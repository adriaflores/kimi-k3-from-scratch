import copy
from dataclasses import asdict

import pytest
import torch
import yaml

from kimi_k3.model import KimiK3, KimiK3Config
from kimi_k3.moe.stable_latent_moe import StableLatentMoE
from kimi_k3.training.moe_metrics import MoEDiagnostics
from kimi_k3.utils import load_config

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
MODES = [
	("cpu", "fp32", torch.float32),
	("cpu", "bf16", torch.float32),
	pytest.param("cuda", "bf16", torch.float32, marks=CUDA),
	pytest.param("cuda", "bf16", torch.bfloat16, marks=CUDA),
]


def _models(config_factory, device, parameter_dtype=torch.float32, **overrides):
	config = config_factory(**{
		"hidden_size": 64, "num_experts": 6, "num_experts_per_token": 2,
		"num_shared_experts": 2, "latent_moe_dim": 32, "expert_intermediate_size": 32,
		"situ_beta": 4.0, "situ_linear_beta": 25.0, "routed_scaling_factor": 1.3,
		**overrides,
	})
	naive = StableLatentMoE(config).to(device=device, dtype=parameter_dtype)
	with torch.no_grad():
		naive.router.weight.zero_()
		naive.router.weight[:, :6] = torch.eye(6, device=device, dtype=parameter_dtype)
		naive.quantile_balancer.biases.copy_(torch.linspace(-0.05, 0.05, 6, device=device))
	grouped = copy.deepcopy(naive)
	# The production model always uses grouped; only this test oracle runs the reference loop.
	naive._forward_grouped_experts = naive._forward_naive_experts
	return config, naive, grouped


def _input(config, selections, device, dtype):
	# Distinct low logits avoid top-k ties; the requested experts always win.
	x = torch.randn(2, len(selections), config.hidden_size, device=device, dtype=dtype)
	x[..., :6] = torch.linspace(-4.0, -6.0, 6, device=device, dtype=dtype)
	for t, selected in enumerate(selections):
		for rank, expert_id in enumerate(selected):
			x[:, t, expert_id] = 3.0 - rank
	return x


def _forward(model, x, precision):
	with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
		return model(x)


def _close(actual, expected, precision, **kwargs):
	rtol, atol = (5e-2, 5e-3) if precision == "bf16" else (2e-5, 2e-6)
	torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, **kwargs)


def _compare_grads(naive, grouped, precision):
	assert dict(naive.named_parameters()).keys() == dict(grouped.named_parameters()).keys()
	for (name, left), right in zip(naive.named_parameters(), grouped.parameters()):
		assert (left.grad is None) == (right.grad is None), name
		if left.grad is not None:
			assert torch.isfinite(right.grad).all(), name
			_close(right.grad, left.grad, precision, msg=name)


@pytest.mark.parametrize("device,precision,parameter_dtype", MODES)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("use_norm,renormalize", [(True, True), (False, False)])
@pytest.mark.parametrize("routing", ["balanced", "irregular", "single"])
def test_full_moe_forward_backward_and_qb(
	kda_config_factory, device, precision, parameter_dtype, training, use_norm, renormalize, routing,
):
	selections = {
		"balanced": [(0, 1), (2, 3), (4, 5)],
		"irregular": [(1, 3), (1, 3), (1, 3), (1, 5), (3, 5)],
		"single": [(5,)],
	}[routing]
	config, naive, grouped = _models(
		kda_config_factory, device, parameter_dtype, num_experts_per_token=len(selections[0]),
		latent_moe_use_norm=use_norm, moe_renormalize=renormalize,
	)
	x = _input(config, selections, device, parameter_dtype)
	left_x, right_x = x.clone().requires_grad_(), x.clone().requires_grad_()
	initial_biases = naive.quantile_balancer.biases.clone()
	naive.train(training)
	grouped.train(training)

	with MoEDiagnostics(naive) as left_diagnostics, MoEDiagnostics(grouped) as right_diagnostics:
		left = _forward(naive, left_x, precision)
		right = _forward(grouped, right_x, precision)
		left_loads = left_diagnostics.snapshot()["<root>"]["expert_counts"]
		right_loads = right_diagnostics.snapshot()["<root>"]["expert_counts"]

	assert left.dtype == right.dtype
	_close(right, left, precision)
	torch.testing.assert_close(grouped.quantile_balancer.biases, naive.quantile_balancer.biases, rtol=0, atol=0)
	if not training:
		torch.testing.assert_close(grouped.quantile_balancer.biases, initial_biases, rtol=0, atol=0)
	else:
		assert not torch.equal(grouped.quantile_balancer.biases, initial_biases)
	counts = [sum(e in selected for selected in selections) * x.shape[0] for e in range(6)]
	assert left_loads == right_loads == (counts if training else [0] * 6)

	grad_output = torch.randn_like(left)
	left.backward(grad_output)
	right.backward(grad_output)
	_close(right_x.grad, left_x.grad, precision)
	_compare_grads(naive, grouped, precision)
	assert all(p.dtype == parameter_dtype for p in grouped.parameters())
	for e, expert in enumerate(grouped.experts):
		for parameter in expert.parameters():
			assert parameter.is_leaf
			assert (parameter.grad is not None) == (counts[e] > 0)
			if counts[e] > 0:
				assert parameter.grad.abs().sum() > 0
	assert grouped.quantile_balancer.biases.grad is None


@pytest.mark.parametrize("device,precision,parameter_dtype", MODES)
def test_grouped_eval_no_grad_keeps_qb_frozen(kda_config_factory, device, precision, parameter_dtype):
	config, naive, grouped = _models(kda_config_factory, device, parameter_dtype)
	naive.eval()
	grouped.eval()
	x = _input(config, [(1, 3), (1, 5)], device, parameter_dtype)
	biases = grouped.quantile_balancer.biases.clone()
	with torch.no_grad():
		_close(_forward(grouped, x, precision), _forward(naive, x, precision), precision)
	torch.testing.assert_close(grouped.quantile_balancer.biases, biases, rtol=0, atol=0)
	assert all(p.grad is None for p in grouped.parameters())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_sparse_adamw_with_accumulation_and_reactivated_experts(kda_config_factory, device):
	precision = "bf16" if device == "cuda" else "fp32"
	config, naive, grouped = _models(kda_config_factory, device, num_experts_per_token=1)
	optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, fused=device == "cuda")
				  for model in (naive, grouped)]
	expected_steps = [0] * 6
	for active_ids in [(0, 2), (1, 2), (0, 1)]:
		before = [{name: p.detach().clone() for name, p in model.experts.named_parameters()}
				  for model in (naive, grouped)]
		before_states = [
			{p: {key: value.clone() for key, value in optimizer.state[p].items()}
			 for p in model.experts.parameters() if p in optimizer.state}
			for model, optimizer in zip((naive, grouped), optimizers)
		]
		for optimizer in optimizers:
			optimizer.zero_grad(set_to_none=True)
		# Two microbatches: grad=None is determined by the union of their active experts.
		for expert_id in active_ids:
			x = _input(config, [(expert_id,)], device, torch.float32)
			grad_output = torch.randn_like(x) / 2
			for model in (naive, grouped):
				with torch.no_grad():
					model.quantile_balancer.biases.zero_() # Keep this test's routing controlled.
				_forward(model, x, precision).backward(grad_output)
		_compare_grads(naive, grouped, precision)
		for e in active_ids:
			expected_steps[e] += 1
		for model, optimizer, weights, states in zip((naive, grouped), optimizers, before, before_states):
			optimizer.step()
			for name, parameter in model.experts.named_parameters():
				e = int(name.split(".")[0])
				assert (parameter.grad is not None) == (e in active_ids)
				if e not in active_ids:
					torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
					if parameter in states:
						for key, value in states[parameter].items():
							torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
				if expected_steps[e]:
					assert optimizer.state[parameter]["step"].item() == expected_steps[e]
				else:
					assert parameter not in optimizer.state
		for (name, left), right in zip(naive.named_parameters(), grouped.parameters()):
			_close(right, left, precision, msg=name)
			for key in optimizers[0].state.get(left, {}):
				_close(optimizers[1].state[right][key], optimizers[0].state[left][key], precision, msg=f"{name}: {key}")
	assert expected_steps == [2, 2, 2, 0, 0, 0]


def test_kimi_always_uses_grouped_and_preserves_state_dict(kda_config_factory, monkeypatch):
	config = kda_config_factory(
		hidden_size=64, num_layers=2, num_dense_layers=1, layer_types=("kda", "mla"),
		num_experts=6, num_experts_per_token=2, latent_moe_dim=32, expert_intermediate_size=32,
	)
	naive = KimiK3(config)
	grouped = KimiK3(config)
	naive.layers[1].ffn._forward_grouped_experts = naive.layers[1].ffn._forward_naive_experts
	assert not hasattr(grouped.layers[1].ffn, "moe_expert_backend")
	def fail_if_naive(*args, **kwargs):
		raise AssertionError("KimiK3 must use grouped execution")
	monkeypatch.setattr(grouped.layers[1].ffn, "_forward_naive_experts", fail_if_naive)
	assert naive.state_dict().keys() == grouped.state_dict().keys()
	grouped.load_state_dict(naive.state_dict(), strict=True)
	idx = torch.randint(0, config.vocab_size, (2, 8))
	left_logits, left_loss = naive(idx, targets=idx)
	right_logits, right_loss = grouped(idx, targets=idx)
	_close(right_logits, left_logits, "fp32")
	_close(right_loss, left_loss, "fp32")
	left_loss.backward()
	right_loss.backward()
	_compare_grads(naive, grouped, "fp32")


def test_yaml_has_no_backend_setting(kda_config_factory, tmp_path):
	data = asdict(kda_config_factory())
	assert "moe_expert_backend" not in data
	path = tmp_path / "model.yaml"
	path.write_text(yaml.safe_dump(data), encoding="utf-8")
	assert asdict(load_config(path, KimiK3Config)) == data
	data["moe_expert_backend"] = "grouped"
	path.write_text(yaml.safe_dump(data), encoding="utf-8")
	with pytest.raises(ValueError, match="Unknown config fields"):
		load_config(path, KimiK3Config)
	del data["moe_expert_backend"]
	del data["hidden_size"]
	path.write_text(yaml.safe_dump(data), encoding="utf-8")
	with pytest.raises(ValueError, match="Missing config fields"):
		load_config(path, KimiK3Config)
