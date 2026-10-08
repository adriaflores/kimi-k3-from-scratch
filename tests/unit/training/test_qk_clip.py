"""Independent score/scaling oracles and observer lifecycle acceptance."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.kimi_k3.attention.gated_mla import GatedMLA
from src.kimi_k3.attention.kda.recurrent import KDARecurrent
from src.kimi_k3.training.optimizers import MuonAdamW
from src.kimi_k3.training.parameter_groups import build_parameter_groups
from src.kimi_k3.training.qk_clip import QKClip, causal_max_logits


def _score_reference(query, key):
	"""Scalar dot products, explicit causal indices; no chunk implementation."""
	q, k = query.double(), key.double()
	return torch.tensor([max((q[b, h, i] * k[b, h, j]).sum().item() / math.sqrt(q.shape[-1])
		for b in range(q.shape[0]) for i in range(q.shape[2]) for j in range(i + 1))
		for h in range(q.shape[1])], dtype=query.dtype, device=query.device)


@pytest.mark.parametrize("shape", [(1, 1, 1, 2), (2, 3, 7, 5)])
@pytest.mark.parametrize("chunk_size", [1, 3, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunked_max_matches_independent_causal_dot_products(shape, chunk_size, dtype):
	q, k = [torch.randn(shape, dtype=dtype, requires_grad=True) for _ in range(2)]
	before = [item.detach().clone() for item in (q, k)]
	actual = causal_max_logits(q, k, chunk_size=chunk_size)
	torch.testing.assert_close(actual, _score_reference(q, k), rtol=1e-6, atol=1e-7)
	assert actual.shape == (shape[1],) and not actual.requires_grad
	for item, original in zip((q, k), before):
		torch.testing.assert_close(item, original, rtol=0, atol=0)
		assert item.grad is None


@pytest.mark.parametrize("chunk_size", [1, 2, 256])
def test_future_scores_are_excluded_and_the_maximum_is_signed(chunk_size):
	q = torch.tensor([[[[1.0], [0.0]], [[-1.0], [-2.0]]]])
	k = torch.tensor([[[[-2.0], [1e8]], [[1.0], [2.0]]]])
	torch.testing.assert_close(causal_max_logits(q, k, chunk_size=chunk_size), torch.tensor([0.0, -1.0]))


def test_threshold_units_include_sqrt_dimension():
	# Raw QK = 320, while the actual pre-softmax score is 80, below tau=100.
	q, k = torch.ones(1, 1, 2, 16), torch.full((1, 1, 2, 16), 20.0)
	torch.testing.assert_close(causal_max_logits(q, k), torch.tensor([80.0]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_scores_use_fp32_even_under_autocast(dtype):
	q = torch.full((1, 2, 3, 2), 250.0, dtype=dtype)
	k = torch.full_like(q, 500.0)
	with torch.autocast("cpu", dtype=torch.bfloat16):
		actual = causal_max_logits(q, k, chunk_size=2)
	assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
	torch.testing.assert_close(actual, _score_reference(q.float(), k.float()))


def _mla(kda_config_factory, *, shared=2):
	module = GatedMLA(kda_config_factory(mla_num_heads=3, mla_q_lora_rank=5, mla_kv_lora_rank=4,
		mla_qk_nope_head_dim=3, mla_qk_shared_head_dim=shared, mla_v_head_dim=5)).double()
	with torch.no_grad():
		module.W_UQ.weight[:3 + shared].mul_(50)
		module.W_UKV.weight[:3].mul_(50)
	return module


def _queries_keys(module, x):
	"""Separate linear projections per head, independent of observer metadata."""
	q_latent = module.q_rmsnorm(F.linear(x, module.W_DQ.weight))
	kv = F.linear(x, module.W_DKV.weight)
	kv_latent = module.kv_rmsnorm(kv[..., :module.kv_latent_dim])
	shared = kv[..., module.kv_latent_dim:]
	queries, keys = [], []
	for h in range(module.num_heads):
		q_start = h * (module.d_head + module.d_shared)
		kv_start = h * (module.d_head + module.dv_head)
		queries.append(F.linear(q_latent, module.W_UQ.weight[q_start:q_start + module.d_head + module.d_shared]))
		key = F.linear(kv_latent, module.W_UKV.weight[kv_start:kv_start + module.d_head])
		keys.append(torch.cat((key, shared), dim=-1))
	return torch.stack(queries, dim=1), torch.stack(keys, dim=1)


@pytest.mark.parametrize("shared", [1, 2])
def test_mla_scaling_caps_fixed_input_scores_and_preserves_other_heads_values_and_shared_key(kda_config_factory, shared):
	module = _mla(kda_config_factory, shared=shared)
	x = torch.randn(2, 7, 16, dtype=torch.float64)
	before = copy.deepcopy(module.state_dict())
	maximum = _score_reference(*_queries_keys(module, x))
	assert maximum[0] > 100 and torch.all(maximum[1:] < 100)
	gamma = 100 / maximum.clamp_min(100)
	with QKClip(module, chunk_size=3) as clip:
		module(x).square().mean().backward()
		torch.testing.assert_close(clip.max_logits[""], maximum, rtol=1e-10, atol=1e-10)
		clip.apply()
		assert not clip.max_logits
		assert clip.summary()["clipped_head_updates"] == 1
	after = _score_reference(*_queries_keys(module, x))
	torch.testing.assert_close(after, maximum * gamma, rtol=1e-10, atol=1e-10)
	q = module.W_UQ.weight.unflatten(0, (3, 3 + shared))
	kv = module.W_UKV.weight.unflatten(0, (3, 8))
	old_q, old_kv = before["W_UQ.weight"].unflatten(0, (3, 3 + shared)), before["W_UKV.weight"].unflatten(0, (3, 8))
	torch.testing.assert_close(q[:, :3], old_q[:, :3] * gamma.sqrt().view(-1, 1, 1))
	torch.testing.assert_close(q[:, 3:], old_q[:, 3:] * gamma.view(-1, 1, 1))
	torch.testing.assert_close(kv[:, :3], old_kv[:, :3] * gamma.sqrt().view(-1, 1, 1))
	torch.testing.assert_close(kv[:, 3:], old_kv[:, 3:], rtol=0, atol=0)
	torch.testing.assert_close(q[1:], old_q[1:], rtol=0, atol=0)
	torch.testing.assert_close(kv[1:], old_kv[1:], rtol=0, atol=0)
	for name, parameter in module.state_dict().items():
		if name not in ("W_UQ.weight", "W_UKV.weight"):
			torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_microbatches_use_max_not_mean_and_clip_only_once(kda_config_factory):
	module = _mla(kda_config_factory)
	for parameter in module.parameters():
		parameter.grad = torch.zeros_like(parameter)
	before = copy.deepcopy(module.state_dict())
	with QKClip(module) as clip:
		# With D=1 these directly encode each microbatch's per-head score.
		for scores in ([400.0, 50.0, -200.0], [10.0, 200.0, -100.0]):
			clip.observe("", torch.ones(1, 3, 1, 1), torch.tensor(scores).view(1, 3, 1, 1))
		torch.testing.assert_close(clip.max_logits[""], torch.tensor([400.0, 200.0, -100.0]))
		clip.apply()
		once = copy.deepcopy(module.state_dict())
		clip.apply()
		for name, value in module.state_dict().items():
			torch.testing.assert_close(value, once[name], rtol=0, atol=0)
		summary = clip.summary(reset=True)
		assert summary["clipped_head_updates"] == 2 and summary["observed_head_updates"] == 3
		assert summary["layers"][""]["minimum_gamma"] == [0.25, 0.5, 1.0]
		assert clip.summary()["observed_head_updates"] == 0
	q = module.W_UQ.weight.unflatten(0, (3, 5))
	old = before["W_UQ.weight"].unflatten(0, (3, 5))
	torch.testing.assert_close(q[:, 3:], old[:, 3:] * torch.tensor([0.25, 0.5, 1.0], dtype=q.dtype).view(-1, 1, 1))


def test_observer_does_not_change_forward_backward_rng_or_collect_eval_and_no_grad(kda_config_factory):
	left, right = _mla(kda_config_factory), _mla(kda_config_factory)
	right.load_state_dict(left.state_dict())
	x = torch.randn(2, 5, 16, dtype=torch.float64)
	left_output = left(x)
	left_output.sum().backward()
	rng = torch.get_rng_state().clone()
	with QKClip(right, chunk_size=2) as clip:
		right_output = right(x)
		right_output.sum().backward()
		captured = clip.max_logits[""].clone()
		right.eval()
		right(x * 100)
		right.train()
		with torch.no_grad():
			right(x * 100)
		torch.testing.assert_close(clip.max_logits[""], captured, rtol=0, atol=0)
	torch.testing.assert_close(right_output, left_output, rtol=0, atol=0)
	for a, b in zip(left.parameters(), right.parameters()):
		torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
	assert torch.equal(torch.get_rng_state(), rng)
	assert not hasattr(right, "_qk_clip_observer")
	assert not clip.max_logits


@pytest.mark.parametrize("inactive", ["frozen", "missing_gradient", "discarded_attempt", "at_threshold"])
def test_inactive_or_nontriggering_clipping_preserves_weights(kda_config_factory, inactive):
	module = _mla(kda_config_factory)
	for parameter in module.parameters():
		parameter.grad = torch.zeros_like(parameter)
	if inactive == "frozen":
		module.W_UQ.weight.requires_grad_(False)
	elif inactive == "missing_gradient":
		module.W_UKV.weight.grad = None
	before = copy.deepcopy(module.state_dict())
	with QKClip(module) as clip:
		clip.observe("", torch.ones(1, 3, 1, 1), torch.full((1, 3, 1, 1), 100.0 if inactive == "at_threshold" else 200.0))
		if inactive == "discarded_attempt":
			clip.reset()
		clip.apply()
		assert clip.summary()["clipped_head_updates"] == 0
	for name, parameter in module.state_dict().items():
		torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_clipping_scales_post_optimizer_weights_without_changing_gradients_or_momentum(kda_config_factory):
	module = _mla(kda_config_factory)
	optimizer = MuonAdamW(build_parameter_groups(module, 0.1), lr=0.01, ns_dtype=torch.float64)
	with QKClip(module) as clip:
		module(torch.randn(2, 7, 16, dtype=torch.float64)).square().mean().backward()
		assert clip.max_logits[""][0] > 100
		gamma = 100 / clip.max_logits[""].clamp_min(100)
		optimizer.step()
		post_step = copy.deepcopy(module.state_dict())
		state = copy.deepcopy(optimizer.state_dict())
		gradients = [parameter.grad.clone() for parameter in module.parameters()]
		clip.apply()
		q = module.W_UQ.weight.unflatten(0, (3, 5))
		kv = module.W_UKV.weight.unflatten(0, (3, 8))
		old_q = post_step["W_UQ.weight"].unflatten(0, (3, 5))
		old_kv = post_step["W_UKV.weight"].unflatten(0, (3, 8))
		torch.testing.assert_close(q[:, :3], old_q[:, :3] * gamma.sqrt().view(-1, 1, 1))
		torch.testing.assert_close(q[:, 3:], old_q[:, 3:] * gamma.view(-1, 1, 1))
		torch.testing.assert_close(kv[:, :3], old_kv[:, :3] * gamma.sqrt().view(-1, 1, 1))
		torch.testing.assert_close(kv[:, 3:], old_kv[:, 3:], rtol=0, atol=0)
		for name, value in module.state_dict().items():
			if name not in ("W_UQ.weight", "W_UKV.weight"):
				torch.testing.assert_close(value, post_step[name], rtol=0, atol=0)
		for parameter, gradient in zip(module.parameters(), gradients):
			torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
		for index, values in optimizer.state_dict()["state"].items():
			for name, value in values.items():
				torch.testing.assert_close(value, state["state"][index][name], rtol=0, atol=0)


def test_nonfinite_stats_reject_all_layers_before_any_clip_mutation(kda_config_factory):
	model = nn.ModuleList([_mla(kda_config_factory), _mla(kda_config_factory)])
	for parameter in model.parameters():
		parameter.grad = torch.zeros_like(parameter)
	before = copy.deepcopy(model.state_dict())
	with QKClip(model) as clip:
		clip.max_logits.update({"0": torch.full((3,), 200.0), "1": torch.full((3,), float("nan"))})
		with pytest.raises(FloatingPointError, match="max logit"):
			clip.apply()
	for name, value in model.state_dict().items():
		torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_observers_are_removed_on_failure_and_nested_observers_are_rejected(kda_config_factory):
	module = _mla(kda_config_factory)
	with pytest.raises(RuntimeError, match="test failure"), QKClip(module):
		raise RuntimeError("test failure")
	assert not hasattr(module, "_qk_clip_observer")
	with QKClip(module):
		original = module._qk_clip_observer
		with pytest.raises(RuntimeError, match="already attached"), QKClip(module):
			pass
		assert module._qk_clip_observer is original
	assert not hasattr(module, "_qk_clip_observer")


def test_kda_has_no_qkclip_observer_or_weight_mutation(kda_config_factory):
	module = KDARecurrent(kda_config_factory())
	before = copy.deepcopy(module.state_dict())
	with QKClip(module) as clip:
		assert not clip.sites
		module(torch.randn(1, 4, 16)).square().mean().backward()
		assert not clip.max_logits
		clip.apply()
	for name, value in module.state_dict().items():
		torch.testing.assert_close(value, before[name], rtol=0, atol=0)
