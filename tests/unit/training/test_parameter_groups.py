"""Assignment acceptance and fused FFN equivalence for the real backbone."""

from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch
from torch import nn

from src.kimi_k3.ffn.dense_ffn import DenseFFN, SituGLU
from src.kimi_k3.model import KimiK3, KimiK3Config
from src.kimi_k3.training.optimizers import MuonAdamW
from src.kimi_k3.training.parameter_groups import build_parameter_groups
from src.kimi_k3.training.trainer import TrainConfig, build_optimizer
from src.kimi_k3.utils import load_config


@pytest.mark.parametrize("full_rank,mla_gate,moe_norm", [(True, True, True), (False, False, False)])
def test_every_backbone_family_has_the_declared_algorithm(kda_config_factory, full_rank, mla_gate, moe_norm):
	model = KimiK3(kda_config_factory(num_layers=2, layer_types=("kda", "mla"), num_dense_layers=1,
		attn_res_block_size=2, kda_full_rank_output_gate=full_rank,
		mla_output_gate=mla_gate, latent_moe_use_norm=moe_norm))
	groups = build_parameter_groups(model, 0.1)
	assignment = {name: group for group in groups for name in group["param_names"]}
	assert set(assignment) == dict(model.named_parameters()).keys()
	parameters = [parameter for group in groups for parameter in group["params"]]
	assert len(parameters) == len({id(parameter) for parameter in parameters})
	assert model.lm_head.weight is model.embed_tokens.weight
	assert "lm_head.weight" not in assignment
	for name, parameter in model.named_parameters():
		group = assignment[name]
		query = name.endswith(("attn_res_proj.weight", "mlp_res_proj.weight", "output_attn_res_proj.weight"))
		if name == "embed_tokens.weight" or name.endswith(("q_conv.weight", "k_conv.weight", "v_conv.weight")):
			assert (group["algorithm"], group["weight_decay"]) == ("adamw", 0.1), name
		elif query or parameter.ndim == 1:
			assert (group["algorithm"], group["weight_decay"]) == ("adamw", 0.0), name
		else:
			assert (group["algorithm"], group["weight_decay"]) == ("muon", 0.1), name
	# Deliverable 2 intentionally retains full attention matrices.
	for name in ("layers.0.attn.QKV.weight", "layers.1.attn.W_UQ.weight", "layers.1.attn.W_UKV.weight"):
		assert assignment[name].get("muon_shape") is None
	for module_name, module in model.named_modules():
		if isinstance(module, DenseFFN):
			assert assignment[module_name + ".gate_up_proj.weight"]["muon_shape"] == (
				2, module.intermediate_size, module.d_model)


def test_structural_exclusions_untied_head_frozen_weights_and_multidimensional_norm():
	model = nn.Module()
	model.tokens = nn.Embedding(11, 3)
	model.lm_head = nn.Linear(3, 11)
	model.attn_res_proj = nn.Linear(3, 1, bias=False)
	model.row_projection = nn.Linear(3, 1, bias=False)
	model.norm = nn.RMSNorm((2, 3))
	model.scalar = nn.Parameter(torch.ones(()))
	model.frozen = nn.Linear(3, 2).requires_grad_(False)
	groups = build_parameter_groups(model, 0.2)
	assignment = {name: group for group in groups for name in group["param_names"]}
	assert not any(name.startswith("frozen.") for name in assignment)
	assert assignment["tokens.weight"]["algorithm"] == assignment["lm_head.weight"]["algorithm"] == "adamw"
	for name in ("lm_head.bias", "norm.weight", "attn_res_proj.weight", "scalar"):
		assert assignment[name]["algorithm"] == "adamw" and assignment[name]["weight_decay"] == 0
	assert assignment["row_projection.weight"]["algorithm"] == "muon"
	assert model.norm.weight.ndim == 2
	zero_decay = build_parameter_groups(model, 0.0)
	assert [group["param_names"] for group in zero_decay] == [group["param_names"] for group in groups]
	model.requires_grad_(False)
	with pytest.raises(ValueError, match="no trainable"):
		build_parameter_groups(model, 0.2)


@pytest.mark.parametrize("rows,columns", [(3, 7), (7, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_gate_up_matches_two_independent_muon_matrices(rows, columns, dtype):
	ffn = DenseFFN(columns, rows, SituGLU(1, 1))
	fused = ffn.gate_up_proj.weight
	gate, up = [nn.Parameter(part.detach().clone()) for part in fused.chunk(2, dim=0)]
	group = next(group for group in build_parameter_groups(ffn, 0.1) if "muon_shape" in group)
	actual = MuonAdamW([group], lr=0.03, ns_dtype=dtype)
	reference = MuonAdamW([{"params": [gate, up], "algorithm": "muon"}], lr=0.03, ns_dtype=dtype)
	for step in range(5):
		gradient = torch.randn_like(fused)
		# Very different branch magnitudes catch shared Frobenius normalization.
		gradient[rows:] *= 100
		fused.grad = gradient if step != 2 else None
		gate.grad, up.grad = gradient[:rows].clone(), gradient[rows:].clone()
		if step == 2:
			gate.grad = up.grad = None
		actual.step()
		reference.step()
		# Stored weights and momentum are FP32 for both NS working dtypes.
		# Batched/full-tensor and separate/sliced operations need not be bitwise
		# identical across platforms (see PyTorch's numerical accuracy notes).
		torch.testing.assert_close(fused, torch.cat((gate, up)), rtol=1e-5, atol=1e-7)
		torch.testing.assert_close(actual.state[fused]["momentum_buffer"],
			torch.cat([reference.state[parameter]["momentum_buffer"] for parameter in (gate, up)]),
			rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("shape", [(2,), (True, 2, 3), (0, 2, 3), (2, 2, 2), "2,2,3"])
def test_invalid_muon_view_is_rejected(shape):
	with pytest.raises(ValueError, match="muon_shape"):
		MuonAdamW([{"params": [nn.Parameter(torch.ones(4, 3))], "algorithm": "muon", "muon_shape": shape}])


@pytest.mark.parametrize("existing_momentum", [False, True])
def test_unrouted_real_experts_skip_decay_and_momentum(kda_config_factory, existing_momentum):
	model = KimiK3(kda_config_factory(num_layers=2, layer_types=("kda", "mla"), num_dense_layers=1,
		num_experts=4, num_experts_per_token=1))
	moe = model.layers[1].ffn
	optimizer = build_optimizer(model, TrainConfig(learning_rate=0.01, weight_decay=0.1))
	if existing_momentum:
		for expert in moe.experts:
			for parameter in expert.parameters():
				parameter.grad = torch.randn_like(parameter)
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)
	with torch.no_grad():
		moe.router.weight.zero_()
		moe.quantile_balancer.biases.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
	inactive = [parameter for expert in moe.experts[1:] for parameter in expert.parameters()]
	before = [(parameter.clone(), optimizer.state[parameter]["momentum_buffer"].clone()
		if existing_momentum else None) for parameter in inactive]
	x = torch.randint(0, model.config.vocab_size, (1, 8))
	_, loss = model(x, targets=x.roll(-1, dims=1))
	loss.backward()
	assert all(parameter.grad is None for parameter in inactive)
	assert all(parameter.grad is not None for parameter in moe.experts[0].parameters())
	optimizer.step()
	for parameter, (weight, buffer) in zip(inactive, before):
		torch.testing.assert_close(parameter, weight, rtol=0, atol=0)
		if buffer is None:
			assert parameter not in optimizer.state
		else:
			torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], buffer, rtol=0, atol=0)


@pytest.mark.parametrize("implementation", ["chunkwise", pytest.param("fused", marks=pytest.mark.skipif(
	find_spec("fla") is None, reason="FLA is unavailable"))])
def test_shipped_300m_assigns_all_weights_without_allocating_optimizer_state(implementation):
	path = Path(__file__).resolve().parents[3] / "configs/model/kimi_k3_300m.yaml"
	config = load_config(path, KimiK3Config)
	# Both implementations share the same learned tensors. The reference checks
	# the complete shipped geometry on CPU without importing CUDA-only FLA.
	config = replace(config, kda_implementation=implementation)
	with torch.device("meta"):
		model = KimiK3(config)
	optimizer = build_optimizer(model, TrainConfig(weight_decay=0.1))
	assert not optimizer.state
	assert len([parameter for group in optimizer.param_groups for parameter in group["params"]]) == len(list(model.parameters()))
	assert sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]) == sum(
		parameter.numel() for parameter in model.parameters())
	assert {group["algorithm"] for group in optimizer.param_groups} == {"muon", "adamw"}
