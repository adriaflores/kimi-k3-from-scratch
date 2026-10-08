"""Head packing, independent update oracles and mixed-size K/V blocks."""

import copy

import pytest
import torch
from torch import nn

from src.kimi_k3.attention.gated_mla import GatedMLA
from src.kimi_k3.attention.kda.recurrent import KDARecurrent
from src.kimi_k3.training.optimizers import MuonAdamW
from src.kimi_k3.training.parameter_groups import build_parameter_groups


def _group(model, name, decay=0.2):
	return next(group for group in build_parameter_groups(model, decay) if name in group["param_names"])


def _independent_blocks(module, name, tensor):
	"""Decode the actual forward packing, without consulting optimizer metadata."""
	if name == "QKV.weight":
		return [head for projection in tensor.split(module.kda_dim, dim=0)
			for head in projection.split(module.d_head, dim=0)]
	if name == "W_UQ.weight":
		return list(tensor.split(module.d_head + module.d_shared, dim=0))
	return [part for head in tensor.split(module.d_head + module.dv_head, dim=0)
		for part in head.split((module.d_head, module.dv_head), dim=0)]


@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("projection", ["QKV.weight", "W_UQ.weight", "W_UKV.weight"])
@pytest.mark.parametrize("ns_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_attention_matches_independent_head_and_projection_optimizers(kda_config_factory, heads, projection, ns_dtype):
	config = kda_config_factory(hidden_size=11, kda_num_heads=heads, kda_head_dim=13,
		mla_num_heads=heads, mla_q_lora_rank=3, mla_kv_lora_rank=4,
		mla_qk_nope_head_dim=5, mla_qk_shared_head_dim=2, mla_v_head_dim=2)
	module = (KDARecurrent(config) if projection == "QKV.weight" else GatedMLA(config)).double()
	parameter = dict(module.named_parameters())[projection]
	parts = [nn.Parameter(part.detach().clone()) for part in _independent_blocks(module, projection, parameter)]
	actual = MuonAdamW([_group(module, projection)], lr=0.03, ns_dtype=ns_dtype)
	reference = MuonAdamW([{"params": parts, "algorithm": "muon"}], lr=0.03, weight_decay=0.2, ns_dtype=ns_dtype)
	# Tall K / wide V test that each part uses its own shape scaling.
	for step in range(6):
		gradient = torch.randn_like(parameter)
		for index, block in enumerate(_independent_blocks(module, projection, gradient)):
			block.mul_(10.0 ** (index % 3 - 1))
		if step == 3:
			gradient.zero_()  # Existing momentum remains active.
		parameter.grad = None if step == 2 else gradient
		for other, block in zip(parts, _independent_blocks(module, projection, gradient)):
			other.grad = None if step == 2 else block.clone()
		original_gradient = gradient.clone()
		actual.step()
		reference.step()
		# BF16 NS can differ between batched and individual GEMMs; judge the
		# update at its working precision. FP32/FP64 comparisons are much tighter.
		rtol, atol = {torch.bfloat16: (0.03, 0.002), torch.float32: (1e-5, 1e-7),
			torch.float64: (1e-10, 1e-12)}[ns_dtype]
		torch.testing.assert_close(parameter, torch.cat(parts), rtol=rtol, atol=atol)
		momentum = actual.state[parameter]["momentum_buffer"]
		torch.testing.assert_close(momentum,
			torch.cat([reference.state[part]["momentum_buffer"] for part in parts]), rtol=1e-12, atol=1e-12)
		assert momentum.shape == parameter.shape
		torch.testing.assert_close(gradient, original_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("projection", ["QKV.weight", "W_UQ.weight", "W_UKV.weight"])
@pytest.mark.parametrize("ns_dtype", [torch.bfloat16, torch.float32])
def test_changing_one_head_or_kv_branch_does_not_change_other_blocks(kda_config_factory, projection, ns_dtype):
	config = kda_config_factory(kda_num_heads=3, mla_num_heads=3,
		mla_qk_nope_head_dim=5, mla_qk_shared_head_dim=2, mla_v_head_dim=3)
	left = KDARecurrent(config) if projection == "QKV.weight" else GatedMLA(config)
	right = copy.deepcopy(left)
	a, b = dict(left.named_parameters())[projection], dict(right.named_parameters())[projection]
	optimizers = [MuonAdamW([_group(module, projection)], lr=0.02, ns_dtype=ns_dtype) for module in (left, right)]
	for _ in range(4):
		a.grad = torch.randn_like(a)
		b.grad = a.grad.clone()
		_independent_blocks(right, projection, b.grad)[0].mul_(100)
		for optimizer in optimizers:
			optimizer.step()
		# Both sides execute the same kernels and geometry, so exact equality
		# is appropriate here, including V of the same head as the altered K.
		for original, altered in zip(_independent_blocks(left, projection, a)[1:],
			_independent_blocks(right, projection, b)[1:]):
			torch.testing.assert_close(original, altered, rtol=0, atol=0)
		for original, altered in zip(_independent_blocks(left, projection, optimizers[0].state[a]["momentum_buffer"])[1:],
			_independent_blocks(right, projection, optimizers[1].state[b]["momentum_buffer"])[1:]):
			torch.testing.assert_close(original, altered, rtol=0, atol=0)
	assert not torch.equal(_independent_blocks(left, projection, a)[0], _independent_blocks(right, projection, b)[0])


@pytest.mark.parametrize("splits", [(2,), (), (True, 4), (0, 5), (-1, 6), (2.0, 3), (2, 2), "2,3"])
def test_invalid_row_splits_are_rejected(splits):
	with pytest.raises(ValueError, match="muon_split_sizes"):
		MuonAdamW([{"params": [nn.Parameter(torch.ones(15, 4))], "algorithm": "muon",
			"muon_shape": (3, 5, 4), "muon_split_sizes": splits}])


def test_adamw_cannot_silently_ignore_muon_geometry():
	with pytest.raises(ValueError, match="Muon matrix views"):
		MuonAdamW([{"params": [nn.Parameter(torch.ones(5, 4))], "algorithm": "adamw", "muon_split_sizes": (2, 3)}])


@pytest.mark.parametrize("existing_momentum", [False, True])
def test_split_parameter_without_gradient_preserves_decay_and_state(existing_momentum):
	parameter = nn.Parameter(torch.randn(15, 4))
	optimizer = MuonAdamW([{"params": [parameter], "algorithm": "muon",
		"muon_shape": (3, 5, 4), "muon_split_sizes": (2, 3)}], lr=0.1, weight_decay=0.3)
	if existing_momentum:
		parameter.grad = torch.randn_like(parameter)
		optimizer.step()
	optimizer.zero_grad(set_to_none=True)
	before = parameter.detach().clone()
	buffer = optimizer.state[parameter]["momentum_buffer"].clone() if existing_momentum else None
	optimizer.step()
	torch.testing.assert_close(parameter, before, rtol=0, atol=0)
	if buffer is None:
		assert parameter not in optimizer.state
	else:
		torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], buffer, rtol=0, atol=0)


def test_zero_split_gradient_decays_the_parameter_once():
	parameter = nn.Parameter(torch.ones(15, 4))
	optimizer = MuonAdamW([{"params": [parameter], "algorithm": "muon",
		"muon_shape": (3, 5, 4), "muon_split_sizes": (2, 3)}], lr=0.1, weight_decay=0.3)
	parameter.grad = torch.zeros_like(parameter)
	optimizer.step()
	torch.testing.assert_close(parameter, torch.full_like(parameter, 0.97), rtol=0, atol=0)
	assert torch.count_nonzero(optimizer.state[parameter]["momentum_buffer"]) == 0


def test_split_update_handles_noncontiguous_parameter_storage():
	parameter = nn.Parameter(torch.randn(4, 15, dtype=torch.float64).T)
	reference = nn.Parameter(parameter.detach().clone().contiguous())
	assert not parameter.is_contiguous()
	optimizers = [MuonAdamW([{"params": [item], "algorithm": "muon",
		"muon_shape": (3, 5, 4), "muon_split_sizes": (2, 3)}], ns_dtype=torch.float64) for item in (parameter, reference)]
	for _ in range(3):
		parameter.grad = torch.randn_like(parameter)
		reference.grad = parameter.grad.clone()
		for optimizer in optimizers:
			optimizer.step()
		torch.testing.assert_close(parameter, reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("key,value", [("muon_shape", (5, 3, 4)), ("muon_split_sizes", (3, 2))])
def test_optimizer_state_rejects_same_size_but_different_geometry_before_mutation(key, value):
	parameter = nn.Parameter(torch.randn(15, 4))
	optimizer = MuonAdamW([{"params": [parameter], "algorithm": "muon",
		"muon_shape": (3, 5, 4), "muon_split_sizes": (2, 3)}])
	parameter.grad = torch.randn_like(parameter)
	optimizer.step()
	before = optimizer.state[parameter]["momentum_buffer"].clone()
	saved = copy.deepcopy(optimizer.state_dict())
	saved["param_groups"][0][key] = value
	if key == "muon_shape":
		saved["param_groups"][0]["muon_split_sizes"] = (1, 2)
	with pytest.raises(ValueError, match="matrix views"):
		optimizer.load_state_dict(saved)
	torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], before, rtol=0, atol=0)
	assert optimizer.param_groups[0]["muon_shape"] == (3, 5, 4)
	assert optimizer.param_groups[0]["muon_split_sizes"] == (2, 3)


def test_assignment_rejects_incorrect_or_conflicting_head_dimensions(kda_config_factory):
	model = nn.Module()
	model.first = KDARecurrent(kda_config_factory(kda_num_heads=3, kda_head_dim=4))
	model.second = KDARecurrent(kda_config_factory(kda_num_heads=2, kda_head_dim=6))
	model.second.QKV.weight = model.first.QKV.weight
	with pytest.raises(ValueError, match="conflicting Muon views"):
		build_parameter_groups(model, 0.1)
	model.first.num_heads = 4
	with pytest.raises(ValueError, match="declared Muon dimensions"):
		build_parameter_groups(model.first, 0.1)
