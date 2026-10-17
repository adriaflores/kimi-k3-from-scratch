"""Batched Muon preserves individual state, head geometry and missing gradients."""

import copy

import pytest
import torch

from kimi_k3.training import optimizers
from kimi_k3.training.optimizers import MuonAdamW


CUDA = pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))
LAYOUTS = [((5, 7), None, None), ((14, 5), (2, 7, 5), None),
	((24, 7), (3, 2, 4, 7), None), ((15, 4), (3, 5, 4), (2, 3)),
	((21, 3), (3, 7, 3), (4, 3))]


@pytest.fixture(autouse=True)
def single_thread():
	previous = torch.get_num_threads()
	torch.set_num_threads(1)
	try:
		yield
	finally:
		torch.set_num_threads(previous)


def _groups(parameters, layout, auxiliary=None):
	_, view, splits = layout
	group = {"params": parameters, "algorithm": "muon"}
	if view:
		group["muon_shape"] = view
	if splits:
		group["muon_split_sizes"] = splits
	return [group] + ([] if auxiliary is None else [{"params": [auxiliary], "algorithm": "adamw"}])


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("nesterov", [False, True])
def test_batched_steps_match_individual_with_partial_chunks_and_inactive_parameters(device, layout, nesterov):
	shape = layout[0]
	reference = [torch.nn.Parameter(torch.randn(shape, dtype=torch.float64, device=device)) for _ in range(7)]
	actual = [torch.nn.Parameter(item.detach().clone()) for item in reference]
	auxiliary = torch.nn.Parameter(torch.randn(3, dtype=torch.float64, device=device))
	auxiliary_copy = torch.nn.Parameter(auxiliary.detach().clone())
	settings = dict(lr=0.03, weight_decay=0.2, ns_dtype=torch.float64, nesterov=nesterov, adamw_fused=False)
	individual = MuonAdamW(_groups(reference, layout, auxiliary), **settings)
	batched = MuonAdamW(_groups(actual, layout, auxiliary_copy), ns_batched=True, ns_batch_size=4, **settings)
	# The last parameter never receives a gradient; others become inactive
	# after acquiring state, or reactivate. Zero gradients are ordinary updates.
	for step, active in enumerate(((0, 1, 2, 3, 4, 5), (1, 2, 3), (), (0, 2, 4, 5))):
		for optimizer in (individual, batched):
			optimizer.zero_grad(set_to_none=True)
			for group in optimizer.param_groups:
				group["lr"] = 0.03 / (step + 1)
		for index in active:
			gradient = torch.randn_like(reference[index]) if index else torch.zeros_like(reference[index])
			reference[index].grad = gradient
			actual[index].grad = gradient.clone()
		if active:
			auxiliary.grad = torch.randn_like(auxiliary)
			auxiliary_copy.grad = auxiliary.grad.clone()
		individual.step()
		batched.step()
		for expected, observed in zip(reference + [auxiliary], actual + [auxiliary_copy]):
			torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-12)
			for name, value in individual.state.get(expected, {}).items():
				torch.testing.assert_close(batched.state[observed][name], value, rtol=0, atol=0)
	assert actual[-1] not in batched.state
	assert all(set(batched.state[item]) == {"momentum_buffer"} for item in actual if item in batched.state)


def test_batched_independence_covers_parameters_and_kda_heads():
	layout = ((24, 7), (3, 2, 4, 7), None)
	values = [torch.nn.Parameter(torch.randn(24, 7, dtype=torch.float64)) for _ in range(3)]
	clones = [torch.nn.Parameter(value.detach().clone()) for value in values]
	for value, clone in zip(values, clones):
		value.grad = torch.randn_like(value)
		clone.grad = value.grad.clone()
	clones[0].grad.reshape(3, 2, 4, 7)[1, 0].mul_(100)
	for parameters in (values, clones):
		MuonAdamW(_groups(parameters, layout), ns_batched=True, ns_dtype=torch.float64).step()
	for index in (1, 2):
		torch.testing.assert_close(values[index], clones[index], rtol=0, atol=0)
	mask = torch.ones(3, 2, dtype=torch.bool)
	mask[1, 0] = False
	torch.testing.assert_close(values[0].reshape(3, 2, 4, 7)[mask], clones[0].reshape(3, 2, 4, 7)[mask], rtol=0, atol=0)


def test_real_cpu_inductor_accepts_packed_kda_heads_and_zero_padding(monkeypatch):
	monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
	matrix = torch.randn(4, 3, 2, 4, 7, dtype=torch.float64)
	matrix[-1].zero_()
	expected = torch.stack([optimizers.zeropower_via_newton_schulz(item, dtype=torch.float64) for item in matrix])
	actual = optimizers.zeropower_via_newton_schulz(matrix, dtype=torch.float64, compiled=True)
	torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
	assert not actual[-1].count_nonzero()


def test_batched_noncontiguous_storage_and_mixed_shapes_are_reassembled_correctly():
	parameters = [torch.nn.Parameter(torch.randn(4, 15, dtype=torch.float64).T) for _ in range(3)]
	parameters += [torch.nn.Parameter(torch.randn(7, 6, dtype=torch.float32)) for _ in range(3)]
	reference = [torch.nn.Parameter(item.detach().clone().contiguous()) for item in parameters]
	groups = [{"algorithm": "muon", "muon_shape": (3, 5, 4), "muon_split_sizes": (2, 3)},
		{"algorithm": "muon"}]
	# Build equivalent groups without assuming contiguous parameter storage.
	def build(items, batched):
		return MuonAdamW([{**group, "params": items[:3] if index == 0 else items[3:]}
			for index, group in enumerate(groups)], ns_dtype=torch.float64, ns_batched=batched)
	for actual, expected in zip(parameters, reference):
		actual.grad = torch.randn_like(actual)
		expected.grad = actual.grad.clone()
	build(parameters, True).step()
	build(reference, False).step()
	for actual, expected in zip(parameters, reference):
		torch.testing.assert_close(actual, expected)


def test_batch_limits_and_active_counts_bound_ns_layouts(monkeypatch):
	parameters = [torch.nn.Parameter(torch.randn(5, 7)) for _ in range(11)]
	optimizer = MuonAdamW(_groups(parameters, LAYOUTS[0]), ns_batched=True, ns_batch_size=4)
	original = optimizers.zeropower_via_newton_schulz
	calls = []
	def counted(matrix, **kwargs):
		calls.append(tuple(matrix.shape))
		return original(matrix, **kwargs)
	monkeypatch.setattr(optimizers, "zeropower_via_newton_schulz", counted)
	for count in (11, 10, 9, 7, 6, 5, 3, 2, 0):
		optimizer.zero_grad(set_to_none=True)
		for parameter in parameters[:count]:
			parameter.grad = torch.ones_like(parameter)
		optimizer.step()
	assert set(calls) == {(4, 5, 7), (2, 5, 7), (5, 7)}
	# A low packing budget chooses individual execution without changing math.
	calls.clear()
	optimizer.param_groups[0]["ns_batch_max_bytes"] = 1
	for parameter in parameters:
		parameter.grad = torch.zeros_like(parameter)
	optimizer.step()
	assert calls == [(5, 7)] * len(parameters)


def test_batch_cache_is_rebuilt_after_adding_groups_and_loading_state():
	parameters = [torch.nn.Parameter(torch.randn(5, 7)) for _ in range(3)]
	optimizer = MuonAdamW(_groups(parameters[:2], LAYOUTS[0]), ns_batched=True)
	optimizer.add_param_group({"params": parameters[2:], "algorithm": "muon"})
	for parameter in parameters:
		parameter.grad = torch.randn_like(parameter)
	optimizer.step()
	saved = copy.deepcopy(optimizer.state_dict())
	optimizer.load_state_dict(saved)
	for parameter in parameters:
		parameter.grad = torch.randn_like(parameter)
	optimizer.step()
	assert all(parameter in optimizer.state for parameter in parameters)
	assert sum(len(bucket) for buckets in optimizer._muon_buckets.values() for bucket in buckets) == 3


def test_legacy_individual_state_and_changed_batch_settings_are_validated_before_mutation():
	parameters = [torch.nn.Parameter(torch.randn(5, 7)) for _ in range(2)]
	optimizer = MuonAdamW(_groups(parameters, LAYOUTS[0]))
	saved = copy.deepcopy(optimizer.state_dict())
	for name in ("ns_batched", "ns_batch_size", "ns_batch_max_bytes"):
		saved["param_groups"][0].pop(name)
	optimizer.load_state_dict(saved)
	assert "ns_batched" not in saved["param_groups"][0]
	batched = MuonAdamW(_groups(parameters, LAYOUTS[0]), ns_batched=True)
	with pytest.raises(ValueError, match="batching"):
		batched.load_state_dict(saved)
	for name, value in (("ns_batched", False), ("ns_batch_size", 16), ("ns_batch_max_bytes", 128)):
		changed = copy.deepcopy(batched.state_dict())
		changed["param_groups"][0][name] = value
		with pytest.raises(ValueError, match="batching"):
			batched.load_state_dict(changed)
	assert not batched.state


@pytest.mark.parametrize("kwargs", [{"ns_batch_size": 0}, {"ns_batch_size": True},
	{"ns_batch_size": 2.5}, {"ns_batch_max_bytes": 0}, {"ns_batch_max_bytes": True}])
def test_invalid_batch_limits_are_rejected(kwargs):
	with pytest.raises(ValueError):
		MuonAdamW(_groups([torch.nn.Parameter(torch.ones(5, 7))], LAYOUTS[0]), **kwargs)


def test_invalid_batch_flag_is_rejected():
	with pytest.raises(TypeError, match="ns_batched"):
		MuonAdamW(_groups([torch.nn.Parameter(torch.ones(5, 7))], LAYOUTS[0]), ns_batched=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_batched_compiled_bf16_updates_match_individual_and_preserve_momentum(monkeypatch):
	monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
	layout = LAYOUTS[3]
	parameters = [torch.nn.Parameter(torch.randn(layout[0], device="cuda")) for _ in range(5)]
	clones = [torch.nn.Parameter(item.detach().clone()) for item in parameters]
	individual = MuonAdamW(_groups(parameters, layout), ns_compile=True, lr=0.05)
	batched = MuonAdamW(_groups(clones, layout), ns_compile=True, ns_batched=True, ns_batch_size=4, lr=0.05)
	for active in ((0, 1, 2, 3), (1, 2, 3), (0, 3)):
		for optimizer in (individual, batched):
			optimizer.zero_grad(set_to_none=True)
		before = [item.detach().clone() for item in parameters]
		before_copy = [item.detach().clone() for item in clones]
		for index in active:
			parameters[index].grad = torch.randn_like(parameters[index])
			clones[index].grad = parameters[index].grad.clone()
		individual.step()
		batched.step()
		for index in active:
			# Compare updates, rather than allowing unchanged initial weights to
			# hide an omitted/misassigned update. BF16 NS may round differently.
			expected = (before[index] - parameters[index]) / 0.05
			actual = (before_copy[index] - clones[index]) / 0.05
			torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
			torch.testing.assert_close(batched.state[clones[index]]["momentum_buffer"],
				individual.state[parameters[index]]["momentum_buffer"], rtol=0, atol=0)
	assert clones[-1] not in batched.state
