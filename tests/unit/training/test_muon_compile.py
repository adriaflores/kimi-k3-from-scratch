"""Real compiled arithmetic, specialization reuse, and execution-mode state."""

import copy

import pytest
import torch

from src.kimi_k3.training import optimizers
from src.kimi_k3.training.optimizers import MuonAdamW, zeropower_via_newton_schulz as ns


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda",
	marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_real_inductor_matches_eager_and_preserves_head_independence(dtype, device, monkeypatch):
	monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
	torch.manual_seed(12)
	matrix = torch.randn(2, 5, 9, device=device)
	matrix[0] *= 100
	original = matrix.clone()
	with torch.autocast(device, dtype=torch.bfloat16):
		actual = ns(matrix, dtype=dtype, compiled=True)
	expected = ns(matrix, dtype=dtype)
	# CPU Inductor's BF16 cast optimizations still differ from eager even with
	# emulate_precision_casts; five polynomial iterations amplify that rounding.
	bf16_tolerance = 0.05 if device == "cpu" else 0.03
	rtol, atol = {torch.bfloat16: (bf16_tolerance, bf16_tolerance), torch.float32: (2e-5, 2e-6),
		torch.float64: (1e-10, 1e-12)}[dtype]
	torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
	if dtype == torch.bfloat16 and device == "cpu":
		assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.05
	assert actual.dtype == dtype and not actual.requires_grad
	torch.testing.assert_close(matrix, original, rtol=0, atol=0)
	altered = matrix.clone()
	altered[0] *= 10
	independent = ns(altered, dtype=dtype, compiled=True)
	torch.testing.assert_close(actual[1], independent[1], rtol=0, atol=0)
	zero = ns(torch.zeros_like(matrix), dtype=dtype, compiled=True)
	assert torch.isfinite(zero).all() and torch.count_nonzero(zero) == 0


def test_real_inductor_handles_tall_noncontiguous_head_views(monkeypatch):
	monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
	# MLA K/V slices need not be contiguous; the tall case transposes internally.
	matrix = torch.randn(3, 14, 4)[:, :9, :]
	assert not matrix.is_contiguous()
	expected = ns(matrix, dtype=torch.float64)
	actual = ns(matrix, dtype=torch.float64, compiled=True)
	torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_real_inductor_preserves_kda_projection_and_head_axes(monkeypatch):
	monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
	matrix = torch.randn(3, 2, 4, 7)
	actual = ns(matrix, dtype=torch.float64, compiled=True)
	expected = torch.stack([ns(head, dtype=torch.float64)
		for head in matrix.flatten(0, 1)]).reshape(matrix.shape)
	torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
	changed = matrix.clone()
	changed[1, 0] = torch.randn(4, 7) * 1000
	updated = ns(changed, dtype=torch.float64, compiled=True)
	mask = torch.ones(3, 2, dtype=torch.bool)
	mask[1, 0] = False
	torch.testing.assert_close(actual[mask], updated[mask], rtol=0, atol=0)


def test_many_static_layouts_compile_once_without_shared_recompile_limit(monkeypatch):
	monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
	compile_function = torch.compile
	graphs = []

	def backend(graph, example_inputs):
		graphs.append(graph)
		return graph.forward

	def compile_counted(function, **kwargs):
		assert kwargs == {"fullgraph": True, "dynamic": False,
			"options": {"emulate_precision_casts": True}}
		kwargs.pop("options")  # This backend counts graphs rather than generating kernels.
		return compile_function(function, backend=backend, **kwargs)

	optimizers._compiled_newton_schulz.cache_clear()
	monkeypatch.setattr(torch, "compile", compile_counted)
	try:
		matrices = [torch.randn(2, rows, 17) for rows in range(3, 15)]
		for _ in range(2):
			for matrix in matrices:
				actual = ns(matrix, dtype=torch.float64, compiled=True)
				torch.testing.assert_close(actual, ns(matrix, dtype=torch.float64), rtol=0, atol=0)
		assert len(graphs) == len(matrices)  # More than Dynamo's default limit of 8.
		info = optimizers._compiled_newton_schulz.cache_info()
		assert info.misses == len(matrices) and info.hits == len(matrices)
		# A different stride, iteration count or arithmetic dtype needs a new graph.
		for matrix, kwargs in ((torch.randn(2, 6, 17)[:, :3], {}),
			(matrices[0], {"steps": 3}), (matrices[0], {"dtype": torch.float32})):
			kwargs.setdefault("dtype", torch.float64)
			ns(matrix, compiled=True, **kwargs)
		assert len(graphs) == len(matrices) + 3
	finally:
		optimizers._compiled_newton_schulz.cache_clear()


def test_cpu_optimizer_retains_eager_execution_and_skips_missing_gradients(monkeypatch):
	def forbidden_compile(*args, **kwargs):
		raise AssertionError("CPU optimizer should use the eager reference")

	monkeypatch.setattr(torch, "compile", forbidden_compile)
	active = torch.nn.Parameter(torch.randn(8, 7))
	inactive = torch.nn.Parameter(torch.randn(8, 7))
	before = inactive.detach().clone()
	optimizer = MuonAdamW([{"params": [active, inactive], "algorithm": "muon",
		"muon_shape": (2, 4, 7), "muon_split_sizes": (1, 3)}], ns_compile=True)
	active.grad = torch.randn_like(active)
	optimizer.step()
	assert inactive not in optimizer.state
	torch.testing.assert_close(inactive, before, rtol=0, atol=0)
	assert optimizer.param_groups[0]["ns_compile"] is True
	assert optimizer.state[active].keys() == {"momentum_buffer"}


def test_legacy_eager_optimizer_state_is_accepted_without_mutating_input():
	parameter = torch.nn.Parameter(torch.randn(4, 7))
	optimizer = MuonAdamW([{"params": [parameter], "algorithm": "muon"}])
	parameter.grad = torch.randn_like(parameter)
	optimizer.step()
	saved = copy.deepcopy(optimizer.state_dict())
	saved["param_groups"][0].pop("ns_compile")
	optimizer.load_state_dict(saved)
	assert optimizer.param_groups[0]["ns_compile"] is False
	assert "ns_compile" not in saved["param_groups"][0]
	compiled = MuonAdamW([{"params": [parameter], "algorithm": "muon"}], ns_compile=True)
	with pytest.raises(ValueError, match="compilation mode"):
		compiled.load_state_dict(saved)
	assert not compiled.state


def test_compile_flags_are_strict_and_disabled_compilation_is_not_reported_as_compiled(monkeypatch):
	parameter = torch.nn.Parameter(torch.randn(4, 7))
	with pytest.raises(TypeError, match="ns_compile"):
		MuonAdamW([{"params": [parameter], "algorithm": "muon"}], ns_compile="yes")
	with pytest.raises(TypeError, match="compiled"):
		ns(parameter, compiled=1)
	monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
	with pytest.raises(RuntimeError, match="TORCH_COMPILE_DISABLE"):
		ns(parameter, compiled=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_hybrid_compiled_updates_match_eager_with_splits_and_lr_changes(monkeypatch):
	monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
	torch.manual_seed(7)
	parameters = [torch.nn.Parameter(torch.randn(24, 7, device="cuda")),
		torch.nn.Parameter(torch.randn(3, device="cuda")),
		torch.nn.Parameter(torch.randn(24, 7, device="cuda"))]
	clones = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]

	def construct(values, compiled):
		return MuonAdamW([
			{"params": [values[0], values[2]], "algorithm": "muon", "muon_shape": (3, 8, 7),
			 "muon_split_sizes": (3, 5)}, {"params": [values[1]], "algorithm": "adamw"}],
			ns_compile=compiled, adamw_fused=True)

	eager, compiled = construct(parameters, False), construct(clones, True)
	for step in range(3):
		for optimizer in (eager, compiled):
			for group in optimizer.param_groups:
				group["lr"] = 3e-4 / (step + 1)
		for left, right in zip(parameters, clones):
			left.grad = torch.randn_like(left)
			right.grad = left.grad.clone()
		parameters[2].grad = clones[2].grad = None
		eager.step()
		compiled.step()
		torch.testing.assert_close(clones[0], parameters[0], rtol=0, atol=1e-4)
		torch.testing.assert_close(clones[1], parameters[1], rtol=0, atol=0)
		torch.testing.assert_close(compiled.state[clones[0]]["momentum_buffer"],
			eager.state[parameters[0]]["momentum_buffer"], rtol=0, atol=0)
	assert clones[2] not in compiled.state
