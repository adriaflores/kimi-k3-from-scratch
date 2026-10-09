import copy
import math

import pytest
import torch

from src.kimi_k3.training.optimizers import MuonAdamW, zeropower_via_newton_schulz


def _parameter(shape, dtype=torch.float64):
	return torch.nn.Parameter(torch.randn(shape, dtype=dtype))


def _svd_polynomial_reference(matrix, steps=5):
	"""Independent oracle: apply the scalar polynomial to singular values."""
	u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
	singular = singular / (torch.linalg.vector_norm(matrix) + 1e-7)
	for _ in range(steps):
		singular = 3.4445 * singular - 4.7750 * singular**3 + 2.0315 * singular**5
	return (u * singular) @ vh


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (4, 4)])
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_newton_schulz_matches_independent_svd_polynomial(shape, rank_deficient):
	matrix = torch.randn(shape, dtype=torch.float64)
	if rank_deficient:
		if shape[0] <= shape[1]:
			matrix[-1] = matrix[0]
		else:
			matrix[:, -1] = matrix[:, 0]
		assert torch.linalg.matrix_rank(matrix) == min(shape) - 1
	original = matrix.clone()
	actual = zeropower_via_newton_schulz(matrix, dtype=torch.float64)
	expected = _svd_polynomial_reference(matrix)
	torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
	torch.testing.assert_close(matrix, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3, 4, 7), (3, 7, 4)])
def test_batched_matrices_are_normalized_and_processed_independently(dtype, shape):
	matrix = torch.randn(shape)
	matrix[0] *= 100  # A batch-wide norm would change the other matrices.
	original = matrix.clone()
	actual = zeropower_via_newton_schulz(matrix, dtype=dtype)
	flat = matrix.reshape(-1, *shape[-2:])
	expected = torch.stack([
		zeropower_via_newton_schulz(item, dtype=dtype) for item in flat
	]).reshape(shape)
	assert actual.dtype == dtype
	# Batched and individual GEMMs may reduce sums in a different order.
	tolerances = {torch.bfloat16: (0.03, 0.03), torch.float32: (1e-5, 1e-6), torch.float64: (1e-10, 1e-12)}
	rtol, atol = tolerances[dtype]
	torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
	altered = matrix.clone()
	altered[0] *= 10
	independent = zeropower_via_newton_schulz(altered, dtype=dtype)
	torch.testing.assert_close(actual[1:], independent[1:], rtol=0, atol=0)
	torch.testing.assert_close(matrix, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_zero_update_stays_zero_and_finite(dtype):
	actual = zeropower_via_newton_schulz(torch.zeros(3, 2, 5), dtype=dtype)
	assert torch.isfinite(actual).all()
	assert torch.count_nonzero(actual) == 0


def test_requested_ns_dtype_is_preserved_under_ambient_autocast():
	matrix = torch.randn(5, 3)
	expected = zeropower_via_newton_schulz(matrix, dtype=torch.float32)
	with torch.autocast("cpu", dtype=torch.bfloat16):
		actual = zeropower_via_newton_schulz(matrix, dtype=torch.float32)
	assert actual.dtype == torch.float32
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (4, 4)])
@pytest.mark.parametrize("nesterov", [False, True])
def test_muon_matches_momentum_svd_update_and_unscaled_weight_decay(shape, nesterov):
	parameter = _parameter(shape)
	expected = parameter.detach().clone()
	momentum = torch.zeros_like(expected)
	lr, beta, decay = 0.012, 0.8, 0.3
	optimizer = MuonAdamW([{"params": [parameter], "algorithm": "muon"}],
		lr=lr, momentum=beta, nesterov=nesterov, weight_decay=decay, ns_dtype=torch.float64)
	for _ in range(4):
		gradient = torch.randn_like(parameter)
		parameter.grad = gradient.clone()
		momentum = beta * momentum + gradient
		direction = gradient + beta * momentum if nesterov else momentum
		update = _svd_polynomial_reference(direction)
		expected = expected * (1 - lr * decay) - lr * 0.2 * math.sqrt(max(shape)) * update
		optimizer.step()
		torch.testing.assert_close(parameter, expected, rtol=1e-10, atol=1e-10)
		torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], momentum)
		torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_auxiliary_groups_match_native_adamw_and_keep_per_parameter_steps(fused, dtype):
	parameters = [_parameter((3, 4), dtype), _parameter((4,), dtype), _parameter((), dtype)]
	reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
	kwargs = {"lr": 0.008, "weight_decay": 0.2, "betas": (0.7, 0.93), "eps": 2e-7}
	optimizer = MuonAdamW([{"params": parameters, "algorithm": "adamw"}],
		adamw_fused=fused, **kwargs)
	native = torch.optim.AdamW(reference, fused=fused, **kwargs)
	for step in range(5):
		for index, (parameter, other) in enumerate(zip(parameters, reference)):
			gradient = None if index == 1 and step % 2 == 0 else torch.randn_like(parameter)
			parameter.grad = gradient
			other.grad = None if gradient is None else gradient.clone()
		optimizer.step()
		native.step()
	for parameter, other in zip(parameters, reference):
		torch.testing.assert_close(parameter, other, rtol=0, atol=0)
		for key in ("step", "exp_avg", "exp_avg_sq"):
			torch.testing.assert_close(optimizer.state[parameter][key], native.state[other][key], rtol=0, atol=0)
	assert optimizer.state[parameters[1]]["step"].item() == 2


def test_none_grad_preserves_weights_and_existing_state_but_zero_grad_is_active():
	muon, adam, never_muon, never_adam = [_parameter((2, 3)) for _ in range(4)]
	optimizer = MuonAdamW([
		{"params": [muon, never_muon], "algorithm": "muon"},
		{"params": [adam, never_adam], "algorithm": "adamw"},
	], lr=0.02, weight_decay=0.4, ns_dtype=torch.float64)
	for parameter in (muon, adam):
		parameter.grad = torch.ones_like(parameter)
	optimizer.step()
	optimizer.zero_grad(set_to_none=True)
	weights = [parameter.detach().clone() for parameter in (muon, adam, never_muon, never_adam)]
	saved = copy.deepcopy(optimizer.state_dict())
	optimizer.step()
	_assert_tree_equal(optimizer.state_dict(), saved)
	for parameter, previous in zip((muon, adam, never_muon, never_adam), weights):
		torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
		assert parameter.grad is None
	assert never_muon not in optimizer.state and never_adam not in optimizer.state
	for parameter in (muon, adam):
		parameter.grad = torch.zeros_like(parameter)
	optimizer.step()
	assert not torch.equal(muon, weights[0]) and not torch.equal(adam, weights[1])
	assert optimizer.state[adam]["step"].item() == 2


def _assert_tree_equal(actual, expected):
	if isinstance(expected, torch.Tensor):
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	elif isinstance(expected, dict):
		assert actual.keys() == expected.keys()
		for key in expected:
			_assert_tree_equal(actual[key], expected[key])
	elif isinstance(expected, (tuple, list)):
		assert len(actual) == len(expected)
		for item, other in zip(actual, expected):
			_assert_tree_equal(item, other)
	else:
		assert actual == expected


@pytest.mark.parametrize("fused", [False, True])
def test_one_checkpoint_restores_both_algorithms_and_exact_continuation(tmp_path, fused):
	parameters = [_parameter((3, 5), torch.float32) for _ in range(4)]

	def build(items):
		return MuonAdamW([
			{"params": items[:2], "algorithm": "muon"},
			{"params": items[2:], "algorithm": "adamw", "weight_decay": 0.0},
		], lr=0.01, weight_decay=0.1, adamw_fused=fused)

	optimizer = build(parameters)
	gradients = [[torch.randn_like(parameter) for parameter in parameters] for _ in range(7)]

	def advance(opt, items, step):
		for index, parameter in enumerate(items):
			parameter.grad = None if index % 2 and step % 3 == 0 else gradients[step][index].clone()
		opt.step()
		opt.zero_grad(set_to_none=True)

	for step in range(3):
		advance(optimizer, parameters, step)
	path = tmp_path / "optimizer.pt"
	torch.save({"weights": [parameter.detach().clone() for parameter in parameters],
		"optimizer": optimizer.state_dict()}, path)
	saved = torch.load(path, weights_only=True)
	resumed = [torch.nn.Parameter(value.clone()) for value in saved["weights"]]
	restored = build(resumed)
	restored.load_state_dict(saved["optimizer"])
	for step in range(3, 7):
		advance(optimizer, parameters, step)
		advance(restored, resumed, step)
	for actual, expected in zip(resumed, parameters):
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	_assert_tree_equal(restored.state_dict(), optimizer.state_dict())


def test_group_lr_changes_and_scheduler_reach_native_adamw():
	muon, adam = _parameter((2, 2)), _parameter((2,))
	reference = torch.nn.Parameter(adam.detach().clone())
	optimizer = MuonAdamW([
		{"params": [muon], "algorithm": "muon"},
		{"params": [adam], "algorithm": "adamw", "lr": 0.03},
	], lr=0.02, adamw_fused=False)
	native = torch.optim.AdamW([reference], lr=0.03, weight_decay=0.1, betas=(0.9, 0.95))
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)
	native_scheduler = torch.optim.lr_scheduler.LambdaLR(native, lambda step: 0.5**step)
	for _ in range(4):
		muon.grad = torch.ones_like(muon)
		adam.grad = reference.grad = torch.ones_like(adam)
		optimizer.step()
		native.step()
		scheduler.step()
		native_scheduler.step()
		torch.testing.assert_close(adam, reference, rtol=0, atol=0)
		assert optimizer.param_groups[0]["lr"] == 0.02 * 0.5**scheduler.last_epoch
		assert optimizer.param_groups[1]["lr"] == native.param_groups[0]["lr"]


def test_adding_a_group_preserves_existing_adamw_state_and_starts_new_state_lazily():
	first, added = _parameter((3,)), _parameter((3,))
	optimizer = MuonAdamW([{"params": [first], "algorithm": "adamw"}])
	first.grad = torch.ones_like(first)
	optimizer.step()
	optimizer.add_param_group({"params": [added], "algorithm": "adamw", "lr": 0.02})
	assert added not in optimizer.state
	added.grad = torch.ones_like(added)
	optimizer.step()
	assert optimizer.state[first]["step"].item() == 2
	assert optimizer.state[added]["step"].item() == 1
	assert optimizer.param_groups[1]["lr"] == 0.02


def test_closure_runs_once_with_gradients_enabled_and_returns_loss():
	muon, adam = _parameter((2, 2)), _parameter((2,))
	optimizer = MuonAdamW([
		{"params": [muon], "algorithm": "muon"},
		{"params": [adam], "algorithm": "adamw"},
	])
	calls = []

	def closure():
		assert torch.is_grad_enabled()
		optimizer.zero_grad(set_to_none=True)
		loss = muon.square().sum() + adam.square().sum()
		loss.backward()
		calls.append(loss)
		return loss

	assert optimizer.step(closure) is calls[0]
	assert len(calls) == 1


@pytest.mark.parametrize("overflow_group", ["muon", "adamw"])
def test_grad_scaler_overflow_skips_the_entire_hybrid_update(overflow_group):
	muon, adam = _parameter((2, 3), torch.float32), _parameter((3,), torch.float32)
	optimizer = MuonAdamW([
		{"params": [muon], "algorithm": "muon"},
		{"params": [adam], "algorithm": "adamw"},
	])
	scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
	weights = [parameter.detach().clone() for parameter in (muon, adam)]
	for overflow in (False, True, False):
		optimizer.zero_grad(set_to_none=True)
		loss = muon.square().sum() + adam.square().sum()
		scaler.scale(loss).backward()
		if overflow:
			parameter = muon if overflow_group == "muon" else adam
			parameter.grad.reshape(-1)[0] = float("inf")
			weights = [item.detach().clone() for item in (muon, adam)]
			saved = copy.deepcopy(optimizer.state_dict())
		scaler.step(optimizer)
		scaler.update()
		if overflow:
			for parameter, expected in zip((muon, adam), weights):
				torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
			_assert_tree_equal(optimizer.state_dict(), saved)
	assert scaler.get_scale() == 4.0
	assert optimizer.state[adam]["step"].item() == 2


def test_small_network_learns_with_the_hybrid_optimizer():
	model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
	optimizer = MuonAdamW([
		{"params": [parameter for parameter in model.parameters() if parameter.ndim == 2], "algorithm": "muon"},
		{"params": [parameter for parameter in model.parameters() if parameter.ndim == 1], "algorithm": "adamw"},
	], lr=0.03, weight_decay=0.0, ns_dtype=torch.float32)
	x = torch.randn(32, 3)
	target = torch.stack((x[:, 0] + 0.5 * x[:, 1], x[:, 2] - x[:, 1]), dim=-1)
	initial = torch.nn.functional.mse_loss(model(x), target).item()
	for _ in range(100):
		optimizer.zero_grad(set_to_none=True)
		loss = torch.nn.functional.mse_loss(model(x), target)
		assert torch.isfinite(loss)
		loss.backward()
		optimizer.step()
	assert torch.nn.functional.mse_loss(model(x), target).item() < initial * 0.1


@pytest.mark.parametrize("kwargs,algorithm", [
	({"lr": -1.0}, "muon"), ({"weight_decay": float("nan")}, "muon"),
	({"momentum": 1.0}, "muon"), ({"momentum": -0.1}, "muon"),
	({"ns_steps": 0}, "muon"), ({"ns_steps": True}, "muon"),
	({"ns_dtype": torch.float16}, "muon"), ({"nesterov": "yes"}, "muon"),
	({"betas": (0.9, 1.0)}, "adamw"), ({"eps": 0}, "adamw"),
	({"adamw_fused": "yes"}, "adamw"),
])
def test_invalid_optimizer_hyperparameters_are_rejected(kwargs, algorithm):
	with pytest.raises((TypeError, ValueError)):
		MuonAdamW([{"params": [_parameter((2, 3))], "algorithm": algorithm}], **kwargs)


@pytest.mark.parametrize("algorithm", [None, "sgd"])
def test_group_algorithms_must_be_explicit_and_supported(algorithm):
	group = {"params": [_parameter((2, 3))]}
	if algorithm is not None:
		group["algorithm"] = algorithm
	with pytest.raises(ValueError, match="algorithm"):
		MuonAdamW([group])


@pytest.mark.parametrize("shape", [(3,), (2, 3, 4), (0, 3)])
def test_muon_rejects_implicit_vector_or_tensor_flattening(shape):
	with pytest.raises(ValueError, match="nonempty 2D"):
		MuonAdamW([{"params": [_parameter(shape)], "algorithm": "muon"}])


@pytest.mark.parametrize("across_groups", [False, True])
def test_duplicate_parameters_are_rejected(across_groups):
	parameter = _parameter((2, 3))
	groups = [{"params": [parameter, parameter], "algorithm": "muon"}]
	if across_groups:
		groups = [{"params": [parameter], "algorithm": algorithm} for algorithm in ("muon", "adamw")]
	with pytest.raises(ValueError, match="only once"):
		MuonAdamW(groups)


@pytest.mark.parametrize("batched", [False, True])
def test_sparse_gradient_is_rejected_before_any_parameter_changes(batched):
	muon, adam = _parameter((2, 3)), _parameter((2, 3))
	optimizer = MuonAdamW([
		{"params": [muon], "algorithm": "muon"},
		{"params": [adam], "algorithm": "adamw"},
	], ns_batched=batched)
	muon.grad = torch.ones_like(muon)
	adam.grad = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.ones(1, dtype=adam.dtype), (2, 3), check_invariants=True)
	expected = muon.detach().clone()
	with pytest.raises(RuntimeError, match="sparse"):
		optimizer.step()
	torch.testing.assert_close(muon, expected, rtol=0, atol=0)
	assert not optimizer.state


def test_loading_state_cannot_silently_change_a_group_algorithm():
	parameter = _parameter((2, 3))
	optimizer = MuonAdamW([{"params": [parameter], "algorithm": "muon"}])
	saved = copy.deepcopy(optimizer.state_dict())
	saved["param_groups"][0]["algorithm"] = "adamw"
	with pytest.raises(ValueError, match="algorithms"):
		optimizer.load_state_dict(saved)
	assert optimizer.param_groups[0]["algorithm"] == "muon"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_autocast_and_grad_scaler_match_an_unscaled_hybrid_step():
	parameters = [_parameter((3, 5), torch.float32).to("cuda") for _ in range(2)]
	parameters = [torch.nn.Parameter(parameter) for parameter in parameters]
	reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]

	def build(items):
		return MuonAdamW([{"params": [parameter], "algorithm": algorithm}
			for parameter, algorithm in zip(items, ("muon", "adamw"))])

	optimizer, native = build(parameters), build(reference)
	scaler = torch.amp.GradScaler("cuda", init_scale=8)
	x = torch.randn(4, 5, device="cuda")
	with torch.autocast("cuda", dtype=torch.float16):
		loss = sum((x @ parameter.T).float().square().mean() for parameter in parameters)
		other_loss = sum((x @ parameter.T).float().square().mean() for parameter in reference)
	scaler.scale(loss).backward()
	other_loss.backward()
	scaler.step(optimizer)
	scaler.update()
	native.step()
	for actual, expected in zip(parameters, reference):
		torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
