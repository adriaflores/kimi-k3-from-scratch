"""Single-device Muon with auxiliary AdamW and one PyTorch optimizer state.

The Newton-Schulz kernel and momentum convention follow MoonshotAI/Moonlight
and KellerJordan/Muon. The shape scaling follows Moonlight, Eq. (4), rather
than the original Muon learning-rate convention. Upstream MIT notices and
pinned source revisions are in references/muon/NOTICE.md.

This module requires explicit parameter groups. Model-specific assignment is
in parameter_groups.py; the trainer applies QK-Clip after successful updates.
"""

import math
import os
from collections.abc import Iterable
from functools import lru_cache

import torch

_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_NS_DTYPES = (torch.bfloat16, torch.float32, torch.float64)
_PARAMETER_DTYPES = (torch.float32, torch.float64)


def _nonnegative(name, value):
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise TypeError(f"{name} must be a real number")
	if not math.isfinite(value) or value < 0:
		raise ValueError(f"{name} must be finite and nonnegative")


def _validate_ns(steps, eps, dtype):
	if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
		raise ValueError("ns_steps must be a positive integer")
	_nonnegative("ns_eps", eps)
	if eps == 0:
		raise ValueError("ns_eps must be positive")
	if dtype not in _NS_DTYPES:
		raise ValueError("ns_dtype must be bfloat16, float32 or float64")


def _newton_schulz_core(matrix, steps, eps, dtype):
	"""Pure tensor arithmetic shared by the eager and compiled routes."""
	x = matrix.to(dtype=dtype)
	transposed = x.size(-2) > x.size(-1)
	if transposed:
		x = x.mT
	# The Frobenius norm bounds the spectral norm. Normalize each matrix.
	x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
	a, b, c = _NS_COEFFICIENTS
	for _ in range(steps):
		gram = x @ x.mT
		polynomial = b * gram + c * gram @ gram
		x = a * x + polynomial @ x
	return x.mT if transposed else x


@lru_cache(maxsize=128)
def _compiled_newton_schulz(shape, stride, device, input_dtype, steps, eps, dtype):
	"""Cache a static full graph per layout, device and arithmetic setting.

	Dynamo caches graphs on Python code objects, not function identities. Give
	each specialization its own code object so the model's many matrix shapes
	do not exhaust one function's recompile limit. This avoids changing global
	compiler settings. Shape/stride/device/input_dtype are cache keys only.
	Compiled functions are process-local and never enter optimizer state.
	"""
	def kernel(matrix):
		return _newton_schulz_core(matrix, steps, eps, dtype)

	kernel.__code__ = kernel.__code__.replace()
	# Preserve eager BF16 truncation between fused pointwise operations. These
	# casts remain inside the generated kernel, without extra launch boundaries.
	return torch.compile(kernel, fullgraph=True, dynamic=False,
		options={"emulate_precision_casts": True})


@torch.no_grad()
def zeropower_via_newton_schulz(matrix, *, steps=5, eps=1e-7,
							   dtype=torch.bfloat16, compiled=False):
	"""Apply Muon's quintic polynomial independently to the last two axes.

	BF16 is the upstream working dtype; FP32/FP64 are useful for numerical
	verification. Five iterations approximate polar orthogonalization: this
	is not an exact SVD/polar factor and its singular values need not be one.
	Leading axes denote independent matrices, never one flattened matrix.
	The input tensor is not modified and ambient autocast is disabled.
	``compiled=True`` lazily compiles just this core with TorchInductor. It is
	also available on CPU for verification; the optimizer enables it on CUDA
	only. Compilation errors propagate instead of silently using eager code.
	"""
	_validate_ns(steps, eps, dtype)
	if not isinstance(compiled, bool):
		raise TypeError("compiled must be a bool")
	if not isinstance(matrix, torch.Tensor) or matrix.ndim < 2:
		raise ValueError("Muon orthogonalization requires at least two dimensions")
	if matrix.layout != torch.strided or not matrix.is_floating_point():
		raise TypeError("Muon orthogonalization requires a dense real floating tensor")
	if matrix.numel() == 0:
		raise ValueError("Muon orthogonalization requires nonempty matrices")
	with torch.autocast(device_type=matrix.device.type, enabled=False):
		if compiled:
			if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
				raise RuntimeError("Compiled Muon requested with TORCH_COMPILE_DISABLE=1")
			kernel = _compiled_newton_schulz(tuple(matrix.shape), tuple(matrix.stride()),
				matrix.device, matrix.dtype, steps, eps, dtype)
			return kernel(matrix)
		return _newton_schulz_core(matrix, steps, eps, dtype)


class MuonAdamW(torch.optim.Optimizer):
	"""One optimizer with explicit ``algorithm='muon'`` / ``'adamw'`` groups.

	Muon uses unnormalized SGD momentum, optional Nesterov, Newton-Schulz,
	and the Moonlight factor ``0.2 * sqrt(max(rows, cols))``. Weight decay
	uses the group's *unscaled* learning rate. A missing gradient skips both
	decay and all state changes. A zero gradient is an ordinary active step.

	Auxiliary groups use native torch.optim.AdamW, including its fused path.
	The native optimizer shares the public groups and state of this object;
	there is no second checkpoint or scheduler. Construct after moving the
	model to its device. Parameters must be FP32 (or FP64 for verification);
	BF16/FP16 training uses autocast and FP32 parameter storage.

	Example::

		optimizer = MuonAdamW([
			{"params": hidden_matrices, "algorithm": "muon"},
			{"params": auxiliary_parameters, "algorithm": "adamw"},
		], lr=3e-4, weight_decay=0.1, adamw_fused=True)

	Muon parameters have 2D storage. A group's optional ``muon_shape`` declares
	a view of independent matrices, e.g. ``(2, intermediate, hidden)`` for fused
	FFN gate/up. Its product must match each parameter's size. Matrix dimensions
	for RMS scaling are those of the view, rather than the fused storage shape.
	``muon_split_sizes`` optionally partitions the view's row axis (-2), e.g.
	``(key_dim, value_dim)`` for MLA's interleaved K/V rows within each head.
	Each part has independent normalization, NS iterations and shape scaling;
	momentum and decay still belong to the original parameter exactly once.
	``ns_compile=True`` compiles the NS core on CUDA; CPU execution stays eager.
	It does not compile momentum, decay, AdamW, the scheduler or QK-Clip.
	"""

	def __init__(self, param_groups: Iterable[dict], *, lr=3e-4,
				 weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5,
				 ns_dtype=torch.bfloat16, ns_compile=False, betas=(0.9, 0.95), eps=1e-8,
				 adamw_fused=True):
		_nonnegative("lr", lr)
		_nonnegative("weight_decay", weight_decay)
		self._muon_defaults = {"momentum": momentum, "nesterov": nesterov,
							   "ns_steps": ns_steps, "ns_dtype": ns_dtype, "ns_compile": ns_compile}
		self._adamw_defaults = {"betas": betas, "eps": eps, "fused": adamw_fused}
		self._initializing = True
		self._adamw = None
		super().__init__(param_groups, {"lr": lr, "weight_decay": weight_decay})
		self._initializing = False
		self._bind_adamw()

	def _validate_group(self, group):
		algorithm = group.get("algorithm")
		if algorithm not in ("muon", "adamw"):
			raise ValueError("Each parameter group needs algorithm='muon' or 'adamw'")
		_nonnegative("lr", group["lr"])
		_nonnegative("weight_decay", group["weight_decay"])
		if algorithm == "muon":
			_nonnegative("momentum", group["momentum"])
			if group["momentum"] >= 1:
				raise ValueError("momentum must be in [0, 1)")
			if not isinstance(group["nesterov"], bool):
				raise TypeError("nesterov must be a bool")
			_validate_ns(group["ns_steps"], 1e-7, group["ns_dtype"])
			if not isinstance(group["ns_compile"], bool):
				raise TypeError("ns_compile must be a bool")
			view = group.get("muon_shape")
			if view is not None and (not isinstance(view, (tuple, list)) or len(view) < 2
					or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in view)):
				raise ValueError("muon_shape must contain at least two positive integer dimensions")
			splits = group.get("muon_split_sizes")
			if splits is not None and (not isinstance(splits, (tuple, list)) or len(splits) < 2
					or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in splits)):
				raise ValueError("muon_split_sizes must contain at least two positive integer row sizes")
		else:
			if group.get("muon_shape") is not None or group.get("muon_split_sizes") is not None:
				raise ValueError("Muon matrix views are only supported in Muon groups")
			betas = group["betas"]
			if not isinstance(betas, (tuple, list)) or len(betas) != 2:
				raise ValueError("betas must contain two values in [0, 1)")
			for beta in betas:
				_nonnegative("beta", beta)
				if beta >= 1:
					raise ValueError("betas must be in [0, 1)")
			_nonnegative("eps", group["eps"])
			if group["eps"] == 0:
				raise ValueError("eps must be positive")
			if not isinstance(group["fused"], bool):
				raise TypeError("fused must be a bool")
			# These variants change the supported recipe or execution contract.
			for flag in ("amsgrad", "maximize", "capturable", "differentiable"):
				if group.get(flag, False):
					raise ValueError(f"AdamW {flag}=True is not supported")
		for parameter in group["params"]:
			if not isinstance(parameter, torch.Tensor):
				raise TypeError("Parameter groups must contain tensors")
			if parameter.dtype not in _PARAMETER_DTYPES or parameter.layout != torch.strided:
				raise TypeError("MuonAdamW needs dense FP32/FP64 parameters; use autocast for mixed precision")
			if algorithm == "muon" and (parameter.ndim != 2 or parameter.numel() == 0):
				raise ValueError("Muon groups require nonempty 2D parameters")
			if (algorithm == "muon" and group.get("muon_shape") is not None
					and math.prod(group["muon_shape"]) != parameter.numel()):
				raise ValueError("muon_shape must match each parameter's number of elements")
			if (algorithm == "muon" and group.get("muon_split_sizes") is not None
					and sum(group["muon_split_sizes"]) != (group.get("muon_shape") or parameter.shape)[-2]):
				raise ValueError("muon_split_sizes must sum to the Muon view's row dimension")

	def add_param_group(self, param_group):
		"""Preserve input ordering and reject duplicates, including within a group."""
		if not isinstance(param_group, dict):
			raise TypeError("MuonAdamW requires explicit parameter-group dictionaries")
		group = dict(param_group)
		parameters = group["params"]
		group["params"] = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
		algorithm = group.get("algorithm")
		specific = self._muon_defaults if algorithm == "muon" else self._adamw_defaults
		for name, value in {**self.defaults, **specific}.items():
			group.setdefault(name, value)
		self._validate_group(group)
		identities = [id(parameter) for parameter in group["params"]]
		existing = {id(parameter) for other in self.param_groups for parameter in other["params"]}
		if len(identities) != len(set(identities)) or existing.intersection(identities):
			raise ValueError("A parameter may occur only once across all optimizer groups")
		super().add_param_group(group)
		if not self._initializing:
			self._bind_adamw()

	def _bind_adamw(self):
		"""Rebind after construction, adding groups, or loading canonical state."""
		groups = [group for group in self.param_groups if group["algorithm"] == "adamw"]
		self._adamw = torch.optim.AdamW(groups) if groups else None
		if self._adamw is not None:
			self._adamw.state = self.state

	@staticmethod
	def _normalize_state_dict(state_dict):
		"""Version-7 checkpoints written before compilation used the eager core."""
		groups = [dict(group) for group in state_dict["param_groups"]]
		for group in groups:
			if group.get("algorithm") == "muon":
				group.setdefault("ns_compile", False)
		return {**state_dict, "param_groups": groups}

	def load_state_dict(self, state_dict):
		"""Keep auxiliary AdamW bound to the groups/state replaced by PyTorch."""
		state_dict = self._normalize_state_dict(state_dict)
		saved = state_dict["param_groups"]
		if len(saved) != len(self.param_groups):
			raise ValueError("Incompatible number of optimizer parameter groups")
		for previous, current in zip(saved, self.param_groups):
			if previous.get("algorithm") != current["algorithm"]:
				raise ValueError("Incompatible optimizer group algorithms")
			if len(previous["params"]) != len(current["params"]):
				raise ValueError("Incompatible optimizer parameter-group sizes")
			self._validate_group({**previous, "params": current["params"]})
			if previous.get("ns_compile") != current.get("ns_compile"):
				raise ValueError("Incompatible Newton-Schulz compilation mode (ns_compile)")
			for key in ("muon_shape", "muon_split_sizes"):
				old, new = previous.get(key), current.get(key)
				if (None if old is None else tuple(old)) != (None if new is None else tuple(new)):
					raise ValueError(f"Incompatible optimizer matrix views ({key})")
		super().load_state_dict(state_dict)
		self._bind_adamw()

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()
		# Reject unsupported gradients before updating any group's parameters.
		for group in self.param_groups:
			for parameter in group["params"]:
				if parameter.grad is not None and parameter.grad.layout != torch.strided:
					raise RuntimeError("MuonAdamW does not support sparse gradients")
		for group in self.param_groups:
			if group["algorithm"] != "muon":
				continue
			for parameter in group["params"]:
				gradient = parameter.grad
				if gradient is None:
					continue
				state = self.state[parameter]
				if not state:
					state["momentum_buffer"] = torch.zeros_like(parameter)
				buffer = state["momentum_buffer"]
				buffer.mul_(group["momentum"]).add_(gradient)
				update = gradient.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
				shape = group.get("muon_shape") or parameter.shape
				update = update.reshape(shape)
				compiled = group["ns_compile"] and parameter.device.type == "cuda"
				splits = group.get("muon_split_sizes")
				if splits is None:
					update = zeropower_via_newton_schulz(update, steps=group["ns_steps"], dtype=group["ns_dtype"], compiled=compiled)
					scale = 0.2 * math.sqrt(max(shape[-2:]))
				else:
					# Copy NS results into parameter precision before scaling. This
					# also avoids updating a reshape copy of noncontiguous storage.
					assembled = torch.empty(shape, dtype=parameter.dtype, device=parameter.device)
					for part, destination in zip(update.split(splits, dim=-2), assembled.split(splits, dim=-2)):
						destination.copy_(zeropower_via_newton_schulz(part, steps=group["ns_steps"], dtype=group["ns_dtype"], compiled=compiled))
						destination.mul_(0.2 * math.sqrt(max(part.shape[-2:])))
					update, scale = assembled, 1.0
				parameter.mul_(1 - group["lr"] * group["weight_decay"])
				parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"] * scale)
		if self._adamw is not None:
			self._adamw.step()
		return loss
