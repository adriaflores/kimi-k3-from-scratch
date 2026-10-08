"""Kimi K2 QK-Clip, translated to this backbone's NoPE MLA packing.

Observe causal QK/sqrt(d) during the training forward; rescale unshared Q/K
rows after a successful optimizer update. The shared key and V stay intact.
KDA's L2-normalized Q/K have bounded dot products and need no softmax clipping.
"""

import math
from functools import partial

import torch
from torch import nn


def _positive(name, value):
	if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
		raise ValueError(f"{name} must be finite and positive")


def _chunk_size(value):
	if isinstance(value, bool) or not isinstance(value, int) or value < 1:
		raise ValueError("qk_clip_chunk_size must be a positive integer")


@torch.no_grad()
def causal_max_logits(query, key, *, chunk_size=256):
	"""Detached per-head maxima for [batch, heads, sequence, qk_dimension].

	Query chunks attend to their entire causal key prefix. Temporary score
	storage is O(B*H*chunk_size*T), rather than a full long-context T*T matrix.
	This still computes the exact causal dot products and adds scoring work.
	FP16/BF16 inputs are promoted to FP32; FP64 is retained for verification.
	"""
	_chunk_size(chunk_size)
	if (query.ndim != 4 or query.shape != key.shape or query.numel() == 0
			or query.device != key.device or query.dtype != key.dtype
			or not query.is_floating_point() or not key.is_floating_point()):
		raise ValueError("QK-Clip needs matching nonempty floating [B,H,T,D] query/key tensors")
	dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
	with torch.autocast(device_type=query.device.type, enabled=False):
		q, k = query.detach().to(dtype), key.detach().to(dtype)
		length = q.shape[-2]
		width = min(chunk_size, length)
		future = torch.ones(width, width, dtype=torch.bool, device=q.device).triu(1)
		maximum = torch.full((q.shape[1],), -torch.inf, dtype=dtype, device=q.device)
		for start in range(0, length, chunk_size):
			end = min(start + chunk_size, length)
			scores = q[:, :, start:end] @ k[:, :, :end].mT
			# All earlier key blocks are valid. Only this final square contains
			# future positions, including the shorter final query chunk.
			scores[..., start:end].masked_fill_(future[:end - start, :end - start], -torch.inf)
			maximum = torch.maximum(maximum, scores.amax(dim=(0, 2, 3)))
			del scores
		return maximum / math.sqrt(q.shape[-1])


def _mla_sites(model):
	"""Structural matching also works with both supported package import paths."""
	for name, module in model.named_modules():
		if not all(isinstance(getattr(module, field, None), nn.Linear) for field in ("W_UQ", "W_UKV", "W_DKV")):
			continue
		h, d, ds, dv = module.num_heads, module.d_head, module.d_shared, module.dv_head
		if (any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in (h, d, dv))
				or isinstance(ds, bool) or not isinstance(ds, int) or ds < 0
				or tuple(module.W_UQ.weight.shape) != (h * (d + ds), module.q_latent_dim)
				or tuple(module.W_UKV.weight.shape) != (h * (d + dv), module.kv_latent_dim)
				or tuple(module.W_DKV.weight.shape) != (module.kv_latent_dim + ds, module.d_model)):
			raise ValueError("MLA weights do not match their declared QK-Clip head geometry")
		yield name, module


def qk_clip_metadata(model, *, threshold, chunk_size):
	"""Plain policy/geometry data, checked before checkpoint state is loaded."""
	_positive("qk_clip_threshold", threshold)
	_chunk_size(chunk_size)
	return {"policy": "mla_shared_key_v1", "score": "causal_qk_over_sqrt_dim",
		"threshold": threshold, "chunk_size": chunk_size,
		"layers": [{"name": name, "num_heads": module.num_heads,
			"key_dim": module.d_head, "shared_dim": module.d_shared, "value_dim": module.dv_head,
			"query_rank": module.q_latent_dim, "kv_rank": module.kv_latent_dim}
			for name, module in _mla_sites(model)]}


class QKClip:
	"""One ephemeral observer for all MLA layers in a trainer invocation.

	Reset at each optimizer attempt; microbatch maxima are combined with max,
	not averaged. Apply only after successful steps, while gradients still
	exist, and discard skipped attempts. No moving statistic or momentum
	rescaling is used. Checkpoints at update boundaries need only the policy
	and the already-clipped model/optimizer state, not pending observations.
	"""

	def __init__(self, model, *, threshold=100.0, chunk_size=256):
		_positive("qk_clip_threshold", threshold)
		_chunk_size(chunk_size)
		self.threshold, self.chunk_size = threshold, chunk_size
		self.sites = dict(_mla_sites(model))
		self.max_logits = {}
		self._validated = False
		self._attachments = []
		self._window = {}

	def __enter__(self):
		try:
			for name, module in self.sites.items():
				previous = getattr(module, "_qk_clip_observer", None)
				if previous is not None:
					raise RuntimeError("An MLA QK-Clip observer is already attached")
				self._attachments.append((module, hasattr(module, "_qk_clip_observer"), previous))
				module._qk_clip_observer = partial(self.observe, name)
		except Exception:
			self.__exit__(None, None, None)
			raise
		return self

	def __exit__(self, *_):
		for module, had_attribute, previous in reversed(self._attachments):
			if had_attribute:
				module._qk_clip_observer = previous
			else:
				del module._qk_clip_observer
		self._attachments.clear()
		self.reset()

	def reset(self):
		self.max_logits.clear()
		self._validated = False

	@torch.no_grad()
	def observe(self, name, query, key):
		with torch.profiler.record_function("attention/qk_clip_stats"):
			maximum = causal_max_logits(query, key, chunk_size=self.chunk_size)
		previous = self.max_logits.get(name)
		self.max_logits[name] = maximum if previous is None else torch.maximum(previous, maximum)
		self._validated = False

	@staticmethod
	def _active(module):
		# Frozen or unused attention parameters retain the same skip contract
		# as the optimizer. Both unshared projections are needed by this rule.
		return all(weight.requires_grad and weight.grad is not None
			for weight in (module.W_UQ.weight, module.W_UKV.weight))

	def validate(self):
		if self._validated:
			return
		for name, maximum in self.max_logits.items():
			if self._active(self.sites[name]) and not torch.isfinite(maximum).all().item():
				raise FloatingPointError(f"Non-finite attention max logit in {name}")
		self._validated = True

	@torch.no_grad()
	def apply(self):
		"""Scale only content Q/K and the query paired with the shared key."""
		self.validate()  # Validate every layer before modifying any clip weight.
		for name, maximum in self.max_logits.items():
			module = self.sites[name]
			if not self._active(module):
				continue
			gamma = self.threshold / maximum.clamp_min(self.threshold)
			triggered = maximum > self.threshold
			if triggered.any().item():
				q = module.W_UQ.weight.unflatten(0, (module.num_heads, module.d_head + module.d_shared))
				kv = module.W_UKV.weight.unflatten(0, (module.num_heads, module.d_head + module.dv_head))
				factors = gamma.to(q.dtype).view(-1, 1, 1)
				root = factors.sqrt()
				q[:, :module.d_head].mul_(root)
				q[:, module.d_head:].mul_(factors)
				kv[:, :module.d_head].mul_(root)
			previous = self._window.get(name)
			if previous is None:
				self._window[name] = {"max_logits": maximum.clone(), "minimum_gamma": gamma.clone(),
					"clipped_head_updates": triggered.to(torch.int64), "updates": 1}
			else:
				previous["max_logits"] = torch.maximum(previous["max_logits"], maximum)
				previous["minimum_gamma"] = torch.minimum(previous["minimum_gamma"], gamma)
				previous["clipped_head_updates"].add_(triggered)
				previous["updates"] += 1
		self.reset()

	def summary(self, *, reset=False):
		"""Successful-update diagnostics; CPU transfer only at a log boundary."""
		layers = {}
		for name, window in self._window.items():
			layers[name] = {"max_logits": window["max_logits"].cpu().tolist(),
				"minimum_gamma": window["minimum_gamma"].cpu().tolist(),
				"clipped_head_updates": window["clipped_head_updates"].cpu().tolist(),
				"updates": window["updates"]}
		clipped = sum(sum(layer["clipped_head_updates"]) for layer in layers.values())
		observed = sum(len(layer["max_logits"]) * layer["updates"] for layer in layers.values())
		result = {"threshold": self.threshold, "max_logit": max(
			(value for layer in layers.values() for value in layer["max_logits"]), default=None),
			"clipped_head_updates": clipped, "observed_head_updates": observed,
			"clipped_fraction": clipped / observed if observed else None, "layers": layers}
		if reset:
			self._window.clear()
		return result
