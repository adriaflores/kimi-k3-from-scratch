"""Per-interval MoE loads, reusing integer counts from the eager dispatch."""

import math


class MoEDiagnostics:
	"""Enable temporary counters; snapshot QB only at training log boundaries.

	StableLatentMoE exposes `_expert_load_counts`: None disables collection,
	a Python list counts routed assignments. Discover that interface rather
	than a class identity, so wrappers or subclasses remain supported. No tensors, RNG or
	persistent model state are retained.
	"""

	def __init__(self, model, *, enabled=True):
		self.layers = [
			(name or "<root>", module) for name, module in model.named_modules()
			if hasattr(module, "_expert_load_counts")
		] if enabled else []

	def __enter__(self):
		if any(module._expert_load_counts is not None for _, module in self.layers):
			raise RuntimeError("MoE load collection is already active")
		for _, module in self.layers:
			module._expert_load_counts = [0] * module.num_routed_experts
		return self

	def snapshot(self):
		"""Return/reset training loads; biases are the latest post-forward QB state."""
		layers = {}
		for name, module in self.layers:
			counts = module._expert_load_counts.copy()
			assignments = sum(counts)
			biases = module.quantile_balancer.get_biases().detach().float().cpu().tolist()
			layers[name] = {
				"tokens": assignments // module.top_k,
				"expert_counts": counts,
				"max_load_over_mean": max(counts) * len(counts) / assignments if assignments else None,
				"zero_load_experts": counts.count(0) if assignments else None,
				"qb_biases": [bias if math.isfinite(bias) else None for bias in biases],
			}
			module._expert_load_counts[:] = [0] * len(counts)
		return layers

	def __exit__(self, exc_type, exc_value, traceback):
		for _, module in self.layers:
			module._expert_load_counts = None
