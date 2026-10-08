"""Structural parameter assignment for the single Muon + auxiliary AdamW recipe."""

from torch import nn


def build_parameter_groups(model, weight_decay):
	"""Partition trainable parameters once, preserving named-parameter order.

	Muon handles hidden 2D matrices. Embeddings/output classifiers, norm gains,
	AttnRes pseudo-query vectors, other scalars/vectors and depthwise filters
	use AdamW. The latter are stored as 3D channel-wise convolution kernels;
	they are not flattened into a coupled hidden matrix.

	Fused FFN gate/up storage represents two independent matrices, specified
	by a batched Muon view. Attention matrices remain whole until the per-head
	delivery. Tied parameters are deduplicated by model.named_parameters().
	"""
	auxiliary = set()
	no_decay = set()
	fused_views = {}
	for module in model.modules():
		if isinstance(module, nn.Embedding):
			auxiliary.update(id(parameter) for parameter in module.parameters())
		if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
			no_decay.update(id(parameter) for parameter in module.parameters())
		# These Linear(C, 1) weights implement the paper's learned query vector.
		for name in ("attn_res_proj", "mlp_res_proj", "output_attn_res_proj"):
			projection = getattr(module, name, None)
			if isinstance(projection, nn.Linear):
				no_decay.update(id(parameter) for parameter in projection.parameters())
		gate_up = getattr(module, "gate_up_proj", None)
		if isinstance(gate_up, nn.Linear) and hasattr(module, "intermediate_size"):
			rows, columns = module.intermediate_size, gate_up.in_features
			if tuple(gate_up.weight.shape) != (2 * rows, columns):
				raise ValueError("Fused FFN gate/up weight does not match its declared dimensions")
			fused_views[id(gate_up.weight)] = (2, rows, columns)
	for name in ("lm_head", "head"):
		output = getattr(model, name, None)
		if isinstance(output, nn.Module):
			auxiliary.update(id(parameter) for parameter in output.parameters())

	groups = {}
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		identity = id(parameter)
		if parameter.ndim < 2 or identity in no_decay:
			key = ("adamw", None, False)
		elif parameter.ndim == 2 and identity not in auxiliary:
			key = ("muon", fused_views.get(identity), True)
		else:
			key = ("adamw", None, True)
		if key not in groups:
			# Keep the layout stable even when the configured decay is zero.
			algorithm, view, decayed = key
			decay = weight_decay if decayed else 0.0
			groups[key] = {"params": [], "param_names": [], "algorithm": algorithm,
						   "weight_decay": decay}
			if view is not None:
				groups[key]["muon_shape"] = view
		groups[key]["params"].append(parameter)
		groups[key]["param_names"].append(name)
	if not groups:
		raise ValueError("The model has no trainable parameters")
	return list(groups.values())
