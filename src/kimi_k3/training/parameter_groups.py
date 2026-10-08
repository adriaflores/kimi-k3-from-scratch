"""Structural parameter assignment, including head-wise attention views."""

import math

from torch import nn


def _register_view(views, projection, shape, split_sizes=None):
	"""Record storage geometry without introducing aliased Parameters."""
	if (any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
			or tuple(projection.weight.shape) != (math.prod(shape[:-1]), shape[-1])):
		raise ValueError("Projection weight does not match its declared Muon dimensions")
	identity = id(projection.weight)
	layout = (shape, split_sizes)
	if identity in views and views[identity] != layout:
		raise ValueError("Tied projection weights have conflicting Muon views")
	views[identity] = layout


def build_parameter_groups(model, weight_decay):
	"""Partition trainable parameters once, preserving named-parameter order.

	Muon handles hidden 2D matrices. Embeddings/output classifiers, norm gains,
	AttnRes pseudo-query vectors, other scalars/vectors and depthwise filters
	use AdamW. The latter are stored as 3D channel-wise convolution kernels;
	they are not flattened into a coupled hidden matrix.

	Fused FFN gate/up storage represents two independent matrices. KDA's QKV
	storage is projection-major; MLA up-projections are head-major, with K/V
	row splits inside each head. Only Q/K/V projections are split by head:
	shared latent projections, output projections and gates remain whole.
	Tied parameters are deduplicated by model.named_parameters().
	"""
	auxiliary = set()
	no_decay = set()
	views = {}
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
			_register_view(views, gate_up, (2, rows, columns))
		# Structural matching also covers FLA's separately implemented KDA class.
		qkv = getattr(module, "QKV", None)
		if isinstance(qkv, nn.Linear) and all(hasattr(module, name) for name in ("num_heads", "d_head", "d_model")):
			_register_view(views, qkv, (3, module.num_heads, module.d_head, module.d_model))
		uq, ukv = getattr(module, "W_UQ", None), getattr(module, "W_UKV", None)
		if isinstance(uq, nn.Linear) and isinstance(ukv, nn.Linear):
			_register_view(views, uq, (module.num_heads, module.d_head + module.d_shared, module.q_latent_dim))
			_register_view(views, ukv, (module.num_heads, module.d_head + module.dv_head, module.kv_latent_dim),
				(module.d_head, module.dv_head))
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
			key = ("adamw", None, None, False)
		elif parameter.ndim == 2 and identity not in auxiliary:
			view, split_sizes = views.get(identity, (None, None))
			key = ("muon", view, split_sizes, True)
		else:
			key = ("adamw", None, None, True)
		if key not in groups:
			# Keep the layout stable even when the configured decay is zero.
			algorithm, view, split_sizes, decayed = key
			decay = weight_decay if decayed else 0.0
			groups[key] = {"params": [], "param_names": [], "algorithm": algorithm,
						   "weight_decay": decay}
			if view is not None:
				groups[key]["muon_shape"] = view
			if split_sizes is not None:
				groups[key]["muon_split_sizes"] = split_sizes
		groups[key]["params"].append(parameter)
		groups[key]["param_names"].append(name)
	if not groups:
		raise ValueError("The model has no trainable parameters")
	return list(groups.values())
