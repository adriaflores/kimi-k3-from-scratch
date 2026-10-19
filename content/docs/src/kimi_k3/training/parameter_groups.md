---
title: "build_parameter_groups"
description: "Assigns model parameters to Muon or AdamW and declares independent matrix views for the optimizer."
sourcePath: "src/kimi_k3/training/parameter_groups.py"
status: published
---

`build_parameter_groups(model, weight_decay)` is the **bridge between model structure and the hybrid optimizer**. It does not update weights. Instead, it visits the modules, records which parameters need special treatment, then assigns each trainable parameter exactly once to a group consumed by `MuonAdamW` [source L20–L33](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L20-L33). Its structural rules also work with fused FLA-backed KDA, not just the readable implementation.

## Recording independent matrix geometry

`_register_view(views,projection,shape,split_sizes=None)` checks that every shape entry is a positive integer and that flattening the proposed view's leading axes produces the stored projection's two-dimensional weight shape [source L8–L17](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L8-L17). It keys metadata by Python parameter identity and refuses conflicting views for a shared weight. Crucially, it does **not** create a new trainable tensor or aliased `Parameter`: a view describes how the optimizer should treat rows independently.

The main traversal records four special layouts [source L47–L59](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L47-L59):

| Projection | Effective Muon matrices | Why the view matters |
| --- | --- | --- |
| Fused FFN `gate_up_proj` | `(2, I, C)` | Gate and up are distinct matrices despite one `(2I,C)` storage tensor. |
| KDA `QKV` | `(3,H,D,C)` | Q, K and V are separate for every KDA head. |
| MLA `W_UQ` | `(H,D+D_s,R_q)` | Each head's query projection is treated independently. |
| MLA `W_UKV` | `(H,D+D_v,R_{kv})`, split `(D,D_v)` | Within each head, key and value rows are separately orthogonalized. |

Here `C` is model width, `I` FFN intermediate width, `H` head count, `D` content dimension, `D_s` shared key dimension, `D_v` value dimension and `R` a latent rank. Ordinary shared latent, output and gate projections do not acquire artificial head axes. These shapes are *optimizer interpretation*, not new network output tensors.

### The layout declarations are literal, not conceptual reshapes

The source does not split gate/up, Q/K/V or attention-head weights into separate `nn.Parameter` objects. It only registers metadata on the original projection weight ([source L47–L59](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L47-L59)):

```python
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
```

For an illustrative `DenseFFN` with `M=8` and intermediate width `I=16`, the stored `gate_up_proj.weight` is `(32,8)`. Metadata `muon_shape=(2,16,8)` asks the optimizer to process two independent `(16,8)` matrices, not one `(32,8)` matrix. For KDA with `H=2,D=4,M=8`, the stored `QKV` weight `(24,8)` is viewed as `(3,2,4,8)`. For MLA, a stored `W_UKV` with `H=2,D=4,D_v=6,R_kv=8` has shape `(20,8)`, but the view `(2,10,8)` and `muon_split_sizes=(4,6)` further distinguish K from V within each head. These are **shape examples**, not measurements or claims about the shipped configuration.

For what the optimizer then does with these matrix views, see the [pinned implementation](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L321-L346) and the adjacent [`optimizers.md` Docs page](/docs/src/kimi_k3/training/optimizers/). This file only assigns structures; it never computes Newton–Schulz or takes an update.

## Exclusion and decay rules

During module traversal [source L34–L46](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L34-L46), embedding weights enter the auxiliary AdamW set. `nn.LayerNorm` and `nn.RMSNorm` weights are assigned **zero weight decay**. Learned AttnRes query projections (`attn_res_proj`, `mlp_res_proj`, `output_attn_res_proj`, all `Linear(C,1)`) likewise receive zero decay. Explicit output classifiers such as `lm_head` and `head` also become auxiliary [source L60–L63](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L60-L63). For tied `lm_head.weight` and `embed_tokens.weight`, parameter identity prevents duplicate optimization.

The final loop visits `model.named_parameters()` in its existing order, skipping frozen parameters [source L65–L91](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L65-L91). One-dimensional tensors, norms and explicitly exempted vectors use non-decayed AdamW; remaining 2D hidden matrices use decayed Muon, with any declared view/splits; other tensors use decayed AdamW. This last case covers depthwise convolutions, which must **not** be flattened into coupled Muon matrices. An empty trainable model raises `ValueError`.

Each returned group contains `params`, parallel `param_names`, `algorithm`, `weight_decay`, and optionally `muon_shape`/`muon_split_sizes`. Those names and shapes later participate in checkpoint compatibility validation; changing the grouping is a change of training recipe, not a cosmetic refactor.

### Classification happens once per original parameter identity

The final traversal and grouping decision are literal ([source L65–L86](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L65-L86)):

```python
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
```

Because the model ties `lm_head.weight` to `embed_tokens.weight`, the same object is visited once by `named_parameters()`: an embedding and vocabulary-classifier weight is not optimized twice. Norm gains and AttnRes learned query vectors use non-decayed AdamW; the depthwise convolution filter retains AdamW with decay because its storage is 3D, not a genuine hidden matrix for Muon. Other eligible 2D hidden matrices receive Muon with decay. This assignment only specifies the **algorithm and geometry**; inactive expert parameters remain untouched during steps because the optimizer later skips weights with no gradient ([source L34–L46](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L34-L46), [source L65–L91](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py#L65-L91)).

## Verification

[test test_parameter_groups.py L19–L56](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_parameter_groups.py#L19-L56) checks complete non-duplicated assignment, tied weights, correct algorithm/decay choices, and head-aware views. [test test_parameter_groups.py L83–L105](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_parameter_groups.py#L83-L105) compares a fused FFN update against independent gate/up matrices. [test test_per_head_muon.py L30–L90](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_per_head_muon.py#L30-L90) checks independence of attention head and K/V submatrices. [test test_parameter_groups.py L115–L174](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_parameter_groups.py#L115-L174) also checks inactive experts, real-model coverage and geometry. These assertions were inspected, not executed in this documentation pass.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py">parameter_groups.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_parameter_groups.py">test_parameter_groups.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_per_head_muon.py">test_per_head_muon.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
