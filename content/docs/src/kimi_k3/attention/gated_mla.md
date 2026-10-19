---
title: "GatedMLA"
description: "Implements causal multi-head latent attention with shared key features and optional output gating."
sourcePath: "src/kimi_k3/attention/gated_mla.py"
status: published
---

`GatedMLA` maps `(B,T,M)` hidden states to a new `(B,T,M)` attention contribution. It compresses Q and KV projections before expanding them into head features, appends a **shared** K component across heads, and calls PyTorch's causal scaled-dot-product attention. It is the decoder's MLA branch, not a complete residual block.

## Projection geometry and stored weights

The constructor [source L9–L43](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/gated_mla.py#L9-L43) reads `H`, query rank `R_q`, KV rank `R_kv`, content-head width `D`, shared width `D_s`, and value width `D_v` from configuration. It validates both latent ranks as strictly smaller than model width `M`. The two down projections have shapes `M → R_q` and `M → R_kv + D_s`; the latter creates the shared key component without a per-head duplicate in storage. The Q up-projection is `R_q → H(D+D_s)` and the KV up-projection is `R_kv → H(D+D_v)`. The final `Wo` maps `HD_v → M`.

`q_rmsnorm` and `kv_rmsnorm` normalize compressed latent features. Their gains are defined by `MixedPrecisionRMSNorm`; no positional rotation is applied by this source file, despite the `nope` spelling in `mla_qk_nope_head_dim`.

## Build per-head Q, K and V

The projection sequence [source L45–L63](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/gated_mla.py#L45-L63) first splits `W_DKV(x)` into the compressed KV feature and the shared K feature. Q is reshaped and transposed to `(B,H,T,D+D_s)`. KV is split into content K `(B,H,T,D)` and V `(B,H,T,D_v)`. The shared key is expanded from `(B,1,T,D_s)` to `(B,H,T,D_s)` and concatenated behind content K.

```python
K_shared = K_shared.unsqueeze(1) # (B, 1, T, D_s)
K_shared = K_shared.expand(B, self.num_heads, T, self.d_shared) # (B, H, T, D_s)
K = torch.cat([K_c, K_shared], dim=-1) # (B, H, T, D + D_s)
```

The explicit `expand` broadcasts the same shared features to every head. They are not separate independently learned shared-key vectors for each head. The packed head layouts are also significant to the optimizer's per-head Muon views, documented in `training/parameter_groups.py`.

## Observe logits, then compute causal attention

[source L64–L75](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/gated_mla.py#L64-L75) optionally invokes a `_qk_clip_observer` attached by the trainer, but **only** while the module is training and autograd is enabled. This hook records detached QK maxima; it does not replace the attention operation. The actual score/value path is:

```python
head_outputs = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=True)
```

The built-in SDPA handles the causal constraint and scaling using the **entire** Q/K head dimension `D+D_s`, not just `D`. Attention-dropout probability is explicitly zero. The commented-out manual attention block is inactive and should not be mistaken for a second runtime path.

## Output projection and optional gate

`head_outputs` becomes `(B,T,HD_v)` by transposing the time and head axes and materializing a contiguous view. With `mla_output_gate`, the implementation multiplies these values by `sigmoid(Wg(x))` before applying `Wo`; otherwise it applies `Wo` directly [source L77–L86](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/gated_mla.py#L77-L86). The gate uses the original `x`, not the attention result. There is no residual addition here.

## Verification

[test test_gated_mla.py L151–L190](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/attention/test_gated_mla.py#L151-L190) tests shared-key identity across heads and geometry. [test test_gated_mla.py L204–L259](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/attention/test_gated_mla.py#L204-L259) compares outputs and gradients with an independent reference and checks the full QK scaling. Other assertions cover causality, length-one behavior, the gate toggle, BF16 output and finite gradients. [test test_kimi_k3.py L96–L113](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_kimi_k3.py#L96-L113) checks causal logits in an assembled decoder. The file makes no KV-cache, RoPE, or custom-attention-kernel claim.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/gated_mla.py">gated_mla.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/attention/test_gated_mla.py">test_gated_mla.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_kimi_k3.py">test_kimi_k3.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
