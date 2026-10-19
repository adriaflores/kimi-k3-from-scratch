---
title: "FusedKDAChunkwise"
description: "Wraps the external FLA chunk_kda kernel with this model’s projections, gates, and output contract."
sourcePath: "src/kimi_k3/attention/kda/fused_chunk.py"
status: review
---

`FusedKDAChunkwise` has the same `(B,T,M) → (B,T,M)` attention-branch contract as the reference KDA, but delegates its recurrent/chunkwise computation to **Flash Linear Attention (FLA)**. It is an integration wrapper, not a Triton kernel implemented locally. Its import is made lazily by `build_kda()` so CPU-only model construction can avoid importing FLA.

## Parameters shared with the reference

The constructor [source L11–L47](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py#L11-L47) builds a `QKV` linear projection, three short depthwise Q/K/V convolutions, write-strength `Wb`, factorized gate input `Wad`/`Wau`, learned `A_log` and `dt_bias`, optional full-rank or low-rank output gate, per-head `MixedPrecisionRMSNorm`, and final `Wo`. The names and shapes match the reference model, allowing weight transfer and equivalence checks. `self.scale` is `D**-0.5`; `chunk_size` is the configured chunk length, not an inferred padding parameter.

## CUDA-only forward and causal preprocessing

The method rejects CPU inputs with a `RuntimeError` before allocating projections [source L49–L51](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py#L49-L51). It then projects Q/K/V, transposes into `(B,HD,T)`, left-pads by `kernel_size-1`, applies independent depthwise filters followed by `SiLU`, and reshapes to FLA's `(B,T,H,D)` representation [source L53–L73](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py#L53-L73). The reference chunkwise implementation uses a different intermediate layout, but the meaning of Q/K/V is the same.

Write-strength `beta` remains `(B,T,H)` and the factorized decay input `a` becomes `(B,T,H,D)` [source L75–L77](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py#L75-L77). This wrapper does not manually normalize Q/K or exponentiate the decay gate at this point; it asks the upstream kernel to perform those operations.

## Exact upstream call contract

The decisive boundary is [source L79–L95](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py#L79-L95):

```python
o, _ = chunk_kda(
q=q,
k=k,
v=v,
g=a,
beta=beta,
A_log=self.A_log,
dt_bias=self.dt_bias,
scale=self.scale,
use_qk_l2norm_in_kernel=True,
use_gate_in_kernel=True,
use_beta_sigmoid_in_kernel=True,
safe_gate=True,
lower_bound=self.gate_lower_bound,
chunk_size=self.chunk_size,
state_v_first=True,
) # (B, T, H, D)
```

The excerpt is literal; the original passes each keyword on its own line. The `True` flags locate L2 normalization, retention gating and sigmoid-beta computation inside FLA rather than in this Python source. `safe_gate=True` and the configured lower bound are part of the backend's supported numerical recipe. `state_v_first=True` chooses the upstream state orientation; do not infer the kernel's memory layout solely from this wrapper. The unused second returned value is not retained as a generation cache.

The configuration validator constrains fused chunk lengths to **32 or 64**, with `kda_gate_lower_bound` in `[-5,0)` [test test_config.py L22–L38](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_config.py#L22-L38). The fused equivalence test deliberately exercises some sequence lengths that are not exact multiples of chunk size [test test_kda_fused_vs_chunkwise.py L91–L101](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_fused_vs_chunkwise.py#L91-L101); this is a different input-length contract from the reference `chunkwise.py`.

## Output handling

After FLA returns `(B,T,H,D)`, `out_norm` normalizes per-head features. The selected full-rank or low-rank output gate is a sigmoid of a projection from the original hidden states; the head tensor is multiplied by that gate, flattened to `(B,T,HD)` and projected through `Wo` [source L97–L110](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py#L97-L110). No decoder residual is added here.

## Verification and limits

[test test_kda_fused_vs_chunkwise.py L44–L88](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_fused_vs_chunkwise.py#L44-L88) checks CUDA FP32 and BF16 forward/backward agreement against the readable chunkwise reference with explicit nonzero numerical tolerances. Further cases verify a CPU rejection, non-multiple lengths, and assembled-model forward/backward comparison [test test_kda_fused_vs_chunkwise.py L91–L146](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_fused_vs_chunkwise.py#L91-L146). Those tests require CUDA/FLA and were not run in this authoring environment. Their source supports a *toleranced comparison*, not bitwise identity or a universal speed claim. Attribution for the kernel belongs to upstream FLA; this repository supplies its wrapper and model-specific projections.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py">fused_chunk.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_fused_vs_chunkwise.py">test_kda_fused_vs_chunkwise.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_config.py">test_config.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
