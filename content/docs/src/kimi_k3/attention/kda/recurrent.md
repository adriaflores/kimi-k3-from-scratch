---
title: "KDARecurrent"
description: "Implements tokenwise Kimi Delta Attention with featurewise state decay and delta writes."
sourcePath: "src/kimi_k3/attention/kda/recurrent.py"
status: review
---

`KDARecurrent` is the readable, token-by-token Kimi Delta Attention implementation. It maps `(B,T,M)` inputs to an `(B,T,M)` attention contribution, starting a **new zero state on every forward call**. Its recurrence provides the reference against which chunkwise implementations can be checked; it is not the backend selected by the model factory in normal model construction.

## Initialization: three streams, per-head gates

The constructor [source L10–L46](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py#L10-L46) creates `QKV: M → 3HD`, three separate depthwise `Conv1d` filters over `HD` channels, a per-head write-strength projection `Wb: M → H`, and a factorized decay-input path `Wad: M → D` then `Wau: D → HD`. `A_log` is a learned per-head positive-scale parameter stored in log form, while `dt_bias` provides learned featurewise offsets. The output branch uses either a full-rank `Wg: M → HD` or low-rank `Wgd`/`Wgu`, controlled by `kda_full_rank_output_gate`; `Wo` projects back to `M`.

The recurrent state itself is **not** an `nn.Parameter`. It is allocated in `forward`, separately for each batch and head.

## Short causal convolutions and dtype boundary

At [source L50–L70](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py#L50-L70), Q, K and V start in `(B,T,HD)` layout, are transposed to `(B,HD,T)`, left-padded by `kernel_size - 1`, passed through separate depthwise convolutions and `SiLU`, and reshaped into `(B,H,T,D)`. Left padding makes each output depend only on its current and earlier positions; no right context is incorporated.

The code saves the post-convolution V dtype as `out_dtype` and promotes Q/K/V to FP32, with unit normalization on the final feature axis of Q and K [source L72–L77](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py#L72-L77). Unlike the chunkwise reference, this implementation does not wrap its entire recurrence in an explicit disabled-autocast region; FP32 tensor conversions are written at the relevant points. Later output normalization restores `out_dtype`.

## Decay and write strength

The write strength comes from `sigmoid(Wb(x).float())`, reshaped to `(B,H,T,1)`. `Wau(Wad(x)).float()` plus the learned `dt_bias` produces `(B,T,H,D)` featurewise gate inputs. The positive `exp(A_log)` is broadcast per head, then [source L79–L90](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py#L79-L90) computes

$$
\alpha_{b,t,h,d}=\exp\!\left(\ell\,\sigma(e^{A_{\log,h}}\,dt_{b,t,h,d})\right),
$$

where `ell` is the configured `gate_lower_bound`. This is featurewise retention, not one scalar decay for an entire head. The notation `D` is both key and value width in this reduced implementation.

## The state recurrence in source order

The loop [source L93–L112](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py#L93-L112) initializes `S` with shape `(B,H,D,D)` and processes each position in sequence:

```python
S = alpha_t * S # (B, H, D_k, D_v)

v_old = torch.einsum("bhkv,bhk->bhv", S, k_t)
delta_v = v_t - v_old
update = k_t.unsqueeze(-1) * delta_v.unsqueeze(-2) # (B, H, D_k, D_v)
beta_t = beta[:, :, t, :].unsqueeze(-1) # (B, H, 1, 1)
S = S + beta_t * update
```

The decay multiplies **rows indexed by key features** (`alpha_t` ends in a singleton value dimension). The key retrieves the currently stored value at that key, `delta_v` measures the correction, and the rank-one outer product updates the key/value memory. Reading happens after writing:

$$
S_t=\operatorname{diag}(\alpha_t)S_{t-1}
  +\beta_t k_t\bigl(v_t-k_t^\top\operatorname{diag}(\alpha_t)S_{t-1}\bigr)^\top,
\qquad y_t=D^{-1/2}q_t^\top S_t.
$$

The `einsum` for `y_t` includes the scaling `self.scale = D**-0.5` [source L110–L110](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py#L110-L110). This state is local to a single invocation; the loop does not return an incremental cache for generation.

## Head normalization, gate, and output

`outputs` are stacked to `(B,T,H,D)`, cast back to the stored V dtype and passed through per-head RMSNorm. The optional gate is projected from the original `x`, sigmoid-activated and reshaped to `(B,T,H,D)` before multiplication. After concatenating the head features, `Wo` returns `(B,T,M)` [source L113–L128](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py#L113-L128). There is no residual add.

## Verification and boundaries

Unit tests check output size, BF16 round-trip, finite gradient propagation and prefix causality [test test_kda_recurrent.py L7–L88](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/attention/test_kda_recurrent.py#L7-L88). The reference equivalence suite compares forward outputs and backward gradients against `KDAChunkwise` for parameterized chunk sizes [test test_kda_chunkwise_vs_recurrent.py L28–L108](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L28-L108). These are source-inspected test assertions rather than a new execution. The recurrent loop is simple but sequential in sequence length; no claim about its comparative speed is made.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py">recurrent.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/attention/test_kda_recurrent.py">test_kda_recurrent.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py">test_kda_chunkwise_vs_recurrent.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
