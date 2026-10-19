---
title: "KDAChunkwise"
description: "A source-level walkthrough of the reference PyTorch Kimi Delta Attention implementation, from causal QKV projections to chunkwise UT solving and recurrent-state propagation."
sourcePath: "src/kimi_k3/attention/kda/chunkwise.py"
status: published
---

This file implements **`KDAChunkwise`**, the readable PyTorch reference for Kimi Delta Attention (KDA). Rather than updating memory token by token, it expresses the interactions within each chunk as batched matrix operations, then carries one matrix-valued state between chunks. The result is intended to reproduce the token-recurrent computation, without introducing a custom fused kernel.

The discussion below follows the [178-line implementation at the documented commit](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py). It assumes familiarity with the KDA update rule. For the intuition behind the memory and the delta correction, see [Hybrid Attention](/architecture/hybrid-attention/).

## Contract and tensor notation

The sole public class in this file is `KDAChunkwise(nn.Module)`, with a configuration-driven constructor and `forward(x)`. The input is a tensor of shape `(B, T, M)`; the returned tensor has the same shape. The output is a **new attention-branch contribution**, not an already residual-added decoder hidden state. The caller handles normalization and residual composition.

| Symbol | Meaning | Code |
| :-- | :-- | :-- |
| `B` | Batch size | `x.shape[0]` |
| `T` | Sequence length | `x.shape[1]` |
| `M` | Model hidden dimension | `self.d_model` |
| `H` | Number of KDA heads | `self.num_heads` |
| `D` | Per-head dimension | `self.d_head` |
| `C` | Tokens per chunk | `self.chunk_size` |
| `N` | Number of chunks | `T // C` |
| `HD` | Concatenated KDA dimension | `self.kda_dim` |

**Important precondition:** `T` must be divisible by `C`. This implementation enforces the restriction with an `assert` rather than internally padding a partial final chunk. Its fused counterpart has a different length-handling contract. See [L95–L102](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L95-L102).

The module allocates the recurrent state inside `forward`, initialized to zero for every call. It does **not** accept an incoming state or expose a state for continuation across separate calls; chunk-to-chunk continuity is internal to one invocation.

## Initialization: projections, gates, and learned timescales

The [constructor, L10–L47](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L10-L47), sets up three sets of transformations.

First, `QKV` projects `M` features to `3HD`, later split into Q, K, and V, with equal per-head widths. Three depthwise `Conv1d` modules then mix a short causal history separately for every projected channel: `groups=self.kda_dim` means there is no learned cross-channel mixing inside these convolutions. The left-padding happens in `forward`, not inside the convolution constructors.

Second, `Wb` predicts a **scalar write strength per head per position**, while `Wad` followed by `Wau` constructs the feature-wise time-scale input for decay. `A_log` has one learnable value per head and is exponentiated during `forward` to remain positive. `dt_bias` starts from the inverse-softplus expression of randomly selected positive time scales; the code subsequently adds that bias to the projected gate input. The factorized `Wad`/`Wau` pair is not a per-head scalar: its output is `(B, T, HD)`.

Third, the output branch can use either the full-rank `Wg: M → HD` or the factorized `Wgd: M → D` and `Wgu: D → HD`, selected by `kda_full_rank_output_gate`. `MixedPrecisionRMSNorm` normalizes each head's `D` output features before the final gate and the `Wo: HD → M` projection.

The values `A_log`, `dt_bias`, and the gate matrices are learned parameters, not cached recurrent memory. In particular, the name `A` will later be **reused** for a different mathematical object: a strictly lower-triangular interaction matrix inside each chunk. These are distinct quantities despite sharing a Python variable name.

## From the hidden states to causal Q, K, and V

[Source L49–L72](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L49-L72)

```python
q, k, v = self.QKV(x).split(self.kda_dim, dim=-1)

q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

padding = self.kernel_size - 1
q = F.pad(q, (padding, 0))
k = F.pad(k, (padding, 0))
v = F.pad(v, (padding, 0))
```

The excerpt shows the Q path; K and V follow the same operations. Before the convolution, a projection is `(B, T, HD)` and transposition turns it into the `Conv1d` layout `(B, HD, T)`. Left-padding by `kernel_size - 1` makes the output at position `t` depend only on positions up to `t`, and keeps the final length at `T` because the convolution has no built-in padding or stride change. `F.silu` acts elementwise after the convolution.

The `contiguous().view(...)` materializes a layout suitable for reshaping the channel dimension as `H × D` while preserving the intended order; the final transpose changes `(B, H, D, T)` into `(B, H, T, D)`. Calling `.contiguous()` here is a layout prerequisite for `view`, not a numerical operation. The code applies these changes independently to Q, K, and V.

The original `x` stays available for later gate projections. The dtype of V **after** convolution and activation is saved as `out_dtype`, to restore the output after the FP32 core.

## FP32 core and log-space decay

[Source L74–L104](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L74-L104)

```python
with torch.autocast(device_type=x.device.type, enabled=False):
    q, k, v = map(lambda x: x.to(torch.float), [q, k, v])
    q = F.normalize(q, p=2, dim=-1, eps=self.eps)
    k = F.normalize(k, p=2, dim=-1, eps=self.eps)
    beta = torch.sigmoid(self.Wb(x.to(self.Wb.weight.dtype)).float())
    beta = beta.transpose(1,2).unsqueeze(-1)

    a = self.Wau(self.Wad(x.to(self.Wad.weight.dtype))).float()
```

The disabled-autocast region covers the **whole chunkwise solve and state update**, not just these projections. `q`, `k`, and `v` are explicitly promoted to float32 before normalization; Q and K are unit-normalized along `D` with `eps=1e-6`. The gate projections use `x` cast to each projection's weight dtype and then convert their results to float32. This detail matters: disabling autocast alone would not cast an incoming lower-precision input or module weights to float32.

The retention gate is computed from `dt = a + dt_bias` and the positive `exp(A_log)` coefficient:

$$
\log \alpha_{t,d}=\ell\,\sigma\!\left(e^{A_{\log,h}}\,dt_{t,h,d}\right),
\qquad
\log \gamma_{t,d}=\sum_{s=0}^{t}\log\alpha_{s,d}
$$

Here $\ell$ is `gate_lower_bound`, and $t$ runs **within a chunk** in the second expression. The notation $A_{\log,h}$ denotes the *learned log-parameter*, not the triangular matrix introduced below. With the project's negative decay bound, `log_alpha` is nonpositive. Storing a cumulative sum of log-decays makes products of many individual retention factors expressible as additions; the code still exponentiates these sums when materializing weighted tensors.

The `view` in [L97–L102](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L97-L102) changes each of Q, K, V, and the decay tensors to `(B, H, N, C, D)`. Beta is initially `(B, H, N, C, 1)` and then squeezed to `(B, H, N, C)`. No chunk state has been updated yet.

## Constructing the triangular interaction system

The first inner loop in [L106–L118](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L106-L118) builds a distinct `C × C` matrix for every batch item, head, and chunk. Its diagonal and upper triangle remain zero.

For positions $i>j$ within one chunk, the assigned coefficient is

$$
A_{ij}=-\beta_i\sum_{d=1}^{D} k_{i,d}\,e^{g_{i,d}-g_{j,d}}\,k_{j,d},
\quad g_i=\log\gamma_i.
$$

That formula is exactly the `k_future * exp(log_delta_gamma) * k_j` product summed along `D`, then multiplied by the **future position's** `beta`. The negative sign encodes the effect of earlier delta corrections on later corrections. For $i\le j$, `A[i,j]` remains zero. Each position can depend on past positions in its chunk; future positions do not enter the corresponding lower-triangular row.

```python
I = torch.eye(self.chunk_size, device=A.device, dtype=A.dtype)
T_ut = torch.linalg.solve_triangular(I - A, I, upper=False)
L = torch.exp(log_gamma) * k
beta_scalar = beta.unsqueeze(-2)
T_b = T_ut * beta_scalar
R = T_b @ L
U = T_b @ v
```

[Source L120–L127](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L120-L127)

Mathematically, `T_ut` is the unique solution to

$$
(I-A)\,T_{\mathrm{UT}}=I,
\qquad T_{\mathrm{UT}}=(I-A)^{-1}.
$$

The implementation uses a **batched triangular solve**, not a generic matrix inverse. Because `A` is strictly lower triangular, `I-A` is lower triangular with ones on the diagonal, so the solve is well-defined under ordinary finite arithmetic. The shared `I` is broadcast across the leading `(B, H, N)` dimensions.

The remaining tensor shapes are worth following:

| Tensor | Shape | Role |
| :-- | :-- | :-- |
| `T_ut` | `(B,H,N,C,C)` | Intra-chunk triangular transform |
| `L` | `(B,H,N,C,D)` | Keys scaled by cumulative retention |
| `beta_scalar` | `(B,H,N,1,C)` | Column-wise broadcast of per-position write strengths |
| `T_b` | `(B,H,N,C,C)` | `T_ut` multiplied by write strengths on the **column** index |
| `R`, `U` | `(B,H,N,C,D)` | Transformed key and value terms |

In other words, `T_b` represents $T_{\mathrm{UT}}\operatorname{Diag}(\beta)$, **not** $\operatorname{Diag}(\beta)T_{\mathrm{UT}}$. That distinction changes the recurrence. `R = T_b @ L` and `U = T_b @ v` prepare the two terms needed to obtain the corrected values for a given incoming memory state.

## Carrying one state between chunks

[Source L129–L159](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L129-L159)

The state `S` starts at zero with shape `(B, H, D, D)`. One axis corresponds to key features and the other to value features. The implementation computes all chunks' `R` and `U` first, but must still propagate `S` **sequentially across chunks**: chunk $n+1$ depends on the outgoing state from chunk $n$.

```python
U_n = U[:, :, chunk, :, :]
R_n = R[:, :, chunk, :, :]
Z = U_n - R_n @ S
q_n = q[:, :, chunk, :, :]
log_gamma_n = log_gamma[:, :, chunk, :, :]
Q_n = torch.exp(log_gamma_n) * q_n
```

Here `Z` has shape `(B,H,C,D)`. It encodes the transformed within-chunk writes **after accounting for the state entering this chunk**. The product `R_n @ S` is valid because `R_n` has `D` key features on its last axis and `S` has key-to-value shape `(D,D)`. Since both `R_n` and `U_n` were prepared in the triangular system, this subtraction incorporates the effect of the incoming memory into all within-chunk write corrections.

The next loop builds `Aqk`, another `C × C` matrix, but **do not confuse it with the preceding `A`**. Its entries include the diagonal, and use query–key interactions rather than key–key interactions:

$$
(\mathrm{Aqk})_{tj}
=\sum_{d=1}^{D}q_{t,d}\,e^{g_{t,d}-g_{j,d}}k_{j,d}
\quad\text{for } t\ge j,
$$

with entries zero for $t<j$. This is how a token at position $t$ reads all permitted corrections from positions $j\le t$ in its chunk. The diagonal accounts for the current position's update, as required by the source's update-before-read convention.

```python
log_gamma_c = log_gamma[:, :, chunk, self.chunk_size-1, :]
S_decay = torch.exp(log_gamma_c.unsqueeze(-1)) * S
K_decay = torch.exp(log_gamma_c.unsqueeze(-2) - g_j) * k[:, :, chunk, :, :]
S_new = S_decay + K_decay.mT @ Z
O_n = self.scale * (Q_n @ S + Aqk @ Z)
outputs.append(O_n)
S = S_new
```

[Source L152–L159](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L152-L159). In this excerpt, `g_j` is the full chunk's cumulative-decay tensor defined at L154; its value is not the single-position variable used earlier.

`S_decay` applies the whole chunk's retention to each key-feature row of the incoming state. `K_decay` instead transports each within-chunk key to the **end of the chunk**. The batched product `K_decay.mT @ Z` accumulates their corrected value contributions into the outgoing `(D,D)` state.

The output combines two independent causal sources:

$$
O_n = D^{-1/2}\left(Q_n S_{\mathrm{in}} + \mathrm{Aqk}\,Z\right).
$$

`Q_n @ S` reads the memory entering the chunk; `Aqk @ Z` reads the writes that happen inside the chunk, including the current token. Crucially, this is calculated while `S` still refers to the **incoming** state. Only after appending the outputs does `S = S_new` make the next chunk's incoming state equal to this chunk's outgoing state.

This avoids a per-token loop over the whole sequence, but does **not** make the implementation fully parallel across the time dimension: the per-chunk state recurrence remains sequential, and the construction of both triangular matrices uses explicit loops over the positions in a chunk. The temporary `C × C` tensors also incur a memory cost that the token-recurrent form does not have. These are algorithmic/code-structure distinctions, not benchmark claims.

## Returning to the model dimension

[Source L161–L178](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py#L161-L178)

The outputs are concatenated along the time axis, transposed to `(B,T,H,D)`, and cast back to `out_dtype`. `out_norm` performs per-head RMS normalization, then the selected output-gate projection is passed through a sigmoid and reshaped from `(B,T,HD)` to `(B,T,H,D)`.

```python
o = o * g
o = o.contiguous().view(B, T, self.kda_dim)
x_new = self.Wo(o)
return x_new
```

The elementwise gate scales each of the `H × D` head features **after** normalization. `Wo` maps the concatenated head representation back to the model width `M`. Nothing here adds `x` as a residual: this module supplies the attention contribution to its caller.

## Verification and boundaries

The strongest directly relevant check is [`test_kda_chunkwise_vs_recurrent.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py). Its parameterized assertions compare [forward outputs](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L28-L39) and [backward gradients](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L89-L108) with [`KDARecurrent`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py), for chunk sizes 1, 2, 4, and 8 and for both output-gate variants. Other assertions check [a single chunk and transitions between chunks](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L40-L64), [output dtype/shape in FP32/BF16 and finite outputs](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L65-L88), [finite gradients](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L109-L123), [prefix causality](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L124-L141), and [the error for a non-divisible sequence length](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py#L143-L149). These statements describe what the tests **assert**, not a fresh test run.

The CUDA-only [`test_kda_fused_vs_chunkwise.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_fused_vs_chunkwise.py) uses this PyTorch class as its reference when comparing with the FLA/Triton-backed [`FusedKDAChunkwise`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/fused_chunk.py). The comparison specifies nonzero numerical tolerances and depends on CUDA and the FLA installation; neither numerical identity nor universal hardware support should be inferred.

This implementation does **not** cache state between forward calls, pad incomplete chunks, or implement a custom backward kernel. Its PyTorch operations are differentiable through autograd. It is selected by [`build_kda()` in `model.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L12-L18) when the fused implementation is not requested. For the broader mathematical motivation, the site explains [Hybrid Attention](/architecture/hybrid-attention/), and the [versioned Kimi K3 paper](https://arxiv.org/abs/2607.24653v2) is the primary architecture reference.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/chunkwise.py">chunkwise.py · pinned source revision</a></dd>
    <dt>Reference</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/kda/recurrent.py">recurrent.py</a> · tokenwise state update</dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_chunkwise_vs_recurrent.py">recurrent equivalence</a> and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_kda_fused_vs_chunkwise.py">fused/reference comparison</a>; source inspection only</dd>
  </dl>
</aside>
