---
title: "Hybrid Attention"
description: "Why Kimi K3 interleaves a correctable recurrent memory with global latent attention—and how we implement both."
question: "Why combine recurrent memory with global attention?"
order: 1
status: draft
---

Imagine reading a long sequence. Some information can be carried forward as a compact summary; other details are useful precisely because we can return to a particular earlier token. These are different ways of using context, and neither makes the other unnecessary.

Kimi K3 combines them *across layers*: three layers of **Kimi Delta Attention (KDA)** followed by one **Gated Multi-head Latent Attention (MLA)** layer. It also ends its backbone with an extra MLA layer. Let's start with the difference between their memories, then see how the two implementations fit together. This is the language backbone only—not Kimi K3's vision pathway. [Paper §2.1](https://arxiv.org/abs/2607.24653v2).

## Two ways to use the past

A recurrent layer carries a state forward. Instead of retaining a separate key and value for every previous token, it updates a fixed-size matrix as tokens arrive. That keeps the *state size* independent of sequence length, but it also forces the model to decide how new information should change what is already stored.

A global attention layer takes the other route: the current query can compare itself with earlier token representations. This permits direct, content-dependent interactions across the prefix, at the cost of attention work that grows with the number of positions. MLA reduces the representation needed for keys and values; it does **not** turn global softmax attention into a fixed-state recurrence.

[Figure 1](#fig-context-access) illustrates this distinction. It is deliberately conceptual; it does not depict the exact kernels.

<figure id="fig-context-access" class="reading-figure" data-kind="conceptual">
  <a href="/assets/site/architecture/hybrid-attention/memory-and-attention.svg" target="_blank" rel="noopener noreferrer" aria-label="Open the recurrent memory and global attention diagram at full size">
    <img src="/assets/site/architecture/hybrid-attention/memory-and-attention.svg" width="900" height="380" loading="lazy" decoding="async" alt="A sequence updates one compact recurrent state in the first diagram. In the second, the current query can address separate earlier key and value representations." />
  </a>
  <figcaption><strong>Figure 1.</strong> Two complementary routes to context: revise a compact state, or address earlier positions directly. Original simplified conceptual illustration; not a measured comparison.</figcaption>
</figure>

Here's the interesting question: **if memory has a fixed size, how can we update it without just piling new information on top of old information?** That leads directly to KDA.

## KDA: correct what the memory already predicts

Consider one KDA head. Its state is a matrix $S_t\in\mathbb{R}^{d_k\times d_v}$. A key $k_t$ selects a direction in that state, a value $v_t$ provides the information to write, and a query $q_t$ later reads from it.

First, KDA applies a **channel-wise decay** to the previous state:

$$
M_t = \operatorname{Diag}(\alpha_t)S_{t-1},
\qquad \alpha_t\in(0,1)^{d_k}.
$$

The vector $\alpha_t$ controls how strongly each key channel is retained. KDA then computes an error between the new value and the value that the decayed state would already return for the current key:

$$
\boxed{S_t=M_t+\beta_t k_t(v_t-M_t^\top k_t)^\top},
\qquad \widetilde{o}_t=S_t^\top q_t.
$$

The scalar $\beta_t\in(0,1)$ controls the write strength for this head. Notice the subtraction. If the state already predicts $v_t$ well for $k_t$, the correction is small; when the prediction differs, the update changes the corresponding memory direction. This is the **delta rule**, after channel-wise forgetting. Expanding the brackets recovers the recurrence given in [paper §2.1.1, Eq. (1)](https://arxiv.org/abs/2607.24653v2).

<aside class="reading-question" aria-label="Question to consider">
  <span class="reading-note-label">Pause and predict</span>
  <p>What happens to the write term when the decayed memory already returns exactly the new value? And why might that be preferable to adding the value again?</p>
</aside>

In the [recurrent PyTorch reference](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/attention/kda/recurrent.py), that update becomes just a few tensor operations (the variable `S` has already been decayed before the prediction):

```python
S = alpha_t * S
v_old = torch.einsum("bhkv,bhk->bhv", S, k_t)
delta_v = v_t - v_old
update = k_t.unsqueeze(-1) * delta_v.unsqueeze(-2)
S = S + beta_t * update
```

The surrounding module performs short causal convolutions on Q, K and V, normalizes Q and K, and applies a learned output gate. The implementation also scales the recurrent readout by the inverse square root of the head dimension. Those details matter for the full module, but the *correction* above is its conceptual center.

## Why the decay is bounded

There is a numerical issue hiding inside the parallel implementation. Chunkwise KDA needs products of retention factors over several positions; their reciprocals can become very large. The paper addresses this by bounding the per-token **log-decay** using a scaled sigmoid:

$$
g_t^h=g_{\min}\,\sigma\!\left(e^{A_h}z_t^h\right),
\qquad \alpha_t^h=\exp(g_t^h),\qquad g_{\min}=-5.
$$

Thus each element of $g_t^h$ lies between $-5$ and $0$. The bound makes the relative-decay factors more manageable for the paper's tiled GPU formulation. It does not imply that the recurrent state itself is free from every numerical problem. See [paper §2.1.1, lower-bounded decay](https://arxiv.org/abs/2607.24653v2).

Our readable [chunkwise reference](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/attention/kda/chunkwise.py) accumulates log-decay, builds a causal triangular system and solves it within each chunk. This lets it compute token interactions in groups while carrying a recurrent state between groups. It currently requires sequence length to be divisible by its configured chunk size. We also provide an optional [fused path](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/attention/kda/fused_chunk.py), which calls **FLA's external Triton KDA kernel**; that optimized kernel is not our from-scratch implementation.

<aside class="reading-note" aria-label="Implementation distinction">
  <span class="reading-note-label">Implementation distinction</span>
  <p>The recurrence explains <em>what</em> KDA computes. The chunkwise transformation changes <em>how</em> we organize the same computation for parallel execution. A detailed UT/WY derivation belongs in a separate, deeper article.</p>
</aside>

## MLA: retrieve from individual positions

So why keep global attention at all? A fixed-size state is a useful compression, but a global layer can still compare its current query against distinct past positions. Kimi K3 uses **Multi-head Latent Attention** for this role. MLA first projects token representations into compact query and key/value latent spaces, then reconstructs the head-specific vectors used by causal attention.

Our [Gated MLA implementation](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/attention/gated_mla.py) makes this structure visible: `W_DQ` and `W_DKV` perform down-projections; `W_UQ` and `W_UKV` produce the head representations. Each key joins a **head-specific content part** with an additional **shared key part**. PyTorch's `scaled_dot_product_attention(..., is_causal=True)` then performs the causal global interaction.

Kimi K3's MLA uses **No Position Encoding (NoPE)**—no RoPE is applied inside these layers. The intervening KDA layers contribute position-sensitive mixing through their causal convolutions and recurrent dynamics. Finally, MLA uses a channel-wise output gate:

$$
y_t=W_o\!\left[\sigma(W_g x_t)\odot\widetilde{o}_t\right].
$$

The gate decides how strongly each output channel contributes before the final projection. This follows [paper §2.1.2](https://arxiv.org/abs/2607.24653v2). One important boundary: the paper describes the inference benefit of caching compressed KV latents, but **our reference `GatedMLA.forward` currently reconstructs Q/K/V and does not expose a persistent KV cache**. It implements the attention computation, not the paper's production caching system.

## The pattern in our scaled backbone

In the paper, the recurring pattern is **KDA → KDA → KDA → MLA**, followed by an extra global MLA at the end. Our [13-layer configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/configs/model/kimi_k3_300m.yaml) preserves that order on a smaller language model:

| Layers (1-based) | Attention mechanism |
| :-- | :-- |
| 1–3, 5–7, 9–11 | KDA (nine layers) |
| 4, 8, 12 | Periodic Gated MLA |
| 13 | Final Gated MLA |

The [decoder block](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/model.py) reads `layer_types` from the configuration; it does not infer the schedule from a hardcoded layer index. This is a reduced-scale architectural reproduction, not a claim to reproduce Kimi K3's parameter count, context capability or model quality.

## What we have verified—and what we have not

The repository contains targeted checks for the two mechanisms:

- [KDA equivalence tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/equivalence/test_kda_chunkwise_vs_recurrent.py) compare recurrent and chunkwise outputs, gradients, causality and chunk boundaries. Separate [GPU tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/equivalence/test_kda_fused_vs_chunkwise.py) specify tolerance-based comparisons with the FLA path.
- [MLA unit tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/unit/attention/test_gated_mla.py) compare outputs and gradients with an explicit per-head reference and check causal behavior.
- [Model integration tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/integration/test_kimi_k3.py) check that the chosen layer schedule assembles the expected attention modules.

These links establish **which checks are implemented**, not that we executed every test while preparing this chapter. They are not proof of equivalence to the original Kimi K3 checkpoint, nor of specific long-context speedups. Those claims require separate measurements and provenance.

The takeaway is a division of labor: **KDA repeatedly compresses and corrects a fixed-size memory; MLA periodically restores direct, global token interactions.** The next architecture chapters will explain how information also flows *between layers* through Attention Residuals and *across channels* through feed-forward and expert modules.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Paper</dt>
    <dd><a href="https://arxiv.org/abs/2607.24653v2">Kimi K3 technical report, v2</a>, §2.1, §2.1.1 and §2.1.2. Recurrence Eq. (1); decay mapping and output gates in the same section.</dd>
    <dt>Implementation</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/attention/kda/recurrent.py">KDA recurrence</a> · <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/attention/kda/chunkwise.py">chunkwise KDA</a> · <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/attention/gated_mla.py">Gated MLA</a> · <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/configs/model/kimi_k3_300m.yaml">scaled configuration</a>.</dd>
    <dt>Verification</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/equivalence/test_kda_chunkwise_vs_recurrent.py">KDA comparisons</a> · <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/unit/attention/test_gated_mla.py">MLA reference checks</a> · <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/integration/test_kimi_k3.py">assembly tests</a>. Test execution was not part of this documentation-only change.</dd>
  </dl>
</aside>
