---
title: "Attention Residuals"
description: "How Kimi K3 lets a layer revisit earlier representations instead of accepting one running sum."
question: "What if a layer could choose which earlier ideas to bring forward?"
order: 2
status: published
---

Imagine you're writing an explanation with a friend. You start with a rough idea, improve the wording, add an example, then refine the conclusion. By the time you reach the last paragraph, your draft contains all those changes. But suppose you now want to recover the *original idea*, before the later edits. You can try to reconstruct it from the latest version, or you can go back and read what you wrote earlier.

Deep neural networks face a surprisingly similar choice. Every layer transforms the representation it receives. A traditional residual connection keeps the earlier information around by *adding* the new transformation to the running state. That's extremely useful, but it also means later layers generally see the accumulated result, not the individual contributions that made it.

**Attention Residuals (AttnRes)** gives Kimi K3 another option. A layer can look back at representations from earlier depths and decide how much to use from each one. The key word is *depth*. In [Hybrid Attention](/architecture/hybrid-attention/), attention connects information at different **token positions**. Here, we're following **the same token through different layers**. [Kimi K3 paper, §2.2](https://arxiv.org/html/2607.24653v2#S2.SS2).

## What gets lost in a running sum?

Imagine that one early layer detects a useful clue about a word. A later layer adds some syntactic context. Another adds something else. With ordinary residual connections, the model repeatedly performs an update like this:

$$
h_{l+1}=h_l+f_l(h_l)
$$

Read it as: *keep what we have, then add what this layer discovered*. It's a great default. Information and gradients have a direct route through the network, which makes deep models much easier to train.

But that convenient running sum comes with a restriction. Suppose a later layer would benefit from the very first clue more than from the subsequent refinements. It only receives the mixture. The clue isn't necessarily gone, but it no longer has its own address. Recovering it requires the network to disentangle earlier contributions from the accumulated state.

So here's the natural question: **what if we kept some of those earlier representations available instead of always collapsing them into one?**

That's the leap AttnRes makes. It borrows the idea of attention, but uses it along the stack of layers rather than along the sequence of tokens.

## Let the next layer choose its sources

Picture three notes that earlier layers left for the same token. The next layer is allowed to read all three, but it doesn't have to value them equally. It might decide that the first note is especially relevant, the second helps a little and the third hardly helps at all.

For example, it could assign weights of 0.7, 0.2 and 0.1. Those numbers are just an illustration, not weights observed in our trained model. The important part is that they can be *different for different tokens*. A representation that matters for one token may matter much less for another.

How does the model make that choice? Each AttnRes read owns a **learned query vector**, sometimes called a *pseudo-query*. Think of it as the kind of information that the upcoming sublayer has learned to look for. Each earlier representation provides a candidate that this query can score. A softmax turns those scores into weights, and the weighted representations are added together.

We can write the whole idea in one equation:

$$
\alpha_i = \operatorname{softmax}_i\!\left(w^{\mathsf T}\operatorname{RMSNorm}(v_i)\right),
\qquad h = \sum_i \alpha_i v_i
$$

The $v_i$ are earlier representations of **one token**. The learned vector $w$ belongs to this particular read. The $\alpha_i$ are the preferences it computes, and $h$ is what the next sublayer actually receives. Importantly, $w$ itself isn't generated from the current token. It's learned during training. The **scores still depend on the token**, because its earlier representations are different.

One small detail makes this more reliable: **RMSNorm is applied when scoring each candidate**. Otherwise, a representation with a much larger magnitude could win attention largely by being big, rather than by matching what the sublayer is looking for. Notice that the values being combined aren't normalized in this equation. We normalize to compare, then mix the original representations. This is the mechanism in [paper §2.2, Eqs. (8–9)](https://arxiv.org/html/2607.24653v2#S2.SS2).

<aside class="reading-question" aria-label="Think about the attention direction">
  <span class="reading-note-label">A quick intuition check</span>
  <p>If two tokens reach the same layer, must they receive the same AttnRes weights just because the learned query is shared?</p>
  <p class="reading-answer"><strong>No.</strong> The query is shared, but each token carries its own earlier representations. Different candidates produce different scores, so the same sublayer can retrieve different parts of the depth history for different tokens.</p>
</aside>

<figure id="fig-depth-retrieval" class="reading-figure" data-kind="conceptual">
  <a
    href="/assets/site/architecture/attention-residuals/depth-retrieval.webp"
    target="_blank"
    rel="noopener noreferrer"
    aria-label="Open Attention Residuals illustration at full size"
  >
    <img
      src="/assets/site/architecture/attention-residuals/depth-retrieval.webp"
      alt="Left: ordinary residuals merge earlier layer contributions into one running representation. Right: Attention Residuals selectively retrieve representations from different network depths for the same token."
      width="2000"
      height="1125"
      loading="lazy"
      decoding="async"
    />
  </a>
  <figcaption>
    <strong>Figure 1.</strong> Ordinary residuals accumulate contributions
    into a running representation. Attention Residuals allow a layer to
    retrieve selected representations from earlier depths.
    Original conceptual illustration with illustrative attention weights.
  </figcaption>
</figure>

## Why not remember *every* layer forever?

The version we've described is **Full AttnRes**. It makes the token embedding and each earlier layer output available as separate candidates. That offers the most freedom, but it also means keeping all those representations alive. In a deep model, the memory and communication overhead can become more troublesome than the dot products themselves.

Here's a compromise that's easy to imagine. Instead of keeping every revision of our document, we keep a snapshot after each small group of edits. We can still choose among earlier snapshots, and we also keep the unfinished changes from the group we're currently working on.

That's **Block AttnRes**. Within a block, Kimi K3 sums the outputs of its layers into a partial running contribution. Once the block finishes, that sum becomes one stored candidate. When a new sublayer needs information, it attends over the original token embedding, the completed block summaries, and, if we're partway through a block, the current partial sum.

The difference is what we keep separately, not how we choose among the candidates. The same normalized scores and softmax can still decide which *block representations* matter. In the paper's full-scale model, the authors describe blocks of 12 layers and a much smaller set of depth-wise candidates than the full history. See [paper §2.2, Eq. (10)](https://arxiv.org/html/2607.24653v2#S2.SS2).

A block summary isn't a magical compression that preserves every layer independently. Inside each block we've returned to plain addition. We're deliberately trading some freedom to revisit individual layers for a much smaller collection of useful checkpoints. That makes the design practical without throwing away the central idea.


<figure id="fig-block-attnres" class="reading-figure" data-kind="conceptual">
  <a
    href="/assets/site/architecture/attention-residuals/block-attnres.webp"
    target="_blank"
    rel="noopener noreferrer"
    aria-label="Open Block Attention Residuals illustration at full size"
  >
    <img
      src="/assets/site/architecture/attention-residuals/block-attnres.webp"
      alt="Successive layers are grouped into blocks of two. Completed blocks contribute stored summaries, while the token embedding and current partial sum remain available for attention across network depth."
      width="2000"
      height="1125"
      loading="lazy"
      decoding="async"
    />
  </a>
  <figcaption>
    <strong>Figure 2.</strong> Block AttnRes stores one summary per
    completed block while keeping the current partial contribution
    available. This example uses two-layer blocks, matching our
    scaled implementation. Original conceptual illustration.
  </figcaption>
</figure>


## What happens in our 13-layer model?

Our [scaled configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/configs/model/kimi_k3_300m.yaml#L9-L21) has **13 decoder layers** and `attn_res_block_size: 2`. So we summarize two layers at a time. There are six complete two-layer groups and a final one-layer group. The token embedding is retained as its own candidate.

This grouping is independent of Hybrid Attention's **three KDA layers followed by one MLA layer**. One choice determines *how tokens exchange information*. The other determines *which earlier depths a sublayer can consult*. They operate in different directions.

The core implementation is pleasantly small. Our [`apply_attn_res`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/src/kimi_k3/model.py#L23-L37) function gathers the completed blocks and current partial sum, scores them and computes their weighted combination:

```python
V = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
scores = proj(norm(V)).squeeze(-1)
weights = scores.softmax(dim=-1)
h = (weights.unsqueeze(-1) * V).sum(dim=1)
```

This is the same story as the equation. `V` contains candidate representations for each token. `norm` lets us compare them on a consistent scale. `proj` is the learned one-dimensional scoring projection for this particular read. Finally, the softmax weights blend the *unnormalized* candidates into a single input representation.

There's one more detail that's worth noticing in the [decoder block](https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/src/kimi_k3/model.py#L80-L114). A decoder layer contains both an attention sublayer and a feed-forward sublayer. **Each gets its own AttnRes read.** In other words, the attention operation and the FFN are allowed to ask for different information from depth. At a block boundary, the completed partial sum is saved before the new block starts accumulating updates. The very first attention sublayer is a special case, since there are no previous completed blocks to choose from yet.

After the thirteenth decoder layer, the model performs **one final AttnRes read** before the output normalization and language-model head. This gives the final prediction access to the embedding and the available block summaries, rather than simply accepting the last partial sum as the only representation. See [`KimiK3.forward`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/src/kimi_k3/model.py#L166-L181).

## What have we actually checked?

We can point to the code and tests for the mechanics, but we should be careful about what that proves. The [model integration tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/tests/integration/test_kimi_k3.py#L57-L113) check the assembled model, finite outputs, gradient flow through used parameters and causal behavior. They even account for the first attention sublayer's unused AttnRes projection. That's valuable engineering evidence, although these are tests *defined in the repository*, not new executions reported by this chapter.

Those tests don't independently establish that Block AttnRes reproduces every numerical detail of the paper's full-scale implementation, nor that it improves downstream quality by a measured amount. We'd need a separate controlled comparison to make those claims.

The reason AttnRes is worth understanding doesn't depend on claiming a particular improvement here. **Attention across tokens answers *where in the sequence should I look?* Attention Residuals answers *which stage of understanding should I bring back?*** Once you can picture that distinction, the mechanism becomes much less mysterious.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Paper</dt>
    <dd><a href="https://arxiv.org/html/2607.24653v2#S2.SS2">Kimi K3 technical report, v2, §2.2</a>, including Full and Block Attention Residuals, Eqs. (8–10).</dd>
    <dt>Implementation</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/src/kimi_k3/model.py#L23-L37">AttnRes mixing function</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/src/kimi_k3/model.py#L80-L114">decoder integration</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/configs/model/kimi_k3_300m.yaml#L9-L21">scaled configuration</a>.</dd>
    <dt>Verification</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/b8b6e12408a50d75a69f92f2e6a9ebd392421560/tests/integration/test_kimi_k3.py#L57-L113">Model assembly and gradient/causality tests</a>. No test execution or performance measurement is claimed for this documentation draft.</dd>
  </dl>
</aside>
