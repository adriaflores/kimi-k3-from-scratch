---
title: "Assembling the Backbone"
description: "Follow a token from its embedding through KDA, MLA, AttnRes and MoE, all the way to the next-token scores."
question: "How do all these pieces become one model that predicts the next token?"
order: 5
status: published
---

We've spent four chapters taking Kimi K3 apart. We've seen a memory that corrects itself, a way to revisit earlier layers, a small network that reshapes what a token knows, and a collection of experts that doesn't need to wake up all at once.

But a pile of clever components isn't a language model. They have to pass information to one another, in the right order, without losing track of what each token has learned.

So let's put the pieces back together. Imagine following a single token through the model, watching its representation change at each step. By the end, we'll have something much more concrete than an architecture diagram. We'll know what happens when we actually call `KimiK3.forward`.

## It starts with a number, not a word

Say our input contains the words “The cat sat.” The model doesn't receive those words as English text. A tokenizer has already turned them into **token IDs**, integers that identify pieces of text. Each integer looks up a learned vector in the embedding table.

That's the first transformation:

$$
\text{token IDs }(B,T)\quad\longrightarrow\quad\text{embeddings }(B,T,C).
$$

Here, $B$ is the number of sequences, $T$ is the number of token positions in each sequence, and $C$ is the width of each token representation. In our [scaled configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/model/kimi_k3_300m.yaml), $C=640$. That means every token begins its journey carrying **640 learned features**, not a little dictionary definition of its meaning.

The model will keep that width through all thirteen decoder layers. The contents of the vector change, but the next layer always knows what shape to expect. This simple agreement is what makes it possible to combine mechanisms that do very different things internally.

## Two questions at every layer

Imagine a room full of people solving a puzzle. Every so often, someone needs to **hear what the others have discovered**. Then they need a moment to **think about what that means for their own answer**.

That's a useful way to picture our decoder layers. Each one has an **attention sublayer** followed by a **feed-forward sublayer**. Attention moves information between positions in the sequence. The FFN transforms features within each position. Together, they let a token gather context and do something useful with it.

The attention operation is chosen by the layer's configuration. A **KDA layer** updates and reads a compact recurrent memory. A **Gated MLA layer** can look directly across earlier token positions. If you'd like to revisit why we need both, that's the story in [Hybrid Attention](/architecture/hybrid-attention/).

The feed-forward operation has its own choice. Our **first decoder layer uses a Dense FFN with SiTU-GLU**. The other twelve use **Stable LatentMoE**, whose shared path runs for every token while its router selects a small number of specialized experts. Keeping the first layer dense is a choice of our scaled model, not an additional rule we're attributing to the original Kimi K3 backbone. We explored the transformations in [Dense FFN & SiTU-GLU](/architecture/dense-ffn/) and sparse routing in [Stable LatentMoE & Quantile Balancing](/architecture/stable-latent-moe/).

Why not use the same attention and the same FFN everywhere? Because the two decisions solve different problems. Attention decides **how information moves between tokens**. The FFN decides **how a token's features are transformed**. We can choose one without forcing the other.

## Thirteen layers, one deliberate rhythm

The [configuration file](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/model/kimi_k3_300m.yaml) tells the story more clearly than a list of class names:

| Decoder layers | Attention | Feed-forward |
| :-- | :-- | :-- |
| 1 | KDA | Dense FFN |
| 2–3 | KDA | Stable LatentMoE |
| 4 | Gated MLA | Stable LatentMoE |
| 5–7 | KDA | Stable LatentMoE |
| 8 | Gated MLA | Stable LatentMoE |
| 9–11 | KDA | Stable LatentMoE |
| 12–13 | Gated MLA | Stable LatentMoE |

Read down the middle column. We get three KDA layers, then one MLA, and repeat. The final layer is **another MLA**, so the backbone ends with global token interaction. This is the hybrid pattern described in [Kimi K3, §2.1](https://arxiv.org/html/2607.24653v2#S2.SS1), reproduced here on a much smaller model.

Notice something else. We don't need thirteen different decoder classes. [`KimiK3DecoderBlock`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/model.py#L39-L78) reads `layer_types` to select KDA or MLA, and `num_dense_layers` to select the dense FFN or MoE. One shared block structure, with a small number of explicit architectural choices.

<!-- OPTIONAL FIGURE BRIEF: A clean 13-layer vertical backbone schematic. Three KDA then one MLA, repeated, final MLA; first FFN dense and all others MoE. Show separate depth-wise AttnRes snapshots at two-layer boundaries and an output AttnRes. Keep labels large enough at 750px, use the site's near-black background, and avoid detailed submodule anatomy. Figure to be integrated later only after the actual asset exists. -->

## The interesting part is what happens *between* the layers

If each layer merely replaced the previous representation with a new one, we would risk losing useful earlier perspectives. Ordinary residual connections help by accumulating updates. Kimi K3 goes further with **Attention Residuals**, or AttnRes.

Here's the mental image. Each token carries its current, unfinished work through a small group of layers. At certain boundaries, the model keeps a snapshot of that work. Later sublayers can decide which earlier snapshots deserve attention, instead of always accepting a single running sum. That's the idea behind [Block Attention Residuals](/architecture/attention-residuals/).

Our model uses **two-layer AttnRes blocks**. For every token, `block_residual` keeps the representations available from earlier block boundaries, while the current block keeps a partial sum of its updates. Each decoder layer has two different learned AttnRes reads, one for the attention sublayer and another for the FFN. So even inside the same layer, those two operations can request different information from the token's history.

There's a lovely little detail at the start. The very first attention sublayer has no completed earlier block to consult. It simply begins from the embedding. As the network advances, the set of available snapshots grows. The code in [`KimiK3DecoderBlock.forward`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/model.py#L80-L114) handles that special case and stores new candidates at block boundaries.

Importantly, **AttnRes is not another way of looking at earlier tokens**. KDA and MLA do that. AttnRes looks at earlier *representations of the same token*, across network depth. That distinction is the reason these mechanisms fit together rather than competing for the same job.

## A tiny loop holds it all together

Now open [`KimiK3.forward`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/model.py#L166-L192). The part that runs the entire backbone is surprisingly short:

```python
x = self.embed_tokens(idx)
block_residual = x.new_zeros(B * T, 0, self.hidden_size)

for layer in self.layers:
    x, block_residual = layer(x, block_residual=block_residual)
```

That's it. `x` carries the evolving representation of all tokens, and `block_residual` carries the earlier depth-wise candidates. The loop doesn't need to know whether the next attention layer is KDA or MLA, or whether its FFN is dense or routed. Each decoder block makes those choices from its configuration.

<aside class="reading-question" aria-label="Check your intuition about the backbone state">
  <span class="reading-note-label">A quick intuition check</span>
  <p>If every decoder layer returns a new <code>x</code>, why does the loop also pass <code>block_residual</code> to the next layer?</p>
  <p class="reading-answer"><strong>Because the latest representation is not the whole depth history.</strong> The extra state keeps selected earlier block representations available for future AttnRes reads. Without it, those later sublayers would have only the most recent running result to work with.</p>
</aside>

One final twist matters. When the thirteenth decoder layer finishes, we don't immediately project its output to vocabulary scores. The model performs **one last AttnRes read**, mixing the final partial representation with earlier block snapshots. Only then does it apply the final RMSNorm and the language-model head. Even the final prediction gets to choose which earlier depth-wise information to use. [See the final read in `model.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/model.py#L176-L183).

## From a representation to a prediction

After all that work, each token still has a vector of width 640. But the next-token prediction needs something different: **one score for each possible token in the vocabulary**.

Our language-model head maps those 640 features to **49,152 logits per position**. A *logit* is simply a raw score. Larger logits mean the corresponding token is more favored, but the scores are not probabilities yet. A softmax can turn them into probabilities when needed for a loss or for generation.

One practical choice is worth calling out. In our reduced model, the input embedding table and the output language-model head **share the same weight matrix**. This is called *weight tying*. The vectors that helped represent tokens on the way in also provide the vocabulary directions used to score them on the way out. That saves parameters. It's an explicit simplification in our implementation, not a feature we're claiming for the original full-scale Kimi K3 model.

The forward method returns **`(logits, loss)`**. The logits have shape $(B,T,49152)$. If the caller supplies target token IDs, the model also computes cross-entropy loss. Otherwise, `loss` is `None`. Notice that `forward` doesn't choose the next token or write any text. It provides the scores from which an inference routine can choose. And the *one-token shift* required for next-token training is performed by the [data loader](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/data/dataset.py#L108-L125), not secretly inside this forward method. That's the handoff into [Training Loop](/training/training-loop/).

## How much of this have we checked?

A model that produces the right tensor shapes might still be wired incorrectly. Maybe a supposed KDA layer is actually MLA. Maybe an expert's parameters never receive gradients. Or maybe later tokens accidentally affect predictions for earlier ones.

The repository's [model integration tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_kimi_k3.py) are written to catch these problems. They check the attention/FFN layer schedule, output shapes, tied weights, finite gradients, causal behavior and whether repeated training on a small batch decreases loss. Separate [configuration tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/test_config.py) guard invalid settings, and [RMSNorm tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/test_rmsnorm.py) check mixed-precision behavior.

Those are descriptions of **tests in the repository**, not claims that we executed them while writing this chapter. They also don't establish that this smaller model matches the original checkpoint or its performance. Our model has thirteen layers and a 2,048-token configured context limit. The paper describes a much larger, native-multimodal system with capabilities that this reproduction does not claim to have. We also select the external FLA fused KDA implementation in our training-oriented configuration, while keeping a readable PyTorch chunkwise reference.

We've reached the end of the architecture tour. The pieces no longer feel independent. **Attention lets tokens exchange information. AttnRes lets each token revisit its earlier depth-wise representations. Dense FFNs and MoE transform its features. The final head turns the result into a prediction.** The [model configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/model/kimi_k3_300m.yaml) chooses the rhythm, and `model.py` makes it all happen.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Paper</dt>
    <dd><a href="https://arxiv.org/html/2607.24653v2#S2">Kimi K3 technical report, v2, §2</a>, especially Hybrid Attention (§2.1), Attention Residuals (§2.2) and Stable LatentMoE (§2.3). This chapter covers the language backbone, not the native vision pathway.</dd>
    <dt>Implementation</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/model.py">KimiK3 model and decoder</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/model/kimi_k3_300m.yaml">scaled model configuration</a>, and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/utils/rmsn.py">mixed-precision RMSNorm</a>.</dd>
    <dt>Verification</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_kimi_k3.py">Model integration tests</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/test_config.py">configuration tests</a>, and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/test_rmsnorm.py">RMSNorm tests</a>. No test execution or measured quality advantage is claimed here.</dd>
  </dl>
</aside>
