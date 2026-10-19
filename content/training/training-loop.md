---
title: "Training Loop"
description: "How a stream of tokens becomes a trustworthy parameter update, and how we make sure the next update is still the right one."
question: "How do you know a model is really learning, rather than just running?"
order: 1
status: published
---

Imagine teaching someone to finish your sentences. You say, “The cat sat on the…” and ask them to guess the next word. They guess, you reveal the answer, and they adjust their intuition a little. Do this enough times and, hopefully, their guesses improve.

That's language-model training in miniature. The idea is wonderfully simple. The engineering isn't quite so forgiving. A model can produce a loss, gradients can flow, and the GPU can stay busy while the system is quietly teaching it the wrong task. Worse, a run might seem perfectly healthy until you have to stop it and discover you can't pick up where you left off.

So when I built the Kimi K3 training loop, I wanted more than a function that updates weights. I wanted to be able to answer three questions. **Did we teach the right thing? Did the update actually happen? And can we trust the next one?**

## First, make sure we're asking the right question

Let's start with the smallest possible example. Suppose our token stream reads:

```text
The | cat | sat | down | .
```

We want the model to see `The` and predict `cat`, see `The cat` and predict `sat`, and so on. There is a nice trick here. We don't have to construct a separate example for every prediction. We can take one sequence, shift it by a token, and train on all those predictions at once.

| What the model sees | What it should predict |
| :-- | :-- |
| `The` | `cat` |
| `The cat` | `sat` |
| `The cat sat` | `down` |
| `The cat sat down` | `.` |

In code, the two tensors are almost the same. The difference is where we start reading them:

```python
x = buffer[:-1].view(batch_size, seq_len)
y = buffer[1:].view(batch_size, seq_len)
```

Those lines come from the [real token dataset](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/data/dataset.py#L108-L125). The extra token at the end matters. Without it, we'd have no target for the last position in the batch. The loader also knows how to read across token-shard boundaries, so the model sees one continuous stream rather than a collection of disconnected files.

Now comes the learning signal. At each position, the model assigns probabilities to possible next tokens. We look at the probability it assigned to the *actual* next token and penalize it when that probability was small. Averaging those penalties gives us the **cross-entropy loss**:

$$
\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p_\theta\!\left(y_{b,t}\mid x_{b,\le t}\right)
$$

There's no need to memorize the formula. The logarithm simply turns a confident wrong guess into a larger penalty. $B$ is the number of sequences, $T$ is their length, and $\theta$ represents the model's parameters. The important part is that **every position teaches the model to predict what comes next, using only what came before**.

<aside class="reading-question" aria-label="Check your intuition about next-token targets">
  <span class="reading-note-label">One small trap</span>
  <p>What would happen if we accidentally used the same tokens for both <code>x</code> and <code>y</code>? The loss might go down beautifully, but what skill would the model actually be learning?</p>
  <p class="reading-answer"><strong>It would learn to copy the answer it has already seen.</strong> A token is part of its own input, so predicting that same token can become an easy shortcut. We might celebrate a tiny loss while the model has learned almost nothing about what comes <em>next</em>. Shifting the targets by one position is what removes that shortcut.</p>
</aside>

That's why a falling loss, on its own, isn't convincing. We first need to know that we're measuring the right game.

## An update is a little more than a forward pass

Once the targets are correct, the next step feels familiar. Run the model, calculate the loss, backpropagate, update the parameters. But there is a practical problem hiding in those four verbs: **the batch we want to learn from may not fit in GPU memory**.

Suppose we want each update to learn from 256 tokens, but we can only process 128 at a time. We could take a batch of 128 tokens, compute its gradients, then take another and *add* its gradients before touching the weights. If we average the two losses correctly, those two passes form one optimizer update. This is **gradient accumulation**.

Think of it as gathering feedback from two pages of exercises before deciding how to revise the lesson. The feedback accumulates. The actual change to the model happens once.

The [trainer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/trainer.py#L510-L560) makes that distinction explicit. Here are the two central operations, excerpted from the update loop:

```python
with _autocast(device, config.precision):
    _, loss = model(x, targets=y)

scaler.scale(loss / config.gradient_accumulation_steps).backward()
```

Dividing by the number of accumulation steps prevents the gradient from becoming artificially larger just because we split a batch into smaller pieces. We use **FP32, BF16 or FP16** through the same training interface, with automatic mixed precision where appropriate. FP16 additionally uses a gradient scaler to handle its narrower numerical range.

There's one subtlety worth keeping in mind. Our MoE router has its own **Quantile Balancing** state, which can change on each training forward pass. That means two accumulated microbatches are not guaranteed to behave *exactly* like one physically larger batch, even if the arithmetic for averaging the losses is right. The trainer documents that distinction rather than promising an equivalence it cannot always guarantee.

## Which weights should we change, and by how much?

A gradient tells us which direction would reduce the loss locally. The optimizer decides how to turn that direction into a parameter update. It also has to decide how big the update should be, which is why the learning-rate schedule matters.

At the beginning of training, charging ahead with the full learning rate can be unnecessarily rough. The loop supports a **warmup**, followed by either a constant rate or a cosine decay. It also measures gradient norms and can clip unusually large gradients before an update. These aren't clever tricks that make a model understand language. They're ways of giving a difficult optimization problem a more controlled environment.

For this project I use a **hybrid Muon + AdamW optimizer**. The intuition is that not every parameter has the same job. A hidden projection matrix transforms whole representations, while an embedding, a normalization scale or a small gating vector has a different geometry. Muon handles selected hidden two-dimensional matrices. AdamW handles the auxiliary parameters, including embeddings and normalization parameters. Some attention projections are even treated as separate head-wise matrix views so the optimizer respects their structure. The exact partition lives in the [parameter-group builder](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/parameter_groups.py).

You don't need the details of Muon's matrix update to understand the loop. What's important is that an update isn't just “subtract the gradient.” It follows an explicit recipe, and that recipe has to remain consistent when training continues.

There's also a guard specific to the MLA layers. The trainer watches for excessively large query-key attention scores and applies **QK-Clip** to the relevant projection weights after a successful update if the threshold is crossed. This is a stability measure, not a replacement for the optimizer, and it doesn't mean every attention score is forcibly clipped during the forward pass. You can inspect the actual behavior in [`qk_clip.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/qk_clip.py).

## A run needs a pulse, not just a progress bar

Imagine watching a training run that reports only “step 10,000.” That sounds reassuring, but what have we really learned? We don't know whether the loss is behaving, whether the gradients have exploded, or whether most of the time is being spent moving data rather than teaching the model.

That's why each run records more than a step counter. The trainer logs training and validation loss, learning rate, gradient statistics, processed tokens, throughput, memory usage and optimizer diagnostics. The [metrics writer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/metrics.py) appends structured events to `metrics.jsonl`, while [`run.json`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/run.py) records the model, data, environment and training recipe behind the run.

Training loss tells us how the model is doing on material it is learning from. **Validation loss asks a different question**: how well does it predict tokens from a separate split? The evaluator reads a fixed prefix of that split, uses an independent data loader and restores the model's training mode and random-number state afterward. That makes repeated checks more comparable, though it doesn't magically prove broad generalization.

<aside class="reading-note" aria-label="What a training step actually counts">
  <span class="reading-note-label">A useful distinction</span>
  <p>The loop counts a <strong>step</strong> only when an optimizer update succeeds. If FP16 overflows, it skips that update, but the tokens used in the attempt have still been processed. So <code>step</code> and <code>tokens_seen</code> tell related, not identical, stories.</p>
</aside>

## What if the machine stops right now?

This is the question that turns a toy training script into an experiment you can depend on.

Imagine saving a video game just before a difficult level. Saving only your character's appearance wouldn't help much if the game forgot your inventory, location and everything you had already done. Training has the same problem. **Model weights are only part of the story.**

The optimizer has momentum and other accumulated state. The data loader has a position in the token stream. Random generators have their own state. FP16 may have a loss scaler. The learning-rate schedule depends on how many *successful* updates have happened. If we restore only the model weights, the next update may be different even though the checkpoint loads without complaint.

Our [checkpoint code](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/trainer.py#L287-L441) saves this information **at update boundaries**, after gradients have been cleared. It writes to a temporary file and then replaces the checkpoint, rather than leaving a half-written final file if something interrupts the save. On resume, it checks that the model, optimizer layout, training recipe and data identity are compatible *before* restoring the saved state.

And here's the key idea: the test of a good resume isn't “it starts running again.” It's **whether the resumed run takes the same next steps as an uninterrupted one under equivalent deterministic conditions**.

The repository includes an [integration test](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/integration/test_checkpoint_resume.py#L144-L193) that constructs both paths and compares subsequent losses, model weights, optimizer state, loader positions and random values. That's a much stronger question than checking whether a `.pt` file exists. Exact agreement depends on preserving the relevant execution conditions, so this is not a promise that any checkpoint can resume identically on arbitrary hardware or software.

## How do we know the whole thing learns?

Before trusting a long run, I start with a much smaller challenge: **can the complete model memorize a tiny, fixed batch?**

This might sound like the opposite of what we want from an intelligent model. It is. That's precisely why it's useful here. If the model can't learn a small set of next-token relationships that we deliberately repeat, there is little reason to trust a larger experiment. Maybe the targets are shifted incorrectly. Maybe a gradient never reaches an attention layer. Maybe an optimizer group is silently missing its parameters.

The [tiny-model integration test](https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/integration/test_tiny_training.py#L83-L178) is written to catch these kinds of mistakes. It checks next-token alignment, nonzero gradients through the architecture, parameter changes, fixed-batch learning and validation without unintended state mutation. Other tests check accumulation behavior and the resume contract.

<aside class="reading-evidence" aria-label="What the implementation tests establish">
  <span class="reading-note-label">What this verifies, and what it doesn't</span>
  <p><strong>The tests are designed to establish</strong> that the training machinery can learn from correctly aligned tokens and that a saved run can continue consistently under controlled conditions.</p>
  <p><strong>They do not establish</strong> that the final pretrained model generalizes well, that a particular learning rate is optimal, or that a production run achieved a given loss. Those conclusions require separately recorded experiments. This chapter references test definitions, not fresh test executions.</p>
</aside>

That's the real purpose of the loop. It isn't to make a GPU do work. It's to turn each batch into an update whose meaning we understand, whose behavior we can inspect and whose history we can recover. Once that foundation is trustworthy, it becomes worth asking the more exciting questions about throughput, optimization and scaling.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Core implementation</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/trainer.py">Training loop and checkpoint contract</a>, pinned to the inspected revision.</dd>
    <dt>Data and optimizer</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/data/dataset.py">Shifted token batches</a> and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/src/kimi_k3/training/parameter_groups.py">Muon/AdamW parameter assignment</a>.</dd>
    <dt>Verification</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/integration/test_tiny_training.py">Tiny-model training test</a> and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/ccf6aad5c6674aae4f994f394d21ef94c7aac7f8/tests/integration/test_checkpoint_resume.py">resume consistency test</a>. Links identify the test code, not an execution report.</dd>
  </dl>
</aside>
