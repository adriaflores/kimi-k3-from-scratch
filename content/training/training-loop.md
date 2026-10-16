---
title: "The Training Loop"
description: "The engineering steps needed to turn tokenized data into a verified parameter update."
question: "What must work before it is safe to scale a training run?"
order: 1
status: draft
---

A training loop is easy to summarize: load a batch, compute a loss, propagate gradients, and update parameters. Verifying that loop is harder. Each step has to agree with the next, and a successful update does not yet prove that the training system can recover after an interruption.

This chapter is a **reading-layout fixture**, not the final account of our experiments. The code below is schematic, and it should not be mistaken for a snippet from the canonical trainer.

## Follow one update

A useful way to reason about a training loop is to follow the data through one update. The important question is not only whether gradients exist, but whether the update uses the right examples, objective, and optimizer state.

```python
# Schematic only — not copied from the Kimi K3 trainer.
for x, y in batches:
    optimizer.zero_grad()
    loss = model(x, labels=y).loss
    loss.backward()
    optimizer.step()
```

The code is intentionally small. Real training adds mixed precision, gradient accumulation, clipping, distributed concerns, and other choices that deserve explanations of their own.

<aside class="reading-note" aria-label="Implementation note">
  <span class="reading-note-label">Implementation note</span>
  <p>Code excerpts in finished chapters should link to the canonical source and explain one decision. They should not become a second copy of the implementation.</p>
</aside>

## What does a checkpoint preserve?

The model weights alone are not the full state of training. To continue consistently, the loop may also need the optimizer state, its position in the data, and the relevant random-number-generator states.

| State | Why it matters |
| :-- | :-- |
| Model parameters | Determines the current predictions |
| Optimizer state | Influences the next parameter update |
| Data position | Determines which tokens are seen next |
| RNG state | Influences stochastic operations |

<blockquote>
  <p><strong>Think about the next step.</strong> If a resumed run loads the same weights but reads a different batch, should we call the continuation equivalent?</p>
</blockquote>

## Evidence, not assumptions

The finished Training chapter will state which recovery tests were run and what they established, with precise links to the recorded results. A passing test name is not evidence by itself; method, conditions and limitations belong beside the claim.

<aside class="reading-evidence" aria-label="Evidence placeholder">
  <span class="reading-note-label">Evidence pattern · illustrative only</span>
  <p><strong>Claim.</strong> A particular run resumed correctly.</p>
  <p><strong>Required support.</strong> Checkpoint identity, comparison method, test or run record, conditions and observed differences. No measurement is asserted here.</p>
</aside>

This page exercises the shared code, table, callout, quote and evidence styles. See the [Training index](/training/) for the planned sequence.

<aside class="reading-sources" aria-label="Source trail for this draft">
  <span class="reading-note-label">Source trail · draft</span>
  <dl>
    <dt>Implementation</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch">Project repository</a>. The excerpt on this page is schematic, not copied from the trainer.</dd>
    <dt>Tests &amp; runs</dt>
    <dd>No experimental measurement is claimed in this reading-layout fixture.</dd>
  </dl>
</aside>
