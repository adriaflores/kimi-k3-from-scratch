---
title: "Reliability"
description: "How checkpoints, recovery checks, and run records turn an interrupted training job into a traceable continuation."
question: "How do you make a long training run safe to interrupt and trust?"
order: 2
status: draft
---

Imagine a training job disappearing halfway through the night. In the morning, you load the latest weights, restart, and the loss looks reasonable.

But did you *resume* the experiment, or did you quietly start a different one?

That's the question behind training reliability. As we saw in [Training Loop](/training/training-loop/), a parameter update is more than a forward and backward pass. It also depends on what the optimizer remembers, which tokens arrive next, and what happened before the interruption. **A reliable checkpoint must preserve the future of the run, not just the appearance of the model at one moment.**

## The next update has a history

Consider a deliberately small token stream split into batches `A`, `B`, `C`, and `D`. Training completes updates on `A` and `B`, then stops. We restore the weights saved after `B`.

What should the first resumed batch be? **`C`.** If the data loader starts over at `A`, the weights may be correct, but the experiment has changed: the model repeats data it already consumed.

Even reading `C` is not enough. AdamW and Muon both remember previous updates. Reset that memory, and the same gradient may change the weights differently. The learning-rate schedule also has a position. The model itself may hold non-parameter state, such as the Quantile Balancing biases used by its MoE router.

Randomness matters too: resetting the initial seed does not restore the *next* random draw. We need the generator state at the checkpoint.

So the object we want to preserve is not merely $\theta$, the model parameters. It is the **training state**: parameters and model buffers, optimizer memory, the completed-update count, data position, and relevant random-number state. Each piece answers the same practical question: *what must remain unchanged for the next update to be the same?*


<figure id="fig-resume-state" class="reading-figure" data-kind="conceptual">
  <a
    href="/assets/site/training/reliability/resume-state-contrast.webp"
    target="_blank"
    rel="noopener noreferrer"
    aria-label="Open checkpoint recovery diagram at full size"
  >
    <img
      src="/assets/site/training/reliability/resume-state-contrast.webp"
      width="1600"
      height="900"
      loading="lazy"
      decoding="async"
      alt="Comparison of restarting from saved model weights, which may repeat batch A, and restoring the complete training state, which continues with batch C."
    />
  </a>
  <figcaption>
    <strong>Figure 1.</strong> Saving weights preserves the model, but preserving the full training state allows the next update to continue from the correct point.
    <span class="figure-provenance">Original conceptual illustration; simplified and not experimental evidence.</span>
  </figcaption>
</figure>


## Save a state you can actually restart from

There is a convenient moment to take that snapshot: **just after a completed optimizer update, when gradients have been cleared**. The trainer deliberately uses this boundary rather than trying to capture an arbitrary point inside gradient accumulation.

That choice makes recovery much simpler. A checkpoint represents completed updates. If the process stops during the next update, the incomplete attempt is not reconstructed; training returns to the last saved boundary. In the real [checkpoint writer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py#L287-L312), the heart of the snapshot looks like this:

```python
"model": model.state_dict(),
"optimizer": optimizer.state_dict(),
"loaders": {"train": train_loader.state_dict(), "val": val_loader.state_dict()},
"rng_state": _capture_rng_state(device),
```

Those are excerpts, not the entire checkpoint. The record also carries the recipe, update counters, optimizer layout, runtime signature and FP16 scaler state when applicable. Model buffers travel with the parameters.

Notice that the loader does **not** serialize the token dataset. It preserves a cursor and a fingerprint of the dataset manifest. At resume, the reader reconstructs the same ordered token stream and checks that identity. The data files themselves are assumed to remain unchanged; the loader validates shard headers and shapes, not the full contents of every shard. That is an important boundary of this guarantee.

<aside class="reading-question" aria-label="Why checkpoint boundaries matter">
  <span class="reading-note-label">Try this thought experiment</span>
  <p>If training uses two microbatches per update, could we safely resume after saving only the first microbatch's gradients?</p>
  <p class="reading-answer"><strong>Not with this checkpoint contract.</strong> The second microbatch has not contributed yet, and the partial gradients are deliberately not saved. By checkpointing only after the optimizer update and gradient clearing, recovery always begins from a well-defined completed state.</p>
</aside>

The writer saves to a temporary file and then replaces the destination. This reduces the risk of exposing a partially written checkpoint if the save itself is interrupted. It does not make every failure recoverable: a crash can still lose work since the previous checkpoint, and file replacement is not a guarantee against every storage or power-loss failure.

## Refuse to resume the wrong experiment

A file loading successfully is a surprisingly weak test. Two models can have compatible tensor shapes while relying on different training recipes. Two optimizers can contain the same number of parameters while assigning them to different matrix groups.

The [checkpoint loader](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py#L314-L441) therefore checks compatibility **before** changing model weights, optimizer state, loader cursors or random generators. Among other things, it checks the model configuration, optimizer type and named parameter layout, data manifest identity, accumulation and precision settings, QK-Clip policy, and an execution signature covering relevant runtime settings.

Increasing the final step count can be legitimate. Changing the sequence length or optimizer matrix assignment is rejected rather than accepted as a continuation.

The RNG state is restored last: reconstructing a model can itself consume random numbers, and those draws must not shift the resumed sequence. For a fixed-batch overfitting run, it also saves the batch's original token offset, then rereads that same batch without advancing the restored cursor.

**Reliability sometimes means refusing to continue.** A clear incompatibility error is more useful than a plausible-looking loss from a run whose history has changed.

## One run, potentially several sessions

We also need to distinguish *what was trained* from *when a particular process was running*.

The project's [run manifest](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/run.py#L119-L192), `run.json`, records a run ID and the sessions belonging to that run. A resumed process can append a new session while preserving the original run identity. Each session records its effective recipe, available source revision, environment, data identity, starting state and checkpoint relationship. A checkpoint resumed outside an existing manifest can instead begin a *new, linked run*; it is not automatically the same run ID.

Meanwhile, [`metrics.jsonl`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/metrics.py#L9-L58) appends timestamped events with both run and session IDs. On reopening, its writer discards an incomplete final line, preserving earlier complete records. This is useful for inspecting interruptions, but a log is **not** a checkpoint: it describes events; it cannot reconstruct optimizer memory or RNG state.

The distinction also changes how to read a graph. An abrupt loss change deserves investigation: is this a new session? Did the learning rate restore? Were FP16 updates skipped? Did evaluation or checkpointing affect throughput? The monitoring code separates training time from those other phases so such questions can be investigated, not guessed from a progress bar.

## Test the continuation, not the existence of a file

How can we tell that all those saved pieces fit together? One strong test is to run the same small experiment along two paths:

- **Uninterrupted:** train from the beginning through the final update.
- **Interrupted:** stop at a checkpoint, reconstruct the process, restore, then finish.

The result should be the same *under equivalent deterministic execution conditions*. The [resume integration test](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_checkpoint_resume.py#L142-L193) compares continuation losses, final model and optimizer state, counters, loader positions and subsequent random values. Its controlled setup enables deterministic algorithms and fixes thread settings; it also exercises fixed-batch and ordinary data loading.

Other tests probe the places where the contract might break. One test consumes Python, NumPy and PyTorch randomness. Another rejects incompatible recovery **without changing the destination state**. A subprocess test exercises the CLI, session records and restored recipe. The [metrics tests](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/training/test_metrics.py#L10-L47) check that complete JSONL history survives a truncated final write.

<aside class="reading-evidence" aria-label="What resume-equivalence establishes">
  <span class="reading-note-label">Evidence &amp; limitations</span>
  <p><strong>Defined verification:</strong> the repository tests compare an uninterrupted path with a checkpoint-and-resume path, including exact state comparisons under controlled conditions.</p>
  <p><strong>Limit:</strong> these are inspected test definitions, not newly executed test results. They do not promise bit-for-bit continuation across arbitrary devices, kernels, software versions or changed datasets, nor do they establish that a long production run was scientifically successful.</p>
</aside>

That is the difference between *the program restarts* and *the experiment continues*. The first is easy to demonstrate. The second requires preserving state, validating assumptions and testing the next updates.

With that foundation, performance work becomes much more meaningful. We can change how efficiently training runs while still asking the essential question: **did we preserve the experiment we intended to run?**

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Checkpoint and recovery</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py#L193-L441">State capture, validation and restoration</a>, at the inspected commit.</dd>
    <dt>Run and data identity</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/run.py#L119-L192">Run manifest and session provenance</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/data/dataset.py#L133-L195">loader cursor and manifest identity</a>.</dd>
    <dt>Verification definitions</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_checkpoint_resume.py">Resume equivalence and rejection tests</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/training/test_metrics.py">interrupted-log handling tests</a>. These links document test code, not an execution report.</dd>
  </dl>
</aside>
