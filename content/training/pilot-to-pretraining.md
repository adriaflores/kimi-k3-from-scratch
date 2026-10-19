---
title: "Pilot to Pretraining"
description: "How a bounded pilot, an unchanged training recipe, and auditable run records determine whether full pretraining is ready to begin."
question: "When is a pilot convincing enough to launch full pretraining?"
order: 6
status: published
---

Imagine you're about to start a training run that will occupy a GPU for a long time. The model loads. The first forward pass works. The loss is finite. You could press *start* now, and there's a good chance the machine would stay busy.

But suppose it fails halfway through. Or worse, suppose it finishes and you can't tell how many tokens it actually processed, whether it silently skipped updates, or which optimizer settings were in effect after a restart. The GPU time was real. The experiment is much harder to trust.

That is the last question I wanted to answer before moving from small experiments to pretraining: **what would a pilot need to show before I'd be willing to leave the full run alone?**

A pilot doesn't have to prove that the eventual model will be good. It has a more modest, and more useful, job. It should show that the planned training system can **learn without obvious instability, keep an honest account of its work, and continue after an interruption**. Then we can freeze the recipe and begin a different kind of experiment: one whose purpose is to train the model, rather than to debug the machinery.

## A pilot is a rehearsal, not a miniature final result

Think about a theater rehearsal. The actors don't need to deliver opening night in full. They do need to know their cues, make the scene work, and recover if something goes wrong. A successful five-minute rehearsal is not proof of a successful two-hour performance—but it tells you whether the performance is ready to begin.

The same distinction matters here. A smoke test asks whether the model and training script can execute. A fixed-batch overfit test asks whether they can learn a deliberately easy problem. A bounded pilot using successive data batches asks something stronger: **does the intended recipe behave sensibly for long enough to expose integration, stability, and continuation problems?**

Those are different questions, so I shouldn't mistake one answer for another. The repository reflects that separation in its [overfit](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/overfit.yaml), [acceptance](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/acceptance.yaml), and [stability](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/stability.yaml) configurations. The acceptance recipe deliberately repeats a batch; the stability recipe requests successive batches. **These YAML files describe experiments. They are not evidence that those experiments passed.**

I find it helpful to imagine three signals arriving at the same decision point, as in [Figure 1](#fig-pilot-gate). A smooth loss curve is only one of them. If the run cannot be accounted for or resumed, a promising loss isn't enough.

<figure id="fig-pilot-gate" class="reading-figure" data-kind="conceptual">
  <a href="/assets/site/training/pilot-to-pretraining/pilot-gate.svg" target="_blank" rel="noopener noreferrer" aria-label="Open the pilot readiness diagram at full size">
    <img src="/assets/site/training/pilot-to-pretraining/pilot-gate.svg" width="1440" height="620" loading="lazy" decoding="async" alt="Three pilot signals—stability, accounting, and resume—converge at one decision gate, from which an arrow begins the full pretraining run." />
  </a>
  <figcaption><strong>Figure 1.</strong> A go/no-go decision depends on more than a finite loss: numerical behavior, trustworthy accounting, and recovery all have to hold together. <span class="figure-provenance">Original conceptual illustration; not a measured run or an automated decision gate.</span></figcaption>
</figure>

## Before training, what exactly is the dataset?

There are two very different quantities that are easy to call “the number of tokens.” One describes **what we've prepared**. The other describes **what the model has processed**. They only tell the same story under additional assumptions.

First comes preparation. The [data script](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/prepare_data.py) resolves the FineWeb-Edu dataset and tokenizer revisions, shuffles documents with a bounded buffer, assigns them deterministically to training or validation, and tokenizes them with an EOS token after each document. It writes ordered `.npy` shards and a `metadata.json` manifest containing split counts, shard information, and checksums. Documents with *exactly* the same text are assigned to the same split; this is not near-duplicate detection.

That manifest describes the prepared corpus. The [dataset reader](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/data/dataset.py) then checks shard existence, dtype, shape, and token totals at startup, and gives training and validation their own sequential cursors. Its `metadata_sha256` identifies the canonicalized manifest, independent of its directory or JSON whitespace.

There is an important limit here: **the reader does not rehash every shard's contents on startup**. The preparation stage records shard checksums, but the training reader assumes the prepared files remain unchanged. A manifest identity check is therefore not, by itself, proof that every data byte has been freshly verified.

Why be this particular about a dataset? Because an experiment can't be continued meaningfully if its input silently changes beneath it. We need an identifiable token stream *before* we can reason about how far we've traveled through it.

## The plan and the journey need separate counters

Now imagine a runner who signs up for a marathon. The distance printed on the event poster isn't evidence that they ran it. We would look for a record of their actual progress.

Training has the same distinction. The [CLI](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/train.py#L143-L176) calculates a **nominal** number of tokens per optimizer update from the configured microbatch, sequence length, and gradient accumulation:

```python
tokens_per_update = config.batch_size * config.seq_len * config.gradient_accumulation_steps
```

If we plan $S$ successful updates, the simple target is $S$ times that quantity. It's useful for budgeting compute. But it isn't a measurement of what happened.

[Figure 2](#fig-three-counters) keeps the three notions apart. The prepared corpus is a fixed inventory; token processing is an activity that can revisit that inventory; successful updates are a smaller set of events, because some attempts may be skipped. The illustration shows the distinction, not actual counts from a run.

<figure id="fig-three-counters" class="reading-figure" data-kind="conceptual">
  <a href="/assets/site/training/pilot-to-pretraining/three-counters.svg" target="_blank" rel="noopener noreferrer" aria-label="Open the three training counters diagram at full size">
    <img src="/assets/site/training/pilot-to-pretraining/three-counters.svg" width="1440" height="560" loading="lazy" decoding="async" alt="Three separate rows: blocks for prepared tokens, dots and a return arrow for processed tokens, and successful update checkmarks with one crossed-out attempt." />
  </a>
  <figcaption><strong>Figure 2.</strong> Dataset size, processed token positions, and successful optimizer updates measure different things. A revisit or a skipped update breaks any simple one-to-one interpretation. <span class="figure-provenance">Original conceptual illustration; symbols are illustrative, not measured quantities.</span></figcaption>
</figure>

In the [trainer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py#L507-L581), `tokens_seen` advances when a microbatch is processed. `step` advances only when the optimizer update succeeds. With FP16, an overflow can skip an update **after the tokens were consumed**. The trainer also restarts its sequential training loader on exhaustion, dropping an incomplete tail. So `tokens_seen` is a count of processed token positions, not necessarily the number of distinct corpus tokens encountered.

<aside class="reading-question" aria-label="Check the meaning of token counts">
  <span class="reading-note-label">Which counter would you trust?</span>
  <p>Suppose the model processes a batch, FP16 overflows, and the optimizer update is skipped. Did training make progress?</p>
  <p class="reading-answer"><strong>The data was processed, but no optimizer step completed.</strong> The token counter increases, the successful-step counter does not, and the skipped update is recorded. Reporting only the planned steps would hide precisely this distinction.</p>
</aside>

And there is one more boundary. A `train_tokens` target in the preparation script means *prepare at least this many training-split tokens*, stopping after a complete document. It does **not** mean that a later model run consumed them. Nor does a `max_steps` setting prove the trainer reached its final step. To establish that, we need records from the actual execution.

## Freeze the recipe before crossing the line

During exploration, changing one setting is often the point. Perhaps gradient accumulation is too expensive, or the learning-rate schedule needs a different warmup. Those are legitimate pilot decisions. But once the long run begins, an unrecorded change stops it from being a single well-defined experiment.

I therefore treat the pilot-to-pretraining boundary as a **recipe freeze**. The model configuration, prepared data identity, optimizer partition and hyperparameters, precision, batch geometry, schedule, random seed, and checkpoint policy should be decided and recorded. The pilot tells us whether the candidate is defensible; the run manifest tells us which candidate actually started.

This is not just a convention scribbled in a notebook. The [run recorder](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/run.py#L119-L193) writes `run.json` with a run ID, per-session IDs, Git commit and dirty-state information when available, model and optimizer details, effective training settings, hardware/software information, and training/validation dataset identities. On resume, it records the **restored effective recipe**, rather than simply trusting the flags typed into the new command.

The choice to freeze is still **a scientific decision, not an automatic property of `run.json`**. The software records and rejects certain incompatibilities; it doesn't independently certify that the selected learning rate is good, that the available GPU budget is sufficient, or that every operating procedure was followed.

A useful go/no-go question is therefore not “did the script run?” but “do I know *which* script, model, data, and settings ran—and do I have a reason to keep them unchanged?”

## What should the pilot show while it is running?

The answer is not one magical threshold. I would inspect a *pattern* of evidence: finite loss on successive batches, sensible validation behavior, gradient norms that don't repeatedly become pathological, the frequency of clipping or skipped updates, usable throughput, and memory behavior compatible with the intended hardware. With a routed model, expert usage and router-balancing diagnostics also deserve attention; in the MLA layers, repeated QK-Clip intervention is another signal worth inspecting.

None of these observations, taken alone, guarantees future stability. A pilot can be quiet for a while and still fail later. Nor should a single early validation number be dressed up as a final quality result. The point is to catch reasons **not** to commit a much larger budget.

The project already emits the raw material for that judgment. Its [metrics writer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/metrics.py) appends timestamped `session_start`, `train`, `val`, `checkpoint`, `overflow`, and `session_end` events to `metrics.jsonl`. The [monitor](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/monitoring.py) separates training time from evaluation, checkpointing, profiling, and reporting. This matters for throughput: a rate measured while saving a checkpoint answers a different question from how quickly training computations themselves run.

<aside class="reading-note" aria-label="What a pilot decision does and does not automate">
  <span class="reading-note-label">The decision is still ours</span>
  <p>The repository supplies configurations, diagnostics, and checks. It does not contain a universal, automated go/no-go threshold for full pretraining. Accepting a pilot requires inspecting its actual run records against the intended hardware budget and stability criteria.</p>
</aside>

## An interruption should change the wall clock, not the experiment

Suppose the pilot passes and the full run starts. Some time later, the process stops. Is the saved model enough to continue?

Not quite. We need more than weights: optimizer state, the successful-step count, tokens consumed, data cursors, random-number generator states, the precision scaler when applicable, and the recipe that gives the next update its meaning. Otherwise we might restart from familiar weights while following a different optimization trajectory.

The [checkpoint implementation](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py#L287-L440) saves at completed-update boundaries, writes a temporary file before replacing the checkpoint, and checks model, optimizer layout, training geometry, data manifest identity, and runtime compatibility before restoring state. Some changes, such as a longer `max_steps`, can be permitted; incompatible optimizer or model changes cannot simply be waved through. Continuing deterministically also depends on equivalent software, hardware, and execution settings.

The most convincing check isn't that a resume command prints “loaded.” It's to compare the *next updates* against a run that never stopped. A [resume integration test](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_checkpoint_resume.py#L144-L193) is written to compare the resulting losses, model and optimizer states, loader positions, and random values. Another test checks the CLI continuation across processes and its appended run sessions. Those are **test definitions inspected in the repository**, not claims of a fresh test execution here.

That is enough detail for our decision. The precise serialization and failure cases belong in **Reliability**. Here, the important idea is that a pilot has to demonstrate a credible path through an interruption before the final run depends on that path.

## When can we call the run complete?

At the end, the question changes. We are no longer deciding whether to start. We are deciding what we can honestly claim happened.

The repository includes a [run summarizer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/summarize_training_run.py) that reads `run.json` and `metrics.jsonl`. It validates logged window counters, checks for non-finite or inconsistent scalar fields, and can require a `session_end` at a requested successful step. It also distinguishes the latest session from earlier ones, and reports training-only throughput separately from wall-clock throughput. A [unit test](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/training/test_training_summary.py) is written to reject inconsistent counters and a missing completion event.

That's useful evidence, but it has an exact boundary: **the summarizer validates the logs of the latest session, not an entire multi-session history, the tensors in every checkpoint, or the scientific quality of the resulting model**. To make an end-to-end claim after an interrupted, resumed run, we also need to reconcile all sessions and their starting and ending counters, the final checkpoint, and the separate evaluation artifacts. A completed training job is not the same thing as a validated research result.

<aside class="reading-evidence" aria-label="What the current sources establish">
  <span class="reading-note-label">Evidence &amp; limits</span>
  <p><strong>Inspected:</strong> data-preparation and loader code, training and resume paths, run logging, summary validation, configurations, and relevant test definitions at one pinned repository revision.</p>
  <p><strong>Not established here:</strong> that a particular pilot passed its gates, that a full pretraining run reached its planned token budget, or that a final model achieved a particular evaluation result. Those require audited run artifacts, not configuration targets or the existence of tests.</p>
</aside>

So when is a pilot convincing enough? **When the evidence is strong enough to stop changing the machinery and start spending compute on the actual experiment.** The loss must be behaving, the data and recipe must be identifiable, the counters must mean what they say, and continuation must be credible. A passing pilot doesn't promise a successful model. It gives us a defensible reason to begin—and the records we'll need to understand what happened afterward.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Data &amp; accounting</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/prepare_data.py">Preparation and manifests</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/data/dataset.py">token stream and cursors</a>, and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py">successful updates versus processed tokens</a>.</dd>
    <dt>Run &amp; continuation</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/run.py">Run provenance</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_checkpoint_resume.py">resume tests</a>, and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/summarize_training_run.py">session summary checks</a>.</dd>
    <dt>Evidence status</dt>
    <dd>Source and test definitions inspected at commit <code>c458fa252152e08a2d22e83af25ddb4e21210037</code>; no fresh execution of the tests and no audited pilot or full-run measurement is claimed in this chapter.</dd>
  </dl>
</aside>
