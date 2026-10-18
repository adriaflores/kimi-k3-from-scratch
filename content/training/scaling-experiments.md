---
title: "Scaling Experiments"
description: "How controlled, low-cost probes expose memory, throughput and learning-rate risks before a larger Kimi K3 training run."
question: "What can small experiments tell you before you commit to a big run?"
order: 4
status: draft
---

Imagine booking a GPU for a long training run. A few hours in, you discover that the sequence length barely fits in memory. Or perhaps the model fits perfectly, but each update takes so long that the planned token budget is out of reach. Neither problem requires a long run to discover. You would much rather learn about it in the first few minutes.

That is the point of a scaling experiment. **Don't ask a small run to predict the final model. Ask it to remove one expensive uncertainty.** A useful probe has a specific question, a controlled change and a measurement that can actually answer it. The hard part is knowing what the result *doesn't* tell us.

## A cheap question before an expensive one

The [training loop](/training/training-loop/) established what an update means. Before scaling those updates, there are several different things worth checking, and they deserve different experiments.

A **smoke test** asks whether a configuration can execute a few updates without an immediate failure. A **fixed-batch overfit test** asks whether the system can learn a deliberately repeated example. A **stability probe** lets successive batches flow through the model and watches what happens to loss and gradients over a longer stretch. These tests get progressively closer to ordinary training, but they are not interchangeable.

The repository makes that progression concrete. Its [smoke configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/smoke.yaml) specifies 20 updates at batch size 1 and context length 128. The [overfit configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/overfit.yaml) repeats a batch for up to 2,000 updates. The [stability configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/stability.yaml) instead specifies 1,000 updates on successive batches, with accumulation, warmup and cosine decay.

Those are **defined experimental recipes**, not three verified outcomes. The YAML tells us what we can attempt. Only a corresponding run manifest and metrics can tell us what actually happened.

## How big is a batch, really?

Suppose one microbatch fits comfortably on the GPU. Is that enough to choose the training batch? Not quite. The model updates its weights only after all accumulated microbatches have contributed their gradients.

For this trainer, the nominal number of training tokens processed per successful update is

$$
N_{\text{update}} = B\,T\,A,
$$

where $B$ is the number of sequences **per microbatch**, $T$ is their length and $A$ is the number of gradient-accumulation passes. In the checked-in stability recipe, $B=1$, $T=128$ and $A=4$. That means **512 nominal tokens per update**, even though each forward pass sees only 128 tokens. The [CLI reports this quantity](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/train.py#L151-L166) explicitly.

Why not simply make $B$ four times larger? Because those choices place different demands on the machine. Increasing the physical microbatch retains more activations at once. Accumulation keeps the microbatch smaller, at the cost of additional forward and backward passes before an optimizer step. In Kimi K3, there is another subtlety: **Quantile Balancing updates routing state on each training forward**, so four microbatches are not guaranteed to be mathematically identical to one concatenated batch. The [trainer documents this distinction](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py#L484-L496).

<aside class="reading-question" aria-label="Question about comparing training progress">
  <span class="reading-note-label">Check your intuition</span>
  <p>If two experiments both reach step 100, have they learned from the same amount of data?</p>
  <p class="reading-answer"><strong>Only if their token budgets per update match, and neither has consumed extra tokens in skipped attempts.</strong> Otherwise the step counter hides a different amount of training exposure. Compare <code>tokens_seen</code> as well as successful updates, and keep the data stream comparable.</p>
</aside>

## Context length changes the experiment

The sequence length $T$ looks like another multiplier in that equation, but changing it does more than add tokens. A longer context changes which earlier tokens the model can use for each prediction. It also changes activation memory and the work performed by the attention layers. An experiment at $T=128$ can therefore tell us that a short-context configuration runs, not that $T=1024$ will be equally efficient or even fit.

There is also an implementation constraint. For chunkwise or fused KDA, the [training entry point requires sequence lengths divisible by the configured KDA chunk size](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/train.py#L50-L69). In the [scaled model configuration](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/model/kimi_k3_300m.yaml), that chunk size is 64. A context sweep must respect this boundary rather than treating every integer as a valid candidate.

A useful context probe therefore measures two separate things: whether the selected configuration fits within memory, and how many *training tokens per second* it processes after startup overhead. Neither observation, on its own, shows that the longer context improves final language-model quality.

## Change one knob, or learn nothing

Now suppose one trial has a lower loss than another. Was the learning rate better? It is tempting to say yes, especially when GPU time is scarce. But if the second trial also changes batch size, context and number of tokens processed, the comparison cannot isolate the learning rate.


<figure id="fig-controlled-comparison"
        class="reading-figure"
        data-kind="conceptual">
  <a href="/assets/site/training/scaling-experiments/controlled-comparison.webp"
     target="_blank"
     rel="noopener noreferrer"
     aria-label="Open controlled comparison diagram at full size">
    <img
      src="/assets/site/training/scaling-experiments/controlled-comparison.webp"
      width="1600"
      height="900"
      loading="lazy"
      decoding="async"
      alt="Two experimental comparisons. On the left, batch size, context length and learning rate all change between runs. On the right, batch size and context remain fixed while only learning rate changes."
    />
  </a>
  <figcaption>
    <strong>Figure 1.</strong> Changing one variable at a time
    makes it easier to attribute differences between experiments.
    Original conceptual illustration, not measured results.
  </figcaption>
</figure>


For a learning-rate probe, hold the model, tokenizer, data split, physical batch, context, accumulation, optimizer recipe and evaluation method fixed. Start from comparable initialization and optimizer states. Change the learning rate, then examine early loss, gradient norms, non-finite events and any skipped updates over the same token budget. The [trainer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/trainer.py#L26-L112) exposes learning rate, warmup and decay as explicit configuration values, making the proposed comparison reproducible in principle.

Even this is a limited result. A rate that looks stable over a short window may misbehave later. A rate that reduces training loss quickly may give worse validation loss after enough tokens. And one seed gives a noisier answer than repeated runs. **A short probe is good at ruling out obvious trouble, not declaring a universal winner.**

The same discipline applies to batch or context comparisons. Vary the quantity under study, keep the other knobs fixed, and record enough metadata to explain any unavoidable differences. A comparison is only as useful as its controls.

## Throughput has two clocks

One setting can perform excellent training updates and still make a poor long-run choice. Perhaps it saves checkpoints slowly, or evaluation occupies a substantial fraction of elapsed time. We need to distinguish the speed of the learning work from the speed of the entire session.

The [monitor](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/monitoring.py) records **training tokens per second** over training windows and **session-wide tokens per second** over elapsed time. Training windows include batch retrieval and the training computation but exclude intervening evaluation, checkpointing and reporting. The [integration test for timing](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_training_metrics.py#L89-L170) checks that distinction with a controlled clock. It is a test of metric accounting, not a GPU benchmark.

A short warmup also matters. Early updates can include initialization and other one-off costs. The [run summarizer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/summarize_training_run.py#L61-L143) can exclude complete logging windows up to a chosen step and compute a time-weighted steady training rate. It separately reports session-wall throughput and memory peaks. Comparing one run's warmup-heavy average to another run's steady rate would make the faster configuration look slower for the wrong reason.

<aside class="reading-evidence" aria-label="What timing and profiling can establish">
  <span class="reading-note-label">What the instrumentation can tell us</span>
  <p><strong>Available:</strong> Session manifests, token counters, training and wall-clock rates, CUDA memory peaks where CUDA is available, and an optional bounded profiler that marks data, forward, backward and optimizer work.</p>
  <p><strong>Not established here:</strong> A measured ranking of GPU settings, an optimal batch/context/learning-rate combination, or the throughput of a future hardware configuration. Those claims require comparable run artifacts, not just the instrumentation code.</p>
</aside>

## What earns the right to scale?

A small experiment earns a narrow conclusion. A successful smoke test supports basic execution under its particular conditions. A fixed-batch overfit supports the ability to learn that repeated example. A controlled throughput probe supports a speed comparison on the hardware and software it actually used. None of these automatically predicts an uninterrupted, stable pretraining run.

The [pilot recipe](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/experiments/pilot/training.yaml) and [pretraining recipe](https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/experiments/pretraining/training.yaml) specify a larger configuration: 32 sequences of length 1,024 per update, with BF16 precision and a shared learning-rate schedule. They show the **chosen configuration recorded in the repository**, not a published batch/context/LR ablation or proof that those values were best. The currently inspected repository snapshot does not include the original logs needed to reconstruct that selection.

So the decision to scale is not “the small run went well.” It is more precise: we have eliminated immediate execution problems, we can state the cost of a candidate configuration, and we know which uncertainties still require a longer experiment. The next task is to test those assumptions on the **target GPU itself**, with memory headroom, profiling and recovery checks. That is where GPU Qualification begins.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Probe recipes</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/stability.yaml">Stability configuration</a> and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/configs/training/overfit.yaml">fixed-batch configuration</a>, both inspected as planned recipes.</dd>
    <dt>Measurement code</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/monitoring.py">Timing and memory monitoring</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/src/kimi_k3/training/profiling.py">bounded profiling</a>, and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/scripts/summarize_training_run.py">run summarization</a>.</dd>
    <dt>Verification boundary</dt>
    <dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/integration/test_training_metrics.py">Timing-accounting tests</a> and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/c458fa252152e08a2d22e83af25ddb4e21210037/tests/unit/training/test_training_summary.py">summary-invariant tests</a>. Test definitions were inspected; no new execution or cross-run scaling measurements are claimed.</dd>
  </dl>
</aside>
