---
title: "TrainingMonitor"
description: "Separates training and session clocks and samples process/CUDA memory at event boundaries."
sourcePath: "src/kimi_k3/training/monitoring.py"
status: published
---

`TrainingMonitor` measures **completed training phases** for a single invocation. It distinguishes active training time from session elapsed time, which also includes evaluation, checkpointing, reporting and profiling [source L13–L18](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L13-L18). It is a scalar/event monitor, not a CUDA kernel profiler: see `training/profiling.py` for optional traces.

## Initialization and phase clocks

At construction [source L20–L33](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L20-L33), the monitor captures the target device and optional metrics writer, attempts to obtain a `psutil.Process` only if metrics are enabled, and initializes a phase dictionary for `train`, `eval`, `checkpoint`, `reporting` and `profiling`. If metrics are enabled on CUDA, it resets CUDA allocator peak-memory statistics **once per session**.

`now()` uses `time.perf_counter()`. `synchronize()` calls `torch.cuda.synchronize(device)` only on CUDA. `begin(state)` synchronizes and remembers session start time and the starting `tokens_seen` value [source L35–L47](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L35-L47). `resume_train()` synchronizes and starts an active-training segment; `pause_train()` synchronizes, measures elapsed time, and adds it to both cumulative `train` time and the current log-window training time [source L48–L56](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L48-L56). `finish_phase(phase,started)` synchronizes and credits a named auxiliary phase [source L58–L62](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L58-L62). Thus GPU asynchronous execution is bounded by synchronization at phase transitions, not forced after **every** optimizer step.

## Which throughput figure is which?

`interval_summary(window)` [source L64–L69](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L64-L69) divides **window tokens by training-only window seconds** for `train_tokens_per_second`, and training seconds by attempted updates for `seconds_per_attempt`. `session_summary(state)` [source L71–L78](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L71-L78) instead uses **new tokens since this invocation began divided by total session elapsed time** for `tokens_per_second`. The denominators are different: adding evaluation or checkpoint cost reduces session throughput even if the training-only rate stays unchanged. Zero-duration rates become `None`.

## Event-boundary memory and reports

`sample_memory()` [source L80–L101](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L80-L101) collects process RSS when available and records the **highest RSS observed at sampling points**, not a continuously monitored true RAM peak. On CUDA it also reports current allocated/reserved bytes and allocator-tracked peak allocated/reserved bytes since session reset. Without a CUDA device, these fields are explicitly `None`; unavailable `psutil` likewise gives `None` RAM readings.

`report(event,state,...)` [source L103–L117](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py#L103-L117) takes a session timing snapshot **before** sampling, printing and JSON writing, optionally includes a MoE snapshot, prints a supplied message, delegates JSON serialization to `MetricsWriter`, and credits this call's overhead to `reporting` for subsequent records. This timing convention is intentional: each event record's session snapshot does not include the cost of writing itself.

## Verification

[test test_monitoring.py L12–L39](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_monitoring.py#L12-L39) checks sampled RSS peaks and the session-versus-training denominator logic; [test test_monitoring.py L60–L77](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_monitoring.py#L60-L77) checks CUDA allocator peak resets. [test test_training_metrics.py L89–L169](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py#L89-L169) injects controlled phase durations and verifies training-only throughput excludes evaluation/checkpoint/reporting. Assertions were reviewed, not freshly executed.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/monitoring.py">monitoring.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_monitoring.py">test_monitoring.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py">test_training_metrics.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
