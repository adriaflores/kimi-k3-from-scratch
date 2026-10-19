---
title: "MetricsWriter and TrainingWindow"
description: "Persists append-only JSONL training events and aggregates scalar statistics over logging windows."
sourcePath: "src/kimi_k3/training/metrics.py"
status: published
---

This module provides two deliberately small components: `MetricsWriter` owns disk records for one training session, and `TrainingWindow` accumulates **detached scalar statistics** until the next training log. Neither module executes the model or collects tensor hooks [source L1–L10](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py#L1-L10). These are operational records, distinct from independently reviewed Results evidence.

## JSONL writer and incomplete-last-record recovery

`MetricsWriter(run)` builds `run.directory / "metrics.jsonl"`, copying `run_id` and `session_id` from its run context [source L9–L17](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py#L9-L17). In `__enter__`, it opens the file in binary append/read mode, checks whether the last byte is a newline, and if not, scans backward in bounded 4096-byte chunks to the last complete line [source L19–L42](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py#L19-L42). It truncates only the interrupted tail and records the count of discarded bytes. A newline acts as the record boundary; the code does not independently validate every historical JSON object.

Every `write(event,state,**values)` constructs a flat event record with `schema_version=1`, UTC ISO timestamp, run/session IDs, `state.step`, cumulative `tokens_seen` and `skipped_updates`, merged with event-specific values [source L44–L54](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py#L44-L54). It serializes with `allow_nan=False`, writes one newline-terminated JSON object, and flushes immediately. Invalid NaN/Infinity values fail instead of creating nonstandard JSON. `__exit__` closes the stream.

## TrainWindow counts attempts separately from successful steps

The dataclass [source L60–L76](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py#L60-L76) starts with zero loss sum, token count, optimizer **attempts**, successful updates, skips, gradient norm sum/max, nonfinite gradient norm count and clipping count. Calling `add(loss,tokens,grad_norm,skipped,max_grad_norm)` increments attempts and weighted loss/token totals **even for a skipped FP16-overflow attempt**, because those input tokens have been consumed [source L78–L88](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py#L78-L88). It increments successful update count and gradient aggregates only when `skipped=False`.

```python
self.loss_sum += loss * tokens
self.tokens += tokens
self.attempts += 1
self.skips += int(skipped)
```

The excerpt is indentation-normalized. `clipped_updates` counts successful updates whose **pre-clipping** gradient norm exceeded a positive configured `max_grad_norm`; it does not report the fraction of QK-Clip heads rescaled.

## Meaning of the summary fields

`summary(lr)` [source L90–L100](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py#L90-L100) returns token-weighted `train_loss`, current LR, tokens, `attempted_updates`, `updates`, `skipped_updates_in_window`, average/max successful gradient norms, `clipped_fraction` and `nonfinite_grad_norms`. When there are no successful updates, norm averages and clipping fraction are `None`, not fabricated zeros. A `TrainingWindow` is reset by `trainer.py` at each training-log boundary; it is not serialized as checkpoint training state.

## Verification

[test test_metrics.py L10–L46](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_metrics.py#L10-L46) checks preservation of prior complete lines, truncation of a broken tail, session IDs and failure on invalid JSON numbers. [test test_metrics.py L49–L70](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_metrics.py#L49-L70) checks loss weighting and skipped-update semantics. [test test_training_metrics.py L43–L86](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py#L43-L86) asserts observed event sequences, token counts and train/validation aggregates in a controlled run. Assertions inspected; no new test execution is claimed.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py">metrics.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_metrics.py">test_metrics.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py">test_training_metrics.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
