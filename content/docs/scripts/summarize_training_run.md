---
title: "summarize_training_run.py — Session log audit"
description: "Validates and aggregates the latest training session from its manifest and newline-delimited metrics."
sourcePath: "scripts/summarize_training_run.py"
status: review
---

[`scripts/summarize_training_run.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py) converts raw run logs into a compact, machine-readable acceptance summary. It reads **only** `run.json` and `metrics.jsonl`, streaming metrics one line at a time. It never loads tensors, checkpoints or profiler traces, so it can inspect a run without importing PyTorch. Its conclusions concern the **logged evidence**, not equality of checkpoint contents or physical proof of uninterrupted execution. The event producer is documented in [metrics.py](/docs/src/kimi_k3/training/metrics/).

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## Input and entry point

The [CLI](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py#L145-L159) takes a positional `run_directory`, with optional `--after-step N` (default `0`) and `--expect-step N` (optional). The first sets the cutoff used for steady training-only throughput. The second insists on a normal `session_end` record at exactly the requested cumulative successful-update step; it does **not** set the number of steps to process. Paths may be relative to the current working directory. A safe, read-only example, using a placeholder path, is:

```bash
python scripts/summarize_training_run.py path/to/run-directory --after-step 1000 --expect-step 2000
```

This example will **reject** a run that has not ended at step 2000; it is not a suggested target for any real run. The command catches `ValueError`, `KeyError` and `OSError` and exits with status 1 and `Invalid training run: ...` on stderr. Malformed JSON also raises `JSONDecodeError` (a `ValueError` subclass). Success prints indented strict JSON; it writes no files.

## Reject non-finite and inconsistent records

The first helper, [`_finite()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py#L13-L17), accepts only finite Python integers or floats above a configurable bound, excluding booleans. `_session_records()` then streams the metrics log and refuses any line lacking a trailing newline—an explicit guard against an uncommitted/torn final record. It skips events from other sessions but checks `run_id` on records belonging to the requested one:

```python
if not line.endswith("\n"):
    raise ValueError(f"Uncommitted metrics tail at line {line_number}")
record = json.loads(line)
if record["session_id"] != session_id:
    continue
if record["run_id"] != run_id:
    raise ValueError("Metric run identity differs from the manifest")
```

The logger's JSON lines must contain the expected event schema. This routine is a parser/validator, not a repair tool: it will not truncate a torn line, merge sessions or reconstruct missing records.

For each `train` event, [`_validate_train_window()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py#L34-L58) checks finite loss, LR and gradient norms. It requires positive successful-update count `updates`, nonnegative skipped count, and

$$\text{attempted updates}=\text{updates}+\text{skipped updates in window}.$$

It also demands a continuous successful-update step history (`record.step == previous_step + updates`) and token accounting equal to `batch_size * seq_len * gradient_accumulation_steps * attempted_updates`. Importantly, these tokens include **attempted** updates, not only successful optimizer updates. It checks the QK-Clip threshold against the session recipe, `0 <= clipped_head_updates <= observed_head_updates`, a finite optional max logit, and gradient-clipping fraction in `[0,1]`. A `val` event has its loss checked for finiteness, but it does not contribute to train-window sums.

## Select the most recent session, not an entire run

[`summarize_run()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py#L61-L94) rejects negative/non-integer `after_step`, loads `run.json`, requires `schema_version == 1` and a nonempty `sessions` list, and selects `sessions[-1]`. It then filters the JSONL stream to that exact session ID and the manifest's run ID. This is deliberate: a resumed run may contain multiple sessions, with different starting steps, time profiles or resources. The returned `sessions` count describes the whole manifest, while all train-window aggregates concern only the **latest** session.

During the scan, memory values are maxima across **any** event of that session, not just train windows. The code tracks `session_end` when present, the latest event (`last`), first/last **logged train-window** loss, largest logged gradient norm and QK logit, summed updates/tokens/skips/nonfinite norms, and the fractional number of successful updates that were gradient-clipped. These fractions are not interchangeable:

- `gradient_clipped_fraction` is `sum(clipped_fraction_in_window * successful_updates_in_window) / sum(successful_updates_in_window)`.
- `qk_clip.clipped_fraction` is `sum(clipped_head_updates) / sum(observed_head_updates)` and is `null` if there were no observed head updates.

The summary's first and last losses come from logged windows, not an average over all batches or an epoch-level metric. `gradient_norm_max` is the maximum recorded window maximum. No measured value is embedded in the script itself.

## Two clocks and the warmup boundary

Steady-state throughput is calculated exclusively from **complete train windows after the cutoff**. The decisive [inclusion rule](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py#L115-L118) is:

```python
if record["step"] - count >= after_step:
    steady_tokens += record["tokens"]
    steady_seconds += _finite(record["train_seconds"], "training seconds")
```

If a window starts before the cutoff and ends after it, **the whole window is excluded**. No fractional interpolation is attempted. `steady_train_tokens_per_second = steady_tokens / steady_seconds`, or JSON `null` if no included training seconds exist. Both operands are summed before division: this is time-weighted, **not** an arithmetic mean of per-window token rates. The step condition uses successful-update steps, even though the recorded token count also includes skipped attempts.

By contrast, `session_wall_tokens_per_second` is copied from the `session.tokens_per_second` field of the **last logged event**, rather than recomputed from the steady windows. It reflects the run logger's session elapsed-time definition, which can include validation, checkpointing, profiling and reporting. If the last record is not `session_end`, the script still reads its session field; no completed-session interpretation should be inferred in that case. Memory peaks similarly cover all observed events, potentially including auxiliary phases.

## Completion and report shape

Once it has seen at least one train window, the script [compares completed-session counters](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py#L119-L126) with the starting checkpoint state. If `session_end` exists, logged successful updates, tokens and skipped updates must match the final cumulative differences. Without `session_end`, `completed` is false, and `--expect-step` produces an error. The step expectation is for a **normally completed** session, not merely a last train window reaching the step.

The [returned object](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py#L127-L142) includes run/session IDs, total session count, start/last steps, completion, logged updates/tokens/skips, a copy of the recipe and hardware, optimizer algorithm names, train-window loss endpoints, gradient/QK metrics, steady train-only and session wall rates, memory peaks, and whether the last event reports positive profiling time. No optimizer buffers, model weights or dataset tensors are inspected. The summary enforces counter consistency **within** the log, not that every update was computed correctly.

## Verification and scope

The source-inspected [`test_training_summary.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_training_summary.py#L49-L107) checks latest-session selection, token/time-weighted rates, distinct wall rate, loss and clipping aggregations, full-window warmup exclusion, many malformed log conditions (including nonfinite values and identity/counter mismatches), required completion, torn tails, and a subprocess invocation under `python -S` outside the repo working directory. These are tests of the actual summarizer, but only their assertions have been inspected for this page; no results are claimed from a new execution.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/summarize_training_run.py">summarize_training_run.py · pinned revision</a></dd>
    <dt>Logs</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/metrics.py">metrics.py</a> · JSONL event producer</dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_training_summary.py">session aggregation and CLI assertions</a>; inspected, not rerun</dd>
  </dl>
</aside>
