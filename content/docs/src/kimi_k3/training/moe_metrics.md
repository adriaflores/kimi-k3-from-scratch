---
title: "MoEDiagnostics"
description: "Collects per-interval routed-expert assignment counts and QuantileBalancer snapshots without persistent hooks."
sourcePath: "src/kimi_k3/training/moe_metrics.py"
status: published
---

`MoEDiagnostics` exists to answer a narrow operational question: **which routed experts were selected by actual training forwards during this log interval?** It avoids reconstructing choices from router probabilities, and it does not change routing or create checkpoint state [source L1–L13](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/moe_metrics.py#L1-L13). The model's `StableLatentMoE` exposes the temporary `_expert_load_counts` interface that this component activates.

## Discover and arm matching modules

At initialization, the collector scans `model.named_modules()` for the `_expert_load_counts` attribute and retains `(module_name,module)` pairs [source L15–L19](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/moe_metrics.py#L15-L19). It uses interface discovery, not a strict class check, so compatible wrappers can contribute. With `enabled=False`, the list is empty. `__enter__` rejects an already active collector and attaches a plain Python integer list of length `num_routed_experts`, initialized to zero, to each discovered module [source L21–L26](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/moe_metrics.py#L21-L26). No gradients, tensors or RNG state are retained by the collector.

## Read/reset an interval snapshot

`snapshot()` [source L28–L43](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/moe_metrics.py#L28-L43) copies the current counts, totals the routed assignments, reads detached FP32 `quantile_balancer.biases` and constructs the per-layer record. For `K = module.top_k`, one token contributes `K` routed assignments, so `tokens = sum(counts)//K`. `max_load_over_mean` is `max(counts)*E/sum(counts)`, where `E` is routed-expert count; `zero_load_experts` counts exactly zero entries. If there were no assignments, both balance diagnostics are `None`. Biases that are nonfinite become JSON-safe `None` in the record **without mutating the actual stored biases**.

The collector then replaces all Python counters with zeros for the next interval. It does **not** reset QB biases, which are model state and represent the latest training-forward balancing update. The `__exit__` method restores `_expert_load_counts=None` [source L45–L48](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/moe_metrics.py#L45-L48), including when the enclosing training context exits due to an exception.

## Interpretation and limits

These metrics count **selected expert assignments**, not expert FLOPs, parameter utilization across the entire run or evaluation-time selections. Shared experts are not counted as routed assignments. They are collected only while the diagnostic context is active, and a disabled collector returns empty dictionaries. Values are aligned with the trainer's logging windows, whose boundaries need not coincide with checkpoint boundaries.

[test test_moe_metrics.py L14–L52](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_moe_metrics.py#L14-L52) checks that actual biased top-k choices are counted, evaluation is excluded, metrics are computed correctly and QB state is preserved. [test test_moe_metrics.py L55–L80](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_moe_metrics.py#L55-L80) checks cleanup and nonfinite bias serialization. [test test_training_metrics.py L173–L275](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py#L173-L275) tests interval integration with accumulation and overflow behavior. Tests inspected, not rerun.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/moe_metrics.py">moe_metrics.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_moe_metrics.py">test_moe_metrics.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py">test_training_metrics.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
