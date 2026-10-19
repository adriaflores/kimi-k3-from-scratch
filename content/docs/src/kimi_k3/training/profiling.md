---
title: "ProfileConfig and TrainingProfiler"
description: "Captures a bounded CPU/CUDA profiling window with temporary module scopes and compressed traces."
sourcePath: "src/kimi_k3/training/profiling.py"
status: review
---

`TrainingProfiler` is an **optional, bounded instrumentation context**, not a continuously active measurement system. Normal training with `config=None` does not construct a Kineto profiler or install module hooks [source L1–L2](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L1-L2). It is designed to observe a fixed number of *optimizer attempts*, including FP16-overflow skips, while retaining clean model/module state after the window.

## Define a finite capture window

`ProfileConfig(wait=1,warmup=1,active=1)` [source L26–L42](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L26-L42) validates nonnegative integer wait and positive integer warmup/active (booleans are explicitly rejected). The `total_attempts` property is their sum. These are **attempt counts**, not successful optimizer-update numbers and not wall-clock seconds. Setting a shorter training session than the scheduled window can therefore produce only a partial capture.

## Initialize without touching the model

The profiler stores references to model, device, training state, chosen config, optional metrics and output directory [source L45–L71](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L45-L71). If profiling is enabled, an explicit directory or a metrics-writer-derived `profiles/<session_id>` directory is required. `describe()` reports the planned config, activities and capture settings [source L73–L86](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L73-L86); it does not imply a trace already exists.

On `__enter__` [source L88–L121](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L88-L121), the profiler creates an empty output directory, records starting step/token/skip counters, chooses CPU activity and adds CUDA activity when using a GPU, then writes `profile.json` with pending status. The PyTorch profiler uses a **single-cycle** wait/warmup/active schedule (`repeat=1`), `profile_memory=True`, `record_shapes=False`, `with_stack=False` and `acc_events=False`. CUDA timing availability depends on the installed profiling support; requesting CUDA activity does not promise per-kernel trace lanes on every machine.

## Temporary scopes and attempt boundaries

`region(name)` creates `torch.profiler.record_function` only during an active profiler action; otherwise it returns a null context [source L123–L126](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L123-L126). `_install_hooks()` [source L128–L147](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L128-L147) visits `model.layers` and the `attn`/`ffn` submodules, installing paired forward-pre and forward hooks that mark module types such as `model/GatedMLA` and `model/StableLatentMoE`. The hook teardown removes handles and closes leftover scopes [source L149–L156](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L149-L156), including on failures.

`step()` [source L158–L167](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L158-L167) advances **once after all work for an attempted update**, potentially including intervening evaluation, checkpoint and reporting. When the configured total is reached, `finish()` is triggered. This distinction matters: profiler phase scopes are nested and their inclusive times can overlap; summing all reported scopes will double count work.

## Export safely, including exceptional paths

`_export()` [source L168–L196](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L168-L196) first writes a Chrome trace, streams it through a **reused 64 KiB buffer** into `trace.json.gz` with gzip compression level 1, removes the raw trace, and writes CPU/CUDA operator and memory summary tables. CPU and CUDA self-memory tables show *net allocations per operator*, not the process's peak RSS or peak VRAM. The compressed trace is published via a temporary filename and rename; a compression exception leaves the prior archive intact [source L12–L24](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L12-L24).

`_write_metadata` [source L198–L212](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L198-L212) writes `profile.json` atomically with run/session identity, planned options, completed attempts, captured active attempts, start/end counters, trace/summary locations, regions, CUDA timing status and stop reason. `finish()` [source L214–L232](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py#L214-L232) closes hooks and profiler and labels results `complete`, `partial`, or `no_capture` based on whether the bounded window finished and whether a usable trace exists. Failure teardown does not leave persistent hooks on the model.

## Verification and limits

[test test_profiling.py L18–L70](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_profiling.py#L18-L70) checks invalid windows, disabled overhead and cleanup after forward errors. [test test_profiling.py L106–L130](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_profiling.py#L106-L130) tests streaming compression and failure preservation. [test test_training_profiling.py L65–L136](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_profiling.py#L65-L136) checks a bounded real capture, scopes, trace metadata and preservation of training state, and [test test_training_profiling.py L138–L196](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_profiling.py#L138-L196) checks incomplete windows and resume-session isolation. Tests inspected; no profiling run was made while authoring this page, so no speed or memory improvement is asserted.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/profiling.py">profiling.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_profiling.py">test_profiling.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_profiling.py">test_training_profiling.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
