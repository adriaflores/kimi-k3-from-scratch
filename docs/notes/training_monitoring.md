# Training monitoring

This guide covers the persistent monitor and the optional short profiler for
the current eager implementation. The aim is to identify a measured cost,
change one component, and check throughput, memory and training behavior.
Run commands from the repository root with prepared format-version-2 data.

## Practical workflow

### 1. Record an unprofiled baseline

This example uses the real tiny Kimi K3 model, with KDA, MLA and routed MoE.
It is a functional and measurement check; use the intended architecture and
batch/context for an optimization decision. Its vocabulary must match the
prepared data. Replace the data path and use a fresh run folder each time.
Use `--device cpu` when checking the workflow without CUDA.

```bash
uv run python scripts/train.py \
  --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml \
  --device cuda --max-steps 20 --log-interval 5 \
  --run-dir results/training/baseline_v1
```

After normal completion, the folder contains `run.json`, `metrics.jsonl`
and `last.pt`. The final metrics event is `session_end`. Inspect the last
train interval, validation result and session totals with this standard-library
snippet, which reads the file a line at a time:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

latest = {}
with Path("results/training/baseline_v1/metrics.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        latest[record["event"]] = record
for event in ("train", "val", "session_end"):
    print(event)
    print(json.dumps(latest[event], indent=2))
PY
```

`train_tokens_per_second` measures the training interval; the final
`session.tokens_per_second` includes evaluation, saves and reports. Inspect
several later train intervals rather than treating the first one as steady
state. A resumed run can have overlapping step numbers; group comparisons by
`session_id`.

To check resume, repeat the baseline command with `--max-steps 40` and add
`--resume results/training/baseline_v1/last.pt`. Keep the same recipe and
run folder. Forty is the total target, so this adds twenty successful updates.
The run ID stays the same and a second session is appended; its timers and
load-count windows start again.

### 2. Capture a short diagnostic window

Use a separate run for diagnosis:

```bash
uv run python scripts/train.py \
  --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml \
  --device cuda --max-steps 10 --log-interval 5 \
  --run-dir results/training/profile_v1 \
  --profile --profile-wait 2 --profile-warmup 1 --profile-active 1
```

The active window is the fourth local optimizer attempt. Read `profile.json`
first: expect `status: complete`, four completed attempts and one active
attempt. FP16 skips still advance this window. On CUDA, also check
`cuda_timings_captured` and the kernel-tracing capability before interpreting
device times or timeline lanes. Then read `summary.txt`; open
`trace.json.gz` in a compatible Chrome trace viewer, decompressing it first
if required. Logging, evaluation and saves appear only if their cadence falls
inside the active attempt.

### 3. Choose one change and measure again

| Observation | Next evidence to inspect |
| --- | --- |
| Forward or backward takes most of the capture | CUDA operator self times and the timeline; forward family totals locate KDA, MLA or MoE work, while backward uses autograd operator names |
| `train/data` is expensive | Timeline of batch preparation and transfers; this region includes both, so its total alone cannot distinguish them |
| Some experts receive most assignments | `expert_counts` and `max_load_over_mean` across several windows, then the MoE region; small windows can leave experts unused without a routing defect |
| Memory limits the intended batch/context | Unprofiled CUDA allocated/reserved peaks and RSS samples, then profiler allocation events; operator net bytes are not peak residency |
| Training rate is good but wall rate is low | `session.eval_seconds`, `checkpoint_seconds` and `reporting_seconds`; their cost is visible outside the training-only rate |

Inclusive phase/family totals overlap. Use them to locate work, then use
operator self times and the timeline to investigate it. CPU totals include
host dispatch and cannot by themselves rank GPU kernel cost. A high forward
family total does not attribute that family's backward work.

Compare fresh, unprofiled runs with the same data, seed, model dimensions,
batch/context, precision, accumulation and reporting/evaluation/save cadences.
Record the code revision or preserve its dirty diff. Discard the same initial
warmup intervals and compare several later windows; repeat when variation is
as large as the apparent gain. Keep both training-only and wall rates, CUDA
allocated/reserved peaks, and sampled RSS. Check losses, gradients, FP16 skips
and MoE loads alongside speed; validate numerical correctness with the tests
appropriate to the change. A gain in the tiny model is evidence for that
configuration, so measure the intended model before projecting a full run.

## Verification

```bash
uv run python -m pytest -q tests/unit/training \
  tests/unit/moe/test_stable_latent_moe.py \
  tests/integration/test_training_metrics.py \
  tests/integration/test_training_profiling.py \
  tests/integration/test_train_cli.py \
  tests/integration/test_checkpoint_resume.py
```

The suite checks JSONL persistence and interrupted-tail recovery, interval
aggregation, timing/memory semantics, actual expert assignments and QB
snapshots, bounded/partial/disabled profiling and hook cleanup. It also checks
continuous versus resumed training in fresh CLI processes and exact training
state/RNG equivalence with profiling enabled. CUDA cases skip when no CUDA
device is available. CPU results validate the workflow; GPU throughput and
memory decisions need measurements on the target device.

## Reference

The definitions below describe the fields and boundaries used by the monitor.

### Run identity and provenance

Every CLI invocation records its effective settings in `run.json` before
training starts. Fresh runs default to unique folders under
`results/training/runs/`, with `last.pt` as their checkpoint. Use `--run-dir`
to choose a new folder; existing run folders are only accepted for resume.
An explicit `--checkpoint` still selects an alternative output path.

The manifest records model/training configurations, seed, actual Muon/AdamW group
settings and Newton–Schulz compilation mode, parameter count, Git commit and dirty status, Python/NumPy/PyTorch
and CUDA/cuDNN versions, hardware and execution settings, and data manifest
hashes plus dataset/tokenizer revisions when present in the data metadata.
Git fields are `null` when unavailable; a dirty flag does not preserve the
uncommitted diff. Archive or commit local code changes for reproducible runs.

Resume reuses the manifest beside its checkpoint, or the folder explicitly
selected with `--run-dir`. It preserves the run ID and prior sessions and
atomically appends a new session AFTER checkpoint restoration, including the
input/output checkpoint paths, starting counters/cursors and effective settings.
The session identifies whether RNG comes from the initial seed or a checkpoint;
on resume, the seed argument does not replace checkpoint RNG. A checkpoint
without a discoverable manifest starts a new run linked to that checkpoint.
When using a checkpoint outside the run folder, pass the original `--run-dir`
to retain its identity. Existing manifests reject unrelated checkpoint paths
or different model/data identities. Checkpoints inside the run folder also
record a relative path, so moving the whole folder retains the run identity.
This is CLI bookkeeping: checkpoint format 7 remains unchanged. The direct
Python `train` API accepts an optional open metrics writer.

### Persistent learning metrics

The CLI writes `metrics.jsonl` beside `run.json`. The JSONL writer uses only
the standard library. Each newline-terminated record is flushed after writing and
includes schema version, UTC timestamp, run/session IDs, successful update
`step`, cumulative `tokens_seen` and cumulative `skipped_updates`.

Events are `session_start`, `train`, `val`, `checkpoint`, `overflow` and `session_end`.
`train` follows `log_interval` and always includes the final partial interval;
`val` follows `eval_interval` and the final update. Validation does not reset
the training aggregation window. `session_end` marks normal completion.

| Train field | Meaning within the current logging interval |
| --- | --- |
| `train_loss` | Mean token loss over consumed tokens, including skipped FP16 attempts |
| `lr` | LR used for the most recent successful update |
| `tokens` | Consumed target tokens, including skipped attempts |
| `attempted_updates`, `updates` | All attempts and successful optimizer updates, respectively |
| `skipped_updates_in_window` | Attempts discarded by FP16 GradScaler |
| `grad_norm_mean`, `grad_norm_max` | Global L2 norm after accumulation/unscaling and before clipping, over successful updates |
| `clipped_fraction` | Fraction of successful updates whose norm exceeds the enabled clipping threshold; zero when disabled |
| `nonfinite_grad_norms` | Number of attempts with a non-finite gradient norm |
| `qk_clip` | Threshold, maximum causal scaled MLA logit, observed/clipped head-update counts and clipped fraction |

`val` records contain `val_loss`. Every FP16 overflow also produces an immediate
`overflow` record with attempted step, LR, loss scale, loss and consumed tokens.
Its gradient norm is `null` when non-finite, with `grad_norm_finite=false`;
non-finite values are never emitted as invalid JSON `NaN`/`Infinity`.

Resume appends to the same file without rewriting completed records. Logging
windows restart per session. If an older checkpoint is resumed, step numbers
can overlap: use `session_id` and the starting checkpoint/counters in `run.json`
to identify each trajectory. An interrupted final line is discarded before
appending, with its byte count recorded in the next `session_start`. A process
failure can lose the unfinished logging interval; completed records are kept.
Flush does not guarantee persistence through an OS or power failure.

The writer reuses scalar losses and gradient norms already computed by the
trainer, without retaining tensors. MoE diagnostics and optional profiling
are described below.

Metric semantics follow PyTorch's [gradient clipping API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.clip_grad_norm_.html)
and [AMP accumulation/clipping examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html).

### Timing, throughput and memory

`train` events add `train_seconds`, `train_tokens_per_second` and
`seconds_per_attempt`. They cover the same interval as the loss/gradient
statistics: data loading, transfers, forward/backward and optimizer work,
including skipped attempts. Validation, checkpoint I/O and reports are excluded.
The console's existing `tokens/s` value uses this training-only interval.
CUDA is synchronized at phase/report boundaries, including overflows, rather
than introducing a new synchronization for each normal update.

`val` records include their individual `eval_seconds`. A `checkpoint` event
after each successful save records `checkpoint_seconds`, including GPU-to-CPU
serialization and file I/O. All events also contain:

| Object / field | Meaning |
| --- | --- |
| `session.elapsed_seconds` | Wall time since the training loop began in this invocation |
| `session.tokens` | Consumed training tokens since this session started; excludes restored tokens and validation tokens |
| `session.tokens_per_second` | Session tokens divided by wall time, including completed validation, checkpoints and reports |
| `session.train_seconds`, `eval_seconds`, `checkpoint_seconds`, `reporting_seconds`, `profiling_seconds` | Cumulative phase costs within this session; reporting covers diagnostic snapshots, memory sampling, console printing and JSONL writes; profiling covers schedule transitions and artifact export, and is zero when disabled |
| `memory.ram_rss_bytes` | Process resident RAM in bytes, sampled with the existing psutil dependency |
| `memory.ram_rss_sampled_peak_bytes` | Largest RSS sample in this session; does not capture short-lived peaks between samples |
| `memory.cuda_allocated_bytes`, `cuda_reserved_bytes` | Current tensor allocation and caching-allocator reservation on the selected device |
| `memory.cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes` | CUDA allocator peaks since session start, including temporary allocations between samples |

Timing is sampled before the current report's own I/O; that report's cost
appears in following records. Use `session_end` for the complete loop's
throughput: it includes the final validation and checkpoint, and excludes only
its own final report. Setup, model construction and checkpoint restoration
precede the measured loop. Resume starts new timers and memory peaks without
altering checkpoint format or RNG. Rates are `null` when no time has elapsed.

Memory is sampled at event boundaries, without polling threads or model hooks.
CUDA peaks are reset once per session; the allocator cache is preserved. CUDA
fields are `null` on CPU. Allocator statistics measure PyTorch-managed memory,
not the full GPU usage shown by system tools or allocations from other processes.
RAM is process RSS, including resident data pages rather than only model tensors.
RSS is `null` when the OS/container does not permit process inspection; valid
previous RSS samples still contribute to the sampled peak.

Compare the same batch/context, precision, hardware and logging/evaluation/save
intervals. First intervals can include optimizer allocation or kernel warmup;
later `train_tokens_per_second` windows are more useful for steady-state
comparisons. The cumulative wall rate measures the cost of the complete recipe.

See PyTorch's [allocated](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html)
and [reserved](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_reserved.html)
peak counters and [CUDA synchronization](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.synchronize.html).

### MoE loads and Quantile Balancing

Each `train` record adds a `moe` object keyed by the module path, such as
`layers.1.ffn`. It covers every StableLatentMoE layer; dense layers and shared
experts are excluded. Models without routed experts produce `moe: {}`.

| Per-layer field | Meaning |
| --- | --- |
| `tokens` | Training tokens routed in the logging interval |
| `expert_counts` | Integer assignments per routed expert, in expert-index order; their sum is `tokens * top_k` |
| `max_load_over_mean` | Busiest expert's count divided by the mean count; 1 means equal loads |
| `zero_load_experts` | Number of routed experts with no assignments during the interval |
| `qb_biases` | Latest QB bias vector, in expert-index order, after the last training forward |

Counts use the actual biased top-k dispatch, excluding the extra `(k+1)`th
selection used by QB. They accumulate across microbatches and optimizer
attempts, including skipped FP16 attempts: those forwards consumed data and
updated QB. Validation neither contributes to nor resets these counts.
Logging and a new/resumed session start a fresh count window. A window with
no routed tokens has `null` load ratio and zero-load summary.

QB biases are snapshots, not interval averages. The most recent forward used
the preceding biases and then updated them for the next forward. Reading the
snapshot does not update QB; evaluation still freezes its buffers. A non-finite
bias is recorded as `null`, without changing the underlying buffer.

Collection is enabled only when `train` receives an open metrics writer, as
the CLI does. The eager dispatch already knows each expert's token count;
diagnostics add Python integer totals, without a second routing histogram,
extra GPU kernels or per-forward synchronization. Only the small bias vectors
are copied to CPU at train-log boundaries, inside the measured reporting cost.
Counters retain no tensors, are removed on normal or exceptional exit, and
are excluded from checkpoints. No training/configuration format changes.

Use load ratios and counts to spot sustained concentration or unused experts
before profiling dispatch. Interval totals can hide imbalance within individual
microbatches; this initial monitor does not collect per-token routes.

### Bounded profiling

Add `--profile` to capture one short PyTorch profiler window. It defaults to
one wait attempt, one warmup attempt and one recorded attempt. These are
local optimizer attempts in the current invocation, including skipped FP16
attempts; each includes all accumulated microbatches. Normal training creates
no profiler, profiling hooks or output directory.

The CLI prints the output folder: `RUN/profiles/SESSION_ID/`. Each profiling
invocation uses a separate folder, including resume; previous captures remain
intact. Its settings and run/session IDs are recorded in `profile.json` and
the `session_start` metric. Match those IDs to `run.json` for seed, commit,
architecture, recipe and data provenance. Profiling is an observation setting,
separate from TrainConfig and checkpoint format 7.

| Artifact | Use |
| --- | --- |
| `trace.json.gz` | Compressed Chrome-format timeline; decompress if your trace viewer requires JSON, then inspect operator ordering, GPU kernels when supported, and allocation events |
| `summary.txt` | Top operators by self CPU/CUDA time and net self tensor memory; also lists inclusive phase/module totals |
| `profile.json` | Capture settings, status, completed attempts, start/end counters, CUDA timing capability and the region totals |

Regions distinguish `train/data` (including transfer), `train/forward`
(including loss extraction), `train/backward`, `train/optimizer` (including
unscaling/clipping), `train/eval`, `train/checkpoint` and `train/report`.
Temporary hooks label training forwards by family: `model/KDAChunkwise`,
`model/GatedMLA`, `model/DenseFFN` and `model/StableLatentMoE`, summed across
layers. Family totals exclude validation and cover forward work; backward
operators appear within `train/backward`. Auxiliary regions appear only when
their cadence falls inside the active window.

The recorder stops and removes its hooks when the window completes; training
continues normally. If training ends early, status is `partial` when any active
work was captured, or `no_capture` when it never reached the active window.
No empty trace is presented as a completed capture. Exceptions also close
temporary hooks and preserve an available partial trace.

CPU activities and tensor-allocation accounting are enabled; CUDA is also
requested on a CUDA device. PyTorch can fall back to legacy CUDA event timings
when its build lacks CUPTI tracing: device times may remain available in the
tables, while the JSON has no on-device kernel lanes. Capability and whether
any CUDA times were captured are explicit in `profile.json`. Shapes, stacks,
FLOP estimates and full memory histories are omitted.

Use this for diagnosis, then measure throughput/memory with `--profile`
disabled. Recording, annotations and extra synchronization at attempt
boundaries perturb profiled training; schedule/export costs also affect wall
time and are tracked in `session.profiling_seconds`. Inclusive totals overlap,
so do not sum nested regions. Memory tables show allocation-minus-release
bytes attributed to operations, not peak residency or total process RAM;
pre-existing tensors may lack allocation history. Session CUDA peaks and
sampled RSS remain in `metrics.jsonl`.
Even a tiny eager model can emit many events. Compression reduces disk size;
recorder memory still depends on event count, including gradient accumulation.

For the Python API, pass `profile_config=ProfileConfig(...)` to `train`; it
uses the metrics writer's run/session folder. Without a writer, also provide
an unused `profile_dir`. Import ProfileConfig from
`src.kimi_k3.training.profiling`.

References: [PyTorch profiler API](https://docs.pytorch.org/docs/2.11/profiler.html)
and [profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

## Compiled Muon comparison

The optimizer can compile the Newton–Schulz core with `--muon-compile`.
Keep the eager/compiled runs unprofiled and discard compilation/warmup windows.
See [the benchmark and 4050 commands](muon_compile.md). The integrated
[Muon + QK-Clip recipe](muon_recipe.md) uses checkpoint format 7; the compilation
setting is recorded in the manifest and must match when resuming.

The second optimization phase adds `--muon-batched`. The manifest also records
batching and its chunk limits; all three must match on resume. Compare a fresh
batched run against the individual compiled baseline with the same recipe and
discard compilation/warmup windows. See [the batched Muon comparison](muon_batched.md).
