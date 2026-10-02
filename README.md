# Kimi K3 from Scratch

A from-scratch PyTorch implementation of the Kimi K3 language backbone,
derived from the papers and verified with reference tests and experiments.

## Philosophy

Every important mechanism should be traceable through:

**paper → derivation → implementation → test → experiment**

## Scope

This repository focuses on the Kimi K3 language backbone.

Planned components:

- Kimi Delta Attention
- Gated MLA
- Attention Residuals
- Stable LatentMoE
- SiTU-GLU
- Quantile Balancing
- Per-Head Muon
- Full language backbone
- KV/recurrent-state caching
- Tiny training and generation

The vision stack and production-optimized kernels are outside the initial scope.

## Status

Work in progress.

## Training entry point

Run from the repository root with prepared uint16 token shards and their
`metadata.json` (format version 2):

```bash
uv run python scripts/train.py --data-dir /path/to/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml --device cpu \
  --batch-size 1 --seq-len 8 --max-steps 2 --eval-batches 1
```

Use `configs/model/kimi_k3_300m.yaml` for the candidate architecture. Inspect its
parameter budget with `uv run python scripts/inspect_config.py configs/model/kimi_k3_300m.yaml`.
The trainer checks that the model vocabulary matches the data, that `seq_len`
fits the context limit, and that KDA sequences are divisible by the chunk size.
`--smoke-test` selects the existing toy model instead of a model YAML.

Load a complete training recipe with `--training-config`; explicit CLI values
override its fields. Training YAMLs use the shared config loader. Runtime
YAMLs remain reserved for later runtime work; select the device with `--device`.
Use `--overfit-batch` to repeat the first training batch. Quantile Balancing
updates only in training mode; validation preserves the training RNG state.

### Run identity and provenance

Every CLI invocation records its effective settings in `run.json` before
training starts. Fresh runs default to unique folders under
`results/training/runs/`, with `last.pt` as their checkpoint. Use `--run-dir`
to choose a new folder; existing run folders are only accepted for resume.
An explicit `--checkpoint` still selects an alternative output path.

The manifest records model/training configurations, seed, actual AdamW group
settings, parameter count, Git commit and dirty status, Python/NumPy/PyTorch
and CUDA/cuDNN versions, hardware and execution settings, and data manifest
hashes plus dataset/tokenizer revisions when present in the data metadata.
Git fields are `null` when unavailable; a dirty flag does not preserve the
uncommitted diff. Archive or commit local code changes for reproducible runs.

```bash
uv run python scripts/train.py --data-dir /path/to/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml --device cpu \
  --run-dir results/training/my_smoke

uv run python scripts/train.py --data-dir /path/to/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml --device cpu --max-steps 40 \
  --resume results/training/my_smoke/last.pt
```

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
This is CLI bookkeeping: checkpoint format 4 remains unchanged. The direct
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
trainer, without retaining tensors. MoE diagnostics are described below;
optional profiling is the following monitoring step.

```bash
uv run python -m pytest -q tests/unit/training/test_metrics.py \
  tests/integration/test_training_metrics.py \
  tests/integration/test_train_cli.py tests/integration/test_checkpoint_resume.py
```

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
| `session.train_seconds`, `eval_seconds`, `checkpoint_seconds`, `reporting_seconds` | Cumulative phase costs within this session; reporting covers diagnostic snapshots, memory sampling, console printing and JSONL writes |
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

```bash
uv run python -m pytest -q tests/unit/training \
  tests/integration/test_training_metrics.py \
  tests/integration/test_train_cli.py tests/integration/test_checkpoint_resume.py
```

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
microbatches; this initial monitor does not collect per-token routes or expert
execution times.

```bash
uv run python -m pytest -q tests/unit/training \
  tests/unit/moe/test_stable_latent_moe.py \
  tests/integration/test_training_metrics.py \
  tests/integration/test_train_cli.py tests/integration/test_checkpoint_resume.py
```

### Initial training recipe

The presets use AdamW with peak LR `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`
and global L2 gradient clipping at `1.0`. Weight decay applies to matrices;
1D norms, biases and gate parameters are excluded. Tied embeddings appear
once in the optimizer. These are initial test settings, not tuned Kimi K3
pretraining hyperparameters.
For direct Python calls to `train`, a fresh optimizer's initial LR supplies
the schedule's peak. The CLI constructs that optimizer from the recipe.

An optimizer step averages `gradient_accumulation_steps` equal-size
microbatches. Nominal tokens per update are `batch_size * seq_len *
gradient_accumulation_steps`. Logging, evaluation, checkpoints, warmup and
`max_steps` count successful optimizer updates. QB updates on every training
forward, so MoE accumulation is not mathematically identical to one
concatenated batch. Overfit mode repeats the same microbatch, including within
an accumulated update.

| Recipe | Updates | Microbatch × sequence | Accumulation | Nominal processed tokens | LR / decay |
| --- | ---: | --- | ---: | ---: | --- |
| `smoke.yaml` | 20 | 1 × 128 | 1 | 2,560 | Constant, no weight decay |
| `overfit.yaml` | 2,000 | 1 × 128 | 1 | 256,000 repeated | Constant, no weight decay |
| `stability.yaml` | 1,000 | 1 × 128 | 4 | 512,000 | 100-step warmup, cosine to 10% of peak; weight decay 0.1 |

Run smoke first to check finite losses/gradients and checkpoint I/O. Overfit
has a 2,000-update budget; inspect whether fixed-batch train loss reaches
`< 0.1` (this is not a generalization test). Stability traverses successive
batches for a full warmup/decay cycle; inspect finite losses and gradients
and unexpected clipping or FP16 skips. Reaching the update budget alone is
not proof of convergence. These are bounded local tests, not a 6B-token run.
Estimate training seconds as nominal tokens divided by measured steady-state
tokens/s, then add evaluation/checkpoint time; their wall-clock duration is
not known before measuring the chosen model and hardware.

All presets start in FP32 as a numerical reference. `--precision bf16` uses
autocast on CPU or CUDA with native BF16 support; `--precision fp16` uses CUDA
autocast and GradScaler. Parameters, gradients and AdamW moments stay FP32.
The KDA solve/recurrent core and MoE router/QB remain FP32 under autocast.
Training and validation use the same precision. Follow PyTorch's
[AMP order](https://docs.pytorch.org/docs/stable/notes/amp_examples.html):
average microbatch losses, accumulate scaled gradients, unscale once, clip,
then step/update the scaler. Non-finite FP32/BF16 gradients fail immediately.
An FP16 gradient overflow reduces the scale and skips AdamW and LR progression;
consumed tokens and QB updates remain, and are checkpointed. The loop fails
after 32 consecutive skips. `tokens_seen` includes skipped/repeated batches;
`grad_norm` is the unscaled global norm before clipping.

```bash
uv run python scripts/train.py --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml --device cuda \
  --checkpoint results/training/recipe_smoke.pt
```

Change the training YAML for the next test and give each run its own checkpoint.
For precision comparisons, start each run from the same seed and choose FP32,
BF16 or FP16 explicitly. `--max-grad-norm 0` disables clipping. The bare CLI
keeps the original constant-LR, FP32, accumulation-one, clipping-disabled
defaults for checkpoint compatibility; the preset YAMLs enable the recipe above.

### Checkpoints and resume

New checkpoints use trainer format version 4 and are written atomically after a
completed optimizer update, with no pending gradients. They contain model
parameters and persistent buffers (including Quantile Balancing biases), AdamW
state and hyperparameters, step/token/skip counters, GradScaler state for FP16,
the training recipe, both data-loader cursors, and Python, NumPy, PyTorch CPU
and active-device CUDA RNG states. In overfit mode,
the fixed batch's starting token offset is saved so resume rereads those same
examples without advancing the restored cursor.

For a deterministic comparison, use `--deterministic` in both the initial and
resumed runs. Resume restores RNG after rebuilding the model and optimizer;
`--seed` does not replace the saved state. `--max-steps` is the total target
number of successful updates, including the saved steps. Saved optimizer
hyperparameters override the CLI learning rate, weight decay, betas and epsilon.
The parameter-group layout must match: this builder uses one group for zero
weight decay and separate matrix/1D groups for nonzero decay.
The LR schedule is derived from the restored step and saved base LR; it has
no separate mutable scheduler object. `decay_steps` fixes its horizon and must
match on resume, even when a first segment uses a smaller `max_steps`.

```bash
uv run python scripts/train.py --data-dir /path/to/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml --device cpu \
  --batch-size 1 --seq-len 128 --max-steps 100 --eval-batches 1 \
  --deterministic --checkpoint results/training/resume_tiny.pt

uv run python scripts/train.py --data-dir /path/to/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml --device cpu \
  --batch-size 1 --seq-len 128 --max-steps 200 --eval-batches 1 \
  --deterministic --checkpoint results/training/resume_tiny.pt \
  --resume results/training/resume_tiny.pt
```

Resume checks model configuration, batch dimensions, accumulation, precision,
overfit mode, clipping, LR schedule, optimizer groups, data manifests and
execution settings before loading state. Resume a preset run with the same
`--training-config` and matching recipe overrides.
Prepared shards must remain unchanged: manifest identity includes their
recorded checksums, but startup does not hash entire shard files. The data
directory can move if its manifest and files are identical. Format version 2
checkpoints lack cursors/RNG and cannot provide this complete resume; keep
existing checkpoints separate when testing the new format.
Version 3 still resumes with its original FP32, accumulation-one, constant-LR,
no-warmup, clipping-disabled recipe and single AdamW group (use the bare CLI
without a new training YAML or weight-decay override). This preserves its
stored optimizer settings; it is not converted silently into a new recipe.

Exact equivalence requires the same software, hardware and execution settings
with deterministic kernels; PyTorch does not promise reproducibility across
[versions or platforms](https://docs.pytorch.org/docs/stable/notes/randomness.html).
The current trainer has no distributed sampler, persistent recurrent cache
or checkpoints inside a partially accumulated update.

```bash
uv run pytest -q tests/unit/training/test_recipe.py tests/integration/test_checkpoint_resume.py
```

These tests compare continuous training with a saved/reconstructed/resumed
run: losses, all model tensors, AdamW state, counters, cursors and subsequent
random draws. They cover real mixed KDA/MLA/MoE layers, a fixed batch starting
at a nonzero offset, shard boundaries and exhaustion, stochastic training,
validation isolation, incompatible resumes, and fresh CLI processes. The
CUDA cases are skipped when CUDA is unavailable, and native BF16 CUDA cases
require a capable device. Recipe tests also check accumulated updates against
a larger stateless batch, clipping after unscaling, warmup/decay boundaries,
FP16 overflow skips, AMP FP32 regions and version-3 compatibility.
