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
fits the context limit, and that KDA sequences are divisible by the chunk size
when the chunkwise KDA is selected.
`--smoke-test` selects the existing toy model instead of a model YAML.

`kda_implementation` in the model YAML selects the KDA kernel. `fused`, used by
the 300M config, runs FLA's Triton `chunk_kda` and needs CUDA, `kda_chunk_size`
32 or 64 and `kda_gate_lower_bound` in [-5, 0); it accepts any sequence length.
`chunkwise`, used by the tiny config, is the readable PyTorch reference and runs
on any device. The first fused run compiles and autotunes Triton kernels, which
can take minutes; later runs reuse the cache. `chunk_size: 32` currently fails
with a misaligned-address error on `triton-windows`, so keep 64 there.

Load a complete training recipe with `--training-config`; explicit CLI values
override its fields. Training YAMLs use the shared config loader. Runtime
YAMLs remain reserved for later runtime work; select the device with `--device`.
Use `--overfit-batch` to repeat the first training batch. Quantile Balancing
updates only in training mode; validation preserves the training RNG state.

See the [tiny training circuit](docs/notes/tiny_training_check.md) for the
pre-GPU next-token check, validation isolation, and checkpoint/resume acceptance.

### Monitoring and profiling

Every CLI run writes `run.json`, `metrics.jsonl` and, by default, `last.pt`
in a unique folder under `results/training/runs/`. Use `--run-dir` to choose
a fresh folder. Resume appends a new session and metrics to the original run.

The monitor records train/validation loss, LR, gradient norms, training and
wall throughput, RAM/CUDA memory, routed expert loads and QB bias snapshots.
The manifest preserves effective configurations, seed, Git commit/dirty status,
software/device information and data identity.

Add `--profile` for one bounded time/memory capture, with phase and
attention/FFN family labels. Defaults are one wait, one warmup and one active
optimizer attempt. Captures live in `RUN/profiles/SESSION_ID/`, with a
compressed timeline, operator tables and capture metadata.

See the [training monitoring guide](docs/notes/training_monitoring.md) for the
baseline → capture → comparison workflow, resume, metric definitions,
interpretation limits and the complete verification command. Measure final
throughput in an unprofiled run with the same hardware and recipe.

### Training recipe

The trainer uses one fixed hybrid optimizer: **Muon for hidden matrices and
native AdamW for auxiliary parameters**. Embeddings and the vocabulary head,
norm gains, biases, scalar/vector gates, AttnRes pseudo-query vectors and KDA's
depthwise short-convolution filters use AdamW. Tied weights appear once. Norms,
biases and logical vectors have no weight decay; matrices and convolution
filters use the configured decay. Fused FFN gate/up storage is viewed as two
independent matrices, each with its own Newton–Schulz normalization and scaling.
Attention Q/K/V projections also use independent head blocks: KDA's fused
`QKV`, MLA's `W_UQ`, and the separate K and V rows of each head in `W_UKV`.
MLA supports unequal key/value dimensions. Its shared latent down-projections,
attention output projections and matrix gates retain whole-matrix Muon.

Muon uses Nesterov momentum `0.95`, five Newton–Schulz iterations in BF16, and
Moonlight's `0.2 * sqrt(max(rows, columns))` update scale. Both algorithms share
the same peak LR and schedule. The presets keep peak LR `3e-4`, auxiliary AdamW
betas `(0.9, 0.95)`, epsilon `1e-8`, and global gradient clipping at `1.0`.
`adam_fused: true` selects native fused AdamW; `--no-adam-fused` changes its
execution backend. `muon_momentum`, `muon_ns_steps` and `muon_ns_dtype` also have
CLI overrides; FP32/FP64 NS arithmetic is available for numerical checks.
These are local test settings, not tuned Kimi K3 pretraining hyperparameters.

Per-head Muon and QK weight clipping are active in this recipe. QK-Clip observes
MLA's scaled causal attention scores across all microbatches, then clips the
corresponding Q/K projection rows after each successful optimizer update.
The default threshold is `100`; KDA's normalized Q/K do not use this clipping.
See [the integrated recipe and head geometry](docs/notes/muon_recipe.md).

`muon_compile: false` retains the eager Newton–Schulz reference. Add
`--muon-compile` to compile just that core on CUDA with TorchInductor; use
`--no-muon-compile` for the comparison run. CPU optimizers remain eager.
Compilation is lazy, cached per matrix layout, and preserves the five iterations,
independent normalization and update scaling. Initial compilation can take
minutes. See [the optimizer benchmark and 4050 comparison](docs/notes/muon_compile.md).

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
| `acceptance.yaml` | 64 | 1 × 128 | 2 | 16,384 repeated | 8-step warmup, cosine to 10%; weight decay 0.1 |
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
autocast and GradScaler. Parameters, gradients, Muon momentum buffers and
AdamW moments stay FP32. NS working precision is independent of autocast.
The chunkwise KDA solve/recurrent core and MoE router/QB remain FP32 under
autocast; the fused KDA follows FLA's kernel precision.
Training and validation use the same precision. Follow PyTorch's
[AMP order](https://docs.pytorch.org/docs/stable/notes/amp_examples.html):
average microbatch losses, accumulate scaled gradients, unscale once, clip,
then step/update the scaler. Non-finite FP32/BF16 gradients fail immediately.
An FP16 gradient overflow reduces the scale and skips both algorithms and LR progression;
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
defaults; it uses the same hybrid optimizer. The preset YAMLs enable the recipe above.

### Checkpoints and resume

New checkpoints use trainer format version 7 and are written atomically after a
completed optimizer update, with no pending gradients. They contain model
parameters and persistent buffers (including Quantile Balancing biases), Muon
momentum and native AdamW state and hyperparameters, named parameter layouts
with head geometry and K/V row splits, step/token/skip counters, GradScaler state for FP16,
the training recipe, both data-loader cursors, and Python, NumPy, PyTorch CPU
and active-device CUDA RNG states. In overfit mode,
the fixed batch's starting token offset is saved so resume rereads those same
examples without advancing the restored cursor.

For a deterministic comparison, use `--deterministic` in both the initial and
resumed runs. Resume restores RNG after rebuilding the model and optimizer;
`--seed` does not replace the saved state. `--max-steps` is the total target
number of successful updates, including the saved steps. Saved optimizer
hyperparameters override CLI learning rate, weight decay, betas, epsilon, Muon
momentum and NS iteration count. The parameter-group layout must match, including
canonical names, shapes, dtypes and declared matrix views. Group separation stays
stable when weight decay is zero. `adam_fused`, `muon_ns_dtype` and
`muon_compile` must match. Format-7 checkpoints from before compilation default
to the eager mode and still resume with `--no-muon-compile`. Compiled functions
are rebuilt lazily after resume and are not serialized.
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
directory can move if its manifest and files are identical. Legacy AdamW trainer
checkpoints (versions 3–5) cannot resume this Muon recipe; start a new run.
Pre-clipping Muon checkpoints (version 6) cannot resume the integrated recipe.
Earlier checkpoints also lack the required state. The loader rejects them before
changing model weights, optimizer state, data cursors or RNG.

Exact equivalence requires the same software, hardware and execution settings
with deterministic kernels; PyTorch does not promise reproducibility across
[versions or platforms](https://docs.pytorch.org/docs/stable/notes/randomness.html).
The current trainer has no distributed sampler, persistent recurrent cache
or checkpoints inside a partially accumulated update.

```bash
uv run pytest -q tests/unit/training/test_recipe.py tests/integration/test_checkpoint_resume.py
```

These tests compare continuous training with a saved/reconstructed/resumed
run: losses, all model tensors, Muon and AdamW state, counters, cursors and subsequent
random draws. They cover real mixed KDA/MLA/MoE layers, a fixed batch starting
at a nonzero offset, shard boundaries and exhaustion, stochastic training,
validation isolation, incompatible resumes, and fresh CLI processes. The
CUDA cases are skipped when CUDA is unavailable, and native BF16 CUDA cases
require a capable device. Recipe tests also check accumulated updates against
a larger stateless batch, clipping after unscaling, warmup/decay boundaries,
FP16 overflow skips, AMP FP32 regions and rejection of legacy checkpoints.
