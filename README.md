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

Training options currently come from the CLI. The training/runtime YAMLs are
not loaded yet; the current loop uses FP32 with one batch per optimizer update.
Use `--overfit-batch` to repeat the first training batch. Quantile Balancing
updates only in training mode; validation preserves the training RNG state.

### Checkpoints and resume

Checkpoints use trainer format version 3 and are written atomically after a
completed optimizer update, with no pending gradients. They contain model
parameters and persistent buffers (including Quantile Balancing biases), AdamW
state and hyperparameters, step/token counters, both data-loader cursors, and
Python, NumPy, PyTorch CPU and active-device CUDA RNG states. In overfit mode,
the fixed batch's starting token offset is saved so resume rereads those same
examples without advancing the restored cursor.

For a deterministic comparison, use `--deterministic` in both the initial and
resumed runs. Resume restores RNG after rebuilding the model and optimizer;
`--seed` does not replace the saved state. `--max-steps` is the total target
number of updates, including the saved steps. Saved optimizer hyperparameters
override the CLI learning rate and weight decay.

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

Resume checks the model configuration, batch dimensions, overfit mode,
optimizer type, data manifests and execution settings before loading state.
Prepared shards must remain unchanged: manifest identity includes their
recorded checksums, but startup does not hash entire shard files. The data
directory can move if its manifest and files are identical. Format version 2
checkpoints lack cursors/RNG and cannot provide this complete resume; keep
existing checkpoints separate when testing the new format.

Exact equivalence requires the same software, hardware and execution settings
with deterministic kernels; PyTorch does not promise reproducibility across
[versions or platforms](https://docs.pytorch.org/docs/stable/notes/randomness.html).
The current trainer has no scheduler, AMP scaler, gradient accumulation,
distributed sampler or persistent recurrent cache. Add their states to the
checkpoint if those features are introduced.

```bash
uv run pytest -q tests/integration/test_checkpoint_resume.py
```

These tests compare continuous training with a saved/reconstructed/resumed
run: losses, all model tensors, AdamW state, counters, cursors and subsequent
random draws. They cover real mixed KDA/MLA/MoE layers, a fixed batch starting
at a nonzero offset, shard boundaries and exhaustion, stochastic training,
validation isolation, incompatible resumes, and fresh CLI processes. The
CUDA stochastic test is skipped when CUDA is unavailable.
