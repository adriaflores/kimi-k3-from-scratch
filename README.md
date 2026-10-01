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
Checkpoint resume restores weights and optimizer state, but restarts the data
stream and does not restore RNG state. Quantile Balancing still updates during
evaluation; freezing it is a separate pending change.
