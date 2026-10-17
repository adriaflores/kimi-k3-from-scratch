# Reproducibility

The project distinguishes **re-running an experiment** from **resuming one without losing state**. A checkpoint of model weights alone is not sufficient to establish exact continuation of a training process.

## Software and data provenance

The repository contains model and training YAMLs, a `uv.lock`, prepared data manifests, run metadata, and test suites. Runtime manifests record model/training settings, seed, software/hardware context, and source revision identity.

Start with the repository’s [`README.md`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/README.md) for installation and basic commands. The canonical configuration of the full run is in [`experiments/pretraining/training.yaml`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/experiments/pretraining/training.yaml).

## Checkpoint contents and resume contract

Current trainer-format checkpoints include:

- Model weights and persistent buffers, including Quantile Balancing biases.
- Muon and AdamW optimizer states, parameter-group geometry, and LR schedule counters.
- Completed update number, tokens seen, and skipped-update counters.
- Training and validation data-loader cursors.
- CPU and active-device CUDA random-number-generator states (plus Python/NumPy).
- Effective training recipe and compatibility metadata.

The checkpoint is written at a completed optimizer update, not partway through a gradient-accumulation cycle. Resume validates configurations and data identity and reconstructs the intended continuation. Cross-hardware or cross-version bitwise reproducibility is not generally guaranteed by PyTorch.

## Test surfaces

| Behavior | Relevant tests |
|---|---|
| Recurrent/chunkwise KDA agreement | `tests/equivalence/test_kda_chunkwise_vs_recurrent.py` |
| Fused/reference KDA agreement | `tests/equivalence/test_kda_fused_vs_chunkwise.py` |
| Model integration | `tests/integration/test_kimi_k3.py` |
| Checkpoint and RNG continuation | `tests/integration/test_checkpoint_resume.py` |
| Training CLI and diagnostics | `tests/integration/test_train_cli.py`, `tests/integration/test_training_metrics.py` |

These tests differ in hardware requirements. GPU kernel tests should be run on an appropriately configured CUDA environment; their existence in the repository does not by itself prove a particular machine passes them.

## Operational artifacts

The public tree retains the pilot scientific assessment and the original pilot/full-run recipes under `experiments/pilot/` and `experiments/pretraining/`. Frozen evaluation artifacts and one-off infrastructure are recoverable from Git history rather than distributed in the current tree. The original model checkpoints and prepared token shards are not included.

Before claiming final reproduction, record the final checkpoint's SHA-256, exact dataset manifest hash, source commit, package environment, and evaluation procedure. These become part of the final results release.
