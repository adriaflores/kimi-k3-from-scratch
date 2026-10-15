# Training recipe

This page records the validated recipe for the scaled ~300M-parameter Kimi K3 pretraining run. It is **configuration documentation**, not a report of final training quality.

## Full-run configuration

The approved full-run YAML is [`experiments/m7/m75_gate/recipe/m8_full_6b.yaml`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/experiments/m7/m75_gate/recipe/m8_full_6b.yaml). The run continues from the verified 80M-token pilot checkpoint rather than restarting the weights or optimizer state.

| Setting | Value |
|---|---:|
| Microbatch size | 32 |
| Sequence length | 1,024 |
| Gradient accumulation | 1 |
| Tokens per successful update | 32,768 |
| Update horizon | 183,106 |
| Peak learning rate | 0.003 |
| Schedule | Cosine |
| Warmup steps | 1,832 |
| Minimum LR / peak LR | 0.1 |
| Weight decay | 0.1 |
| Gradient norm clipping | 1.0 |
| Precision | BF16 autocast with FP32 model parameters |
| Seed | 1337 |

The target is approximately six billion training tokens. Do not mistake the model's maximum configured context of 2,048 for the **1,024-token training context**.

## Optimizers and numerical choices

The trainer partitions trainable parameters between **Muon** for eligible hidden matrices and **AdamW** for auxiliary parameters. The implementation includes per-head Muon handling for compatible attention projections, five Newton–Schulz iterations, momentum `0.95`, batched/compiled Muon updates, and fused AdamW.

QK-Clip uses a threshold of `100` for relevant MLA attention projections. The global unscaled gradient norm is clipped at `1.0`; Quantile Balancing updates routing biases during training and is frozen during validation.

See [`src/kimi_k3/training/optimizers.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/training/optimizers.py), [`parameter_groups.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/training/parameter_groups.py), and [`qk_clip.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/training/qk_clip.py).

## Runtime and monitoring

The qualified runtime uses a single H100 NVL GPU through Vast.ai and the optimized FLA KDA path. Logs record loss, learning rate, gradient norm, throughput, memory use, expert loads, and Quantile Balancing diagnostics.

The full-run YAML specifies logging every 64 updates, validation every 512 updates (8 batches), and checkpointing every 4,096 updates. Separate operational scripts can impose additional backup cadence and verification.

The training entry point is [`scripts/train.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/scripts/train.py); the core trainer resides in [`src/kimi_k3/training/trainer.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/training/trainer.py).

## Evidence boundary

The 80M-token pilot was reviewed for learning, finite gradients, and MoE routing behavior before authorizing full continuation. Its validation measurements covered a repeated validation prefix, so the pilot should not be mistaken for a broad downstream evaluation. See the [M7.4 scientific decision](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/experiments/m7/m75_gate/docs/M7_4_scientific_decision.md).
