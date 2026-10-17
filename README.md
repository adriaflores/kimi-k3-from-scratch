# Kimi K3 from Scratch

An independent PyTorch implementation of the **Kimi K3 language backbone**, built to study how its architecture and training recipe work in practice. The project scales the design down to **294.7M total parameters** (~100.4M active per token) and follows a **~6B-token pretraining run** on FineWeb-Edu.

**[Architecture, training and results — companion website](https://kimi-k3.adriaflores.com)** · **[Research references](references/README.md)** · **[Kimi K3 paper](https://arxiv.org/abs/2607.24653v2)**

This repository is the **research engineering artifact**: model code, training infrastructure, configurations, experiments and tests. The companion website hosts the longer explanations and visualizations.

## At a glance

| | Scaled implementation |
| --- | --- |
| Model | 13-layer hybrid KDA / Gated MLA decoder |
| Parameters | ~294.7M total; ~100.4M active per token |
| Attention | 9 Kimi Delta Attention layers + 4 Gated MLA layers |
| Feed-forward | 1 dense layer + 12 Stable LatentMoE layers |
| MoE routing | 64 routed experts, top-2 per token, plus 2 shared experts |
| Residual connections | Block Attention Residuals (block size 2) |
| Vocabulary | 49,152; tied input embeddings and LM head |
| Data | FineWeb-Edu `sample-10BT`, SmolLM2-135M tokenizer |
| Training context | 1,024 tokens (model configuration supports up to 2,048) |
| Optimizer | Muon for eligible matrices + AdamW for auxiliary parameters |

Counts refer to the **reduced-scale model**, not to the original Kimi K3 checkpoint. You can inspect the exact parameter accounting from the configuration:

```bash
uv run python scripts/inspect_config.py configs/model/kimi_k3_300m.yaml
```

## What is implemented

| Component | Implementation | Verification |
| --- | --- | --- |
| Kimi Delta Attention | [Recurrent](src/kimi_k3/attention/kda/recurrent.py), [chunkwise PyTorch](src/kimi_k3/attention/kda/chunkwise.py), and [FLA-backed fused](src/kimi_k3/attention/kda/fused_chunk.py) | [Recurrent/chunkwise equivalence](tests/equivalence/test_kda_chunkwise_vs_recurrent.py), [fused/chunkwise tests](tests/equivalence/test_kda_fused_vs_chunkwise.py) |
| Gated Multi-head Latent Attention | [Gated MLA](src/kimi_k3/attention/gated_mla.py) | [Unit tests](tests/unit/attention/test_gated_mla.py) |
| Block Attention Residuals and decoder | [Model and decoder blocks](src/kimi_k3/model.py) | [Model integration](tests/integration/test_kimi_k3.py) |
| SiTU-GLU and dense FFN | [Dense FFN](src/kimi_k3/ffn/dense_ffn.py) | [Unit tests](tests/unit/ffn/test_dense_ffn.py) |
| Stable LatentMoE and Quantile Balancing | [MoE and routing](src/kimi_k3/moe/stable_latent_moe.py) | [MoE unit tests](tests/unit/moe/test_stable_latent_moe.py), [grouped-expert equivalence](tests/equivalence/test_moe_grouped_vs_naive.py) |
| Muon, AdamW and QK-Clip | [Optimizer](src/kimi_k3/training/optimizers.py), [parameter grouping](src/kimi_k3/training/parameter_groups.py), [QK-Clip](src/kimi_k3/training/qk_clip.py) | [Optimizer tests](tests/unit/training/test_optimizers.py), [checkpoint/resume](tests/integration/test_checkpoint_resume.py) |

The recurrent and chunkwise KDA paths are readable PyTorch implementations. The high-throughput **fused CUDA path uses the upstream Flash-Linear-Attention (FLA) kernel**; this repository does not claim to implement that Triton kernel from scratch. The model is a language-only, reduced-scale research implementation, not an official Moonshot AI release.

## Getting started

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/). GPU training uses a supported NVIDIA CUDA environment; the project specifies PyTorch CUDA 12.8 wheels and makes FLA available through its `gpu` extra.

```bash
git clone https://github.com/adriaflores/kimi-k3-from-scratch.git
cd kimi-k3-from-scratch
uv sync                  # Base environment
uv sync --extra gpu      # Optional: CUDA/FLA training backend
```

For a bounded **CPU training smoke test**, first prepare compatible token shards (see below), then run the tiny reference model:

```bash
uv run python scripts/train.py \
  --data-dir /path/to/prepared-data \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml \
  --device cpu
```

This smoke test checks the training circuit; it is **not** a miniature reproduction of the full-run results. For available options, use `uv run python scripts/train.py --help`.

## Data and pretraining

Data preparation uses [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) and the [SmolLM2-135M tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Documents are deterministically assigned to train/validation, tokenized without automatic special tokens, separated with EOS, and stored as contiguous NumPy token shards with a provenance manifest. **The dataset is not distributed in this repository.**

```bash
uv run python scripts/prepare_data.py \
  --output-dir /path/to/prepared-data \
  --train-tokens 6000000000
```

This is a large data-preparation job, not a quick-start command. It downloads source data and stores generated token shards. The manifest records the resolved dataset/tokenizer revisions and the data identity required for checkpoint resume.

The recorded training recipe is split into an ~80M-token pilot and continuation to the ~6B-token budget:

- [Pilot recipe](experiments/pilot/training.yaml)
- [Full-run continuation recipe](experiments/pretraining/training.yaml)
- [Pilot scientific assessment](experiments/pilot/scientific_assessment.md)

The scaled run uses BF16, an effective batch of **32 × 1,024 = 32,768 tokens/update**, a **3e-3** peak learning rate, cosine decay, Muon + AdamW, QK-Clip and Quantile Balancing. The full schedule ends at **183,106 optimizer updates**. The continuation recipe expects the original pilot model **and optimizer/training state**, rather than a weights-only restart.

```bash
# Example: continue from a compatible pilot checkpoint on CUDA.
uv run --extra gpu python scripts/train.py \
  --data-dir /path/to/prepared-data \
  --model-config configs/model/kimi_k3_300m.yaml \
  --training-config experiments/pretraining/training.yaml \
  --resume /path/to/pilot/last.pt \
  --device cuda
```

Checkpoints contain model and optimizer states, training counters, data cursors, and RNG state. Runs write `run.json`, `metrics.jsonl`, and checkpoints; profiling is available through `--profile`. See the [trainer](src/kimi_k3/training/trainer.py), [run metadata](src/kimi_k3/training/run.py), and [monitoring](src/kimi_k3/training/monitoring.py) implementations.

## Inference and evaluation

The public CLI tools accept compatible **trainer-format v7 checkpoints**. Model weights are **not included** in this repository.

```bash
# Generate from a local checkpoint.
uv run python scripts/generate.py \
  --checkpoint /path/to/last.pt \
  --prompt "Once upon a time, in a small village" \
  --device auto --backend chunkwise --max-new-tokens 48

# Exploratory next-token NLL, perplexity and top-1 accuracy.
uv run python scripts/evaluate.py \
  --checkpoint /path/to/last.pt \
  --data-dir /path/to/prepared-data \
  --split val --seq-len 1024 --num-windows 4 \
  --device auto --backend chunkwise
```

`evaluate.py` scores configurable windows and **does not reproduce the original pre-registered held-out evaluation**. Do not compare its numbers with results obtained using different offsets, data exclusions, precision, or attention backends. For interactive experimentation, see the [generation playground](notebooks/generation_playground.ipynb).

Generation currently recomputes the prompt without KV/recurrent-state caching; it is intended for correctness and inspection rather than fast serving.

## Validation and evidence

```bash
uv run python -m pytest -q tests
```

The suite covers component behavior, recurrent/chunkwise/fused equivalence, MoE execution, training integration, optimizer geometry, checkpoint recovery, inference causality and CLI behavior. CUDA-specific checks need compatible hardware and dependencies; some tests are skipped when those are absent.

The [pilot assessment](experiments/pilot/scientific_assessment.md) documents a validation-loss reduction from **7.8979 to 4.1003** over 80,019,456 processed tokens. Importantly, those pilot evaluations reused the **same 262,144-token validation prefix**: they demonstrate a training trend, **not** independent held-out generalization. The repository does not bundle original model weights or the complete historical evaluation outputs. No final-model benchmark comparison is claimed here.

## Repository layout

```text
configs/       Model configurations and bounded training presets
experiments/   Pilot assessment and pretraining recipes
notebooks/     Interactive generation and evaluation playground
references/    Paper-to-code reading map and required source notices
scripts/       Data preparation, training, inspection, inference and evaluation
src/kimi_k3/   Model, attention, MoE, data and training implementation
tests/        Unit, integration, equivalence and smoke tests
```

## References, attribution and license

The [research reference index](references/README.md) links the papers, upstream software, implementation files and relevant tests. The Muon Newton–Schulz kernel and momentum convention adapt MIT-licensed source material; see the preserved [source attributions and licenses](references/muon/NOTICE.md). Other dependencies, including PyTorch and Flash-Linear-Attention, retain their respective upstream licenses.

This project is licensed under [Apache License 2.0](LICENSE). It is an independent research and educational implementation and is **not affiliated with or endorsed by Moonshot AI**.
