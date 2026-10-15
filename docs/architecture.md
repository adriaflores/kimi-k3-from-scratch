# Architecture

A concise map of the language backbone implemented by this repository. For mathematical background, refer to the cited research papers; this page documents **this implementation**, not every feature of the original full-scale system.

## Model configuration

The principal scaled configuration lives at [`configs/model/kimi_k3_300m.yaml`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/configs/model/kimi_k3_300m.yaml).

| Setting | Scaled configuration |
|---|---:|
| Vocabulary | 49,152 |
| Hidden width | 640 |
| Decoder layers | 13 |
| Total parameters | ~294.7M |
| Active parameters per token | ~100.4M (approximate) |
| Attention heads | 8 |
| Sequence length supported by config | 2,048 |
| Dense feed-forward layers | 1 |
| Routed experts per MoE layer | 64 |
| Routed experts selected per token | 2 |
| Shared experts | 2 |
| Attention Residual block size | 2 |

The **training sequence length** is 1,024, which is distinct from the 2,048 sequence length in the model configuration. Parameter counts are for this scaled configuration, not the original full-scale Kimi K3.

## Decoder layout

The attention type is selected per layer:

```text
layer  1: KDA     layer  2: KDA     layer  3: KDA     layer  4: MLA
layer  5: KDA     layer  6: KDA     layer  7: KDA     layer  8: MLA
layer  9: KDA     layer 10: KDA     layer 11: KDA     layer 12: MLA
layer 13: MLA
```

The first layer uses a dense feed-forward network; the remaining twelve use Stable LatentMoE. Residual mixing uses Attention Residuals with a block size of two. The embedding and output vocabulary weights are tied.

## Kimi Delta Attention

- [`src/kimi_k3/attention/kda/recurrent.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/attention/kda/recurrent.py) is the direct recurrence reference.
- [`src/kimi_k3/attention/kda/chunkwise.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/attention/kda/chunkwise.py) is the readable PyTorch chunkwise implementation.
- [`src/kimi_k3/attention/kda/fused_chunk.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/attention/kda/fused_chunk.py) wraps the optimized **FLA Triton `chunk_kda` kernel** used on CUDA for the scaled model.

All three include layer-level projections, causal depthwise short convolutions, decay and write gates, output normalization, and an output gate. The GPU fused kernel is an external optimized dependency; its internals are **not claimed as independently implemented** here.

Relevant tests: [`test_kda_chunkwise_vs_recurrent.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/tests/equivalence/test_kda_chunkwise_vs_recurrent.py) and [`test_kda_fused_vs_chunkwise.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/tests/equivalence/test_kda_fused_vs_chunkwise.py). CUDA tests require compatible GPU and FLA installation.

## Gated MLA, Attention Residuals and feed-forward layers

- **Gated MLA:** [`attention/gated_mla.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/attention/gated_mla.py), with low-rank query and KV projections and an output gate.
- **Attention Residuals:** [`model.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/model.py) contains block-level residual aggregation and the top-level model composition.
- **Dense FFN / SiTU-GLU:** [`ffn/dense_ffn.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/ffn/dense_ffn.py).
- **Stable LatentMoE / Quantile Balancing:** [`moe/stable_latent_moe.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/moe/stable_latent_moe.py).

## Implementation boundaries

The project is focused on the language backbone rather than the vision stack. It uses the SmolLM2-135M tokenizer instead of claiming original-checkpoint tokenizer compatibility. The scaled geometry and training recipe are research choices and should not be confused with those of the original model.

**Primary reference:** [Kimi K3 paper](https://arxiv.org/abs/2607.24653v2). See also [Kimi Linear](https://arxiv.org/abs/2510.26692v2) for the KDA architecture.
