# Research references

This index connects the research behind **Kimi K3 from Scratch** to the code,
tests, configurations, and data choices in this repository. It is a reading map,
not a claim that every cited method has been implemented here.

**Legend:** **Core** = directly defines a mechanism or recipe used in this
project; **Supporting** = informs a related implementation decision;
**Background** = historical or comparative context; **Software/data** = upstream
resource used or consulted. The scaled model is an independent, reduced-size
implementation of the **language backbone**, not a reproduction of the original
2.8T-parameter Kimi K3, its vision stack, or its complete production systems.

## Kimi model family and primary architecture

| Reference | Role in this project | Relevance |
| --- | --- | --- |
| **[Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653v2)** — Kimi Team (2026) | **Core** | Primary specification for the hybrid language backbone, Block Attention Residuals, Stable LatentMoE, SiTU-GLU, Quantile Balancing, and optimizer recipe. |
| **[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)** — Kimi Team (2025) | **Core** | Kimi Delta Attention (KDA), channelwise decay, recurrent/chunkwise formulation, and hybrid KDA/MLA design. |
| **[Attention Residuals](https://arxiv.org/abs/2603.15031)** — Kimi Team (2026) | **Core** | Content-dependent residual retrieval across **network depth** and the blockwise approximation used in the model. |
| **[Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)** — Kimi Team (2025) | **Supporting** | MuonClip/QK-Clip training-stability precedent. Our QK-Clip is adapted to this model's MLA packing, not a verbatim implementation. |
| **[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)** — DeepSeek-AI (2024) | **Supporting** | Multi-head Latent Attention (MLA) and latent KV projection design. Kimi's gated/NoPE variant is governed by the Kimi papers, rather than reproduced from DeepSeek-V2. |
| **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** — DeepSeek-AI (2024) | **Supporting** | Auxiliary-loss-free load balancing and modern MoE training design; useful context for contrasting routing-bias methods with Quantile Balancing. |

**Code:** [`model.py`](../src/kimi_k3/model.py),
[`gated_mla.py`](../src/kimi_k3/attention/gated_mla.py),
[`stable_latent_moe.py`](../src/kimi_k3/moe/stable_latent_moe.py),
[`qk_clip.py`](../src/kimi_k3/training/qk_clip.py).
**Tests:** [`test_kimi_k3.py`](../tests/integration/test_kimi_k3.py),
[`test_gated_mla.py`](../tests/unit/attention/test_gated_mla.py),
[`test_stable_latent_moe.py`](../tests/unit/moe/test_stable_latent_moe.py).

## Linear attention, delta rules, and chunkwise algebra

| Reference | Role | Relevance |
| --- | --- | --- |
| **[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)** — Yang et al. (2024) | **Core** | Gating and delta-rule memory updates underlying KDA. |
| **[Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)** — Schlag et al. (2021) | **Supporting** | Fast-weight interpretation and delta corrections to key/value memory. |
| **[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)** — Katharopoulos et al. (2020) | **Supporting** | Equivalence of causal linear attention and recurrent state updates. |
| **[Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)** — Yang et al. (2023) | **Supporting** | Gated linear attention and efficient chunk-oriented execution. |
| **[The WY Representation for Products of Householder Matrices](https://doi.org/10.1137/0908009)** — Bischof & Van Loan (1987) | **Background** | WY factorization and matrix-product organization; mathematical context for compact transformations, **not** a claim that KDA performs Householder QR. |
| **[A Storage-Efficient WY Representation for Products of Householder Transformations](https://doi.org/10.1137/0910005)** — Schreiber & Van Loan (1989) | **Background** | Compact triangular-factor representation. Useful mathematical analogy when studying chunkwise triangular solves. |
| **[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)** — Sun et al. (2023) | **Background** | Parallel, recurrent, and chunkwise perspectives on sequence modeling. |
| **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)** — Gu & Dao (2023) | **Background** | Input-dependent recurrent state-space models; related design space, not a component of this implementation. |

**Code:** [`recurrent.py`](../src/kimi_k3/attention/kda/recurrent.py),
[`chunkwise.py`](../src/kimi_k3/attention/kda/chunkwise.py),
[`fused_chunk.py`](../src/kimi_k3/attention/kda/fused_chunk.py).
**Validation:** [`chunkwise vs recurrent`](../tests/equivalence/test_kda_chunkwise_vs_recurrent.py),
[`fused vs chunkwise`](../tests/equivalence/test_kda_fused_vs_chunkwise.py).

## Depth mixing, normalization, and feed-forward networks

| Reference | Role | Relevance |
| --- | --- | --- |
| **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — Vaswani et al. (2017) | **Background** | Transformer attention, residual connections, and layerwise architecture. |
| **[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)** — Zhang & Sennrich (2019) | **Core** | RMSNorm used throughout the backbone. |
| **[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)** — Shazeer (2020) | **Supporting** | Gated feed-forward structure; background for SiTU-GLU, whose actual definition follows Kimi K3. |
| **[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)** — Ramachandran et al. (2017) | **Background** | Swish/SiLU activation family; do not conflate the Swish paper with the Kimi-specific SiTU function. |
| **[Layer Normalization](https://arxiv.org/abs/1607.06450)** — Ba et al. (2016) | **Background** | Normalization lineage and contrast with RMSNorm. |

**Code:** [`model.py`](../src/kimi_k3/model.py),
[`dense_ffn.py`](../src/kimi_k3/ffn/dense_ffn.py),
[`rmsn.py`](../src/kimi_k3/utils/rmsn.py).
**Validation:** [`dense FFN tests`](../tests/unit/ffn/test_dense_ffn.py),
[`RMSNorm tests`](../tests/unit/test_rmsnorm.py).

## Sparse mixtures of experts and routing

| Reference | Role | Relevance |
| --- | --- | --- |
| **[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)** — Dai et al. (2024) | **Supporting** | Shared/routed experts and fine-grained expert specialization. |
| **[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)** — Shazeer et al. (2017) | **Background** | Sparse conditional computation and learned expert routing. |
| **[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)** — Lepikhin et al. (2020) | **Background** | Top-k expert routing and expert parallelism; this repository does **not** implement GShard distributed sharding. |
| **[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)** — Fedus et al. (2021) | **Background** | Routing, load balancing, and expert-training stability. |
| **[GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)** — Du et al. (2021) | **Background** | Sparse MoE scaling, activated-versus-total parameter budgets. |

**Project-specific methods:** Stable LatentMoE and Quantile Balancing are primarily
specified by **Kimi K3**, rather than attributed to the general MoE works above.
**Code:** [`stable_latent_moe.py`](../src/kimi_k3/moe/stable_latent_moe.py),
[`moe_metrics.py`](../src/kimi_k3/training/moe_metrics.py).
**Tests:** [`MoE unit tests`](../tests/unit/moe/test_stable_latent_moe.py),
[`grouped-vs-naive equivalence`](../tests/equivalence/test_moe_grouped_vs_naive.py).

## Optimization, training, and numerical stability

| Reference | Role | Relevance |
| --- | --- | --- |
| **[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)** — Liu et al. (2025) | **Core** | Muon scaling, parameter classes, decoupled weight decay, and Moonlight's update scale. |
| **[Muon reference implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py)** — Keller Jordan | **Code attribution** | Pinned source for the Newton–Schulz polynomial and momentum convention. See the retained [MIT notices](muon/NOTICE.md). |
| **[Moonlight training example](https://github.com/MoonshotAI/Moonlight/blob/c2ad5b20c605086526a179d36901bfc41b52b44b/examples/toy_train.py)** — Moonshot AI | **Code attribution** | Pinned source for the scaled Muon training convention. See the retained [MIT notices](muon/NOTICE.md). |
| **[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)** — Kingma & Ba (2014) | **Supporting** | Adaptive-moment optimizer foundations; auxiliary parameters use PyTorch AdamW. |
| **[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)** — Loshchilov & Hutter (2017) | **Supporting** | AdamW and independent weight-decay policy. |
| **[Mixed Precision Training](https://arxiv.org/abs/1710.03740)** — Micikevicius et al. (2017) | **Supporting** | FP32 master state and loss scaling considerations for reduced-precision training. |
| **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** — Hoffmann et al. (2022) | **Supporting** | Context for the approximate tokens-per-parameter budget; this scaled Kimi model is **not** a new compute-optimality study. |
| **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** — Kaplan et al. (2020) | **Background** | Model/data/compute scaling relationships. |

**Code:** [`optimizers.py`](../src/kimi_k3/training/optimizers.py),
[`parameter_groups.py`](../src/kimi_k3/training/parameter_groups.py),
[`qk_clip.py`](../src/kimi_k3/training/qk_clip.py),
[`trainer.py`](../src/kimi_k3/training/trainer.py).
**Tests:** [`optimizer tests`](../tests/unit/training/test_optimizers.py),
[`Muon head geometry`](../tests/unit/training/test_per_head_muon.py),
[`QK-Clip tests`](../tests/unit/training/test_qk_clip.py),
[`checkpoint/resume`](../tests/integration/test_checkpoint_resume.py).

## Pretraining data, tokenization, and reproducibility

| Reference | Role | Relevance |
| --- | --- | --- |
| **[The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)** — Penedo et al. (2024) | **Core** | Provenance and filtering rationale for FineWeb-Edu. |
| **[FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)** — Hugging Face | **Software/data** | The project tokenizes a bounded subset of `sample-10BT`, rather than reproducing FineWeb's data-curation pipeline. |
| **[SmolLM2-135M model and tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)** — Hugging Face | **Software/data** | Supplies the tokenizer and vocabulary; **not** a source for the model weights. |
| **[Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)** — Hugging Face | **Software/data** | Fast tokenizer implementation used during data preprocessing. |
| **[Hugging Face Datasets](https://github.com/huggingface/datasets)** — Hugging Face | **Software/data** | Streaming dataset access during data preparation. |

**Code:** [`prepare_data.py`](../scripts/prepare_data.py),
[`dataset.py`](../src/kimi_k3/data/dataset.py),
[`inspect_data.py`](../scripts/inspect_data.py).
**Evidence:** [`pilot assessment`](../experiments/pilot/scientific_assessment.md),
[`pilot recipe`](../experiments/pilot/training.yaml),
[`full-run recipe`](../experiments/pretraining/training.yaml).
Dataset-revision IDs, tokenizer-revision IDs, checksums, seed and exact corpus
counts should be taken from the **run's metadata/checkpoint**, not inferred from
these upstream links.

## Kernels and implementation infrastructure

| Reference | Role | Relevance |
| --- | --- | --- |
| **[Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)** | **Software** | External optimized KDA kernels used by the CUDA `fused` backend; the kernel internals are **not** claimed as independently implemented in this repository. |
| **[Triton](https://github.com/triton-lang/triton)** | **Software** | GPU kernel/compiler stack used by FLA. |
| **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)** — Dao et al. (2022) | **Background** | IO-aware attention design; **not** the attention implementation used by this model's MLA module. |
| **[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)** — Dao (2023) | **Background** | GPU attention implementation and parallelism context. |
| **[PyTorch](https://pytorch.org/)** / **[automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)** / **[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)** | **Software** | Training framework, AMP, compiled optimizer option and checkpoint serialization. |

**Code:** [`fused_chunk.py`](../src/kimi_k3/attention/kda/fused_chunk.py),
[`profiling.py`](../src/kimi_k3/training/profiling.py),
[`bench_muon.py`](../scripts/bench_muon.py).

## Scope and attribution policy

- **Research citation is not code provenance.** Citing a paper means it informed
  the design or provides context; it does not mean its source code was copied.
- **Muon is a documented exception:** the Newton–Schulz numerical kernel and
  conventions are adapted from specific MIT-licensed upstream revisions. Keep
  [`muon/NOTICE.md`](muon/NOTICE.md) with the repository and any relevant redistribution.
- **KDA fused kernels are external** through FLA. The readable recurrent and
  chunkwise KDA paths, model composition, training machinery and integration
  are part of this repository's implementation; no authorship is claimed for
  third-party dependencies.
- **Check evidence boundaries.** Tests establish specified behaviors, not
  equivalence to all aspects of the original production Kimi K3. Historical
  pre-registered evaluation artifacts remain recoverable from Git history.
- Links point to published sources or upstream project homes; arXiv links may
  resolve to newer revisions unless a version is explicitly pinned. The primary
  Kimi K3 report above is pinned to `v2`.
