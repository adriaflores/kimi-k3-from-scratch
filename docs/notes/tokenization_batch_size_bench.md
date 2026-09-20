# Tokenization Batch-Size Benchmark

## Goal

Before changing `prepare_data.py` to tokenize documents in batches, we benchmarked several batch sizes on a fixed sample of **10,000 FineWeb-Edu documents** using the `HuggingFaceTB/SmolLM2-135M` fast tokenizer.

Each configuration was run **3 times**, and the benchmark reports median throughput. It also verifies that batched tokenization produces exactly the same output shards as document-at-a-time tokenization by comparing shard hashes.

## Results

| Batch size | Tokenizer throughput | Local pipeline throughput | Peak RSS |
|---:|---:|---:|---:|
| 1 | 446,890 tok/s | 428,048 tok/s | 0.816 GiB |
| 8 | 919,080 tok/s | 852,940 tok/s | 0.896 GiB |
| 16 | 1,197,999 tok/s | 1,094,383 tok/s | 0.908 GiB |
| 32 | 1,532,362 tok/s | 1,381,762 tok/s | 0.910 GiB |
| 64 | 1,909,307 tok/s | 1,679,778 tok/s | 0.922 GiB |
| 128 | 2,280,788 tok/s | 1,967,574 tok/s | 0.934 GiB |
| 256 | 2,406,935 tok/s | 2,059,683 tok/s | 0.957 GiB |

All tested configurations produced matching hashes.

The selected candidate, **batch size 128**, was then measured with the actual streaming pipeline:

- Streaming pipeline throughput: **301,652 tok/s**
- Peak RSS: **1.492 GiB**
- Rough extrapolation for 6B training tokens: **~5.8 hours**

## Interpretation

Batching provides a large improvement over encoding documents individually. Local pipeline throughput increased from roughly **0.43M tok/s at batch size 1** to **1.97M tok/s at batch size 128**, a gain of about **4.6×**.

The improvement starts to saturate around batch size 128. Moving from 128 to 256 increases local pipeline throughput by only about **4.7%**, while using a larger batch and slightly more memory.

For this reason, we use the benchmark rule:

> Choose the smallest batch size that reaches at least 95% of the best eligible local pipeline throughput.

Batch size 128 satisfies this condition:

- Best observed local throughput: 2,059,683 tok/s at batch 256
- 95% threshold: ~1,956,699 tok/s
- Batch 128 throughput: 1,967,574 tok/s

Therefore, **128 is the preferred batch size**.

The streaming measurement also shows that tokenization is no longer the main end-to-end bottleneck. The local pipeline reaches ~1.97M tok/s at batch 128, whereas the complete streaming pipeline reaches only ~0.30M tok/s. This indicates that remote dataset access, iteration, and related I/O dominate the real preparation time much more than tokenizer compute.

Testing substantially larger batches such as 512 or 1024 is therefore unlikely to materially improve the complete pipeline at this stage.

## Decision

`prepare_data.py` should be updated to:

1. Read source documents in batches of **128**.
2. Tokenize the complete batch in one call to the fast tokenizer.
3. Keep `add_special_tokens=False` and `truncation=False`.
4. Append `eos_token_id` manually to every document exactly as before.
5. Preserve the existing deterministic train/validation split and shard-writing logic.

No other optimization is currently justified by the measurements.

The batch size is a **hardware-dependent tuning parameter**, not a universal constant. If preprocessing is later moved to a substantially different machine or CPU configuration, this benchmark should be rerun there before preparing the final multi-billion-token dataset.

## References

Hugging Face's fast tokenizers are backed by the Rust `tokenizers` library, and the official documentation specifically notes that batched tokenization provides significant speedups and allows the backend to parallelize work efficiently.

- Hugging Face Transformers — Fast tokenizers:  
  https://huggingface.co/docs/transformers/en/fast_tokenizers
- Hugging Face Transformers — Tokenizer API:  
  https://huggingface.co/docs/transformers/main_classes/tokenizer
- Hugging Face Tokenizers — Quicktour, batch encoding:  
  https://huggingface.co/docs/tokenizers/quicktour

These references motivate batching in general; the choice of **128** is empirical and comes from the benchmark above rather than from a fixed recommendation in the library documentation.
