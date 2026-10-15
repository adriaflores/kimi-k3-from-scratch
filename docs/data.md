# Data pipeline

The data pipeline converts a bounded subset of FineWeb-Edu into compact, deterministic token shards suitable for sequential next-token pretraining.

## Inputs and provenance

- **Dataset:** [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT`.
- **Tokenizer:** [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) tokenizer, vocabulary size 49,152.
- **Storage:** NumPy `uint16` `.npy` shards, with a `metadata.json` manifest (format version 2).
- **Budget:** at least six billion training tokens; validation data is additional.

The generated corpus used for qualification contains 241 train shards and 13 validation shards. The source dataset and tokenizer revisions are pinned in the generated manifest: consult that manifest for exact provenance rather than relying on a moving repository default.

## Preparation decisions

[`scripts/prepare_data.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/scripts/prepare_data.py) prepares the corpus. The pipeline:

1. Iterates through source documents with a bounded deterministic shuffle.
2. Splits train/validation deterministically using a hash of original text.
3. Tokenizes without automatic special tokens and appends EOS (`0`) between documents.
4. Writes fixed-size shards (25M tokens per standard shard) without padding every document to context length.
5. Saves metadata/checkpoints to permit a validated restart of preprocessing.

The implementation evolved after diagnosing cumulative memory growth in the initial streaming reader. The detailed investigation remains in [`docs/notes/data_preparation.md`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/docs/notes/data_preparation.md).

## Loading batches

[`src/kimi_k3/data/dataset.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/data/dataset.py) provides a sharded token dataset and a next-token dataloader. Each batch produces causal `(x, y)` sequences with the target shifted one token ahead. Training and validation have separate, checkpointed cursors.

Data quality, sharding identity, and cursor semantics matter for reproducible continuation. Before reusing a checkpoint, verify that its manifest matches the current prepared corpus; file paths alone are not sufficient evidence of identity.
