---
title: "inspect_data.py — Prepared shard inspection"
description: "Loads one causal token batch and independently checks a boundary between prepared NumPy shards."
sourcePath: "scripts/inspect_data.py"
status: review
---

[`scripts/inspect_data.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_data.py) is a read-only inspection command for a prepared token dataset. It does not prepare data, train a model, or verify every token in the corpus. The reusable reader is documented in [dataset.py](/docs/src/kimi_k3/data/dataset/). Instead, it asks two concrete questions: does the first batch obey the next-token shift, and can the reader reconstruct a sequence that crosses from the first shard into the second?

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## Command and inputs

The [argument parser](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_data.py#L13-L19) requires `--data-dir` and accepts `--split {train,val}` (default `train`), `--batch-size` (default `4`) and `--seq-len` (default `128`). The directory must contain a compatible `metadata.json` and the named `.npy` shards. Relative paths are interpreted relative to the process working directory. The script adjusts `sys.path` to include the checkout's `src/` directory before importing the data classes; it can therefore run without installing the local package first.

For example, from the repository root, with `path/to/prepared-data` replaced by a real prepared-data directory:

```bash
python scripts/inspect_data.py --data-dir path/to/prepared-data --split val --batch-size 2 --seq-len 128
```

There is no output file, no random seed, and no GPU selection. This is a CPU/NumPy/PyTorch inspection. A single batch requires at least `batch_size * seq_len + 1` tokens in the chosen split; exhaustion raises `StopIteration`.

## Read and validate a causal batch

The [first part of `main()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_data.py#L21-L28) instantiates the metadata-ordered stream and a sequential loader, then requests exactly one batch:

```python
dataset = ShardedTokenDataset(args.data_dir, split=args.split)
loader = TokenDataLoader(dataset, args.batch_size, args.seq_len)
x, y = loader.next_batch()
assert x.shape == y.shape == (args.batch_size, args.seq_len)
assert x.dtype == y.dtype == torch.long
assert torch.equal(x[:, 1:], y[:, :-1])
```

Both tensors have shape `(B, T)` and `torch.long` dtype. In the flattened token stream, `x[b,t]` is the input token and `y[b,t]` is the following token, so `x[:,1:] == y[:,:-1]`. This checks the interior shift within each row, while the loader itself decides where each row begins. The remaining assertions check nonnegative IDs and IDs below `dataset.vocab_size`, including the last target of each row. The script does **not** assert that the vocabulary includes every ID, or that the data has any particular language distribution.

The [diagnostics](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_data.py#L29-L35) print the selected split, shard and token counts, tensor shape/dtype, vocabulary size, EOS ID, token minima/maxima, and EOS occurrence counts in this one batch. `Causal shift: PASS` is a message emitted **after the assertions succeed**, not an independent test-suite result.

## Check the first shard boundary independently

The second check deliberately avoids trusting `get_sequence()` to construct its own expected answer. It [memory-maps the first two `.npy` files](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_data.py#L37-L49), copies a tail from the first and a head from the second, and compares these to a reader request spanning the same range.

```python
expected = np.concatenate((left[-tail_count:], right[:count - tail_count])).astype(np.int64)
bx, by = dataset.get_sequence(start, count - 1)
np.testing.assert_array_equal(bx.numpy(), expected[:-1])
np.testing.assert_array_equal(by.numpy(), expected[1:])
```

Here `count` is no greater than `seq_len + 1` and the combined length of the two shards; `tail_count` is adjusted to fit both sources. `start` is an absolute token offset equal to the beginning of that tail within shard one. Since `get_sequence(start, count-1)` reads one extra lookahead token, the expected inputs are `expected[:-1]` and the targets are `expected[1:]`. The assertion therefore checks not only token values but also the shifted target at the boundary.

With fewer than two shards, the command [prints `Boundary check: SKIPPED`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_data.py#L39-L41) and returns normally. With two or more it reports the tail/head lengths only after both array comparisons succeed. If one of the two selected shards is too small to form the intended cross-boundary range, the check may become a short boundary probe rather than a full-length one.

## Limits and verification

The underlying [`ShardedTokenDataset`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L19-L94) validates file headers, metadata counts and dtypes but **does not hash every shard on initialization**. This command checks one loader batch and one boundary only. It neither sweeps the entire dataset nor compares stored SHA-256 hashes. The reader requires uint16 `.npy` storage, even though data preparation has a wider-integer fallback for larger tokenizers.

The reader behavior is implemented in [dataset.py](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L95-L193); the [train CLI fixture](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_train_cli.py#L22-L41) exercise prepared-stream assumptions. The latter creates metadata and tiny shards for a downstream integration test. No dedicated test of the `inspect_data.py` entry point was identified in the inspected suite; the command contains its own runtime assertions. These are source-inspection observations, not claims of a fresh execution.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_data.py">inspect_data.py · pinned revision</a></dd>
    <dt>Dependency</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py">dataset.py</a> · shard and loader semantics</dd>
    <dt>Verification</dt><dd>Local assertions and source-inspected <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_train_cli.py#L22-L41">prepared fixture</a>; no CLI execution performed here</dd>
  </dl>
</aside>
