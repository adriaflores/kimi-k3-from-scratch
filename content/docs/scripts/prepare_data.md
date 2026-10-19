---
title: "prepare_data.py — Reproducible token-shard preparation"
description: "Resolves a versioned Parquet corpus, shuffles and tokenizes documents, and publishes resumable train/validation shards."
sourcePath: "scripts/prepare_data.py"
status: review
---

[`scripts/prepare_data.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py) converts a Hugging Face-hosted Parquet corpus into fixed-capacity NumPy token shards for the model's streaming data loader. The hard part is not calling a tokenizer. The file must establish **which exact rows were visited, in what order, how each was assigned to a split, and how an interrupted preparation can continue without silently changing the dataset**. Its solution is a pinned source revision, two seeded random streams, a deterministic content-hash split, checksummed shard/buffer snapshots, and a replay check before resuming.

The command sets `USE_TORCH=0` by default before importing Transformers to avoid bringing PyTorch into this tokenization path. It **does not train a model**, construct batches of context length `T`, create the training optimizer or write trainer-format checkpoints. The `.npy` sequences are flat token streams; [`ShardedTokenDataset`](/docs/src/kimi_k3/data/dataset/) later exposes windows and causal next-token targets.

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## CLI contract: two ways to bound the source

The [argument parser](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L28-L57) offers two mutually exclusive stopping modes. `--max-documents N` limits the number of source rows visited in the shuffled stream; `--train-tokens N` stops once the **training** token budget (including EOS) is reached, after finishing the document that crossed it. If neither is specified, the program sets `max_documents=10_000`. Validation tokens are additional to the train-token budget.

| Argument | Default | Role |
|---|---|---|
| `--output-dir` | `C:/datasets/fineweb_edu_6b` | Final output directory; a developer-specific, nonportable default |
| `--config` | `sample-10BT` | Dataset card configuration for Parquet train paths |
| `--revision` | `main` | Requested dataset ref, resolved to a specific SHA |
| `--tokenizer` | `HuggingFaceTB/SmolLM2-135M` | Tokenizer repository |
| `--tokenizer-revision` | `main` | Requested tokenizer ref, resolved to a specific SHA |
| `--tokenization-batch-size` | `128` | Nonempty documents passed per tokenizer call |
| `--val-fraction` | `0.05` | Expected fraction of documents assigned to validation |
| `--seed` | `1337` | Seed for both separated RNG streams and split hashing |
| `--shuffle-buffer` | `1000` | Maximum number of buffered document values |
| `--shard-tokens` | `25_000_000` | Capacity in token elements for each saved shard |
| `--log-interval` | `10` seconds | Minimum elapsed time between progress messages |
| `--checkpoint-interval` | `300` seconds | Minimum elapsed time between checkpoint opportunities at batch boundaries |
| `--resume` | false | Resume the `.incomplete` directory associated with `--output-dir` |

All positive integer counts and intervals must be greater than zero; `0 < val_fraction < 1`, and the seed must be an integer in `[0,2**32)`. The parser does not test network access, disk capacity or split nonemptiness up front. Time-based intervals are only **checked after work advances**; a slow tokenizer call cannot be interrupted mid-batch simply because the checkpoint interval elapsed.

An illustrative small-data command from the repository root is:

```bash
python scripts/prepare_data.py --output-dir path/to/new-data --max-documents 1000 --seed 1337
```

`path/to/new-data` is a placeholder for a **new, nonexistent output directory**. This example contacts Hugging Face to resolve/download data and writes files; it is not a safe dry run. There is no `--dry-run` mode. Always inspect `python scripts/prepare_data.py --help` when a read-only command is needed.

## Pin the tokenizer and its true vocabulary range

After defining the dataset and tokenizer defaults, [`load_tokenizer()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L60-L78) resolves the requested tokenizer revision using `HfApi().model_info(...).sha`. A moving ref such as `main` is converted into the content-addressed revision actually used to load a **fast** tokenizer. The function rejects missing SHA, a non-fast tokenizer or a tokenizer without an EOS ID.

```python
vocab_size = max(max(tokenizer.get_vocab().values()), tokenizer.eos_token_id) + 1
metadata = {
    "repository": repo_id, "revision": sha,
    "vocab_size": vocab_size, "eos_id": tokenizer.eos_token_id,
    "use_fast": True, "add_special_tokens": False,
    "append_eos": True,
}
```

Using `max ID + 1`, rather than `tokenizer.vocab_size`, includes added tokens and preserves possible ID gaps. This number is a storage bound, **not** the number of unique tokens observed in the selected documents. The tokenization path does not add tokenizer-defined BOS or other special tokens; a single EOS is appended explicitly per nonempty document later. Pinned revision IDs, tokenizer settings and package versions are kept in saved metadata so a restart has a concrete compatibility contract.

## Choose a reproducible source order

The helper [`source_rng(seed, stream)`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L81-L83) builds a NumPy `PCG64` generator from `SeedSequence([seed, stream])`. Stream zero controls file order; stream one controls document-level buffered shuffling. Separating streams prevents changes in file-count RNG consumption from directly sharing state with the document shuffle.

[`resolve_parquet_files()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L86-L106) reads the dataset card at the pinned dataset revision. It finds the configured `config_name`, collects its **train** `data_files` globs, intersects them with `.parquet` siblings, sorts the matching repository-relative paths lexicographically, then shuffles the list with RNG stream zero. It rejects absent config or no matching train Parquet files. This is **not** arbitrary directory enumeration and does not use a separate Hugging Face validation split.

The [`reader_metadata()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L109-L125) record stores the ordered filenames, Parquet interface, batch size `1024`, column `text`, pre-buffer/memory-map/thread settings, and the file and row shuffle algorithms with seed sequences. These choices become part of checkpoint compatibility. Row order within each Parquet file is the stored row order.

[`parquet_texts()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L128-L143) downloads each pinned file through `hf_hub_download()`, opens it with `pyarrow.parquet.ParquetFile`, yields the Python value of each `text` cell, then closes that file on normal or early exit. Materialized Parquet downloads live in the Hugging Face cache; the final prepared-data directory stores NumPy shards, not copies of source Parquet files. The process needs network/cache access to the selected versions, and a failure downloading or decoding a file stops the run.

### Bounded document shuffle

[`shuffled_texts()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L146-L164) initially fills an in-memory list up to `shuffle_buffer` documents. For each subsequent input, it chooses a uniformly random buffer index, emits the value currently there, and replaces that slot with the new text:

```python
index = int(rng.integers(0, len(buffer)))
yield buffer[index]
buffer[index] = text
```

Once source files are exhausted, it shuffles and drains the remaining buffer. This is a **bounded-buffer shuffle**, not a full permutation over all corpus rows. Memory is bounded by *document count*, not document byte length: a few unusually long text entries can still require significant memory. RNG calls depend on row sequence rather than Arrow batch boundaries, an important property when rebuilding the source prefix on resume. `closing(...)` around the generator helps ensure underlying Parquet reading can be closed after early termination.

## Split documents by their original text

[`split_for()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L167-L174) hashes the original string rather than drawing from a shared RNG:

```python
digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
return "val" if int.from_bytes(digest[:8], "big") < int(val_fraction * 2**64) else "train"
```

For a fixed seed and exact input text, the first 64 digest bits deterministically select one of the two splits regardless of visitation order. Equal texts always go to the same split, limiting **exact-duplicate cross-split leakage**. The function neither deduplicates nor recognizes near-duplicates. Since selection is a hash threshold, `val_fraction=0.05` is an *expected* document fraction, not a promise of exactly 5% of documents or tokens. In particular, long documents can make the validation token proportion noticeably different.

Whitespace-only documents are skipped later. Nonempty text is preserved unchanged before tokenization, so different whitespace, encoding-normalized spellings or small edits can have different split hashes. The splitting step does not produce fixed-length training examples; it only chooses which flat token stream receives each document.

## Append tokens into two bounded shard writers

[`ShardWriter`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L177-L213) owns one NumPy buffer for each split, with shape `(shard_tokens,)` and an unsigned integer dtype. `append()` increments the document and token counts and copies incoming tokens into the current buffer. A document can cross a shard boundary; when the buffer fills, `flush()` writes the occupied region to a new numbered file such as `train_00000.npy` or `val_00000.npy`.

```python
size = min(len(tokens) - offset, len(self.buffer) - self.used)
self.buffer[self.used : self.used + size] = tokens[offset : offset + size]
self.used += size
offset += size
if self.used == len(self.buffer):
    self.flush()
```

A shard records `file`, token count and SHA-256 of the **saved file bytes**. These are checksums of NumPy `.npy` artifacts, including their header, not hashes of raw text or decoded tokens alone. The writer's `summary()` also exposes its document/token totals. An incomplete final buffer is not published as a normal shard until the finalization phase. This architecture uses one buffer per split; a large `--shard-tokens` means memory proportional to roughly twice the specified token capacity times the dtype width, plus current texts/tokenization arrays and runtime overhead.

### Atomic file writes and storage width

[`save_array()` and `save_json()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L216-L237) write a sibling temporary file, flush it and `os.fsync()` its file descriptor, then `os.replace()` it into the target path. This ensures readers do not normally see half-written committed `.npy` or JSON files, subject to filesystem guarantees. They are **not** a general multi-file transaction. The helpers do not guarantee recovery from arbitrary storage or hardware corruption.

The storage dtype is selected [after loading the pinned tokenizer](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L403-L415): little-endian uint16 (`<u2`) if vocabulary ID range fits 65,536 entries, otherwise little-endian uint32 (`<u4`) up to a range of `2**32`. Larger ranges raise. **Compatibility limit:** the current [`ShardedTokenDataset`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L35-L49) accepts only uint16 shards. Thus the uint32 path can prepare artifacts that the present model data loader will not consume without a corresponding reader change.

## Recoverability: checkpoint buffers, pointer, replay

Preparation checkpoints are **distinct from training checkpoints**. A preparation checkpoint consists of `checkpoint.json`, saved partial-buffer `.npy` snapshots for both splits, and references to already committed normal shards. It is designed to restore the exact prepared prefix, not model weights, optimizer state or RNG tensors.

### Commit a coherent checkpoint

[`save_checkpoint()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L256-L275) creates a new UUID generation, writes a snapshot for the partial buffer of each split and records its hash, plus each writer's document/token/shard summary. Only **after** both new buffer files exist does it atomically replace `checkpoint.json`:

```python
checkpoint = {**state, "writers": snapshots,
              "elapsed_seconds": time.perf_counter() - started}
save_json(directory / "checkpoint.json", checkpoint)
clean_checkpoint_buffers(directory, {s["buffer_file"] for s in snapshots.values()})
```

The JSON file is the committed pointer to the current writer state. Old buffer snapshots remain valid until that replacement, then `clean_checkpoint_buffers()` tries to remove snapshots not referenced by the new checkpoint. A failed cleanup is logged as a warning, rather than invalidating the published pointer. A crash before pointer replacement should leave the earlier checkpoint addressable; a crash afterwards should leave the new snapshots named in the committed JSON. That design does not make work since the last checkpoint durable.

### Restore only committed files

[`restore_writers()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L278-L304) verifies SHA-256 for every named normal shard and each split's buffer snapshot. It rejects a buffer with wrong dimensionality/dtype or greater-than-capacity length. It also checks

$$\text{saved split tokens}=\sum\text{saved shard tokens}+\text{saved buffer length}.$$

Once those checks pass, it reconstructs `used`, the buffer prefix, document/token totals and shard lists. It then **deletes** `train_*.npy` and `val_*.npy` files in the `.incomplete` staging directory that are *not* named by the checkpoint. Those are treated as uncommitted output from a failed batch, not evidence to be silently adopted. It finally cleans obsolete checkpoint buffers. This is why a resume must point at the correct staging directory: it is not a read-only validation operation.

### Rebuild and verify the source prefix

The checkpoint records `source_rows_visited` and a source-prefix SHA-256 digest, together with the reader algorithm, pinned revisions, tokenizer, config, package versions and writer state. On resume, [`_replay_source_prefix()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L328-L345) consumes that many rows from the reconstructed shuffled stream. It feeds each original text to `read_text()`, which hashes an eight-byte length followed by its UTF-8 bytes:

```python
raw = text.encode("utf-8")
source_digest.update(len(raw).to_bytes(8, "big"))
source_digest.update(raw)
```

The replay **does not retokenize or rewrite** committed prefix documents. It rejects a source that ends too early or whose replayed order/content digest disagrees with the checkpoint. After replay, the iterator is positioned at the first unread source row. This favors verifiable reproducibility over instant seeking: a late restart may still need to download/read and reshuffle a long already-processed prefix. Source hashing verifies the prefix actually encountered; it is not a checksum of every file in the remote dataset or of unvisited source rows.

## Tokenization and stopping within a document batch

[`_write_token_batch()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L348-L364) calls the fast tokenizer on a batch of nonempty texts in their preserved order:

```python
encoded = tokenizer(
    texts, add_special_tokens=False, padding=False, truncation=False,
    return_attention_mask=False, return_token_type_ids=False, verbose=False,
)["input_ids"]
for text, tokens in zip(texts, encoded, strict=True):
    split = split_for(text, seed, val_fraction)
    tokens.append(tokenizer.eos_token_id)
    writers[split].append(tokens)
```

There is no truncation, padding or model-context packing. Every visited **nonempty and written** document contributes its own EOS; empty or whitespace-only documents do not enter this function. Documents can be longer than a single token shard. The `strict=True` zip rejects unequal counts of input texts and encoded outputs rather than silently dropping unmatched items.

For `--train-tokens`, the helper checks the accumulated **train-split** token count immediately after each written document and breaks upon reaching the requested threshold. This can overshoot the budget by the remainder of that complete document; the code does not cut a sequence to hit an exact total. The surrounding batch may have been prefetched and tokenized more widely than was actually written. Those already-read but unwritten documents are accounted for in final `prefetched_documents_not_written`. The validation split receives tokens only from documents encountered before stopping, and is not capped by the train target.

## `prepare()` orchestrates staging, resume and publication

The top-level [`prepare()` function](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L366-L540) links the earlier primitives into one lifecycle. Its order makes the safety model easier to audit.

### 1. Guard the output and recover saved settings

The final requested output is resolved to an absolute path. Working files live in a **sibling** directory whose name has `.incomplete` appended. If the final output directory already exists, a normal run rejects it. The one exception is `--resume` plus an existing `metadata.json` in that final directory: the function logs "already complete" and returns that metadata **without rehashing all completed shards**. If no final directory exists but an unfinished staging directory does, starting without `--resume` also fails rather than overwriting the partial work.

For `--resume`, the program reads staging `checkpoint.json`, requires `checkpoint_version=2` and **exact** saved runtime version metadata. It restores preparation settings from the checkpoint, apart from `output_dir`, `resume`, `log_interval` and `checkpoint_interval`. Hence the user's new CLI source/budget/tokenizer options generally do **not** replace the checkpointed recipe: the intended continuation keeps the original source and tokenization algorithm, while output destination and logging cadence remain current. Resuming an obsolete checkpoint format or different package/Python versions is an error, not a best-effort conversion.

### 2. Resolve identities and create writers

It resolves the dataset revision to an immutable commit SHA for new runs; resumed runs reuse the saved revision and source repository. The saved reader settings are checked against the expected `reader_metadata()` algorithm, and the loaded tokenizer metadata is checked for exact equality. It selects uint16/uint32, creates the staging directory for a fresh run, builds the two `ShardWriter`s, restores checkpoint buffers if needed, and initializes source-row/empty-document counters and the prefix digest. A fresh run records an **initial checkpoint** even before processing a source row.

The runtime metadata captures Python and selected `huggingface-hub`, NumPy, PyArrow, Transformers and Tokenizers package versions. These are checked on resume; they do not prove that all platform/hardware behavior beyond the specified algorithms is invariant.

### 3. Stream, skip empties, then checkpoint only between batches

If the restored checkpoint does not mark the source as finished, the program rebuilds the buffered-shuffle iterator and optionally limits it with `islice(..., max_documents)`. It replays the saved prefix if necessary. The inner `nonempty_texts()` generator increments `visited`, validates that the text column actually contains strings, updates the length-delimited source digest and skips whitespace-only strings. For each group of up to `tokenization_batch_size` nonempty documents, it tokenizes, assigns split and appends tokens.

Progress is logged at elapsed-time intervals. After a batch, if the train budget is reached the loop exits; otherwise an elapsed checkpoint may be saved. The code deliberately avoids checkpointing *inside* an incomplete batch or in an exception handler, so the last saved cursor and writer snapshots always describe a coherent processed prefix. When the stream is done, it calls `checkpoint(finished=True)` **before** final buffer flushing. If interrupted in this short finalization stage, resume can finish without replaying the already-completed source. On exceptions, staging is intentionally left in place for recovery.

### 4. Flush, record provenance and publish

Before publication the program requires **both** writers to contain at least one document. It flushes partial buffers to numbered shards, logs progress and evaluates whether the requested train-token target was reached. If the source ran out before that threshold, it **warns and still saves the available data**, setting `train_token_target_reached=false`; reaching a target is not an implicit fatal postcondition.

[`metadata.json`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L502-L519) carries format version 2, pinned source/config/split SHA, original preparation settings, reader order/algorithm, source-prefix digest, split rule, text processing rule, tokenizer metadata, NumPy storage format/dtype, visited/skipped/prefetched counts, target status, performance estimates, split/shard counts and SHA-256 hashes, runtime package versions and a SHA-256 of the preparation script's bytes. It is a provenance record and the loader's manifest, not a trainer run manifest.

The script writes this metadata **within staging**, refuses publication if the final directory appeared meanwhile, and renames the whole staging directory to the final output name. Only after publication does it attempt to remove stale checkpoint buffer files and `checkpoint.json`/its temporary variant. The completed dataset normally consists of `metadata.json` and numbered `train_*.npy`/`val_*.npy` shards. Cleanup failures are warnings, not evidence that the dataset never published. The final log reports split document/token totals and the output directory.

## Throughput and memory must be interpreted as preparation metrics

[`report_progress()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py#L307-L325) measures elapsed wall time from `time.perf_counter()`, adjusted by saved elapsed time across resumes. The reported `train_tokens_per_second` is **prepared train-split tokens** divided by elapsed preparation time; `total_tokens_per_second` also includes validation tokens. ETA uses the explicit train target when given or a nominal **six-billion-train-token projection** otherwise. These estimates include I/O, source reading, tokenization, hashing, checkpoint work and interruptions accounted for by the saved elapsed counter; they exclude the time while a preparation process was shut down and are not GPU training throughput or evidence of a full 6B dataset being produced by a small default run.

The two dense buffers themselves require approximately `2 * shard_tokens * dtype.itemsize` bytes, excluding all text buffers, Arrow row batches, tokenizer allocations, other Python objects and filesystem caches. The shuffle buffer is capped in number of documents, not bytes. The script processes source files sequentially and does not launch multiple tokenizer-worker processes in this file, though its dependencies may perform their own internal work. There is no GPU requirement.

## Failure modes, data integrity and verification

The main protection is **refuse rather than guess**: existing output directories are not overwritten, mismatched checkpoint versions/settings/hashes abort resume, missing or corrupted shards abort restoration, a changed source prefix aborts replay, non-string texts raise, and empty final train or validation split blocks publication. A dataset can still be *complete but below target*, and an already-published output fast path does not perform a full integrity rescan. Verification of saved SHA-256 values on restore does not turn the prepared corpus into a deduplicated or statistically representative evaluation set.

The inspected [unit tests for preparation](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_prepare_data.py#L35-L75) use a fake tokenizer and sinks to assert tokenization flags, document order, EOS insertion, early stop *after* the document crossing a token threshold, replay digest matching and preservation of the unread iterator suffix, and rejection of changed/shortened replay sources. They do **not** by themselves test live Hugging Face downloads, multi-gigabyte sharding, crash consistency on a real filesystem, or a full interruption-and-resume invocation through `prepare()`. No expensive dataset creation or live download was performed for this documentation review.

The handoff distinction is important: this script's checkpointing guards **data preparation**. The training state and optimizer checkpoints belong to [`trainer.py`](/docs/src/kimi_k3/training/trainer/), while [`dataset.py`](/docs/src/kimi_k3/data/dataset/) defines what downstream consumers accept. In particular, the uint32 writer fallback is a visible limit at that interface, not a feature of the current data loader.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/prepare_data.py">prepare_data.py · pinned revision</a></dd>
    <dt>Consumer</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py">dataset.py</a> · prepared-shard format and loader constraints</dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_prepare_data.py">EOS, batch stopping and replay unit checks</a>; only source inspected</dd>
  </dl>
</aside>
