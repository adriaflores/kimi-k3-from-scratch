# Data preparation notes — FineWeb-Edu → 6B tokens

These are mostly implementation notes for why `prepare_data.py` ended up looking the way it does.

## Starting point

The goal was simple: take a bounded slice of `HuggingFaceFW/fineweb-edu`, tokenize it with the SmolLM2 tokenizer, split it deterministically into train/validation, append EOS between documents, and write compact `.npy` shards that can later be memory-mapped by the training dataloader.

The first version used the convenient Hugging Face streaming path:

```python
load_dataset(..., streaming=True)
    .shuffle(seed=..., buffer_size=1000)
```

Then documents were tokenized in batches and written to fixed-size NumPy shards.

A few design choices were already intentional:

- pin both the dataset and tokenizer to commit SHAs;
- keep the source text unchanged except for skipping whitespace-only documents;
- assign train/val from a SHA-256 hash of the original text, so split membership does not depend on stream order;
- append EOS ourselves instead of asking the tokenizer to add special tokens;
- keep document continuity across shard boundaries;
- stop on a **train-token budget**, not an approximate document count;
- write checkpoints and partial buffers atomically so a long preprocessing run can be resumed.

For the final run we targeted **6B train tokens**, with validation additional.

## First optimization: tokenizer batching

Before touching the full dataset, we benchmarked document-at-a-time tokenization against batched tokenization.

The useful part of the curve was roughly:

| Batch | Local pipeline throughput |
|---:|---:|
| 1 | 428k tok/s |
| 64 | 1.68M tok/s |
| 128 | 1.97M tok/s |
| 256 | 2.06M tok/s |

Batch 128 was already within ~95% of the best observed local throughput, while using less memory than 256, so we picked **128**.

That decision stayed in the final pipeline.

## Then the real problem appeared: RAM kept growing

The first full runs looked healthy at the beginning, then died with errors like:

```text
memory allocation of 1048576 bytes failed
```

At first this looked like a simple memory-pressure problem:

- Windows page file?
- too many applications open?
- shard buffers too large?
- tokenizer batch too large?
- tokenizer/Rayon parallelism?
- BPE cache?

We reduced the shard size to **25M tokens**, closed other applications, limited tokenizer parallelism, disabled the tokenizer BPE cache, and even tried recreating the tokenizer periodically.

None of that fixed the actual pattern.

The important observation was not the final allocation error. It was the shape of RAM usage:

```text
new process
   ↓
RAM starts low
   ↓
RAM grows steadily
   ↓
~98% RAM
   ↓
allocation fails
```

Even more useful: after every crash, resume would start from low RAM again and process roughly another chunk before dying. At first this was around ~300M new tokens per process; later runs varied depending on available memory and replay cost.

That strongly suggested **memory retained over the lifetime of the reader/process**, not one unusually large tokenizer batch.

## Things we checked along the way

There were enough related reports that several hypotheses were plausible.

### Tokenizers

Hugging Face has had a reported memory-growth issue when repeatedly tokenizing large strings:

- `huggingface/tokenizers#1539` — *Memory leak for large strings*  
  https://github.com/huggingface/tokenizers/issues/1539

This made the tokenizer a reasonable suspect, so we tried disabling its BPE cache and periodically recreating the tokenizer.

The RAM still grew. That was useful: it pushed us away from continuing to patch the tokenizer.

### Hugging Face `datasets` streaming

There are also reports of memory increasing during `load_dataset(..., streaming=True)`:

- `huggingface/datasets#7269` — *Memory leak when streaming*  
  https://github.com/huggingface/datasets/issues/7269
- `huggingface/datasets#7722` — *Out of memory even though using load_dataset(..., streaming=True)*  
  https://github.com/huggingface/datasets/issues/7722

These reports do not prove that our exact failure had the same root cause, but they matched the symptom much better: memory increasing continuously while iterating a supposedly streaming dataset.

### PyArrow dataset/scanner path

This was the strongest clue.

Apache Arrow has reports where the dataset/scanner API accumulates memory while reading Parquet, while `ParquetFile.iter_batches()` does not show the same behavior:

- `apache/arrow#39808` — *Dataset.to_batches accumulates memory usage and leaks*  
  https://github.com/apache/arrow/issues/39808
- `apache/arrow#44799` — *OOM with Dataset.scanner but ok with ParquetFile.iter_batches*  
  https://github.com/apache/arrow/issues/44799
- `apache/arrow#47266` — *21.0.0 release introduced memory leak when reading parquet*  
  https://github.com/apache/arrow/issues/47266
- `apache/arrow#49474` — *Memory Leak while iterating batches of pyarrow dataset*  
  https://github.com/apache/arrow/issues/49474

This finally gave us a concrete alternative to test instead of another workaround.

## Isolating the reader

Before rewriting `prepare_data.py`, we built a tiny benchmark that bypassed `datasets` entirely:

```text
Hugging Face Hub
    ↓
download/open pinned Parquet file
    ↓
pyarrow.parquet.ParquetFile
    ↓
iter_batches(columns=["text"])
```

The result was the turning point.

The direct reader processed:

- **3.00 GiB** of decoded text;
- about **704M estimated tokens**;
- **674,816 documents**;

while memory stayed around:

- final RSS: **0.203 GiB**;
- peak RSS: **0.206 GiB**;
- RSS growth over the full test: **136 MiB**.

Most importantly, RAM did **not** show the previous monotonic climb toward OOM.

This was enough evidence for us to stop debugging the tokenizer and replace the reader.

For reference, Hugging Face also documents direct PyArrow access to Parquet files on the Hub:

- Hugging Face — *PyArrow datasets*  
  https://huggingface.co/docs/hub/datasets-pyarrow

## Final reader design

We removed `datasets` from `prepare_data.py`.

The final source pipeline is:

```text
pinned FineWeb-Edu revision
        ↓
resolve exact Parquet file list
        ↓
deterministic file order
        ↓
hf_hub_download / local HF cache
        ↓
ParquetFile.iter_batches()
        ↓
our deterministic bounded shuffle
        ↓
tokenize 128 documents at a time
        ↓
deterministic train/val split
        ↓
append EOS
        ↓
25M-token .npy shards
```

A few details matter here.

### Deterministic bounded shuffle

The old reader relied on:

```python
IterableDataset.shuffle(seed=..., buffer_size=1000)
```

Once we removed `datasets`, we implemented the shuffle ourselves.

It keeps at most **1000 documents** in memory and uses a seeded NumPy PCG64 RNG. File-order RNG and document-shuffle RNG use separate deterministic streams.

This gives us:

- bounded memory;
- reproducibility;
- no dependence on Arrow batch boundaries;
- an implementation we can describe completely in metadata.

We do **not** try to reproduce the old Hugging Face shuffle ordering. The direct-Parquet pipeline defines a new dataset preparation run.

### Exact source provenance

We store the exact ordered Parquet file list and reader/shuffle configuration in metadata/checkpoints.

That means a run is tied to:

- FineWeb-Edu repository;
- pinned dataset commit SHA;
- config (`sample-10BT`);
- exact ordered Parquet files;
- shuffle algorithm/version;
- seed;
- shuffle-buffer size.

### Resume by deterministic replay

Serializing the entire Parquet iterator + shuffle buffer would make resume much more complicated.

Instead, checkpoints save:

- writer state and partial shard buffers;
- number of source rows visited;
- source-prefix SHA-256;
- exact reader configuration/file list.

On resume we:

1. restore committed shards and partial writer buffers;
2. recreate the exact Parquet stream and shuffle;
3. replay the already-visited source rows **without tokenizing or writing them**;
4. recompute the prefix hash;
5. refuse to continue if the hash differs;
6. continue from the next document.

This is slower than a native iterator checkpoint, but it is simple and, more importantly, auditable.

Old checkpoints from the `datasets` reader are deliberately incompatible (`checkpoint_version = 2`).

## Resume validation

Before trusting the new pipeline with 6B tokens, we ran two 100M-token tests.

### Clean run

Result:

```text
train: 95,933 documents, 100,001,598 tokens
val:    5,004 documents,   5,353,777 tokens
```

RAM remained stable.

### Interrupted + resumed run

We intentionally interrupted the second run after a committed checkpoint, then resumed it.

It finished with exactly the same counts:

```text
train: 95,933 documents, 100,001,598 tokens
val:    5,004 documents,   5,353,777 tokens
```

Then we compared the complete shard metadata/hashes:

```text
clean_run["splits"] == resumed_run["splits"]
→ True
```

So the resumed run reproduced the clean run byte-for-byte at the shard level.

At that point the reader + shuffle + checkpoint design was considered validated.

## Final 6B run

The final preparation completed successfully:

```text
train documents:   5,740,283
train tokens:      6,000,001,675

validation docs:     302,339
validation tokens:   316,623,851

elapsed:            ~1.54 h
final train rate:   ~1.08M tok/s
final total rate:   ~1.14M tok/s
```

The train set exceeds 6B by only **1,675 tokens** because we deliberately finish the document that crosses the target instead of truncating it.

Storage uses **25M-token shards**, except for the final partial shard.

## Current state

At this point I consider `prepare_data.py` closed unless we find a concrete correctness issue later.

The main lesson from this part of the project was pretty simple:

> "streaming" does not automatically mean constant-memory.

The original pipeline was conceptually fine, and most of our early debugging focused on the obvious memory consumers. What actually fixed the problem was isolating the reader, reproducing the memory behavior separately, and replacing the higher-level dataset/scanner path with the simpler `ParquetFile.iter_batches()` path that we could measure directly.

The final pipeline is a little more code than `load_dataset(..., streaming=True)`, but now the ordering, memory bound, provenance and resume behavior are all explicit.
