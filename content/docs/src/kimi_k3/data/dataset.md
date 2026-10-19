---
title: "ShardedTokenDataset and TokenDataLoader"
description: "Reads metadata-ordered uint16 shards and provides causal batches with resumable stream offsets."
sourcePath: "src/kimi_k3/data/dataset.py"
status: published
---

This file is the runtime boundary between prepared token storage and model inputs. `ShardedTokenDataset` exposes a flat, metadata-ordered token stream across `.npy` shards. `TokenDataLoader` layers a sequential, non-wrapping batch cursor on that stream. Neither class downloads documents, invokes a tokenizer, builds shards or stores full-dataset tensors on the GPU.

## Manifest identity and startup validation

`ShardedTokenDataset(directory, split)` accepts only `"train"` or `"val"` [source L28–L30](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L28-L30). It reads `metadata.json` and computes `metadata_sha256` from a compact JSON serialization with sorted object keys [source L33–L38](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L33-L38). This digest identifies parsed metadata independently of incidental whitespace or directory location. It is **not** a SHA-256 scan of every token shard.

The constructor requires `format_version == 2`, storage format `"npy"`, an unsigned two-byte NumPy dtype, and tokenizer IDs within uint16 capacity [source L39–L49](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L39-L49). It keeps the manifest's shard ordering exactly, rejects nonconforming or repeated filenames, checks that each file exists, and validates its advertised dtype and one-dimensional `(tokens,)` shape via memory mapping [source L51–L83](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L51-L83). The prefix-sum array `offsets` gives the global beginning of each shard and ends at `total_tokens`. It also cross-checks the total against the split summary. Startup reads shard headers, not the entire binary contents, so silent changes within a shard after preparation are outside this reader's integrity guarantee.

## Random access across shard boundaries

`read_tokens(start, count)` returns an **owned** `uint16` NumPy array for half-open interval `[start,start+count)` [source L85–L106](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L85-L106). Both arguments must be nonnegative integers and the requested range must end inside the stream. Zero-length reads are allowed. The implementation finds the starting shard using `bisect_right(self.offsets, start) - 1` and then iterates across as many shards as necessary.

```python
		shard_index = bisect_right(self.offsets, start) - 1
		written = 0
		while written < count:
			local_start = start - self.offsets[shard_index]
			take = min(count - written, self.offsets[shard_index + 1] - start)
			array = np.load(self.paths[shard_index], mmap_mode="r", allow_pickle=False)
			result[written:written + take] = array[local_start:local_start + take]
			written += take
			start += take
			shard_index += 1
```

These are the exact lines in the disk reader; every chunk is copied into the owned `result` buffer. `np.load(..., mmap_mode="r", allow_pickle=False)` is reopened for each participating shard. The relevant slices are copied into `result`; the returned array is not an alias of the disk mapping. This bounded-memory design makes crossing shard boundaries transparent but deliberately performs no prefetch or persistent mmap caching.

### Walk across a shard boundary

Imagine two manifest-ordered shards containing `[10,11,12,13]` and `[14,15,16]`. The resulting offsets are `[0,4,7]`. Reading `start=3, count=3` chooses shard 0 with `bisect_right([0,4,7],3)-1=0`, copies token `13`, advances the global position to `4`, and then copies `14,15` from shard 1. It returns the **owned** array `[13,14,15]`; neither the manifest nor token order is rearranged. An input beginning at *exactly* offset 4 chooses shard 1 immediately.

The reader allocates an output buffer of `count` uint16 values, opens a `np.load(..., mmap_mode="r")` view of the participating shard, and copies only the relevant slice ([source L85–L106](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L85-L106)). It does **not** read every shard into RAM or retain a persistent mapping. OS page caching and mapped-page residency remain outside the reader's explicit memory accounting. Startup likewise maps files only to check dtype and shape, not to hash their full contents ([source L77–L83](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L77-L83)).

## Shifted causal batches with one lookahead token

`get_batch(start, B, T)` first reads `B*T + 1` tokens and rejects any token ID outside the advertised vocabulary [source L108–L125](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L108-L125). It promotes only this slice to PyTorch `int64` and forms two views:

```python
		buffer = torch.from_numpy(tokens.astype(np.int64))
		x = buffer[:-1].view(batch_size, seq_len)
		y = buffer[1:].view(batch_size, seq_len)
```

Thus `x[b,t]` and `y[b,t]` are adjacent tokens in the global stream, and the final target of the final row is defined. The rows are laid out consecutively without inserting padding or restarting at a shard boundary. Both tensors have shape `(B,T)` and share backing storage, so the docstring asks consumers to treat them as read-only. `get_sequence(start,T)` returns the two one-dimensional rows for evaluation [source L127–L130](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L127-L130).

## Sequential loading, exhaustion and saved cursors

`TokenDataLoader` stores its own `position`; independent instances allow training and validation to maintain separate cursors [source L133–L147](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L133-L147). A successful `next_batch()` reads at `position`, advances by exactly `B*T`, and preserves the lookahead token as the next batch's first input [source L188–L194](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L188-L194). If fewer than `B*T+1` tokens remain, it raises `StopIteration` **without wrapping**; the training loop, not the loader, decides to reset and restart.

The checkpoint contract [source L149–L180](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L149-L180) contains `format_version=1`, position, batch size, sequence length, split and `metadata_sha256`. `validate_state_dict` checks all these fields and returns the candidate position without mutating the cursor; `load_state_dict` applies it only after validation. The manifest files must remain unchanged while a loader is active. Saving a cursor does not serialize dataset contents or the current numpy memory maps.

### A cursor is a position, not an epoch counter

The two-line shift in `get_batch` is literal: a batch covering global tokens `[10,11,12,13,14]` with `B=2`, `T=2` gives `x=[[10,11],[12,13]]` and `y=[[11,12],[13,14]]`. The one lookahead token is enough for the last target ([source L119–L125](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L119-L125)). Subsequent batches advance by `B*T`, not `B*T+1`, so adjacent batches overlap at the **single lookahead/input token** without rescoring the previous batch's targets as an additional row.

`TokenDataLoader.state_dict()` serializes exactly the cursor and compatibility fields ([source L155–L164](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L155-L164)):

```python
		return {
			"format_version": 1,
			"position": self.position,
			"batch_size": self.batch_size,
			"seq_len": self.seq_len,
			"split": self.dataset.split,
			"metadata_sha256": self.dataset.metadata_sha256,
		}
```

The restore path first compares configuration and manifest identity without moving the cursor ([source L166–L180](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L166-L180)):

```python
		current = self.state_dict()
		for name in ("batch_size", "seq_len", "split", "metadata_sha256"):
			if state.get(name) != current[name]:
				raise ValueError(f"Incompatible loader {name}")
		position = _integer(state["position"], "position")
		if position > self.dataset.total_tokens:
			raise ValueError("Saved loader position exceeds the split length")
		return position
```

A successful read in `next_batch` increments `self.position` by `size=B*T`; if `position + size + 1 > total_tokens`, it raises `StopIteration` **before reading or advancing** ([source L188–L194](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py#L188-L194)). The loader never resets itself or stores an epoch number. The trainer may explicitly `reset()` it on exhaustion; calling code can also do so. Therefore any claim that the dataset loader transparently wraps each epoch would be incorrect.

## Verification and boundaries

[test test_tiny_training.py L83–L100](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_tiny_training.py#L83-L100) inspects the causal one-token shift in the integrated tiny recipe. [test test_checkpoint_resume.py L275–L289](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py#L275-L289) covers relocating a manifest and preserving its identity, while [test test_checkpoint_resume.py L290–L340](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py#L290-L340) covers incompatible resume state rejection. The source is the authority for exact format fields and boundary behavior; these tests were inspected, not executed. Shard creation and source-corpus consistency belong to `scripts/prepare_data.py` (outside this author's scope).

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/dataset.py">dataset.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_tiny_training.py">test_tiny_training.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py">test_checkpoint_resume.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
