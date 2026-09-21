import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


class ShardedTokenDataset:
    """A metadata-ordered token stream, with explicit token offsets.

    Startup checks every selected shard's header, not its entire contents or
    SHA-256. Reads map only the needed shards and copy only the requested range.
    This intentionally simple reader reopens mappings per read; it does not
    retain mappings, create worker processes, or prefetch batches.
    """

    def __init__(self, directory: str | Path, split: str = "train"):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.directory = Path(directory)
        self.split = split
        self.metadata = json.loads(
            (self.directory / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata["format_version"] != 2:
            raise ValueError("Expected prepared-data format_version=2")
        storage = self.metadata["storage"]
        self.dtype = np.dtype(storage["dtype"])
        if storage["format"] != "npy" or self.dtype.kind != "u" or self.dtype.itemsize != 2:
            raise ValueError("Expected uint16 .npy token shards")
        tokenizer = self.metadata["tokenizer"]
        self.vocab_size = _integer(tokenizer["vocab_size"], "vocab_size", 1)
        self.eos_id = _integer(tokenizer["eos_id"], "eos_id")
        if not self.eos_id < self.vocab_size <= 2**16:
            raise ValueError("Tokenizer IDs must fit the uint16 storage")

        summary = self.metadata["splits"][split]
        self.shards = summary["shards"]  # Preserve the manifest order exactly.
        self.paths: list[Path] = []
        self.offsets = [0]
        seen = set()
        for entry in self.shards:
            filename = entry["file"]
            if (not isinstance(filename, str) or "/" in filename or "\\" in filename
                    or not filename.startswith(f"{split}_") or not filename.endswith(".npy")):
                raise ValueError(f"Invalid {split} shard filename: {filename!r}")
            if filename in seen:
                raise ValueError(f"Duplicate shard in metadata: {filename}")
            seen.add(filename)
            count = _integer(entry["tokens"], f"{filename} tokens", 1)
            path = self.directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Metadata references missing shard: {path}")
            self._validate_shard(path, count)
            self.paths.append(path)
            self.offsets.append(self.offsets[-1] + count)
        self.total_tokens = self.offsets[-1]
        if self.total_tokens != _integer(summary["tokens"], "split tokens"):
            raise ValueError("Split token count does not equal the sum of shard lengths")
        if not self.paths:
            raise ValueError(f"No shards for split {split!r}")

    def _validate_shard(self, path: Path, count: int) -> None:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != self.dtype or array.shape != (count,):
            raise ValueError(
                f"Invalid shard {path.name}: expected {self.dtype} shape {(count,)}, "
                f"got {array.dtype} shape {array.shape}"
            )

    def read_tokens(self, start: int, count: int) -> np.ndarray:
        """Return an owned uint16 copy of [start, start + count), across shards.

        Never wraps at the end of a split. A zero-length range is allowed.
        Prepared files must remain unchanged while the reader is in use.
        """
        start = _integer(start, "start")
        count = _integer(count, "count")
        if start + count > self.total_tokens:
            raise IndexError(f"Range [{start}, {start + count}) exceeds {self.total_tokens} tokens")
        result = np.empty(count, dtype=self.dtype)
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
        return result

    def get_batch(
        self, start: int, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows start at start + row * seq_len, each reading seq_len + 1 tokens.

        Each input token has its next token as target, including the final
        position in each row. Only this batch is converted to int64. x and y
        share backing storage: treat returned tensors as read-only.
        """
        batch_size = _integer(batch_size, "batch_size", 1)
        seq_len = _integer(seq_len, "seq_len", 1)
        tokens = self.read_tokens(start, batch_size * seq_len + 1)
        if int(tokens.max()) >= self.vocab_size:
            raise ValueError(f"Batch contains a token ID >= vocab_size ({self.vocab_size})")
        buffer = torch.from_numpy(tokens.astype(np.int64))
        x = buffer[:-1].view(batch_size, seq_len)
        y = buffer[1:].view(batch_size, seq_len)
        return x, y

    def get_sequence(self, start: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one sequence at an arbitrary token offset, including boundaries."""
        x, y = self.get_batch(start, batch_size=1, seq_len=seq_len)
        return x[0], y[0]


class TokenDataLoader:
    """Sequential full batches; exhaustion raises StopIteration without wrapping.

    position is the next input-token offset. Each successful batch advances
    B*T, retaining the lookahead token as the next batch's first input. At the
    very end of the split, fewer than B*T+1 remaining tokens cannot form a full
    batch; they remain readable through the dataset's explicit range methods.
    Independent instances have independent cursors (use one for evaluation).
    """

    def __init__(self, dataset: ShardedTokenDataset, batch_size: int, seq_len: int):
        self.dataset = dataset
        self.batch_size = _integer(batch_size, "batch_size", 1)
        self.seq_len = _integer(seq_len, "seq_len", 1)
        self.position = 0

    def reset(self, position: int = 0) -> None:
        position = _integer(position, "position")
        if position > self.dataset.total_tokens:
            raise IndexError("position exceeds the split length")
        self.position = position

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.next_batch()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.batch_size * self.seq_len
        if self.position + size + 1 > self.dataset.total_tokens:
            raise StopIteration
        batch = self.dataset.get_batch(self.position, self.batch_size, self.seq_len)
        self.position += size
        return batch
