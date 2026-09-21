import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Also work in a uv project that has not installed the src package yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kimi_k3.data import ShardedTokenDataset, TokenDataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    dataset = ShardedTokenDataset(args.data_dir, split=args.split)
    loader = TokenDataLoader(dataset, args.batch_size, args.seq_len)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (args.batch_size, args.seq_len)
    assert x.dtype == y.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert 0 <= min(x.min().item(), y.min().item())
    assert max(x.max().item(), y.max().item()) < dataset.vocab_size
    print(f"{args.split}: {len(dataset.paths)} shards; {dataset.total_tokens:,} tokens")
    print(f"x/y: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"Vocabulary: {dataset.vocab_size}; EOS ID: {dataset.eos_id}")
    for name, tensor in (("x", x), ("y", y)):
        print(f"{name}: min={tensor.min().item()}, max={tensor.max().item()}, "
              f"EOS count={(tensor == dataset.eos_id).sum().item()}")
    print("Causal shift: PASS")

    # Independent reference: concatenate the tail and head of the first two
    # shards. Shorten only the boundary check if running against tiny fixtures.
    if len(dataset.paths) < 2:
        print("Boundary check: SKIPPED (this split has only one shard)")
        return
    left = np.load(dataset.paths[0], mmap_mode="r", allow_pickle=False)
    right = np.load(dataset.paths[1], mmap_mode="r", allow_pickle=False)
    count = min(args.seq_len + 1, len(left) + len(right))
    tail_count = min(len(left), max(1, count // 2))
    tail_count = max(tail_count, count - len(right))
    start = len(left) - tail_count
    expected = np.concatenate((left[-tail_count:], right[:count - tail_count])).astype(np.int64)
    bx, by = dataset.get_sequence(start, count - 1)
    np.testing.assert_array_equal(bx.numpy(), expected[:-1])
    np.testing.assert_array_equal(by.numpy(), expected[1:])
    print(f"Shard boundary: PASS ({tail_count} tail + {count - tail_count} head tokens)")


if __name__ == "__main__":
    main()
