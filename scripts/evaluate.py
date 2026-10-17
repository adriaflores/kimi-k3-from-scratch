"""Exploratory next-token NLL/PPL on prepared train/validation token shards.

Not a reproduction of the frozen M9 held-out evaluation protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kimi_k3.data.dataset import ShardedTokenDataset
from src.kimi_k3.evaluation import evaluate_windows
from src.kimi_k3.inference import load_checkpoint_model, resolve_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--backend", choices=("chunkwise", "fused"), default="chunkwise")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-windows", type=int, default=4)
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--output", type=Path, help="Optional location for a JSON report")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    dataset = ShardedTokenDataset(args.data_dir, split=args.split)
    model, metadata = load_checkpoint_model(args.checkpoint, device=device, backend=args.backend)
    summary = evaluate_windows(
        model, dataset, device=device, seq_len=args.seq_len,
        num_windows=args.num_windows, start_offset=args.start_offset,
        stride=args.stride, precision=args.precision,
    )
    result = {"checkpoint": metadata, "device": str(device), "summary": summary}
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
