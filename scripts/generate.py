"""Generate text or token IDs from any compatible trainer-format Kimi K3 checkpoint.

For an entirely offline smoke test, provide --token-ids instead of --prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kimi_k3.inference import generate_tokens, load_checkpoint_model, resolve_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Text prompt, tokenized with SmolLM2")
    source.add_argument("--token-ids", help="Comma-separated raw token IDs (offline)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--backend", choices=("chunkwise", "fused"), default="chunkwise")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eos-id", type=int, default=None,
                   help="Override EOS token ID; defaults to tokenizer EOS or 0 for raw IDs")
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--tokenizer-revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 requires CUDA")
    tokenizer = None
    if args.token_ids is not None:
        try:
            ids = [int(x.strip()) for x in args.token_ids.split(",")]
        except ValueError as exc:
            raise ValueError("--token-ids must be comma-separated integers") from exc
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.tokenizer_revision,
            use_fast=True, local_files_only=args.local_files_only,
        )
        ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    model, metadata = load_checkpoint_model(args.checkpoint, device=device, backend=args.backend)
    if tokenizer is not None and len(tokenizer) != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match checkpoint")
    eos_id = args.eos_id if args.eos_id is not None else (
        tokenizer.eos_token_id if tokenizer is not None and tokenizer.eos_token_id is not None else 0
    )
    generated = generate_tokens(
        model, ids, device=device, max_new_tokens=args.max_new_tokens,
        mode=args.mode, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, seed=args.seed, precision=args.precision, eos_id=eos_id,
    )
    result = {
        "checkpoint": metadata,
        "device": str(device),
        "precision": args.precision,
        "prompt_token_ids": ids,
        "generated_token_ids": generated,
    }
    if tokenizer is not None:
        result["generated_text"] = tokenizer.decode(generated, skip_special_tokens=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
