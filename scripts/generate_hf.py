"""Run greedy inference with the published Kimi K3 PyTorch weights.

Run from the kimi-k3-from-scratch repository root:
    uv run --with safetensors python scripts/generate_hf.py \
      --source C:/datasets/kimi_k3_hf_validation \
      --prompt "The capital of France is" --max-new-tokens 4

The --source argument also accepts a Hugging Face model repository ID.
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from kimi_k3.inference import generate_tokens
from kimi_k3.model import KimiK3, KimiK3Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="adriaflores/kimi-k3-300m-from-scratch",
        help="HF model repo ID or an existing local model directory",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    model_dir = Path(args.source)
    if not model_dir.is_dir():
        model_dir = Path(snapshot_download(repo_id=args.source))

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config_data["kda_implementation"] = "chunkwise"
    config = KimiK3Config(**config_data)
    if len(tokenizer) != config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match model config")

    # Build without allocating randomly initialized weights in CPU RAM.
    with torch.device("meta"):
        model = KimiK3(config)
    weights = load_file(str(model_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True, assign=True)
    model.lm_head.weight = model.embed_tokens.weight
    model = model.eval()

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    token_ids = generate_tokens(
        model,
        prompt_ids,
        device=torch.device("cpu"),
        max_new_tokens=args.max_new_tokens,
        mode="greedy",
        eos_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
        precision="fp32",
    )
    print("Prompt:", repr(args.prompt))
    print("Completion:", repr(tokenizer.decode(token_ids, skip_special_tokens=True)))
    print("Generated token IDs:", token_ids)


if __name__ == "__main__":
    main()
