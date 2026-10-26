
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

out = args.output
out.mkdir(parents=True, exist_ok=True)

# Load the final training checkpoint
ckpt = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=True,
    mmap=True,
)

assert ckpt["format_version"] == 7
assert ckpt["step"] == 183106

# Export inference weights, without optimizer state.
# Clone tensors to handle tied-weight aliases safely.
weights = {
    k: v.detach().cpu().contiguous().clone()
    for k, v in ckpt["model"].items()
}
save_file(weights, str(out / "model.safetensors"))

# Preserve the original architecture configuration
config = ckpt["model_config"]
(out / "config.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

# Export the exact tokenizer revision
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
)
assert len(tokenizer) == config["vocab_size"]
tokenizer.save_pretrained(out)

print("Export complete:", out)
