"""General read-only inference utilities for trainer-format Kimi K3 checkpoints.

Unlike the original pre-registered research evaluation, this module is checkpoint-agnostic.
It does not load optimizer state or assert a particular model/checkpoint hash.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch

from kimi_k3.model import KimiK3, KimiK3Config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def inference_autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("precision must be fp32 or bf16")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 inference requires a supported CUDA device")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def load_checkpoint_model(
    checkpoint_path: str | Path, *, device: torch.device, backend: str = "chunkwise"
) -> tuple[KimiK3, dict]:
    """Load only model/config from a trusted local trainer-format checkpoint.

    ``meta`` initialization and assign=True avoid first allocating a redundant
    random model in host RAM. Re-tie embeddings after assignment; otherwise the
    tied embedding and vocabulary-head Parameter objects may diverge.
    """
    if backend not in ("chunkwise", "fused"):
        raise ValueError("backend must be chunkwise or fused")
    if backend == "fused" and device.type != "cuda":
        raise ValueError("The fused KDA backend requires CUDA")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("format_version") != 7:
        raise ValueError("Expected a format-version-7 trainer checkpoint")
    config_data = checkpoint.get("model_config")
    weights = checkpoint.get("model")
    if not isinstance(config_data, dict) or not isinstance(weights, dict):
        raise ValueError("Checkpoint must contain model_config and model state_dict")
    original = KimiK3Config(**config_data)
    effective = KimiK3Config(**{**config_data, "kda_implementation": backend})
    with torch.device("meta"):
        model = KimiK3(effective)
    model.load_state_dict(weights, strict=True, assign=True)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Unmaterialized model parameter after strict checkpoint load")
    model.lm_head.weight = model.embed_tokens.weight
    model.eval()
    model = model.to(device)
    metadata = {
        "checkpoint": str(path),
        "format_version": checkpoint["format_version"],
        "step": checkpoint.get("step"),
        "tokens_seen": checkpoint.get("tokens_seen"),
        "vocab_size": original.vocab_size,
        "backend_trained": original.kda_implementation,
        "backend_inference": backend,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del checkpoint, weights
    return model, metadata


def next_token_logits(
    model: KimiK3, token_ids: list[int], *, device: torch.device,
    precision: str = "fp32", pad_id: int = 0,
) -> torch.Tensor:
    """Score the last real token; any right-padding is strictly future context.

    Reference chunkwise KDA requires a full chunk. Pad only when necessary,
    without scoring fabricated suffix tokens. No implicit context truncation.
    """
    config = model.config
    if not token_ids or len(token_ids) > config.max_seq_len:
        raise ValueError("Prompt must contain 1..max_seq_len tokens")
    if any(type(token) is not int or not 0 <= token < config.vocab_size for token in token_ids):
        raise ValueError("Invalid prompt token ID")
    if not 0 <= pad_id < config.vocab_size:
        raise ValueError("pad_id outside vocabulary")
    pad_count = 0
    if config.kda_implementation == "chunkwise" and "kda" in config.layer_types:
        pad_count = -len(token_ids) % config.kda_chunk_size
    if len(token_ids) + pad_count > config.max_seq_len:
        raise ValueError("Chunk padding would exceed model max_seq_len")
    ids = token_ids + [pad_id] * pad_count
    x = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.inference_mode(), inference_autocast(device, precision):
        logits, _ = model(x)
        last = logits[0, len(token_ids) - 1].float()
    if last.shape != (config.vocab_size,) or not bool(torch.isfinite(last).all()):
        raise ValueError("Invalid or nonfinite next-token logits")
    return last


def select_token(logits: torch.Tensor, *, mode: str, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.95,
                 generator: torch.Generator | None = None) -> int:
    if mode == "greedy":
        return int(logits.argmax().item())
    if mode != "sample" or temperature <= 0 or top_k <= 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling arguments")
    # CPU sampling makes the seeded generator work independently of the model device.
    scores = (logits.detach().float() / temperature).cpu()
    values, indices = scores.topk(min(top_k, scores.numel()))
    probs = values.softmax(-1)
    # Keep the minimal leading prefix whose cumulative probability reaches top_p.
    excluded = (probs.cumsum(0) - probs) >= top_p
    probs = probs.masked_fill(excluded, 0)
    probs = probs / probs.sum()
    return int(indices[torch.multinomial(probs, 1, generator=generator)].item())


def generate_tokens(
    model: KimiK3, prompt_ids: list[int], *, device: torch.device,
    max_new_tokens: int = 32, mode: str = "greedy", temperature: float = 1.0,
    top_k: int = 50, top_p: float = 0.95, seed: int = 1337,
    eos_id: int = 0, precision: str = "fp32",
) -> list[int]:
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if len(prompt_ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("Generation would exceed max_seq_len; no sliding window is implemented")
    if mode not in ("greedy", "sample"):
        raise ValueError("mode must be greedy or sample")
    if not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("eos_id outside vocabulary")
    if mode == "sample" and (temperature <= 0 or top_k < 1 or not 0 < top_p <= 1):
        raise ValueError("Invalid sampling arguments")
    rng = torch.Generator(device="cpu").manual_seed(seed) if mode == "sample" else None
    seq = list(prompt_ids)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = next_token_logits(model, seq, device=device, precision=precision, pad_id=eos_id)
        token = select_token(logits, mode=mode, temperature=temperature, top_k=top_k,
                             top_p=top_p, generator=rng)
        seq.append(token)
        new_ids.append(token)
        if token == eos_id:
            break
    return new_ids
