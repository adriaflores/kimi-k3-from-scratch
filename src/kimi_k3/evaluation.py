"""Exploratory next-token evaluation over an existing prepared token dataset.

This is NOT the original pre-registered 64-window evaluation, and does not reproduce its
reported metrics or offer its pre-registered exclusion/identity guarantees.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kimi_k3.inference import inference_autocast


def evaluate_windows(
    model, dataset, *, device: torch.device, seq_len: int,
    num_windows: int, start_offset: int = 0, stride: int | None = None,
    precision: str = "fp32",
) -> dict:
    if seq_len <= 0 or seq_len > model.config.max_seq_len or num_windows <= 0:
        raise ValueError("Invalid seq_len or num_windows")
    if start_offset < 0:
        raise ValueError("start_offset must be >= 0")
    if model.config.kda_implementation == "chunkwise" and "kda" in model.config.layer_types:
        if seq_len % model.config.kda_chunk_size:
            raise ValueError("Evaluation seq_len must be divisible by KDA chunk size")
    stride = seq_len if stride is None else stride
    if stride < seq_len:
        raise ValueError("stride must be >= seq_len to avoid overlapping scored windows")
    last_end = start_offset + (num_windows - 1) * stride + seq_len + 1
    if last_end > dataset.total_tokens:
        raise ValueError("Requested windows exceed the token stream")
    if model.config.vocab_size != dataset.vocab_size:
        raise ValueError("Checkpoint and dataset vocabulary sizes differ")

    model.eval()
    nll_sum = 0.0
    correct = 0
    n = 0
    with torch.inference_mode():
        for i in range(num_windows):
            offset = start_offset + i * stride
            x, y = dataset.get_sequence(offset, seq_len)
            x, y = x.unsqueeze(0).to(device), y.to(device)
            with inference_autocast(device, precision):
                logits, _ = model(x)
            if logits.shape != (1, seq_len, model.config.vocab_size):
                raise ValueError("Unexpected model logits shape")
            logits = logits[0].float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite logits")
            loss = F.cross_entropy(logits, y, reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite NLL")
            nll_sum += float(loss.item())
            correct += int((logits.argmax(-1) == y).sum().item())
            n += seq_len
    mean_nll = nll_sum / n
    return {
        "kind": "exploratory_contiguous_val_windows",
        "split": dataset.split,
        "dataset_metadata_sha256": dataset.metadata_sha256,
        "start_offset": start_offset,
        "stride": stride,
        "sequence_length": seq_len,
        "windows": num_windows,
        "scored_tokens": n,
        "nll_sum": nll_sum,
        "mean_nll_nats_per_token": mean_nll,
        "perplexity_same_tokenizer": math.exp(mean_nll),
        "top1_accuracy": correct / n,
        "precision": precision,
        "note": "Exploratory evaluation; not the pre-registered held-out protocol.",
    }
