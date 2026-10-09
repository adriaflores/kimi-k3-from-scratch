#!/usr/bin/env python3
"""Full-shape proxy plumbing check on synthetic tokens, not a training study.

Runs one accumulated update, checkpoint/restore, a second update and a
QB-frozen evaluation through the existing trainer. CPU uses chunkwise KDA;
CUDA keeps the YAML's backend. Parameters, vocabulary and depth are unchanged.
Temporary checkpoints are removed; an optional JSON retains scalar evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from src.kimi_k3.model import KimiK3, KimiK3Config
from src.kimi_k3.training.metrics import MetricsWriter
from src.kimi_k3.training.qk_clip import qk_clip_metadata
from src.kimi_k3.training.run import start_run
from src.kimi_k3.training.trainer import (
    TrainConfig,
    TrainState,
    build_optimizer,
    evaluate,
    load_checkpoint,
    train,
    validate_precision,
)
from src.kimi_k3.utils import load_config


def summarize_update(event: dict) -> dict:
    """Retain relevant scalar evidence, without CPU timing or large QB arrays."""
    fields = ("step", "tokens_seen", "train_loss", "lr", "grad_norm_mean", "grad_norm_max",
              "clipped_fraction", "nonfinite_grad_norms", "skipped_updates")
    result = {name: event[name] for name in fields}
    result["qk_clip"] = {name: value for name, value in event["qk_clip"].items() if name != "layers"}
    result["moe"] = {
        name: {"tokens": layer["tokens"], "assignments": sum(layer["expert_counts"]),
               "max_load_over_mean": layer["max_load_over_mean"],
               "zero_load_experts": layer["zero_load_experts"],
               "qb_bias_abs_max": max(abs(value) for value in layer["qb_biases"])}
        for name, layer in event["moe"].items()
    }
    return result


def check(path: Path, args: argparse.Namespace) -> dict:
    configured = load_config(path, KimiK3Config)
    effective = replace(configured, kda_implementation="chunkwise") if args.device == "cpu" else configured
    if effective.kda_implementation == "chunkwise" and args.seq_len % effective.kda_chunk_size:
        raise ValueError("CPU/reference sequence length must be divisible by kda_chunk_size")
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = TrainConfig(
        batch_size=args.batch_size, seq_len=args.seq_len,
        gradient_accumulation_steps=args.accumulation,
        max_steps=1, log_interval=1, eval_interval=1, eval_batches=1,
        checkpoint_interval=1, learning_rate=3e-4, weight_decay=0.1,
        max_grad_norm=1.0, precision=args.precision,
        muon_batched=True, muon_compile=args.muon_compile,
        lr_schedule="cosine", decay_steps=2, warmup_steps=0, seed=seed,
    )
    device = torch.device(args.device)
    validate_precision(device, config.precision)
    model = KimiK3(effective).to(device)
    optimizer = build_optimizer(model, config)
    all_parameters = list(model.parameters())
    grouped_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len(grouped_parameters) != len(all_parameters) or {id(p) for p in grouped_parameters} != {id(p) for p in all_parameters}:
        raise RuntimeError("Optimizer parameters are missing or duplicated")
    initial_embedding_sample = model.embed_tokens.weight[0, :8].detach().clone()
    biases = [layer.ffn.quantile_balancer.biases for layer in model.layers if hasattr(layer.ffn, "quantile_balancer")]
    with tempfile.TemporaryDirectory(prefix="k3_proxy_check_") as temporary:
        root = Path(temporary)
        data = root / "data"
        data.mkdir()
        rng = np.random.default_rng(seed)
        tokens = 2 * args.accumulation * args.batch_size * args.seq_len + 1
        metadata = {"format_version": 2, "storage": {"format": "npy", "dtype": "uint16"},
                    "tokenizer": {"vocab_size": effective.vocab_size, "eos_id": 0}, "splits": {}}
        for split in ("train", "val"):
            filename = split + "_000.npy"
            np.save(data / filename, rng.integers(0, effective.vocab_size, size=tokens, dtype=np.uint16))
            metadata["splits"][split] = {"tokens": tokens, "shards": [{"file": filename, "tokens": tokens}]}
        (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        train_loader, val_loader = (
            TokenDataLoader(ShardedTokenDataset(data, split), args.batch_size, args.seq_len)
            for split in ("train", "val")
        )
        run = start_run(model=model, optimizer=optimizer, config=config, state=TrainState(),
                        train_loader=train_loader, val_loader=val_loader,
                        repo_root=REPO_ROOT, run_dir=root / "initial")
        with MetricsWriter(run) as metrics:
            state = train(model, optimizer, train_loader, val_loader, config, device,
                          checkpoint_path=run.checkpoint_path, metrics=metrics)
        saved_sample = model.embed_tokens.weight[0, :8].detach().clone()
        saved_biases = [bias.detach().clone() for bias in biases]
        saved_rng = torch.get_rng_state().clone()
        saved_cursor = train_loader.position
        with torch.no_grad():
            model.embed_tokens.weight[0, :8].add_(1)
            for bias in biases:
                bias.add_(1)
        train_loader.reset()
        torch.manual_seed(999)
        config = replace(config, max_steps=2)
        state = load_checkpoint(run.checkpoint_path, model, optimizer, config,
                                vocab_size=effective.vocab_size,
                                train_loader=train_loader, val_loader=val_loader)
        torch.testing.assert_close(model.embed_tokens.weight[0, :8], saved_sample, rtol=0, atol=0)
        for bias, expected in zip(biases, saved_biases, strict=True):
            torch.testing.assert_close(bias, expected, rtol=0, atol=0)
        if state.step != 1 or train_loader.position != saved_cursor or not torch.equal(torch.get_rng_state(), saved_rng):
            raise RuntimeError("Checkpoint did not restore step, data cursor and RNG")
        resumed = start_run(model=model, optimizer=optimizer, config=config, state=state,
                            train_loader=train_loader, val_loader=val_loader,
                            repo_root=REPO_ROOT, run_dir=root / "resumed", resume_path=run.checkpoint_path)
        with MetricsWriter(resumed) as metrics:
            state = train(model, optimizer, train_loader, val_loader, config, device,
                          state=state, checkpoint_path=resumed.checkpoint_path, metrics=metrics)
        expected_tokens = 2 * args.batch_size * args.seq_len * args.accumulation
        if state.step != 2 or state.tokens_seen != expected_tokens or state.skipped_updates != 0 or train_loader.position != expected_tokens:
            raise RuntimeError("Successful updates/tokens/cursor mismatch")
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise RuntimeError("Non-finite parameters")
        if torch.equal(initial_embedding_sample, model.embed_tokens.weight[0, :8]):
            raise RuntimeError("Embedding weights were not updated")
        frozen_biases = [bias.detach().clone() for bias in biases]
        val_loss = evaluate(model, val_loader, device, 1, precision=args.precision)
        for bias, expected in zip(biases, frozen_biases, strict=True):
            torch.testing.assert_close(bias, expected, rtol=0, atol=0)
        if not math.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss")
        events = [json.loads(line) for directory in (run.directory, resumed.directory)
                  for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        updates = [event for event in events if event["event"] == "train"]
        heads = configured.layer_types.count("mla") * configured.mla_num_heads
        if len(updates) != 2 or any(event["qk_clip"]["observed_head_updates"] != heads for event in updates):
            raise RuntimeError("QK-Clip did not observe every MLA head")
        tokens_per_update = args.batch_size * args.seq_len * args.accumulation
        moe_layers = configured.num_layers - configured.num_dense_layers
        for event in updates:
            if len(event["moe"]) != moe_layers or any(
                layer["tokens"] != tokens_per_update
                or sum(layer["expert_counts"]) != tokens_per_update * configured.num_experts_per_token
                or not all(math.isfinite(bias) for bias in layer["qb_biases"])
                for layer in event["moe"].values()
            ):
                raise RuntimeError("MoE counters or QB biases are inconsistent")
        report = {
            "config": path.as_posix(), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "pass", "device": str(device), "precision": args.precision,
            "configured_backend": configured.kda_implementation,
            "effective_model_config": asdict(effective), "effective_train_config": asdict(config),
            "synthetic_data": True, "full_vocabulary": effective.vocab_size,
            "steps": state.step, "tokens_seen": state.tokens_seen,
            "skipped_updates": state.skipped_updates, "qk_clip_heads_per_update": heads,
            "checkpoint_restore": "weights/QB/step/data cursor/RNG restored; continued successfully",
            "evaluation_qb_frozen": True, "parameters_finite": True,
            "moe_counter_consistency": True,
            "validation_loss_synthetic": val_loss,
            "qk_clip_geometry": qk_clip_metadata(model, threshold=100.0, chunk_size=256),
            "update_metrics": [summarize_update(event) for event in updates],
        }
    del model, optimizer, all_parameters, grouped_parameters, biases
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2, help="CPU intra-op threads")
    parser.add_argument("--muon-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    results = []
    for path in args.configs:
        print(f"Checking {path} on {args.device} ({args.precision})", flush=True)
        results.append(check(path, args))
        print(f"PASS: {path.name}", flush=True)
    report = {"schema_version": 1, "torch_version": str(torch.__version__),
              "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(),
              "purpose": "full-shape plumbing check; synthetic losses are not research results",
              "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
