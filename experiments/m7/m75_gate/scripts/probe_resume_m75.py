#!/usr/bin/env python3
"""M7.5 no-update, real-format-7 checkpoint resume gate for the pinned H100 runtime.

Constructs the actual Kimi K3, Muon+AdamW optimizer and ShardedTokenDataset
loaders; calls the repository's load_checkpoint, then verifies restored state.
Never calls train(), optimizer.step(), save_checkpoint() or starts a run.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source(reference, manifest_path):
    expected = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    bad = []
    for rel, wanted in expected.items():
        path = reference / rel
        if not path.is_file() or sha256(path) != wanted:
            bad.append(rel)
    if bad:
        raise AssertionError("Remote source differs from the M7.3 snapshot: " + ", ".join(bad))
    return len(expected)


def main(args):
    # No write access to reference, data, source checkpoint or pilot artifacts.
    reference = args.reference.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_count = check_source(reference, args.source_manifest)
    if checkpoint.stat().st_size != 2_485_700_423:
        raise AssertionError("Unexpected checkpoint size")
    checksum_before = sha256(checkpoint)
    required_sha = "cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12"
    if checksum_before != required_sha:
        raise AssertionError("Checkpoint SHA-256 mismatch; aborting")
    sys.path.insert(0, str(reference))
    import numpy as np
    import torch
    from src.kimi_k3.data import ShardedTokenDataset, TokenDataLoader
    from src.kimi_k3.model import KimiK3, KimiK3Config
    from src.kimi_k3.training.trainer import (
        TrainConfig, build_optimizer, learning_rate_at_step, load_checkpoint,
        _runtime_signature,
    )
    from src.kimi_k3.utils import load_config

    torch.set_num_threads(8)  # M7.3's recorded runtime signature.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.are_deterministic_algorithms_enabled():
        raise AssertionError("M7.3 used deterministic_algorithms=False")
    if str(torch.__version__) != "2.11.0+cu128":
        raise AssertionError(f"Different torch version: {torch.__version__}")
    if not torch.cuda.is_available():
        raise AssertionError("The real resume probe requires the qualified H100 CUDA runtime")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.name != "NVIDIA H100 NVL":
        raise AssertionError(f"Unqualified GPU: {properties.name}")

    pilot = load_config(args.pilot_config, TrainConfig)
    config = load_config(args.m8_config, TrainConfig)
    allowed = {"max_steps", "log_interval", "eval_interval", "eval_batches", "checkpoint_interval"}
    changed = {key for key in vars(pilot) if getattr(pilot, key) != getattr(config, key)}
    if changed - allowed or config.max_steps != 183106 or pilot.max_steps != 2442:
        raise AssertionError(f"Unauthorized recipe change: {sorted(changed)}")
    if (config.batch_size, config.seq_len, config.gradient_accumulation_steps, config.warmup_steps,
        config.decay_steps, config.precision, config.learning_rate) != (32, 1024, 1, 1832, 183106, "bf16", 0.003):
        raise AssertionError("Unexpected M8 learning contract")

    # Real loaders (all dataset headers are inspected, just like scripts/train.py).
    train_data = ShardedTokenDataset(args.data_dir, split="train")
    val_data = ShardedTokenDataset(args.data_dir, split="val")
    if (train_data.total_tokens, val_data.total_tokens) != (6_000_001_675, 316_623_851):
        raise AssertionError("Unexpected corpus token counts")
    expected_metadata = "06709535cc3067f61fc1b76847f7d4b3798dbfe3bb5b4cde3dc1755832fd98a9"
    if train_data.metadata_sha256 != expected_metadata or val_data.metadata_sha256 != expected_metadata:
        raise AssertionError("Training dataset metadata differs from M7.3")
    train_loader = TokenDataLoader(train_data, config.batch_size, config.seq_len)
    val_loader = TokenDataLoader(val_data, config.batch_size, config.seq_len)

    model_config = load_config(args.model_config, KimiK3Config)
    if model_config.vocab_size != train_data.vocab_size:
        raise AssertionError("model/data vocab mismatch")
    # Match the standard CLI construction and order; load_checkpoint restores RNG last.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = KimiK3(model_config).to(device)
    optimizer = build_optimizer(model, config)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 294_676_808:
        raise AssertionError(f"Different parameter count: {parameter_count}")

    # mmap limits extra CPU RAM during evidence comparison. load_checkpoint itself
    # uses the official weights_only=True path without modification.
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if saved["format_version"] != 7:
        raise AssertionError("Unexpected checkpoint format")
    if saved["runtime"] != _runtime_signature(model):
        raise AssertionError("Runtime signature mismatch before calling load_checkpoint")

    state = load_checkpoint(checkpoint, model, optimizer, config,
                            vocab_size=train_data.vocab_size,
                            train_loader=train_loader, val_loader=val_loader)
    assert state.step == 2442 and state.tokens_seen == 80_019_456
    assert state.skipped_updates == 0
    assert train_loader.position == saved["loaders"]["train"]["position"] == 80_019_456
    assert val_loader.position == saved["loaders"]["val"]["position"] == 262_144
    assert train_loader.state_dict() == saved["loaders"]["train"]
    assert val_loader.state_dict() == saved["loaders"]["val"]
    assert random.getstate() == saved["rng_state"]["python"]
    restored_numpy = np.random.get_state()
    original_numpy = saved["rng_state"]["numpy"]
    assert restored_numpy[0] == original_numpy[0]
    assert np.array_equal(restored_numpy[1], np.asarray(original_numpy[1], dtype=np.uint32))
    assert tuple(restored_numpy[2:]) == tuple(original_numpy[2:])
    assert torch.equal(torch.get_rng_state(), saved["rng_state"]["torch_cpu"])
    assert torch.equal(torch.cuda.get_rng_state(device).cpu(), saved["rng_state"]["torch_cuda"].cpu())

    assert math.isclose(config.learning_rate, saved["train_config"]["learning_rate"], rel_tol=0, abs_tol=0)
    saved_groups = saved["optimizer"]["param_groups"]
    assert len(saved_groups) == len(optimizer.param_groups)
    for restored, previous in zip(optimizer.param_groups, saved_groups):
        assert restored["lr"] == previous["lr"]
    assert len(optimizer.state_dict()["state"]) == len(saved["optimizer"]["state"])
    # No train cursor mutation: read the next batch directly from the dataset.
    x, y = train_data.get_batch(train_loader.position, config.batch_size, config.seq_len)
    assert x.shape == y.shape == (32, 1024)
    assert train_loader.position == 80_019_456
    next_step = state.step + 1
    next_lr = learning_rate_at_step(config, next_step)
    if not math.isclose(next_lr, 0.0029999243148206077, abs_tol=1e-16, rel_tol=0):
        raise AssertionError("First resumed LR changed")
    checksum_after = sha256(checkpoint)
    if checksum_after != checksum_before:
        raise AssertionError("Source checkpoint changed during read-only gate")

    return {
        "status": "PASS", "no_optimizer_updates_executed": True,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256_before": checksum_before,
        "checkpoint_sha256_after": checksum_after,
        "source_files_sha256_verified": source_count,
        "torch": str(torch.__version__), "gpu": properties.name,
        "source_snapshot": str(reference), "model_parameters": parameter_count,
        "original_step": state.step, "original_tokens_seen": state.tokens_seen,
        "original_skipped_updates": state.skipped_updates,
        "next_step": next_step, "next_lr": next_lr,
        "restored_train_position": train_loader.position,
        "restored_val_position": val_loader.position,
        "optimizer_state_entries": len(optimizer.state_dict()["state"]),
        "optimizer_groups": len(optimizer.param_groups),
        "rng_verified": ["python", "numpy", "torch_cpu", "torch_cuda"],
        "total_steps": config.max_steps, "remaining_updates": config.max_steps - state.step,
        "remaining_tokens": (config.max_steps - state.step) * 32768,
        "m8_recipe_sha256": sha256(args.m8_config),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=Path("/workspace/kimi/reference"))
    p.add_argument("--checkpoint", type=Path, default=Path(
        "/workspace/kimi/runs/m73/m73_pilot_80m_v1_20261014T065121Z/last.pt"))
    p.add_argument("--model-config", type=Path, default=Path(
        "/workspace/kimi/reference/configs/model/kimi_k3_300m.yaml"))
    p.add_argument("--data-dir", type=Path, default=Path("/workspace/kimi/data/full"))
    base = Path(__file__).resolve().parents[1]
    p.add_argument("--source-manifest", type=Path, default=base / "evidence/m73_source_manifest.json")
    p.add_argument("--pilot-config", type=Path, default=base / "recipe/m73_pilot_original.yaml")
    p.add_argument("--m8-config", type=Path, default=base / "recipe/m8_full_6b.yaml")
    p.add_argument("--receipt", type=Path, default=Path("/workspace/kimi/results/m75/m75_resume_receipt.json"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    status = {"status": "FAIL"}
    try:
        status = main(args)
    except Exception as exc:
        status.update(error=str(exc), traceback=traceback.format_exc())
    status["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    # Only the dedicated M7.5 results directory is written.
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    sys.exit(0 if status["status"] == "PASS" else 1)
