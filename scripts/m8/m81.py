#!/usr/bin/env python3
"""M8.1 fail-closed preflight. Local contract + optional read-only remote environment gate.

No training, writes only --receipt (atomically); never rewrites weights, source or data.
Remote stage is deliberately postponed until the single planned Vast activation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

M8_RECIPE_SHA = "2917190c3db675d18f0e439696e77222c87a8e577c5eba96aadaaf4708516ca6"
SOURCE_CHECKPOINT_SHA = "cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12"
SOURCE_CHECKPOINT_BYTES = 2_485_700_423
SOURCE_CHECKPOINT_POSIX = "/workspace/kimi/runs/m73/m73_pilot_80m_v1_20261014T065121Z/last.pt"
# Path is used for filesystem access on Vast (Linux). Evidence comparison
# must use the immutable POSIX string, even when run from Windows Git Bash.
SOURCE_CHECKPOINT = Path(SOURCE_CHECKPOINT_POSIX)
META_SHA = "15d7fedd033e919aceb76f79b002935bd7cb2e0dcee3ec598f46b5f42fd2805f"
META_CANON_SHA = "06709535cc3067f61fc1b76847f7d4b3798dbfe3bb5b4cde3dc1755832fd98a9"
NEXT_LR = 0.0029999243148206077
TOKENS_PER_STEP = 32768
SOURCE_STEP = 2442
TARGET_STEP = 183106
EXPECTED_YAML = {
    "batch_size": "32", "seq_len": "1024", "learning_rate": "0.003", "weight_decay": "0.1",
    "max_steps": "183106", "log_interval": "64", "eval_interval": "512", "eval_batches": "8",
    "checkpoint_interval": "4096", "overfit_batch": "false", "gradient_accumulation_steps": "1",
    "precision": "bf16", "adam_beta1": "0.9", "adam_beta2": "0.95", "adam_eps": "1.0e-08",
    "adam_fused": "true", "muon_momentum": "0.95", "muon_ns_steps": "5",
    "muon_ns_dtype": "bf16", "muon_compile": "true", "muon_batched": "true",
    "muon_batch_size": "32", "muon_batch_max_mb": "64", "qk_clip_threshold": "100.0",
    "qk_clip_chunk_size": "256", "lr_schedule": "cosine", "warmup_steps": "1832",
    "decay_steps": "183106", "min_lr_ratio": "0.1", "max_grad_norm": "1.0", "seed": "1337",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flat_recipe(path: Path) -> dict[str, str]:
    """Fail closed on this exact flat scalar-only TrainConfig YAML format."""
    data = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        key, colon, value = stripped.partition(":")
        key, value = key.strip(), value.strip()
        if not colon or not key.isidentifier() or not value or key in data:
            raise ValueError(f"Invalid or duplicate YAML key at {path}:{line_no}")
        if any(c in value for c in "{}[]&*!|>"):
            raise ValueError("Only scalar recipe values supported")
        data[key] = value
    return data


def validate_recipe(path: Path) -> dict:
    if sha256(path) != M8_RECIPE_SHA:
        raise ValueError("M8 recipe SHA-256 mismatch; refuse altered bytes")
    recipe = flat_recipe(path)
    if recipe != EXPECTED_YAML:
        raise ValueError("M8 recipe differs from pinned M7.5 contract")
    assert len(recipe) == 31
    tps = int(recipe["batch_size"]) * int(recipe["seq_len"]) * int(recipe["gradient_accumulation_steps"])
    if tps != TOKENS_PER_STEP or int(recipe["max_steps"]) != TARGET_STEP:
        raise ValueError("Invalid token arithmetic")
    return {
        "sha256": M8_RECIPE_SHA, "tokens_per_update": tps, "source_step": SOURCE_STEP,
        "first_resumed_step": SOURCE_STEP + 1, "target_global_step": TARGET_STEP,
        "remaining_updates": TARGET_STEP - SOURCE_STEP,
        "remaining_tokens": (TARGET_STEP - SOURCE_STEP) * tps,
        "nominal_total_tokens": TARGET_STEP * tps,
        "first_periodic_checkpoint_step": 4096,
        "first_periodic_checkpoint_updates_from_resume": 4096 - SOURCE_STEP,
        "dataset_tail_dropped_tokens": 17035,
        "last_batch_restart_tokens": tps,
    }


def validate_m75_receipt(path: Path) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "status": "PASS", "no_optimizer_updates_executed": True,
        "checkpoint_path": SOURCE_CHECKPOINT_POSIX,
        "checkpoint_sha256_before": SOURCE_CHECKPOINT_SHA,
        "checkpoint_sha256_after": SOURCE_CHECKPOINT_SHA,
        "source_files_sha256_verified": 35, "gpu": "NVIDIA H100 NVL", "torch": "2.11.0+cu128",
        "source_snapshot": "/workspace/kimi/reference", "model_parameters": 294676808,
        "original_step": SOURCE_STEP, "original_tokens_seen": 80019456,
        "original_skipped_updates": 0, "next_step": SOURCE_STEP + 1,
        "restored_train_position": 80019456, "restored_val_position": 262144,
        "optimizer_state_entries": 1830, "optimizer_groups": 9,
        "total_steps": TARGET_STEP, "remaining_updates": TARGET_STEP - SOURCE_STEP,
        "remaining_tokens": (TARGET_STEP - SOURCE_STEP) * TOKENS_PER_STEP,
        "m8_recipe_sha256": M8_RECIPE_SHA,
    }
    bad = [name for name, expected in exact.items()
           if type(receipt.get(name)) is not type(expected) or receipt.get(name) != expected]
    if bad:
        raise ValueError("M7.5 PASS receipt invalid: " + ", ".join(bad))
    if not math.isclose(receipt.get("next_lr", float("nan")), NEXT_LR, abs_tol=1e-16, rel_tol=0):
        raise ValueError("M7.5 expected resumed LR differs")
    if set(receipt.get("rng_verified", [])) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("M7.5 RNG completeness invalid")
    return {"status": "PASS", "original_step": SOURCE_STEP, "checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
            "optimizer_state_entries": 1830, "rng": receipt["rng_verified"]}


def check_source_manifest(reference: Path, manifest_path: Path, expected_count: int = 35) -> dict:
    """Must use the M7.3 frozen manifest. Never generate a baseline from current source."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or len(manifest) != expected_count:
        raise ValueError(f"Missing/incomplete original source manifest (expected {expected_count} entries)")
    if expected_count == 35 and not {"scripts/train.py", "src/kimi_k3/training/trainer.py",
                                     "configs/model/kimi_k3_300m.yaml"}.issubset(manifest):
        raise ValueError("Frozen manifest is missing required training files")
    root = reference.resolve(strict=True)
    invalid = []
    for rel, expected_hash in sorted(manifest.items()):
        if not isinstance(rel, str) or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            invalid.append(str(rel)); continue
        entry = root / rel
        if Path(rel).is_absolute() or not entry.resolve().is_relative_to(root) or not entry.is_file():
            invalid.append(rel); continue
        if sha256(entry) != expected_hash:
            invalid.append(rel)
    if invalid:
        raise ValueError("Training reference changed/missing: " + ", ".join(invalid[:15]))
    return {"source_file_count": len(manifest), "manifest_sha256": sha256(manifest_path)}


def check_cost(hourly_eur: float, cap_eur: float, projected_hours: float) -> dict:
    for name, value in (("price_per_hour_eur", hourly_eur), ("cost_cap_eur", cap_eur),
                        ("projected_hours", projected_hours)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be an explicitly supplied positive finite number")
    estimate = hourly_eur * projected_hours
    if estimate > cap_eur:
        raise ValueError(f"Estimated cost {estimate:.2f} EUR exceeds cap {cap_eur:.2f} EUR")
    return {"price_per_hour_eur_user_confirmed": hourly_eur, "cost_cap_eur": cap_eur,
            "projected_hours": projected_hours, "estimated_compute_eur_excluding_fees": round(estimate, 2),
            "note": "Price is user supplied, not a Vast API quotation; excludes fees/taxes/storage."}


def check_paths(remote_root: Path, reference: Path, data: Path, checkpoint: Path,
                proposed_run: Path, receipt: Path) -> dict:
    root = remote_root.resolve(strict=True)
    ref = reference.resolve(strict=True)
    data = data.resolve(strict=True)
    source = checkpoint.resolve(strict=True)
    run = proposed_run.resolve(strict=False)
    report = receipt.resolve(strict=False)
    if ref != root / "reference" or data != root / "data/full" or source != SOURCE_CHECKPOINT:
        raise ValueError("Unexpected reference, dataset, or source checkpoint path")
    if run == root / "runs/m8" or not run.is_relative_to(root / "runs/m8") or run.exists():
        raise ValueError("M8 run dir must be a NEW, non-existing child of /workspace/kimi/runs/m8")
    if report == root / "artifacts/m8/m81" or not report.is_relative_to(root / "artifacts/m8/m81"):
        raise ValueError("Receipt must be inside dedicated M8.1 artifact folder")
    if proposed_run == checkpoint or run / "last.pt" == source:
        raise ValueError("M8 output may never reuse M7.3 source checkpoint")
    return {"reference": str(ref), "dataset": str(data), "source_checkpoint": str(source),
            "proposed_run_dir": str(run), "receipt": str(report), "run_dir_does_not_exist": True}


def check_disk(path: Path, minimum_gib: float) -> dict:
    if not math.isfinite(minimum_gib) or minimum_gib < 20:
        raise ValueError("Minimum free disk must be at least 20 GiB")
    stat = shutil.disk_usage(path)
    if stat.free < minimum_gib * (1024 ** 3):
        raise ValueError(f"Free workspace {stat.free / (1024**3):.2f} GiB below {minimum_gib} GiB")
    return {"free_gib": round(stat.free / (1024 ** 3), 2), "required_gib": minimum_gib,
            "note": "M8.2 must verify staging/rotation/backups separately; headroom is a minimum."}


def check_checkpoint_header(checkpoint: Path, torch) -> dict:
    if checkpoint.stat().st_size != SOURCE_CHECKPOINT_BYTES:
        raise ValueError("M7.3 checkpoint byte size changed")
    if sha256(checkpoint) != SOURCE_CHECKPOINT_SHA:
        raise ValueError("M7.3 source checkpoint SHA-256 mismatch")
    # A header-level state test (mmap) not a re-run of M7.5 full load_checkpoint.
    record = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)
    if (record.get("format_version"), record.get("step"), record.get("tokens_seen"),
        record.get("skipped_updates")) != (7, SOURCE_STEP, 80019456, 0):
        raise ValueError("M7.3 checkpoint header differs")
    for key in ("model", "optimizer", "optimizer_layout", "loaders", "rng_state", "runtime", "qk_clip"):
        if key not in record:
            raise ValueError(f"Missing M7.3 resume key: {key}")
    if record["loaders"]["train"]["position"] != 80019456 or record["loaders"]["val"]["position"] != 262144:
        raise ValueError("M7.3 restored cursors differ")
    del record
    return {"sha256": SOURCE_CHECKPOINT_SHA, "bytes": SOURCE_CHECKPOINT_BYTES,
            "format_version": 7, "step": SOURCE_STEP, "tokens_seen": 80019456, "intact": True}


def check_runtime_and_corpus(reference: Path, data_path: Path) -> dict:
    # Import from M7.3 immutable reference rather than potentially drifted local source.
    sys.path.insert(0, str(reference))
    import torch
    if platform.python_version() != "3.12.13" or str(torch.__version__) != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise ValueError(f"Unqualified Python/torch/CUDA: {platform.python_version()}, {torch.__version__}, {torch.version.cuda}")
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA H100 NVL":
        raise ValueError("Qualified NVIDIA H100 NVL is unavailable")
    if torch.get_num_threads() != 8 or torch.are_deterministic_algorithms_enabled():
        raise ValueError("Thread/determinism profile differs from pilot")
    numerical = {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                 "matmul_precision": torch.get_float32_matmul_precision(),
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                 "cudnn_benchmark": torch.backends.cudnn.benchmark,
                 "cudnn_deterministic": torch.backends.cudnn.deterministic}
    expected = {"tf32_matmul": False, "matmul_precision": "highest", "cudnn_allow_tf32": True,
                "cudnn_benchmark": False, "cudnn_deterministic": False}
    if numerical != expected:
        raise ValueError("Numerical backend profile changed from M7.3")
    gpu_used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], text=True)
    if int(gpu_used.splitlines()[0].strip()) > 2048:
        raise ValueError("GPU already occupied by another workload (>2048 MiB)")
    if sha256(data_path / "metadata.json") != META_SHA:
        raise ValueError("Dataset metadata.json bytes differ")
    from src.kimi_k3.data import ShardedTokenDataset
    totals = {}
    for split, count, shards in (("train", 6_000_001_675, 241), ("val", 316_623_851, 13)):
        ds = ShardedTokenDataset(data_path, split=split)
        found = len(list(data_path.glob(split + "_*.npy")))
        if (ds.total_tokens, found, ds.metadata_sha256) != (count, shards, META_CANON_SHA):
            raise ValueError(f"Corpus split {split} tokens/shards/manifest differs")
        totals[split] = {"tokens": count, "shards": shards, "metadata_sha256": ds.metadata_sha256}
    return {"runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                        "num_threads": torch.get_num_threads(), "numerical": numerical},
            "corpus": totals, "torch_module": torch}


def write_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    # Refuse to overwrite an in-progress concurrent receipt.
    with tmp.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(tmp, path)


def execute(args) -> dict:
    root = args.repo_root.resolve()
    kit = root / "experiments/m8/m81"
    recipe = root / "experiments/m7/m75_gate/recipe/m8_full_6b.yaml"
    checks = {"recipe": validate_recipe(recipe),
              "m75_resumption_evidence": validate_m75_receipt(kit / "inputs/m75_resume_receipt.json")}
    if args.mode == "local":
        return {"stage": "local", "status": "PASS", "checks": checks, "gpu_accessed": False,
                "training_started": False, "remote_preflight_status": "PENDING"}
    if args.price_per_hour_eur is None or args.cost_cap_eur is None:
        raise ValueError("Remote preflight REQUIRES --price-per-hour-eur and --cost-cap-eur")
    checks["cost"] = check_cost(args.price_per_hour_eur, args.cost_cap_eur, args.projected_hours)
    checks["paths"] = check_paths(args.remote_root, args.reference, args.data_dir,
                                  args.checkpoint, args.proposed_run_dir, args.receipt)
    checks["storage"] = check_disk(args.remote_root, args.min_free_gib)
    checks["reference"] = check_source_manifest(args.reference, args.source_manifest)
    result = check_runtime_and_corpus(args.reference, args.data_dir)
    torch = result.pop("torch_module")
    checks.update(result)
    checks["checkpoint"] = check_checkpoint_header(args.checkpoint, torch)
    return {"stage": "remote", "status": "PASS", "checks": checks,
            "training_started": False, "next_phase": "M8.2/M8.3 operational GO gate (NOT cleared by M8.1 alone)"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("local", "remote"), required=True)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--remote-root", type=Path, default=Path("/workspace/kimi"))
    p.add_argument("--reference", type=Path, default=Path("/workspace/kimi/reference"))
    p.add_argument("--data-dir", type=Path, default=Path("/workspace/kimi/data/full"))
    p.add_argument("--checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    p.add_argument("--source-manifest", type=Path, default=Path("/workspace/kimi/experiments/m7/m75_gate/evidence/m73_source_manifest.json"))
    p.add_argument("--proposed-run-dir", type=Path, default=Path("/workspace/kimi/runs/m8/m8_full_6b_v1"))
    p.add_argument("--price-per-hour-eur", type=float)
    p.add_argument("--cost-cap-eur", type=float)
    p.add_argument("--projected-hours", type=float, default=30.0)
    p.add_argument("--min-free-gib", type=float, default=20.0)
    p.add_argument("--receipt", type=Path)
    args = p.parse_args(argv)
    if args.receipt is None:
        args.receipt = (args.repo_root / "results/m8/m81_local_preflight.json" if args.mode == "local"
                        else Path("/workspace/kimi/artifacts/m8/m81/preflight.json"))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        payload = execute(args)
        code = 0
    except Exception as exc:
        payload = {"status": "FAIL", "stage": args.mode, "error": f"{type(exc).__name__}: {exc}",
                   "training_started": False}
        code = 2
    payload["checked_at_utc"] = utc()
    try:
        # Remote mode never trusts a caller's arbitrary receipt path.
        if args.mode == "remote":
            parent = Path("/workspace/kimi/artifacts/m8/m81")
            if not args.receipt.resolve(strict=False).is_relative_to(parent):
                raise ValueError("Receipt path outside dedicated M8.1 artifact folder")
        write_receipt(args.receipt, payload)
    except Exception as exc:
        print(f"FAIL: could not write M8.1 receipt safely: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
