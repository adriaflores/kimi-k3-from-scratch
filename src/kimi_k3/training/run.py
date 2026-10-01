"""Run provenance for the CLI; recording metadata never executes the model."""

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .trainer import _model_config, _runtime_signature


@dataclass(frozen=True)
class RunContext:
    directory: Path
    checkpoint_path: Path
    run_id: str
    session_id: str


def _git_metadata(repo_root):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo_root), *args], check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    try:
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _environment(model):
    device = next(model.parameters()).device
    hardware = {"device": str(device), "machine": platform.machine(),
                "cpu_count": os.cpu_count(), "gpu": None}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "hardware": hardware,
        "execution": {
            **_runtime_signature(model),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def _dataset_metadata(loader):
    dataset = loader.dataset
    return {
        "directory": str(dataset.directory.resolve()),
        "metadata_sha256": dataset.metadata_sha256,
        "total_tokens": dataset.total_tokens,
        "source": dataset.metadata.get("source"),
        "tokenizer": dataset.metadata["tokenizer"],
    }


def start_run(*, model, optimizer, config, state, train_loader, val_loader,
              repo_root, run_dir=None, resume_path=None, checkpoint_path=None,
              runs_root=Path("results/training/runs")) -> RunContext:
    """Record effective settings AFTER load_checkpoint has restored the recipe.

    A fresh invocation creates its own folder. Resume appends a session when
    run_dir is provided or run.json is beside the input checkpoint; otherwise
    it creates a new run linked to that checkpoint. Explicit checkpoint paths
    remain supported. Checkpoint format and the training loop are unchanged.
    """
    resume_path = None if resume_path is None else Path(resume_path).resolve()
    if (run_dir is None and resume_path is not None
            and (resume_path.parent / "run.json").is_file()):
        run_dir = resume_path.parent
    now = datetime.now(UTC)
    run_id = uuid4().hex
    if run_dir is None:
        run_dir = Path(runs_root) / f"{now:%Y%m%dT%H%M%SZ}-{run_id[:8]}"
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run.json"
    manifest = None
    if run_dir.exists():
        if resume_path is None or not manifest_path.is_file():
            raise FileExistsError("Run directory already exists; use a new --run-dir or resume its checkpoint")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or not manifest.get("run_id")
                or not isinstance(manifest.get("sessions"), list) or not manifest["sessions"]):
            raise ValueError("Invalid or unsupported run manifest")
        if not any(
            str(resume_path) == session["checkpoint_path"]
            or (session.get("checkpoint_relative_path") is not None
                and resume_path == (run_dir / session["checkpoint_relative_path"]).resolve())
            for session in manifest["sessions"]
        ):
            raise ValueError("Checkpoint is not associated with this run; choose a new --run-dir")
        run_id = manifest["run_id"]

    checkpoint_path = (Path(checkpoint_path).resolve() if checkpoint_path is not None
                       else resume_path if manifest is not None else run_dir / "last.pt")
    if checkpoint_path.name == "run.json":
        raise ValueError("Checkpoint path cannot overwrite run.json")
    session_id = uuid4().hex
    session = {
        "session_id": session_id, "started_at": now.isoformat(),
        "git": _git_metadata(repo_root),
        "model_config": _model_config(model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_config": asdict(config),
        "optimizer": {
            "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "param_groups": [{key: value for key, value in group.items() if key != "params"}
                             for group in optimizer.param_groups],
        },
        "environment": _environment(model),
        "data": {"train": _dataset_metadata(train_loader), "val": _dataset_metadata(val_loader)},
        "start_state": {
            "step": state.step, "tokens_seen": state.tokens_seen,
            "skipped_updates": state.skipped_updates,
            "loaders": {"train": train_loader.state_dict(), "val": val_loader.state_dict()},
        },
        "rng_source": "checkpoint" if resume_path is not None else "seed",
        "resume_checkpoint": None if resume_path is None else str(resume_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_relative_path": (checkpoint_path.relative_to(run_dir).as_posix()
                                     if checkpoint_path.is_relative_to(run_dir) else None),
    }
    # Normalize tuples (e.g. layer_types and AdamW betas) before comparing JSON.
    session = json.loads(json.dumps(session, allow_nan=False))
    if manifest is not None:
        previous = manifest["sessions"][-1]
        if session["model_config"] != previous["model_config"] or any(
            session["data"][split]["metadata_sha256"] != previous["data"][split]["metadata_sha256"]
            for split in ("train", "val")
        ):
            raise ValueError("Model or data differs from this run's manifest")
    else:
        manifest = {"schema_version": 1, "run_id": run_id,
                    "created_at": now.isoformat(), "sessions": []}
        run_dir.mkdir(parents=True, exist_ok=False)
    manifest["sessions"].append(session)
    temporary = run_dir / f"run.{session_id}.tmp"
    try:
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return RunContext(run_dir, checkpoint_path, run_id, session_id)
