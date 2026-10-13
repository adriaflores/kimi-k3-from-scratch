#!/usr/bin/env python3
"""Small, standard-library remote operations invoked over an existing SSH session."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from branch4_data_common import (FINAL_TOKENS, canonical_hash, inspect_shard,
                                qualification_metadata, read_json, sha256_file,
                                validate_metadata, write_json)


def file_status(path: Path, size: int, digest: str) -> dict:
    if path.is_symlink():
        raise ValueError("Symlinks are not accepted")
    if not path.exists():
        return {"exists": False, "valid": False, "bytes": 0}
    if not path.is_file():
        raise ValueError("Expected a regular file")
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path) if actual_size == size else None
    return {"exists": True, "valid": actual_size == size and actual_hash == digest,
            "bytes": actual_size, "actual_sha256": actual_hash}


def promote(partial: Path, final: Path, size: int, digest: str) -> dict:
    expected_name = final.name + ".b4part." + digest
    if partial.parent != final.parent or partial.name != expected_name:
        raise ValueError("Invalid managed partial name")
    status = file_status(partial, size, digest)
    if not status["valid"]:
        raise ValueError("Incomplete/corrupt partial; it was not published")
    if final.exists():
        existing = file_status(final, size, digest)
        if not existing["valid"]:
            raise FileExistsError("Refusing to overwrite a different final file")
    else:
        # Exclusive creation: another process cannot be overwritten by a race.
        os.link(partial, final)
    partial.unlink()
    return {"published": True, "sha256": digest, "bytes": size}


def verify_dataset(directory: Path, reference: Path, scope: str) -> dict:
    full = read_json(reference)
    expected = full if scope == "full" else qualification_metadata(full)
    actual = read_json(directory / "metadata.json")
    if canonical_hash(actual) != canonical_hash(expected):
        raise ValueError("Destination metadata fingerprint differs from the selected scope")
    rows = [inspect_shard(directory, entry) for entry in validate_metadata(expected)]
    passed = all(r["header_ok"] and r["hash_ok"] for r in rows)
    if scope == "full" and actual["splits"]["train"]["tokens"] < FINAL_TOKENS + 1:
        passed = False
    return {"scope": scope, "verified": passed, "metadata_sha256": canonical_hash(actual),
            "files": len(rows), "bytes": sum(r.get("file_bytes", 0) for r in rows),
            "failed_files": [r for r in rows if not (r["header_ok"] and r["hash_ok"])],
            "cuda_qualification": False}


def copy_fixed_prefix(source: Path, target: Path, lines_only: bool = False) -> None:
    # Snapshot an append-only log at its initial size; do not chase new writes.
    with source.open("rb") as src, target.open("xb") as dst:
        remaining = os.fstat(src.fileno()).st_size
        while remaining:
            chunk = src.read(min(1024 ** 2, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    if lines_only:
        with target.open("r+b") as dst:
            size = dst.seek(0, 2)
            if size:
                dst.seek(max(0, size - 1024 ** 2))
                tail = dst.read()
                last_newline = tail.rfind(b"\n")
                if last_newline < 0:
                    raise ValueError("No complete final JSONL line in the bounded log tail")
                dst.truncate(size - len(tail) + last_newline + 1)


def snapshot_run(run: Path, snapshots: Path) -> dict:
    run = run.resolve()
    checkpoint = run / "last.pt"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError("A completed regular last.pt is required; .tmp is never copied")
    provenance = read_json(run / "run.json")
    run_id = provenance.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.json does not identify the run")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = snapshots / ("snapshot_" + stamp)
    target.mkdir(parents=True, exist_ok=False)
    method = "hardlink"
    try:
        os.link(checkpoint, target / "last.pt")
    except OSError:
        method = "copy_from_open_descriptor"
        copy_fixed_prefix(checkpoint, target / "last.pt")
    copy_fixed_prefix(run / "run.json", target / "run.json")
    copied_provenance = read_json(target / "run.json")
    if copied_provenance.get("run_id") != run_id:
        raise ValueError("Run identity changed during snapshot")
    if (run / "metrics.jsonl").is_file():
        copy_fixed_prefix(run / "metrics.jsonl", target / "metrics.jsonl", lines_only=True)
    for name in ("console.log", "training.log"):
        if (run / name).is_file():
            copy_fixed_prefix(run / name, target / name)
    files = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(target.iterdir()) if p.is_file()]
    manifest = {"schema_version": 1, "snapshot_id": target.name, "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "original_run_dir": str(run), "checkpoint_capture": method,
                "checkpoint_restore_tested": False,
                "metrics_note": "bounded log prefix; may include events after captured checkpoint",
                "files": files}
    write_json(target / "snapshot.json", manifest)
    return {"snapshot_dir": str(target), "snapshot": manifest}


def validate_snapshot(manifest: dict) -> None:
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("Snapshot run identity is missing")
    seen = set()
    for entry in manifest["files"]:
        name = entry["file"]
        if name not in {"last.pt", "run.json", "metrics.jsonl", "console.log", "training.log"} or name in seen:
            raise ValueError("Unexpected or duplicate snapshot file")
        seen.add(name)
        if type(entry["bytes"]) is not int or entry["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Invalid snapshot size/digest")
    if not {"last.pt", "run.json"}.issubset(seen):
        raise ValueError("Checkpoint/provenance missing from snapshot")


def restore_snapshot(snapshot: Path, destination: Path) -> dict:
    manifest = read_json(snapshot / "snapshot.json")
    validate_snapshot(manifest)
    names = []
    for entry in manifest["files"]:
        name = entry["file"]
        if name not in {"last.pt", "run.json", "metrics.jsonl", "console.log", "training.log"} or name in names:
            raise ValueError("Unexpected or duplicate snapshot file")
        names.append(name)
        if not file_status(snapshot / name, entry["bytes"], entry["sha256"])["valid"]:
            raise ValueError("Snapshot integrity failed: " + name)
    if not {"last.pt", "run.json"}.issubset(names):
        raise ValueError("Checkpoint/provenance missing from snapshot")
    if read_json(snapshot / "run.json").get("run_id") != manifest["run_id"]:
        raise ValueError("Snapshot run identity mismatch")
    if destination.exists():
        raise FileExistsError("Restore requires a new destination; existing runs are not overwritten")
    destination.mkdir(parents=True)
    for name in names:
        shutil.copyfile(snapshot / name, destination / name)
    return {"restored_files": names, "destination": str(destination),
            "trainer_resume_tested": False, "run_id": manifest["run_id"]}


def release_snapshot(directory: Path, root: Path, digest: str) -> dict:
    if directory.is_symlink() or directory.parent.resolve() != root.resolve() or not re.fullmatch(r"snapshot_[0-9TZ]+", directory.name):
        raise ValueError("Only an exact managed staging snapshot may be released")
    manifest = read_json(directory / "snapshot.json")
    validate_snapshot(manifest)
    if canonical_hash(manifest) != digest:
        raise ValueError("Snapshot receipt no longer matches; nothing removed")
    allowed = {entry["file"] for entry in manifest["files"]} | {"snapshot.json"}
    if allowed - {"last.pt", "run.json", "metrics.jsonl", "console.log", "training.log", "snapshot.json"}:
        raise ValueError("Unexpected snapshot file; nothing removed")
    if {p.name for p in directory.iterdir()} != allowed or any(p.is_symlink() for p in directory.iterdir()):
        raise ValueError("Snapshot contains unmanaged files; nothing removed")
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    return {"staging_snapshot_released": True, "training_run_touched": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    status = sub.add_parser("file-status")
    status.add_argument("--path", type=Path, required=True)
    status.add_argument("--bytes", type=int, required=True)
    status.add_argument("--sha256", required=True)
    pub = sub.add_parser("promote")
    pub.add_argument("--partial", type=Path, required=True)
    pub.add_argument("--final", type=Path, required=True)
    pub.add_argument("--bytes", type=int, required=True)
    pub.add_argument("--sha256", required=True)
    mkdir = sub.add_parser("mkdir")
    mkdir.add_argument("--path", type=Path, required=True)
    reset = sub.add_parser("reset-partial")
    reset.add_argument("--path", type=Path, required=True)
    verify = sub.add_parser("verify-dataset")
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--reference-metadata", type=Path, required=True)
    verify.add_argument("--scope", choices=("qualification", "full"), required=True)
    snap = sub.add_parser("snapshot-run")
    snap.add_argument("--run-dir", type=Path, required=True)
    snap.add_argument("--snapshot-root", type=Path, required=True)
    restore = sub.add_parser("restore-snapshot")
    restore.add_argument("--snapshot-dir", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    release = sub.add_parser("release-snapshot")
    release.add_argument("--snapshot-dir", type=Path, required=True)
    release.add_argument("--snapshot-root", type=Path, required=True)
    release.add_argument("--manifest-sha256", required=True)
    args = p.parse_args()
    try:
        if args.command == "file-status":
            result = file_status(args.path, args.bytes, args.sha256)
        elif args.command == "promote":
            result = promote(args.partial, args.final, args.bytes, args.sha256)
        elif args.command == "mkdir":
            args.path.mkdir(parents=True, exist_ok=True)
            result = {"directory": str(args.path)}
        elif args.command == "reset-partial":
            if not re.search(r"\.b4part\.[0-9a-f]{64}$", args.path.name) or args.path.is_symlink():
                raise ValueError("Only managed partial files can be reset")
            args.path.unlink(missing_ok=True)
            result = {"reset_partial": str(args.path)}
        elif args.command == "verify-dataset":
            result = verify_dataset(args.data_dir, args.reference_metadata, args.scope)
            if not result["verified"]:
                print(json.dumps(result)); return 1
        elif args.command == "snapshot-run":
            result = snapshot_run(args.run_dir, args.snapshot_root)
        elif args.command == "release-snapshot":
            result = release_snapshot(args.snapshot_dir, args.snapshot_root, args.manifest_sha256)
        else:
            result = restore_snapshot(args.snapshot_dir, args.destination)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
