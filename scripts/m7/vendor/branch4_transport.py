#!/usr/bin/env python3
"""Native SSH/SFTP transfer and backup; never rents, starts, stops or deletes instances."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from branch4_data_common import canonical_hash, read_json, sha256_file, write_json
from branch4_remote_ops import file_status, validate_snapshot

ROOT = Path(__file__).resolve().parents[3]


def sftp_quote(path: str) -> str:
    if any(c in path for c in ("\r", "\n", "\0")):
        raise ValueError("Invalid newline/NUL in SFTP path")
    value = path.replace("\\", "\\\\").replace('"', '\\"')
    for character in "*?[]":
        value = value.replace(character, "\\" + character)
    return '"' + value + '"'


class Connection:
    def __init__(self, configuration: dict):
        self.config = configuration
        host, user, port = configuration.get("host"), configuration.get("user", "root"), configuration.get("port")
        if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.:-]+", host):
            raise ValueError("Fill host from the actual Vast direct SSH connection")
        if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", user) or user.startswith("-"):
            raise ValueError("Invalid SSH user")
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError("Fill a valid actual SSH port")
        self.address = user + "@" + ("[" + host + "]" if ":" in host else host)
        self.port = str(port)
        self.remote_root = configuration.get("remote_root", "/workspace/kimi")
        if not isinstance(self.remote_root, str) or not re.fullmatch(r"/workspace/[A-Za-z0-9_./-]+", self.remote_root) or ".." in PurePosixPath(self.remote_root).parts:
            raise ValueError("Remote root must be a controlled /workspace subdirectory")
        self.python = configuration.get("remote_python", "python3")
        self.ssh = shutil.which("ssh")
        self.sftp = shutil.which("sftp")
        if not self.ssh or not self.sftp:
            raise RuntimeError("Native ssh and sftp must be available in PATH")
        self.options = ["-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oConnectTimeout=20",
                        "-oServerAliveInterval=20", "-oServerAliveCountMax=3"]
        identity = configuration.get("identity_file")
        if identity:
            self.options += ["-i", str(Path(identity).expanduser().resolve())]
        self.ops = self.remote_root + "/ops/m7/branch4_remote_ops.py"

    def shell(self, arguments: list[str]) -> str:
        command = [self.ssh, "-T", "-p", self.port, *self.options, self.address, shlex.join(arguments)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"SSH failed ({result.returncode})")
        return result.stdout

    def rpc(self, arguments: list[str]) -> dict:
        return json.loads(self.shell([self.python, self.ops, *arguments]))

    def batch(self, text: str) -> None:
        command = [self.sftp, "-P", self.port, *self.options, "-B", "131072", "-R", "64", "-b", "-", self.address]
        result = subprocess.run(command, input=text, text=True)
        if result.returncode:
            raise RuntimeError(f"SFTP failed ({result.returncode}); partial transfers were retained")

    def install_helpers(self) -> None:
        self.shell(["mkdir", "-p", "--", self.remote_root + "/ops/m7"])
        commands = []
        for name in ("branch4_data_common.py", "branch4_remote_ops.py"):
            local = ROOT / "scripts/m7/vendor" / name
            commands.append("put " + sftp_quote(local.as_posix()) + " " + sftp_quote(self.remote_root + "/ops/m7/" + name))
        local = ROOT / "inputs/m7/reference_metadata.json"
        commands.append("put " + sftp_quote(local.as_posix()) + " " + sftp_quote(self.remote_root + "/ops/m7/reference_metadata.json"))
        self.batch("\n".join(commands) + "\n")


def upload_one(connection: Connection, local: Path, remote: str, size: int, digest: str,
               restart_partial: bool = False) -> dict:
    if local.stat().st_size != size or sha256_file(local) != digest:
        raise ValueError("Origin changed since verification: " + local.name)
    existing = connection.rpc(["file-status", "--path", remote, "--bytes", str(size), "--sha256", digest])
    if existing["valid"]:
        return {"file": local.name, "bytes": size, "skipped_verified": True}
    if existing["exists"]:
        raise FileExistsError("A different destination file exists; it was not overwritten: " + remote)
    partial = remote + ".b4part." + digest
    part = connection.rpc(["file-status", "--path", partial, "--bytes", str(size), "--sha256", digest])
    if restart_partial:
        connection.rpc(["reset-partial", "--path", partial])
        part = {"bytes": 0, "exists": False, "valid": False}
    if part["bytes"] > size or (part["bytes"] == size and not part["valid"]):
        raise ValueError("Corrupt/oversized managed partial; use --restart-partial explicitly")
    started = time.perf_counter()
    if not part["valid"]:
        connection.batch(("reput " if part["exists"] else "put ") + sftp_quote(local.as_posix()) + " " + sftp_quote(partial) + "\n")
    connection.rpc(["promote", "--partial", partial, "--final", remote, "--bytes", str(size), "--sha256", digest])
    seconds = time.perf_counter() - started
    moved = max(0, size - part["bytes"])
    return {"file": local.name, "bytes": size, "resumed_bytes": part["bytes"],
            "operation_seconds": seconds, "effective_mbps": moved * 8 / seconds / 1e6 if seconds else None,
            "rate_includes_protocol_and_hash_overhead": True, "sha256": digest}


def upload_data(connection: Connection, data: Path, inspection: Path, scope: str, restart: bool) -> dict:
    origin = read_json(inspection / "inspection.json")
    if not origin.get("dataset_verified") or origin.get("errors"):
        raise ValueError("A successful origin inspection is required")
    if scope == "qualification":
        manifest = read_json(inspection / "qualification_transfer_manifest.json")
        metadata_path = inspection / "qualification_metadata.json"
    else:
        manifest = read_json(inspection / "verified_data_manifest.json")
        metadata_path = data / "metadata.json"
        if sha256_file(metadata_path) != manifest["metadata_file_sha256"]:
            raise ValueError("Full metadata changed since origin inspection")
    destination = connection.remote_root + "/data/" + ("qualification" if scope == "qualification" else "full")
    connection.rpc(["mkdir", "--path", destination])
    rows = []
    for entry in manifest["files"]:
        name = entry["file"]
        if not re.fullmatch(r"(train|val)_[0-9]{5}\.npy", name):
            raise ValueError("Invalid shard name in transfer manifest")
        print("Transfer: " + name, flush=True)
        rows.append(upload_one(connection, data / name, destination + "/" + name,
                               entry["file_bytes"], entry["sha256"], restart))
    # Publish metadata last; a new dataset is not advertised before its shards.
    rows.append(upload_one(connection, metadata_path, destination + "/metadata.json",
                           metadata_path.stat().st_size, sha256_file(metadata_path), restart))
    verified = connection.rpc(["verify-dataset", "--data-dir", destination, "--scope", scope,
                              "--reference-metadata", connection.remote_root + "/ops/m7/reference_metadata.json"])
    return {"scope": scope, "destination": destination, "files": rows, "destination_verification": verified,
            "paid_instance_lifecycle_changed": False, "training_started": False}


def pull_snapshot(connection: Connection, remote: str, snapshot: dict, backups: Path) -> dict:
    validate_snapshot(snapshot)
    snapshot_id = snapshot["snapshot_id"]
    if not re.fullmatch(r"snapshot_[0-9TZ]+", snapshot_id):
        raise ValueError("Invalid snapshot identifier")
    if remote != connection.remote_root + "/backup_staging/" + snapshot_id:
        raise ValueError("Snapshot path is outside the managed backup namespace")
    parent = backups.resolve() / hashlib.sha256(snapshot["run_id"].encode()).hexdigest()[:16]
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / snapshot_id
    incoming = parent / (snapshot_id + ".partial")
    if target.exists():
        manifest = read_json(target / "snapshot.json")
        if manifest != snapshot:
            raise ValueError("Existing snapshot has a different manifest")
        for entry in snapshot["files"]:
            if not file_status(target / entry["file"], entry["bytes"], entry["sha256"])["valid"]:
                raise ValueError("Existing complete snapshot has corrupt contents")
        return {"backup_dir": str(target), "verified": True, "already_present": True}
    incoming.mkdir(exist_ok=True)
    for entry in snapshot["files"]:
        name = entry["file"]
        if name not in {"last.pt", "run.json", "metrics.jsonl", "console.log", "training.log"}:
            raise ValueError("Unexpected snapshot file")
        final = incoming / name
        if file_status(final, entry["bytes"], entry["sha256"])["valid"]:
            continue
        if final.exists():
            raise ValueError("Different local snapshot file; it was not overwritten")
        part = incoming / (name + ".b4part." + entry["sha256"])
        status = file_status(part, entry["bytes"], entry["sha256"])
        if status["bytes"] > entry["bytes"] or (status["bytes"] == entry["bytes"] and not status["valid"]):
            raise ValueError("Corrupt download partial; remove only that managed partial before retrying")
        if not status["valid"]:
            connection.batch(("reget " if status["exists"] else "get ") + sftp_quote(remote + "/" + name) + " " + sftp_quote(part.as_posix()) + "\n")
        if not file_status(part, entry["bytes"], entry["sha256"])["valid"]:
            raise ValueError("Downloaded snapshot hash failed: " + name)
        os.link(part, final)
        part.unlink()
    if read_json(incoming / "run.json").get("run_id") != snapshot["run_id"]:
        raise ValueError("Downloaded run identity disagrees with snapshot")
    write_json(incoming / "snapshot.json", snapshot)
    incoming.rename(target)
    return {"backup_dir": str(target), "verified": True, "checkpoint_restore_tested": False,
            "retention_note": "Prune only managed verified snapshots when explicitly requested"}


def prune_managed(parent: Path, run_id: str, keep: int = 2) -> dict:
    if keep < 2:
        raise ValueError("Keep at least two verified external copies")
    valid = []
    for path in parent.iterdir():
        if path.is_symlink() or not path.is_dir() or not re.fullmatch(r"snapshot_[0-9TZ]+", path.name):
            continue
        try:
            manifest = read_json(path / "snapshot.json")
            names = {entry["file"] for entry in manifest["files"]} | {"snapshot.json"}
            if manifest["run_id"] != run_id or manifest["snapshot_id"] != path.name:
                continue
            if names - {"last.pt", "run.json", "metrics.jsonl", "console.log", "training.log", "snapshot.json"}:
                continue
            if {p.name for p in path.iterdir()} != names or any(p.is_symlink() for p in path.iterdir()):
                continue
            if not all(file_status(path / e["file"], e["bytes"], e["sha256"])["valid"] for e in manifest["files"]):
                continue
            valid.append(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    valid.sort(key=lambda p: p.name, reverse=True)
    removed = []
    for path in valid[keep:]:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
        removed.append(path.name)
    return {"kept_verified": [p.name for p in valid[:keep]], "removed_managed_snapshots": removed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--connection", type=Path, required=True)
    p.add_argument("--output", type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("install-helpers")
    upload = sub.add_parser("upload-data")
    upload.add_argument("--data-dir", type=Path, default=Path("C:/datasets/fineweb_edu_6b"))
    upload.add_argument("--inspection-dir", type=Path, default=ROOT / "inputs/m7/origin_inspection")
    upload.add_argument("--scope", choices=("qualification", "full"), required=True)
    upload.add_argument("--restart-partial", action="store_true")
    artifact = sub.add_parser("upload-artifact")
    artifact.add_argument("--file", type=Path, required=True)
    backup = sub.add_parser("backup-run")
    backup.add_argument("--remote-run-dir", required=True)
    backup.add_argument("--backup-dir", type=Path, default=Path("C:/datasets/kimi_k3_backups"))
    backup.add_argument("--prune-managed", action="store_true")
    backup.add_argument("--keep", type=int, default=2)
    pull = sub.add_parser("pull-snapshot")
    pull.add_argument("--receipt", type=Path, required=True)
    pull.add_argument("--backup-dir", type=Path, default=Path("C:/datasets/kimi_k3_backups"))
    pull.add_argument("--prune-managed", action="store_true")
    pull.add_argument("--keep", type=int, default=2)
    args = p.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = Path("results/storage") / ("branch4_" + args.command + "_" + stamp + ".json")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "command": args.command,
              "status": "failed", "instance_lifecycle_changed": False}
    code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = Connection(read_json(args.connection))
        if args.command == "install-helpers":
            connection.install_helpers()
            result = {"helpers_installed": True}
        elif args.command == "upload-data":
            result = upload_data(connection, args.data_dir.resolve(), args.inspection_dir.resolve(), args.scope, args.restart_partial)
        elif args.command == "upload-artifact":
            name = args.file.name
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("Rename artifact to a simple filename before upload")
            directory = connection.remote_root + "/artifacts"
            connection.rpc(["mkdir", "--path", directory])
            result = upload_one(connection, args.file.resolve(), directory + "/" + name,
                                args.file.stat().st_size, sha256_file(args.file))
        elif args.command == "backup-run":
            if not args.remote_run_dir.startswith(connection.remote_root + "/runs/") or ".." in PurePosixPath(args.remote_run_dir).parts:
                raise ValueError("Run must be in the managed runs directory")
            captured = connection.rpc(["snapshot-run", "--run-dir", args.remote_run_dir,
                                       "--snapshot-root", connection.remote_root + "/backup_staging"])
            # Save the receipt before download, so an interruption can resume the
            # SAME immutable snapshot rather than capture another changing last.pt.
            report["snapshot_receipt"] = captured
            write_json(args.output, report)
            result = pull_snapshot(connection, captured["snapshot_dir"], captured["snapshot"], args.backup_dir)
        else:
            old = read_json(args.receipt)
            captured = old["snapshot_receipt"]
            report["snapshot_receipt"] = captured
            result = pull_snapshot(connection, captured["snapshot_dir"], captured["snapshot"], args.backup_dir)
        if args.command in ("backup-run", "pull-snapshot"):
            if args.prune_managed:
                result["retention"] = prune_managed(Path(result["backup_dir"]).parent,
                                                     captured["snapshot"]["run_id"], args.keep)
            try:
                result["remote_cleanup"] = connection.rpc(["release-snapshot", "--snapshot-dir", captured["snapshot_dir"],
                     "--snapshot-root", connection.remote_root + "/backup_staging",
                     "--manifest-sha256", canonical_hash(captured["snapshot"])])
            except (RuntimeError, ValueError) as exc:
                result["remote_cleanup_pending"] = str(exc)
        report.update(status="passed", result=result)
        code = 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
