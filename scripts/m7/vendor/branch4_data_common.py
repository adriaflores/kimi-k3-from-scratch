"""Data checks for milestone 5; standard library only, no tensor allocation."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import struct
from pathlib import Path

EFFECTIVE_BATCH = 2048
FINAL_TOKENS = 6_000_001_024
CHECKPOINT_PAYLOAD_BYTES = 8 * 263_042_048 + 12 * 31_634_760
GIB = 1024 ** 3
SHARD_NAME = re.compile(r"^(train|val)_[0-9]{5}\.npy$")


def canonical_hash(value: object) -> str:
    # Identical serialization to ShardedTokenDataset's metadata identity.
    data = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 ** 2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {path.name}")
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_metadata(metadata: dict) -> list[dict]:
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    if metadata.get("format_version") != 2:
        raise ValueError("Expected format_version=2")
    storage = metadata.get("storage", {})
    if not isinstance(storage, dict) or storage.get("format") != "npy" or storage.get("dtype") != "<u2":
        raise ValueError("Expected little-endian uint16 NPY storage (<u2)")
    tokenizer = metadata.get("tokenizer", {})
    if not isinstance(tokenizer, dict):
        raise ValueError("Tokenizer metadata must be an object")
    vocab = tokenizer.get("vocab_size")
    eos = tokenizer.get("eos_id")
    if type(vocab) is not int or not 0 < vocab <= 65536:
        raise ValueError("Invalid vocabulary size")
    if type(eos) is not int or not 0 <= eos < vocab:
        raise ValueError("Invalid EOS ID")
    records = []
    seen = set()
    for split in ("train", "val"):
        summary = metadata["splits"][split]
        shards = summary["shards"]
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"Empty or invalid {split} shards")
        total = 0
        for entry in shards:
            name, tokens, digest = entry["file"], entry["tokens"], entry["sha256"]
            if not isinstance(name, str) or not SHARD_NAME.fullmatch(name) or not name.startswith(split + "_"):
                raise ValueError(f"Invalid shard name: {name!r}")
            if name in seen:
                raise ValueError(f"Duplicate shard: {name}")
            seen.add(name)
            if type(tokens) is not int or tokens <= 0:
                raise ValueError(f"Invalid token count: {name}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid SHA-256: {name}")
            total += tokens
            records.append({"split": split, "file": name, "tokens": tokens, "sha256": digest})
        if type(summary["tokens"]) is not int or summary["tokens"] != total:
            raise ValueError(f"Split count disagrees with shards: {split}")
    return records


def metadata_differences(actual: dict, reference: dict) -> list[str]:
    # Paths, timings and package inventories do not change the ordered token stream.
    required = ("format_version", "source", "storage", "tokenizer", "splits")
    provenance = ("reader", "source_prefix_sha256", "split_rule", "text_processing", "script_sha256")
    return [key for key in required + provenance if actual.get(key) != reference.get(key)]


def npy_header(path: Path, expected_tokens: int) -> dict:
    if path.is_symlink():
        raise ValueError("Shard symlinks are not accepted for the transfer inventory")
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("Invalid NPY magic")
        version_bytes = stream.read(2)
        if len(version_bytes) != 2:
            raise ValueError("Truncated NPY version")
        version = tuple(version_bytes)
        if version not in ((1, 0), (2, 0), (3, 0)):
            raise ValueError(f"Unsupported NPY version: {version}")
        width = 2 if version == (1, 0) else 4
        raw_length = stream.read(width)
        if len(raw_length) != width:
            raise ValueError("Truncated NPY header length")
        length = struct.unpack("<H" if width == 2 else "<I", raw_length)[0]
        if length > 65536:
            raise ValueError("NPY header exceeds 64 KiB")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Truncated NPY header")
        header = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin-1"))
        if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
            raise ValueError("Invalid NPY header fields")
        shape = header["shape"]
        if (not isinstance(shape, tuple) or len(shape) != 1 or type(shape[0]) is not int or
                header["descr"] != "<u2" or shape != (expected_tokens,) or header["fortran_order"] is not False):
            raise ValueError(f"Expected <u2 shape {(expected_tokens,)} in C order; got {header}")
        offset = stream.tell()
        size = os.fstat(stream.fileno()).st_size
        if size != offset + 2 * expected_tokens:
            raise ValueError(f"NPY file size mismatch: {size} != {offset + 2 * expected_tokens}")
    return {"npy_version": list(version), "data_offset_bytes": offset, "file_bytes": size,
            "payload_bytes": 2 * expected_tokens, "tokens": expected_tokens, "dtype": "<u2"}


def inspect_shard(root: Path, record: dict) -> dict:
    path = root / record["file"]
    result = {**record, "exists": path.is_file(), "header_ok": False, "hash_ok": False,
              "actual_sha256": None, "error": None}
    try:
        result.update(npy_header(path, record["tokens"]))
        result["header_ok"] = True
        before = path.stat()
        result["actual_sha256"] = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("Shard changed or was replaced during inspection")
        result["hash_ok"] = result["actual_sha256"] == record["sha256"]
        if not result["hash_ok"]:
            result["error"] = "SHA-256 mismatch"
    except (OSError, ValueError, SyntaxError, KeyError, UnicodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def qualification_metadata(parent: dict) -> dict:
    validate_metadata(parent)
    splits = {}
    for split in ("train", "val"):
        first = dict(parent["splits"][split]["shards"][0])
        splits[split] = {"tokens": first["tokens"], "shards": [first]}
    max_tokens = ((splits["train"]["tokens"] - 1) // EFFECTIVE_BATCH) * EFFECTIVE_BATCH
    value = {"format_version": 2, "source": parent.get("source"),
             "storage": parent["storage"], "tokenizer": parent["tokenizer"], "splits": splits,
             "qualification_prefix": {
                 "scope": "qualification only; original whole shards, unchanged bytes",
                 "parent_metadata_sha256": canonical_hash(parent),
                 "max_tokens_per_run_without_wrap": max_tokens,
                 "max_updates_per_run_without_wrap": max_tokens // EFFECTIVE_BATCH,
                 "effective_batch_tokens": EFFECTIVE_BATCH,
                 "continue_as_full_corpus_pilot": False}}
    validate_metadata(value)
    return value


def disk_for(path: Path) -> dict:
    import shutil
    intended = path.resolve()
    ancestor = intended
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    usage = shutil.disk_usage(ancestor)
    return {"intended_path": str(intended), "exists": intended.exists(),
            "queried_ancestor": str(ancestor), "total_bytes": usage.total,
            "free_bytes": usage.free, "free_gib": usage.free / GIB}
