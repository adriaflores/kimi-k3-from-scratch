import argparse
import hashlib
import json
import logging
import os
import platform
import time
from contextlib import closing
from fnmatch import fnmatchcase
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

os.environ.setdefault("USE_TORCH", "0")

from transformers import AutoTokenizer

DATASET = "HuggingFaceFW/fineweb-edu"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("C:/datasets/fineweb_edu_6b"))
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--revision", default="main", help="Dataset commit or ref; resolved to a SHA.")
    parser.add_argument("--tokenizer", default=TOKENIZER, help="Hugging Face tokenizer repository.")
    parser.add_argument("--tokenizer-revision", default="main", help="Tokenizer commit or ref; resolved to a SHA.")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-documents", type=int, help="Maximum source rows visited; default 10,000 without a token budget.")
    budget.add_argument("--train-tokens", type=int, help="Target train tokens including EOS; validation is additional.")
    parser.add_argument("--tokenization-batch-size", type=int, default=128, help="Nonempty documents per tokenizer call.")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Expected document fraction.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    parser.add_argument("--log-interval", type=int, default=10, help="Seconds between progress reports.")
    parser.add_argument("--checkpoint-interval", type=int, default=300, help="Seconds between checkpoints, checked after each batch.")
    parser.add_argument("--resume", action="store_true", help="Restore saved data settings and resume the specified output directory.")
    args = parser.parse_args()
    if args.max_documents is None and args.train_tokens is None:
        args.max_documents = 10_000
    for name in ("max_documents", "train_tokens", "tokenization_batch_size", "shuffle_buffer", "shard_tokens", "log_interval", "checkpoint_interval"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1 (exclusive)")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def load_tokenizer(repo_id: str, revision: str):
    """Pin the tokenizer and include added tokens in the required ID range."""
    sha = HfApi().model_info(repo_id, revision=revision).sha
    if not sha:
        raise RuntimeError("Could not resolve the tokenizer revision to a commit SHA.")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=sha, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required.")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")
    # vocab_size alone may exclude added tokens. Max ID also handles ID gaps.
    vocab_size = max(max(tokenizer.get_vocab().values()), tokenizer.eos_token_id) + 1
    metadata = {
        "repository": repo_id, "revision": sha,
        "vocab_size": vocab_size, "eos_id": tokenizer.eos_token_id,
        "use_fast": True, "add_special_tokens": False,
        "append_eos": True,
    }
    return tokenizer, metadata


def source_rng(seed: int, stream: int):
    """Separate PCG64 streams: 0 for file order, 1 for document shuffle."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))


def resolve_parquet_files(repository: str, revision: str, config: str, seed: int) -> list[str]:
    """Resolve train globs from the pinned dataset card, never local enumeration."""
    info = HfApi().dataset_info(repository, revision=revision)
    configs = info.card_data.get("configs", []) if info.card_data else []
    selected = next((item for item in configs if item["config_name"] == config), None)
    if selected is None:
        raise ValueError(f"Dataset config is absent from the pinned card: {config}")
    patterns = []
    for entry in selected["data_files"]:
        if entry["split"] == "train":
            paths = entry["path"]
            patterns.extend([paths] if isinstance(paths, str) else paths)
    files = sorted({
        item.rfilename for item in info.siblings
        if item.rfilename.endswith(".parquet")
        and any(fnmatchcase(item.rfilename, pattern) for pattern in patterns)
    })
    if not files:
        raise ValueError(f"No train Parquet files matched config {config} at {revision}.")
    source_rng(seed, 0).shuffle(files)
    return files


def reader_metadata(files: list[str], seed: int, buffer_size: int) -> dict:
    return {
        "type": "pyarrow.parquet.ParquetFile.iter_batches",
        "batch_rows": 1024, "columns": ["text"],
        "pre_buffer": False, "memory_map": False, "use_threads": True,
        "parquet_files": files,
        "file_resolution": "pinned dataset card train globs, Parquet paths only",
        "file_order": {
            "algorithm": "lexicographic_sort_then_numpy_shuffle_v1",
            "rng": "PCG64", "seed_sequence": [seed, 0],
        },
        "row_order": "stored row order within each Parquet file",
        "shuffle": {
            "algorithm": "bounded_buffer_v1", "buffer_size": buffer_size,
            "seed": seed, "rng": "PCG64", "seed_sequence": [seed, 1],
        },
    }


def parquet_texts(repository: str, revision: str, reader: dict):
    """Decode one text batch at a time; close each file even on early exit."""
    for filename in reader["parquet_files"]:
        LOG.info("Reading Parquet: %s", filename)
        local_path = hf_hub_download(
            repo_id=repository, filename=filename, repo_type="dataset", revision=revision,
        )
        with pq.ParquetFile(
            local_path, pre_buffer=reader["pre_buffer"], memory_map=reader["memory_map"],
        ) as parquet:
            for batch in parquet.iter_batches(
                batch_size=reader["batch_rows"], columns=reader["columns"],
                use_threads=reader["use_threads"],
            ):
                for value in batch.column(0):
                    yield value.as_py()


def shuffled_texts(repository: str, revision: str, reader: dict):
    """Bound retained documents by buffer_size; drain the final partial buffer.

    The bound is on document count, not bytes; individual texts vary in size.
    RNG calls depend only on rows, never on Arrow batch or row-group boundaries.
    """
    shuffle = reader["shuffle"]
    rng = source_rng(shuffle["seed"], 1)
    buffer = []
    with closing(parquet_texts(repository, revision, reader)) as documents:
        for text in documents:
            if len(buffer) < shuffle["buffer_size"]:
                buffer.append(text)
                continue
            index = int(rng.integers(0, len(buffer)))
            yield buffer[index]
            buffer[index] = text
    rng.shuffle(buffer)
    yield from buffer


def split_for(text: str, seed: int, val_fraction: float) -> str:
    """Assign equal texts to the same split, independent of stream order.

    This groups exact duplicates; it neither removes them nor detects near
    duplicates. Python's built-in hash() is not stable across processes.
    """
    digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:8], "big") < int(val_fraction * 2**64) else "train"


class ShardWriter:
    """Append tokens using one fixed-size NumPy buffer per split."""

    def __init__(self, directory: Path, split: str, capacity: int, dtype: np.dtype):
        self.directory = directory
        self.split = split
        self.buffer = np.empty(capacity, dtype=dtype)
        self.used = 0
        self.documents = 0
        self.tokens = 0
        self.shards = []

    def append(self, tokens: list[int]) -> None:
        self.documents += 1
        self.tokens += len(tokens)
        offset = 0
        while offset < len(tokens):
            size = min(len(tokens) - offset, len(self.buffer) - self.used)
            self.buffer[self.used : self.used + size] = tokens[offset : offset + size]
            self.used += size
            offset += size
            if self.used == len(self.buffer):
                self.flush()

    def flush(self) -> None:
        if not self.used:
            return
        path = self.directory / f"{self.split}_{len(self.shards):05d}.npy"
        save_array(path, self.buffer[: self.used])
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        self.shards.append({"file": path.name, "tokens": self.used, "sha256": checksum})
        LOG.info("Wrote %s (%s tokens)", path.name, f"{self.used:,}")
        self.used = 0

    def summary(self) -> dict:
        return {"documents": self.documents, "tokens": self.tokens, "shards": self.shards}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("huggingface-hub", "numpy", "pyarrow", "transformers", "tokenizers")},
    }


def clean_checkpoint_buffers(directory: Path, keep=()) -> None:
    for path in directory.glob("checkpoint-*.npy"):
        if path.name not in keep:
            try:
                path.unlink()
            except OSError:
                LOG.warning("Could not remove obsolete checkpoint buffer: %s", path)


def save_checkpoint(directory, writers, state, started):
    """Publish a JSON pointer only after both new buffer snapshots are durable.

    Never overwrite buffers referenced by the previous checkpoint. A crash
    before the JSON replacement leaves the old checkpoint usable.
    """
    generation = uuid4().hex
    snapshots = {}
    for split, writer in writers.items():
        path = directory / f"checkpoint-{generation}-{split}.npy"
        save_array(path, writer.buffer[:writer.used])
        snapshots[split] = {
            **writer.summary(), "buffer_file": path.name,
            "buffer_sha256": file_sha256(path),
        }
    checkpoint = {**state, "writers": snapshots,
                  "elapsed_seconds": time.perf_counter() - started}
    save_json(directory / "checkpoint.json", checkpoint)
    clean_checkpoint_buffers(directory, {s["buffer_file"] for s in snapshots.values()})
    LOG.info("Checkpoint saved: %s train tokens", f"{writers['train'].tokens:,}")


def restore_writers(directory, writers, checkpoint):
    keep = set()
    for split, writer in writers.items():
        saved = checkpoint["writers"][split]
        for shard in saved["shards"]:
            path = directory / shard["file"]
            if file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Checkpoint shard checksum mismatch: {path}")
            keep.add(path.name)
        path = directory / saved["buffer_file"]
        if file_sha256(path) != saved["buffer_sha256"]:
            raise ValueError(f"Checkpoint buffer checksum mismatch: {path}")
        buffer = np.load(path, allow_pickle=False)
        if buffer.ndim != 1 or buffer.dtype != writer.buffer.dtype or len(buffer) > len(writer.buffer):
            raise ValueError(f"Invalid checkpoint buffer: {path}")
        if saved["tokens"] != sum(s["tokens"] for s in saved["shards"]) + len(buffer):
            raise ValueError(f"Checkpoint token count mismatch: {split}")
        writer.used = len(buffer)
        writer.buffer[:writer.used] = buffer
        writer.documents, writer.tokens = saved["documents"], saved["tokens"]
        writer.shards = saved["shards"]
    # A failed batch may have created shards absent from the committed state.
    for pattern in ("train_*.npy", "val_*.npy"):
        for path in directory.glob(pattern):
            if path.name not in keep:
                path.unlink()
    clean_checkpoint_buffers(directory, {s["buffer_file"] for s in checkpoint["writers"].values()})


def report_progress(writers: dict, started: float, target: int | None) -> dict:
    elapsed = max(time.perf_counter() - started, 1e-9)
    train = writers["train"].tokens
    total = sum(writer.tokens for writer in writers.values())
    rate = train / elapsed
    remaining = max(0, (target or 6_000_000_000) - train) / rate if rate else None
    eta = f"{remaining / 3600:.2f} h" if remaining is not None else "calculating"
    goal = f" / {target:,} ({min(100, 100 * train / target):.1f}%)" if target else ""
    LOG.info(
        "Train %s%s | %.0f train tok/s | %.0f total tok/s | elapsed %.2f h | ETA%s %s",
        f"{train:,}", goal, rate, total / elapsed, elapsed / 3600,
        " to 6B" if target is None else "", eta,
    )
    return {
        "elapsed_seconds": elapsed, "train_tokens_per_second": rate,
        "total_tokens_per_second": total / elapsed,
        "estimated_remaining_seconds": remaining,
        "estimated_hours_for_6B_train_tokens": 6_000_000_000 / rate / 3600 if rate else None,
    }


def prepare(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists():
        if args.resume and (output / "metadata.json").exists():
            LOG.info("Dataset is already complete: %s", output)
            return json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory.")
    versions = runtime_versions()
    saved = None
    if args.resume:
        saved = json.loads((staging / "checkpoint.json").read_text(encoding="utf-8"))
        if saved.get("checkpoint_version") != 2:
            raise ValueError("This reader requires checkpoint_version=2. Start a fresh output directory; old datasets checkpoints cannot be resumed.")
        if saved["versions"] != versions:
            raise ValueError("Checkpoint/environment mismatch. Use the original Python version and uv.lock.")
        # Resume uses original data settings, even when CLI defaults differ.
        for key, value in saved["preparation"].items():
            if key not in ("output_dir", "resume", "log_interval", "checkpoint_interval"):
                setattr(args, key, value)
        LOG.info("Restored original data settings and pinned revisions from checkpoint.")
    elif staging.exists():
        raise FileExistsError(f"Unfinished run exists: {staging}. Use --resume or a new output directory.")

    revision = saved["source"]["revision"] if saved else HfApi().dataset_info(DATASET, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve the dataset revision to a commit SHA.")
    repository = saved["source"]["dataset"] if saved else DATASET
    if saved:
        reader = saved["reader"]
        if saved["source"]["config"] != args.config or reader != reader_metadata(
            reader["parquet_files"], args.seed, args.shuffle_buffer,
        ):
            raise ValueError("Checkpoint reader settings/algorithm are incompatible.")
    else:
        files = resolve_parquet_files(repository, revision, args.config, args.seed)
        reader = reader_metadata(files, args.seed, args.shuffle_buffer)
    tokenizer_ref = saved["tokenizer"]["revision"] if saved else args.tokenizer_revision
    tokenizer, tokenizer_metadata = load_tokenizer(args.tokenizer, tokenizer_ref)
    if saved and tokenizer_metadata != saved["tokenizer"]:
        raise ValueError("Tokenizer does not match the checkpoint.")
    vocab_size = tokenizer_metadata["vocab_size"]
    if vocab_size > 2**32:
        raise ValueError("Tokenizer IDs do not fit in uint32.")
    dtype = np.dtype("<u2" if vocab_size <= 2**16 else "<u4")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not saved:
        staging.mkdir()
    started = time.perf_counter() - (saved["elapsed_seconds"] if saved else 0)
    writers = {split: ShardWriter(staging, split, args.shard_tokens, dtype) for split in ("train", "val")}
    if saved:
        LOG.info("Checking and restoring committed shards and partial buffers...")
        restore_writers(staging, writers, saved)
    visited = saved["source_rows_visited"] if saved else 0
    skipped = saved["empty_documents_skipped"] if saved else 0
    source_digest = hashlib.sha256()
    state = {
        "checkpoint_version": 2,
        "preparation": {**vars(args), "output_dir": str(output)},
        "source": {"dataset": repository, "config": args.config, "split": "train", "revision": revision},
        "reader": reader,
        "tokenizer": tokenizer_metadata, "versions": versions,
    }

    def checkpoint(finished=False):
        state.update(
            source_rows_visited=visited, empty_documents_skipped=skipped,
            source_prefix_sha256=source_digest.hexdigest(), finished=finished,
        )
        save_checkpoint(staging, writers, state, started)

    # Do not delete staging on exceptions: the previous checkpoint stays usable.
    if not saved:
        checkpoint()
    if not (saved and saved["finished"]):
        with closing(shuffled_texts(repository, revision, reader)) as documents:
            source_iterator = iter(documents) if args.max_documents is None else islice(documents, args.max_documents)
            LOG.info("Source: %s / %s @ %s", repository, args.config, revision)

            def read_text(text):
                if not isinstance(text, str):
                    raise TypeError("Expected a string in the source text column.")
                raw = text.encode("utf-8")
                source_digest.update(len(raw).to_bytes(8, "big"))
                source_digest.update(raw)
                return text

            # Replay reconstructs the bounded shuffle from the stored file order,
            # without re-tokenizing or rewriting the committed prefix.
            if saved:
                LOG.info("Replaying %s source rows without tokenization; this may take time.", f"{visited:,}")
                last_replay_log = time.perf_counter()
                for index in range(visited):
                    try:
                        read_text(next(source_iterator))
                    except StopIteration as exc:
                        raise ValueError("Source ended before the checkpoint position.") from exc
                    if time.perf_counter() - last_replay_log >= args.log_interval:
                        LOG.info("Replay: %s / %s source rows", f"{index + 1:,}", f"{visited:,}")
                        last_replay_log = time.perf_counter()
                if source_digest.hexdigest() != saved["source_prefix_sha256"]:
                    raise ValueError("Source order/content differs from the checkpoint; refusing to resume.")
            last_report = last_checkpoint = time.perf_counter()

            def nonempty_texts():
                nonlocal visited, skipped
                for document in source_iterator:
                    visited += 1
                    text = read_text(document)
                    if not text.strip():
                        skipped += 1
                        continue
                    yield text

            text_iterator = nonempty_texts()
            while texts := list(islice(text_iterator, args.tokenization_batch_size)):
                # Preserve document order and full lengths; training builds blocks later.
                encoded = tokenizer(
                    texts, add_special_tokens=False, padding=False, truncation=False,
                    return_attention_mask=False, return_token_type_ids=False, verbose=False,
                )["input_ids"]
                for text, tokens in zip(texts, encoded, strict=True):
                    split = split_for(text, args.seed, args.val_fraction)
                    tokens.append(tokenizer.eos_token_id)
                    writers[split].append(tokens)
                    if args.train_tokens is not None and writers["train"].tokens >= args.train_tokens:
                        break
                if time.perf_counter() - last_report >= args.log_interval:
                    report_progress(writers, started, args.train_tokens)
                    last_report = time.perf_counter()
                if args.train_tokens is not None and writers["train"].tokens >= args.train_tokens:
                    break
                # At this boundary every prefetched nonempty document was written.
                # Never checkpoint in the middle of a batch or in an exception handler.
                if time.perf_counter() - last_checkpoint >= args.checkpoint_interval:
                    checkpoint()
                    last_checkpoint = time.perf_counter()
            text_iterator.close()
            # Protect final partial buffers before flushing and publishing the output.
            # If interrupted here, resume can finalize without replaying the source.
            checkpoint(finished=True)

    if any(writer.documents == 0 for writer in writers.values()):
        raise ValueError("An output split is empty. Use a larger sample or adjust --val-fraction in a new run.")
    for writer in writers.values():
        writer.flush()
    performance = report_progress(writers, started, args.train_tokens)
    target_reached = None if args.train_tokens is None else writers["train"].tokens >= args.train_tokens
    if target_reached is False:
        LOG.warning("Source exhausted before the train-token target; saving available data with target_reached=false.")

    metadata = {
        "format_version": 2,
        "source": state["source"], "preparation": state["preparation"],
        "reader": reader,
        "source_prefix_sha256": saved["source_prefix_sha256"] if saved and saved["finished"] else source_digest.hexdigest(),
        "selection": "seeded buffer shuffle; stop at document limit or after the whole document reaching the train-token target",
        "split_rule": "sha256(seed + NUL + original UTF-8 text), first 64 bits below val_fraction * 2**64",
        "text_processing": "skip whitespace-only documents; otherwise preserve text unchanged",
        "tokenizer": tokenizer_metadata,
        "storage": {"format": "npy", "dtype": dtype.str, "eos": "after every document"},
        "source_rows_visited": visited, "empty_documents_skipped": skipped,
        "prefetched_documents_not_written": visited - skipped - sum(w.documents for w in writers.values()),
        "train_token_target_reached": target_reached, "performance": performance,
        "splits": {split: writer.summary() for split, writer in writers.items()},
        "versions": versions,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save_json(staging / "metadata.json", metadata)
    if output.exists():
        raise FileExistsError(f"Output appeared during preparation: {output}")
    staging.rename(output)
    # Cleanup follows publication, so an interruption cannot destroy recovery.
    clean_checkpoint_buffers(output)
    for name in ("checkpoint.json", "checkpoint.json.tmp"):
        try:
            (output / name).unlink(missing_ok=True)
        except OSError:
            LOG.warning("Could not remove obsolete checkpoint file: %s", output / name)

    for split, writer in writers.items():
        LOG.info("%s: %s documents, %s tokens", split, f"{writer.documents:,}", f"{writer.tokens:,}")
    LOG.info("Dataset ready: %s", output)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    prepare(parse_args())
