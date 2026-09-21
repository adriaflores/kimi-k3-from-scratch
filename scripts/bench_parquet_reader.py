from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_PREFIX = "sample/10BT/"
DEFAULT_BYTES_PER_TOKEN = 4.58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DATASET)
    parser.add_argument("--revision", default="main", help="Dataset ref; resolved to a commit SHA.")
    parser.add_argument("--path-prefix", default=DEFAULT_PREFIX,
                        help="Only benchmark .parquet files under this repository path.")
    parser.add_argument("--batch-rows", type=int, default=1024,
                        help="Maximum Parquet rows per RecordBatch.")
    parser.add_argument("--max-text-gib", type=float, default=3.0,
                        help="Stop after decoding approximately this many GiB of UTF-8 text.")
    parser.add_argument("--max-files", type=int,
                        help="Optional hard limit on Parquet files, in addition to --max-text-gib.")
    parser.add_argument("--log-interval", type=float, default=10.0,
                        help="Seconds between progress reports.")
    parser.add_argument("--bytes-per-token", type=float, default=DEFAULT_BYTES_PER_TOKEN,
                        help="Only used for a rough equivalent-token estimate.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/parquet_reader"))
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional Hugging Face cache directory.")
    parser.add_argument("--use-threads", action="store_true",
                        help="Enable PyArrow threaded column reads. Off by default to minimize memory.")
    args = parser.parse_args()

    for name in ("batch_rows", "max_text_gib", "log_interval", "bytes_per_token"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")
    return args


def gib(value: int | float) -> float:
    return float(value) / (1024**3)


def mib(value: int | float) -> float:
    return float(value) / (1024**2)


def current_memory(process: psutil.Process) -> tuple[int, int]:
    return process.memory_info().rss, pa.total_allocated_bytes()


def print_progress(*, started, files_done, batches_done, documents, text_bytes,
                   baseline_rss, peak_rss, rss, arrow_bytes, bytes_per_token) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    approx_tokens = text_bytes / bytes_per_token
    print(
        f"files={files_done:,} | batches={batches_done:,} | docs={documents:,} | "
        f"text={gib(text_bytes):.2f} GiB | ~tokens={approx_tokens / 1e6:.0f}M | "
        f"RSS={gib(rss):.3f} GiB (+{mib(rss - baseline_rss):.0f} MiB) | "
        f"peak={gib(peak_rss):.3f} GiB | Arrow={mib(arrow_bytes):.0f} MiB | "
        f"elapsed={elapsed / 60:.1f} min",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision)
    revision = info.sha
    if not revision:
        raise RuntimeError("Could not resolve the dataset revision to a commit SHA.")

    print(f"Resolving Parquet files for {args.repo_id} @ {revision}...", flush=True)
    files = sorted(
        path
        for path in api.list_repo_files(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=revision,
        )
        if path.startswith(args.path_prefix) and path.endswith(".parquet")
    )
    if not files:
        raise ValueError(
            f"No .parquet files found under prefix {args.path_prefix!r}. Check --path-prefix."
        )
    if args.max_files is not None:
        files = files[: args.max_files]

    print(
        f"Found {len(files):,} candidate Parquet files. "
        f"Target decoded text: {args.max_text_gib:.2f} GiB.",
        flush=True,
    )

    process = psutil.Process()
    baseline_rss, baseline_arrow = current_memory(process)
    peak_rss = baseline_rss
    peak_arrow = baseline_arrow
    started = last_log = time.perf_counter()

    files_done = 0
    batches_done = 0
    documents = 0
    text_bytes = 0
    null_texts = 0
    samples: list[dict] = []
    file_summaries: list[dict] = []
    target_bytes = int(args.max_text_gib * 1024**3)
    stop = False

    for remote_path in files:
        if stop:
            break

        print(f"Downloading/opening {remote_path}...", flush=True)
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
        )

        file_docs_before = documents
        file_bytes_before = text_bytes
        file_started = time.perf_counter()

        parquet = pq.ParquetFile(
            local_path,
            memory_map=False,
            buffer_size=0,
            pre_buffer=False,
        )

        try:
            if "text" not in parquet.schema_arrow.names:
                raise ValueError(f"'text' column missing from {remote_path}")

            for record_batch in parquet.iter_batches(
                batch_size=args.batch_rows,
                columns=["text"],
                use_threads=args.use_threads,
                use_pandas_metadata=False,
            ):
                batches_done += 1
                texts = record_batch.column(0).to_pylist()
                documents += len(texts)

                for text in texts:
                    if text is None:
                        null_texts += 1
                        continue
                    if not isinstance(text, str):
                        raise TypeError(
                            f"Expected string text, got {type(text).__name__} in {remote_path}"
                        )
                    text_bytes += len(text.encode("utf-8"))

                rss, arrow_bytes = current_memory(process)
                peak_rss = max(peak_rss, rss)
                peak_arrow = max(peak_arrow, arrow_bytes)

                now = time.perf_counter()
                if now - last_log >= args.log_interval:
                    samples.append({
                        "elapsed_seconds": now - started,
                        "files_done": files_done,
                        "batches_done": batches_done,
                        "documents": documents,
                        "text_bytes": text_bytes,
                        "approx_tokens": text_bytes / args.bytes_per_token,
                        "rss_bytes": rss,
                        "peak_rss_bytes": peak_rss,
                        "arrow_allocated_bytes": arrow_bytes,
                    })
                    print_progress(
                        started=started,
                        files_done=files_done,
                        batches_done=batches_done,
                        documents=documents,
                        text_bytes=text_bytes,
                        baseline_rss=baseline_rss,
                        peak_rss=peak_rss,
                        rss=rss,
                        arrow_bytes=arrow_bytes,
                        bytes_per_token=args.bytes_per_token,
                    )
                    last_log = now

                if text_bytes >= target_bytes:
                    stop = True
                    break
        finally:
            parquet.close()

        files_done += 1
        rss, arrow_bytes = current_memory(process)
        peak_rss = max(peak_rss, rss)
        peak_arrow = max(peak_arrow, arrow_bytes)
        file_summaries.append({
            "file": remote_path,
            "documents": documents - file_docs_before,
            "text_bytes": text_bytes - file_bytes_before,
            "seconds": time.perf_counter() - file_started,
            "rss_bytes_after_file": rss,
            "arrow_allocated_bytes_after_file": arrow_bytes,
        })

    rss, arrow_bytes = current_memory(process)
    peak_rss = max(peak_rss, rss)
    peak_arrow = max(peak_arrow, arrow_bytes)
    elapsed = time.perf_counter() - started

    growth_mib = mib(rss - baseline_rss)
    text_gib = gib(text_bytes)
    growth_mib_per_text_gib = growth_mib / text_gib if text_gib else None

    if len(samples) >= 2:
        median_rss_mib = statistics.median(mib(s["rss_bytes"]) for s in samples)
    else:
        median_rss_mib = mib(rss)

    result = {
        "source": {
            "repository": args.repo_id,
            "revision": revision,
            "path_prefix": args.path_prefix,
        },
        "settings": {
            "batch_rows": args.batch_rows,
            "max_text_gib": args.max_text_gib,
            "max_files": args.max_files,
            "use_threads": args.use_threads,
            "pre_buffer": False,
            "memory_map": False,
            "buffer_size": 0,
            "bytes_per_token": args.bytes_per_token,
        },
        "result": {
            "files_done": files_done,
            "batches_done": batches_done,
            "documents": documents,
            "null_texts": null_texts,
            "text_bytes": text_bytes,
            "text_gib": text_gib,
            "approx_tokens": text_bytes / args.bytes_per_token,
            "elapsed_seconds": elapsed,
            "baseline_rss_gib": gib(baseline_rss),
            "final_rss_gib": gib(rss),
            "peak_rss_gib": gib(peak_rss),
            "median_sampled_rss_gib": median_rss_mib / 1024,
            "rss_growth_mib": growth_mib,
            "rss_growth_mib_per_text_gib": growth_mib_per_text_gib,
            "baseline_arrow_mib": mib(baseline_arrow),
            "final_arrow_mib": mib(arrow_bytes),
            "peak_arrow_mib": mib(peak_arrow),
        },
        "samples": samples,
        "files": file_summaries,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine_ram_gib": psutil.virtual_memory().total / 1024**3,
            "logical_cpus": psutil.cpu_count(),
            "versions": {
                package: version(package)
                for package in ("pyarrow", "huggingface-hub", "psutil")
            },
        },
    }

    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print_progress(
        started=started,
        files_done=files_done,
        batches_done=batches_done,
        documents=documents,
        text_bytes=text_bytes,
        baseline_rss=baseline_rss,
        peak_rss=peak_rss,
        rss=rss,
        arrow_bytes=arrow_bytes,
        bytes_per_token=args.bytes_per_token,
    )
    if growth_mib_per_text_gib is not None:
        print(
            f"Done. RSS growth: {growth_mib:.0f} MiB over {text_gib:.2f} GiB of decoded text "
            f"({growth_mib_per_text_gib:.1f} MiB/GiB).",
            flush=True,
        )
    print(f"Report: {result_path}", flush=True)


if __name__ == "__main__":
    main()
