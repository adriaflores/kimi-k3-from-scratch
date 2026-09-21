import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import os
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import psutil
from datasets import load_dataset
from huggingface_hub import HfApi
from pyarrow.dataset import ParquetFragmentScanOptions
from transformers import AutoTokenizer

from prepare_data import DATASET, TOKENIZER, ShardWriter, load_tokenizer, split_for

os.environ["USE_TORCH"] = "0"

def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", type=Path, default=Path("results/tokenizer_batches"))
	parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256])
	parser.add_argument("--max-documents", type=int, default=10_000)
	parser.add_argument("--repeats", type=int, default=3, help="Use 3 for more stable medians.")
	parser.add_argument("--near-best", type=float, default=0.95, help="Minimum fraction of best local throughput.")
	parser.add_argument("--max-ram-gib", type=float, help="Optional measured worker RSS budget.")
	parser.add_argument("--config", default="sample-10BT")
	parser.add_argument("--revision", default="main")
	parser.add_argument("--tokenizer", default=TOKENIZER)
	parser.add_argument("--tokenizer-revision", default="main")
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--shuffle-buffer", type=int, default=1000)
	parser.add_argument("--shard-tokens", type=int, default=1_000_000)
	parser.add_argument("--val-fraction", type=float, default=0.05)
	args = parser.parse_args()
	if min(args.batch_sizes + [args.max_documents, args.repeats, args.shuffle_buffer, args.shard_tokens]) <= 0:
		parser.error("Batch sizes, documents, repeats, buffer and shard size must be positive.")
	if not 0 < args.near_best <= 1 or not 0 < args.val_fraction < 1:
		parser.error("Require 0 < near-best <= 1 and 0 < val-fraction < 1.")
	if not 0 <= args.seed < 2**32 or (args.max_ram_gib is not None and args.max_ram_gib <= 0):
		parser.error("Invalid seed or RAM budget.")
	return args


def source_texts(job):
	cfg = job["config"]
	dataset = load_dataset(
		DATASET, name=cfg["config"], revision=job["dataset_sha"], split="train",
		streaming=True, columns=["text"],
		fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False),
	).shuffle(seed=cfg["seed"], buffer_size=cfg["shuffle_buffer"])
	for row in dataset.take(cfg["max_documents"]):
		text = row["text"]
		if not isinstance(text, str):
			raise TypeError("Expected a string in the text column.")
		if text.strip():
			yield text


class PeakRSS:
	"""Sample process memory, including the tokenizer's native allocations."""

	def __enter__(self):
		self.process = psutil.Process()
		self.baseline = self.peak = self.process.memory_info().rss
		self.stop = threading.Event()
		self.thread = threading.Thread(target=self.sample, daemon=True)
		self.thread.start()
		return self

	def sample(self):
		while not self.stop.wait(0.01):
			self.peak = max(self.peak, self.process.memory_info().rss)

	def __exit__(self, *exc):
		self.stop.set()
		self.thread.join()
		self.peak = max(self.peak, self.process.memory_info().rss)


def encode(tokenizer, texts, batch_size):
	if batch_size == 1:
		return [tokenizer.encode(texts[0], add_special_tokens=False, truncation=False, verbose=False)]
	return tokenizer(
		texts, add_special_tokens=False, padding=False, truncation=False,
		return_attention_mask=False, return_token_type_ids=False, verbose=False,
	)["input_ids"]


def worker(job, batch_size, streaming):
	cfg, info = job["config"], job["tokenizer"]
	tokenizer = AutoTokenizer.from_pretrained(
		info["repository"], revision=info["revision"], use_fast=True, local_files_only=True,
	)
	# Warm native code/thread pools without including startup in the timer.
	encode(tokenizer, ["Warm up tokenization."] * min(batch_size, 16), batch_size)
	if streaming:
		texts = source_texts(job)  # Lazy: remote setup happens inside the timer.
	else:
		with Path(job["sample"]).open(encoding="utf-8") as stream:
			texts = iter([json.loads(line) for line in stream])

	dtype = np.dtype("<u2" if info["vocab_size"] <= 2**16 else "<u4")
	with TemporaryDirectory(prefix="shards-", dir=cfg["output_dir"]) as directory, PeakRSS() as ram:
		start = time.perf_counter()
		writers = {s: ShardWriter(Path(directory), s, cfg["shard_tokens"], dtype) for s in ("train", "val")}
		token_seconds = 0.0
		text_tokens = 0
		while batch := list(islice(texts, batch_size)):
			tick = time.perf_counter()
			encoded = encode(tokenizer, batch, batch_size)
			token_seconds += time.perf_counter() - tick
			for text, ids in zip(batch, encoded, strict=True):
				text_tokens += len(ids)
				ids.append(tokenizer.eos_token_id)
				writers[split_for(text, cfg["seed"], cfg["val_fraction"])].append(ids)
		for writer in writers.values():
			writer.flush()
		seconds = time.perf_counter() - start
		signature = {split: writer.summary() for split, writer in writers.items()}
		total_tokens = sum(writer.tokens for writer in writers.values())
	if not text_tokens or not writers["train"].tokens:
		raise ValueError("Sample has no text/train tokens. Increase --max-documents.")
	return {
		"batch_size": batch_size, "streaming": streaming,
		"seconds": seconds, "tokenization_seconds": token_seconds,
		"text_tokens": text_tokens, "tokens_with_eos": total_tokens,
		"tokenization_tokens_s": text_tokens / token_seconds,
		"pipeline_tokens_s": total_tokens / seconds,
		"train_tokens_s": writers["train"].tokens / seconds,
		"peak_rss_gib": ram.peak / 2**30, "baseline_rss_gib": ram.baseline / 2**30,
		"signature": signature,
	}


def run_worker(job_path, batch_size, streaming=False):
	result = subprocess.run(
		[sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path),
		 str(batch_size), str(int(streaming))],
		check=True, stdout=subprocess.PIPE, text=True,
	)
	return json.loads(result.stdout)


def main():
	args = parse_args()
	output = args.output_dir.resolve()
	output.mkdir(parents=True, exist_ok=False)
	cfg = {**vars(args), "output_dir": str(output)}
	sha = HfApi().dataset_info(DATASET, revision=args.revision).sha
	if not sha:
		raise RuntimeError("Could not resolve the dataset SHA.")
	tokenizer, info = load_tokenizer(args.tokenizer, args.tokenizer_revision)
	del tokenizer
	if info["vocab_size"] > 2**32:
		raise ValueError("Tokenizer IDs do not fit in uint32.")
	sample = output / "sample.jsonl"
	job = {"config": cfg, "dataset_sha": sha, "tokenizer": info, "sample": str(sample)}
	print("Downloading the fixed sample...", flush=True)
	started = time.perf_counter()
	count = 0
	with sample.open("w", encoding="utf-8", newline="\n") as stream:
		for text in source_texts(job):
			stream.write(json.dumps(text, ensure_ascii=False) + "\n")
			count += 1
	if not count:
		raise ValueError("Empty sample.")
	acquisition_seconds = time.perf_counter() - started
	with sample.open("rb") as stream:
		sample_sha = hashlib.file_digest(stream, "sha256").hexdigest()
	job_path = output / "job.json"
	job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

	rows, runs, reference = [], [], None
	print(f"{count:,} documents. RSS is per worker; token counts include EOS only in pipeline rates.")
	print(f"{'Batch':>6} {'Tokenizer tok/s':>17} {'Local pipeline tok/s':>22} {'Peak GiB':>10} {'Hashes':>8}")
	for size in sorted(set([1] + args.batch_sizes)):
		group = [run_worker(job_path, size) for _ in range(args.repeats)]
		if reference is None:
			reference = group[0]["signature"]
		matches = all(run["signature"] == reference for run in group)
		row = {
			"batch_size": size,
			"tokenization_tokens_s": statistics.median(r["tokenization_tokens_s"] for r in group),
			"pipeline_tokens_s": statistics.median(r["pipeline_tokens_s"] for r in group),
			"peak_rss_gib": max(r["peak_rss_gib"] for r in group),
			"checksums_match": matches,
		}
		rows.append(row)
		runs.extend(group)
		print(f"{size:6d} {row['tokenization_tokens_s']:17,.0f} {row['pipeline_tokens_s']:22,.0f} "
			  f"{row['peak_rss_gib']:10.3f} {str(matches):>8}", flush=True)

	eligible = [r for r in rows if r["checksums_match"] and
				(args.max_ram_gib is None or r["peak_rss_gib"] <= args.max_ram_gib)]
	chosen = None
	streaming_result = None
	if eligible:
		best = max(r["pipeline_tokens_s"] for r in eligible)
		chosen = min(r["batch_size"] for r in eligible if r["pipeline_tokens_s"] >= args.near_best * best)
		print(f"Candidate: {chosen}. Measuring the streaming pipeline...", flush=True)
		streaming_result = run_worker(job_path, chosen, streaming=True)
		streaming_result["checksums_match"] = streaming_result["signature"] == reference
		streaming_result["estimated_hours_for_6B_train_tokens"] = 6_000_000_000 / streaming_result["train_tokens_s"] / 3600
		if not streaming_result["checksums_match"] or (args.max_ram_gib is not None and
				streaming_result["peak_rss_gib"] > args.max_ram_gib):
			chosen = None
	report = {
		**job, "sample_documents": count, "sample_sha256": sample_sha,
		"sample_acquisition_seconds": acquisition_seconds,
		"summary": rows, "runs": runs, "streaming": streaming_result,
		"recommended_batch_size": chosen,
		"recommendation_rule": "smallest batch within near-best of best eligible local pipeline throughput",
		"environment": {
			"python": platform.python_version(), "platform": platform.platform(),
			"logical_cpus": psutil.cpu_count(), "machine_ram_gib": psutil.virtual_memory().total / 2**30,
			"TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM"),
			"RAYON_NUM_THREADS": os.getenv("RAYON_NUM_THREADS"),
			"versions": {name: version(name) for name in ("transformers", "tokenizers", "datasets", "pyarrow", "numpy", "psutil")},
		},
	}
	(output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	if streaming_result:
		print(f"Streaming: {streaming_result['pipeline_tokens_s']:,.0f} tok/s; "
			  f"{streaming_result['peak_rss_gib']:.3f} GiB; "
			  f"6B train tokens: ~{streaming_result['estimated_hours_for_6B_train_tokens']:.1f} h (rough estimate).")
	print(f"Recommended batch: {chosen if chosen is not None else 'none: inspect checksums / RAM'}. "
		  f"Report: {output / 'results.json'}")


if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] == "--worker":
		job = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
		print(json.dumps(worker(job, int(sys.argv[3]), bool(int(sys.argv[4])))))
	else:
		main()
