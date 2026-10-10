"""Run short, real-data Branch 3 benchmarks using the existing training CLI.

From the repository root:
  uv run python scripts/benchmark_pretraining_proxies.py --data-dir C:/datasets/fineweb_edu_6b

Only orchestration and log analysis live here. The backbone/trainer are unchanged.
The result ZIP contains scalar logs/configs; checkpoints stay in the run folders.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
from datetime import datetime, timezone
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "82m": "configs/model/kimi_k3_proxy_82m.yaml",
    "122m": "configs/model/kimi_k3_proxy_122m.yaml",
    "300m": "configs/model/kimi_k3_300m.yaml",
}
TRAIN_CONFIG = "configs/training/branch3_benchmark.yaml"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def commands(args, output):
    micro_tokens = args.microbatch * args.seq_len
    if 4096 % micro_tokens:
        raise ValueError("microbatch * seq_len must divide the fixed 4096-token validation prefix")
    return [(name, [
        sys.executable, "-u", "scripts/train.py", "--data-dir", str(args.data_dir.resolve()),
        "--model-config", MODELS[name], "--training-config", TRAIN_CONFIG,
        "--device", "cuda", "--batch-size", str(args.microbatch),
        "--seq-len", str(args.seq_len),
        "--gradient-accumulation-steps", str(args.accumulation),
        "--max-steps", str(args.steps), "--eval-batches", str(4096 // micro_tokens),
        "--checkpoint-interval", str(args.steps), "--run-dir", str(output / name),
    ]) for name in args.models]


def environment():
    # Run in a separate process so this orchestrator retains no CUDA allocations.
    code = """
import importlib.metadata as m, json, torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in this Python environment')
if not torch.cuda.is_bf16_supported():
    raise RuntimeError('Native BF16 is unavailable')
p = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
versions = {}
for name in ('torch', 'flash-linear-attention', 'fla-core', 'triton', 'triton-windows'):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': __import__('sys').version, 'torch': str(torch.__version__),
 'cuda': torch.version.cuda, 'packages': versions, 'gpu': p.name,
 'compute_capability': [p.major, p.minor], 'total_memory_bytes': total,
 'free_memory_bytes_before_runs': free}))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def run_logged(command, log):
    started = perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with log.open("w", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
    return returncode, perf_counter() - started


def analyze_run(directory, args, wall_seconds):
    spec = importlib.util.spec_from_file_location("existing_run_summary", ROOT / "scripts/summarize_training_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.summarize_run(directory, after_step=args.exclude_through_step,
                                  expected_step=args.steps)
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    session = manifest["sessions"][-1]
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r["session_id"] == session["session_id"]]
    train = [r for r in records if r["event"] == "train"]
    steady = [r for r in train if r["step"] - r["updates"] >= args.exclude_through_step]
    expected_tokens = args.steps * args.microbatch * args.seq_len * args.accumulation
    if result["logged_tokens"] != expected_tokens or result["skipped_updates"] or result["nonfinite_grad_norms"]:
        raise ValueError("Unexpected token count, skipped update or nonfinite gradient")
    config = session["train_config"]
    if config["overfit_batch"] or not config["muon_compile"] or not config["muon_batched"] or config["precision"] != "bf16":
        raise ValueError("Effective recipe differs from the requested optimized BF16 benchmark")
    if session["model_config"]["kda_implementation"] != "fused":
        raise ValueError("Benchmark did not use fused KDA")
    layer_names = set(train[0]["moe"])
    expected_layers = session["model_config"]["num_layers"] - session["model_config"]["num_dense_layers"]
    if len(layer_names) != expected_layers:
        raise ValueError("Missing MoE diagnostics")
    qb_max = load_max = 0.0
    zero_max = 0
    for record in train:
        if set(record["moe"]) != layer_names:
            raise ValueError("MoE diagnostic layers changed during the run")
        for layer in record["moe"].values():
            if layer["tokens"] != record["tokens"] or sum(layer["expert_counts"]) != 2 * record["tokens"]:
                raise ValueError("MoE assignment count differs from processed training tokens")
            for bias in layer["qb_biases"]:
                if bias is None or not math.isfinite(bias):
                    raise ValueError("Nonfinite QB bias")
                qb_max = max(qb_max, abs(bias))
            load_max = max(load_max, layer["max_load_over_mean"])
            zero_max = max(zero_max, layer["zero_load_experts"])
    end = next(r for r in reversed(records) if r["event"] == "session_end")
    val = [r for r in records if r["event"] == "val"]
    timing = end["session"]
    early_seconds = sum(r["train_seconds"] for r in train if r["step"] <= args.exclude_through_step)
    early_tokens = sum(r["tokens"] for r in train if r["step"] <= args.exclude_through_step)
    rate = result["steady_train_tokens_per_second"]
    auxiliary = timing["eval_seconds"] + timing["checkpoint_seconds"] + timing["reporting_seconds"]
    result.update({
        "process_wall_seconds": wall_seconds,
        "session_phase_seconds": {k: v for k, v in timing.items() if k.endswith("_seconds")},
        "import_model_optimizer_setup_seconds_approx": max(0.0, wall_seconds - timing["elapsed_seconds"]),
        "first_excluded_windows_train_seconds": early_seconds,
        "startup_excess_train_seconds_approx": max(0.0, early_seconds - early_tokens / rate) if rate else None,
        "startup_excess_note": "Includes compilation, state allocation, routing differences and warmup; not isolated compiler time.",
        "auxiliary_fraction_of_session": auxiliary / timing["elapsed_seconds"],
        "validation_prefix_tokens": 4096,
        "validation": [{"step": r["step"], "tokens_seen": r["tokens_seen"], "loss": r["val_loss"],
                        "seconds": r["eval_seconds"]} for r in val],
        "steady_windows": [{"start_step": r["step"] - r["updates"], "end_step": r["step"],
                            "tokens_per_second": r["train_tokens_per_second"], "seconds": r["train_seconds"]} for r in steady],
        "steady_window_min_tokens_per_second": min(r["train_tokens_per_second"] for r in steady),
        "steady_window_max_tokens_per_second": max(r["train_tokens_per_second"] for r in steady),
        "steady_window_median_tokens_per_second": statistics.median(r["train_tokens_per_second"] for r in steady),
        "moe": {"layers": len(layer_names), "qb_bias_abs_max": qb_max,
                "max_load_over_mean_max": load_max, "zero_load_experts_max_in_one_window": zero_max},
    })
    return result


def archive_results(output):
    archive = output / "branch3_results.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".yaml"):
                zip_file.write(path, path.relative_to(output).as_posix())
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seq-len", type=int, choices=(64, 128, 256, 512), default=256)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--exclude-through-step", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Print the exact commands without using CUDA or creating files")
    args = parser.parse_args(argv)
    if args.microbatch < 1 or args.accumulation < 1 or args.steps < 40 or args.steps % 10:
        parser.error("microbatch/accumulation must be positive; steps must be a multiple of 10 and >= 40")
    if args.exclude_through_step < 10 or args.exclude_through_step % 10 or args.exclude_through_step > args.steps - 20:
        parser.error("exclude-through-step must be a multiple of 10 with at least two later logging windows")
    if len(set(args.models)) != len(args.models):
        parser.error("models must not be repeated")
    output = ROOT / "results/training/branch3" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    required = [ROOT / "scripts/train.py", ROOT / "scripts/summarize_training_run.py", ROOT / TRAIN_CONFIG,
                *(ROOT / MODELS[name] for name in args.models)]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing repository files: " + ", ".join(missing))
    try:
        plan = commands(args, output)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"runs": [{"model": name, "argv": command} for name, command in plan]}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"purpose": "Branch 3 feasibility/timing; not a comparison of optimal learning rates",
                "source_files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in required},
                "planned_runs": [{"model": name, "argv": command} for name, command in plan], "runs": []}
    snapshots = output / "configs"
    snapshots.mkdir()
    for path in [ROOT / TRAIN_CONFIG, *(ROOT / MODELS[name] for name in args.models)]:
        (snapshots / path.name).write_bytes(path.read_bytes())
    status = 0
    try:
        metadata["environment"] = environment()
        for name, command in plan:
            print(f"\nBranch 3 benchmark: {name}\n", flush=True)
            returncode, wall_seconds = run_logged(command, output / (name + ".stdout.log"))
            entry = {"model": name, "returncode": returncode, "process_wall_seconds": wall_seconds}
            metadata["runs"].append(entry)
            if returncode:
                entry["error"] = "Training failed; see the captured stdout log. Remaining runs were not started."
                status = 1
                break
            summary = analyze_run(output / name, args, wall_seconds)
            write_json(output / name / "benchmark_summary.json", summary)
            entry["summary"] = summary
            rate = summary["steady_train_tokens_per_second"]
            peak = summary["memory"]["cuda_peak_allocated_bytes"] / 1024**3
            print(f"\n{name}: {rate:.1f} steady tok/s | peak allocated {peak:.2f} GiB | {wall_seconds:.1f}s total\n", flush=True)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        metadata["error"] = str(error)
        print(f"\nBenchmark stopped: {error}", file=sys.stderr)
        status = 1
    finally:
        write_json(output / "benchmark_report.json", metadata)
        archive = archive_results(output)
        print(f"\nShare this results file: {archive}\n", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
