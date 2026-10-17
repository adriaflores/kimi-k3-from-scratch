"""Compare complete MuonAdamW execution modes on CUDA without a profiler.

Uses the model's actual parameter groups and fixed synthetic gradients for all
parameters. This isolates optimizer cost; it does not measure training tokens/s.
Compare individual eager/compiled NS, or individual/batched compiled NS.
Run from the repository root: uv run python scripts/bench_muon.py --help.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.model import KimiK3, KimiK3Config
from kimi_k3.training.optimizers import (
	MuonAdamW, _batch_capacity, _compiled_newton_schulz, zeropower_via_newton_schulz,
)
from kimi_k3.training.parameter_groups import build_parameter_groups
from kimi_k3.utils import load_config


def _blueprint(path):
	# No forward is performed. The reference has the same parameter geometry
	# and allows inspecting a fused-KDA config without importing FLA/Triton.
	config = replace(load_config(path, KimiK3Config), kda_implementation="chunkwise")
	with torch.device("meta"):
		model = KimiK3(config)
	groups = build_parameter_groups(model, 0.1)
	return [{**group, "params": [tuple(parameter.shape) for parameter in group["params"]]}
		for group in groups]


def _materialize(blueprint, device, seed):
	generator = torch.Generator(device=device).manual_seed(seed)
	groups = []
	for group in blueprint:
		parameters = []
		for shape in group["params"]:
			parameter = torch.nn.Parameter(torch.randn(shape, device=device, generator=generator) * 0.02)
			parameter.grad = torch.randn(shape, device=device, generator=generator) * 0.01
			parameters.append(parameter)
		groups.append({**group, "params": parameters})
	return groups


def _measure(blueprint, device, *, compiled, batched, batch_size, batch_max_mb, warmup, repeats, seed):
	groups = _materialize(blueprint, device, seed)
	optimizer = MuonAdamW(groups, ns_compile=compiled, ns_batched=batched,
		ns_batch_size=batch_size, ns_batch_max_bytes=batch_max_mb * 1024**2,
		ns_steps=5, ns_dtype=torch.bfloat16,
		adamw_fused=True)
	if compiled:
		_compiled_newton_schulz.cache_clear()
	torch.cuda.synchronize(device)
	torch.cuda.reset_peak_memory_stats(device)
	start = time.perf_counter()
	optimizer.step()
	torch.cuda.synchronize(device)
	first_step = time.perf_counter() - start
	for _ in range(warmup):
		optimizer.step()
	torch.cuda.synchronize(device)
	walls, gpu_times = [], []
	begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	for _ in range(repeats):
		start = time.perf_counter()
		begin.record()
		optimizer.step()
		end.record()
		end.synchronize()
		walls.append((time.perf_counter() - start) * 1000)
		gpu_times.append(begin.elapsed_time(end))
	cache = _compiled_newton_schulz.cache_info()._asdict() if compiled else None
	result = {"backend": "compiled" if compiled else "eager", "batched": batched,
		"first_step_seconds": first_step,
		"warmup_updates_after_first": warmup, "measured_updates": repeats,
		"wall_ms_median": statistics.median(walls), "wall_ms_mean": statistics.mean(walls),
		"wall_ms_min": min(walls), "wall_ms_max": max(walls),
		"cuda_event_ms_median": statistics.median(gpu_times),
		"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
		"cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
		"compile_cache": cache,
		"ns_calls_per_update": (cache["hits"] + cache["misses"]) / (1 + warmup + repeats) if cache else None}
	# Each mode starts from the same weights/gradients with fresh momentum and
	# AdamW state. Keep only one model's storage on the GPU at any time.
	del optimizer, groups
	gc.collect()
	torch.cuda.empty_cache()
	return result


def _verify_core(blueprint, device, seed, *, batched, batch_size, batch_max_mb):
	"""Compare every distinct logical layout after timing, including K/V strides."""
	generator = torch.Generator(device=device).manual_seed(seed)
	seen, results = set(), []
	for group in blueprint:
		if group["algorithm"] != "muon":
			continue
		for shape in group["params"]:
			view = group.get("muon_shape") or shape
			# Shape/stride depend only on storage/view, not on its numeric contents.
			meta = torch.empty(shape, device="meta").reshape(view)
			splits = group.get("muon_split_sizes")
			parts = meta.split(splits, dim=-2) if splits else (meta,)
			keys = [(tuple(part.shape), tuple(part.stride())) for part in parts]
			if all(key in seen for key in keys):
				continue
			capacity = _batch_capacity(view, splits, torch.float32, torch.bfloat16,
				batch_size, batch_max_mb * 1024**2, sum(item == shape for item in group["params"]))
			lanes = capacity if batched else 1
			matrix = torch.randn((lanes, *shape), device=device, generator=generator).reshape(lanes, *view)
			parts = matrix.split(splits, dim=-2) if splits else (matrix,)
			for part, key in zip(parts, keys):
				if key in seen:
					continue
				seen.add(key)
				eager = torch.stack([zeropower_via_newton_schulz(item) for item in part])
				compiled = (zeropower_via_newton_schulz(part, compiled=True) if lanes > 1
					else torch.stack([zeropower_via_newton_schulz(part[0], compiled=True)]))
				torch.testing.assert_close(compiled, eager, rtol=0.03, atol=0.03)
				difference = compiled.float() - eager.float()
				results.append({"shape": key[0], "stride": key[1], "parameter_lanes": lanes,
					"max_absolute_difference": difference.abs().max().item(),
					"relative_l2_difference": (difference.norm() / eager.float().norm()).item()})
	return {"rtol": 0.03, "atol": 0.03, "layouts": results}


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-config", type=Path, required=True)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--warmup", type=int, default=5)
	parser.add_argument("--repeats", type=int, default=30)
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--compare", choices=("compile", "batch"), default="compile",
		help="compile: individual eager vs compiled; batch: individual compiled vs batched compiled")
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--batch-max-mb", type=int, default=64)
	parser.add_argument("--order", choices=("eager-first", "compiled-first", "reference-first", "candidate-first"), default="reference-first")
	parser.add_argument("--output", type=Path, help="Write the JSON result as well as printing it")
	args = parser.parse_args(argv)
	device = torch.device(args.device)
	if device.type != "cuda" or not torch.cuda.is_available():
		parser.error("This comparison requires CUDA; CPU optimizer execution stays eager")
	if args.warmup < 1 or args.repeats < 2:
		parser.error("Use --warmup >= 1 and --repeats >= 2")
	if args.batch_size < 1 or args.batch_max_mb < 1:
		parser.error("Use positive --batch-size and --batch-max-mb")
	if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
		parser.error("Remove TORCH_COMPILE_DISABLE=1 before comparing compilation")
	with torch.cuda.device(device):
		blueprint = _blueprint(args.model_config)
		order = (False, True) if args.order in ("eager-first", "reference-first") else (True, False)
		results = []
		for candidate in order:
			compiled = args.compare == "batch" or candidate
			batched = args.compare == "batch" and candidate
			print(f"Measuring {'compiled' if compiled else 'eager'}, {'batched' if batched else 'individual'}; first compilation can take minutes...",
				file=sys.stderr, flush=True)
			result = _measure(blueprint, device, compiled=compiled, batched=batched,
				batch_size=args.batch_size, batch_max_mb=args.batch_max_mb, warmup=args.warmup,
				repeats=args.repeats, seed=args.seed)
			result["role"] = "candidate" if candidate else "reference"
			results.append(result)
		print("Verifying eager/compiled arithmetic for every logical layout...", file=sys.stderr, flush=True)
		verification = _verify_core(blueprint, device, args.seed, batched=args.compare == "batch",
			batch_size=args.batch_size, batch_max_mb=args.batch_max_mb)
	by_mode = {result["role"]: result for result in results}
	report = {"torch": torch.__version__, "cuda": torch.version.cuda,
		"device": torch.cuda.get_device_name(device), "model_config": str(args.model_config),
		"seed": args.seed, "order": args.order, "compare": args.compare,
		"batch_size": args.batch_size, "batch_max_mb": args.batch_max_mb, "all_gradients_active": True,
		"ns_steps": 5, "ns_dtype": "bf16", "parameter_count": sum(
			torch.Size(shape).numel() for group in blueprint for shape in group["params"]),
		"results": results, "core_verification": verification, "optimizer_wall_speedup":
			by_mode["reference"]["wall_ms_median"] / by_mode["candidate"]["wall_ms_median"]}
	payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(payload, encoding="utf-8")
	print(payload, end="")


if __name__ == "__main__":
	main()
