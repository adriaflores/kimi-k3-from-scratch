"""Summarize the latest training session from JSON logs, without loading tensors.

Run from any directory: python scripts/summarize_training_run.py RUN_DIRECTORY.
Read one JSONL record at a time; never load the checkpoint or profiler trace.
"""

import argparse
import json
import math
from pathlib import Path


def _finite(value, name, *, minimum=0):
	if (isinstance(value, bool) or not isinstance(value, (int, float))
			or not math.isfinite(value) or value < minimum):
		raise ValueError(f"Invalid {name}: expected a finite number >= {minimum}")
	return value


def summarize_run(directory, *, after_step=0, expected_step=None):
	"""Latest session only; steady rates include whole windows after after_step.

	Losses compare the first and last logged windows, not individual batches.
	Memory peaks cover the session, including warmup and auxiliary phases.
	Completion/count checks concern logs, not checkpoint tensor equivalence.
	"""
	if isinstance(after_step, bool) or not isinstance(after_step, int) or after_step < 0:
		raise ValueError("after_step must be a nonnegative integer")
	directory = Path(directory)
	manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
	if manifest.get("schema_version") != 1 or not manifest.get("sessions"):
		raise ValueError("Unsupported or empty run manifest")
	session = manifest["sessions"][-1]
	config, start = session["train_config"], session["start_state"]
	first_loss = last_loss = max_logit = grad_max = None
	updates = tokens = skips = nonfinite = clipped_heads = observed_heads = windows = 0
	steady_tokens = 0
	steady_seconds = clipped_updates = 0.0
	last = end = None
	memory = dict.fromkeys(("ram_rss_sampled_peak_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"))
	with (directory / "metrics.jsonl").open(encoding="utf-8") as stream:
		for line_number, line in enumerate(stream, 1):
			if not line.endswith("\n"):
				raise ValueError(f"Uncommitted metrics tail at line {line_number}")
			record = json.loads(line)
			if record["session_id"] != session["session_id"]:
				continue
			if record["run_id"] != manifest["run_id"]:
				raise ValueError("Metric run identity differs from the manifest")
			last = record
			for name in memory:
				value = record.get("memory", {}).get(name)
				if value is not None:
					memory[name] = max(memory[name] or 0, _finite(value, name))
			if record["event"] == "session_end":
				end = record
			if record["event"] == "val":
				_finite(record["val_loss"], "validation loss")
			if record["event"] != "train":
				continue
			loss = _finite(record["train_loss"], "training loss")
			_finite(record["lr"], "learning rate")
			_finite(record["grad_norm_mean"], "gradient norm mean")
			grad_max = max(grad_max or 0, _finite(record["grad_norm_max"], "gradient norm maximum"))
			if first_loss is None:
				first_loss = loss
			last_loss = loss
			count, skipped = record["updates"], record["skipped_updates_in_window"]
			if count < 1 or skipped < 0 or record["attempted_updates"] != count + skipped:
				raise ValueError("Inconsistent update counts in training window")
			if record["step"] != start["step"] + updates + count:
				raise ValueError("Discontinuous successful-update history")
			attempt_tokens = config["batch_size"] * config["seq_len"] * config["gradient_accumulation_steps"]
			if record["tokens"] != attempt_tokens * record["attempted_updates"]:
				raise ValueError("Token count differs from the configured microbatches")
			clip = record["qk_clip"]
			if clip["threshold"] != config["qk_clip_threshold"]:
				raise ValueError("QK-Clip threshold differs from the session recipe")
			observed, clipped = clip["observed_head_updates"], clip["clipped_head_updates"]
			if not 0 <= clipped <= observed:
				raise ValueError("Invalid QK-Clip head-update counts")
			if clip["max_logit"] is not None:
				value = _finite(clip["max_logit"], "maximum logit", minimum=-math.inf)
				max_logit = value if max_logit is None else max(max_logit, value)
			clipped_heads += clipped
			observed_heads += observed
			updates += count
			tokens += record["tokens"]
			skips += skipped
			nonfinite += record["nonfinite_grad_norms"]
			fraction = _finite(record["clipped_fraction"], "gradient clipping fraction")
			if fraction > 1:
				raise ValueError("Gradient clipping fraction exceeds one")
			clipped_updates += fraction * count
			windows += 1
			# Exclude a straddling window in full, rather than mixing warmup in.
			if record["step"] - count >= after_step:
				steady_tokens += record["tokens"]
				steady_seconds += _finite(record["train_seconds"], "training seconds")
	if not windows:
		raise ValueError("No logged training windows for the latest session")
	if end is not None and (updates != end["step"] - start["step"]
			or tokens != end["tokens_seen"] - start["tokens_seen"]
			or skips != end["skipped_updates"] - start["skipped_updates"]):
		raise ValueError("Completed session counters differ from its logged windows")
	if expected_step is not None and (end is None or end["step"] != expected_step):
		raise ValueError(f"Expected normal completion at step {expected_step}")
	return {"run_id": manifest["run_id"], "session_id": session["session_id"],
		"sessions": len(manifest["sessions"]), "completed": end is not None,
		"start_step": start["step"], "last_step": last["step"], "logged_updates": updates,
		"logged_tokens": tokens, "skipped_updates": skips, "nonfinite_grad_norms": nonfinite,
		"recipe": config, "hardware": session["environment"]["hardware"],
		"optimizer_algorithms": sorted({group["algorithm"] for group in session["optimizer"]["param_groups"]}),
		"first_logged_train_loss": first_loss, "last_logged_train_loss": last_loss,
		"gradient_norm_max": grad_max, "gradient_clipped_fraction": clipped_updates / updates,
		"qk_clip": {"threshold": config["qk_clip_threshold"], "max_logit": max_logit,
			"clipped_head_updates": clipped_heads, "observed_head_updates": observed_heads,
			"clipped_fraction": clipped_heads / observed_heads if observed_heads else None},
		"steady_train_tokens_per_second": steady_tokens / steady_seconds if steady_seconds else None,
		"steady_logged_tokens": steady_tokens, "exclude_through_step": after_step,
		"session_wall_tokens_per_second": last["session"]["tokens_per_second"],
		"profiled": last["session"].get("profiling_seconds", 0) > 0,
		"memory": memory}


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("run_directory", type=Path)
	parser.add_argument("--after-step", type=int, default=0, help="Exclude any train window containing an earlier step")
	parser.add_argument("--expect-step", type=int, help="Require normal completion at this total successful-update step")
	args = parser.parse_args(argv)
	try:
		result = summarize_run(args.run_directory, after_step=args.after_step, expected_step=args.expect_step)
	except (ValueError, KeyError, OSError) as error:
		parser.exit(1, f"Invalid training run: {error}\n")
	print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
	main()
