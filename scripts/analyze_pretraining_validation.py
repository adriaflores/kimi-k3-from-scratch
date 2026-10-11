"""Audit Branch 6 300M runs and measured proxy references; export CSV, PNG/PDF and a report."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import yaml

MANIFEST_HASH = "5dc9200c399f4aa46e71a17dc9cd472af66b5564f662e0db93a4a9c2ae1cf579"


def write_json(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
	temporary.replace(path)


def canonical_hash(value):
	return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
									 allow_nan=False).encode("utf-8")).hexdigest()


def finite(value, name, *, minimum=0):
	if (isinstance(value, bool) or not isinstance(value, (float, int))
			or not math.isfinite(value) or value < minimum):
		raise ValueError(f"Invalid {name}: expected a finite number >= {minimum}")
	return value


def integer(value, name, *, minimum=0):
	if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
		raise ValueError(f"Invalid {name}: expected an integer >= {minimum}")
	return value


class Evidence:
	"""Read the runner's directory or ZIP directly; never extract or execute it."""

	def __init__(self, path):
		self.path = Path(path)
		self.archive = None if self.path.is_dir() else ZipFile(self.path)

	def exists(self, name):
		return ((self.path / name).is_file() if self.archive is None
				else name in self.archive.namelist())

	def read(self, name):
		return ((self.path / name).read_bytes() if self.archive is None
				else self.archive.read(name))

	def json(self, name):
		return json.loads(self.read(name).decode("utf-8-sig"))

	def close(self):
		if self.archive is not None:
			self.archive.close()


def expected_recipe(common, run):
	batch = run["effective_batch_tokens"]
	return {
		"batch_size": common["microbatch_sequences"], "seq_len": common["seq_len"],
		"learning_rate": run["learning_rate"], "weight_decay": 0.1,
		"max_steps": common["training_tokens_D"] // batch,
		"log_interval": common["log_interval_tokens_L"] // batch,
		"eval_interval": common["eval_interval_tokens_E"] // batch,
		"eval_batches": common["validation_prefix_tokens_V"] //
						(common["microbatch_sequences"] * common["seq_len"]),
		"checkpoint_interval": common["checkpoint_interval_tokens"] // batch,
		"overfit_batch": False, "gradient_accumulation_steps": run["gradient_accumulation_steps"],
		"precision": "bf16", "adam_beta1": 0.9, "adam_beta2": 0.95,
		"adam_eps": 1e-8, "adam_fused": True, "muon_momentum": 0.95,
		"muon_ns_steps": 5, "muon_ns_dtype": "bf16", "muon_compile": True,
		"muon_batched": True, "muon_batch_size": 32, "muon_batch_max_mb": 64,
		"qk_clip_threshold": 100.0, "qk_clip_chunk_size": 256,
		"lr_schedule": "cosine", "warmup_steps": common["warmup_tokens_W"] // batch,
		"decay_steps": common["training_tokens_D"] // batch,
		"min_lr_ratio": 0.1, "max_grad_norm": 1.0, "seed": common["seed"],
	}


def lr_at_step(config, step):
	# This is the trainer's documented 1-based schedule, used to check its logs.
	if step <= config["warmup_steps"]:
		return config["learning_rate"] * step / config["warmup_steps"]
	if config["lr_schedule"] == "constant":
		return config["learning_rate"]
	start = max(1, config["warmup_steps"])
	progress = min(1.0, (step - start) / (config["decay_steps"] - start))
	factor = config["min_lr_ratio"] + (1 - config["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))
	return config["learning_rate"] * factor


def read_records(data):
	lines = data.decode("utf-8").splitlines(keepends=True)
	records, warnings = [], []
	for index, line in enumerate(lines):
		if not line.endswith("\n"):
			if index != len(lines) - 1:
				raise ValueError("Uncommitted metrics line before EOF")
			warnings.append("Uncommitted final metrics line; it is excluded.")
			break
		try:
			records.append(json.loads(line))
		except json.JSONDecodeError as error:
			raise ValueError(f"Invalid committed metrics line {index + 1}") from error
	return records, warnings


def reconstruct(manifest, records):
	"""Return one trajectory, with old work beyond a resumed cursor removed."""
	sessions = manifest.get("sessions", [])
	if manifest.get("schema_version") != 1 or not sessions:
		raise ValueError("Unsupported or empty run manifest")
	ids = [session["session_id"] for session in sessions]
	if len(set(ids)) != len(ids):
		raise ValueError("Duplicate session identities")
	groups = {session_id: [] for session_id in ids}
	for record in records:
		if record.get("run_id") != manifest["run_id"] or record.get("session_id") not in groups:
			raise ValueError("Metrics identity differs from run.json")
		groups[record["session_id"]].append(record)
	trajectory = []
	for index, session in enumerate(sessions):
		start = session["start_state"]
		step = integer(start["step"], "session start step")
		tokens = integer(start["tokens_seen"], "session start tokens")
		if index == 0 and (step != 0 or tokens != 0 or session.get("rng_source") != "seed"):
			raise ValueError("The candidate did not start from seed at token zero")
		if index:
			if session.get("rng_source") != "checkpoint":
				raise ValueError("A later session did not resume a checkpoint")
			prior_steps = {0, *(r["step"] for r in trajectory if r["event"] == "train")}
			if step not in prior_steps:
				raise ValueError("Resume cursor has no matching logged training boundary")
			trajectory = [r for r in trajectory if r["step"] <= step and r["tokens_seen"] <= tokens]
		for record in groups[session["session_id"]]:
			if record["event"] in ("train", "val"):
				trajectory.append(record)
	# A final checkpoint may be resumed without any additional update. Keep one
	# validation value at each cursor, after rollback has removed the old future.
	val_by_tokens = {r["tokens_seen"]: r for r in trajectory if r["event"] == "val"}
	train = [r for r in trajectory if r["event"] == "train"]
	return train, sorted(val_by_tokens.values(), key=lambda r: r["tokens_seen"]), groups


def check_window(record, config, model, previous_step, previous_tokens):
	updates = integer(record["updates"], "updates", minimum=1)
	attempted = integer(record["attempted_updates"], "attempted updates", minimum=1)
	skips = integer(record["skipped_updates_in_window"], "skips")
	batch = config["batch_size"] * config["seq_len"] * config["gradient_accumulation_steps"]
	if (skips or attempted != updates or record["skipped_updates"]
			or record["nonfinite_grad_norms"]):
		raise ValueError("Skipped update or nonfinite gradient in the candidate")
	if record["step"] != previous_step + updates:
		raise ValueError("Discontinuous successful-update history")
	if record["tokens"] != updates * batch or record["tokens_seen"] != previous_tokens + record["tokens"]:
		raise ValueError("Discontinuous token history")
	finite(record["train_loss"], "training loss")
	finite(record["grad_norm_mean"], "gradient norm mean")
	finite(record["grad_norm_max"], "gradient norm maximum")
	if not 0 <= finite(record["clipped_fraction"], "gradient clipping fraction") <= 1:
		raise ValueError("Gradient clipping fraction exceeds one")
	if not math.isclose(finite(record["lr"], "LR"), lr_at_step(config, record["step"]), rel_tol=1e-9, abs_tol=1e-12):
		raise ValueError("Logged LR differs from the frozen token schedule")
	finite(record["train_seconds"], "training seconds")
	clip = record["qk_clip"]
	if clip["threshold"] != config["qk_clip_threshold"]:
		raise ValueError("QK-Clip threshold differs from the recipe")
	expected_heads = model["mla_num_heads"] * model["layer_types"].count("mla")
	if clip["observed_head_updates"] != expected_heads * updates:
		raise ValueError("Missing QK-Clip head observations")
	expected_mla_layers = {f"layers.{i}.attn" for i, kind in enumerate(model["layer_types"]) if kind == "mla"}
	if set(clip["layers"]) != expected_mla_layers:
		raise ValueError("Missing or unexpected MLA QK diagnostic layers")
	clipped = integer(clip["clipped_head_updates"], "clipped head updates")
	if clipped > clip["observed_head_updates"]:
		raise ValueError("Invalid clipped head-update count")
	finite(clip["max_logit"], "QK maximum", minimum=-math.inf)
	if sum(len(layer["max_logits"]) * layer["updates"] for layer in clip["layers"].values()) != clip["observed_head_updates"]:
		raise ValueError("QK-Clip per-layer observations differ from the total")
	if sum(sum(layer["clipped_head_updates"]) for layer in clip["layers"].values()) != clipped:
		raise ValueError("QK-Clip per-head clipping counts differ from the total")
	for layer in clip["layers"].values():
		if layer["updates"] != updates or len(layer["max_logits"]) != model["mla_num_heads"]:
			raise ValueError("QK per-layer observation count differs")
		for value in layer["max_logits"]:
			finite(value, "head max logit", minimum=-math.inf)
		if len(layer["minimum_gamma"]) != model["mla_num_heads"]:
			raise ValueError("Missing per-head minimum gamma")
		if len(layer["clipped_head_updates"]) != model["mla_num_heads"]:
			raise ValueError("Missing per-head clipping counts")
		for count in layer["clipped_head_updates"]:
			if integer(count, "per-head clipping count") > updates:
				raise ValueError("Per-head clipping count exceeds updates")
		for value in layer["minimum_gamma"]:
			if not 0 < finite(value, "minimum gamma") <= 1:
				raise ValueError("QK-Clip gamma is outside (0,1]")
	expected_layers = {f"layers.{i}.ffn" for i in range(model["num_dense_layers"], model["num_layers"])}
	if set(record["moe"]) != expected_layers:
		raise ValueError("Missing MoE diagnostic layers")
	experts, top_k = model["num_experts"], model["num_experts_per_token"]
	for layer in record["moe"].values():
		counts, biases = layer["expert_counts"], layer["qb_biases"]
		if len(counts) != experts or len(biases) != experts:
			raise ValueError("Incorrect expert count or QB bias shape")
		for value in counts:
			integer(value, "expert assignments")
		if layer["tokens"] != record["tokens"] or sum(counts) != top_k * record["tokens"]:
			raise ValueError("MoE assignments differ from the processed tokens")
		for value in biases:
			finite(value, "QB bias", minimum=-math.inf)
		if not math.isclose(layer["max_load_over_mean"], max(counts) * experts / sum(counts), rel_tol=1e-9):
			raise ValueError("MoE load ratio differs from expert counts")
		if layer["zero_load_experts"] != counts.count(0):
			raise ValueError("MoE zero-load count differs from expert counts")


def summarize_candidate(evidence, plan, state, entry, model):
	run_id = entry["id"]
	row = {"id": run_id, "learning_rate": entry["learning_rate"],
		   "effective_batch_tokens": entry["effective_batch_tokens"], "status": "pending",
		   "eligible": False, "issues": [], "warnings": [], "tokens_seen": 0,
		   "final_val_loss": None, "train": [], "validation": [], "moe_windows": [],
		   "expert_totals": {}, "final_qb_biases": {}, "sessions": 0}
	ledger = state["runs"][run_id]
	row["wall_time_complete"] = all(a.get("wall_seconds") is not None for a in ledger.get("attempts", []))
	row["process_wall_seconds"] = sum(finite(a["wall_seconds"], "attempt wall seconds")
									  for a in ledger.get("attempts", []) if a.get("wall_seconds") is not None)
	if not row["wall_time_complete"]:
		row["warnings"].append("A terminated runner did not capture an attempt's wall time; recorded process cost is a lower bound.")
	if ledger.get("restart_note"):
		row["warnings"].append(ledger["restart_note"])
	prefix = f"runs/{run_id}"
	if not evidence.exists(prefix + "/run.json"):
		row["status"] = ledger["status"]
		if ledger.get("error"):
			row["issues"].append(ledger["error"])
		return row
	try:
		manifest = evidence.json(prefix + "/run.json")
		records, tail_warnings = read_records(evidence.read(prefix + "/metrics.jsonl"))
		row["warnings"].extend(tail_warnings)
		config = expected_recipe(plan["common"], entry)
		for session in manifest["sessions"]:
			if session["train_config"] != config:
				raise ValueError("Effective TrainConfig differs from the frozen recipe")
			if any(session["model_config"].get(k) != v for k, v in model.items()):
				raise ValueError("Effective model differs from the frozen 300M model")
			if session["parameter_count"] != 294676808:
				raise ValueError("Parameter count differs from the 300M proxy")
			for split in ("train", "val"):
				if session["data"][split]["metadata_sha256"] != plan["common"]["dataset_metadata_sha256"]:
					raise ValueError("Dataset identity differs from Branch 3")
			start = session["start_state"]
			if (start["skipped_updates"] or start["tokens_seen"] != start["step"] * entry["effective_batch_tokens"]
					or start["loaders"]["train"]["position"] != start["tokens_seen"]
					or start["loaders"]["val"]["position"] != (0 if start["step"] == 0 else plan["common"]["validation_prefix_tokens_V"])):
				raise ValueError("Initial/resumed counters or data cursors differ")
			if {group["algorithm"] for group in session["optimizer"]["param_groups"]} != {"muon", "adamw"}:
				raise ValueError("Unexpected optimizer algorithm")
			if (session["environment"]["execution"]["dtype"] != "torch.float32"
					or not session["environment"]["hardware"]["device"].startswith("cuda")):
				raise ValueError("Expected FP32 parameter storage on CUDA")
		row["runtime_environment"] = manifest["sessions"][0]["environment"]
		row["dataset_identity"] = {s: manifest["sessions"][0]["data"][s]["metadata_sha256"] for s in ("train", "val")}
		row["parameter_count"] = 294676808
		row["model"] = "300m"
		train, val, groups = reconstruct(manifest, records)
		row["sessions"] = len(manifest["sessions"])
		previous_step = previous_tokens = 0
		for record in train:
			check_window(record, config, model, previous_step, previous_tokens)
			previous_step, previous_tokens = record["step"], record["tokens_seen"]
		row["tokens_seen"] = previous_tokens
		row["discarded_logged_tokens"] = sum(r["tokens"] for r in records if r["event"] == "train") - previous_tokens
		for record in val:
			finite(record["val_loss"], "validation loss")
			if (record["tokens_seen"] != record["step"] * entry["effective_batch_tokens"]
					or record["tokens_seen"] % plan["common"]["eval_interval_tokens_E"]):
				raise ValueError("Validation is outside the common token grid")
		if train:
			total_updates = sum(r["updates"] for r in train)
			steady = [r for r in train if r["tokens_seen"] - r["tokens"] >=
					  max(plan["common"]["warmup_tokens_W"], 40960)]
			seconds = sum(r["train_seconds"] for r in steady)
			row.update({
				"last_train_loss": train[-1]["train_loss"],
				"grad_norm_mean": sum(r["grad_norm_mean"] * r["updates"] for r in train) / total_updates,
				"grad_norm_max": max(r["grad_norm_max"] for r in train),
				"grad_clipped_fraction": sum(r["clipped_fraction"] * r["updates"] for r in train) / total_updates,
				"steady_tokens_per_second": sum(r["tokens"] for r in steady) / seconds if seconds else None,
				"qk_max_logit": max(r["qk_clip"]["max_logit"] for r in train),
				"qk_clipped_fraction": sum(r["qk_clip"]["clipped_head_updates"] for r in train) /
									   sum(r["qk_clip"]["observed_head_updates"] for r in train),
				"minimum_gamma": min(v for r in train for layer in r["qk_clip"]["layers"].values()
									 for v in layer["minimum_gamma"]),
				"moe_max_load_over_mean": max(layer["max_load_over_mean"] for r in train for layer in r["moe"].values()),
				"moe_max_zero_load_experts": max(layer["zero_load_experts"] for r in train for layer in r["moe"].values()),
				"qb_bias_abs_max": max(abs(v) for r in train for layer in r["moe"].values() for v in layer["qb_biases"]),
			})
		all_memory = [r.get("memory", {}) for r in records]
		for field in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "ram_rss_sampled_peak_bytes"):
			values = [finite(m[field], field) for m in all_memory if m.get(field) is not None]
			row[field] = max(values, default=None)
		phase_totals = {name: 0.0 for name in ("train_seconds", "eval_seconds", "checkpoint_seconds", "reporting_seconds", "elapsed_seconds")}
		for session_records in groups.values():
			if session_records:
				timing = session_records[-1].get("session", {})
				for name in phase_totals:
					phase_totals[name] += finite(timing.get(name, 0), name)
				if timing.get("profiling_seconds", 0):
					raise ValueError("The comparison contains profiler overhead")
		row["session_phase_seconds"] = phase_totals
		row["wall_tokens_per_second"] = previous_tokens / row["process_wall_seconds"] if row["process_wall_seconds"] else None
		for record in train:
			clip = record["qk_clip"]
			row["train"].append({
				"id": run_id, "step": record["step"], "tokens_seen": record["tokens_seen"],
				"tokens_in_window": record["tokens"], "updates": record["updates"],
				"train_loss": record["train_loss"], "lr": record["lr"],
				"grad_norm_mean": record["grad_norm_mean"], "grad_norm_max": record["grad_norm_max"],
				"grad_clipped_fraction": record["clipped_fraction"],
				"qk_max_logit": clip["max_logit"], "qk_clipped_fraction": clip["clipped_fraction"],
				"minimum_gamma": min(v for layer in clip["layers"].values() for v in layer["minimum_gamma"]),
				"tokens_per_second": record["train_tokens_per_second"], "train_seconds": record["train_seconds"],
				"moe_max_load_over_mean": max(layer["max_load_over_mean"] for layer in record["moe"].values()),
				"moe_max_zero_load_experts": max(layer["zero_load_experts"] for layer in record["moe"].values()),
				"qb_bias_abs_max": max(abs(v) for layer in record["moe"].values() for v in layer["qb_biases"]),
			})
			for name, layer in record["moe"].items():
				counts = layer["expert_counts"]
				probabilities = [count / sum(counts) for count in counts]
				entropy = -sum(p * math.log(p) for p in probabilities if p) / math.log(len(counts))
				row["moe_windows"].append({
					"id": run_id, "tokens_seen": record["tokens_seen"], "layer": name,
					"tokens_in_window": record["tokens"], "max_load_over_mean": layer["max_load_over_mean"],
					"zero_load_experts": layer["zero_load_experts"], "normalized_load_entropy": entropy,
					"qb_bias_abs_max": max(abs(v) for v in layer["qb_biases"]),
				})
				if name not in row["expert_totals"]:
					row["expert_totals"][name] = [0] * len(counts)
				row["expert_totals"][name] = [a + b for a, b in zip(row["expert_totals"][name], counts)]
				row["final_qb_biases"][name] = layer["qb_biases"]
		row["validation"] = [{"id": run_id, "step": r["step"], "tokens_seen": r["tokens_seen"],
							  "val_loss": r["val_loss"], "eval_seconds": r["eval_seconds"]} for r in val]
		final_records = groups[manifest["sessions"][-1]["session_id"]]
		end = next((r for r in reversed(final_records) if r["event"] == "session_end"), None)
		target = plan["common"]["training_tokens_D"]
		completed = (end is not None and end["tokens_seen"] == target and end["step"] == config["max_steps"]
					 and previous_tokens == target and end["skipped_updates"] == 0)
		if completed:
			expected_val_tokens = list(range(plan["common"]["eval_interval_tokens_E"], target + 1,
											plan["common"]["eval_interval_tokens_E"]))
			if [r["tokens_seen"] for r in val] != expected_val_tokens:
				raise ValueError("Missing evaluations on the frozen token grid")
			row["final_val_loss"] = val[-1]["val_loss"]
			row["eligible"] = not tail_warnings and ledger["status"] == "completed"
			row["status"] = "complete" if row["eligible"] else "review_required"
		else:
			row["status"] = "failed" if ledger["status"] == "failed" else "incomplete"
		if ledger.get("error"):
			row["issues"].append(ledger["error"])
		row["comparison_signature"] = canonical_hash({
			"model": manifest["sessions"][0]["model_config"],
			"environment": manifest["sessions"][0]["environment"],
			"data": {s: manifest["sessions"][0]["data"][s]["metadata_sha256"] for s in ("train", "val")},
		})
		environments = {canonical_hash(s["environment"]) for s in manifest["sessions"]}
		if len(environments) > 1:
			row["eligible"] = False
			row["status"] = "review_required"
			row["issues"].append("Runtime environment changed between resumed sessions.")
	except (KeyError, ValueError, TypeError, OSError, ZeroDivisionError) as error:
		row["status"] = "invalid"
		row["eligible"] = False
		row["issues"].append(str(error))
	if row["train"]:
		import statistics
		steady = [r for r in row["train"] if r["tokens_seen"] - r["tokens_in_window"] >= 40960]
		row["median_window_tokens_per_second"] = statistics.median(r["tokens_per_second"] for r in steady) if steady else None
		last_quarter = [r for r in row["train"] if r["tokens_seen"] > .75 * plan["common"]["training_tokens_D"]]
		updates = sum(r["updates"] for r in last_quarter)
		row["last_quarter_grad_clipped_fraction"] = sum(r["grad_clipped_fraction"] * r["updates"] for r in last_quarter) / updates if updates else None
		row["min_normalized_load_entropy"] = min((r["normalized_load_entropy"] for r in row["moe_windows"]), default=None)
	return row


def write_csv(path, rows, fields):
	with Path(path).open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


SUMMARY_FIELDS = ["id", "learning_rate", "effective_batch_tokens", "status", "eligible", "tokens_seen",
				  "final_val_loss", "delta_from_best", "last_train_loss", "grad_norm_mean", "grad_norm_max",
				  "grad_clipped_fraction", "qk_max_logit", "qk_clipped_fraction", "minimum_gamma",
				  "moe_max_load_over_mean", "moe_max_zero_load_experts", "qb_bias_abs_max",
				  "steady_tokens_per_second", "wall_tokens_per_second", "process_wall_seconds",
				  "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "sessions", "discarded_logged_tokens"]


def summarize_study(evidence):
	plan, state = evidence.json("study_manifest.json"), evidence.json("study_state.json")
	if canonical_hash(plan) != MANIFEST_HASH:
		raise ValueError("The declared Branch 6 manifest changed")
	if state.get("phase") != "branch6" or state.get("schema_version") != 1:
		raise ValueError("Expected a Branch 6 runner bundle")
	if canonical_hash(plan) != state["manifest_sha256"]:
		raise ValueError("Study manifest differs from the runner's frozen copy")
	model = yaml.safe_load(evidence.read("snapshots/model.yaml"))
	if canonical_hash(model) != "9d455836820541befff1b4fda57b42ec2f4fc4dd5b497e86f79aea4a7b6746c9":
		raise ValueError("Evidence model differs from the benchmarked 300M model")
	snapshot_issues = []
	if canonical_hash(evidence.json("snapshots/data_metadata.json")) != plan["common"]["dataset_metadata_sha256"]:
		snapshot_issues.append("Dataset metadata snapshot differs from the frozen data")
	for name, expected in state["snapshot_sha256"].items():
		if not evidence.exists(name) or hashlib.sha256(evidence.read(name)).hexdigest() != expected:
			snapshot_issues.append(f"Missing or changed evidence snapshot: {name}")
	for name, expected in plan["reference"]["training_source_sha256"].items():
		snapshot_name = "source_snapshot/" + name
		if not evidence.exists(snapshot_name) or hashlib.sha256(evidence.read(snapshot_name)).hexdigest() not in plan["reference"]["source_allowed_sha256"][name]:
			snapshot_issues.append("Actual training source snapshot differs from calibrated runtime: " + name)
	candidates = [summarize_candidate(evidence, plan, state, entry, model)
				  for entry in plan["branch6_validation_runs"]]
	signatures = {r.get("comparison_signature") for r in candidates if r["eligible"]}
	if len(signatures) > 1:
		snapshot_issues.append("Effective model/runtime differs between completed candidates.")
	if snapshot_issues:
		for row in candidates:
			if row["eligible"]:
				row["eligible"], row["status"] = False, "review_required"
			row["issues"].extend(snapshot_issues)
	eligible = sorted((r for r in candidates if r["eligible"]), key=lambda r: r["final_val_loss"])
	best = eligible[0]["final_val_loss"] if eligible else None
	for row in eligible:
		row["delta_from_best"] = row["final_val_loss"] - best
	tolerance = plan["selection"]["practical_tie_nats_per_token"]
	tied = [r["id"] for r in eligible if r["delta_from_best"] <= tolerance + 1e-12]
	finished = all(state["runs"][r["id"]]["status"] in ("completed", "failed") for r in candidates)
	complete_grid = len(eligible) == len(candidates)
	best_row = eligible[0] if eligible else None
	selection = {
		"grid_complete_and_valid": complete_grid, "all_candidates_attempted": finished,
		"ready_for_candidate_review": finished and bool(eligible) and not snapshot_issues,
		"best_observed_run": best_row["id"] if best_row else None,
		"within_practical_tie": tied, "practical_tie_nats_per_token": tolerance,
		"best_lr_is_grid_endpoint": (best_row["learning_rate"] in
									  (min(plan["nominal_learning_rates"]), max(plan["nominal_learning_rates"]))) if best_row else None,
		"interpretation": ("Ready for transfer and trajectory/stability review; these are early-horizon candidates, not a 6B recipe."
						   if complete_grid else "All candidates attempted with exclusions: review failure reasons before selecting among valid runs."
						   if finished and eligible and not snapshot_issues else
						   "Partial or invalid grid: do not finalize the candidate selection."),
		"statistical_significance_claim": False, "automatic_additional_runs": False,
	}
	return {"study_id": plan["study_id"], "phase": "branch6", "common": plan["common"],
			"selection": selection, "candidates": candidates, "evidence_issues": snapshot_issues,
			"orchestration_error": state.get("error"),
			"environment_probe": state.get("environment"), "source_files_sha256": state.get("source_files_sha256", {}),
			"power_profile": state.get("power_profile", "unknown"),
			"attempt_ledger": [{"id": run_id, "attempt": i+1, **attempt}
							   for run_id, ledger in state["runs"].items() for i, attempt in enumerate(ledger.get("attempts", []))],
			"analysis_seconds_recorded": state.get("analysis_seconds", 0),
			"packaging_seconds_recorded": state.get("packaging_seconds", 0),
			"measurement_note": "Training windows are token-weighted losses. Steady throughput excludes whole windows starting before 40,960 tokens. Wall cost includes interrupted/superseded attempts."}




def make_figures(output, result):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
						 "axes.spines.top": False, "axes.spines.right": False,
						 "axes.grid": True, "grid.alpha": 0.18,
						 "savefig.dpi": 180, "pdf.fonttype": 42})
	directory = output / "figures"
	directory.mkdir(exist_ok=True)
	for previous in directory.iterdir():
		if previous.suffix in (".png", ".pdf"):
			previous.unlink()
	candidates = result["candidates"]
	rates = sorted({r["learning_rate"] for r in candidates})
	batches = sorted({r["effective_batch_tokens"] for r in candidates})
	colors = dict(zip(rates, ["#2874a6", "#e67e22", "#7d3c98"]))
	warmup = result["common"]["warmup_tokens_W"] / 1e6
	target = result["common"]["training_tokens_D"] / 1e6

	def save(fig, name):
		fig.savefig(directory / (name + ".png"), bbox_inches="tight")
		fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
		plt.close(fig)

	def label(row):
		return f"LR {row['learning_rate']:.0e}, B {row['effective_batch_tokens']:,}"

	def style(row):
		return {"color": colors[row["learning_rate"]],
				"linestyle": "-" if row["effective_batch_tokens"] == batches[0] else "--"}

	def curves(ax, key, *, kind="train", scale=1, title=None):
		plotted = False
		for row in candidates:
			data = row[kind]
			if data:
				ax.plot([r["tokens_seen"] / 1e6 for r in data],
						[r[key] * scale for r in data], label=label(row),
						marker="o" if kind == "validation" else None,
						markersize=3, linewidth=1.5, **style(row))
				plotted = True
		if not plotted:
			ax.text(0.5, 0.5, "No measurements yet", ha="center", va="center", transform=ax.transAxes)
		ax.axvline(warmup, color="#555555", alpha=0.5, linewidth=1, linestyle=":")
		ax.set_xlim(0, target * 1.015)
		ax.set_xlabel("Training tokens (millions)")
		if title:
			ax.set_title(title)
		return plotted

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
	shown = curves(axes[0], "train_loss", title="Training loss · logged token windows")
	curves(axes[1], "val_loss", kind="validation", title="Validation loss · fixed prefix")
	for ax in axes:
		ax.set_ylabel("Cross entropy (nats/token)")
	if shown:
		axes[0].legend(fontsize=8, frameon=False)
	save(fig, "loss_curves")

	fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
	for row in candidates:
		if row["train"]:
			ax.plot([r["tokens_seen"] / 1e6 for r in row["train"]],
					[r["lr"] / row["learning_rate"] for r in row["train"]], label=label(row), **style(row))
	ax.axvline(warmup, color="#555555", alpha=0.5, linestyle=":")
	ax.set_xlim(0, target * 1.015)
	ax.set_ylim(0, 1.05)
	ax.set_xlabel("Training tokens (millions)")
	ax.set_ylabel("Applied LR / nominal peak LR")
	ax.set_title("Common token schedule · recorded points")
	save(fig, "lr_schedule")

	fig, ax = plt.subplots(figsize=(6.7, 4.6), layout="constrained")
	grid = np.full((len(rates), len(batches)), np.nan)
	for row in candidates:
		if row["eligible"]:
			grid[rates.index(row["learning_rate"]), batches.index(row["effective_batch_tokens"])] = row["final_val_loss"]
	cmap = plt.get_cmap("viridis_r").copy()
	cmap.set_bad("#eeeeee")
	image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto")
	for row in candidates:
		i, j = rates.index(row["learning_rate"]), batches.index(row["effective_batch_tokens"])
		text = (f"{row['final_val_loss']:.4f}\nΔ {row['delta_from_best']:+.4f}"
				if row["eligible"] else row["status"].replace("_", " "))
		ax.text(j, i, text, ha="center", va="center", fontsize=10,
				bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 4})
	measured = {(r["learning_rate"], r["effective_batch_tokens"]) for r in candidates}
	for i, rate in enumerate(rates):
		for j, batch in enumerate(batches):
			if (rate, batch) not in measured:
				ax.text(j, i, "Not tested", ha="center", va="center", color="#64748b")
	ax.set_xticks(range(len(batches)), [f"{b:,}" for b in batches])
	ax.set_yticks(range(len(rates)), [f"{r:.0e}" for r in rates])
	ax.set_xlabel("Effective batch (tokens/update)")
	ax.set_ylabel("Peak nominal learning rate")
	ax.set_title("Final validation loss · complete, valid runs only")
	ax.grid(False)
	if np.isfinite(grid).any():
		fig.colorbar(image, ax=ax, label="Cross entropy (nats/token)")
	save(fig, "lr_batch_grid")

	fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), layout="constrained")
	curves(axes[0], "grad_norm_mean", title="Gradient norm before clipping · mean over updates")
	curves(axes[1], "grad_clipped_fraction", scale=100, title="Updates with gradient clipping")
	axes[0].set_ylabel("Gradient norm")
	axes[0].axhline(1, color="#777777", linestyle=":", linewidth=1)
	axes[1].set_ylabel("Clipped updates (%)")
	axes[1].set_ylim(-2, 102)
	if shown:
		axes[0].legend(fontsize=8, frameon=False, ncol=2)
	save(fig, "gradients")

	fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
	curves(axes[0], "qk_max_logit", title="Maximum causal QK score across MLA heads")
	axes[0].axhline(100, color="#777777", linestyle=":", label="Clip threshold")
	if all(r["qk_max_logit"] > 0 for row in candidates for r in row["train"]):
		axes[0].set_yscale("log")
	curves(axes[1], "qk_clipped_fraction", scale=100, title="Head updates with QK-Clip")
	curves(axes[2], "minimum_gamma", title="Minimum QK-Clip rescaling factor")
	axes[0].set_ylabel("QK / √dimension")
	axes[1].set_ylabel("Clipped head updates (%)")
	axes[1].set_ylim(-2, 102)
	axes[2].set_ylabel("Gamma")
	axes[2].set_ylim(-0.02, 1.02)
	save(fig, "qk_clip")

	fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
	curves(axes[0], "moe_max_load_over_mean", title="Worst layer's maximum / mean expert load")
	curves(axes[1], "qb_bias_abs_max", title="Largest absolute QB bias across layers/experts")
	curves(axes[2], "moe_max_zero_load_experts", title="Worst layer's zero-load experts in each token window")
	axes[0].set_ylabel("Load ratio")
	axes[1].set_ylabel("Absolute bias")
	axes[2].set_ylabel("Expert count")
	axes[2].set_ylim(-1, 65)
	if shown:
		axes[0].legend(fontsize=8, frameon=False, ncol=2)
	save(fig, "moe_health")

	fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
	positions = np.arange(len(candidates))
	ticks = [f"{r['learning_rate']:.0e}\nB={r['effective_batch_tokens']}" for r in candidates]
	for ax, key, scale, ylabel, title in [
		(axes[0], "steady_tokens_per_second", 1, "Tokens/second", "Measured steady training throughput"),
		(axes[1], "process_wall_seconds", 1/60, "Minutes", "Process wall time · includes retries"),
		(axes[2], "cuda_peak_allocated_bytes", 1/1024**3, "GiB", "Peak allocated CUDA memory"),
	]:
		values = [np.nan if r.get(key) is None else r[key] * scale for r in candidates]
		ax.bar(positions, values, color=[colors[r["learning_rate"]] for r in candidates], alpha=0.85)
		ax.set_xticks(positions, ticks, fontsize=7)
		ax.set_ylabel(ylabel)
		ax.set_title(title, fontsize=10)
		ax.set_ylim(bottom=0)
	save(fig, "efficiency")

	maxima = [max(counts) * len(counts) / sum(counts)
			  for row in candidates for counts in row["expert_totals"].values() if sum(counts)]
	load_max = max(maxima, default=1)
	bias_max = max((abs(v) for row in candidates for biases in row["final_qb_biases"].values() for v in biases), default=1) or 1
	for row in candidates:
		if not row["expert_totals"]:
			continue
		names = sorted(row["expert_totals"], key=lambda name: int(name.split(".")[1]))
		loads = np.array([[v * len(counts) / sum(counts) for v in counts]
						  for counts in (row["expert_totals"][name] for name in names)])
		biases = np.array([row["final_qb_biases"][name] for name in names])
		fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained")
		for ax, data, cmap_name, lower, upper, title, units in [
			(axes[0], loads, "viridis", 0, load_max, "Cumulative routing assignments / mean expert load", "Relative load"),
			(axes[1], biases, "coolwarm", -bias_max, bias_max, "QB biases at last logged window", "QB bias"),
		]:
			image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap_name, vmin=lower, vmax=upper)
			ax.set_yticks(range(len(names)), [name.split(".")[1] for name in names], fontsize=8)
			ax.set_xlabel("Expert index")
			ax.set_ylabel("Decoder layer")
			ax.set_title(title, fontsize=10)
			ax.grid(False)
			fig.colorbar(image, ax=ax, label=units)
		fig.suptitle(label(row) + f" · {row['tokens_seen']:,} tokens · {row['status']}")
		save(fig, "experts_" + row["id"])



def paired_contrast(treatment, control):
	a = {r['tokens_seen']: r['val_loss'] for r in treatment['validation']}
	b = {r['tokens_seen']: r['val_loss'] for r in control['validation']}
	return [{'tokens_seen': t, 'delta_loss': a[t] - b[t]} for t in sorted(a.keys() & b.keys())]


def validation_review(result, baseline):
	tolerance = result['selection']['practical_tie_nats_per_token']
	curves, summary = [], []
	for scale, candidates in [*baseline['scales'].items(), ('300m', result['candidates'])]:
		by_lr = {r['learning_rate']: r for r in candidates}
		treatment, control = by_lr[.003], by_lr[.001]
		complete = scale != '300m' or (treatment['eligible'] and control['eligible'])
		points = paired_contrast(treatment, control)
		curves.extend({'model': scale, **p} for p in points)
		delta = treatment['final_val_loss'] - control['final_val_loss'] if complete else None
		summary.append({'model': scale, 'delta_3e3_minus_1e3': delta,
						'paired_points': len(points), 'complete_and_valid': complete,
						'outcome': ('3e-3_lower_loss' if delta < -tolerance else
									'1e-3_lower_loss' if delta > tolerance else 'practical_tie')
								   if complete else 'pending_or_invalid'})
	measured = summary[-1]
	# A ranking is an input to a human stability review, not a certification.
	return {'status': 'ready_for_stability_review' if measured['complete_and_valid'] else 'awaiting_valid_300m_pair',
			'contrasts': summary, 'curves': curves,
			'lr_ranking_transfer': measured['outcome'],
			'batch_transfer_tested_in_300m': False,
			'recipe_recommendation': 'Review loss trajectories, clipping and routing before closing Branch 6.',
			'statistical_significance_claim': False, 'production_6b_optimum_claim': False}


def reference_figures(output, result, baseline):
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
	colors = {'82m': '#94a3b8', '122m': '#007f83', '300m': '#e67e22'}
	for scale, candidates in [*baseline['scales'].items(), ('300m', result['candidates'])]:
		for row in candidates:
			if row['validation']:
				axes[0].plot([p['tokens_seen']/1e6 for p in row['validation']],
							 [p['val_loss'] for p in row['validation']], color=colors[scale],
							 linestyle='-' if row['learning_rate']==.003 else '--',
							 marker='o', markersize=3,
							 label=f"{scale.upper()} / LR {row['learning_rate']:.0e}")
		points = [r for r in result['validation_review']['curves'] if r['model']==scale]
		if points:
			axes[1].plot([p['tokens_seen']/1e6 for p in points], [p['delta_loss'] for p in points],
						 color=colors[scale], marker='o', label=scale.upper())
	axes[0].set_ylabel('Validation cross entropy (nats/token)')
	axes[0].set_title('Measured proxies and 300M progress')
	axes[1].set_ylabel('Loss LR 3e-3 minus loss LR 1e-3')
	axes[1].axhspan(-.01, .01, color='#64748b', alpha=.12, label='Operational tie band')
	axes[1].axhline(0, color='#64748b', linewidth=1)
	axes[1].set_title('LR contrast at B=2,048 / negative favors 3e-3')
	for ax in axes:
		ax.set_xlabel('Training tokens (millions)')
		ax.legend(fontsize=8, frameon=False)
	fig.suptitle('Same 2.048M-token horizon / no IsoFLOPs or power-law fit')
	for suffix in ('png', 'pdf'):
		fig.savefig(output/'figures'/('proxy_to_300m.'+suffix), bbox_inches='tight')
	plt.close(fig)


def write_report(output, result):
	def fmt(v): return '—' if v is None else f'{v:.5f}'
	lines = ['# Kimi K3 · Branch 6: validación de candidatos en 300M', '',
			 '**Estado:** '+result['validation_review']['status']+'.', '',
			 'Dos recetas: LR 3e-3 y 1e-3, ambas B=2.048; microbatch 1 × contexto 256; acumulación 8. '
			 'Cada run comienza desde seed 1337 y token cero. D=2.048.000; warmup 20.480; '
			 'cosine hasta D, mínimo 10%; ocho validaciones de un prefijo fijo de 65.536 tokens.', '',
			 '| LR | Estado | Tokens válidos | Loss final | Grad norm máx. | Clip global | QK máx. | Tok/s estables |',
			 '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
	for row in result['candidates']:
		lines.append(f"| {row['learning_rate']:.0e} | {row['status']} | {row['tokens_seen']:,} | "
					 f"{fmt(row['final_val_loss'])} | {fmt(row.get('grad_norm_max'))} | "
					 f"{fmt(row.get('grad_clipped_fraction'))} | {fmt(row.get('qk_max_logit'))} | "
					 f"{fmt(row.get('steady_tokens_per_second'))} |")
	lines += ['', '## Contraste de LR entre escalas', '',
			  '| Escala | Δ loss 3e-3 menos 1e-3 | Puntos pareados | Lectura |',
			  '| --- | ---: | ---: | --- |']
	for row in result['validation_review']['contrasts']:
		lines.append(f"| {row['model']} | {fmt(row['delta_3e3_minus_1e3'])} | {row['paired_points']} | {row['outcome']} |")
	lines += ['', 'La banda ±0,01 nats/token es un empate operativo, no un intervalo de confianza. '
			  'El punto principal es el final declarado; los puntos anteriores ayudan a evaluar el ranking. '
			  'Las curvas parciales no se clasifican como resultados finales.', '',
			  '## Evidencia y revisión de estabilidad', '',
			  'Revisar pérdidas finitas, updates omitidos, clipping global temprano/tardío, máximos QK/gamma, '
			  'cargas y entropía por capa, expertos sin carga y sesgos QB. Clipping frecuente o un cero temporal '
			  'no invalidan automáticamente una run; se interpretan junto con la trayectoria. '
			  'CSV y métricas crudas conservan todas las ventanas y los intentos.', '']
	for row in result['candidates']:
		for message in row['issues'] + row['warnings']:
			lines.append('- '+row['id']+': '+message)
	for message in result['evidence_issues']:
		lines.append('- Auditoría: '+message)
	lines += ['', '## Coste y reanudación', '',
			  'Tiempo de procesos registrado: '+fmt(sum(r['process_wall_seconds'] for r in result['candidates']))+' s. '
			  'Incluye arranque, compilación, evaluación, checkpoints e intentos anteriores. '
			  'Throughput estable excluye ventanas que comienzan antes de 40.960 tokens. '
			  'Los picos CUDA corresponden al allocator. El perfil energético es declarado: `'+str(result['power_profile'])+'`.', '',
			  'Checkpoints permanecen en el directorio local y se excluyen del ZIP. La reconstrucción recorta '
			  'métricas posteriores al cursor reanudado y conserva su coste. Si no existe checkpoint, '
			  'el intento previo se archiva y la receta comienza desde cero.', '',
			  '## Alcance y siguiente decisión', '',
			  'Esta prueba valida el contraste de LR en 300M a horizonte corto. No vuelve a probar B=512. '
			  'Una seed, validación compartida, contexto 256 y cosine de 2M tokens no determinan el óptimo a 6B. '
			  '3e-3 permanece en el borde superior estudiado; no se ajusta una ley de potencia.', '',
			  'Tras revisar ambos resultados, seleccionar una receta candidata para Branch 7. '
			  'El pilot de producción debe arrancar una trayectoria nueva con horizonte cosine de 6B, '
			  'atravesar el warmup de producción y confirmar contexto/microbatch/QB, memoria, '
			  'rendimiento y checkpoint→resume. Si falta evidencia por memoria o fallo, documentar '
			  'la comprobación pendiente; no sustituir silenciosamente el batch.', '']
	for name in ['loss_curves','proxy_to_300m','gradients','qk_clip','moe_health','efficiency']:
		lines += [f'![{name}](figures/{name}.png)', '']
	(output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze(results, output_dir=None):
	source = Path(results).resolve()
	output = Path(output_dir).resolve() if output_dir else (source/'analysis' if source.is_dir() else source.with_name(source.stem+'_analysis'))
	evidence = Evidence(source)
	try:
		result = summarize_study(evidence)
		plan = evidence.json('study_manifest.json')
		baseline = evidence.json('snapshots/reference_proxies.json')
		if canonical_hash(baseline) != plan['reference']['canonical_sha256']:
			raise ValueError('Proxy reference changed')
		state = evidence.json('study_state.json')
		if state.get('environment') is not None and state['environment'] != plan['reference']['environment']:
			result['evidence_issues'].append('Environment probe differs from calibrated runtime')
		if result['evidence_issues']:
			for row in result['candidates']:
				row['eligible'] = False
			result['selection']['grid_complete_and_valid'] = False
			result['selection']['ready_for_candidate_review'] = False
			result['selection']['best_observed_run'] = None
			result['selection']['within_practical_tie'] = []
		result['validation_review'] = validation_review(result, baseline)
	finally:
		evidence.close()
	output.mkdir(parents=True, exist_ok=True)
	write_json(output/'study_summary.json', result)
	write_json(output/'selection.json', result['selection'])
	write_json(output/'validation_review.json', result['validation_review'])
	fields = SUMMARY_FIELDS + ['last_quarter_grad_clipped_fraction','min_normalized_load_entropy','ram_rss_sampled_peak_bytes','wall_time_complete']
	write_csv(output/'summary.csv', result['candidates'], fields)
	for filename,key in [('train_curves','train'),('validation_curves','validation'),('moe_windows','moe_windows')]:
		rows = [p for r in result['candidates'] for p in r[key]]
		default = ['id','tokens_seen']
		write_csv(output/(filename+'.csv'), rows, list(rows[0]) if rows else default)
	write_csv(output/'lr_contrasts.csv', result['validation_review']['contrasts'], ['model','delta_3e3_minus_1e3','paired_points','complete_and_valid','outcome'])
	write_csv(output/'lr_contrast_curves.csv', result['validation_review']['curves'], ['model','tokens_seen','delta_loss'])
	write_csv(output/'attempt_ledger.csv', result['attempt_ledger'], ['id','attempt','started_at','ended_at','wall_seconds','returncode','interrupted','power_profile','log'])
	experts = []
	for row in result['candidates']:
		for layer, counts in row['expert_totals'].items():
			for index,count in enumerate(counts):
				experts.append({'id':row['id'],'layer':layer,'expert':index,'assignments':count,
								'load_over_mean':count*len(counts)/sum(counts),'final_qb_bias':row['final_qb_biases'][layer][index]})
	write_csv(output/'expert_summary.csv', experts, ['id','layer','expert','assignments','load_over_mean','final_qb_bias'])
	make_figures(output, result)
	reference_figures(output, result, baseline)
	write_report(output, result)
	return result


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--results', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path)
	args = parser.parse_args(argv)
	result = analyze(args.results, args.output_dir)
	print(json.dumps(result['validation_review'], indent=2))
	return 1 if result['evidence_issues'] or any(r['issues'] for r in result['candidates']) else 0


if __name__ == '__main__':
	raise SystemExit(main())
