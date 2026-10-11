"""Execute two declared 300M validation runs; checkpoint/resume and export branch6_results.zip.

	uv run python scripts/run_pretraining_validation.py --data-dir C:/datasets/fineweb_edu_6b \
		--output-dir results/training/branch6/300m_validation --power-profile turbo

Only existing train.py/YAMLs train the model. This orchestrator holds no CUDA state.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import perf_counter
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

if __package__:
	from .analyze_pretraining_validation import analyze, canonical_hash, expected_recipe, write_json
else:
	from analyze_pretraining_validation import analyze, canonical_hash, expected_recipe, write_json


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("docs/notes/pretraining_validation_manifest.json")
MODEL = Path("configs/model/kimi_k3_300m.yaml")
REFERENCE = Path("docs/notes/pretraining_validation_reference_proxies.json")
MANIFEST_HASH = "5dc9200c399f4aa46e71a17dc9cd472af66b5564f662e0db93a4a9c2ae1cf579"
MODEL_HASH = "9d455836820541befff1b4fda57b42ec2f4fc4dd5b497e86f79aea4a7b6746c9"


def now():
	return datetime.now(timezone.utc).isoformat()


def file_hash(path):
	return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan(root):
	plan = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
	if canonical_hash(plan) != MANIFEST_HASH:
		raise ValueError("The declared Branch 6 manifest changed")
	common = plan["common"]
	required = {"microbatch_sequences": 1, "seq_len": 256, "seed": 1337,
				"training_tokens_D": 2048000, "warmup_tokens_W": 20480,
				"eval_interval_tokens_E": 256000, "log_interval_tokens_L": 20480,
				"validation_prefix_tokens_V": 65536, "checkpoint_interval_tokens": 1024000,
				"lr_schedule": "cosine", "min_lr_ratio": 0.1, "overfit_batch": False}
	if any(common.get(k) != v for k, v in required.items()):
		raise ValueError("The Branch 3 token controls changed")
	entries = plan["branch6_validation_runs"]
	if {(e["learning_rate"], e["effective_batch_tokens"]) for e in entries} != {(.003,2048),(.001,2048)} or len(entries) != 2:
		raise ValueError("Expected exactly the two declared validation candidates")
	if [e["execution_order"] for e in entries] != [1,2]:
		raise ValueError("The declared execution order changed")
	model = yaml.safe_load((root / MODEL).read_text(encoding="utf-8"))
	if canonical_hash(model) != MODEL_HASH:
		raise ValueError("The model differs from the declared 300M model")
	reference = json.loads((root / REFERENCE).read_text(encoding="utf-8"))
	if canonical_hash(reference) != plan["reference"]["canonical_sha256"]:
		raise ValueError("The measured proxy reference changed")
	for name, digest in plan["reference"]["training_source_sha256"].items():
		if not (root / name).is_file() or file_hash(root / name) not in plan["reference"]["source_allowed_sha256"][name]:
			raise ValueError("Training code/dependencies differ from calibrated proxies: " + name)
	for entry in entries:
		if (entry["model"] != "300m" or entry["expected_tokens"] != common["training_tokens_D"]
				or entry["expected_updates"] != common["training_tokens_D"] // entry["effective_batch_tokens"]
				or entry["gradient_accumulation_steps"] * 256 != entry["effective_batch_tokens"]):
			raise ValueError("Inconsistent candidate tokens/accumulation")
		recipe = yaml.safe_load((root / entry["training_config"]).read_text(encoding="utf-8"))
		if recipe != expected_recipe(common, entry):
			raise ValueError("Training recipe differs from its selected proxy candidate: " + entry["training_config"])
	return plan


def source_files(root, plan):
	files = {Path("scripts/train.py"), MODEL, MANIFEST, REFERENCE,
			 Path("scripts/run_pretraining_validation.py"), Path("scripts/analyze_pretraining_validation.py"),
			 *(Path(entry["training_config"]) for entry in plan["branch6_validation_runs"])}
	files.update(path.relative_to(root) for path in (root / "src/kimi_k3").rglob("*.py"))
	for name in ("pyproject.toml", "uv.lock"):
		if (root / name).is_file():
			files.add(Path(name))
	return {path.as_posix(): file_hash(root / path) for path in sorted(files)}


def check_dataset(data_dir, plan):
	metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
	if canonical_hash(metadata) != plan["common"]["dataset_metadata_sha256"]:
		raise ValueError("Dataset metadata differs from the Branch 3 dataset")
	if metadata["tokenizer"]["vocab_size"] != 49152:
		raise ValueError("Unexpected tokenizer vocabulary")
	for split, needed in (("train", plan["common"]["training_tokens_D"] + 1),
						  ("val", plan["common"]["validation_prefix_tokens_V"] + 1)):
		if metadata["splits"][split]["tokens"] < needed:
			raise ValueError(f"Not enough {split} tokens for the frozen prefix")
	return metadata


def environment(root):
	# This process exits before training: the orchestrator holds no CUDA state.
	code = """
import importlib.metadata as m, json, platform, torch
if not torch.cuda.is_available():
	raise RuntimeError('CUDA is unavailable in this Python environment')
if not torch.cuda.is_bf16_supported(including_emulation=False):
	raise RuntimeError('Native BF16 is unavailable')
p = torch.cuda.get_device_properties(0)
versions = {}
for name in ('torch', 'numpy', 'flash-linear-attention', 'fla-core', 'triton', 'triton-windows'):
	try: versions[name] = m.version(name)
	except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': platform.python_version(), 'torch': str(torch.__version__),
 'cuda': torch.version.cuda, 'packages': versions, 'gpu': p.name,
 'compute_capability': [p.major, p.minor], 'total_memory_bytes': p.total_memory}))
"""
	result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
							text=True, encoding="utf-8", errors="replace", timeout=60)
	if result.returncode:
		raise RuntimeError(result.stderr.strip() or result.stdout.strip())
	return json.loads(result.stdout)


def command(root, study, data_dir, entry, checkpoint=None):
	argv = [sys.executable, "-u", str(root / "scripts/train.py"),
			"--data-dir", str(data_dir), "--model-config", str(root / MODEL),
			"--training-config", str(root / entry["training_config"]),
			"--device", "cuda", "--run-dir", str(study / "runs" / entry["id"])]
	if checkpoint is not None:
		argv += ["--resume", str(checkpoint)]
	return argv


@contextmanager
def study_lock(directory):
	"""An OS lock releases on process death; no stale PID lock to delete."""
	with (directory / ".study.lock").open("a+b") as stream:
		stream.seek(0, 2)
		if stream.tell() == 0:
			stream.write(b"0")
			stream.flush()
		stream.seek(0)
		if os.name == "nt":
			import msvcrt
			try:
				msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
			except OSError as error:
				raise RuntimeError("Another runner is using this study directory") from error
		else:
			import fcntl
			try:
				fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
			except OSError as error:
				raise RuntimeError("Another runner is using this study directory") from error
		try:
			yield
		finally:
			stream.seek(0)
			if os.name == "nt":
				msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
			else:
				fcntl.flock(stream, fcntl.LOCK_UN)


def run_logged(argv, log, root):
	started = perf_counter()
	options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
			   else {"start_new_session": True})
	env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
	interrupted = False
	with log.open("w", encoding="utf-8") as stream:
		with subprocess.Popen(argv, cwd=root, env=env, stdout=subprocess.PIPE,
							  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
							  errors="replace", bufsize=1, **options) as process:
			try:
				for line in process.stdout:
					stream.write(line)
					stream.flush()
					print(line, end="", flush=True)
				returncode = process.wait()
			except KeyboardInterrupt:
				interrupted = True
				print("\nStopping the current trainer and saving the partial study...", flush=True)
				try:
					if os.name == "nt":
						process.send_signal(signal.CTRL_BREAK_EVENT)
					else:
						os.killpg(process.pid, signal.SIGINT)
					process.wait(timeout=10)
				except (OSError, subprocess.TimeoutExpired):
					process.terminate()
					try:
						process.wait(timeout=5)
					except subprocess.TimeoutExpired:
						process.kill()
						process.wait()
				remaining = process.stdout.read()
				if remaining:
					stream.write(remaining)
				returncode = process.returncode
	return returncode, perf_counter() - started, interrupted


def create_state(root, study, data_dir, plan):
	hashes = source_files(root, plan)
	snapshots = {}
	for name, digest in hashes.items():
		destination = study / "source_snapshot" / name
		destination.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(root / name, destination)
		snapshots[destination.relative_to(study).as_posix()] = digest
	for path in (MODEL, MANIFEST, REFERENCE):
		target = study / ("snapshots/model.yaml" if path == MODEL else "snapshots/reference_proxies.json" if path == REFERENCE else "study_manifest.json")
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(root / path, target)
		snapshots[target.relative_to(study).as_posix()] = file_hash(target)
	data_snapshot = study / "snapshots/data_metadata.json"
	shutil.copyfile(data_dir / "metadata.json", data_snapshot)
	snapshots[data_snapshot.relative_to(study).as_posix()] = file_hash(data_snapshot)
	for name in ("run_pretraining_validation.py", "analyze_pretraining_validation.py"):
		target = study / "source_snapshot/scripts" / name
		shutil.copyfile(root / "scripts" / name, target)
		snapshots[target.relative_to(study).as_posix()] = file_hash(target)
	return {
		"schema_version": 1, "phase": "branch6", "created_at": now(), "updated_at": now(),
		"manifest_sha256": canonical_hash(plan), "data_dir": str(data_dir),
		"source_files_sha256": hashes, "snapshot_sha256": snapshots, "environment": None,
		"runs": {entry["id"]: {"status": "pending", "attempts": [], "error": None}
				 for entry in plan["branch6_validation_runs"]},
	}


def save_state(study, state):
	state["updated_at"] = now()
	write_json(study / "study_state.json", state)


def archive_results(study):
	archive = study / "branch6_results.zip"
	temporary = study / "branch6_results.zip.tmp"
	suffixes = {".json", ".jsonl", ".log", ".yaml", ".toml", ".lock", ".py", ".csv", ".md", ".png", ".pdf"}
	with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as bundle:
		for path in sorted(study.rglob("*")):
			if (path.is_file() and path != study / ".study.lock" and path.suffix in suffixes
					and "__pycache__" not in path.parts):
				bundle.write(path, path.relative_to(study).as_posix())
	temporary.replace(archive)
	return archive


def refresh_results(study, state):
	save_state(study, state)
	result = None
	started = perf_counter()
	try:
		result = analyze(study)
		state.pop("analysis_error", None)
	except (OSError, ValueError, KeyError, ImportError, RuntimeError, TypeError) as error:
		state["analysis_error"] = str(error)
		print(f"Analysis error (raw evidence is preserved): {error}", file=sys.stderr)
	state["analysis_seconds"] = state.get("analysis_seconds", 0) + perf_counter() - started
	save_state(study, state)
	started = perf_counter()
	archive = archive_results(study)
	state["packaging_seconds"] = state.get("packaging_seconds", 0) + perf_counter() - started
	save_state(study, state)
	print(f"\nResults ZIP: {archive}", flush=True)
	return result


def prepare_attempt(root, study, data_dir, entry, ledger):
	directory = study / "runs" / entry["id"]
	checkpoint = directory / "last.pt"
	resume = checkpoint if checkpoint.is_file() and (directory / "run.json").is_file() else None
	if directory.exists() and resume is None:
		# Preserve evidence and start from seed; never append fresh weights to
		# an existing run identity when there is no usable checkpoint.
		archive = study / "abandoned_attempts" / entry["id"] / f"attempt_{len(ledger['attempts']):02d}"
		archive.parent.mkdir(parents=True, exist_ok=True)
		directory.replace(archive)
		ledger["restart_note"] = "No checkpoint: prior attempt archived; candidate restarted from token zero."
	logs = study / "logs"
	logs.mkdir(exist_ok=True)
	log = logs / f"{entry['id']}_attempt_{len(ledger['attempts']) + 1:02d}.stdout.log"
	return command(root, study, data_dir, entry, resume), log


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path)
	parser.add_argument("--phase", choices=("branch6",), default="branch6")
	parser.add_argument("--output-dir", type=Path, help="New study folder; required with --resume")
	parser.add_argument("--resume", action="store_true", help="Continue this study; completed/failed candidates are not rerun")
	parser.add_argument("--power-profile", help="User-declared energy/fan profile, e.g. turbo; not measured automatically")
	parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry failed candidates using their original recipes; requires --resume")
	parser.add_argument("--max-runs", type=int, help="Execute this many remaining candidates, in the frozen order")
	parser.add_argument("--dry-run", action="store_true", help="Print exact commands; no CUDA, training or file writes")
	parser.add_argument("--pack-only", action="store_true", help="Rebuild analysis/ZIP from --output-dir without CUDA")
	args = parser.parse_args(argv)
	if args.retry_failed and not args.resume:
		parser.error("--retry-failed requires --resume")
	if args.max_runs is not None and args.max_runs < 1:
		parser.error("--max-runs must be positive")
	if (args.resume or args.pack_only) and args.output_dir is None:
		parser.error("--output-dir is required with --resume or --pack-only")
	study = (args.output_dir.resolve() if args.output_dir else ROOT / "results/training/branch6" /
			 (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]))
	try:
		if args.pack_only:
			with study_lock(study):
				state = json.loads((study / "study_state.json").read_text(encoding="utf-8"))
				refresh_results(study, state)
			return 0 if not state.get("analysis_error") else 1
		plan = load_plan(ROOT)
		if args.resume:
			state = json.loads((study / "study_state.json").read_text(encoding="utf-8"))
			data_dir = (args.data_dir or Path(state["data_dir"])).resolve()
			if state["manifest_sha256"] != canonical_hash(plan):
				raise ValueError("The manifest changed; cannot resume this study")
			if state["source_files_sha256"] != source_files(ROOT, plan):
				raise ValueError("Training source/configs/dependency lock changed; cannot resume this study")
		else:
			if args.data_dir is None:
				parser.error("--data-dir is required for a new study")
			data_dir = args.data_dir.resolve()
			if study.exists() and not args.dry_run:
				raise FileExistsError("Study directory exists; use --resume or a new --output-dir")
		entries = [entry for entry in plan["branch6_validation_runs"]
				   if not args.resume or state["runs"][entry["id"]]["status"] not in (("completed",) if args.retry_failed else ("completed", "failed"))]
		if args.max_runs:
			entries = entries[:args.max_runs]
		if args.dry_run:
			print(json.dumps({"output_dir": str(study), "runs": [
				{"id": entry["id"], "tokens": entry["expected_tokens"],
				 "estimated_minutes": round(entry["estimated_cost"]["total_seconds"] / 60, 1),
				 "argv": command(ROOT, study, data_dir, entry,
								 study / "runs" / entry["id"] / "last.pt" if args.resume and
								 (study / "runs" / entry["id"] / "last.pt").is_file() else None)}
				for entry in entries]}, indent=2))
			return 0
		# Fail on a missing plotting dependency before spending any GPU time.
		import matplotlib  # noqa: F401
		import numpy  # noqa: F401
		check_dataset(data_dir, plan)
		if not args.resume:
			study.mkdir(parents=True, exist_ok=False)
			state = create_state(ROOT, study, data_dir, plan)
		status = 0
		with study_lock(study):
			refreshed = False
			try:
				if args.resume:
					state = json.loads((study / "study_state.json").read_text(encoding="utf-8"))
					for ledger in state["runs"].values():
						if ledger["status"] == "running":
							ledger["status"] = "interrupted"
							if ledger["attempts"][-1]["returncode"] is None:
								ledger["attempts"][-1]["wall_seconds"] = None
					entries = [entry for entry in plan["branch6_validation_runs"]
							   if state["runs"][entry["id"]]["status"] not in (("completed",) if args.retry_failed else ("completed", "failed"))]
					if args.max_runs:
						entries = entries[:args.max_runs]
				current_environment = environment(ROOT)
				if current_environment != plan["reference"]["environment"]:
					raise ValueError("Runtime differs from calibrated proxies; preserve the calibrated environment before transferring")
				if state["environment"] is not None and current_environment != state["environment"]:
					raise ValueError("Training environment changed; cannot resume this study")
				state["environment"] = current_environment
				state["power_profile"] = args.power_profile or state.get("power_profile", "unknown")
				state.pop("error", None)
				save_state(study, state)
				print(f"Study folder: {study}\nRemaining runs in this invocation: {len(entries)}", flush=True)
				for entry in entries:
					refreshed = False
					if source_files(ROOT, plan) != state["source_files_sha256"]:
						raise ValueError("Training source/configs changed during the study")
					check_dataset(data_dir, plan)
					ledger = state["runs"][entry["id"]]
					argv_run, log = prepare_attempt(ROOT, study, data_dir, entry, ledger)
					attempt = {"started_at": now(), "argv": argv_run, "log": log.relative_to(study).as_posix(),
							   "wall_seconds": 0, "returncode": None, "interrupted": False, "power_profile": state["power_profile"]}
					ledger["attempts"].append(attempt)
					ledger["status"], ledger["error"] = "running", None
					save_state(study, state)
					print(f"\n[{entry['execution_order']}/2] {entry['id']} · {entry['expected_tokens']:,} tokens\n", flush=True)
					code, wall, interrupted = run_logged(argv_run, log, ROOT)
					attempt.update({"returncode": code, "wall_seconds": wall, "interrupted": interrupted, "ended_at": now()})
					if interrupted:
						ledger["status"] = "interrupted"
						status = 130
					elif code:
						ledger["status"] = "failed"
						ledger["error"] = f"Trainer exited with code {code}; inspect {attempt['log']}."
						status = 1
					else:
						ledger["status"] = "completed"
					result = refresh_results(study, state)
					refreshed = True
					if result is not None and result["evidence_issues"]:
						state["error"] = "Validation controls failed: " + "; ".join(result["evidence_issues"])
						status = 1
						refreshed = False
					if result is not None and ledger["status"] == "completed":
						summary = next(r for r in result["candidates"] if r["id"] == entry["id"])
						if not summary["eligible"]:
							ledger["status"] = "failed"
							ledger["error"] = "Completed trainer failed evidence checks: " + "; ".join(summary["issues"])
							status = 1
							refresh_results(study, state)
						else:
							print(f"{entry['id']}: final val {summary['final_val_loss']:.4f} | "
								  f"{summary['steady_tokens_per_second']:.1f} logged tok/s | {wall/60:.1f} min", flush=True)
					if status or state.get("analysis_error"):
						status = status or 1
						break
			except KeyboardInterrupt:
				status = 130
				refreshed = False
				for ledger in state["runs"].values():
					if ledger["status"] == "running":
						ledger["status"] = "interrupted"
			except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
				refreshed = False
				state["error"] = str(error)
				print(f"Study stopped: {error}", file=sys.stderr)
				status = 1
			finally:
				if not refreshed:
					refresh_results(study, state)
		return status or (1 if state.get("error") or any(r["status"] == "failed" for r in state["runs"].values()) else 0)
	except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
		parser.exit(1, f"Cannot run study: {error}\n")


if __name__ == "__main__":
	raise SystemExit(main())
