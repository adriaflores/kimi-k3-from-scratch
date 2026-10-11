"""CPU orchestration/evidence tests using explicit synthetic trajectories, not model training."""
"""Scalar/protocol tests; no torch import or GPU training is performed.

The diagnostic fixtures are real Branch 3 logs. Full-grid logs below are
synthetic control fixtures ONLY, to test accounting, rollback and reports.

	python -m unittest discover -s tests/unit/training -p test_pretraining_study.py -v
"""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import scripts.analyze_pretraining_transfer as analysis
import scripts.run_pretraining_transfer as runner

FIXTURES = ROOT / "tests/fixtures/pretraining_transfer"
TEMPLATES = json.loads((FIXTURES / "templates.json").read_text())
BASE_RUN = TEMPLATES["run_manifest"]
BASE_RECORDS = TEMPLATES["records"]
TRAIN_TEMPLATE = next(r for r in BASE_RECORDS if r["event"] == "train")


def trainer_oracle():
	"""Use the actual trainer's config validator/schedule without its GPU imports."""
	module = types.ModuleType("pretraining_test_trainer_oracle")
	module.__dict__.update({"dataclass": dataclass, "math": math})
	sys.modules[module.__name__] = module
	qk = ast.parse((ROOT / "src/kimi_k3/training/qk_clip.py").read_text())
	trainer = ast.parse((ROOT / "src/kimi_k3/training/trainer.py").read_text())
	nodes = [n for n in qk.body if isinstance(n, ast.FunctionDef) and n.name in ("_positive", "_chunk_size")]
	nodes += [n for n in trainer.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
			  and n.name in ("TrainConfig", "learning_rate_at_step")]
	exec(compile(ast.Module(body=nodes, type_ignores=[]), "real_trainer_definitions", "exec"), module.__dict__)
	return module


ORACLE = trainer_oracle()
FINAL_LOSSES = {(.003,2048):6.0, (.001,2048):6.006, (.003,512):6.2}


def dump_records(path, records):
	path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")


def create_candidate(root, entry, directory, *, until=None, completed=True):
	config = yaml.safe_load((root / entry["training_config"]).read_text())
	oracle_config = ORACLE.TrainConfig(**config)
	batch = config["batch_size"] * config["seq_len"] * config["gradient_accumulation_steps"]
	target = config["max_steps"] * batch
	until = target if until is None else until
	manifest = deepcopy(BASE_RUN)
	manifest["run_id"] = entry["id"]
	session = manifest["sessions"][0]
	session["session_id"] = entry["id"] + "_session1"
	session["train_config"] = config
	session["checkpoint_path"] = str(directory / "last.pt")
	session["resume_checkpoint"] = None
	records = []
	for step in range(config["log_interval"], until // batch + 1, config["log_interval"]):
		record = deepcopy(TRAIN_TEMPLATE)
		tokens = step * batch
		record.update({"run_id": manifest["run_id"], "session_id": session["session_id"],
					   "step": step, "tokens_seen": tokens, "updates": config["log_interval"],
					   "attempted_updates": config["log_interval"], "tokens": config["log_interval"] * batch,
					   "lr": ORACLE.learning_rate_at_step(oracle_config, step),
					   "train_loss": 9 - 2 * tokens / target, "clipped_fraction": 0.5,
					   "train_seconds": 20.48, "train_tokens_per_second": 1000.0})
		record["qk_clip"]["observed_head_updates"] = 20 * config["log_interval"]
		for layer in record["qk_clip"]["layers"].values():
			layer["updates"] = config["log_interval"]
		record["session"] = {"elapsed_seconds": tokens / 1000, "train_seconds": tokens / 1000,
							 "eval_seconds": 0, "checkpoint_seconds": 0, "reporting_seconds": 0,
							 "profiling_seconds": 0, "tokens": tokens, "tokens_per_second": 1000}
		records.append(record)
	for step in range(config["eval_interval"], until // batch + 1, config["eval_interval"]):
		tokens = step * batch
		record = deepcopy(next(r for r in BASE_RECORDS if r["event"] == "val"))
		record.update({"run_id": manifest["run_id"], "session_id": session["session_id"],
					   "step": step, "tokens_seen": tokens, "eval_seconds": 1.0,
					   "val_loss": FINAL_LOSSES[(entry["learning_rate"], entry["effective_batch_tokens"])]
								   + 2 * (target - tokens) / target})
		record["session"] = {"elapsed_seconds": tokens / 1000, "train_seconds": tokens / 1000,
							 "eval_seconds": 0, "checkpoint_seconds": 0, "reporting_seconds": 0,
							 "profiling_seconds": 0}
		records.append(record)
	records.sort(key=lambda r: (r["step"], r["event"] == "val"))
	start = deepcopy(next(r for r in BASE_RECORDS if r["event"] == "session_start"))
	start.update({"run_id": manifest["run_id"], "session_id": session["session_id"]})
	records.insert(0, start)
	if completed and until == target:
		end = deepcopy(next(r for r in BASE_RECORDS if r["event"] == "session_end"))
		end.update({"run_id": manifest["run_id"], "session_id": session["session_id"],
					"step": config["max_steps"], "tokens_seen": target})
		end["session"] = deepcopy(records[-1]["session"])
		records.append(end)
	directory.mkdir(parents=True, exist_ok=True)
	analysis.write_json(directory / "run.json", manifest)
	dump_records(directory / "metrics.jsonl", records)
	return manifest, records


def create_study(path, *, complete_ids=None):
    plan = runner.load_plan(ROOT)
    path.mkdir()
    data = path.parent / "data"
    data.mkdir(exist_ok=True)
    analysis.write_json(data / "metadata.json", TEMPLATES["metadata"])
    state = runner.create_state(ROOT,path,data,plan)
    state["environment"] = plan["reference"]["environment"]
    state["power_profile"] = "synthetic-test"
    for entry in plan["branch5_transfer_runs"]:
        complete = complete_ids is None or entry["id"] in complete_ids
        state["runs"][entry["id"]] = {"status": "completed" if complete else "pending", "error": None,
                                     "attempts": [{"wall_seconds":2400}] if complete else []}
        if complete:
            create_candidate(ROOT,entry,path / "runs" / entry["id"])
    analysis.write_json(path / "study_state.json",state)
    return plan,state


def summarize(path):
	evidence = analysis.Evidence(path)
	try:
		return analysis.summarize_study(evidence)
	finally:
		evidence.close()


class ProtocolAndAnalysisTests(unittest.TestCase):
	def setUp(self):
		self.temp = tempfile.TemporaryDirectory()
		self.base = Path(self.temp.name)
		self.study = self.base / "study"
		self.plan, self.state = create_study(self.study)
		self.entry = self.plan["branch5_transfer_runs"][0]
		self.run = self.study / "runs" / self.entry["id"]

	def tearDown(self):
		self.temp.cleanup()

	def mutate_record(self, function):
		records = [json.loads(line) for line in (self.run / "metrics.jsonl").read_text().splitlines()]
		function(records)
		dump_records(self.run / "metrics.jsonl", records)

	def assert_candidate_invalid(self, fragment):
		candidate = summarize(self.study)["candidates"][0]
		self.assertFalse(candidate["eligible"])
		self.assertEqual(candidate["status"], "invalid")
		self.assertTrue(any(fragment in issue for issue in candidate["issues"]), candidate["issues"])

	def test_frozen_configs_match_actual_trainconfig_and_schedule(self):
		plan = runner.load_plan(ROOT)
		fractions = []
		for entry in plan["branch5_transfer_runs"]:
			config = ORACLE.TrainConfig(**yaml.safe_load((ROOT / entry["training_config"]).read_text()))
			self.assertEqual(config.max_steps * entry["effective_batch_tokens"], 2048000)
			values = [ORACLE.learning_rate_at_step(config, tokens // entry["effective_batch_tokens"]) /
					  config.learning_rate for tokens in (20480, 256000, 1024000, 2048000)]
			fractions.append(values)
		for values in fractions:
			for actual, expected in zip(values, fractions[0]):
				self.assertAlmostEqual(actual, expected)

	def test_complete_grid_has_known_ranking_tie_and_weighted_metrics(self):
		result = summarize(self.study)
		self.assertTrue(result["selection"]["grid_complete_and_valid"])
		self.assertEqual(result["selection"]["best_observed_run"], "122m_lr3e-3_b2048")
		self.assertEqual(set(result["selection"]["within_practical_tie"]),
						 {"122m_lr3e-3_b2048", "122m_lr1e-3_b2048"})
		self.assertTrue(result["selection"]["best_lr_is_grid_endpoint"])
		for row in result["candidates"]:
			self.assertEqual(row["tokens_seen"], 2048000)
			self.assertEqual(len(row["train"]), 100)
			self.assertEqual(len(row["validation"]), 8)
			self.assertAlmostEqual(row["grad_clipped_fraction"], 0.5)
			self.assertAlmostEqual(row["steady_tokens_per_second"], 1000)

	def test_real_short_diagnostic_is_not_a_grid_candidate(self):
		analysis.write_json(self.run / "run.json", BASE_RUN)
		dump_records(self.run / "metrics.jsonl", BASE_RECORDS)
		self.assert_candidate_invalid("TrainConfig differs")

	def test_token_discontinuity_is_excluded(self):
		self.mutate_record(lambda rs: next(r for r in rs if r["event"] == "train").update(tokens_seen=1))
		self.assert_candidate_invalid("token history")

	def test_wrong_schedule_is_excluded(self):
		self.mutate_record(lambda rs: next(r for r in rs if r["event"] == "train" and r["step"] > 10).update(lr=0.004))
		self.assert_candidate_invalid("Logged LR")

	def test_missing_qk_observations_are_excluded(self):
		self.mutate_record(lambda rs: next(r for r in rs if r["event"] == "train")["qk_clip"].update(observed_head_updates=0))
		self.assert_candidate_invalid("Missing QK")

	def test_nonfinite_qb_snapshot_is_excluded(self):
		def mutate(records):
			train = next(r for r in records if r["event"] == "train")
			next(iter(train["moe"].values()))["qb_biases"][0] = None
		self.mutate_record(mutate)
		self.assert_candidate_invalid("QB bias")

	def test_missing_expert_assignments_are_excluded(self):
		def mutate(records):
			train = next(r for r in records if r["event"] == "train")
			next(iter(train["moe"].values()))["expert_counts"][0] += 1
		self.mutate_record(mutate)
		self.assert_candidate_invalid("MoE assignments")

	def test_missing_evaluation_is_excluded(self):
		records = [json.loads(line) for line in (self.run / "metrics.jsonl").read_text().splitlines()]
		records = [r for r in records if not (r["event"] == "val" and r["tokens_seen"] == 256000)]
		dump_records(self.run / "metrics.jsonl", records)
		self.assert_candidate_invalid("Missing evaluations")

	def test_torn_tail_is_preserved_but_not_eligible(self):
		with (self.run / "metrics.jsonl").open("a") as stream:
			stream.write('{"event":')
		row = summarize(self.study)["candidates"][0]
		self.assertEqual(row["status"], "review_required")
		self.assertFalse(row["eligible"])
		self.assertIn("Uncommitted", row["warnings"][0])

	def test_changed_runtime_blocks_cross_candidate_comparison(self):
		manifest = json.loads((self.run / "run.json").read_text())
		manifest["sessions"][0]["environment"]["torch"] = "different"
		analysis.write_json(self.run / "run.json", manifest)
		result = summarize(self.study)
		self.assertFalse(any(r["eligible"] for r in result["candidates"]))
		self.assertIn("runtime differs", result["evidence_issues"][0])

	def test_changed_snapshot_blocks_comparison(self):
		with (self.study / "snapshots/model.yaml").open("a") as stream:
			stream.write("\n# changed evidence\n")
		result = summarize(self.study)
		self.assertFalse(any(r["eligible"] for r in result["candidates"]))
		self.assertIn("changed evidence snapshot", result["evidence_issues"][0])

	def test_checkpoint_rollback_removes_old_future_and_keeps_its_cost(self):
		manifest = json.loads((self.run / "run.json").read_text())
		full = [json.loads(line) for line in (self.run / "metrics.jsonl").read_text().splitlines()]
		checkpoint_tokens, old_end = 1024000, 1433600
		first = [r for r in full if r["tokens_seen"] <= old_end and r["event"] != "session_end"]
		second_session = deepcopy(manifest["sessions"][0])
		second_session["session_id"] = "resumed_session"
		second_session["rng_source"] = "checkpoint"
		second_session["resume_checkpoint"] = str(self.run / "last.pt")
		start = second_session["start_state"]
		start["step"] = checkpoint_tokens // self.entry["effective_batch_tokens"]
		start["tokens_seen"] = checkpoint_tokens
		start["loaders"]["train"]["position"] = checkpoint_tokens
		start["loaders"]["val"]["position"] = 65536
		second = [deepcopy(r) for r in full if r["tokens_seen"] > checkpoint_tokens]
		for r in second:
			r["session_id"] = "resumed_session"
		manifest["sessions"].append(second_session)
		analysis.write_json(self.run / "run.json", manifest)
		dump_records(self.run / "metrics.jsonl", first + second)
		self.state["runs"][self.entry["id"]]["attempts"] = [{"wall_seconds": 1700}, {"wall_seconds": 1200}]
		analysis.write_json(self.study / "study_state.json", self.state)
		row = summarize(self.study)["candidates"][0]
		self.assertTrue(row["eligible"], row["issues"])
		self.assertEqual(len(row["train"]), 100)
		self.assertEqual(len(row["validation"]), 8)
		self.assertEqual(row["discarded_logged_tokens"], 409600)
		self.assertEqual(row["process_wall_seconds"], 2900)

	def test_partial_grid_does_not_finalize_selection(self):
		partial = self.base / "partial"
		create_study(partial, complete_ids={self.entry["id"]})
		result = summarize(partial)
		self.assertFalse(result["selection"]["ready_for_candidate_review"])
		self.assertEqual(sum(r["eligible"] for r in result["candidates"]), 1)

	def test_zip_analysis_is_equivalent_and_excludes_checkpoints(self):
		(self.run / "last.pt").write_bytes(b"not a tensor checkpoint; exclusion fixture")
		archive = runner.archive_results(self.study)
		with ZipFile(archive) as z:
			self.assertFalse(any(name.endswith(".pt") for name in z.namelist()))
		self.assertEqual(summarize(archive)["selection"], summarize(self.study)["selection"])


class RunnerTests(unittest.TestCase):
	def setUp(self):
		self.temp = tempfile.TemporaryDirectory()
		self.base = Path(self.temp.name)

	def tearDown(self):
		self.temp.cleanup()

	def test_dry_run_has_three_commands_and_no_output_directory(self):
		output = self.base / "new_study"
		with redirect_stdout(io.StringIO()) as stream:
			status = runner.main(["--data-dir", str(self.base / "no_dataset_needed"), "--output-dir", str(output), "--dry-run"])
		plan = json.loads(stream.getvalue())
		self.assertEqual(status, 0)
		self.assertEqual(len(plan["runs"]), 3)
		self.assertFalse(output.exists())
		for row in plan["runs"]:
			self.assertNotIn("--profile", row["argv"])
			self.assertEqual(row["tokens"], 2048000)
		tree = ast.parse((ROOT / "scripts/train.py").read_text())
		node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
		import argparse
		namespace = {"argparse": argparse, "Path": Path, "__doc__": "CLI validation",
					 "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}
		exec(compile(ast.Module(body=[node], type_ignores=[]), "real_train_cli", "exec"), namespace)
		for row in plan["runs"]:
			parsed = namespace["parse_args"](row["argv"][3:])
			self.assertEqual(parsed.device, "cuda")
			self.assertFalse(parsed.profile)
			self.assertIsNotNone(parsed.model_config)

	def test_capture_real_child_failure_and_log(self):
		log = self.base / "child.log"
		with redirect_stdout(io.StringIO()):
			code, elapsed, interrupted = runner.run_logged(
				[sys.executable, "-c", "import sys; print('failure evidence',flush=True);sys.exit(7)"], log, ROOT)
		self.assertEqual(code, 7)
		self.assertGreater(elapsed, 0)
		self.assertFalse(interrupted)
		self.assertIn("failure evidence", log.read_text())

	@unittest.skipIf(os.name == "nt", "POSIX signal integration test")
	def test_capture_real_child_interrupt(self):
		log = self.base / "interrupt.log"
		timer = threading.Timer(0.15, lambda: os.kill(os.getpid(), signal.SIGINT))
		timer.start()
		try:
			with redirect_stdout(io.StringIO()):
				code, elapsed, interrupted = runner.run_logged(
					[sys.executable, "-u", "-c", "import time\nfor i in range(500):\n print(i,flush=True)\n time.sleep(.01)"], log, ROOT)
		finally:
			timer.cancel()
		self.assertTrue(interrupted)
		self.assertNotEqual(code, 0)
		self.assertLess(elapsed, 5)
		self.assertTrue(log.read_text())

	def test_study_lock_prevents_two_writers(self):
		with runner.study_lock(self.base):
			with self.assertRaisesRegex(RuntimeError, "Another runner"):
				with runner.study_lock(self.base):
					pass

	def test_end_to_end_runner_preserves_fixed_order_and_skips_completed_on_resume(self):
		fake_root = self.base / "repo"
		for source in [ROOT / name for name in runner.source_files(ROOT,runner.load_plan(ROOT))]:
			target = fake_root / source.relative_to(ROOT)
			target.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(source, target)
		data = self.base / "data"
		data.mkdir()
		analysis.write_json(data / "metadata.json",TEMPLATES["metadata"])
		output = self.base / "runner_study"
		calls = []
		entries = runner.load_plan(ROOT)["branch5_transfer_runs"]
		def fake_train(argv, log, root):
			destination = Path(argv[argv.index("--run-dir") + 1])
			entry = next(e for e in entries if e["id"] == destination.name)
			calls.append(entry["id"])
			create_candidate(fake_root, entry, destination)
			log.write_text("SYNTHETIC CONTROL FIXTURE — no GPU training\n")
			return 0, 2400, False
		with patch.object(runner, "ROOT", fake_root), patch.object(runner, "environment", return_value=runner.load_plan(ROOT)["reference"]["environment"]), \
			 patch.object(runner, "check_dataset", return_value={}), patch.object(runner, "run_logged", side_effect=fake_train), \
			 patch.object(analysis, "make_figures"), patch.object(analysis, "transfer_figures"), redirect_stdout(io.StringIO()):
			self.assertEqual(runner.main(["--data-dir", str(data), "--output-dir", str(output), "--max-runs", "1"]), 0)
			self.assertEqual(calls, [entries[0]["id"]])
			self.assertEqual(runner.main(["--output-dir", str(output), "--resume"]), 0)
		self.assertEqual(calls, [e["id"] for e in entries])
		self.assertTrue(summarize(output)["selection"]["grid_complete_and_valid"])
		self.assertTrue((output / "branch5_results.zip").is_file())
		self.assertTrue((output / "analysis/summary.csv").is_file())


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.study = self.base / "study"
        self.plan, self.state = create_study(self.study)

    def tearDown(self):
        self.temp.cleanup()

    def review(self, path=None):
        evidence = analysis.Evidence(path or self.study)
        try:
            result = analysis.summarize_study(evidence)
            result["transfer"] = analysis.transfer_review(result, evidence.json("snapshots/reference_82m.json"),
                                                          evidence.json("study_manifest.json"))
            return result
        finally:
            evidence.close()

    def test_known_cross_scale_contrasts_and_operational_tie(self):
        result = self.review()
        self.assertEqual(result["transfer"]["status"], "ready_for_review")
        contrasts = result["transfer"]["contrasts"]
        self.assertAlmostEqual(contrasts[0]["delta_82m"], -0.12434100545942783)
        self.assertAlmostEqual(contrasts[0]["delta_122m"], -0.006)
        self.assertEqual(contrasts[0]["outcome_122m"], "practical_tie")
        self.assertAlmostEqual(contrasts[1]["delta_122m"], -0.2)
        self.assertEqual(contrasts[1]["outcome_122m"], "treatment_lower_loss")
        self.assertEqual(len(contrasts[0]["curve_122m"]), 8)

    def test_incomplete_pair_does_not_supply_final_contrast(self):
        partial = self.base / "partial"
        create_study(partial, complete_ids={"122m_lr3e-3_b2048"})
        result = self.review(partial)
        self.assertFalse(result["selection"]["ready_for_candidate_review"])
        self.assertTrue(all(c["delta_122m"] is None for c in result["transfer"]["contrasts"]))
        self.assertEqual(sum(r["loss_122m"] is not None for r in result["transfer"]["comparisons"]), 1)

    def test_wrong_parameter_count_invalidates_122m_candidate(self):
        entry = self.plan["branch5_transfer_runs"][0]
        path = self.study / "runs" / entry["id"] / "run.json"
        manifest = json.loads(path.read_text())
        manifest["sessions"][0]["parameter_count"] = 82221540
        analysis.write_json(path, manifest)
        row = self.review()["candidates"][0]
        self.assertFalse(row["eligible"])
        self.assertIn("122M proxy", row["issues"][0])

    def test_matching_122m_runtime_can_still_fail_cross_scale_transfer(self):
        for entry in self.plan["branch5_transfer_runs"]:
            path = self.study / "runs" / entry["id"] / "run.json"
            manifest = json.loads(path.read_text())
            manifest["sessions"][0]["environment"]["different_runtime_for_test"] = True
            analysis.write_json(path, manifest)
        result = self.review()
        self.assertTrue(all(r["eligible"] for r in result["candidates"]))
        self.assertFalse(result["selection"]["cross_scale_comparison_valid"])
        self.assertFalse(result["selection"]["ready_for_candidate_review"])
        self.assertTrue(all(c["delta_122m"] is None for c in result["transfer"]["contrasts"]))

    def test_changed_reference_is_rejected_even_if_state_hash_is_updated(self):
        path = self.study / "snapshots/reference_82m.json"
        value = json.loads(path.read_text())
        value["candidates"][0]["final_val_loss"] = 0.0
        analysis.write_json(path, value)
        self.state["snapshot_sha256"]["snapshots/reference_82m.json"] = runner.file_hash(path)
        analysis.write_json(self.study / "study_state.json", self.state)
        with self.assertRaisesRegex(ValueError, "82M reference changed"):
            self.review()

    def test_no_checkpoint_archives_attempt_and_restarts_at_zero(self):
        entry = self.plan["branch5_transfer_runs"][0]
        ledger = self.state["runs"][entry["id"]]
        old_manifest = (self.study / "runs" / entry["id"] / "run.json").read_bytes()
        argv, log = runner.prepare_attempt(ROOT, self.study, self.base / "data", entry, ledger)
        archived = self.study / "abandoned_attempts" / entry["id"] / "attempt_01" / "run.json"
        self.assertEqual(archived.read_bytes(), old_manifest)
        self.assertNotIn("--resume", argv)
        self.assertFalse((self.study / "runs" / entry["id"]).exists())
        self.assertIn("attempt_02", log.name)

    def test_plot_failure_keeps_raw_evidence_and_produces_zip(self):
        with patch.object(runner, "analyze", side_effect=RuntimeError("synthetic plotting failure")), redirect_stdout(io.StringIO()):
            self.assertIsNone(runner.refresh_results(self.study, self.state))
        self.assertIn("synthetic plotting failure", self.state["analysis_error"])
        with ZipFile(self.study / "branch5_results.zip") as bundle:
            self.assertIn("study_state.json", bundle.namelist())
            self.assertIn("runs/122m_lr3e-3_b2048/metrics.jsonl", bundle.namelist())


if __name__ == "__main__":
	unittest.main()

