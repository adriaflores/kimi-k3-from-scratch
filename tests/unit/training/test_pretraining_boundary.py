"""Boundary protocol tests; generated losses below are synthetic test controls."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_pretraining_study as fixtures

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import scripts.analyze_pretraining_study as analysis
import scripts.run_pretraining_boundary as boundary


class BoundaryTests(unittest.TestCase):
	def setUp(self):
		self.temp = tempfile.TemporaryDirectory()
		self.directory = Path(self.temp.name)
		self.plan = boundary.load_plan(ROOT)

	def tearDown(self):
		self.temp.cleanup()

	def clone_root(self):
		root = self.directory / "repo"
		names = set(self.plan["parent_grid"]["source_files_sha256"])
		names |= {str(boundary.MANIFEST), "scripts/run_pretraining_boundary.py",
				  "scripts/analyze_pretraining_study.py"}
		names |= {row["training_config"] for row in self.plan["branch4_exploration_runs"]}
		for name in names:
			destination = root / name
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(ROOT / name, destination)
		return root

	def result(self, *, first_loss=6.20, second_loss=5.90, complete=True):
		signature = self.plan["parent_grid"]["candidates"][0]["comparison_signature"]
		candidates = []
		for entry, loss in zip(self.plan["branch4_exploration_runs"], (first_loss, second_loss)):
			candidates.append({**entry, "eligible": complete, "status": "complete" if complete else "incomplete",
							   "tokens_seen": 2048000 if complete else 512000,
							   "final_val_loss": loss if complete else None,
							   "comparison_signature": signature, "process_wall_seconds": 2400,
							   "issues": [], "warnings": []})
		state = {"runs": {r["id"]: {"status": "completed" if complete else "pending"} for r in candidates}}
		return {"candidates": candidates, "evidence_issues": []}, state

	def test_only_two_complete_recipes_at_unchanged_token_budget(self):
		for entry in self.plan["branch4_exploration_runs"]:
			import yaml
			config = yaml.safe_load((ROOT / entry["training_config"]).read_text())
			config = fixtures.ORACLE.TrainConfig(**config)
			self.assertEqual(config.learning_rate, 0.003)
			self.assertEqual(config.max_steps * entry["effective_batch_tokens"], 2048000)
			self.assertEqual(config.warmup_steps * entry["effective_batch_tokens"], 20480)
			self.assertAlmostEqual(fixtures.ORACLE.learning_rate_at_step(config, config.max_steps), 0.0003, places=14)
		self.assertEqual(len(self.plan["branch4_exploration_runs"]), 2)

	def test_changed_parent_source_is_blocked(self):
		root = self.clone_root()
		(root / "scripts/train.py").write_text("# changed training code\n")
		with self.assertRaisesRegex(ValueError, "differs from the completed parent"):
			boundary.load_plan(root)

	def test_changed_manifest_is_blocked(self):
		root = self.clone_root()
		changed = deepcopy(self.plan)
		changed["common"]["seed"] = 2027
		analysis.write_json(root / boundary.MANIFEST, changed)
		with self.assertRaisesRegex(ValueError, "manifest changed"):
			boundary.load_plan(root)

	def test_dry_run_uses_two_new_runs_without_gpu_or_output(self):
		output = self.directory / "not_created"
		with patch.object(boundary, "environment") as probe, redirect_stdout(io.StringIO()) as stream:
			self.assertEqual(boundary.main(["--dry-run", "--data-dir", str(self.directory),
											"--output-dir", str(output)]), 0)
		probe.assert_not_called()
		self.assertFalse(output.exists())
		runs = json.loads(stream.getvalue())["runs"]
		self.assertEqual([r["id"] for r in runs], [r["id"] for r in self.plan["branch4_exploration_runs"]])

	def test_joint_selection_and_practical_tie_include_parent(self):
		result, state = self.result(second_loss=6.04)
		joint = boundary.joint_review(result, self.plan, state)
		self.assertTrue(joint["all_eight_complete_and_valid"])
		self.assertEqual(joint["best_observed_run"], "82m_lr3e-3_b2048")
		self.assertIn("82m_lr1e-3_b2048", joint["within_practical_tie"])
		self.assertTrue(joint["best_lr_is_upper_endpoint"])
		self.assertEqual(joint["further_local_82m_runs"], 0)

	def test_partial_extension_does_not_finalize(self):
		result, state = self.result(complete=False)
		joint = boundary.joint_review(result, self.plan, state)
		self.assertEqual(joint["status"], "provisional")
		self.assertFalse(joint["all_boundary_candidates_attempted"])
		self.assertEqual(joint["best_observed_run"], "82m_lr1e-3_b2048")

	def test_runtime_mismatch_excludes_boundary_from_parent_ranking(self):
		result, state = self.result()
		result["candidates"][1]["comparison_signature"] = "changed-runtime"
		joint = boundary.joint_review(result, self.plan, state)
		self.assertFalse(joint["all_eight_complete_and_valid"])
		self.assertEqual(joint["status"], "provisional")
		self.assertEqual(joint["best_observed_run"], "82m_lr1e-3_b2048")
		self.assertTrue(joint["issues"])
		self.assertFalse(result["candidates"][1]["eligible"])

	def test_changed_environment_stops_before_training(self):
		root = self.clone_root()
		data = self.directory / "data"
		data.mkdir()
		(data / "metadata.json").write_text("{}")
		output = self.directory / "blocked"
		with patch.object(boundary, "ROOT", root), \
			 patch.object(boundary, "environment", return_value={"changed": True}), \
			 patch.object(boundary, "check_dataset", return_value={}), \
			 patch.object(boundary, "run_logged") as training, \
			 patch.object(analysis, "make_figures"), \
			 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
			self.assertEqual(boundary.main(["--data-dir", str(data), "--output-dir", str(output)]), 1)
		training.assert_not_called()
		state = json.loads((output / "study_state.json").read_text())
		self.assertIn("differs from the completed parent", state["error"])
		joint = json.loads((output / "analysis/joint_selection.json").read_text())
		self.assertEqual(joint["status"], "provisional")

	def test_runner_partial_then_resume_writes_joint_zip_without_rerun(self):
		root = self.clone_root()
		data = self.directory / "data"
		data.mkdir()
		(data / "metadata.json").write_text("{}")
		output = self.directory / "study"
		observed = []
		identity = json.loads((ROOT / "tests/fixtures/pretraining_study/branch4_parent_identity.json").read_text())

		def fake_train(argv, log, working_root):
			run_dir = Path(argv[argv.index("--run-dir") + 1])
			entry = next(e for e in self.plan["branch4_exploration_runs"] if e["id"] == run_dir.name)
			observed.append(entry["id"])
			manifest, _ = fixtures.create_candidate(root, entry, run_dir)
			manifest["sessions"][0].update(identity)
			analysis.write_json(run_dir / "run.json", manifest)
			log.write_text("Synthetic test trainer; no GPU training.\n")
			return 0, 2400.0, False

		with patch.object(boundary, "ROOT", root), \
			 patch.object(boundary, "environment", return_value=self.plan["parent_grid"]["environment"]), \
			 patch.object(boundary, "check_dataset", return_value={}), \
			 patch.object(boundary, "run_logged", side_effect=fake_train), \
			 patch.object(analysis, "make_figures"), \
			 patch.dict(fixtures.FINAL_LOSSES, {(0.003, 512): 6.20, (0.003, 2048): 5.90}), \
			 redirect_stdout(io.StringIO()):
			self.assertEqual(boundary.main(["--data-dir", str(data), "--output-dir", str(output), "--max-runs", "1"]), 0)
			partial = json.loads((output / "analysis/joint_selection.json").read_text())
			self.assertFalse(partial["all_boundary_candidates_attempted"])
			self.assertEqual(boundary.main(["--output-dir", str(output), "--resume"]), 0)
			self.assertEqual(boundary.main(["--output-dir", str(output), "--pack-only"]), 0)
		self.assertEqual(observed, [e["id"] for e in self.plan["branch4_exploration_runs"]])
		joint = json.loads((output / "analysis/joint_selection.json").read_text())
		self.assertTrue(joint["all_eight_complete_and_valid"])
		self.assertEqual(joint["best_observed_run"], "82m_lr3e-3_b2048")
		from zipfile import ZipFile
		with ZipFile(output / "branch4_results.zip") as archive:
			self.assertIn("analysis/joint_selection.json", archive.namelist())
			self.assertIn("source_snapshot/scripts/run_pretraining_boundary.py", archive.namelist())
			self.assertFalse(any(name.endswith(".pt") for name in archive.namelist()))


if __name__ == "__main__":
	unittest.main()
