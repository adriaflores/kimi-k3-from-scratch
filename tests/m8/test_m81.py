"""Stdlib-only M8.1 regression gates: no GPU or training dependencies."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("m81", ROOT / "scripts/m8/m81.py")
m81 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m81)
RECIPE = ROOT / "experiments/m7/m75_gate/recipe/m8_full_6b.yaml"
RECEIPT = ROOT / "experiments/m8/m81/inputs/m75_resume_receipt.json"


class ContractTests(unittest.TestCase):
    def test_exact_recipe_and_horizon(self):
        r = m81.validate_recipe(RECIPE)
        self.assertEqual(r["sha256"], m81.M8_RECIPE_SHA)
        self.assertEqual((r["remaining_updates"], r["remaining_tokens"]), (180664, 5919997952))
        self.assertEqual(r["nominal_total_tokens"], 6000017408)
        self.assertEqual(r["first_periodic_checkpoint_updates_from_resume"], 1654)

    def test_changed_yaml_must_fail_even_if_meaning_identical(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "changed.yaml"
            p.write_bytes(RECIPE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                m81.validate_recipe(p)

    def test_duplicate_yaml_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "duplicate.yaml"
            p.write_text("max_steps: 183106\nmax_steps: 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                m81.flat_recipe(p)

    def test_m75_gate_receipt(self):
        self.assertEqual(m81.validate_m75_receipt(RECEIPT)["status"], "PASS")

    def test_m75_receipt_is_host_platform_independent(self):
        """Regression: WindowsPath.__str__ uses backslashes for Linux receipt paths."""
        source = json.loads(RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(source["checkpoint_path"], m81.SOURCE_CHECKPOINT_POSIX)
        # On Linux, simulate what pathlib does natively on Windows.
        with mock.patch.object(m81, "SOURCE_CHECKPOINT", PureWindowsPath(m81.SOURCE_CHECKPOINT_POSIX)):
            self.assertEqual(m81.validate_m75_receipt(RECEIPT)["status"], "PASS")

    def test_tampered_receipt_fails(self):
        with tempfile.TemporaryDirectory() as t:
            j = json.loads(RECEIPT.read_text(encoding="utf-8"))
            for key, changed in [("status", "FAIL"), ("checkpoint_sha256_after", "0" * 64),
                                 ("restored_train_position", 0), ("next_lr", 0.0),
                                 ("rng_verified", ["python"])]:
                original = j[key]
                j[key] = changed
                p = Path(t) / "mutated.json"
                p.write_text(json.dumps(j), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    m81.validate_m75_receipt(p)
                j[key] = original

    def test_cost_must_be_current_and_within_cap(self):
        self.assertEqual(m81.check_cost(2.3, 100.0, 30)["estimated_compute_eur_excluding_fees"], 69)
        for price, cap, hours in [(2.3, 50, 30), (0, 100, 30), (float("nan"), 100, 30),
                                  (2, -1, 30), (float("inf"), 100, 30), (2, 100, 0)]:
            with self.subTest(price=price, cap=cap, hours=hours), self.assertRaises(ValueError):
                m81.check_cost(price, cap, hours)

    def test_wrong_disk_policy_fails(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError, "20 GiB"):
                m81.check_disk(Path(t), 10)
            with mock.patch.object(m81.shutil, "disk_usage", return_value=mock.Mock(free=19 * 1024**3)):
                with self.assertRaisesRegex(ValueError, "below"):
                    m81.check_disk(Path(t), 20)

    def test_manifest_matches_immutable_files_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "reference"
            root.mkdir()
            names = ["scripts/train.py", "src/kimi_k3/training/trainer.py",
                     "configs/model/kimi_k3_300m.yaml"] + [f"src/fake_{i:02d}.py" for i in range(32)]
            manifest = {}
            for name in names:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(name, encoding="utf-8")
                manifest[name] = m81.sha256(p)
            mf = Path(t) / "reference_manifest.json"
            mf.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(m81.check_source_manifest(root, mf)["source_file_count"], 35)
            (root / names[-1]).write_text("different", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed/missing"):
                m81.check_source_manifest(root, mf)
            manifest[names[-1]] = m81.sha256(root / names[-1])
            manifest["../outside"] = "0" * 64
            del manifest[names[-2]]
            mf.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed/missing"):
                m81.check_source_manifest(root, mf)

    def test_missing_source_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                m81.check_source_manifest(Path(t), Path(t) / "missing.json")

    def test_output_path_is_new_and_does_not_overwrite_source(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "workspace" / "kimi"
            (root / "reference").mkdir(parents=True)
            (root / "data/full").mkdir(parents=True)
            ckpt = root / "runs/m73/original/last.pt"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_text("checkpoint", encoding="utf-8")
            (root / "runs/m8").mkdir(parents=True)
            new = root / "runs/m8/full_v1"
            receipt = root / "artifacts/m8/m81/preflight.json"
            with mock.patch.object(m81, "SOURCE_CHECKPOINT", ckpt.resolve()):
                paths = m81.check_paths(root, root / "reference", root / "data/full", ckpt, new, receipt)
                self.assertTrue(paths["run_dir_does_not_exist"])
                new.mkdir()
                with self.assertRaisesRegex(ValueError, "NEW"):
                    m81.check_paths(root, root / "reference", root / "data/full", ckpt, new, receipt)
                new.rmdir()
                with self.assertRaisesRegex(ValueError, "NEW"):
                    m81.check_paths(root, root / "reference", root / "data/full", ckpt,
                                    root / "runs/m73/escape", receipt)
                with self.assertRaisesRegex(ValueError, "dedicated"):
                    m81.check_paths(root, root / "reference", root / "data/full", ckpt,
                                    new, root / "reference/unsafe.json")

    def test_checkpoint_header_real_contract_mock_only(self):
        """Logic under test, not a claim that the real 2.5GB checkpoint was read."""
        with tempfile.TemporaryDirectory() as t:
            ckpt = Path(t) / "fake.pt"
            ckpt.write_bytes(b"fake-checkpoint")
            saved = {"format_version": 7, "step": 2442, "tokens_seen": 80019456,
                     "skipped_updates": 0, "model": {}, "optimizer": {}, "optimizer_layout": [],
                     "loaders": {"train": {"position": 80019456}, "val": {"position": 262144}},
                     "rng_state": {}, "runtime": {}, "qk_clip": {}}
            torch = mock.Mock()
            torch.load.return_value = saved
            with mock.patch.object(m81, "SOURCE_CHECKPOINT_BYTES", ckpt.stat().st_size), \
                 mock.patch.object(m81, "sha256", return_value=m81.SOURCE_CHECKPOINT_SHA):
                self.assertTrue(m81.check_checkpoint_header(ckpt, torch)["intact"])
                self.assertTrue(torch.load.call_args.kwargs["weights_only"])
                saved["loaders"]["train"]["position"] = 0
                with self.assertRaisesRegex(ValueError, "cursors"):
                    m81.check_checkpoint_header(ckpt, torch)

    def test_integrated_local_receipt_and_no_training(self):
        with tempfile.TemporaryDirectory() as t:
            dest = Path(t) / "receipt.json"
            self.assertEqual(m81.main(["--mode", "local", "--repo-root", str(ROOT),
                                       "--receipt", str(dest)]), 0)
            obj = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(obj["status"], "PASS")
            self.assertFalse(obj["training_started"])
            self.assertFalse(obj["gpu_accessed"])
            self.assertEqual(obj["remote_preflight_status"], "PENDING")

    def test_remote_missing_price_is_no_go_before_gpu_import(self):
        args = m81.parse_args(["--mode", "remote", "--repo-root", str(ROOT)])
        with self.assertRaisesRegex(ValueError, "REQUIRES"):
            m81.execute(args)


if __name__ == "__main__":
    unittest.main()
