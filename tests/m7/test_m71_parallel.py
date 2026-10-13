"""Offline safety checks for bounded concurrent M7 shard uploads."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/m7/vendor"))
import scripts.m7.vendor.branch4_transport as transfer


class ConnectionStub:
    remote_root = "/workspace/kimi"

    def __init__(self):
        self.calls = []

    def rpc(self, args):
        self.calls.append(args[0])
        if args[0] == "verify-dataset":
            return {"verified": True, "files": 2}
        if args[0] == "mkdir":
            return {"created": True}
        raise AssertionError(args)


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        (self.root / "data").mkdir()
        (self.root / "inspection").mkdir()
        (self.root / "data/metadata.json").write_text("{}", encoding="utf-8")
        self.manifest = {
            "files": [
                {"file": f"train_{i:05d}.npy", "file_bytes": 1, "sha256": "a" * 64}
                for i in range(2)
            ],
            "metadata_file_sha256": "localhash",
        }
        for i in range(2):
            (self.root / f"data/train_{i:05d}.npy").write_bytes(b"x")
        self.connection = ConnectionStub()

    def run_upload(self, upload_one, workers="2"):
        def fake_read_json(path):
            if path.name == "inspection.json":
                return {"dataset_verified": True, "errors": []}
            if path.name == "verified_data_manifest.json":
                return self.manifest
            raise AssertionError(path)
        with patch.dict(os.environ, {"M7_TRANSFER_WORKERS": workers}), \
             patch.object(transfer, "read_json", side_effect=fake_read_json), \
             patch.object(transfer, "sha256_file", return_value="localhash"), \
             patch.object(transfer, "upload_one", side_effect=upload_one):
            return transfer.upload_data(self.connection, self.root / "data", self.root / "inspection", "full", False)

    def test_concurrent_completion_preserves_manifest_order_and_metadata_last(self):
        barrier = threading.Barrier(2, timeout=3)
        lock = threading.Lock()
        finished = set()
        def upload_one(connection, local, remote, size, digest, restart):
            if local.suffix == ".npy":
                barrier.wait()
                if local.name.endswith("00000.npy"):
                    time.sleep(0.025)
                with lock:
                    finished.add(local.name)
                return {"file": local.name, "bytes": 1, "skipped_verified": True}
            self.assertEqual(len(finished), 2)
            return {"file": local.name, "bytes": 2, "skipped_verified": True}
        result = self.run_upload(upload_one)
        self.assertEqual([row["file"] for row in result["files"]][:2],
                         ["train_00000.npy", "train_00001.npy"])
        self.assertEqual(self.connection.calls, ["mkdir", "verify-dataset"])

    def test_error_never_publishes_metadata(self):
        touched = []
        def upload_one(connection, local, remote, size, digest, restart):
            touched.append(local.name)
            if local.name.endswith("00000.npy"):
                raise OSError("broken socket")
            return {"file": local.name, "bytes": 1, "skipped_verified": True}
        with self.assertRaisesRegex(OSError, "broken socket"):
            self.run_upload(upload_one)
        self.assertNotIn("metadata.json", touched)
        self.assertNotIn("verify-dataset", self.connection.calls)

    def test_rejects_excessive_concurrency(self):
        with self.assertRaisesRegex(ValueError, "M7_TRANSFER_WORKERS"):
            self.run_upload(lambda *args: None, workers="10")
        self.assertEqual(self.connection.calls, ["mkdir"])


if __name__ == "__main__":
    unittest.main()
