import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.train import build_model
from src.kimi_k3.model import KimiK3Config
from src.kimi_k3.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPO_ROOT / "configs/model/kimi_k3_tiny.yaml"
TRAIN_SCRIPT = REPO_ROOT / "scripts/train.py"


@pytest.fixture
def prepared_data(tmp_path):
	directory = tmp_path / "data"
	directory.mkdir()
	metadata = {
		"format_version": 2,
		"storage": {"format": "npy", "dtype": "uint16"},
		"tokenizer": {"vocab_size": 49152, "eos_id": 0},
		"splits": {},
	}
	for split in ("train", "val"):
		tokens = np.arange(33, dtype=np.uint16)
		filename = f"{split}_000.npy"
		np.save(directory / filename, tokens)
		metadata["splits"][split] = {
			"tokens": len(tokens),
			"shards": [{"file": filename, "tokens": len(tokens)}],
		}
	(directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
	return directory


@pytest.mark.parametrize("smoke_test", [False, True], ids=["kimi_k3_yaml", "legacy_smoke_test"])
def test_cli_trains_and_saves_selected_model(prepared_data, tmp_path, smoke_test):
	checkpoint = tmp_path / "last.pt"
	run_dir = tmp_path / "run"
	output_args = ["--run-dir", str(run_dir), "--checkpoint", str(checkpoint)] if smoke_test else []
	selection = ["--smoke-test"] if smoke_test else ["--model-config", str(MODEL_CONFIG)]
	result = subprocess.run(
		[sys.executable, str(TRAIN_SCRIPT), "--data-dir", str(prepared_data),
		 *selection, "--device", "cpu", "--batch-size", "1", "--seq-len", "8",
		 "--max-steps", "2", "--log-interval", "1", "--eval-batches", "1",
		 "--seed", "123", *output_args],
		# Exercise direct script imports outside the repository root too.
		cwd=tmp_path, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
		capture_output=True, text=True, timeout=60, check=False,
	)
	assert result.returncode == 0, result.stdout + result.stderr
	assert "step 2 | train loss" in result.stdout
	assert "step 2 | val loss" in result.stdout
	if not smoke_test:
		run_dirs = list((tmp_path / "results/training/runs").iterdir())
		assert len(run_dirs) == 1
		run_dir = run_dirs[0]
		checkpoint = run_dir / "last.pt"
	manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
	session = manifest["sessions"][0]
	records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(
		encoding="utf-8").splitlines()]
	assert [record["event"] for record in records] == [
		"session_start", "train", "train", "val", "checkpoint", "session_end"]
	assert all(record["run_id"] == manifest["run_id"] and record["session_id"] == session["session_id"]
			   for record in records)
	assert [record["step"] for record in records if record["event"] == "train"] == [1, 2]
	assert records[-1]["step"] == 2
	assert records[-1]["tokens_seen"] == 16
	assert records[-1]["session"]["tokens"] == 16
	assert records[-1]["session"]["elapsed_seconds"] > 0
	assert records[-1]["session"]["tokens_per_second"] > 0
	rss = records[-1]["memory"]["ram_rss_bytes"]
	assert rss is None or rss > 0
	if rss is not None:
		assert records[-1]["memory"]["ram_rss_sampled_peak_bytes"] >= rss
	assert all(value is None for key, value in records[-1]["memory"].items() if key.startswith("cuda_"))
	assert session["train_config"]["seed"] == 123
	assert session["train_config"]["seq_len"] == 8
	assert session["checkpoint_path"] == str(checkpoint)
	assert len(session["git"]["commit"]) == 40
	saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
	assert saved["format_version"] == 6
	assert saved["optimizer_type"].endswith(".MuonAdamW")
	assert "Muon momentum=" in result.stdout
	assert {group["algorithm"] for group in saved["optimizer"]["param_groups"]} == (
		{"adamw"} if smoke_test else {"muon", "adamw"})
	assert saved["step"] == 2
	assert saved["tokens_seen"] == 16
	assert saved["optimizer"]["state"]
	assert session["model_config"] == json.loads(json.dumps(saved["model_config"]))
	assert all(torch.isfinite(weight).all() for weight in saved["model"].values())
	for record in records:
		if record["event"] != "train":
			continue
		if smoke_test:
			assert record["moe"] == {}
		else:
			architecture = saved["model_config"]
			assert set(record["moe"]) == {
				f"layers.{i}.ffn" for i in range(architecture["num_dense_layers"], architecture["num_layers"])}
			for layer in record["moe"].values():
				assert layer["tokens"] == record["tokens"] == 8
				assert sum(layer["expert_counts"]) == 8 * architecture["num_experts_per_token"]
				assert len(layer["qb_biases"]) == architecture["num_experts"]
	if smoke_test:
		assert saved["model_config"]["model_type"] == "smoke_test_lm"
	else:
		assert "KimiK3 | model config:" in result.stdout
		assert saved["model_config"] == asdict(load_config(MODEL_CONFIG, KimiK3Config))
		# A real update must change the initialized KimiK3 weights.
		torch.manual_seed(123)
		initial = build_model(49152, 8, model_config_path=MODEL_CONFIG)
		assert not torch.equal(saved["model"]["embed_tokens.weight"], initial.embed_tokens.weight)


@pytest.mark.parametrize("vocab_size,seq_len,message", [
	(32, 8, "does not match dataset vocab_size"),
	(49152, 132, "exceeds model max_seq_len"),
	(49152, 6, "must be divisible by kda_chunk_size"),
])
def test_incompatible_data_dimensions_are_rejected(vocab_size, seq_len, message):
	with pytest.raises(ValueError, match=message):
		build_model(vocab_size, seq_len, model_config_path=MODEL_CONFIG)


def test_fused_kda_model_does_not_require_kda_chunk_alignment(tmp_path):
	pytest.importorskip("fla.ops.kda", reason="FLA is required to construct the fused KDA")
	data = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
	data.update(kda_implementation="fused", kda_chunk_size=64)
	path = tmp_path / "fused_kda.yaml"
	path.write_text(yaml.safe_dump(data), encoding="utf-8")
	with torch.device("meta"):
		model = build_model(49152, 8, model_config_path=path)
	assert type(model.layers[0].attn).__name__ == "FusedKDAChunkwise"


def test_mla_only_model_does_not_require_kda_chunk_alignment(tmp_path):
	data = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
	data["layer_types"] = ["mla"] * data["num_layers"]
	path = tmp_path / "mla_only.yaml"
	path.write_text(yaml.safe_dump(data), encoding="utf-8")
	with torch.device("meta"):
		model = build_model(49152, 6, model_config_path=path)
	assert model.config.layer_types == ("mla",) * data["num_layers"]
