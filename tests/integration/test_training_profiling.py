import gzip
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from kimi_k3.model import KimiK3
from kimi_k3.training.metrics import MetricsWriter
from kimi_k3.training.profiling import ProfileConfig
from kimi_k3.training.run import start_run
from kimi_k3.training.trainer import (
	TrainConfig,
	TrainState,
	_capture_rng_state,
	build_optimizer,
	train,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def prepared_data(tmp_path):
	directory = tmp_path / "data"
	directory.mkdir()
	metadata = {"format_version": 2, "storage": {"format": "npy", "dtype": "uint16"},
				"tokenizer": {"vocab_size": 32, "eos_id": 0}, "splits": {}}
	for split in ("train", "val"):
		name = f"{split}_000.npy"
		np.save(directory / name, np.arange(65, dtype=np.uint16) % 32)
		metadata["splits"][split] = {"tokens": 65, "shards": [{"file": name, "tokens": 65}]}
	(directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
	return directory


def _architecture(factory):
	return factory(num_layers=3, layer_types=("kda", "mla", "kda"), num_dense_layers=1,
				   attn_res_block_size=2, num_experts=4, num_experts_per_token=2)


def _assert_equal(actual, expected):
	if isinstance(actual, torch.Tensor):
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	elif isinstance(actual, dict):
		assert actual.keys() == expected.keys()
		for key in actual:
			_assert_equal(actual[key], expected[key])
	elif isinstance(actual, (tuple, list)):
		assert type(actual) is type(expected) and len(actual) == len(expected)
		for left, right in zip(actual, expected, strict=True):
			_assert_equal(left, right)
	else:
		assert actual == expected


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
	not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_real_profile_is_bounded_and_preserves_training_state(
	tmp_path, prepared_data, kda_config_factory, monkeypatch, device,
):
	monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
	deterministic = torch.are_deterministic_algorithms_enabled()
	torch.use_deterministic_algorithms(True)
	try:
		def run(profile):
			random.seed(123)
			np.random.seed(123)
			torch.manual_seed(123)
			config = TrainConfig(batch_size=1, seq_len=8, max_steps=6, log_interval=1,
								 eval_interval=1, eval_batches=1, checkpoint_interval=1,
								 gradient_accumulation_steps=2,
								 precision="bf16" if device == "cuda" else "fp32")
			model = KimiK3(_architecture(kda_config_factory)).to(device)
			optimizer = build_optimizer(model, config)
			loaders = [TokenDataLoader(ShardedTokenDataset(prepared_data, split), 1, 8)
					   for split in ("train", "val")]
			state = TrainState()
			run = start_run(model=model, optimizer=optimizer, config=config, state=state,
							train_loader=loaders[0], val_loader=loaders[1], repo_root=tmp_path,
							runs_root=tmp_path / "runs")
			with MetricsWriter(run) as metrics:
				train(model, optimizer, *loaders, config, device, state=state,
					  checkpoint_path=run.checkpoint_path, metrics=metrics,
					  profile_config=ProfileConfig(active=2) if profile else None)
			assert all(not module._forward_hooks and not module._forward_pre_hooks
					   for module in model.modules())
			return run, _capture_rng_state(device), torch.load(run.checkpoint_path, weights_only=True)

		baseline_run, baseline_rng, baseline = run(False)
		profiled_run, profiled_rng, profiled = run(True)
		_assert_equal(profiled_rng, baseline_rng)
		_assert_equal(profiled, baseline)
	finally:
		torch.use_deterministic_algorithms(deterministic)
	assert not (baseline_run.directory / "profiles").exists()
	directory = profiled_run.directory / "profiles" / profiled_run.session_id
	metadata = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
	assert metadata["status"] == "complete" and metadata["stop_reason"] == "window_complete"
	assert metadata["run_id"] == profiled_run.run_id and metadata["session_id"] == profiled_run.session_id
	assert metadata["attempts_completed"] == 4 and metadata["active_attempts_completed"] == 2
	assert metadata["start_state"]["step"] == 0 and metadata["end_state"]["step"] == 4
	assert metadata["end_state"]["tokens_seen"] == 64 and profiled["tokens_seen"] == 96
	assert not metadata["options"]["record_shapes"] and not metadata["options"]["with_stack"]
	regions = {region["name"]: region for region in metadata["regions"]}
	assert regions["train/forward"]["calls"] == regions["train/backward"]["calls"] == 4
	assert regions["train/optimizer"]["calls"] == 2
	assert regions["train/qk_clip"]["calls"] == 2
	assert regions["model/StableLatentMoE"]["calls"] == 8
	assert regions["model/KDAChunkwise"]["calls"] == 8
	assert regions["model/GatedMLA"]["calls"] == 4
	assert regions["model/DenseFFN"]["calls"] == 4
	assert {"train/data", "train/eval", "train/checkpoint", "train/report"} <= regions.keys()
	with gzip.open(directory / "trace.json.gz", "rt", encoding="utf-8") as stream:
		trace = json.load(stream)
	names = {event.get("name", "") for event in trace["traceEvents"]}
	assert {name for name in names if name.startswith("ProfilerStep#")} == {"ProfilerStep#2", "ProfilerStep#3"}
	assert "aten::mm" in names
	assert {"train/qk_clip", "attention/qk_clip_stats"} <= names
	summary = (directory / "summary.txt").read_text(encoding="utf-8")
	assert "Self CPU" in summary and "Self CPU Mem" in summary
	if device == "cuda":
		assert "CUDA operators" in summary and metadata["cuda_timings_captured"]
	records = [json.loads(line) for line in (profiled_run.directory / "metrics.jsonl").read_text(
		encoding="utf-8").splitlines()]
	assert records[0]["profiling"]["active"] == 2
	assert records[-1]["session"]["profiling_seconds"] > 0


@pytest.mark.parametrize("steps,expected,active_attempts", [(1, "no_capture", 0), (3, "partial", 1)])
def test_short_training_reports_incomplete_window(
	tmp_path, prepared_data, kda_config_factory, steps, expected, active_attempts,
):
	config = TrainConfig(batch_size=1, seq_len=8, max_steps=steps, log_interval=1,
						 eval_interval=10, eval_batches=1)
	model = KimiK3(_architecture(kda_config_factory))
	loaders = [TokenDataLoader(ShardedTokenDataset(prepared_data, split), 1, 8)
			   for split in ("train", "val")]
	directory = tmp_path / "profile"
	state = train(model, build_optimizer(model, config), *loaders, config, "cpu",
				  profile_config=ProfileConfig(active=3), profile_dir=directory)
	metadata = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
	assert metadata["status"] == expected and metadata["stop_reason"] == "training_end"
	assert metadata["attempts_completed"] == steps
	assert metadata["active_attempts_completed"] == active_attempts
	assert metadata["end_state"]["step"] == state.step == steps
	assert (directory / "trace.json.gz").exists() == (expected == "partial")
	assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_cli_resume_creates_a_separate_profile_and_preserves_trajectory(
	tmp_path, prepared_data, kda_config_factory,
):
	architecture = _architecture(kda_config_factory)
	path = tmp_path / "model.yaml"
	path.write_text(yaml.safe_dump(asdict(architecture)), encoding="utf-8")
	run_dir = tmp_path / "run"

	def execute(steps, directory, *, profile=False, resume=None):
		args = [sys.executable, str(REPO_ROOT / "scripts/train.py"), "--data-dir", str(prepared_data),
				"--model-config", str(path), "--run-dir", str(directory), "--device", "cpu",
				"--batch-size", "1", "--seq-len", "8", "--gradient-accumulation-steps", "2",
				"--max-steps", str(steps), "--log-interval", "1", "--eval-batches", "1", "--deterministic"]
		if profile:
			args += ["--profile", "--profile-wait", "0", "--profile-active", "1"]
		if resume:
			args += ["--resume", str(resume)]
		result = subprocess.run(args, cwd=tmp_path, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
								capture_output=True, text=True, timeout=60, check=False)
		assert result.returncode == 0, result.stdout + result.stderr
		return torch.load(directory / "last.pt", weights_only=True)

	execute(3, run_dir, profile=True)
	first_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
	first_session = first_manifest["sessions"][0]["session_id"]
	first_dir = run_dir / "profiles" / first_session
	original = {name: (first_dir / name).read_bytes() for name in ("profile.json", "trace.json.gz", "summary.txt")}
	resumed = execute(5, run_dir, profile=True, resume=run_dir / "last.pt")
	continuous = execute(5, tmp_path / "continuous")
	_assert_equal(resumed, continuous)
	manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
	second_session = manifest["sessions"][1]["session_id"]
	assert first_session != second_session
	for name, data in original.items():
		assert (first_dir / name).read_bytes() == data
	second = json.loads((run_dir / "profiles" / second_session / "profile.json").read_text(encoding="utf-8"))
	assert second["status"] == "complete" and second["attempts_completed"] == 2
	assert second["start_state"]["step"] == 3 and second["end_state"]["step"] == 5
	assert second["run_id"] == manifest["run_id"] and second["session_id"] == second_session
