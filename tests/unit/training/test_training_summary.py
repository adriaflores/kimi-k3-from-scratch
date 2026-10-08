"""Acceptance summaries use session-local, token-weighted measurements."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.summarize_training_run import summarize_run

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def recorded_run(tmp_path):
	config = {"batch_size": 1, "seq_len": 128, "gradient_accumulation_steps": 2, "qk_clip_threshold": 100.0}
	session = {"session_id": "latest", "train_config": config,
		"start_state": {"step": 0, "tokens_seen": 0, "skipped_updates": 0},
		"environment": {"hardware": {"device": "cpu"}},
		"optimizer": {"param_groups": [{"algorithm": "adamw"}, {"algorithm": "muon"}]}}
	manifest = {"schema_version": 1, "run_id": "run", "sessions": [{"session_id": "old"}, session]}
	(tmp_path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
	old = {"event": "train", "session_id": "old", "run_id": "run", "train_loss": 1000}
	records = [old]
	for step, seconds, loss, clipped in ((2, 1.0, 10.0, 0), (4, 3.0, 8.0, 1)):
		records.append({"event": "train", "session_id": "latest", "run_id": "run",
			"step": step, "tokens_seen": step * 256, "skipped_updates": 0,
			"updates": 2, "attempted_updates": 2, "skipped_updates_in_window": 0,
			"nonfinite_grad_norms": 0, "tokens": 512, "train_seconds": seconds,
			"train_loss": loss, "lr": 3e-4, "grad_norm_mean": 1.0, "grad_norm_max": 2.0,
			"clipped_fraction": 0.5 if clipped else 0.0,
			"qk_clip": {"threshold": 100.0, "max_logit": 200.0 if clipped else 20.0,
				"clipped_head_updates": clipped, "observed_head_updates": 4},
			"memory": {"ram_rss_sampled_peak_bytes": 1000 * step},
			"session": {"tokens_per_second": 128.0, "profiling_seconds": 0.0}})
	records.append({"event": "session_end", "session_id": "latest", "run_id": "run",
		"step": 4, "tokens_seen": 1024, "skipped_updates": 0,
		"memory": {"ram_rss_sampled_peak_bytes": 3000},
		"session": {"tokens_per_second": 128.0, "profiling_seconds": 0.0}})
	_write_records(tmp_path, records)
	return tmp_path, records


def _write_records(path, records):
	(path / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_summary_weights_rates_by_time_and_uses_only_latest_session(recorded_run):
	path, _ = recorded_run
	result = summarize_run(path, expected_step=4)
	assert result["completed"] and result["logged_updates"] == 4
	assert result["logged_tokens"] == 1024 and result["steady_train_tokens_per_second"] == 256
	assert result["session_wall_tokens_per_second"] == 128
	assert result["first_logged_train_loss"] == 10 and result["last_logged_train_loss"] == 8
	assert result["gradient_clipped_fraction"] == 0.25
	assert result["qk_clip"]["clipped_fraction"] == 1 / 8
	assert result["qk_clip"]["max_logit"] == 200
	assert result["memory"]["ram_rss_sampled_peak_bytes"] == 4000
	assert result["memory"]["cuda_peak_allocated_bytes"] is None
	assert result["optimizer_algorithms"] == ["adamw", "muon"]


def test_warmup_exclusion_discards_the_entire_straddling_window(recorded_run):
	path, _ = recorded_run
	result = summarize_run(path, after_step=1)
	assert result["steady_logged_tokens"] == 512
	assert result["steady_train_tokens_per_second"] == pytest.approx(512 / 3)
	assert summarize_run(path, after_step=4)["steady_train_tokens_per_second"] is None


@pytest.mark.parametrize("case", ["loss", "grad_norm", "threshold", "head_counts", "tokens", "step", "identity", "end_counts"])
def test_inconsistent_or_nonfinite_runs_are_rejected(recorded_run, case):
	path, records = recorded_run
	if case == "threshold":
		records[1]["qk_clip"]["threshold"] = 80
	elif case == "head_counts":
		records[1]["qk_clip"]["clipped_head_updates"] = 5
	elif case == "end_counts":
		records[-1]["tokens_seen"] += 1
	else:
		field, value = {"loss": ("train_loss", float("nan")), "grad_norm": ("grad_norm_mean", float("inf")),
			"tokens": ("tokens", 1), "step": ("step", 99), "identity": ("run_id", "wrong")}[case]
		records[1][field] = value
	_write_records(path, records)
	with pytest.raises(ValueError):
		summarize_run(path)


def test_expected_completion_and_uncommitted_tail_are_checked(recorded_run):
	path, records = recorded_run
	_write_records(path, records[:-1])
	assert not summarize_run(path)["completed"]
	with pytest.raises(ValueError, match="normal completion"):
		summarize_run(path, expected_step=4)
	with (path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
		stream.write('{"interrupted":')
	with pytest.raises(ValueError, match="Uncommitted"):
		summarize_run(path)


def test_summary_cli_runs_without_torch_or_repository_working_directory(recorded_run):
	path, _ = recorded_run
	result = subprocess.run([sys.executable, "-S", str(REPO_ROOT / "scripts/summarize_training_run.py"),
		str(path), "--expect-step", "4"], cwd=path, capture_output=True, text=True, timeout=10, check=False)
	assert result.returncode == 0, result.stderr
	assert json.loads(result.stdout)["logged_tokens"] == 1024
