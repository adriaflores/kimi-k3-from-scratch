import json

import pytest

from src.kimi_k3.training.metrics import MetricsWriter, TrainingWindow
from src.kimi_k3.training.run import RunContext
from src.kimi_k3.training.trainer import TrainState


@pytest.mark.parametrize("tail", [b"", b'{"interrupted":' + b"x" * 5000])
def test_metrics_flush_and_resume_preserve_complete_history(tmp_path, tail):
	run = RunContext(tmp_path, tmp_path / "last.pt", "run", "first")
	state = TrainState(step=2, tokens_seen=16)
	path = tmp_path / "metrics.jsonl"
	with MetricsWriter(run) as metrics:
		metrics.write("train", state, train_loss=1.25)
		# The log is readable before closing the writer.
		record = json.loads(path.read_text(encoding="utf-8"))
		assert record["run_id"] == "run"
		assert record["session_id"] == "first"
		assert record["step"] == 2
		assert record["tokens_seen"] == 16
	original = path.read_bytes()
	path.write_bytes(original + tail)
	resumed = RunContext(tmp_path, tmp_path / "last.pt", "run", "second")
	with MetricsWriter(resumed) as metrics:
		assert metrics.discarded_tail_bytes == len(tail)
		assert path.read_bytes() == original
		metrics.write("session_start", state, discarded_tail_bytes=metrics.discarded_tail_bytes)
	assert path.read_bytes().startswith(original)
	records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
	assert [record["session_id"] for record in records] == ["first", "second"]


def test_interrupted_first_record_and_invalid_values_do_not_corrupt_jsonl(tmp_path):
	path = tmp_path / "metrics.jsonl"
	path.write_bytes(b'{"unfinished":')
	run = RunContext(tmp_path, tmp_path / "last.pt", "run", "session")
	with MetricsWriter(run) as metrics:
		assert metrics.discarded_tail_bytes == len(b'{"unfinished":')
		metrics.write("session_start", TrainState())
		before = path.read_bytes()
		with pytest.raises(ValueError, match="JSON compliant"):
			metrics.write("train", TrainState(), train_loss=float("nan"))
		assert path.read_bytes() == before
	assert json.loads(path.read_text(encoding="utf-8"))["event"] == "session_start"


def test_window_counts_skipped_tokens_but_uses_successful_preclip_gradients():
	window = TrainingWindow()
	window.add(4.0, 4, float("inf"), skipped=True, max_grad_norm=0.5)
	window.add(1.0, 8, 0.2, skipped=False, max_grad_norm=0.5)
	window.add(2.0, 8, 0.6, skipped=False, max_grad_norm=0.5)
	record = window.summary(0.001)
	assert record["train_loss"] == 2.0  # Token-weighted; includes the skipped attempt.
	assert record["tokens"] == 20
	assert record["attempted_updates"] == 3
	assert record["updates"] == 2
	assert record["skipped_updates_in_window"] == 1
	assert record["grad_norm_mean"] == pytest.approx(0.4)
	assert record["grad_norm_max"] == 0.6
	assert record["clipped_fraction"] == 0.5
	assert record["nonfinite_grad_norms"] == 1
	json.dumps(record, allow_nan=False)
	only_skips = TrainingWindow()
	only_skips.add(1.0, 8, float("nan"), skipped=True, max_grad_norm=0.5)
	record = only_skips.summary(0.001)
	assert record["grad_norm_mean"] is record["grad_norm_max"] is record["clipped_fraction"] is None
	json.dumps(record, allow_nan=False)
