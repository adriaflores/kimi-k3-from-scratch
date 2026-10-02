import gzip
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.kimi_k3.training.profiling import ProfileConfig, TrainingProfiler
from src.kimi_k3.training.trainer import TrainState


@pytest.mark.parametrize("values", [
	{"wait": -1}, {"warmup": 0}, {"active": 0}, {"active": True}, {"wait": 1.5},
])
def test_invalid_profile_window_is_rejected(values):
	with pytest.raises(ValueError, match="profile"):
		ProfileConfig(**values)


def test_disabled_mode_uses_no_profiler_ranges_or_hooks(monkeypatch, tmp_path):
	def unexpected(*args, **kwargs):
		raise AssertionError("Disabled profiling must not initialize recording")

	monkeypatch.setattr(torch.profiler, "profile", unexpected)
	monkeypatch.setattr(torch.profiler, "record_function", unexpected)
	model = nn.Linear(4, 4)
	with TrainingProfiler(model, "cpu", TrainState(), directory=tmp_path / "unused") as profiling:
		with profiling.region("train/forward"):
			model(torch.ones(1, 4)).sum().backward()
		profiling.step()
		assert not profiling.running and profiling.describe() is None
		assert not model._forward_hooks and not model._forward_pre_hooks
	assert not (tmp_path / "unused").exists()


def test_partial_trace_and_hooks_are_closed_when_forward_raises(tmp_path):
	class ExplodingAttention(nn.Module):
		def forward(self, x):
			raise RuntimeError("forward failed")

	model = nn.Module()
	attention = ExplodingAttention()
	model.layers = [SimpleNamespace(attn=attention)]
	directory = tmp_path / "profile"
	with pytest.raises(RuntimeError, match="forward failed"), TrainingProfiler(
		model, "cpu", TrainState(), ProfileConfig(wait=0, active=2), directory=directory,
	) as profiling:
		with profiling.region("train/forward"):
			torch.ones(2, 2).square()
		profiling.step()  # Warmup done; the next attempt is recorded.
		with profiling.region("train/forward"):
			attention(torch.ones(2, 2))
	assert not attention._forward_hooks and not attention._forward_pre_hooks
	metadata = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
	assert metadata["status"] == "partial" and metadata["stop_reason"] == "training_error"
	assert metadata["attempts_completed"] == 1 and metadata["active_attempts_completed"] == 0
	with gzip.open(directory / "trace.json.gz", "rt", encoding="utf-8") as stream:
		trace = json.load(stream)
	names = {event.get("name") for event in trace["traceEvents"]}
	assert "model/ExplodingAttention" in names and "train/forward" in names


def test_profile_requires_a_destination_without_metrics():
	with pytest.raises(ValueError, match="profile_dir"):
		TrainingProfiler(nn.Linear(2, 2), "cpu", TrainState(), ProfileConfig())
