import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.kimi_k3.training.profiling import (
	ProfileConfig,
	TrainingProfiler,
	_compress_trace,
)
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
	assert not (directory / "trace.json").exists()


def test_profile_requires_a_destination_without_metrics():
	with pytest.raises(ValueError, match="profile_dir"):
		TrainingProfiler(nn.Linear(2, 2), "cpu", TrainState(), ProfileConfig())


def _limit_trace_reads(monkeypatch, raw, *, fail_after=None):
	"""Simulate a reader that cannot allocate fresh large bytes objects."""
	original_open = Path.open
	buffers = []

	class ConstrainedReader:
		def __init__(self, source):
			self.source = source
		def __enter__(self):
			self.source.__enter__()
			return self
		def __exit__(self, *args):
			return self.source.__exit__(*args)
		def read(self, _size=-1):
			raise MemoryError("Cannot allocate a new trace read buffer")
		def readinto(self, buffer):
			if len(buffer) > 64 * 1024:
				raise MemoryError("Trace buffer exceeds the available memory")
			if fail_after is not None and len(buffers) >= fail_after:
				raise MemoryError("Trace read failed after compression started")
			buffers.append(id(buffer))
			return self.source.readinto(buffer)

	def constrained_open(path, mode="r", *args, **kwargs):
		source = original_open(path, mode, *args, **kwargs)
		return ConstrainedReader(source) if path == raw and mode == "rb" else source

	monkeypatch.setattr(Path, "open", constrained_open)
	return buffers


def test_trace_compression_reuses_a_small_buffer_under_read_memory_pressure(tmp_path, monkeypatch):
	raw, compressed = tmp_path / "trace.json", tmp_path / "trace.json.gz"
	# Multiple full blocks and a short last block catch truncation and stale bytes.
	payload = bytes(range(256)) * 8192 + b"short final block"
	raw.write_bytes(payload)
	buffers = _limit_trace_reads(monkeypatch, raw)
	_compress_trace(raw, compressed)
	with gzip.open(compressed, "rb") as source:
		assert source.read() == payload
	assert len(buffers) > 2 and len(set(buffers)) == 1
	assert not compressed.with_name(compressed.name + ".tmp").exists()


def test_failed_trace_compression_preserves_raw_and_previous_archive(tmp_path, monkeypatch):
	raw, compressed = tmp_path / "trace.json", tmp_path / "trace.json.gz"
	payload = b"trace data" * 20000
	raw.write_bytes(payload)
	previous = gzip.compress(b"previous complete trace")
	compressed.write_bytes(previous)
	with monkeypatch.context() as limited:
		_limit_trace_reads(limited, raw, fail_after=1)
		with pytest.raises(MemoryError, match="after compression started"):
			_compress_trace(raw, compressed)
	assert raw.read_bytes() == payload and compressed.read_bytes() == previous
	assert not compressed.with_name(compressed.name + ".tmp").exists()
