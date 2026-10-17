from types import SimpleNamespace

import pytest
import torch

from kimi_k3.training import monitoring
from kimi_k3.training.metrics import MetricsWriter
from kimi_k3.training.run import RunContext
from kimi_k3.training.trainer import TrainState


def test_ram_peak_is_explicitly_sampled_and_cpu_gpu_fields_are_null(tmp_path, monkeypatch):
	samples = iter((100, 150, 80))
	monkeypatch.setattr(monitoring.psutil, "Process", lambda: SimpleNamespace(
		memory_info=lambda: SimpleNamespace(rss=next(samples))))
	writer = MetricsWriter(RunContext(tmp_path, tmp_path / "last.pt", "run", "session"))
	monitor = monitoring.TrainingMonitor("cpu", writer)
	records = [monitor.sample_memory() for _ in range(3)]
	assert [record["ram_rss_bytes"] for record in records] == [100, 150, 80]
	assert [record["ram_rss_sampled_peak_bytes"] for record in records] == [100, 150, 150]
	assert all(value is None for record in records for key, value in record.items()
			   if key.startswith("cuda_"))


def test_session_rates_use_new_tokens_and_handle_no_training(monkeypatch):
	now = 10.0
	monkeypatch.setattr(monitoring, "perf_counter", lambda: now)
	state = TrainState(step=2, tokens_seen=16)
	monitor = monitoring.TrainingMonitor("cpu")
	monitor.begin(state)
	record = monitor.session_summary(state)
	assert record["tokens"] == 0
	assert record["tokens_per_second"] is None
	assert record["train_seconds"] == 0
	now += 2.0
	state.tokens_seen += 8
	record = monitor.session_summary(state)
	assert record["tokens"] == 8
	assert record["tokens_per_second"] == 4.0


def test_unavailable_process_inspection_still_records_metrics(tmp_path, monkeypatch):
	def unavailable():
		raise monitoring.psutil.NoSuchProcess(123)

	monkeypatch.setattr(monitoring.psutil, "Process", unavailable)
	writer = MetricsWriter(RunContext(tmp_path, tmp_path / "last.pt", "run", "session"))
	with writer:
		state = TrainState()
		monitor = monitoring.TrainingMonitor("cpu", writer)
		monitor.begin(state)
		monitor.report("session_start", state)
		memory = monitor.sample_memory()
	assert memory["ram_rss_bytes"] is None
	assert memory["ram_rss_sampled_peak_bytes"] is None
	assert writer.path.is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_peak_resets_per_session_and_captures_released_allocations(tmp_path):
	device = torch.device("cuda", torch.cuda.current_device())
	old = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
	old_peak = torch.cuda.max_memory_allocated(device)
	del old
	torch.cuda.synchronize(device)
	baseline = torch.cuda.memory_allocated(device)
	writer = MetricsWriter(RunContext(tmp_path, tmp_path / "last.pt", "run", "session"))
	monitor = monitoring.TrainingMonitor(device, writer)
	assert monitor.sample_memory()["cuda_peak_allocated_bytes"] == baseline
	temporary = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device=device)
	del temporary
	torch.cuda.synchronize(device)
	memory = monitor.sample_memory()
	assert memory["cuda_allocated_bytes"] == baseline
	assert baseline + 4 * 1024 * 1024 <= memory["cuda_peak_allocated_bytes"] < old_peak
	assert memory["cuda_reserved_bytes"] >= memory["cuda_allocated_bytes"]
	assert memory["cuda_peak_reserved_bytes"] >= memory["cuda_reserved_bytes"]
