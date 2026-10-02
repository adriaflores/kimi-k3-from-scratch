import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from src.kimi_k3.model import KimiK3
from src.kimi_k3.training import monitoring, trainer
from src.kimi_k3.training.metrics import MetricsWriter
from src.kimi_k3.training.run import start_run
from src.kimi_k3.training.trainer import TrainConfig, TrainState, train


class QuadraticLoss(nn.Module):
	"""Known loss/gradient values independently verify the logged aggregates."""

	def __init__(self):
		super().__init__()
		self.weight = nn.Parameter(torch.tensor(2.0))

	def forward(self, x, targets):
		return None, self.weight.square()


@pytest.fixture
def prepared_data(tmp_path):
	directory = tmp_path / "data"
	directory.mkdir()
	metadata = {"format_version": 2, "storage": {"format": "npy", "dtype": "uint16"},
				"tokenizer": {"vocab_size": 32, "eos_id": 0}, "splits": {}}
	for split in ("train", "val"):
		filename = f"{split}_000.npy"
		np.save(directory / filename, np.arange(65, dtype=np.uint16) % 32)
		metadata["splits"][split] = {"tokens": 65, "shards": [{"file": filename, "tokens": 65}]}
	(directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
	return directory


@pytest.mark.parametrize("max_grad_norm", [0.0, 3.0])
def test_interval_metrics_accumulation_clipping_and_final_partial_window(tmp_path, prepared_data, max_grad_norm):
	config = TrainConfig(batch_size=1, seq_len=8, learning_rate=0.1, max_steps=3, log_interval=2,
						 eval_interval=1, eval_batches=1, gradient_accumulation_steps=2,
						 max_grad_norm=max_grad_norm)
	loaders = [TokenDataLoader(ShardedTokenDataset(prepared_data, split), 1, 8)
			   for split in ("train", "val")]
	model = QuadraticLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
	state = TrainState()
	run = start_run(model=model, optimizer=optimizer, config=config, state=state,
					train_loader=loaders[0], val_loader=loaders[1], repo_root=tmp_path,
					runs_root=tmp_path / "runs")
	with MetricsWriter(run) as metrics:
		train(model, optimizer, *loaders, config, "cpu", state=state, metrics=metrics)
	records = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text(
		encoding="utf-8").splitlines()]
	assert [record["event"] for record in records] == [
		"session_start", "val", "train", "val", "train", "val", "session_end"]
	assert all(record["run_id"] == run.run_id and record["session_id"] == run.session_id
			   for record in records)
	windows = [record for record in records if record["event"] == "train"]
	assert [record["step"] for record in windows] == [2, 3]
	assert [record["tokens"] for record in windows] == [32, 16]
	assert [record["tokens_seen"] for record in windows] == [32, 48]
	assert [record["updates"] for record in windows] == [2, 1]
	assert [record["attempted_updates"] for record in windows] == [2, 1]
	# Two microbatches produce the average gradient, not twice that gradient.
	expected_weights = [2.0, 1.7, 1.4, 1.12] if max_grad_norm else [2.0, 1.6, 1.28, 1.024]
	expected_norms = [2 * weight for weight in expected_weights[:3]]
	assert windows[0]["train_loss"] == pytest.approx(
		sum(weight**2 for weight in expected_weights[:2]) / 2, abs=1e-5)
	assert windows[1]["train_loss"] == pytest.approx(expected_weights[2]**2, abs=1e-5)
	assert windows[0]["grad_norm_mean"] == pytest.approx(sum(expected_norms[:2]) / 2, abs=1e-5)
	assert windows[0]["grad_norm_max"] == expected_norms[0]
	assert windows[1]["grad_norm_mean"] == pytest.approx(expected_norms[2], abs=1e-5)
	assert [record["clipped_fraction"] for record in windows] == ([1.0, 0.0] if max_grad_norm else [0.0, 0.0])
	assert all(record["lr"] == 0.1 and record["skipped_updates_in_window"] == 0
			   and record["nonfinite_grad_norms"] == 0 for record in windows)
	validation = [record for record in records if record["event"] == "val"]
	assert [record["step"] for record in validation] == [1, 2, 3]
	assert [record["val_loss"] for record in validation] == pytest.approx(
		[weight**2 for weight in expected_weights[1:]], abs=1e-5)
	assert records[-1]["tokens_seen"] == state.tokens_seen == 48


def test_training_rate_excludes_eval_save_and_reporting_cost(tmp_path, prepared_data, monkeypatch):
	now = 0.0

	def advance(seconds):
		nonlocal now
		now += seconds

	monkeypatch.setattr(monitoring, "perf_counter", lambda: now)
	snapshot = trainer.MoEDiagnostics.snapshot

	def timed_snapshot(diagnostics):
		result = snapshot(diagnostics)
		advance(0.4)
		return result

	monkeypatch.setattr(trainer.MoEDiagnostics, "snapshot", timed_snapshot)
	def memory_info():
		advance(0.1)
		return SimpleNamespace(rss=128 * 1024 * 1024)

	monkeypatch.setattr(monitoring.psutil, "Process", lambda: SimpleNamespace(memory_info=memory_info))

	class TimedLoss(QuadraticLoss):
		def forward(self, x, targets):
			advance(1.0 if self.training else 3.0)
			return super().forward(x, targets)

	config = TrainConfig(batch_size=1, seq_len=8, learning_rate=0.1, max_steps=3,
						 log_interval=2, eval_interval=1, eval_batches=1, checkpoint_interval=1,
						 gradient_accumulation_steps=2)
	loaders = [TokenDataLoader(ShardedTokenDataset(prepared_data, split), 1, 8)
			   for split in ("train", "val")]
	next_batch = loaders[0].next_batch

	def timed_batch():
		advance(0.5)
		return next_batch()

	monkeypatch.setattr(loaders[0], "next_batch", timed_batch)
	save = trainer.save_checkpoint

	def timed_save(*args, **kwargs):
		save(*args, **kwargs)
		advance(2.0)

	monkeypatch.setattr(trainer, "save_checkpoint", timed_save)
	model = TimedLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	state = TrainState()
	run = start_run(model=model, optimizer=optimizer, config=config, state=state,
					train_loader=loaders[0], val_loader=loaders[1], repo_root=tmp_path,
					runs_root=tmp_path / "runs")
	with MetricsWriter(run) as metrics:
		write = metrics.write

		def timed_write(*args, **kwargs):
			write(*args, **kwargs)
			advance(0.25)

		monkeypatch.setattr(metrics, "write", timed_write)
		train(model, optimizer, *loaders, config, "cpu", state=state,
			  checkpoint_path=run.checkpoint_path, metrics=metrics)
	records = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text(
		encoding="utf-8").splitlines()]
	windows = [record for record in records if record["event"] == "train"]
	# Train includes batch retrieval and all accumulated forwards, while
	# intervening evaluation, checkpoint I/O and reporting belong to wall time.
	assert [record["train_seconds"] for record in windows] == pytest.approx([6.0, 3.0])
	assert [record["seconds_per_attempt"] for record in windows] == pytest.approx([3.0, 3.0])
	assert [record["train_tokens_per_second"] for record in windows] == pytest.approx([32 / 6, 16 / 3])
	assert [record["eval_seconds"] for record in records if record["event"] == "val"] == pytest.approx([3.0] * 3)
	assert [record["checkpoint_seconds"] for record in records if record["event"] == "checkpoint"] == pytest.approx([2.0] * 3)
	session = records[-1]["session"]
	assert session["train_seconds"] == pytest.approx(9.0)
	assert session["eval_seconds"] == pytest.approx(9.0)
	assert session["checkpoint_seconds"] == pytest.approx(6.0)
	report_seconds = 9 * 0.35 + 2 * 0.4  # Two MoE snapshots; before the final report itself.
	assert session["reporting_seconds"] == pytest.approx(report_seconds)
	assert session["elapsed_seconds"] == pytest.approx(9 + 9 + 6 + report_seconds)
	assert session["tokens"] == 48
	assert session["tokens_per_second"] == pytest.approx(48 / session["elapsed_seconds"])
	assert records[-1]["memory"]["ram_rss_bytes"] > 0


@pytest.mark.parametrize("device,precision,force_overflow", [
	("cpu", "fp32", False),
	pytest.param("cuda", "fp32", False, marks=pytest.mark.skipif(
		not torch.cuda.is_available(), reason="CUDA is unavailable")),
	pytest.param("cuda", "fp16", True, marks=pytest.mark.skipif(
		not torch.cuda.is_available(), reason="CUDA is unavailable")),
])
def test_moe_records_match_expert_dispatch_with_accumulation_and_intervening_eval(
	tmp_path, prepared_data, kda_config_factory, monkeypatch, device, precision, force_overflow,
):
	config = TrainConfig(batch_size=1, seq_len=8, max_steps=3, log_interval=2,
						 eval_interval=1, eval_batches=2, gradient_accumulation_steps=2,
						 precision=precision)
	architecture = kda_config_factory(num_layers=3, layer_types=("kda", "mla", "kda"),
									  num_dense_layers=1, attn_res_block_size=2,
									  num_experts=4, num_experts_per_token=2)
	model = KimiK3(architecture).to(device)
	first = True

	def overflow_once(gradient):
		nonlocal first
		if first:
			first = False
			return torch.full_like(gradient, float("inf"))
		return gradient

	if force_overflow:
		model.embed_tokens.weight.register_hook(overflow_once)
	optimizer = trainer.build_optimizer(model, config)
	loaders = [TokenDataLoader(ShardedTokenDataset(prepared_data, split), 1, 8)
			   for split in ("train", "val")]
	state = TrainState()
	run = start_run(model=model, optimizer=optimizer, config=config, state=state,
					train_loader=loaders[0], val_loader=loaders[1], repo_root=tmp_path,
					runs_root=tmp_path / "runs")
	layers = {f"layers.{i}.ffn": model.layers[i].ffn for i in (1, 2)}
	expected_counts = {name: [0] * 4 for name in layers}
	handles = []

	def observe(name, expert_index):
		def count(module, args):
			if model.training:
				expected_counts[name][expert_index] += args[0].shape[0]
		return count

	for name, layer in layers.items():
		for index, expert in enumerate(layer.experts):
			handles.append(expert.register_forward_pre_hook(observe(name, index)))
	truth = []
	try:
		with MetricsWriter(run) as metrics:
			write = metrics.write

			def capture(event, state, **values):
				if event == "train":
					truth.append({name: {
						"counts": expected_counts[name].copy(),
						"biases": layer.quantile_balancer.biases.detach().cpu().tolist(),
					} for name, layer in layers.items()})
					for counts in expected_counts.values():
						counts[:] = [0] * 4
				write(event, state, **values)

			monkeypatch.setattr(metrics, "write", capture)
			train(model, optimizer, *loaders, config, device, state=state,
				  checkpoint_path=run.checkpoint_path, metrics=metrics)
	finally:
		for handle in handles:
			handle.remove()
	records = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text(
		encoding="utf-8").splitlines()]
	windows = [record for record in records if record["event"] == "train"]
	assert [record["step"] for record in windows] == [2, 3]
	assert sum(record["tokens"] for record in windows) == state.tokens_seen
	if force_overflow:
		assert state.skipped_updates >= 1
		assert any(record["skipped_updates_in_window"] for record in windows)
	else:
		assert [record["tokens"] for record in windows] == [32, 16]
	for record, reference in zip(windows, truth, strict=True):
		assert record["moe"].keys() == layers.keys()
		for name, layer_record in record["moe"].items():
			counts = reference[name]["counts"]
			assert layer_record["tokens"] == record["tokens"]
			assert layer_record["expert_counts"] == counts
			assert sum(counts) == record["tokens"] * 2
			assert layer_record["max_load_over_mean"] == max(counts) / (sum(counts) / 4)
			assert layer_record["zero_load_experts"] == counts.count(0)
			assert layer_record["qb_biases"] == reference[name]["biases"]
	assert all("moe" not in record for record in records if record["event"] != "train")
	assert all(layer._expert_load_counts is None for layer in layers.values())
	checkpoint = torch.load(run.checkpoint_path, weights_only=True)
	assert not any("expert_load_counts" in name for name in checkpoint["model"])
