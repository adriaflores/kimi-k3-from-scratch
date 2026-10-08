"""Active QK-Clip in real training, skip handling and checkpoint policy."""

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from src.kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from src.kimi_k3.model import KimiK3
from src.kimi_k3.training.metrics import MetricsWriter
from src.kimi_k3.training.qk_clip import QKClip
from src.kimi_k3.training.run import start_run
from src.kimi_k3.training.trainer import (
	TrainConfig,
	TrainState,
	build_optimizer,
	load_checkpoint,
	train,
)


@pytest.fixture
def prepared_data(tmp_path):
	directory = tmp_path / "data"
	directory.mkdir()
	metadata = {"format_version": 2, "storage": {"format": "npy", "dtype": "uint16"},
		"tokenizer": {"vocab_size": 32, "eos_id": 0}, "splits": {}}
	for split in ("train", "val"):
		filename = split + "_000.npy"
		np.save(directory / filename, (np.arange(129) * 7 % 32).astype(np.uint16))
		metadata["splits"][split] = {"tokens": 129, "shards": [{"file": filename, "tokens": 129}]}
	(directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
	return directory


def _loaders(directory, config):
	return tuple(TokenDataLoader(ShardedTokenDataset(directory, split), config.batch_size, config.seq_len)
		for split in ("train", "val"))


def _model(kda_config_factory, seed=123):
	torch.manual_seed(seed)
	model = KimiK3(kda_config_factory(num_layers=2, layer_types=("kda", "mla"), num_dense_layers=2,
		mla_num_heads=3, mla_q_lora_rank=5, mla_kv_lora_rank=4,
		mla_qk_nope_head_dim=3, mla_qk_shared_head_dim=2, mla_v_head_dim=5))
	# Trigger the production tau=100, with only the first MLA head amplified.
	with torch.no_grad():
		mla = model.layers[1].attn
		mla.W_UQ.weight[:mla.d_head].mul_(1000)
		mla.W_UKV.weight[:mla.d_head].mul_(1000)
	return model


def _config(**kwargs):
	return TrainConfig(batch_size=1, seq_len=8, max_steps=4, log_interval=1,
		eval_interval=4, eval_batches=1, checkpoint_interval=2,
		gradient_accumulation_steps=2, learning_rate=1e-3, weight_decay=0.1,
		max_grad_norm=0.1, qk_clip_chunk_size=3, **kwargs)


def _equal(actual, expected):
	if isinstance(actual, torch.Tensor):
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	elif isinstance(actual, dict):
		assert actual.keys() == expected.keys()
		for key in actual:
			_equal(actual[key], expected[key])
	elif isinstance(actual, (list, tuple)):
		assert type(actual) is type(expected)
		for a, b in zip(actual, expected, strict=True):
			_equal(a, b)
	else:
		assert actual == expected


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_active_default_threshold_clipping_resumes_exactly(prepared_data, tmp_path, kda_config_factory, precision):
	config = _config(precision=precision, lr_schedule="cosine", warmup_steps=1, decay_steps=4)

	def execute(seed, requested, directory, resume=None):
		model = _model(kda_config_factory, seed)
		optimizer = build_optimizer(model, requested)
		loaders = _loaders(prepared_data, requested)
		state = (load_checkpoint(resume, model, optimizer, requested, vocab_size=32,
			train_loader=loaders[0], val_loader=loaders[1]) if resume else TrainState())
		run = start_run(model=model, optimizer=optimizer, config=requested, state=state,
			train_loader=loaders[0], val_loader=loaders[1], repo_root=tmp_path, run_dir=directory)
		with MetricsWriter(run) as metrics:
			train(model, optimizer, *loaders, requested, "cpu", state=state,
				checkpoint_path=run.checkpoint_path, metrics=metrics)
		assert all(not hasattr(module, "_qk_clip_observer") for module in model.modules())
		return torch.load(run.checkpoint_path, weights_only=True), run.checkpoint_path

	continuous, _ = execute(123, replace(config), tmp_path / "continuous")
	_, partial = execute(123, replace(config, max_steps=2), tmp_path / "partial")
	resumed, _ = execute(999, replace(config), tmp_path / "resumed", partial)
	_equal(resumed, continuous)
	assert resumed["format_version"] == 7
	assert resumed["qk_clip"]["threshold"] == 100.0
	records = [json.loads(line) for line in (tmp_path / "continuous/metrics.jsonl").read_text().splitlines()]
	updates = [record for record in records if record["event"] == "train"]
	assert sum(record["qk_clip"]["clipped_head_updates"] for record in updates) > 0
	assert updates[0]["qk_clip"]["max_logit"] > 100
	assert all(record["qk_clip"]["observed_head_updates"] == 3 for record in updates)


@pytest.mark.parametrize("case", ["format6", "missing_policy", "policy", "head_geometry", "threshold", "chunk_size"])
def test_clip_checkpoint_rejection_is_atomic(prepared_data, tmp_path, kda_config_factory, case):
	config = replace(_config(), max_steps=1)
	model = _model(kda_config_factory)
	path = tmp_path / "checkpoint.pt"
	train(model, build_optimizer(model, config), *_loaders(prepared_data, config), config, "cpu", checkpoint_path=path)
	saved = torch.load(path, weights_only=True)
	if case == "format6":
		saved["format_version"] = 6
		saved.pop("qk_clip")
		for field in ("qk_clip_threshold", "qk_clip_chunk_size"):
			saved["train_config"].pop(field)
	elif case == "missing_policy":
		saved.pop("qk_clip")
	elif case == "policy":
		saved["qk_clip"]["score"] = "unscaled_qk"
	elif case == "head_geometry":
		saved["qk_clip"]["layers"][0]["shared_dim"] += 1
	else:
		config = replace(config, **{"qk_clip_" + case: 80.0 if case == "threshold" else 4})
	torch.save(saved, path)
	target = _model(kda_config_factory, seed=999)
	optimizer = build_optimizer(target, config)
	loaders = _loaders(prepared_data, config)
	before = copy.deepcopy(target.state_dict())
	config_before, rng = asdict(config), torch.get_rng_state().clone()
	with pytest.raises(ValueError, match="QK-Clip|qk_clip"):
		load_checkpoint(path, target, optimizer, config, vocab_size=32,
			train_loader=loaders[0], val_loader=loaders[1])
	_equal(target.state_dict(), before)
	assert not optimizer.state and asdict(config) == config_before
	assert all(loader.position == 0 for loader in loaders)
	assert torch.equal(torch.get_rng_state(), rng)


def test_skipped_attempt_does_not_clip_or_leak_scores_into_next_attempt(
	prepared_data, tmp_path, kda_config_factory, monkeypatch,
):
	# Exercise the trainer's skip branch on CPU. The existing real CUDA
	# GradScaler integration also forces clipping and checks frozen weights.
	model = _model(kda_config_factory)
	initial = copy.deepcopy(model.state_dict())
	config = replace(_config(), max_steps=1)
	observations = 0

	def synthetic_stats(name, query, key):
		nonlocal observations
		observations += 1
		maximum = 200.0 if observations <= config.gradient_accumulation_steps else 10.0
		observer.max_logits[name] = torch.full((3,), maximum)

	observer = None
	enter = QKClip.__enter__
	def capture_observer(self):
		nonlocal observer
		observer = self
		return enter(self)
	monkeypatch.setattr(QKClip, "__enter__", capture_observer)
	monkeypatch.setattr(QKClip, "observe", lambda self, name, q, k: synthetic_stats(name, q, k))

	class OneSkippedAttempt:
		def __init__(self, *_args, **_kwargs):
			self.value, self.attempt = 64.0, 0
		def is_enabled(self):
			return True
		def scale(self, loss):
			return loss
		def unscale_(self, optimizer):
			pass
		def get_scale(self):
			return self.value
		def step(self, optimizer):
			_equal(model.state_dict(), initial)
			assert not optimizer.state
			if self.attempt:
				optimizer.step()
		def update(self):
			if not self.attempt:
				self.value /= 2
			self.attempt += 1
		def state_dict(self):
			return {"scale": self.value}
	monkeypatch.setattr(torch.amp, "GradScaler", OneSkippedAttempt)
	optimizer = build_optimizer(model, config)
	loaders = _loaders(prepared_data, config)
	run = start_run(model=model, optimizer=optimizer, config=config, state=TrainState(),
		train_loader=loaders[0], val_loader=loaders[1], repo_root=tmp_path, run_dir=tmp_path / "skip")
	with MetricsWriter(run) as metrics:
		state = train(model, optimizer, *loaders, config, "cpu", metrics=metrics)
	assert state.step == state.skipped_updates == 1
	assert observations == 4
	records = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().splitlines()]
	update = next(record for record in records if record["event"] == "train")
	assert update["qk_clip"]["max_logit"] == 10.0
	assert update["qk_clip"]["clipped_head_updates"] == 0
	assert not observer.max_logits


def test_nonfinite_attention_stat_fails_before_optimizer_changes(
	prepared_data, kda_config_factory, monkeypatch,
):
	model = _model(kda_config_factory)
	before = copy.deepcopy(model.state_dict())
	config = replace(_config(), max_steps=1)
	optimizer = build_optimizer(model, config)
	def invalid_stats(self, name, query, key):
		self.max_logits[name] = torch.full((3,), float("nan"))
	monkeypatch.setattr(QKClip, "observe", invalid_stats)
	with pytest.raises(FloatingPointError, match="max logit"):
		train(model, optimizer, *_loaders(prepared_data, config), config, "cpu")
	_equal(model.state_dict(), before)
	assert not optimizer.state
	assert all(not hasattr(module, "_qk_clip_observer") for module in model.modules())
