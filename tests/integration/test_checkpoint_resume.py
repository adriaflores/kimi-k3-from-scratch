import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from src.kimi_k3.model import KimiK3
from src.kimi_k3.training.trainer import TrainConfig, evaluate, load_checkpoint, train

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICES = ["cpu", pytest.param(
	"cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))]


def _seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def deterministic_execution(monkeypatch):
	# Set before a CUDA model performs its first matrix multiplication.
	monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
	enabled = torch.are_deterministic_algorithms_enabled()
	warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
	threads = torch.get_num_threads()
	torch.set_num_threads(1)
	torch.use_deterministic_algorithms(True)
	yield
	torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
	torch.set_num_threads(threads)


def _prepare_data(directory, offset=0):
	directory.mkdir()
	metadata = {
		"format_version": 2,
		"storage": {"format": "npy", "dtype": "uint16"},
		"tokenizer": {"vocab_size": 32, "eos_id": 0},
		"splits": {},
	}
	# A batch has 16 inputs plus lookahead: cross a shard, then wrap at exhaustion.
	for split, lengths in (("train", (17, 20)), ("val", (33,))):
		stream = ((np.arange(sum(lengths)) * 7 + offset) % 32).astype(np.uint16)
		entries = []
		position = 0
		for index, length in enumerate(lengths):
			filename = f"{split}_{index:03}.npy"
			path = directory / filename
			np.save(path, stream[position:position + length])
			entries.append({"file": filename, "tokens": length,
							"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
			position += length
		metadata["splits"][split] = {"tokens": len(stream), "shards": entries}
	(directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
	return directory


@pytest.fixture
def prepared_data(tmp_path):
	return _prepare_data(tmp_path / "data")


def _loaders(directory, config):
	return tuple(TokenDataLoader(ShardedTokenDataset(directory, split),
								 config.batch_size, config.seq_len)
				 for split in ("train", "val"))


def _train_config(**overrides):
	return TrainConfig(batch_size=2, seq_len=8, learning_rate=1e-3, max_steps=5,
					   log_interval=1, eval_interval=3, eval_batches=2,
					   checkpoint_interval=2, **overrides)


def _assert_equal(actual, expected):
	if isinstance(actual, torch.Tensor):
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	elif isinstance(actual, dict):
		assert actual.keys() == expected.keys()
		for key in actual:
			_assert_equal(actual[key], expected[key])
	elif isinstance(actual, (list, tuple)):
		assert type(actual) is type(expected)
		assert len(actual) == len(expected)
		for left, right in zip(actual, expected):
			_assert_equal(left, right)
	else:
		assert actual == expected


def _record_training_losses(model):
	losses = []

	def record(module, args, output):
		if module.training:
			losses.append(output[1].item())

	model.register_forward_hook(record)
	return losses


def _next_random_values(device="cpu"):
	return {"python": [random.random(), random.gauss(0, 1)],
			"numpy": [np.random.random(), np.random.normal()],
			"torch": torch.rand(4),
			"cuda": torch.rand(4, device=device).cpu() if device == "cuda" else None}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("overfit,stop_step", [(False, 1), (False, 2), (True, 2)])
def test_kimi_resume_matches_continuous_updates(prepared_data, tmp_path, kda_config_factory,
											  overfit, stop_step, device):
	architecture = kda_config_factory(num_layers=4, layer_types=("kda", "kda", "kda", "mla"),
									 num_dense_layers=1, attn_res_block_size=2,
									 num_experts=4, num_experts_per_token=2)

	def construct(seed, max_steps):
		_seed(seed)
		model = KimiK3(architecture).to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
		config = replace(_train_config(overfit_batch=overfit), max_steps=max_steps)
		loaders = _loaders(prepared_data, config)
		# Catch a resume implementation that assumes the fixed batch always starts at zero.
		if overfit:
			loaders[0].reset(3)
		return model, optimizer, config, loaders

	full_model, full_optimizer, full_config, full_loaders = construct(123, 5)
	full_losses = _record_training_losses(full_model)
	full_state = train(full_model, full_optimizer, *full_loaders, full_config, device)
	expected_random = _next_random_values(device)

	partial_model, partial_optimizer, partial_config, partial_loaders = construct(123, stop_step)
	checkpoint = tmp_path / "interrupted.pt"
	train(partial_model, partial_optimizer, *partial_loaders, partial_config, device,
		  checkpoint_path=checkpoint)
	resumed_model, resumed_optimizer, resumed_config, resumed_loaders = construct(999, 5)
	state = load_checkpoint(checkpoint, resumed_model, resumed_optimizer, resumed_config,
							vocab_size=32, train_loader=resumed_loaders[0], val_loader=resumed_loaders[1])
	resumed_losses = _record_training_losses(resumed_model)
	resumed_state = train(resumed_model, resumed_optimizer, *resumed_loaders,
						  resumed_config, device, state=state)

	assert resumed_losses == full_losses[stop_step:]
	assert asdict(resumed_state) == asdict(full_state)
	_assert_equal(resumed_model.state_dict(), full_model.state_dict())
	_assert_equal(resumed_optimizer.state_dict(), full_optimizer.state_dict())
	for actual, expected in zip(resumed_loaders, full_loaders):
		assert actual.state_dict() == expected.state_dict()
	_assert_equal(_next_random_values(device), expected_random)


class StochasticLM(nn.Module):
	"""Exercise all global RNGs; KimiK3 currently has no stochastic training layers."""

	def __init__(self):
		super().__init__()
		self.config = {"vocab_size": 32, "hidden_size": 8, "dropout": 0.4}
		self.embedding = nn.Embedding(32, 8)
		self.dropout = nn.Dropout(0.4)
		self.head = nn.Linear(8, 32)

	def forward(self, x, targets=None):
		# Draw even in eval to catch extra end-of-segment validation consuming RNG.
		scale = 1 + 0.1 * random.gauss(0, 1) + 0.1 * np.random.normal()
		logits = self.head(self.dropout(self.embedding(x)) * scale)
		loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
		return logits, loss


@pytest.mark.parametrize("device", DEVICES)
def test_resume_restores_rng_used_by_training(prepared_data, tmp_path, device):
	def construct(seed, max_steps):
		_seed(seed)
		# Include cached Gaussian values in Python and NumPy RNG state.
		random.gauss(0, 1)
		np.random.normal()
		model = StochasticLM().to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
		config = replace(_train_config(), max_steps=max_steps)
		return model, optimizer, config, _loaders(prepared_data, config)

	full_model, full_optimizer, full_config, full_loaders = construct(123, 5)
	full_losses = _record_training_losses(full_model)
	train(full_model, full_optimizer, *full_loaders, full_config, device)
	expected_random = _next_random_values(device)

	partial_model, partial_optimizer, partial_config, partial_loaders = construct(123, 2)
	checkpoint = tmp_path / "rng.pt"
	train(partial_model, partial_optimizer, *partial_loaders, partial_config, device,
		  checkpoint_path=checkpoint)
	resumed_model, resumed_optimizer, resumed_config, resumed_loaders = construct(999, 5)
	state = load_checkpoint(checkpoint, resumed_model, resumed_optimizer, resumed_config,
							vocab_size=32, train_loader=resumed_loaders[0], val_loader=resumed_loaders[1])
	resumed_losses = _record_training_losses(resumed_model)
	train(resumed_model, resumed_optimizer, *resumed_loaders, resumed_config, device, state=state)

	assert resumed_losses == full_losses[2:]
	_assert_equal(resumed_model.state_dict(), full_model.state_dict())
	_assert_equal(resumed_optimizer.state_dict(), full_optimizer.state_dict())
	_assert_equal(_next_random_values(device), expected_random)


def test_evaluation_preserves_model_and_rng_on_success_and_failure(prepared_data):
	_seed(123)
	model = StochasticLM()
	loader = _loaders(prepared_data, _train_config())[1]
	before = copy.deepcopy(model.state_dict())
	expected_random = _next_random_values()
	_seed(123)
	model = StochasticLM()
	evaluate(model, loader, "cpu", max_batches=2)
	assert model.training
	_assert_equal(model.state_dict(), before)
	_assert_equal(_next_random_values(), expected_random)

	_seed(123)
	model = StochasticLM()
	model.eval()
	loader.reset()
	# Force a failure after forward has drawn random numbers.
	with torch.no_grad():
		model.head.weight.fill_(float("nan"))
	expected_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
	with pytest.raises(FloatingPointError, match="validation loss"):
		evaluate(model, loader, "cpu", max_batches=1)
	assert not model.training
	assert random.getstate() == expected_state[0]
	np.testing.assert_equal(np.random.get_state(), expected_state[1])
	torch.testing.assert_close(torch.get_rng_state(), expected_state[2], rtol=0, atol=0)


def test_loader_resume_accepts_relocated_manifest_with_different_json_formatting(prepared_data, tmp_path):
	config = _train_config()
	original = _loaders(prepared_data, config)[0]
	original.next_batch()
	state = original.state_dict()
	relocated = tmp_path / "relocated"
	shutil.copytree(prepared_data, relocated)
	metadata_path = relocated / "metadata.json"
	metadata_path.write_text(json.dumps(json.loads(metadata_path.read_text()), indent=2, sort_keys=True),
							 encoding="utf-8")
	restored = _loaders(relocated, config)[0]
	restored.load_state_dict(state)
	_assert_equal(restored.next_batch(), original.next_batch())


@pytest.mark.parametrize("case", ["batch_size", "seq_len", "overfit_batch", "manifest", "cursor", "legacy"])
def test_incompatible_resume_is_rejected_before_changing_state(prepared_data, tmp_path, case):
	_seed(123)
	model = StochasticLM()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
	config = replace(_train_config(), max_steps=2)
	loaders = _loaders(prepared_data, config)
	checkpoint = tmp_path / "resume.pt"
	train(model, optimizer, *loaders, config, "cpu", checkpoint_path=checkpoint)

	config = replace(config, max_steps=5)
	if case in ("batch_size", "seq_len", "overfit_batch"):
		config = replace(config, **{case: {"batch_size": 1, "seq_len": 4, "overfit_batch": True}[case]})
	directory = _prepare_data(tmp_path / "changed", offset=1) if case == "manifest" else prepared_data
	target_loaders = _loaders(directory, config)
	if case in ("cursor", "legacy"):
		saved = torch.load(checkpoint, weights_only=True)
		if case == "cursor":
			saved["loaders"]["train"]["position"] = 1000
		else:
			saved["format_version"] = 2
		torch.save(saved, checkpoint)

	_seed(999)
	target = StochasticLM()
	target_optimizer = torch.optim.AdamW(target.parameters(), lr=3e-4)
	model_before = copy.deepcopy(target.state_dict())
	optimizer_before = copy.deepcopy(target_optimizer.state_dict())
	cursors_before = [loader.position for loader in target_loaders]
	rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
	with pytest.raises(ValueError):
		load_checkpoint(checkpoint, target, target_optimizer, config, vocab_size=32,
						train_loader=target_loaders[0], val_loader=target_loaders[1])
	_assert_equal(target.state_dict(), model_before)
	_assert_equal(target_optimizer.state_dict(), optimizer_before)
	assert [loader.position for loader in target_loaders] == cursors_before
	assert random.getstate() == rng_before[0]
	np.testing.assert_equal(np.random.get_state(), rng_before[1])
	torch.testing.assert_close(torch.get_rng_state(), rng_before[2], rtol=0, atol=0)


@pytest.mark.parametrize("overfit", [False, True])
def test_cli_resume_in_fresh_process_matches_continuous_run(prepared_data, tmp_path,
														  kda_config_factory, overfit):
	architecture = kda_config_factory(num_layers=4, layer_types=("kda", "kda", "kda", "mla"),
									 num_dense_layers=1, attn_res_block_size=2,
									 num_experts=4, num_experts_per_token=2)
	path = tmp_path / "model.yaml"
	path.write_text(yaml.safe_dump(asdict(architecture)), encoding="utf-8")

	def run(max_steps, checkpoint, *, resume=None, seed=123):
		args = [sys.executable, str(REPO_ROOT / "scripts/train.py"),
				"--data-dir", str(prepared_data), "--model-config", str(path),
				"--device", "cpu", "--batch-size", "2", "--seq-len", "8",
				"--max-steps", str(max_steps), "--log-interval", "1", "--eval-interval", "3",
				"--eval-batches", "2", "--checkpoint-interval", "2", "--seed", str(seed),
				"--deterministic", "--checkpoint", str(checkpoint)]
		if overfit:
			args.append("--overfit-batch")
		if resume is not None:
			args.extend(["--resume", str(resume)])
		result = subprocess.run(args, cwd=tmp_path,
								env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
								capture_output=True, text=True, check=False, timeout=60)
		assert result.returncode == 0, result.stdout + result.stderr
		return torch.load(checkpoint, map_location="cpu", weights_only=True)

	continuous = run(5, tmp_path / "full.pt")
	partial_path = tmp_path / "partial.pt"
	run(2, partial_path)
	resumed = run(5, tmp_path / "resumed.pt", resume=partial_path, seed=999)
	for key in ("model", "optimizer", "loaders", "rng_state", "step", "tokens_seen", "overfit_batch_start"):
		_assert_equal(resumed[key], continuous[key])
