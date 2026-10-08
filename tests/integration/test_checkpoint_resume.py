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
from src.kimi_k3.training.metrics import MetricsWriter
from src.kimi_k3.training.optimizers import MuonAdamW
from src.kimi_k3.training.parameter_groups import build_parameter_groups
from src.kimi_k3.training.run import start_run
from src.kimi_k3.training.trainer import (
	TrainConfig,
	TrainState,
	build_optimizer,
	evaluate,
	learning_rate_at_step,
	load_checkpoint,
	train,
)

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


def _skipped_by_grad_scaler(optimizer):
	# GradScaler still calls step() on fused optimizers and passes found_inf so the kernel skips the update.
	found_inf = getattr(optimizer, "found_inf", None)
	return found_inf is not None and found_inf.item() > 0


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
		config = replace(_train_config(overfit_batch=overfit), max_steps=max_steps,
						 precision="bf16" if device == "cuda" else "fp32")
		optimizer = build_optimizer(model, config)
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
	partial_run = start_run(model=partial_model, optimizer=partial_optimizer, config=partial_config,
							state=TrainState(), train_loader=partial_loaders[0], val_loader=partial_loaders[1],
							repo_root=tmp_path, runs_root=tmp_path / "runs", checkpoint_path=checkpoint)
	with MetricsWriter(partial_run) as metrics:
		train(partial_model, partial_optimizer, *partial_loaders, partial_config, device,
			  checkpoint_path=checkpoint, metrics=metrics)
	resumed_model, resumed_optimizer, resumed_config, resumed_loaders = construct(999, 5)
	state = load_checkpoint(checkpoint, resumed_model, resumed_optimizer, resumed_config,
							vocab_size=32, train_loader=resumed_loaders[0], val_loader=resumed_loaders[1])
	resumed_losses = _record_training_losses(resumed_model)
	resumed_run = start_run(model=resumed_model, optimizer=resumed_optimizer, config=resumed_config,
							state=state, train_loader=resumed_loaders[0], val_loader=resumed_loaders[1],
							repo_root=tmp_path, run_dir=partial_run.directory, resume_path=checkpoint)
	with MetricsWriter(resumed_run) as metrics:
		resumed_state = train(resumed_model, resumed_optimizer, *resumed_loaders,
							  resumed_config, device, state=state, metrics=metrics)

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
		config = replace(_train_config(), max_steps=max_steps)
		optimizer = build_optimizer(model, config)
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


@pytest.mark.parametrize("case", ["batch_size", "seq_len", "overfit_batch", "manifest", "cursor", "legacy",
	"gradient_accumulation_steps", "precision", "lr_schedule", "warmup_steps", "max_grad_norm",
	"decay_steps", "min_lr_ratio", "adam_fused", "muon_ns_dtype"])
def test_incompatible_resume_is_rejected_before_changing_state(prepared_data, tmp_path, case):
	_seed(123)
	model = StochasticLM()
	config = replace(_train_config(), max_steps=2)
	optimizer = build_optimizer(model, config)
	if case in ("decay_steps", "min_lr_ratio"):
		config = replace(config, lr_schedule="cosine")
	loaders = _loaders(prepared_data, config)
	checkpoint = tmp_path / "resume.pt"
	train(model, optimizer, *loaders, config, "cpu", checkpoint_path=checkpoint)

	config = replace(config, max_steps=5)
	incompatible = {"batch_size": 1, "seq_len": 4, "overfit_batch": True,
					"gradient_accumulation_steps": 2, "precision": "bf16", "lr_schedule": "cosine",
					"warmup_steps": 1, "max_grad_norm": 0.1, "decay_steps": 1001, "min_lr_ratio": 0.2,
					"adam_fused": False, "muon_ns_dtype": "fp32"}
	if case in incompatible:
		config = replace(config, **{case: incompatible[case]})
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
	target_optimizer = build_optimizer(target, config)
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


@pytest.mark.parametrize("version", [3, 4, 5])
def test_legacy_adamw_checkpoint_is_rejected_before_mutation(prepared_data, tmp_path, version):
	config = replace(_train_config(), max_steps=2)
	model = StochasticLM()
	# Produce real auxiliary AdamW moments, representing the old trainer recipe.
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
	path = tmp_path / "legacy.pt"
	train(model, optimizer, *_loaders(prepared_data, config), config, "cpu", checkpoint_path=path)
	saved = torch.load(path, weights_only=True)
	saved["format_version"] = version
	saved.pop("optimizer_layout")
	for name in ("muon_momentum", "muon_ns_steps", "muon_ns_dtype"):
		saved["train_config"].pop(name)
	torch.save(saved, path)
	target = StochasticLM()
	target_optimizer = build_optimizer(target, config)
	loaders = _loaders(prepared_data, config)
	before = copy.deepcopy(target.state_dict())
	optimizer_before = copy.deepcopy(target_optimizer.state_dict())
	with pytest.raises(ValueError, match="Legacy AdamW.*start a new run"):
		load_checkpoint(path, target, target_optimizer, config, vocab_size=32,
			train_loader=loaders[0], val_loader=loaders[1])
	_assert_equal(target.state_dict(), before)
	_assert_equal(target_optimizer.state_dict(), optimizer_before)
	assert all(loader.position == 0 for loader in loaders)


@pytest.mark.parametrize("case", ["layout_name", "missing_layout", "group_names", "group_indices",
	"matrix_view", "momentum", "missing_recipe_field", "layout_splits", "group_splits", "missing_splits"])
def test_hybrid_checkpoint_rejects_changed_assignment_before_mutation(
	prepared_data, tmp_path, kda_config_factory, case,
):
	config = replace(_train_config(), max_steps=1, weight_decay=0.1)
	architecture = kda_config_factory(num_layers=2, layer_types=("kda", "mla"), num_dense_layers=1)
	model = KimiK3(architecture)
	path = tmp_path / "hybrid.pt"
	train(model, build_optimizer(model, config), *_loaders(prepared_data, config), config, "cpu", checkpoint_path=path)
	saved = torch.load(path, weights_only=True)
	group = next(group for group in saved["optimizer"]["param_groups"]
		if group["algorithm"] == "muon" and "muon_shape" not in group)
	if case == "layout_name":
		saved["optimizer_layout"][0]["parameters"][0]["name"] = "different.weight"
	elif case == "missing_layout":
		saved.pop("optimizer_layout")
	elif case == "group_names":
		group["param_names"][:2] = reversed(group["param_names"][:2])
	elif case == "group_indices":
		group["params"][:2] = reversed(group["params"][:2])
	elif case == "matrix_view":
		group["muon_shape"] = (1, 24, 16)
	elif case == "momentum":
		group["momentum"] = 0.8
	elif case in ("layout_splits", "group_splits", "missing_splits"):
		if case == "layout_splits":
			group = next(item for item in saved["optimizer_layout"] if "muon_split_sizes" in item)
		else:
			group = next(item for item in saved["optimizer"]["param_groups"] if "muon_split_sizes" in item)
		if case == "missing_splits":
			group.pop("muon_split_sizes")
		else:
			group["muon_split_sizes"] = (2, 6)  # Same total rows; different K/V boundary.
	else:
		saved["train_config"].pop("muon_ns_dtype")
	torch.save(saved, path)
	target = KimiK3(architecture)
	optimizer = build_optimizer(target, config)
	loaders = _loaders(prepared_data, config)
	before = copy.deepcopy(target.state_dict())
	optimizer_before = copy.deepcopy(optimizer.state_dict())
	rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
	config_before = asdict(config)
	with pytest.raises(ValueError):
		load_checkpoint(path, target, optimizer, config, vocab_size=32,
			train_loader=loaders[0], val_loader=loaders[1])
	_assert_equal(target.state_dict(), before)
	_assert_equal(optimizer.state_dict(), optimizer_before)
	assert asdict(config) == config_before
	assert all(loader.position == 0 for loader in loaders)
	assert random.getstate() == rng_before[0]
	np.testing.assert_equal(np.random.get_state(), rng_before[1])
	_assert_equal(torch.get_rng_state(), rng_before[2])


def test_deliverable_2_full_attention_checkpoint_cannot_resume_per_head_recipe(
	prepared_data, tmp_path, kda_config_factory,
):
	config = replace(_train_config(), max_steps=1, weight_decay=0.1)
	architecture = kda_config_factory(num_layers=2, layer_types=("kda", "mla"), num_dense_layers=1)
	model = KimiK3(architecture)
	# Reconstruct milestone 2's grouping: same tensors and algorithms, fused
	# gate/up still split, but QKV/UQ/UKV are complete attention matrices.
	assignment = {name: group for group in build_parameter_groups(model, config.weight_decay)
		for name in group["param_names"]}
	previous_groups = {}
	for name, parameter in model.named_parameters():
		group = assignment[name]
		view = None if name.endswith(("QKV.weight", "W_UQ.weight", "W_UKV.weight")) else group.get("muon_shape")
		key = (group["algorithm"], view, group["weight_decay"])
		if key not in previous_groups:
			previous_groups[key] = {"params": [], "param_names": [], "algorithm": group["algorithm"],
				"weight_decay": group["weight_decay"]}
			if view is not None:
				previous_groups[key]["muon_shape"] = view
		previous_groups[key]["params"].append(parameter)
		previous_groups[key]["param_names"].append(name)
	old_optimizer = MuonAdamW(list(previous_groups.values()), lr=config.learning_rate,
		momentum=config.muon_momentum, ns_steps=config.muon_ns_steps)
	path = tmp_path / "full_attention.pt"
	train(model, old_optimizer, *_loaders(prepared_data, config), config, "cpu", checkpoint_path=path)
	saved = torch.load(path, weights_only=True)
	assert saved["format_version"] == 6
	assert not any("muon_split_sizes" in group for group in saved["optimizer_layout"])
	assert saved["optimizer"]["state"]
	target = KimiK3(architecture)
	optimizer = build_optimizer(target, config)
	loaders = _loaders(prepared_data, config)
	before = copy.deepcopy(target.state_dict())
	with pytest.raises(ValueError, match="Incompatible optimizer.*start a new run"):
		load_checkpoint(path, target, optimizer, config, vocab_size=32,
			train_loader=loaders[0], val_loader=loaders[1])
	_assert_equal(target.state_dict(), before)
	assert not optimizer.state
	assert all(loader.position == 0 for loader in loaders)


def test_hybrid_resume_restores_all_hyperparameters_even_after_zero_decay_build(
	prepared_data, tmp_path, kda_config_factory,
):
	config = replace(_train_config(), max_steps=2, weight_decay=0.2, muon_momentum=0.9,
		muon_ns_steps=7, adam_beta1=0.8, adam_beta2=0.98, adam_eps=1e-7)
	architecture = kda_config_factory(num_layers=2, layer_types=("kda", "mla"), num_dense_layers=1)
	model = KimiK3(architecture)
	path = tmp_path / "restore.pt"
	train(model, build_optimizer(model, config), *_loaders(prepared_data, config), config, "cpu", checkpoint_path=path)
	target = KimiK3(architecture)
	requested = replace(config, max_steps=3, learning_rate=0.5, weight_decay=0,
		muon_momentum=0.7, muon_ns_steps=3, adam_beta1=0.6, adam_beta2=0.7, adam_eps=0.1)
	optimizer = build_optimizer(target, requested)
	loaders = _loaders(prepared_data, requested)
	state = load_checkpoint(path, target, optimizer, requested, vocab_size=32,
		train_loader=loaders[0], val_loader=loaders[1])
	assert state.step == 2
	for name in ("learning_rate", "weight_decay", "muon_momentum", "muon_ns_steps",
		"adam_beta1", "adam_beta2", "adam_eps"):
		assert getattr(requested, name) == getattr(config, name)
	train(target, optimizer, *loaders, requested, "cpu", state=state)
	assert state.step == 3
	assert {group["lr"] for group in optimizer.param_groups} == {config.learning_rate}


@pytest.mark.parametrize("overfit", [False, True])
@pytest.mark.parametrize("use_recipe", [False, True])
def test_cli_resume_in_fresh_process_matches_continuous_run(prepared_data, tmp_path,
														  kda_config_factory, overfit, use_recipe):
	architecture = kda_config_factory(num_layers=4, layer_types=("kda", "kda", "kda", "mla"),
									 num_dense_layers=1, attn_res_block_size=2,
									 num_experts=4, num_experts_per_token=2,
									 mla_qk_nope_head_dim=3, mla_v_head_dim=5)
	path = tmp_path / "model.yaml"
	path.write_text(yaml.safe_dump(asdict(architecture)), encoding="utf-8")
	recipe_path = tmp_path / "train.yaml"
	recipe = replace(_train_config(), gradient_accumulation_steps=2, precision="bf16",
					 lr_schedule="cosine", warmup_steps=2, decay_steps=5,
					 max_grad_norm=0.1, weight_decay=0.1)
	recipe_path.write_text(yaml.safe_dump(asdict(recipe)), encoding="utf-8")

	def run(max_steps, checkpoint, *, resume=None, seed=123):
		run_dir = tmp_path / ("full_run" if resume is None and checkpoint.name == "full.pt" else "partial_run")
		args = [sys.executable, str(REPO_ROOT / "scripts/train.py"),
				"--data-dir", str(prepared_data), "--model-config", str(path),
				"--device", "cpu", "--batch-size", "2", "--seq-len", "8",
				"--max-steps", str(max_steps), "--log-interval", "1", "--eval-interval", "3",
				"--eval-batches", "2", "--checkpoint-interval", "2", "--seed", str(seed),
				"--deterministic", "--run-dir", str(run_dir), "--checkpoint", str(checkpoint)]
		if overfit:
			args.append("--overfit-batch")
		if use_recipe:
			args.extend(["--training-config", str(recipe_path)])
		if resume is not None:
			# Saved optimizer settings must win, including in the run manifest.
			args.extend(["--resume", str(resume), "--learning-rate", "0.5",
						 "--adam-beta1", "0.5", "--adam-beta2", "0.8", "--adam-eps", "0.01",
						 "--muon-momentum", "0.8", "--muon-ns-steps", "7"])
		result = subprocess.run(args, cwd=tmp_path,
								env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
								capture_output=True, text=True, check=False, timeout=60)
		assert result.returncode == 0, result.stdout + result.stderr
		return torch.load(checkpoint, map_location="cpu", weights_only=True)

	continuous = run(5, tmp_path / "full.pt")
	partial_path = tmp_path / "partial.pt"
	partial = run(3 if use_recipe else 2, partial_path)
	metrics_path = tmp_path / "partial_run" / "metrics.jsonl"
	original_metrics = metrics_path.read_bytes()
	manifest_path = tmp_path / "partial_run" / "run.json"
	original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	resumed = run(5, tmp_path / "resumed.pt", resume=partial_path, seed=999)
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	assert manifest["run_id"] == original_manifest["run_id"]
	assert manifest["sessions"][0] == original_manifest["sessions"][0]
	assert len(manifest["sessions"]) == 2
	session = manifest["sessions"][-1]
	assert session["start_state"]["step"] == partial["step"]
	assert session["start_state"]["tokens_seen"] == partial["tokens_seen"]
	assert session["start_state"]["loaders"] == partial["loaders"]
	assert session["train_config"]["max_steps"] == 5
	assert session["train_config"]["learning_rate"] == partial["train_config"]["learning_rate"]
	for name in ("adam_beta1", "adam_beta2", "adam_eps", "weight_decay", "muon_momentum", "muon_ns_steps"):
		assert session["train_config"][name] == partial["train_config"][name]
	assert session["optimizer"]["param_groups"][0]["lr"] == partial["optimizer"]["param_groups"][0]["lr"]
	assert session["rng_source"] == "checkpoint"
	assert metrics_path.read_bytes().startswith(original_metrics)
	records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
	assert all(record["run_id"] == manifest["run_id"] for record in records)
	for run_session, start, end in ((manifest["sessions"][0], 0, partial["step"]),
									  (session, partial["step"], 5)):
		segment = [record for record in records if record["session_id"] == run_session["session_id"]]
		assert segment[0]["event"] == "session_start" and segment[0]["step"] == start
		assert segment[-1]["event"] == "session_end" and segment[-1]["step"] == end
		assert segment[0]["session"]["tokens"] == 0
		assert segment[-1]["session"]["tokens"] == segment[-1]["tokens_seen"] - segment[0]["tokens_seen"]
		assert segment[-1]["session"]["elapsed_seconds"] > 0
		assert segment[-1]["session"]["eval_seconds"] > 0
		assert segment[-1]["session"]["checkpoint_seconds"] > 0
		updates = [record for record in segment if record["event"] == "train"]
		assert [record["step"] for record in updates] == list(range(start + 1, end + 1))
		assert [record["lr"] for record in updates] == [
			learning_rate_at_step(TrainConfig(**resumed["train_config"]), step) for step in range(start + 1, end + 1)]
		for record in updates:
			assert set(record["moe"]) == {f"layers.{i}.ffn" for i in (1, 2, 3)}
			for layer in record["moe"].values():
				assert layer["tokens"] == record["tokens"] == (32 if use_recipe else 16)
				assert sum(layer["expert_counts"]) == layer["tokens"] * 2
				assert len(layer["qb_biases"]) == 4
	assert resumed["format_version"] == 6
	assert resumed["train_config"]["gradient_accumulation_steps"] == (2 if use_recipe else 1)
	for key in ("model", "optimizer", "loaders", "rng_state", "step", "tokens_seen", "overfit_batch_start",
				"skipped_updates", "grad_scaler_state"):
		_assert_equal(resumed[key], continuous[key])


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
BF16_CUDA_REQUIRED = pytest.mark.skipif(
	not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False),
	reason="Native CUDA BF16 is unavailable",
)


@pytest.mark.parametrize("device,precision,overfit", [
	("cpu", "fp32", False), ("cpu", "bf16", False), ("cpu", "fp32", True),
	pytest.param("cuda", "fp32", False, marks=CUDA_REQUIRED),
	pytest.param("cuda", "fp16", False, marks=CUDA_REQUIRED),
	pytest.param("cuda", "bf16", False, marks=BF16_CUDA_REQUIRED),
])
def test_recipe_resume_keeps_accumulation_lr_precision_and_scaler(
	prepared_data, tmp_path, kda_config_factory, device, precision, overfit,
):
	architecture = kda_config_factory(num_layers=4, layer_types=("kda", "kda", "kda", "mla"),
									 num_dense_layers=1, attn_res_block_size=2,
									 num_experts=4, num_experts_per_token=2,
									 mla_qk_nope_head_dim=3, mla_v_head_dim=5)

	def construct(seed, max_steps):
		_seed(seed)
		config = replace(_train_config(overfit_batch=overfit), max_steps=max_steps,
						 gradient_accumulation_steps=3, precision=precision,
						 lr_schedule="cosine", warmup_steps=2, decay_steps=5,
						 max_grad_norm=0.1, weight_decay=0.1)
		model = KimiK3(architecture).to(device)
		if device == "cuda" and precision != "bf16":
			# Keep FP32/FP16 trainer/scaler coverage using the pedagogical reference in this test only.
			for layer in model.layers:
				if hasattr(layer.ffn, "_forward_naive_experts"):
					layer.ffn._forward_grouped_experts = layer.ffn._forward_naive_experts
		optimizer = build_optimizer(model, config)
		loaders = _loaders(prepared_data, config)
		if overfit:
			loaders[0].reset(3)
		lrs, clipped_norms = [], []

		def record(optimizer, args, kwargs):
			if _skipped_by_grad_scaler(optimizer):
				return
			assert len({group["lr"] for group in optimizer.param_groups}) == 1
			lrs.append(optimizer.param_groups[0]["lr"])
			clipped_norms.append(torch.cat([
				p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]).norm().item())

		optimizer.register_step_pre_hook(record)
		return model, optimizer, config, loaders, lrs, clipped_norms

	full_model, full_optimizer, config, loaders, full_lrs, norms = construct(123, 5)
	full_losses = _record_training_losses(full_model)
	full_state = train(full_model, full_optimizer, *loaders, config, device)
	expected_random = _next_random_values(device)
	assert full_lrs == [learning_rate_at_step(config, step) for step in range(1, 6)]
	assert all(0 < norm <= 0.10001 for norm in norms)
	assert full_state.tokens_seen == (5 + full_state.skipped_updates) * 2 * 8 * 3

	# Stop inside warmup or after decay has already lowered the current LR.
	stop_step = 1 if overfit else 3
	partial, partial_optimizer, partial_config, partial_loaders, partial_lrs, _ = construct(123, stop_step)
	checkpoint = tmp_path / "recipe.pt"
	partial_state = train(partial, partial_optimizer, *partial_loaders, partial_config, device,
						  checkpoint_path=checkpoint)
	resumed, optimizer, resumed_config, restored_loaders, resumed_lrs, _ = construct(999, 5)
	resumed_config.learning_rate = 7e-4  # Resume must restore the saved peak, not this or the current LR.
	state = load_checkpoint(checkpoint, resumed, optimizer, resumed_config, vocab_size=32,
							train_loader=restored_loaders[0], val_loader=restored_loaders[1])
	assert resumed_config.learning_rate == config.learning_rate
	resumed_losses = _record_training_losses(resumed)
	state = train(resumed, optimizer, *restored_loaders, resumed_config, device, state=state)
	assert resumed_losses == full_losses[(stop_step + partial_state.skipped_updates) * 3:]
	assert partial_lrs + resumed_lrs == full_lrs
	_assert_equal(asdict(state), asdict(full_state))
	_assert_equal(resumed.state_dict(), full_model.state_dict())
	_assert_equal(optimizer.state_dict(), full_optimizer.state_dict())
	for actual, expected in zip(restored_loaders, loaders):
		assert actual.state_dict() == expected.state_dict()
	_assert_equal(_next_random_values(device), expected_random)


def test_accumulation_matches_large_batch_for_stateless_model(prepared_data):
	from scripts.train import SmokeTestLM

	def run(batch_size, accumulation):
		_seed(123)
		model = SmokeTestLM(32)
		config = replace(_train_config(), batch_size=batch_size, gradient_accumulation_steps=accumulation,
						 max_steps=2, max_grad_norm=0)
		optimizer = build_optimizer(model, config)
		loaders = _loaders(prepared_data, config)
		state = train(model, optimizer, *loaders, config, "cpu")
		return model, state, loaders

	large, large_state, large_loaders = run(2, 1)
	accumulated, accumulated_state, accumulated_loaders = run(1, 2)
	for actual, expected in zip(accumulated.parameters(), large.parameters()):
		torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
	assert accumulated_state.tokens_seen == large_state.tokens_seen == 32
	assert accumulated_loaders[0].position == large_loaders[0].position


@CUDA_REQUIRED
@pytest.mark.parametrize("real_backbone", [False, True])
def test_cuda_fp16_overflow_does_not_advance_optimizer_or_warmup(
	prepared_data, tmp_path, kda_config_factory, real_backbone,
):
	from scripts.train import SmokeTestLM

	_seed(123)
	model = (KimiK3(kda_config_factory(num_layers=2, layer_types=("kda", "mla"),
		num_dense_layers=2)) if real_backbone else SmokeTestLM(32)).cuda()
	config = replace(_train_config(), max_steps=3, precision="fp16", warmup_steps=2,
					 gradient_accumulation_steps=2, max_grad_norm=0.1)
	optimizer = build_optimizer(model, config)
	loaders = _loaders(prepared_data, config)
	first = True

	def overflow_once(gradient):
		nonlocal first
		if first:
			first = False
			return torch.full_like(gradient, float("inf"))
		return gradient

	(model.layers[0].attn.QKV.weight if real_backbone else model.head.weight).register_hook(overflow_once)
	initial_weights = copy.deepcopy(model.state_dict())
	initial_buffers = {name for name, _ in model.named_buffers()}
	lrs = []
	def record_first_success(optimizer, args, kwargs):
		if not lrs:
			# No successful update has happened, despite at least one overflow.
			assert not optimizer.state
			for name, tensor in model.state_dict().items():
				if name not in initial_buffers:
					_assert_equal(tensor, initial_weights[name])
		assert len({group["lr"] for group in optimizer.param_groups}) == 1
		lrs.append(optimizer.param_groups[0]["lr"])
	optimizer.register_step_pre_hook(record_first_success)
	initial_state = TrainState()
	run = start_run(model=model, optimizer=optimizer, config=config, state=initial_state,
					train_loader=loaders[0], val_loader=loaders[1], repo_root=tmp_path,
					runs_root=tmp_path / "runs")
	with MetricsWriter(run) as metrics:
		state = train(model, optimizer, *loaders, config, "cuda", state=initial_state, metrics=metrics)
	assert state.step == 3
	assert state.skipped_updates >= 1
	assert state.grad_scaler_state["scale"] < 65536
	assert state.tokens_seen == (3 + state.skipped_updates) * 2 * 8 * 2
	assert lrs == [learning_rate_at_step(config, step) for step in range(1, 4)]
	records = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text(
		encoding="utf-8").splitlines()]
	overflows = [record for record in records if record["event"] == "overflow"]
	assert len(overflows) == state.skipped_updates
	assert overflows[0]["step"] == 0 and overflows[0]["attempted_step"] == 1
	assert overflows[0]["grad_norm"] is None and not overflows[0]["grad_norm_finite"]
	updates = [record for record in records if record["event"] == "train"]
	assert sum(record["tokens"] for record in updates) == state.tokens_seen
	assert sum(record["skipped_updates_in_window"] for record in updates) == state.skipped_updates
	assert sum(record["nonfinite_grad_norms"] for record in updates) == state.skipped_updates
	assert [record["lr"] for record in updates] == lrs
	assert records[-1]["session"]["tokens"] == state.tokens_seen
	assert sum(record["train_seconds"] for record in updates) == pytest.approx(
		records[-1]["session"]["train_seconds"])
	assert records[-1]["memory"]["cuda_peak_allocated_bytes"] > 0
