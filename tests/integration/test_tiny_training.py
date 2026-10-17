"""Close the next-token gap using the shipped tiny YAML and the real trainer."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.summarize_training_run import summarize_run
from scripts.train import build_model
from kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from kimi_k3.training.metrics import MetricsWriter
from kimi_k3.training.run import start_run
from kimi_k3.training.trainer import (
	TrainConfig,
	TrainState,
	build_optimizer,
	evaluate,
	load_checkpoint,
	train,
)
from kimi_k3.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPO_ROOT / "configs/model/kimi_k3_tiny.yaml"


@pytest.fixture
def single_thread():
	previous = torch.get_num_threads()
	torch.set_num_threads(1)
	try:
		yield
	finally:
		torch.set_num_threads(previous)


@pytest.fixture
def deterministic_execution(monkeypatch, device):
	if device != "cuda":
		yield
		return
	monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
	enabled = torch.are_deterministic_algorithms_enabled()
	warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
	torch.use_deterministic_algorithms(True)
	try:
		yield
	finally:
		torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _prepare_data(directory):
	directory.mkdir()
	metadata = {
		"format_version": 2,
		"storage": {"format": "npy", "dtype": "uint16"},
		"tokenizer": {"vocab_size": 49152, "eos_id": 0},
		"source": {"kind": "synthetic", "purpose": "tiny next-token circuit"},
		"splits": {},
	}
	# Token 7 has two successors. A token-local lookup cannot memorize this batch.
	# Validation uses different IDs; its loss need only be finite.
	for split, pattern in (("train", [7, 13, 7, 29]), ("val", [101, 203, 101, 307])):
		tokens = np.resize(np.array(pattern, dtype=np.uint16), 129)
		filename = f"{split}_000.npy"
		path = directory / filename
		np.save(path, tokens)
		metadata["splits"][split] = {
			"tokens": len(tokens),
			"shards": [{"file": filename, "tokens": len(tokens),
						"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
		}
	(directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
	return directory


def test_shipped_tiny_memorizes_next_tokens_and_evaluates_without_mutation(tmp_path, single_thread):
	# This LR only shortens a deterministic synthetic regression check. The
	# shipped overfit recipe remains 3e-4; this is not a pretraining recipe.
	config = replace(load_config(REPO_ROOT / "configs/training/overfit.yaml", TrainConfig),
					 learning_rate=1e-2, max_steps=64, log_interval=8,
					 eval_interval=32, checkpoint_interval=64)
	torch.manual_seed(config.seed)
	data = _prepare_data(tmp_path / "data")
	train_data = ShardedTokenDataset(data, "train")
	val_data = ShardedTokenDataset(data, "val")
	train_loader = TokenDataLoader(train_data, config.batch_size, config.seq_len)
	val_loader = TokenDataLoader(val_data, config.batch_size, config.seq_len)
	x, y = train_data.get_batch(0, config.batch_size, config.seq_len)
	raw = torch.from_numpy(train_data.read_tokens(0, 129).astype(np.int64))
	assert torch.equal(x.flatten(), raw[:-1])
	assert torch.equal(y.flatten(), raw[1:])
	assert not torch.equal(x, y)
	assert set(y[x == 7].tolist()) == {13, 29}
	val_x, val_y = val_data.get_batch(0, config.batch_size, config.seq_len)
	assert not torch.equal(x, val_x)

	model = build_model(train_data.vocab_size, config.seq_len, model_config_path=MODEL_CONFIG)
	initial = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
	logits, loss = model(x, targets=y)
	assert logits.shape == (1, 128, 49152)
	assert torch.isfinite(logits).all() and torch.isfinite(loss)
	torch.testing.assert_close(loss, F.cross_entropy(logits.flatten(0, 1), y.flatten()))
	initial_loss = loss.item()
	loss.backward()
	gradients = {name: parameter.grad for name, parameter in model.named_parameters()
				 if parameter.grad is not None}
	assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients.values())
	# Every attention/FFN family and the final AttnRes mixture participates.
	prefixes = [f"layers.{i}.{family}." for i in range(4) for family in ("attn", "ffn")]
	prefixes += ["embed_tokens.", "output_attn_res_proj."]
	for prefix in prefixes:
		assert any(name.startswith(prefix) and torch.count_nonzero(gradient).item()
				   for name, gradient in gradients.items()), f"no nonzero gradient for {prefix}"
	del logits, loss, gradients
	model.zero_grad(set_to_none=True)
	model.load_state_dict(initial)  # Discard the diagnostic forward's QB update.
	optimizer = build_optimizer(model, config)

	observed = {"train": 0, "val": 0}

	def check_targets(module, args, kwargs):
		key = "train" if module.training else "val"
		expected_x, expected_y = (x, y) if module.training else (val_x, val_y)
		assert torch.equal(args[0], expected_x)
		assert torch.equal(kwargs["targets"], expected_y)
		observed[key] += 1

	hook = model.register_forward_pre_hook(check_targets, with_kwargs=True)
	run = start_run(model=model, optimizer=optimizer, config=config, state=TrainState(),
					train_loader=train_loader, val_loader=val_loader,
					repo_root=REPO_ROOT, run_dir=tmp_path / "run")
	try:
		with MetricsWriter(run) as metrics:
			state = train(model, optimizer, train_loader, val_loader, config, "cpu",
						  checkpoint_path=run.checkpoint_path, metrics=metrics)
	finally:
		hook.remove()
	assert observed == {"train": config.max_steps, "val": 2}
	assert state.step == config.max_steps and state.skipped_updates == 0
	assert state.tokens_seen == config.max_steps * config.seq_len
	assert state.overfit_batch_start == 0
	assert train_loader.position == config.seq_len
	for prefix in prefixes:
		assert any(name.startswith(prefix) and not torch.equal(parameter, initial[name])
				   for name, parameter in model.named_parameters()), f"unchanged weights for {prefix}"

	before_eval = {name: tensor.clone() for name, tensor in model.state_dict().items()}
	cursor, rng = train_loader.state_dict(), torch.get_rng_state().clone()
	val_loss = evaluate(model, val_loader, "cpu", max_batches=1)
	assert math.isfinite(val_loss) and model.training
	assert train_loader.state_dict() == cursor
	assert val_loader.position == config.seq_len
	assert torch.equal(torch.get_rng_state(), rng)
	for name, tensor in model.state_dict().items():
		assert torch.equal(tensor, before_eval[name]), f"evaluation changed {name}"

	model.eval()
	with torch.no_grad():
		logits, final_loss = model(x, targets=y)
		predictions = logits.argmax(dim=-1)
	# Low loss and both successors rule out learning only the target vocabulary
	# or copying the input. This demonstrates memorization, not generalization.
	assert final_loss.item() < 0.1 and final_loss.item() < initial_loss / 2
	assert torch.equal(predictions, y)
	print("tiny_next_token: " + json.dumps({
		"updates": state.step, "tokens_seen": state.tokens_seen,
		"initial_loss": initial_loss, "fixed_batch_loss": final_loss.item(),
		"next_token_accuracy": (predictions == y).float().mean().item(),
		"val_loss": val_loss, "train_cursor": train_loader.position,
		"qb_frozen_in_eval": True, "run_directory": str(run.directory),
	}))


def _assert_checkpoint_equal(actual, expected):
	if isinstance(actual, torch.Tensor):
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	elif isinstance(actual, dict):
		assert actual.keys() == expected.keys()
		for name in actual:
			_assert_checkpoint_equal(actual[name], expected[name])
	elif isinstance(actual, (list, tuple)):
		assert type(actual) is type(expected) and len(actual) == len(expected)
		for left, right in zip(actual, expected, strict=True):
			_assert_checkpoint_equal(left, right)
	else:
		assert actual == expected


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("muon_compile,muon_batched,device", [(False, False, "cpu"), (True, False, "cpu"),
	(False, True, "cpu"),
	pytest.param(True, False, "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
	pytest.param(True, True, "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_shipped_tiny_integrated_recipe_learns_and_resumes_exactly(tmp_path, single_thread, precision,
																   muon_compile, muon_batched, device, deterministic_execution):
	# The full 64-update recipe is a separate measured experiment. Keep this
	# regression at 16 updates, retaining its LR, accumulation and fixed horizon.
	base = replace(load_config(REPO_ROOT / "configs/training/acceptance.yaml", TrainConfig),
		max_steps=16, precision=precision, muon_compile=muon_compile,
		muon_batched=muon_batched, muon_batch_size=2)
	data = _prepare_data(tmp_path / "data")

	def execute(directory, config, *, checkpoint=None, seed=None):
		torch.manual_seed(config.seed if seed is None else seed)
		loaders = [TokenDataLoader(ShardedTokenDataset(data, split), config.batch_size, config.seq_len)
			for split in ("train", "val")]
		model = build_model(49152, config.seq_len, model_config_path=MODEL_CONFIG).to(device)
		optimizer = build_optimizer(model, config)
		state = (load_checkpoint(checkpoint, model, optimizer, config, vocab_size=49152,
			train_loader=loaders[0], val_loader=loaders[1]) if checkpoint else TrainState())
		run = start_run(model=model, optimizer=optimizer, config=config, state=state,
			train_loader=loaders[0], val_loader=loaders[1], repo_root=REPO_ROOT, run_dir=directory)
		with MetricsWriter(run) as metrics:
			train(model, optimizer, *loaders, config, device, state=state,
				checkpoint_path=run.checkpoint_path, metrics=metrics)
		assert all(not hasattr(module, "_qk_clip_observer") for module in model.modules())
		assert all(parameter.grad is None for parameter in model.parameters())
		return torch.load(run.checkpoint_path, weights_only=True), run

	continuous, run = execute(tmp_path / "continuous", replace(base))
	_, partial = execute(tmp_path / "partial", replace(base, max_steps=8))
	resumed, resumed_run = execute(tmp_path / "resumed", replace(base), checkpoint=partial.checkpoint_path, seed=999)
	_assert_checkpoint_equal(resumed, continuous)
	assert resumed["format_version"] == 7
	assert resumed["train_config"]["learning_rate"] == 3e-4
	assert resumed["train_config"]["muon_compile"] == muon_compile
	assert resumed["train_config"]["muon_batched"] == muon_batched
	assert resumed["step"] == 16 and resumed["tokens_seen"] == 4096
	assert resumed["qk_clip"]["threshold"] == 100.0
	states = resumed["optimizer"]["state"].values()
	assert any("momentum_buffer" in state for state in states)
	assert any("exp_avg" in state for state in states)
	result = summarize_run(run.directory, after_step=8, expected_step=16)
	assert result["last_logged_train_loss"] < result["first_logged_train_loss"]
	assert result["skipped_updates"] == result["nonfinite_grad_norms"] == 0
	assert result["optimizer_algorithms"] == ["adamw", "muon"]
	assert result["qk_clip"]["observed_head_updates"] == 32
	assert math.isfinite(result["qk_clip"]["max_logit"])
	assert result["steady_logged_tokens"] == 2048
	assert result["steady_train_tokens_per_second"] > 0
	assert not result["profiled"]
	continuation = summarize_run(resumed_run.directory, expected_step=16)
	assert continuation["start_step"] == 8 and continuation["logged_updates"] == 8
	print("tiny_hybrid_recipe: " + json.dumps({
		"precision": precision, "batched": muon_batched,
		"updates": result["logged_updates"], "tokens": result["logged_tokens"],
		"first_window_loss": result["first_logged_train_loss"], "last_window_loss": result["last_logged_train_loss"],
		"qk_max_logit": result["qk_clip"]["max_logit"], "exact_resume": True,
	}))
