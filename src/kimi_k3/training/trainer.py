"""Single-device causal-LM training; batches are independent context windows.

The model returns (logits, mean_token_loss) from model(x, targets=y).
It must not retain an inference cache or recurrent state between calls.
"""

import math
import os
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainConfig:
	batch_size: int = 4
	seq_len: int = 128
	learning_rate: float = 3e-4
	weight_decay: float = 0.0
	max_steps: int = 1000
	log_interval: int = 10
	eval_interval: int = 100
	eval_batches: int = 10
	checkpoint_interval: int = 100
	overfit_batch: bool = False

	def __post_init__(self):
		for name in ("batch_size", "seq_len", "max_steps", "log_interval",
					 "eval_interval", "eval_batches", "checkpoint_interval"):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int) or value < 1:
				raise ValueError(f"{name} must be a positive integer")
		if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
			raise ValueError("learning_rate must be finite and positive")
		if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
			raise ValueError("weight_decay must be finite and nonnegative")
		if not isinstance(self.overfit_batch, bool):
			raise TypeError("overfit_batch must be a bool")


@dataclass
class TrainState:
	step: int = 0
	tokens_seen: int = 0
	overfit_batch_start: int | None = None

	def __post_init__(self):
		for name in ("step", "tokens_seen"):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int) or value < 0:
				raise ValueError(f"{name} must be a nonnegative integer")
		if self.overfit_batch_start is not None:
			value = self.overfit_batch_start
			if isinstance(value, bool) or not isinstance(value, int) or value < 0:
				raise ValueError("overfit_batch_start must be a nonnegative integer or None")


def _model_config(model):
	"""Optional model.config: a dataclass instance or a dict of plain values."""
	config = getattr(model, "config", None)
	if config is None:
		return None
	if is_dataclass(config) and not isinstance(config, type):
		return asdict(config)
	if isinstance(config, dict):
		return dict(config)
	raise TypeError("model.config must be a dataclass instance or dict")


def _capture_rng_state(device) -> dict:
	"""Global RNGs used by this single-device trainer; safe for weights_only=True."""
	device = torch.device(device)
	numpy_state = np.random.get_state()
	return {
		"python": random.getstate(),
		# NumPy's ndarray cannot be loaded by the restricted torch unpickler.
		"numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
		"torch_cpu": torch.get_rng_state(),
		"torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
	}


def _restore_rng_state(state: dict, device) -> None:
	device = torch.device(device)
	random.setstate(state["python"])
	numpy_state = state["numpy"]
	np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
	torch.set_rng_state(state["torch_cpu"])
	if device.type == "cuda":
		torch.cuda.set_rng_state(state["torch_cuda"], device)


def _runtime_signature(model) -> dict:
	"""Reject changes in execution settings that can change the training trajectory."""
	parameter = next(model.parameters())
	return {
		"torch_version": str(torch.__version__),
		"device_type": parameter.device.type,
		"dtype": str(parameter.dtype),
		"num_threads": torch.get_num_threads(),
		"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
		"deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
		"cudnn_benchmark": torch.backends.cudnn.benchmark,
		"cudnn_deterministic": torch.backends.cudnn.deterministic,
		"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
			if parameter.device.type == "cuda" else None,
	}


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> float:
	"""Mean token loss on a fixed validation prefix, without repeating data."""
	if max_batches < 1:
		raise ValueError("max_batches must be positive")
	was_training = model.training
	rng_state = _capture_rng_state(device)
	model.eval()
	total_loss = 0.0
	total_tokens = 0
	try:
		loader.reset()
		for _ in range(max_batches):
			try:
				x, y = loader.next_batch()
			except StopIteration:
				break
			x, y = x.to(device), y.to(device)
			_, loss = model(x, targets=y)
			value = loss.item()
			if not math.isfinite(value):
				raise FloatingPointError("Non-finite validation loss")
			total_loss += value * y.numel()
			total_tokens += y.numel()
		if total_tokens == 0:
			raise ValueError("Validation split cannot provide a full batch")
		return total_loss / total_tokens
	finally:
		model.train(was_training)
		_restore_rng_state(rng_state, device)


def save_checkpoint(path, model, optimizer, state: TrainState, config: TrainConfig,
					*, vocab_size: int, train_loader, val_loader):
	"""Save at an update boundary, after zero_grad; no partial gradients are saved."""
	if any(parameter.grad is not None for parameter in model.parameters()):
		raise ValueError("Checkpoint must be saved after optimizer.zero_grad(set_to_none=True)")
	device = next(model.parameters()).device
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + ".tmp")
	torch.save({
		"format_version": 3,
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		**asdict(state),
		"train_config": asdict(config),
		"vocab_size": vocab_size,
		"model_config": _model_config(model),
		"optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
		"loaders": {"train": train_loader.state_dict(), "val": val_loader.state_dict()},
		"rng_state": _capture_rng_state(device),
		"runtime": _runtime_signature(model),
	}, temporary)
	temporary.replace(path)


def load_checkpoint(path, model, optimizer, config: TrainConfig,
					*, vocab_size: int, train_loader, val_loader) -> TrainState:
	"""Restore completed updates, data cursors and RNG into a reconstructed run.

	Move the model to its training device before constructing the optimizer.
	Optimizer hyperparameters (including LR) are restored from the checkpoint.
	RNG is restored last, after model/optimizer construction and state loading.
	Continue on the same data, software and device with deterministic kernels
	for bitwise equivalence. Format 2 lacks the required state and is rejected.
	"""
	checkpoint = torch.load(path, map_location="cpu", weights_only=True)
	if checkpoint["format_version"] != 3:
		raise ValueError("Expected trainer checkpoint format_version=3; older checkpoints lack cursor/RNG state")
	state = TrainState(
		step=checkpoint["step"], tokens_seen=checkpoint["tokens_seen"],
		overfit_batch_start=checkpoint["overfit_batch_start"],
	)
	# Validate structure before modifying model weights or optimizer state.
	for name in ("batch_size", "seq_len", "overfit_batch"):
		if checkpoint["train_config"][name] != getattr(config, name):
			raise ValueError(f"Incompatible {name} in checkpoint")
	if checkpoint["vocab_size"] != vocab_size:
		raise ValueError("Incompatible vocab_size in checkpoint")
	if checkpoint["model_config"] != _model_config(model):
		raise ValueError("Incompatible model_config in checkpoint (including missing config)")
	if config.max_steps < state.step:
		raise ValueError("max_steps cannot be smaller than the checkpoint step")
	if checkpoint["optimizer_type"] != f"{type(optimizer).__module__}.{type(optimizer).__qualname__}":
		raise ValueError("Incompatible optimizer type in checkpoint")
	if checkpoint["runtime"] != _runtime_signature(model):
		raise ValueError("Incompatible runtime in checkpoint (version, device, dtype, threads or determinism)")
	train_position = train_loader.validate_state_dict(checkpoint["loaders"]["train"])
	val_loader.validate_state_dict(checkpoint["loaders"]["val"])
	if config.overfit_batch and state.step > 0 and state.overfit_batch_start is None:
		raise ValueError("Checkpoint is missing the fixed batch offset")
	if state.overfit_batch_start is not None:
		batch_end = state.overfit_batch_start + config.batch_size * config.seq_len
		if not config.overfit_batch or batch_end + 1 > train_loader.dataset.total_tokens:
			raise ValueError("Incompatible fixed batch offset in checkpoint")
		if train_position != batch_end:
			raise ValueError("Fixed batch offset does not match the saved training cursor")
	model.load_state_dict(checkpoint["model"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	optimizer.zero_grad(set_to_none=True)
	config.learning_rate = optimizer.param_groups[0]["lr"]
	config.weight_decay = optimizer.param_groups[0]["weight_decay"]
	train_loader.load_state_dict(checkpoint["loaders"]["train"])
	val_loader.load_state_dict(checkpoint["loaders"]["val"])
	_restore_rng_state(checkpoint["rng_state"], next(model.parameters()).device)
	return state


def train(model, optimizer, train_loader, val_loader, config: TrainConfig,
		  device, *, state: TrainState | None = None, checkpoint_path=None) -> TrainState:
	"""Train until max_steps total optimizer updates, including restored steps.

	Uses TokenDataLoader.next_batch()/reset(); the loader itself is unchanged.
	Exhaustion restarts the training stream, dropping an incomplete tail.
	Validation always uses its own loader and starts from offset zero.
	In overfit mode the batch is identified by its token offset; resume rereads
	that batch without advancing the restored loader cursor.
	"""
	state = TrainState() if state is None else state
	device = torch.device(device)
	if device.type not in ("cpu", "cuda"):
		raise ValueError("This skeleton supports CPU or one CUDA device")
	if train_loader is val_loader:
		raise ValueError("Training and validation require independent loaders")
	if state.step > config.max_steps:
		raise ValueError("state.step cannot exceed max_steps")
	if not config.overfit_batch and state.overfit_batch_start is not None:
		raise ValueError("A fixed batch state requires overfit_batch=True")
	if train_loader.dataset.vocab_size != val_loader.dataset.vocab_size:
		raise ValueError("Training and validation vocab_size must match")
	for loader in (train_loader, val_loader):
		if (loader.batch_size, loader.seq_len) != (config.batch_size, config.seq_len):
			raise ValueError("Loader batch dimensions must match TrainConfig")
		if loader.dataset.total_tokens < config.batch_size * config.seq_len + 1:
			raise ValueError("Each split must contain at least one full batch")

	def synchronize():
		if device.type == "cuda":
			torch.cuda.synchronize(device)

	model.train()
	optimizer.zero_grad(set_to_none=True)
	window_loss = 0.0
	window_tokens = 0
	window_steps = 0
	training_seconds = 0.0
	fixed_batch = None
	if config.overfit_batch and state.overfit_batch_start is not None:
		x, y = train_loader.dataset.get_batch(
			state.overfit_batch_start, config.batch_size, config.seq_len,
		)
		fixed_batch = (x.to(device), y.to(device))

	synchronize()
	started = time.perf_counter()
	for step in range(state.step + 1, config.max_steps + 1):
		if fixed_batch is None:
			try:
				batch_start = train_loader.position
				x, y = train_loader.next_batch()
			except StopIteration:
				train_loader.reset()
				batch_start = train_loader.position
				x, y = train_loader.next_batch()

			x, y = x.to(device), y.to(device)

			if config.overfit_batch:
				state.overfit_batch_start = batch_start
				fixed_batch = (x, y)
		else:
			x, y = fixed_batch

		_, loss = model(x, targets=y)
		value = loss.item()
		if not math.isfinite(value):
			raise FloatingPointError(f"Non-finite training loss at step {step}")
		loss.backward()
		optimizer.step()
		state.step = step
		state.tokens_seen += y.numel()
		optimizer.zero_grad(set_to_none=True)

		window_loss += value
		window_tokens += x.numel()
		window_steps += 1
		final = step == config.max_steps
		log_now = step % config.log_interval == 0 or final
		eval_now = step % config.eval_interval == 0 or final
		save_now = checkpoint_path is not None and (
			step % config.checkpoint_interval == 0 or final
		)
		if log_now or eval_now or save_now:
			synchronize()
			training_seconds += time.perf_counter() - started
			if log_now:
				print(f"step {step} | train loss {window_loss / window_steps:.4f}"
					  f" | tokens_seen {state.tokens_seen:,}"
					  f" | tokens/s {window_tokens / training_seconds:,.0f}", flush=True)
				window_loss = 0.0
				window_tokens = window_steps = 0
				training_seconds = 0.0
			if eval_now:
				val_loss = evaluate(model, val_loader, device, config.eval_batches)
				print(f"step {step} | val loss {val_loss:.4f}"
					  f" | tokens_seen {state.tokens_seen:,}", flush=True)
			if save_now:
				save_checkpoint(checkpoint_path, model, optimizer, state, config,
								vocab_size=train_loader.dataset.vocab_size,
								train_loader=train_loader, val_loader=val_loader)
			# Exclude validation, checkpoint I/O and logging from throughput.
			synchronize()
			started = time.perf_counter()
	return state
