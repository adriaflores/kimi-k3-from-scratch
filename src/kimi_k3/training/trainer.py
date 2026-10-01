"""Single-device causal-LM training; batches are independent context windows.

The model returns (logits, mean_token_loss) from model(x, targets=y).
It must not retain an inference cache or recurrent state between calls.
"""

import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

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


@dataclass
class TrainState:
	step: int = 0
	tokens_seen: int = 0

	def __post_init__(self):
		for name in ("step", "tokens_seen"):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int) or value < 0:
				raise ValueError(f"{name} must be a nonnegative integer")


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


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> float:
	"""Mean token loss on a fixed validation prefix, without repeating data."""
	if max_batches < 1:
		raise ValueError("max_batches must be positive")
	was_training = model.training
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


def save_checkpoint(path, model, optimizer, state: TrainState, config: TrainConfig,
					*, vocab_size: int):
	"""Save completed updates; no loader cursor or RNG state is saved yet."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + ".tmp")
	torch.save({
		"format_version": 2,
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		**asdict(state),
		"train_config": asdict(config),
		"vocab_size": vocab_size,
		"model_config": _model_config(model),
	}, temporary)
	temporary.replace(path)


def load_checkpoint(path, model, optimizer, config: TrainConfig,
					*, vocab_size: int) -> TrainState:
	"""Restore weights and optimizer into an already constructed model.

	Move the model to its training device before constructing the optimizer.
	Optimizer hyperparameters (including LR) are restored from the checkpoint.
	Data cursors and random-number generators are intentionally not restored.
	"""
	checkpoint = torch.load(path, map_location="cpu", weights_only=True)
	if checkpoint["format_version"] != 2:
		raise ValueError("Expected trainer checkpoint format_version=2; legacy resume is unsupported")
	state = TrainState(step=checkpoint["step"], tokens_seen=checkpoint["tokens_seen"])
	# Validate structure before modifying model weights or optimizer state.
	if checkpoint["train_config"]["seq_len"] != config.seq_len:
		raise ValueError("Incompatible seq_len in checkpoint")
	if checkpoint["vocab_size"] != vocab_size:
		raise ValueError("Incompatible vocab_size in checkpoint")
	if checkpoint["model_config"] != _model_config(model):
		raise ValueError("Incompatible model_config in checkpoint (including missing config)")
	if config.max_steps < state.step:
		raise ValueError("max_steps cannot be smaller than the checkpoint step")
	model.load_state_dict(checkpoint["model"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	return state


def train(model, optimizer, train_loader, val_loader, config: TrainConfig,
		  device, *, state: TrainState | None = None, checkpoint_path=None) -> TrainState:
	"""Train until max_steps total optimizer updates, including restored steps.

	Uses TokenDataLoader.next_batch()/reset(); the loader itself is unchanged.
	Exhaustion restarts the training stream, dropping an incomplete tail.
	Validation always uses its own loader and starts from offset zero.
	"""
	state = TrainState() if state is None else state
	device = torch.device(device)
	if device.type not in ("cpu", "cuda"):
		raise ValueError("This skeleton supports CPU or one CUDA device")
	if train_loader is val_loader:
		raise ValueError("Training and validation require independent loaders")
	if state.step > config.max_steps:
		raise ValueError("state.step cannot exceed max_steps")
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

	synchronize()
	started = time.perf_counter()
	for step in range(state.step + 1, config.max_steps + 1):
		if fixed_batch is None:
			try:
				x, y = train_loader.next_batch()
			except StopIteration:
				train_loader.reset()
				x, y = train_loader.next_batch()
			x, y = x.to(device), y.to(device)
			if config.overfit_batch:
				fixed_batch = (x,y)
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
								vocab_size=train_loader.dataset.vocab_size)
			# Exclude validation, checkpoint I/O and logging from throughput.
			synchronize()
			started = time.perf_counter()
	return state
