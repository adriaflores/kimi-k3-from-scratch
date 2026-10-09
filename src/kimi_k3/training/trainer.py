"""Single-device causal-LM training; batches are independent context windows.

The model returns (logits, mean_token_loss) from model(x, targets=y).
It must not retain an inference cache or recurrent state between calls.
"""

import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import torch

from .metrics import TrainingWindow
from .moe_metrics import MoEDiagnostics
from .monitoring import TrainingMonitor
from .optimizers import MuonAdamW
from .parameter_groups import build_parameter_groups
from .profiling import TrainingProfiler
from .qk_clip import QKClip, _chunk_size, _positive, qk_clip_metadata


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
	gradient_accumulation_steps: int = 1
	precision: str = "fp32"
	adam_beta1: float = 0.9
	adam_beta2: float = 0.95
	adam_eps: float = 1e-8
	adam_fused: bool = True
	muon_momentum: float = 0.95
	muon_ns_steps: int = 5
	muon_ns_dtype: str = "bf16"
	muon_compile: bool = False
	qk_clip_threshold: float = 100.0
	qk_clip_chunk_size: int = 256
	lr_schedule: str = "constant"
	warmup_steps: int = 0
	decay_steps: int = 1000
	min_lr_ratio: float = 0.1
	max_grad_norm: float = 0.0
	seed: int = 1337

	def __post_init__(self):
		for name in ("batch_size", "seq_len", "max_steps", "log_interval",
					 "eval_interval", "eval_batches", "checkpoint_interval",
					 "gradient_accumulation_steps", "decay_steps"):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int) or value < 1:
				raise ValueError(f"{name} must be a positive integer")
		if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
			raise ValueError("learning_rate must be finite and positive")
		if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
			raise ValueError("weight_decay must be finite and nonnegative")
		if not isinstance(self.overfit_batch, bool):
			raise TypeError("overfit_batch must be a bool")
		if not isinstance(self.adam_fused, bool):
			raise TypeError("adam_fused must be a bool")
		if not isinstance(self.muon_compile, bool):
			raise TypeError("muon_compile must be a bool")
		if isinstance(self.muon_momentum, bool) or not math.isfinite(self.muon_momentum) or not 0 <= self.muon_momentum < 1:
			raise ValueError("muon_momentum must be finite and in [0, 1)")
		if isinstance(self.muon_ns_steps, bool) or not isinstance(self.muon_ns_steps, int) or self.muon_ns_steps < 1:
			raise ValueError("muon_ns_steps must be a positive integer")
		if self.muon_ns_dtype not in ("bf16", "fp32", "fp64"):
			raise ValueError("muon_ns_dtype must be bf16, fp32 or fp64")
		_positive("qk_clip_threshold", self.qk_clip_threshold)
		_chunk_size(self.qk_clip_chunk_size)
		if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
			raise ValueError("warmup_steps must be a nonnegative integer")
		if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
			raise ValueError("seed must be an integer in [0, 2**32)")
		if self.precision not in ("fp32", "bf16", "fp16"):
			raise ValueError("precision must be fp32, bf16 or fp16")
		if self.lr_schedule not in ("constant", "cosine"):
			raise ValueError("lr_schedule must be constant or cosine")
		if self.lr_schedule == "cosine" and self.decay_steps <= max(1, self.warmup_steps):
			raise ValueError("cosine decay_steps must exceed max(1, warmup_steps)")
		for name in ("adam_beta1", "adam_beta2"):
			value = getattr(self, name)
			if not math.isfinite(value) or not 0 <= value < 1:
				raise ValueError(f"{name} must be finite and in [0, 1)")
		if not math.isfinite(self.adam_eps) or self.adam_eps <= 0:
			raise ValueError("adam_eps must be finite and positive")
		if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
			raise ValueError("min_lr_ratio must be finite and in [0, 1]")
		if not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
			raise ValueError("max_grad_norm must be finite and nonnegative (0 disables clipping)")


@dataclass
class TrainState:
	step: int = 0
	tokens_seen: int = 0
	overfit_batch_start: int | None = None
	skipped_updates: int = 0
	grad_scaler_state: dict | None = None

	def __post_init__(self):
		for name in ("step", "tokens_seen", "skipped_updates"):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int) or value < 0:
				raise ValueError(f"{name} must be a nonnegative integer")
		if self.overfit_batch_start is not None:
			value = self.overfit_batch_start
			if isinstance(value, bool) or not isinstance(value, int) or value < 0:
				raise ValueError("overfit_batch_start must be a nonnegative integer or None")
		if self.grad_scaler_state is not None and not isinstance(self.grad_scaler_state, dict):
			raise TypeError("grad_scaler_state must be a dict or None")


def build_optimizer(model, config: TrainConfig):
	"""The only CLI recipe: Muon hidden matrices plus native auxiliary AdamW."""
	config.__post_init__()
	return MuonAdamW(build_parameter_groups(model, config.weight_decay),
		lr=config.learning_rate, weight_decay=config.weight_decay,
		momentum=config.muon_momentum, nesterov=True, ns_steps=config.muon_ns_steps,
		ns_dtype={"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[config.muon_ns_dtype],
		ns_compile=config.muon_compile,
		betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps,
		adamw_fused=config.adam_fused)


def learning_rate_at_step(config: TrainConfig, step: int) -> float:
	"""LR for a 1-based successful update; independent of an interruption's max_steps."""
	if step < 1:
		raise ValueError("LR step must be positive")
	if step <= config.warmup_steps:
		return config.learning_rate * step / config.warmup_steps
	if config.lr_schedule == "constant":
		return config.learning_rate
	start = max(1, config.warmup_steps)
	progress = min(1.0, (step - start) / (config.decay_steps - start))
	factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
	return config.learning_rate * factor


def validate_precision(device, precision: str) -> None:
	device = torch.device(device)
	if precision not in ("fp32", "bf16", "fp16"):
		raise ValueError("precision must be fp32, bf16 or fp16")
	if precision == "fp16" and device.type != "cuda":
		raise ValueError("fp16 requires CUDA; use fp32 or bf16 on CPU")
	if precision == "bf16" and device.type == "cuda":
		with torch.cuda.device(device):
			if not torch.cuda.is_bf16_supported(including_emulation=False):
				raise ValueError("This CUDA device lacks native BF16; choose fp32 or fp16")


def _autocast(device, precision):
	if precision == "fp32":
		return nullcontext()
	dtype = torch.bfloat16 if precision == "bf16" else torch.float16
	return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


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


def _optimizer_layout(model, optimizer) -> list[dict]:
	"""Canonical names and matrix views; group lengths alone cannot detect swaps."""
	names = {id(parameter): name for name, parameter in model.named_parameters()}
	layout = []
	for group in optimizer.param_groups:
		parameters = [{"name": names[id(parameter)], "shape": tuple(parameter.shape),
					   "dtype": str(parameter.dtype)} for parameter in group["params"]]
		if "param_names" in group and group["param_names"] != [item["name"] for item in parameters]:
			raise ValueError("Optimizer param_names do not match model parameters")
		view = group.get("muon_shape")
		description = {"algorithm": group.get("algorithm"),
			"muon_shape": None if view is None else tuple(view), "parameters": parameters}
		# Omit absent row splits so format-6 checkpoints without attention
		# (e.g. the smoke model) keep their existing layout representation.
		if group.get("muon_split_sizes") is not None:
			description["muon_split_sizes"] = tuple(group["muon_split_sizes"])
		layout.append(description)
	return layout


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int, *, precision: str = "fp32") -> float:
	"""Mean token loss on a fixed validation prefix, without repeating data."""
	if max_batches < 1:
		raise ValueError("max_batches must be positive")
	validate_precision(device, precision)
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
			with _autocast(device, precision):
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
		"format_version": 7,
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		**asdict(state),
		"train_config": asdict(config),
		"vocab_size": vocab_size,
		"model_config": _model_config(model),
		"optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
		"optimizer_layout": _optimizer_layout(model, optimizer),
		"qk_clip": qk_clip_metadata(model, threshold=config.qk_clip_threshold, chunk_size=config.qk_clip_chunk_size),
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
	for bitwise equivalence. Legacy AdamW checkpoints cannot resume Muon.
	"""
	checkpoint = torch.load(path, map_location="cpu", weights_only=True)
	version = checkpoint["format_version"]
	if version in (3, 4, 5):
		raise ValueError("Legacy AdamW checkpoints cannot resume the Muon recipe; start a new run")
	if version == 6:
		raise ValueError("Pre-clipping Muon checkpoints cannot resume QK-Clip; start a new run")
	if version != 7:
		raise ValueError("Expected trainer checkpoint format_version=7")
	state = TrainState(
		step=checkpoint["step"], tokens_seen=checkpoint["tokens_seen"],
		overfit_batch_start=checkpoint["overfit_batch_start"],
		skipped_updates=checkpoint.get("skipped_updates", 0),
		grad_scaler_state=checkpoint.get("grad_scaler_state"),
	)
	# Validate structure before modifying model weights or optimizer state.
	saved_config = checkpoint["train_config"]
	if isinstance(saved_config, dict):
		# The only added execution field has an explicit backwards-compatible default.
		saved_config = {"muon_compile": False, **saved_config}
	if not isinstance(saved_config, dict) or saved_config.keys() != asdict(config).keys():
		raise ValueError("Incomplete or unknown training recipe fields in checkpoint")
	try:
		TrainConfig(**saved_config)
	except (TypeError, ValueError) as error:
		raise ValueError("Invalid training recipe in checkpoint") from error
	for name in ("batch_size", "seq_len", "overfit_batch", "gradient_accumulation_steps",
				 "precision", "lr_schedule", "warmup_steps", "max_grad_norm", "adam_fused", "muon_ns_dtype", "muon_compile",
				 "qk_clip_threshold", "qk_clip_chunk_size"):
		if saved_config[name] != getattr(config, name):
			raise ValueError(f"Incompatible {name} in checkpoint")
	if config.lr_schedule == "cosine":
		for name in ("decay_steps", "min_lr_ratio"):
			if saved_config[name] != getattr(config, name):
				raise ValueError(f"Incompatible {name} in checkpoint")
	if config.precision == "fp16" and (state.step or state.skipped_updates) and not state.grad_scaler_state:
		raise ValueError("FP16 checkpoint is missing GradScaler state")
	if config.precision != "fp16" and state.grad_scaler_state is not None:
		raise ValueError("Unexpected GradScaler state for this precision")
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
	if checkpoint.get("qk_clip") != qk_clip_metadata(
		model, threshold=config.qk_clip_threshold, chunk_size=config.qk_clip_chunk_size,
	):
		raise ValueError("Incompatible QK-Clip policy or attention geometry in checkpoint")
	optimizer_state = checkpoint["optimizer"]
	if isinstance(optimizer, MuonAdamW):
		optimizer_state = optimizer._normalize_state_dict(optimizer_state)
	groups = optimizer_state["param_groups"]
	if len(groups) != len(optimizer.param_groups) or any(
		len(saved["params"]) != len(current["params"])
		for saved, current in zip(groups, optimizer.param_groups)
	):
		raise ValueError("Incompatible optimizer parameter groups; start a new run")
	layout = _optimizer_layout(model, optimizer)
	if checkpoint.get("optimizer_layout") != layout:
		raise ValueError("Incompatible optimizer parameter layout (names, shapes, dtype or matrix views); start a new run")
	current_groups = optimizer.state_dict()["param_groups"]
	for saved, current, description in zip(groups, current_groups, layout):
		view = saved.get("muon_shape")
		splits = saved.get("muon_split_sizes")
		if (saved["params"] != current["params"]
				or saved.get("algorithm") != description["algorithm"]
				or saved.get("ns_compile") != current.get("ns_compile")
				or (None if view is None else tuple(view)) != description["muon_shape"]
				or (None if splits is None else tuple(splits)) != description.get("muon_split_sizes")
				or ("param_names" in current and saved.get("param_names") != current["param_names"])):
			raise ValueError("Incompatible optimizer parameter layout in saved groups")
	if isinstance(optimizer, MuonAdamW):
		expected_groups = build_parameter_groups(model, saved_config["weight_decay"])
		for saved, expected in zip(groups, expected_groups):
			optimizer._validate_group({**saved, "params": expected["params"]})
			if saved["algorithm"] == "muon":
				expected.update(momentum=saved_config["muon_momentum"], nesterov=True,
					ns_compile=saved_config["muon_compile"],
					ns_steps=saved_config["muon_ns_steps"], ns_dtype={"bf16": torch.bfloat16,
						"fp32": torch.float32, "fp64": torch.float64}[saved_config["muon_ns_dtype"]])
			else:
				expected.update(betas=(saved_config["adam_beta1"], saved_config["adam_beta2"]),
					eps=saved_config["adam_eps"], fused=saved_config["adam_fused"])
			if any(saved[key] != value for key, value in expected.items()
					if key not in ("params", "param_names", "muon_shape", "muon_split_sizes")):
				raise ValueError("Optimizer hyperparameters differ from the saved training recipe")
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
	optimizer.load_state_dict(optimizer_state)
	optimizer.zero_grad(set_to_none=True)
	for name in ("learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps",
				 "muon_momentum", "muon_ns_steps"):
		setattr(config, name, saved_config[name])
	train_loader.load_state_dict(checkpoint["loaders"]["train"])
	val_loader.load_state_dict(checkpoint["loaders"]["val"])
	_restore_rng_state(checkpoint["rng_state"], next(model.parameters()).device)
	return state


def train(model, optimizer, train_loader, val_loader, config: TrainConfig,
		  device, *, state: TrainState | None = None, checkpoint_path=None, metrics=None,
		  profile_config=None, profile_dir=None) -> TrainState:
	"""Train until max_steps successful optimizer updates, including restored steps.

	Uses TokenDataLoader.next_batch()/reset(); the loader itself is unchanged.
	Exhaustion restarts the training stream, dropping an incomplete tail.
	Validation always uses its own loader and starts from offset zero.
	In overfit mode the batch is identified by its token offset; resume rereads
	that batch without advancing the restored loader cursor.
	Each update averages equal-size microbatch losses. QB updates per forward,
	so accumulation is not equivalent to concatenation for stateful MoE routing.
	FP16 overflow skips the whole optimizer and LR progression; consumed tokens and QB remain.
	An optional open MetricsWriter records scalar metrics without changing state.
	"""
	state = TrainState() if state is None else state
	# Preserve the direct API: a fresh optimizer supplies the schedule's peak LR.
	# The CLI constructs that optimizer from TrainConfig; resume restores the peak.
	if state.step == 0 and state.skipped_updates == 0:
		config.learning_rate = optimizer.param_groups[0]["lr"]
		config.__post_init__()
	device = torch.device(device)
	if device.type not in ("cpu", "cuda"):
		raise ValueError("This skeleton supports CPU or one CUDA device")
	validate_precision(device, config.precision)
	if config.precision != "fp32" and any(p.dtype != torch.float32 for p in model.parameters()):
		raise ValueError("Mixed precision requires FP32 model parameters; use autocast, not model.half()")
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

	model.train()
	optimizer.zero_grad(set_to_none=True)
	scaler = torch.amp.GradScaler(device.type, enabled=config.precision == "fp16")
	if state.grad_scaler_state is not None:
		scaler.load_state_dict(state.grad_scaler_state)
	window = TrainingWindow()
	fixed_batch = None
	if config.overfit_batch and state.overfit_batch_start is not None:
		x, y = train_loader.dataset.get_batch(
			state.overfit_batch_start, config.batch_size, config.seq_len,
		)
		fixed_batch = (x.to(device), y.to(device))

	monitor = TrainingMonitor(device, metrics)
	with TrainingProfiler(model, device, state, profile_config,
						  directory=profile_dir, metrics=metrics) as profiling, \
		 MoEDiagnostics(model, enabled=metrics is not None) as moe, \
		 QKClip(model, threshold=config.qk_clip_threshold, chunk_size=config.qk_clip_chunk_size) as qk_clip:
		monitor.begin(state)
		monitor.report("session_start", state,
					   discarded_tail_bytes=metrics.discarded_tail_bytes if metrics is not None else 0,
					   profiling=profiling.describe())
		monitor.resume_train()
		consecutive_skips = 0
		while state.step < config.max_steps:
			qk_clip.reset()
			step = state.step + 1
			lr = learning_rate_at_step(config, step)
			for group in optimizer.param_groups:
				group["lr"] = lr
			update_loss = 0.0
			update_tokens = 0
			for _ in range(config.gradient_accumulation_steps):
				with profiling.region("train/data"):
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
				with profiling.region("train/forward"):
					with _autocast(device, config.precision):
						_, loss = model(x, targets=y)
					value = loss.item()
				if not math.isfinite(value):
					raise FloatingPointError(f"Non-finite training loss at step {step}")
				with profiling.region("train/backward"):
					scaler.scale(loss / config.gradient_accumulation_steps).backward()
				update_loss += value / config.gradient_accumulation_steps
				update_tokens += y.numel()
				state.tokens_seen += y.numel()
			with profiling.region("train/optimizer"):
				scaler.unscale_(optimizer)
				grad_norm = torch.nn.utils.clip_grad_norm_(
					model.parameters(), config.max_grad_norm or float("inf"),
					error_if_nonfinite=not scaler.is_enabled(),
				).item()
				previous_scale = scaler.get_scale()
				qk_clip.validate()
				scaler.step(optimizer)
				scaler.update()
				state.grad_scaler_state = scaler.state_dict() if scaler.is_enabled() else None
				skipped = scaler.get_scale() < previous_scale
				with profiling.region("train/qk_clip"):
					if skipped:
						qk_clip.reset()
					else:
						qk_clip.apply()
				optimizer.zero_grad(set_to_none=True)
			window.add(update_loss, update_tokens, grad_norm,
					   skipped=skipped, max_grad_norm=config.max_grad_norm)
			if skipped:
				state.skipped_updates += 1
				consecutive_skips += 1
				monitor.pause_train()
				with profiling.region("train/report"):
					monitor.report("overflow", state, attempted_step=step, lr=lr,
								   loss_scale=scaler.get_scale(), train_loss=update_loss,
								   grad_norm=grad_norm if math.isfinite(grad_norm) else None,
								   grad_norm_finite=math.isfinite(grad_norm), tokens=update_tokens,
								   message=f"step {step} | FP16 overflow: update skipped"
								   f" | loss_scale {scaler.get_scale():g} | skipped {state.skipped_updates}")
				if profiling.running:
					started = monitor.now()
					profiling.step()
					monitor.finish_phase("profiling", started)
				if consecutive_skips >= 32:
					raise FloatingPointError("32 consecutive FP16 updates skipped; check numerics or use FP32")
				monitor.resume_train()
				continue
			consecutive_skips = 0
			state.step = step
			final = step == config.max_steps
			log_now = step % config.log_interval == 0 or final
			eval_now = step % config.eval_interval == 0 or final
			save_now = checkpoint_path is not None and (
				step % config.checkpoint_interval == 0 or final
			)
			if log_now or eval_now or save_now or profiling.running:
				monitor.pause_train()
				if log_now:
					interval = monitor.interval_summary(window)
					with profiling.region("train/report"):
						monitor.report("train", state, **window.summary(lr), **interval,
									   qk_clip=qk_clip.summary(reset=True),
									   moe_diagnostics=moe,
									   message=f"step {step} | train loss {window.loss_sum / window.tokens:.4f}"
									   f" | lr {lr:.6g} | grad_norm {grad_norm:.4f}"
									   f" | tokens_seen {state.tokens_seen:,}"
									   f" | tokens/s {interval['train_tokens_per_second'] or 0:,.0f}")
					window = TrainingWindow()
					monitor.window_train_seconds = 0.0
				if eval_now:
					started = monitor.now()
					with profiling.region("train/eval"):
						val_loss = evaluate(model, val_loader, device, config.eval_batches, precision=config.precision)
					elapsed = monitor.finish_phase("eval", started)
					with profiling.region("train/report"):
						monitor.report("val", state, val_loss=val_loss, eval_seconds=elapsed,
									   message=f"step {step} | val loss {val_loss:.4f}"
									   f" | tokens_seen {state.tokens_seen:,}")
				if save_now:
					started = monitor.now()
					with profiling.region("train/checkpoint"):
						save_checkpoint(checkpoint_path, model, optimizer, state, config,
										vocab_size=train_loader.dataset.vocab_size,
										train_loader=train_loader, val_loader=val_loader)
					elapsed = monitor.finish_phase("checkpoint", started)
					with profiling.region("train/report"):
						monitor.report("checkpoint", state, checkpoint_seconds=elapsed)
				if profiling.running:
					started = monitor.now()
					profiling.step()
					monitor.finish_phase("profiling", started)
				monitor.resume_train()
		if profiling.running:
			started = monitor.now()
			profiling.finish()
			monitor.finish_phase("profiling", started)
		monitor.report("session_end", state)
		return state
