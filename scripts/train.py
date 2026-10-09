"""Run from the repository root: uv run python scripts/train.py --help."""

import argparse
import os
import random
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Match the model's imports when running the script directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from src.kimi_k3.model import KimiK3, KimiK3Config
from src.kimi_k3.training.metrics import MetricsWriter
from src.kimi_k3.training.profiling import ProfileConfig
from src.kimi_k3.training.run import start_run
from src.kimi_k3.training.trainer import (
	TrainConfig,
	TrainState,
	build_optimizer,
	load_checkpoint,
	train,
	validate_precision,
)
from src.kimi_k3.utils import load_config


class SmokeTestLM(nn.Module):
	"""Token-local toy model for plumbing checks only; this is not Kimi K3."""

	def __init__(self, vocab_size):
		super().__init__()
		self.config = {"model_type": "smoke_test_lm", "vocab_size": vocab_size, "hidden_size": 32}
		self.embedding = nn.Embedding(vocab_size, 32)
		self.head = nn.Linear(32, vocab_size)

	def forward(self, x, targets=None):
		logits = self.head(self.embedding(x))
		loss = None if targets is None else F.cross_entropy(
			logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
		)
		return logits, loss


def build_model(vocab_size, seq_len, *, model_config_path=None, smoke_test=False):
	if smoke_test:
		return SmokeTestLM(vocab_size)
	if model_config_path is None:
		raise ValueError("Provide --model-config or use --smoke-test")
	config = load_config(model_config_path, KimiK3Config)
	if config.vocab_size != vocab_size:
		raise ValueError(
			f"Model vocab_size={config.vocab_size} does not match dataset vocab_size={vocab_size}"
		)
	if seq_len > config.max_seq_len:
		raise ValueError(f"seq_len={seq_len} exceeds model max_seq_len={config.max_seq_len}")
	# The fused kernel pads the last chunk itself; only the PyTorch reference needs whole chunks.
	if ("kda" in config.layer_types and config.kda_implementation == "chunkwise"
			and seq_len % config.kda_chunk_size != 0):
		raise ValueError(
			f"seq_len={seq_len} must be divisible by kda_chunk_size={config.kda_chunk_size}"
		)
	return KimiK3(config)


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path, required=True)
	parser.add_argument("--training-config", type=Path, help="Complete TrainConfig YAML; CLI values override it")
	for name in ("batch_size", "seq_len", "max_steps", "log_interval", "eval_interval",
				 "eval_batches", "checkpoint_interval", "gradient_accumulation_steps",
				 "warmup_steps", "decay_steps", "seed", "muon_ns_steps", "qk_clip_chunk_size"):
		parser.add_argument("--" + name.replace("_", "-"), type=int)
	for name in ("learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps",
				 "min_lr_ratio", "max_grad_norm", "muon_momentum", "qk_clip_threshold"):
		parser.add_argument("--" + name.replace("_", "-"), type=float)
	parser.add_argument("--adam-fused", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--muon-compile", action=argparse.BooleanOptionalAction, default=None,
						help="Compile Newton-Schulz on CUDA; CPU keeps the eager reference")
	parser.add_argument("--muon-ns-dtype", choices=("bf16", "fp32", "fp64"))
	parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
	parser.add_argument("--lr-schedule", choices=("constant", "cosine"))
	parser.add_argument("--run-dir", type=Path, help="Run folder; defaults to a unique folder under results/training/runs")
	parser.add_argument("--checkpoint", type=Path, help="Checkpoint output; defaults to last.pt in the run folder")
	parser.add_argument("--resume", type=Path)
	parser.add_argument("--overfit-batch", action=argparse.BooleanOptionalAction, default=None,
						help="Repeat one fixed training batch")
	parser.add_argument("--deterministic", action="store_true", help="Require deterministic PyTorch kernels")
	parser.add_argument("--profile", action="store_true", help="Capture one short time/memory profiler window")
	parser.add_argument("--profile-wait", type=int, default=1, help="Unprofiled optimizer attempts before warmup")
	parser.add_argument("--profile-warmup", type=int, default=1, help="Profiler warmup attempts (>=1)")
	parser.add_argument("--profile-active", type=int, default=1, help="Recorded optimizer attempts (>=1)")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	model_selection = parser.add_mutually_exclusive_group(required=True)
	model_selection.add_argument("--model-config", type=Path, help="Path to a Kimi K3 model YAML")
	model_selection.add_argument("--smoke-test", action="store_true", help="Use a toy model, not Kimi K3")
	return parser.parse_args(argv)


def build_train_config(args):
	config = load_config(args.training_config, TrainConfig) if args.training_config else TrainConfig()
	overrides = {field.name: getattr(args, field.name) for field in fields(TrainConfig)
				 if getattr(args, field.name, None) is not None}
	return replace(config, **overrides)


def main():
	args = parse_args()
	config = build_train_config(args)
	profile_config = ProfileConfig(args.profile_wait, args.profile_warmup, args.profile_active) if args.profile else None
	device = torch.device(args.device)
	if device.type not in ("cpu", "cuda"):
		raise ValueError("Choose cpu, cuda, or cuda:<index>")
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable")
	validate_precision(device, config.precision)
	if args.deterministic:
		os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
		torch.use_deterministic_algorithms(True)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	train_data = ShardedTokenDataset(args.data_dir, split="train")
	val_data = ShardedTokenDataset(args.data_dir, split="val")
	if train_data.vocab_size != val_data.vocab_size:
		raise ValueError("Training and validation vocab_size must match")
	train_loader = TokenDataLoader(train_data, config.batch_size, config.seq_len)
	val_loader = TokenDataLoader(val_data, config.batch_size, config.seq_len)
	model = build_model(
		train_data.vocab_size, config.seq_len,
		model_config_path=args.model_config, smoke_test=args.smoke_test,
	).to(device)
	optimizer = build_optimizer(model, config)
	state = TrainState()
	if args.resume:
		state = load_checkpoint(args.resume, model, optimizer, config,
								vocab_size=train_data.vocab_size,
								train_loader=train_loader, val_loader=val_loader)
		print(f"Resumed step {state.step}; tokens_seen={state.tokens_seen:,}; "
			  f"base LR={config.learning_rate:g}; saved LR={optimizer.param_groups[0]['lr']:g}. "
			  f"Data cursor={train_loader.position:,}; RNG restored.")
	run = start_run(
		model=model, optimizer=optimizer, config=config, state=state,
		train_loader=train_loader, val_loader=val_loader,
		repo_root=Path(__file__).resolve().parents[1], run_dir=args.run_dir,
		resume_path=args.resume, checkpoint_path=args.checkpoint,
	)
	print(f"Run: {run.directory} | session: {run.session_id}")
	print(f"Manifest: {run.directory / 'run.json'} | checkpoint: {run.checkpoint_path}")
	if profile_config is not None:
		print(f"Profile: {run.directory / 'profiles' / run.session_id} | "
			  f"wait={profile_config.wait}, warmup={profile_config.warmup}, active={profile_config.active} attempts")
	print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
	tokens_per_update = config.batch_size * config.seq_len * config.gradient_accumulation_steps
	print(f"Muon momentum={config.muon_momentum} | Nesterov=True"
		  f" | NS={config.muon_ns_steps} x {config.muon_ns_dtype}")
	print(f"Muon NS backend={'compiled' if config.muon_compile and device.type == 'cuda' else 'eager'}"
		  f" | muon_compile={config.muon_compile}")
	print(f"Auxiliary AdamW betas=({config.adam_beta1}, {config.adam_beta2}) | eps={config.adam_eps:g}"
		  f" | weight_decay={config.weight_decay:g} | precision={config.precision} | fused={config.adam_fused}")
	for algorithm in ("muon", "adamw"):
		parameters = [p for group in optimizer.param_groups if group["algorithm"] == algorithm for p in group["params"]]
		print(f"{algorithm}: {len(parameters)} parameter tensors, {sum(p.numel() for p in parameters):,} elements")
	print(f"microbatch={config.batch_size} x {config.seq_len}"
		  f" | accumulation={config.gradient_accumulation_steps} | tokens/update={tokens_per_update:,}"
		  f" | target updates={config.max_steps:,}"
		  f" | remaining nominal tokens={(config.max_steps - state.step) * tokens_per_update:,}")
	print(f"LR={config.learning_rate:g} | schedule={config.lr_schedule}"
		  f" | warmup={config.warmup_steps} | decay end={config.decay_steps}"
		  f" | min LR ratio={config.min_lr_ratio:g} | clipping={config.max_grad_norm:g}")
	print(f"QK-Clip: MLA threshold={config.qk_clip_threshold:g} | score=QK/sqrt(d)"
		  f" | query chunk={config.qk_clip_chunk_size}")
	if args.smoke_test:
		print("SMOKE TEST: toy model only; these are not Kimi K3 results.")
	else:
		print(f"KimiK3 | model config: {args.model_config}")
	with MetricsWriter(run) as metrics:
		print(f"Metrics: {metrics.path}")
		train(model, optimizer, train_loader, val_loader, config, device,
			  state=state, checkpoint_path=run.checkpoint_path, metrics=metrics,
			  profile_config=profile_config)


if __name__ == "__main__":
	main()
