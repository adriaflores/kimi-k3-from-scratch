"""Run from the repository root: uv run python scripts/train.py --help."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Also support uv projects that have not installed their src package yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from kimi_k3.training.trainer import TrainConfig, TrainState, load_checkpoint, train


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


def build_model(vocab_size, *, smoke_test):
	if smoke_test:
		return SmokeTestLM(vocab_size)
	# Replace this branch with KimiK3Config(...) and KimiK3(config) later.
	raise NotImplementedError(
		"Kimi K3 is not implemented yet. Use --smoke-test to check the trainer."
	)


def parse_args():
	defaults = TrainConfig()
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path, required=True)
	parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
	parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
	parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
	parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
	parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
	parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
	parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
	parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
	parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
	parser.add_argument("--checkpoint", type=Path, default=Path("results/training/last.pt"))
	parser.add_argument("--resume", type=Path)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--smoke-test", action="store_true", help="Use a toy model, not Kimi K3")
	return parser.parse_args()


def main():
	args = parse_args()
	config = TrainConfig(
		batch_size=args.batch_size, seq_len=args.seq_len,
		learning_rate=args.learning_rate, weight_decay=args.weight_decay,
		max_steps=args.max_steps, log_interval=args.log_interval,
		eval_interval=args.eval_interval, eval_batches=args.eval_batches,
		checkpoint_interval=args.checkpoint_interval,
	)
	device = torch.device(args.device)
	if device.type not in ("cpu", "cuda"):
		raise ValueError("Choose cpu, cuda, or cuda:<index>")
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable")
	torch.manual_seed(args.seed)
	train_data = ShardedTokenDataset(args.data_dir, split="train")
	val_data = ShardedTokenDataset(args.data_dir, split="val")
	train_loader = TokenDataLoader(train_data, config.batch_size, config.seq_len)
	val_loader = TokenDataLoader(val_data, config.batch_size, config.seq_len)
	model = build_model(train_data.vocab_size, smoke_test=args.smoke_test).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
	)
	state = TrainState()
	if args.resume:
		state = load_checkpoint(args.resume, model, optimizer, config,
								vocab_size=train_data.vocab_size)
		# Keep saved config metadata consistent with restored optimizer settings.
		config.learning_rate = optimizer.param_groups[0]["lr"]
		config.weight_decay = optimizer.param_groups[0]["weight_decay"]
		print(f"Resumed step {state.step}; tokens_seen={state.tokens_seen:,}; "
			  f"optimizer LR={config.learning_rate:g}. "
			  "Data restarts at offset 0; RNG is not restored. Resume is not exact.")
	print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
	if args.smoke_test:
		print("SMOKE TEST: toy model only; these are not Kimi K3 results.")
	train(model, optimizer, train_loader, val_loader, config, device,
		  state=state, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
	main()
