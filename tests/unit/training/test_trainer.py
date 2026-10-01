from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from kimi_k3.training.trainer import TrainConfig, train

VOCAB_SIZE = 16
BATCH_SIZE, SEQ_LEN = 2, 4


class _CountingLoader:
	"""Yields a new deterministic batch on every call, advancing a token cursor and counting the calls."""

	def __init__(self):
		self.batch_size, self.seq_len = BATCH_SIZE, SEQ_LEN
		self.dataset = SimpleNamespace(vocab_size=VOCAB_SIZE, total_tokens=10_000)
		self.calls = 0
		self.position = 0

	def next_batch(self):
		self.calls += 1
		self.position += BATCH_SIZE * SEQ_LEN
		tokens = (torch.arange(BATCH_SIZE * SEQ_LEN + 1) + self.calls) % VOCAB_SIZE
		return tokens[:-1].view(BATCH_SIZE, SEQ_LEN), tokens[1:].view(BATCH_SIZE, SEQ_LEN)

	def reset(self):
		self.calls = 0
		self.position = 0


class _RecordingLM(nn.Module):
	"""Token-local LM that records every training input it receives."""

	def __init__(self):
		super().__init__()
		self.embedding = nn.Embedding(VOCAB_SIZE, 8)
		self.head = nn.Linear(8, VOCAB_SIZE)
		self.train_inputs = []

	def forward(self, x, targets=None):
		if self.training:
			self.train_inputs.append(x.clone())
		logits = self.head(self.embedding(x))
		loss = None if targets is None else F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
		return logits, loss


def _run(overfit_batch, max_steps=5, log_interval=None):
	model = _RecordingLM()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
	train_loader, val_loader = _CountingLoader(), _CountingLoader()
	config = TrainConfig(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, max_steps=max_steps, log_interval=log_interval or max_steps,
						 eval_interval=max_steps, eval_batches=1, checkpoint_interval=max_steps,
						 overfit_batch=overfit_batch)
	state = train(model, optimizer, train_loader, val_loader, config, "cpu")
	return model, train_loader, state


def test_overfit_batch_repeats_the_first_training_batch():
	model, train_loader, state = _run(overfit_batch=True)

	assert train_loader.calls == 1
	assert len(model.train_inputs) == 5
	for x in model.train_inputs[1:]:
		torch.testing.assert_close(x, model.train_inputs[0], rtol=0, atol=0)
	assert state.tokens_seen == 5 * BATCH_SIZE * SEQ_LEN


def test_without_overfit_batch_every_step_reads_a_new_batch():
	model, train_loader, _ = _run(overfit_batch=False)

	assert train_loader.calls == 5
	assert not torch.equal(model.train_inputs[0], model.train_inputs[1])


def test_overfit_batch_drives_training_loss_down(capsys):
	_run(overfit_batch=True, max_steps=30, log_interval=1)

	train_losses = [float(line.split("train loss ")[1].split(" |")[0])
					for line in capsys.readouterr().out.splitlines() if "train loss" in line]
	assert len(train_losses) == 30
	# Memorizing one batch should take the loss well below its starting value.
	assert train_losses[-1] < 0.5 * train_losses[0]
