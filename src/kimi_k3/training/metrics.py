"""Small, append-only JSONL records; no tensors, hooks or device synchronization."""

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime


class MetricsWriter:
	"""One writer per CLI session, appending to the run's metrics.jsonl."""

	def __init__(self, run):
		self.path = run.directory / "metrics.jsonl"
		self.run_id = run.run_id
		self.session_id = run.session_id
		self.discarded_tail_bytes = 0
		self._file = None

	def __enter__(self):
		# A newline commits a record. Discard an interrupted final write, using
		# bounded reads rather than loading the training history into memory.
		with self.path.open("a+b") as stream:
			stream.seek(0, 2)
			end = stream.tell()
			boundary = end
			if end:
				stream.seek(end - 1)
				if stream.read(1) != b"\n":
					boundary = 0
					offset = end
					while offset:
						start = max(0, offset - 4096)
						stream.seek(start)
						newline = stream.read(offset - start).rfind(b"\n")
						if newline >= 0:
							boundary = start + newline + 1
							break
						offset = start
					stream.truncate(boundary)
			self.discarded_tail_bytes = end - boundary
		self._file = self.path.open("a", encoding="utf-8", newline="\n")
		return self

	def write(self, event, state, **values):
		record = {
			**values, "schema_version": 1, "event": event,
			"timestamp": datetime.now(UTC).isoformat(),
			"run_id": self.run_id, "session_id": self.session_id,
			"step": state.step, "tokens_seen": state.tokens_seen,
			"skipped_updates": state.skipped_updates,
		}
		line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
		self._file.write(line)
		self._file.flush()

	def __exit__(self, exc_type, exc_value, traceback):
		self._file.close()


@dataclass
class TrainingWindow:
	"""Detached scalar statistics between train logs; reset on a new session.

	Loss and token counts include skipped attempts, whose data was consumed.
	Gradient statistics and clipping fractions cover successful updates only.
	"""

	loss_sum: float = 0.0
	tokens: int = 0
	attempts: int = 0
	updates: int = 0
	skips: int = 0
	grad_norm_sum: float = 0.0
	grad_norm_max: float | None = None
	nonfinite_grad_norms: int = 0
	clipped_updates: int = 0

	def add(self, loss, tokens, grad_norm, *, skipped, max_grad_norm):
		self.loss_sum += loss * tokens
		self.tokens += tokens
		self.attempts += 1
		self.skips += int(skipped)
		self.nonfinite_grad_norms += int(not math.isfinite(grad_norm))
		if not skipped:
			self.updates += 1
			self.grad_norm_sum += grad_norm
			self.grad_norm_max = max(self.grad_norm_max or 0.0, grad_norm)
			self.clipped_updates += int(max_grad_norm > 0 and grad_norm > max_grad_norm)

	def summary(self, lr):
		return {
			"train_loss": self.loss_sum / self.tokens,
			"lr": lr, "tokens": self.tokens,
			"attempted_updates": self.attempts, "updates": self.updates,
			"skipped_updates_in_window": self.skips,
			"grad_norm_mean": self.grad_norm_sum / self.updates if self.updates else None,
			"grad_norm_max": self.grad_norm_max,
			"clipped_fraction": self.clipped_updates / self.updates if self.updates else None,
			"nonfinite_grad_norms": self.nonfinite_grad_norms,
		}
