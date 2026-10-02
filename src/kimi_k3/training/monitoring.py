"""Interval timing and event-boundary memory samples for one training session."""

from time import perf_counter

import psutil
import torch


def _rate(amount, seconds):
	return amount / seconds if seconds > 0 else None


class TrainingMonitor:
	"""Measure completed phases, without synchronizing every optimizer update.

	RAM is sampled RSS. CUDA peaks come from the allocator's counters, reset
	once per session; they include temporary allocations between samples.
	"""

	def __init__(self, device, metrics=None):
		self.device = torch.device(device)
		self.metrics = metrics
		self._process = None
		if metrics is not None:
			try:
				self._process = psutil.Process()
			except (psutil.Error, OSError, NotImplementedError):
				pass  # Process inspection may be unavailable in a container.
		self._rss_peak = None
		self.seconds = dict.fromkeys(("train", "eval", "checkpoint", "reporting", "profiling"), 0.0)
		self.window_train_seconds = 0.0
		if metrics is not None and self.device.type == "cuda":
			torch.cuda.reset_peak_memory_stats(self.device)

	@staticmethod
	def now():
		return perf_counter()

	def synchronize(self):
		if self.device.type == "cuda":
			torch.cuda.synchronize(self.device)

	def begin(self, state):
		self.synchronize()
		self._session_started = self.now()
		self._tokens_at_start = state.tokens_seen

	def resume_train(self):
		self.synchronize()
		self._train_started = self.now()

	def pause_train(self):
		self.synchronize()
		elapsed = self.now() - self._train_started
		self.seconds["train"] += elapsed
		self.window_train_seconds += elapsed

	def finish_phase(self, phase, started):
		self.synchronize()
		elapsed = self.now() - started
		self.seconds[phase] += elapsed
		return elapsed

	def interval_summary(self, window):
		return {
			"train_seconds": self.window_train_seconds,
			"train_tokens_per_second": _rate(window.tokens, self.window_train_seconds),
			"seconds_per_attempt": _rate(self.window_train_seconds, window.attempts),
		}

	def session_summary(self, state, *, now=None):
		elapsed = (self.now() if now is None else now) - self._session_started
		tokens = state.tokens_seen - self._tokens_at_start
		return {
			"elapsed_seconds": elapsed, "tokens": tokens,
			"tokens_per_second": _rate(tokens, elapsed),
			**{phase + "_seconds": seconds for phase, seconds in self.seconds.items()},
		}

	def sample_memory(self):
		rss = None
		if self._process is not None:
			try:
				rss = self._process.memory_info().rss
			except (psutil.Error, OSError, NotImplementedError):
				pass
		if rss is not None:
			self._rss_peak = max(self._rss_peak or 0, rss)
		memory = {
			"ram_rss_bytes": rss, "ram_rss_sampled_peak_bytes": self._rss_peak,
			"cuda_allocated_bytes": None, "cuda_reserved_bytes": None,
			"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
		}
		if self.device.type == "cuda":
			memory.update({
				"cuda_allocated_bytes": torch.cuda.memory_allocated(self.device),
				"cuda_reserved_bytes": torch.cuda.memory_reserved(self.device),
				"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
				"cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
			})
		return memory

	def report(self, event, state, *, message=None, moe_diagnostics=None, **values):
		started = self.now()
		# Snapshot timing before this report's own sampling/printing/writing.
		# Its measured cost is included in the following reports.
		if self.metrics is not None:
			session = self.session_summary(state, now=started)
			if moe_diagnostics is not None:
				values["moe"] = moe_diagnostics.snapshot()
			memory = self.sample_memory()
		if message is not None:
			print(message, flush=True)
		if self.metrics is not None:
			self.metrics.write(event, state, session=session, memory=memory, **values)
		self.seconds["reporting"] += self.now() - started
