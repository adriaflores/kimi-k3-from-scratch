"""One bounded profiler window; normal training creates no Kineto profiler."""

import gzip
import json
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ProfileConfig:
	"""Counts optimizer attempts in this invocation, including FP16 skips."""

	wait: int = 1
	warmup: int = 1
	active: int = 1

	def __post_init__(self):
		for name, minimum in (("wait", 0), ("warmup", 1), ("active", 1)):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
				raise ValueError(f"profile {name} must be an integer >= {minimum}")

	@property
	def total_attempts(self):
		return self.wait + self.warmup + self.active


class TrainingProfiler:
	"""Capture time/allocation events and label attention/FFN forward families.

	Temporary hooks annotate eager modules only while this window is open.
	No shapes, stacks or input/output tensors are retained by Python hooks.
	"""

	def __init__(self, model, device, state, config=None, *, directory=None, metrics=None):
		self.model, self.device, self.state = model, torch.device(device), state
		self.config = config
		self.directory = None
		self.run_id = getattr(metrics, "run_id", None)
		self.session_id = getattr(metrics, "session_id", None)
		self._prof = None
		self._handles = []
		self._scopes = {}
		self.attempts = 0
		self.trace_available = False
		self.regions = []
		self.cuda_timings = False
		self.stop_reason = None
		if config is not None:
			if directory is None and metrics is not None:
				directory = metrics.path.parent / "profiles" / metrics.session_id
			if directory is None:
				raise ValueError("Profiling requires a metrics writer or an explicit profile_dir")
			self.directory = Path(directory).resolve()

	@property
	def running(self):
		return self._prof is not None

	def _counters(self):
		return {name: getattr(self.state, name) for name in ("step", "tokens_seen", "skipped_updates")}

	def describe(self):
		if self.config is None:
			return None
		return {"directory": str(self.directory), **asdict(self.config),
				"profile_memory": True, "record_shapes": False, "with_stack": False,
				"activities": [activity.name for activity in self.activities],
				"cuda_kernel_tracing_supported": self.cuda_kernel_tracing_supported}

	def __enter__(self):
		if self.config is None:
			return self
		self.directory.mkdir(parents=True, exist_ok=False)
		self.start_state = self.end_state = self._counters()
		self.activities = [torch.profiler.ProfilerActivity.CPU]
		self.cuda_kernel_tracing_supported = False
		if self.device.type == "cuda":
			# Include CUDA even without CUPTI: PyTorch can provide legacy
			# device-event timings in its tables, without on-device trace lanes.
			self.activities.append(torch.profiler.ProfilerActivity.CUDA)
			self.cuda_kernel_tracing_supported = (
				torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities())
		self._write_metadata("pending")
		self._prof = torch.profiler.profile(
			activities=self.activities,
			schedule=torch.profiler.schedule(**asdict(self.config), repeat=1),
			on_trace_ready=self._export,
			profile_memory=True, record_shapes=False, with_stack=False,
			acc_events=True,
		)
		self._prof.preset_metadata_json("training_profile", json.dumps({
			"run_id": self.run_id, "session_id": self.session_id,
			"schedule": asdict(self.config), "start_state": self.start_state,
		}))
		self._prof.__enter__()
		try:
			self._install_hooks()
		except BaseException:
			self.finish(error=True)
			raise
		return self

	def region(self, name):
		if self.running and self._prof.current_action != torch.profiler.ProfilerAction.NONE:
			return torch.profiler.record_function(name)
		return nullcontext()

	def _install_hooks(self):
		for block in getattr(self.model, "layers", ()):
			for attribute in ("attn", "ffn"):
				module = getattr(block, attribute, None)
				if module is None:
					continue
				label = "model/" + type(module).__name__
				self._scopes[module] = []

				def before(module, args, label=label):
					scope = self.region(label) if module.training else nullcontext()
					scope.__enter__()
					self._scopes[module].append(scope)

				def after(module, args, output):
					if self._scopes[module]:
						self._scopes[module].pop().__exit__(None, None, None)

				self._handles.append(module.register_forward_pre_hook(before))
				self._handles.append(module.register_forward_hook(after, always_call=True))

	def _remove_hooks(self):
		for handle in self._handles:
			handle.remove()
		self._handles.clear()
		for scopes in self._scopes.values():
			while scopes:
				scopes.pop().__exit__(None, None, None)
		self._scopes.clear()

	def step(self):
		"""Advance once after ALL work for an attempt, including auxiliary phases."""
		if not self.running:
			return
		self.attempts += 1
		self.end_state = self._counters()
		self._prof.step()
		if self.attempts == self.config.total_attempts:
			self.finish()

	def _export(self, prof):
		raw = self.directory / "trace.json"
		prof.export_chrome_trace(str(raw))
		# Export to a closed path first: reopening an open NamedTemporaryFile
		# for native trace writing is not portable to Windows.
		with raw.open("rb") as source, gzip.open(self.directory / "trace.json.gz", "wb", compresslevel=1) as target:
			shutil.copyfileobj(source, target, length=1024 * 1024)
		raw.unlink()
		averages = prof.key_averages()
		self.cuda_timings = any(event.device_time_total > 0 for event in averages)
		tables = [
			"CPU operators: self time\n" + averages.table(sort_by="self_cpu_time_total", row_limit=20),
			"CPU tensor memory: net self bytes, not peak RAM\n"
			+ averages.table(sort_by="self_cpu_memory_usage", row_limit=20),
		]
		if self.device.type == "cuda":
			tables.extend([
				"CUDA operators: self time\n" + averages.table(sort_by="self_device_time_total", row_limit=20),
				"CUDA tensor memory: net self bytes, not peak VRAM\n"
				+ averages.table(sort_by="self_device_memory_usage", row_limit=20),
			])
		self.regions = [{
			"name": event.key, "calls": event.count,
			"cpu_total_time_us": event.cpu_time_total,
			"device_total_time_us": event.device_time_total,
		} for event in averages if event.key.startswith(("train/", "model/"))]
		tables.append("Phase/module regions: inclusive times overlap; do not sum nested regions\n"
					  + json.dumps(self.regions, indent=2))
		(self.directory / "summary.txt").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
		self.trace_available = True

	def _write_metadata(self, status):
		active_completed = max(0, self.attempts - self.config.wait - self.config.warmup)
		record = {
			"schema_version": 1, "status": status, "stop_reason": self.stop_reason,
			"run_id": self.run_id, "session_id": self.session_id,
			"device": str(self.device), "options": self.describe(),
			"attempts_completed": self.attempts, "active_attempts_completed": active_completed,
			"start_state": self.start_state, "end_state": self.end_state,
			"trace": "trace.json.gz" if self.trace_available else None,
			"summary": "summary.txt" if self.trace_available else None,
			"cuda_timings_captured": self.cuda_timings, "regions": self.regions,
		}
		temporary = self.directory / "profile.json.tmp"
		temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
		temporary.replace(self.directory / "profile.json")

	def finish(self, *, error=False):
		if not self.running:
			return
		complete = self.attempts == self.config.total_attempts
		self.stop_reason = "window_complete" if complete else "training_error" if error else "training_end"
		if not complete:
			self.end_state = self._counters()
		profiler, self._prof = self._prof, None
		try:
			self._remove_hooks()
			profiler.__exit__(None, None, None)
		finally:
			self._remove_hooks()
		status = "complete" if complete else "partial" if self.trace_available else "no_capture"
		self._write_metadata(status)

	def __exit__(self, exc_type, exc_value, traceback):
		self.finish(error=exc_type is not None)
