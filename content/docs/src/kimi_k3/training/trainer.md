---
title: "TrainConfig, TrainState and training loop"
description: "Executes single-device causal-LM updates, evaluation, clipped hybrid optimization and resumable checkpointing."
sourcePath: "src/kimi_k3/training/trainer.py"
status: published
---

`trainer.py` is the **execution authority** for one-device Kimi K3 pretraining. It constructs the hybrid optimizer, defines the update schedule, drives model forward/backward passes and coordinates QK-Clip, validation, checkpointing, metrics and bounded profiling [source L1–L23](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L1-L23). The input `model(x,targets=y)` must return `(logits, mean_token_loss)`, and each batch starts with no cached attention or recurrent state. This is deliberately distinct from the run-provenance writer (`training/run.py`) and the command-line parser in `scripts/`.

## Configuration and persistent counters

`TrainConfig` [source L26–L111](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L26-L111) is a dataclass that supplies batch size and sequence length, peak LR and weight decay, max successful updates, logging/evaluation/checkpoint intervals, gradient accumulation count, precision, AdamW and Muon options, QK-Clip policy, scheduler, clipping threshold and seed. `__post_init__` validates positive integer dimensions and intervals, numeric ranges and allowed strings. Precision is one of `fp32`, `bf16`, `fp16`; LR schedule is `constant` or `cosine`; `max_grad_norm=0` disables ordinary gradient-norm clipping. `muon_ns_dtype` is independently chosen from BF16/FP32/FP64 even when model parameters remain FP32.

`TrainState` [source L113–L131](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L113-L131) is smaller: it holds successful `step`, cumulative `tokens_seen`, optional fixed overfit-batch offset, cumulative `skipped_updates` and optional FP16 GradScaler state. These are **different clocks**: `step` counts successful optimizer updates only, whereas `tokens_seen` increments for every consumed microbatch, including later-overflowed attempts. A resumed session starts with the restored counters rather than restarting them at zero.

## Optimizer assembly and LR at a successful step

`build_optimizer(model,config)` [source L134–L145](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L134-L145) creates `MuonAdamW` using `build_parameter_groups`, Nesterov Muon for eligible hidden matrices and native AdamW for auxiliary parameters. The one optimizer exposes all groups to the same LR assignment, gradient scaler, checkpoint and step call.

`learning_rate_at_step(config,step)` [source L148–L159](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L148-L159) accepts a **one-based successful-update number** and rejects step zero. During warmup, LR increases linearly to the configured peak; thereafter `constant` stays at the peak. For cosine decay it uses a bounded progress fraction between the warmup boundary and `decay_steps`, and a cosine interpolation down to `min_lr_ratio * peak_lr`. Importantly, it is computed from the absolute successful step: dividing a run into sessions does **not** restart warmup or change the intended cosine horizon. An interruption's temporary `max_steps` is not itself the decay horizon.

## Device and precision contracts

`validate_precision` [source L162–L171](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L162-L171) rejects unsupported precision values, FP16 on CPU and CUDA BF16 without native hardware support. `_autocast` [source L174–L178](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L174-L178) returns a null context for FP32 and otherwise creates a PyTorch CPU/CUDA autocast context. The trainer explicitly requires FP32 **stored model parameters** for mixed-precision training [source L442–L461](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L442-L461); computation may autocast parts of the forward. A `torch.amp.GradScaler` is enabled **only for FP16**, not BF16. Model-specific routines such as KDA triangular solves and MoE router FP32 operations own their own dtype policy.

## Training-time validation is deliberately local

`evaluate(model,loader,device,max_batches,precision)` [source L253–L284](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L253-L284) temporarily moves the model to `eval`, resets its **independent validation loader to the start** and computes a token-weighted mean over up to `max_batches` full batches. It captures and restores Python, NumPy and torch RNG, and restores the caller's previous model `training` flag even on failure. The validation loader cursor is **not restored**: by design the next validation starts again from offset zero, while the training loader stays independent. This fixed-prefix validation is not the later exploratory-window API and is not, by itself, an independently preregistered evaluation protocol.

## Version-7 checkpoints contain a complete update boundary

`save_checkpoint(...)` [source L287–L311](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L287-L311) insists that gradients have been cleared with `optimizer.zero_grad(set_to_none=True)` before saving. It writes a temporary file and replaces the target only after serialization. Its format-7 payload stores model weights, the **one** hybrid optimizer state, `TrainState` counters/GradScaler, full train and model configs, vocab size, optimizer identity and per-parameter layout, QK-Clip policy, both loader cursor states, Python/NumPy/torch CPU/CUDA RNG, and a runtime signature. It does **not** save partial gradients or a half-accumulated attempt.

`load_checkpoint(...)` [source L314–L439](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L314-L439) follows an intentionally strict order: read with `weights_only=True`, reject incompatible historical recipe formats, validate the configuration and scalar counters, require matching vocab/model config/runtime/QK-Clip geometry/optimizer class/group layout and matching loader identities, then restore model/optimizer weights, cursors and RNG. Only after structural validation does it mutate weights. It restores effective optimizer hyperparameters from the checkpoint; increasing `max_steps` for continuation is supported, but the cosine decay settings are compatibility-checked. RNG restoration happens **last**, after model/optimizer construction, to avoid constructor draws changing the resumed sequence.

A checkpoint's promised continuation is conditional: matching dataset, software/device and deterministic kernel settings are required for bitwise equivalence. A saved filename alone is not enough. Training loaders hold a stream offset and manifest digest, not a copy of the entire dataset; prepared shards must remain immutable for meaningful resume.

### Checkpoint writes and the resume gate, in code

A checkpoint must represent an **update boundary**. The saver first rejects any live gradient, then writes the versioned payload to a temporary file ([source L287–L301](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L287-L301)):

```python
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
```

The saved dictionary additionally contains the counters, loader positions/manifest identities, RNG and optimizer layout; `temporary.replace(path)` publishes it only after the write finishes ([source L300–L311](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L300-L311)). There is no capture of half-accumulated gradients, and a partial temporary file is not a valid completion marker.

On loading, the trainer rejects changed recipe/runtime geometry **before** restoring parameters. The following checks are representative, not the entire compatibility contract ([source L364–L377](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L364-L377)):

```python
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
```

It also refuses formats 3–6, rejects changed QK-Clip policy, checks the full parameter-name/view layout and both loader identities, and validates the restored fixed-batch offset. Only at [source L430–L439](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L430-L439) does it load weights/optimizer, clear gradients, restore loader cursors and finally restore RNG. A validation rejection is designed to leave those mutable training objects unchanged.

## The update loop in source order

`train(model,optimizer,train_loader,val_loader,config,device,...)` [source L484–L509](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L484-L509) validates setup, sets `model.train()`, clears existing gradients, constructs the optional scaler and local stats, and reconstructs an overfit batch by saved **token offset** when requested. Inside one set of contexts for `TrainingProfiler`, `MoEDiagnostics` and `QKClip`, it records the session start and begins active-training timing [source L512–L529](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L512-L529).

The central loop repeats while `state.step < config.max_steps` [source L530–L572](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L530-L572):

1. Reset QK observations for the attempted update. Compute LR for candidate successful step `state.step + 1` and assign it to every optimizer group.
2. For each accumulation microbatch, read a fresh causal batch (unless fixed-overfit mode), run the forward inside autocast, reject nonfinite loss, backpropagate `loss/gradient_accumulation_steps`, and increment consumed `tokens_seen`.
3. Unscale FP16 gradients, calculate global norm and apply optional ordinary `clip_grad_norm_` (or infinity when disabled). Validate QK observations before parameter mutation.
4. Call `scaler.step(optimizer)` and `scaler.update()`. If the scale has dropped, the optimizer update was skipped; discard QK observations. Otherwise apply per-head **weight rescaling after** the optimizer step. Always clear gradients at the boundary.

The effective token count for an ordinary attempted update is `batch_size * seq_len * gradient_accumulation_steps`. Equal-sized microbatch losses are averaged before backward. A model with stateful QuantileBalancer updates QB on **each training forward**, however, so gradient accumulation is **not generally equivalent** to one concatenated larger batch. A skipped FP16 attempt still consumed tokens and ran QB forward updates, but it does not advance successful `step`, Muon/AdamW state or the LR schedule [source L494–L498](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L494-L498).

### An attempted step is not necessarily an update

Inside one attempt, the candidate LR uses `state.step + 1`. Every microbatch increments `tokens_seen` after backward, regardless of whether the later FP16 scaler skips the optimizer ([source L539–L554](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L539-L554)):

```python
			for _ in range(config.gradient_accumulation_steps):
				with profiling.region("train/data"):
					(x, y), fixed_batch = _get_training_batch(
						train_loader, device, config, state, fixed_batch
					)
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
```

The loss check fails the attempt *before backward and before this microbatch's token increment* if a nonfinite scalar is detected. A later overflow is different: all the attempted microbatches have already been read, backpropagated and counted, and the MoE's `QuantileBalancer` may already have updated on each training forward. Those QB biases are **buffers changed during forward**, not optimizer steps, so skipping the optimizer does not roll them back.

The next excerpt shows the exact ordering around a gradient/optimizer boundary ([source L555–L572](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L555-L572)):

```python
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
```

`unscale_` precedes norm clipping; `qk_clip.validate()` checks recorded score maxima *before* any optimizer weight mutation; `qk_clip.apply()` mutates eligible attention weights *only after* a successful optimizer step. A drop in GradScaler scale marks a skipped attempt. After clearing gradients, the skipped branch increments only `skipped_updates` and reports the overflow; the successful branch increments `state.step` and can trigger a checkpoint ([source L573–L600](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L573-L600)). Repeating the candidate successful step number does **not** repeat the consumed data. The profiler's attempt clock advances even for an overflow, whereas the LR schedule and interval triggers use successful `step`.

The true failure branches matter: FP16 on CPU or non-FP32 mixed-precision model weights are rejected during setup; missing/changed checkpoint metadata rejects resume before loading; a nonfinite loss raises immediately; `qk_clip.validate()` rejects nonfinite active-head observations before the optimizer; and 32 consecutive FP16 skipped attempts raise `FloatingPointError` ([source L442–L461](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L442-L461), [source L545–L549](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L545-L549), [source L590–L593](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L590-L593)). The code does not guarantee progress if the data stream or model repeatedly fails.

## Skips, progress, auxiliary phases and finalization

On FP16 overflow [source L573–L594](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L573-L594), the trainer increments `skipped_updates`, logs an `overflow` event, advances the profiler attempt boundary if needed, and retries the same *successful step index* with the next data. After 32 consecutive skips it raises to prevent unbounded failure. A valid update increments `state.step`, then decides which intervals are due for logging, evaluation and checkpointing; **all three occur on the final successful step** [source L595–L600](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L595-L600).

During a due auxiliary boundary [source L602–L638](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L602-L638), training-only timing pauses. The trainer writes a token-weighted `TrainingWindow` summary including pre-clip gradient norms and separate QK/MoE diagnostics; it evaluates the fixed validation prefix and stores a `val_loss`; it saves the checkpoint after gradients were cleared; and it advances any active profiler attempt window. Phase timing separates these costs from the active training rate. Finally, any unfinished profiler context is closed and `session_end` reported [source L639–L644](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L639-L644).

`_get_training_batch` [source L465–L481](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L465-L481) resets an exhausted training loader to offset zero, discarding an incomplete final tail, whereas `TokenDataLoader` itself never implicitly wraps. Overfit mode rereads one selected batch without incrementing the restored loader cursor. These distinctions matter when interpreting token counts or comparing uninterrupted and resumed runs.

## What the tests establish

[test test_recipe.py L20–L35](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_recipe.py#L20-L35) verifies schedule boundaries remain consistent across interrupted runs; [test test_trainer.py L61–L85](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_trainer.py#L61-L85) checks overfit-batch reuse and loss decrease in a controlled case. [test test_checkpoint_resume.py L142–L191](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py#L142-L191) compares resumed and continuous training state, and [test test_checkpoint_resume.py L290–L425](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py#L290-L425) checks that incompatible checkpoints are rejected without changing weights/cursors/RNG. [test test_qk_clip_training.py L79–L109](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_qk_clip_training.py#L79-L109) covers successful post-step clipping with resume, and [test test_training_metrics.py L43–L86](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py#L43-L86) checks logged events. The tests were read at the pinned revision, **not executed during this documentation pass**. In particular, the code's rigorous saved-state contract is not a universal guarantee of hardware-independent bitwise determinism.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py">trainer.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_recipe.py">test_recipe.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_trainer.py">test_trainer.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py">test_checkpoint_resume.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_qk_clip_training.py">test_qk_clip_training.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_training_metrics.py">test_training_metrics.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
