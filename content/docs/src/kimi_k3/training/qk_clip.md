---
title: "QKClip"
description: "Observes causal MLA QK scores and rescales selected projection rows after successful updates."
sourcePath: "src/kimi_k3/training/qk_clip.py"
status: published
---

`QKClip` implements a training-time intervention for **MLA**, not KDA: it observes the maximum causal attention score per head and, when the configured threshold is exceeded, rescales selected Q/K projection weights. The observer is temporary, and rescaling is performed **after a successful optimizer step**, never as part of ordinary evaluation [source L1–L6](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L1-L6). This file implements neither a different attention forward kernel nor a general gradient clip.

## Compute causal maxima in bounded score blocks

`causal_max_logits(query,key,chunk_size=256)` receives matching nonempty floating tensors `(B,H,T,D_qk)`, refuses invalid shapes/devices/dtypes, and detaches both tensors [source L25–L42](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L25-L42). BF16 and FP16 are promoted to FP32; FP64 remains FP64 for numeric verification. For query block `start:end`, keys run from sequence start through `end`; only future positions within the current block are masked to `-inf` [source L43–L54](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L43-L54).

```python
			scores = q[:, :, start:end] @ k[:, :, :end].mT
			# All earlier key blocks are valid. Only this final square contains
			# future positions, including the shorter final query chunk.
			scores[..., start:end].masked_fill_(future[:end - start, :end - start], -torch.inf)
			maximum = torch.maximum(maximum, scores.amax(dim=(0, 2, 3)))
```

These lines are copied from the inner chunk loop. Reducing over batch and both time axes yields one **signed maximum per head** `(H,)`, scaled by `1/sqrt(D_qk)`. Working score memory scales with `B*H*chunk_size*T`, rather than allocating the full `B*H*T*T` scores at once; arithmetic still computes the causal dot products. A negative maximum remains negative rather than being clamped to zero.

## Discover MLA sites without changing the module

`_mla_sites(model)` structurally identifies modules with `W_UQ`, `W_UKV` and `W_DKV` and validates all per-head projection geometries [source L57–L69](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L57-L69). It does not capture KDA sites. `qk_clip_metadata` packages the threshold, chunk size and head geometry under policy `mla_shared_key_v1` for manifests/checkpoint validation [source L72–L81](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L72-L81). The model's MLA `forward` conditionally invokes the observer only during training with gradients enabled.

## Temporary observer lifecycle

`QKClip.__enter__` installs `_qk_clip_observer` callables on identified MLA modules, rejecting already attached observers; `__exit__` removes them and clears per-attempt observations even when an exception occurs [source L84–L129](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L84-L129). `observe` records the head-wise maximum and merges successive microbatch samples by `maximum`, **not average** [source L130–L136](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L130-L136). `validate` rejects nonfinite maxima on active trainable projections before mutating any clipping weights [source L138–L151](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L138-L151). An inactive, frozen or unused site's rows are not clipped.

### Observation, validation, and weight mutation are separate calls

The context manager attaches one callable to each structurally matched MLA site; the model invokes it only under training and enabled gradients ([source L104–L125](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L104-L125); [MLA forward L64–L67](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/gated_mla.py#L64-L67)). During accumulation, the observer combines actual scores from multiple forwards using a per-head maximum ([source L130–L136](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L130-L136)):

```python
		with torch.profiler.record_function("attention/qk_clip_stats"):
			maximum = causal_max_logits(query, key, chunk_size=self.chunk_size)
		previous = self.max_logits.get(name)
		self.max_logits[name] = maximum if previous is None else torch.maximum(previous, maximum)
		self._validated = False
```

At this point **no projection weights have changed**. `validate()` checks that each active MLA site's maxima are finite, iterating through all recorded sites before `apply()` modifies anything ([source L145–L151](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L145-L151)). The active predicate requires both unshared Q and KV projections to be trainable and have gradients; an observation of a frozen or unused site does not authorize rescaling. The trainer calls validation *before* `scaler.step(optimizer)` and applies QK-Clip *after* a successful optimizer update ([source L555–L572](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/trainer.py#L555-L572)). A nonfinite active statistic raises `FloatingPointError` before the optimizer changes weights, while a skipped FP16 step discards the current maxima with `reset()` and does not call `apply()`.

The context's `__exit__` restores/removes the temporary `_qk_clip_observer` attributes and clears pending observation state, including on exceptions. `summary()` covers recorded **successful applied updates** and does not retroactively correct a previously completed forward. QK-Clip is a post-update intervention, not an extra term in the attention loss.

## Exact rows rescaled after the step

For every active head, `gamma = threshold / maximum.clamp_min(threshold)` gives `gamma <= 1` [source L153–L180](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L153-L180). If the maximum does not exceed threshold, no rows change. Otherwise the code takes unflattened Q and KV projection views and applies the following factors:

| Weight rows | Applied factor |
| --- | --- |
| Q content rows `:D` | `sqrt(gamma)` |
| Q shared-key-paired rows `D:` | `gamma` |
| K content rows `:D` in `W_UKV` | `sqrt(gamma)` |
| V rows `D:` in `W_UKV` | unchanged |
| Shared K from `W_DKV` | unchanged |

Splitting the factors across content Q and K reduces their dot product by `gamma`; scaling the Q half paired with the untouched shared K by `gamma` does the same for the shared component. The operation mutates post-update **weights**, not stored gradients, momentum or AdamW state. The trainer discards observations from FP16-overflow attempts before applying any rescaling.

`summary(reset=False)` [source L182–L198](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L182-L198) records per-layer head maxima, smallest gamma, count of clipped head-updates and total observed head-updates. The summary is a training diagnostic; it is not direct evidence of downstream model quality.

### Exact per-head weight rescaling

`apply()` first validates and then computes each head's `gamma`, which is one when the observed maximum is at or below the threshold ([source L153–L171](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py#L153-L171)):

```python
			gamma = self.threshold / maximum.clamp_min(self.threshold)
			triggered = maximum > self.threshold
			if triggered.any().item():
				q = module.W_UQ.weight.unflatten(0, (module.num_heads, module.d_head + module.d_shared))
				kv = module.W_UKV.weight.unflatten(0, (module.num_heads, module.d_head + module.dv_head))
				factors = gamma.to(q.dtype).view(-1, 1, 1)
				root = factors.sqrt()
				q[:, :module.d_head].mul_(root)
				q[:, module.d_head:].mul_(factors)
				kv[:, :module.d_head].mul_(root)
```

Each row operation is performed under `@torch.no_grad()` *on the post-optimizer stored parameter*, without modifying `.grad` or optimizer moments. In particular, `W_UKV` value rows and the `W_DKV` shared-key projection are left alone. The logged maxima are gathered from the **pre-update forward**, so the calculation is not a claim that every future input's score is globally bounded. The original Q/K clipping test verifies the fixed-input scaling behavior and unchanged non-target weights, not a bound over every possible context.

## Verification

[test test_qk_clip.py L26–L60](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_qk_clip.py#L26-L60) compares bounded causal maxima with an independent reference, including dtype and masking cases. [test test_qk_clip.py L88–L115](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_qk_clip.py#L88-L115) checks exactly which projection rows change; [test test_qk_clip.py L118–L135](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_qk_clip.py#L118-L135) verifies microbatch maxima and single application. [test test_qk_clip_training.py L79–L109](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_qk_clip_training.py#L79-L109) tests resume equivalence with active clipping, while later cases reject incompatible checkpoints and skipped-update contamination. These tests were inspected but not rerun.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/qk_clip.py">qk_clip.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_qk_clip.py">test_qk_clip.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_qk_clip_training.py">test_qk_clip_training.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
