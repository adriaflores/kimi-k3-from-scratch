---
title: "MuonAdamW and Newton–Schulz"
description: "Combines per-matrix Muon updates with native AdamW in one checkpointable optimizer and optional compiled batching."
sourcePath: "src/kimi_k3/training/optimizers.py"
status: published
---

`MuonAdamW` is a **single PyTorch optimizer object** that dispatches explicitly configured parameter groups to two algorithms. Hidden two-dimensional matrices receive Muon; auxiliary tensors receive PyTorch AdamW. Both routes share one optimizer state dictionary and parameter-group schedule [source L134–L169](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L134-L169). The module also implements the Newton–Schulz update approximation and optional compiled/batched execution. It does **not** choose which model parameters deserve Muon—that is the responsibility of `training/parameter_groups.py`—or apply QK-Clip, which the trainer calls **after** successful optimizer updates.

## Newton–Schulz begins with an independently normalized matrix

`zeropower_via_newton_schulz(matrix,steps=5,eps=1e-7,dtype=torch.bfloat16,compiled=False)` is the public helper [source L101–L131](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L101-L131). It accepts a real, dense, nonempty tensor with at least two dimensions. The **last two axes** form a matrix; earlier axes enumerate *independent* matrices, not one flattened large matrix. It rejects unsupported working dtypes and shapes, and disables ambient autocast so the requested arithmetic dtype is respected.

The core [source L65–L78](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L65-L78) casts to the working dtype, transposes tall matrices for a convenient orientation, and normalizes each matrix by its Frobenius norm plus epsilon. In code, the quintic iteration uses coefficients `(a,b,c)=(3.4445,-4.7750,2.0315)`:

```python
	x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
	a, b, c = _NS_COEFFICIENTS
	for _ in range(steps):
		gram = x @ x.mT
		polynomial = b * gram + c * gram @ gram
		x = a * x + polynomial @ x
```

These are the exact source statements inside `_newton_schulz_core`. In mathematical terms, one update is `X ← aX + (bXXᵀ + c(XXᵀ)²)X`. It approximates the orthogonal/polar direction without an SVD, **not** exact polar decomposition; after a finite number of polynomial iterations its singular values are not guaranteed to equal one. A zero matrix stays zero. The implementation follows the credited upstream Moonlight/Muon approach and references the pinned notices at `references/muon/NOTICE.md` [source L1–L10](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L1-L10).

## Lazy compilation is local to the numerical core

With `compiled=True`, `_compiled_newton_schulz` [source L81–L99](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L81-L99) caches a full-graph TorchInductor specialization keyed by shape, stride, device, input dtype, iteration count, epsilon and working dtype. It gives each specialization its own code object and enables precision-cast emulation to preserve eager BF16 truncation semantics. Compilation happens lazily when first used, and the process-local compiled kernels do not enter checkpoints. Direct helper callers can request compilation for CPU verification; `MuonAdamW` itself enables that route **on CUDA only**. A compile error propagates rather than triggering an unannounced fallback.

## Parameter groups declare the real optimizer contract

The `MuonAdamW` constructor [source L171–L187](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L171-L187) receives groups with mandatory `algorithm="muon"` or `"adamw"`, then shares the resulting `param_groups` and `state` with a native `torch.optim.AdamW` created for only the AdamW groups [source L269–L285](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L269-L285). The optimizer requires dense FP32 or FP64 *parameter storage*; training in BF16/FP16 is performed with autocast around the model, not by storing trainable parameters in a half dtype.

Group validation [source L189–L247](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L189-L247) also enforces that Muon parameters have nonempty **2D storage**, that a declared `muon_shape` has the same number of elements, and that `muon_split_sizes` sums to the independent matrix view's row dimension. `add_param_group` rejects duplicate parameter objects within or across groups [source L249–L267](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L249-L267). These conditions prevent an unrelated vector, convolution or tied tensor from silently being optimized as a flattened matrix.

`muon_shape` tells the algorithm how to reinterpret packed rows into independent matrices. For example `(2,I,C)` treats the fused FFN gate and up matrices separately, while `(3,H,D,C)` separates KDA's Q/K/V heads. A `muon_split_sizes=(D,D_v)` further partitions each MLA head's **key and value** rows before the normalization/orthogonalization steps. Parameter storage, momentum and weight decay still belong to the **single original tensor**: splitting affects the Muon update direction and shape scaling only.

## One individual Muon update, from gradient to new weight

`_step_muon_parameter(parameter,group)` [source L321–L347](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L321-L347) first returns without action when `parameter.grad is None`. No momentum state is initialized, no decay is applied and no step is counted for that inactive parameter. A present **zero** gradient is different: it is a valid active update and existing momentum and weight decay can still change the weight.

For a present gradient `G` and stored momentum `M`, the code computes `M ← μM + G`; with Nesterov enabled (the trainer's recipe) it forms `U = G + μM`, otherwise `U = M` [source L326–L334](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L326-L334). It reshapes `U` to the declared independent matrix layout, runs Newton–Schulz per matrix or per specified row part, and applies the Moonlight factor `0.2*sqrt(max(rows,columns))` calculated **for each independent matrix's true dimensions** [source L335–L344](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L335-L344). Finally, it applies decoupled weight decay and the normalized update:

$$
W \leftarrow (1-\eta\lambda)W - \eta\,s\,\widehat U,
$$

where `η` is the group's unscaled learning rate, `λ` its weight decay, `s` the matrix's geometry-dependent factor and `Û` the numerical Newton–Schulz result. Decay uses the **unscaled** learning rate; matrix scaling affects only the update term. For split matrices the scale is applied to each part before reassembly, so the outer scale becomes `1.0`.

### The literal individual-update path

The no-gradient guard and momentum calculation are adjacent in the source ([source L321–L334](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L321-L334)):

```python
		gradient = parameter.grad
		if gradient is None:
			return
		state = self.state[parameter]
		if not state:
			state["momentum_buffer"] = torch.zeros_like(parameter)
		buffer = state["momentum_buffer"]
		buffer.mul_(group["momentum"]).add_(gradient)
		update = gradient.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
		shape = group.get("muon_shape") or parameter.shape
		update = update.reshape(shape)
```

For a packed FFN weight stored as `(2I,C)`, the `muon_shape=(2,I,C)` view produces two independent matrices; for KDA `QKV`, `(3,H,D,C)` keeps three projections and their `H` heads separate. The code does not flatten those leading axes into a single orthogonalization matrix. When a group declares additional MLA K/V row splits, each part is orthogonalized and shape-scaled independently before returning to the one original parameter storage ([source L335–L346](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L335-L346)):

```python
		if splits is None:
			update = zeropower_via_newton_schulz(update, steps=group["ns_steps"], dtype=group["ns_dtype"], compiled=compiled)
			scale = 0.2 * math.sqrt(max(shape[-2:]))
		else:
			assembled = torch.empty(shape, dtype=parameter.dtype, device=parameter.device)
			for part, destination in zip(update.split(splits, dim=-2), assembled.split(splits, dim=-2)):
				destination.copy_(zeropower_via_newton_schulz(part, steps=group["ns_steps"], dtype=group["ns_dtype"], compiled=compiled))
				destination.mul_(0.2 * math.sqrt(max(part.shape[-2:])))
			update, scale = assembled, 1.0
```

The remaining lines apply decay and the update to the original storage once ([source L345–L346](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L345-L346)); no separate per-head or per-expert `Parameter` objects are created.

## Optional batched execution preserves independent updates

`_step_muon_bucket` [source L348–L399](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L348-L399) groups parameters of identical physical shape, device and dtype, filters out those with `grad=None`, and processes active parameters in bounded chunks. `_batch_capacity` [source L30–L45](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L30-L45) estimates temporary packing/orthogonalization storage from matrix and Gram-matrix sizes, subject to both batch-size and byte-limit options. Its bound is an **estimate of local live intermediates**, not a guarantee on total optimizer VRAM, allocator reservations or third-party library workspace.

For a multi-parameter chunk, momentum updates use `torch._foreach_*`, inputs are temporarily stacked, and the last partial chunk is padded with zero lanes to a power of two for stable compiled layouts. Those padded lanes are not parameters: they receive neither decay nor optimizer state. Newton–Schulz runs along the leading independent axes; row splits remain separate. Updated matrices are assigned back through foreach weight operations. Singleton chunks use the individual reference implementation. This is a packaging/launch optimization, **not** an algorithmic decision to orthogonalize multiple experts jointly.

### What batching changes, and what it cannot promise

The actual `step()` dispatch is explicit ([source L400–L422](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L400-L422)):

```python
		for group in self.param_groups:
			if group["algorithm"] != "muon":
				continue
			if group["ns_batched"]:
				for bucket in self._muon_buckets[id(group)]:
					self._step_muon_bucket(bucket, group)
				continue
			for parameter in group["params"]:
				self._step_muon_parameter(parameter, group)
		if self._adamw is not None:
			self._adamw.step()
```

Only parameters whose `.grad is not None` enter a batched bucket ([source L348–L357](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L348-L357)). For active groups of the same physical shape/device/dtype, `_batch_capacity` computes a **conservative bound** on the pack's temporary matrix/Gram storage, then rounds the allowed capacity down to a power of two ([source L30–L45](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L30-L45)). This is not a measured peak VRAM limit: the estimator explicitly excludes persistent optimizer state, PyTorch allocator reservations and possible GEMM workspaces. Internally, a partial chunk pads to the next power of two in *temporary* storage only; padded lanes acquire no optimizer state and no weight decay ([source L374–L397](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L374-L397)).

Autocast is disabled inside `zeropower_via_newton_schulz`, which uses its **requested** NS working dtype. On CPU, the optimizer keeps Newton–Schulz eager even if `ns_compile=True`; on CUDA that flag selects compiled NS, not compiled AdamW or the entire training update. The public numerical helper separately permits CPU compilation for verification ([source L101–L131](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L101-L131), [source L334–L357](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L334-L357)). Inputs remain dense FP32/FP64 parameters; BF16/FP16 outer model computation relies on the trainer's autocast with FP32 parameter storage.

## Sharing AdamW state and restoring a saved recipe

`_bind_adamw()` [source L269–L285](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L269-L285) creates the auxiliary native optimizer over the same group dictionaries and directly aliases its `.state` to this optimizer's state. Consequently the trainer can save one canonical `optimizer.state_dict()` containing both Muon momentum buffers and AdamW moments/steps, not two partially independent optimizers.

`load_state_dict` [source L288–L319](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L288-L319) normalizes backward-compatible format-7 execution defaults (`ns_compile`, batching settings), checks group counts, algorithm identity, parameter counts, compiled/batched modes and declared matrix geometry, then delegates state loading and rebinds the native AdamW wrapper. This is compatibility validation, not a promise that a changed model or runtime can safely resume; `trainer.load_checkpoint` adds more constraints.

Finally, `step(closure=None)` [source L400–L422](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py#L400-L422) optionally evaluates the closure once with gradients enabled, **prevalidates** that all present gradients are dense, processes Muon groups individually or in batches, then calls native AdamW once for its groups. Rejecting sparse gradients before updating either algorithm prevents partial hybrid updates in that failure case. The enclosing trainer owns LR progression, gradient scaling/overflow skips and post-step QK-Clip.

## Verification and limits

[test test_optimizers.py L23–L98](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_optimizers.py#L23-L98) compares Newton–Schulz with an independently constructed polynomial/SVD reference and checks momentum, Nesterov and decay semantics; [test test_optimizers.py L101–L121](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_optimizers.py#L101-L121) compares the auxiliary groups with native AdamW. [test test_muon_batched.py L38–L71](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_muon_batched.py#L38-L71) compares grouped and individual updates with partial chunks and inactive parameters. [test test_per_head_muon.py L30–L90](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_per_head_muon.py#L30-L90) checks head/row-split independence, and [test test_optimizers.py L164–L198](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_optimizers.py#L164-L198) tests restoration of both optimizer branches. No benchmark was rerun: these files establish what the tests **assert**, not a fresh performance measurement or an exact SVD guarantee.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py">optimizers.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_optimizers.py">test_optimizers.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_muon_batched.py">test_muon_batched.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_per_head_muon.py">test_per_head_muon.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
