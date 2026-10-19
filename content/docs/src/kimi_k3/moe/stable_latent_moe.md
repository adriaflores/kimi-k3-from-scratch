---
title: "StableLatentMoE and QuantileBalancer"
description: "Implements shared and top-k routed expert paths, quantile-based router correction, and grouped expert execution."
sourcePath: "src/kimi_k3/moe/stable_latent_moe.py"
status: review
---

This file defines two cooperating classes: `QuantileBalancer`, which maintains non-gradient routing biases, and `StableLatentMoE`, which computes a shared FFN output plus a routed-expert output in a compressed latent space. The input and output are `(B,T,M)`. Experts are selected using corrected router scores, but contributions are weighted with **uncorrected sigmoid scores**. That distinction is central to reading this forward correctly.

## QuantileBalancer: persistent biases, no gradient update

`QuantileBalancer(num_experts,k)` registers a persistent `(E,)` buffer of zeros named `biases` [source L10–L21](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L10-L21). It is in `state_dict`, but is not an optimizable `Parameter`. The no-grad `update(top_scores, router_scores)` computes $p=k/E$. The caller supplies the highest `k+1` *bias-corrected* scores; the `(k+1)`-th score becomes each token's comparison threshold [source L23–L32](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L23-L32).

```python
		thresh = top_scores.view(-1, self.top_k + 1)[:, -1].unsqueeze(-1) # (BT, 1)
		relative_thresh = thresh - router_scores # (BT, E)
		new_biases = torch.quantile(relative_thresh, p, dim=0, interpolation="higher") # (E)
		new_biases = new_biases - new_biases.mean()
```

Each expert's new bias is an empirical quantile of the difference between a per-token threshold and that expert's raw sigmoid score. Centering removes a common offset across experts, which would not change the ranking. `biases.copy_(new_biases)` mutates the persistent buffer during training forwards. This is neither backpropagation nor an EMA; the current batch's quantile directly replaces the previous buffer. The thresholds are based on `top_k+1`, not only the routed `top_k` entries.

## Construction separates shared and routed capacities

The constructor [source L37–L73](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L37-L73) creates a `shared_experts` instance of `DenseFFN(M, num_shared_experts * expert_intermediate_size)` evaluated for **every token**. The routed side has a full-width router `Linear(M,E)`, one compression `down_latent_proj: M → L`, a `ModuleList` of `E` `DenseFFN(L,I)` experts, optional post-routing RMSNorm in latent width `L`, and `up_latent_proj: L → M`. Both expert families use SiTU-GLU; routed experts enable the conditional compiled activation path.

The `num_shared_experts` count determines aggregate intermediate capacity inside one shared FFN, not a loop over distinct separately routed shared modules. The temporary `_expert_load_counts` starts as `None` and is set by `MoEDiagnostics` only during active diagnostic collection; it is not part of model state.

## FP32 router and training-only QB update

The forward computes `shared = self.shared_experts(x)` and then enters an explicit disabled-autocast block for routing [source L76–L93](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L76-L93). Inside it, `F.linear` uses FP32 versions of both activations and router weights to keep router arithmetic in FP32 even under a mixed-precision outer context.

```python
			router_logits = F.linear(x.to(torch.float32), self.router.weight.to(torch.float32), bias=None) # (B, T, E)
			router_scores = torch.sigmoid(router_logits)
			biases = self.quantile_balancer.get_biases() # (E)
			corrected_scores = router_scores + biases # (B, T, E)
			top_scores, top_indices = torch.topk(corrected_scores, k=self.top_k + 1, dim=-1) # (B, T, K + 1), (B, T, K + 1)
```

The code calls `quantile_balancer.update(...)` only when `self.training` is true. Crucially, it uses the **already selected** `top_indices` for the current forward, then updates the buffer for subsequent calls. Evaluation reads biases but leaves them unchanged. Gradient tracking for the router remains meaningful through the gathered raw scores, even though the discrete top-k indices are not differentiable.

### The QB buffer changes during forward, not at optimizer.step

The selected `top_indices` are computed from corrected scores **before** `update` changes the buffer. The update is gated by `self.training`, not by an optimizer, a backward call or `torch.is_grad_enabled()` ([source L85–L94](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L85-L94)):

```python
			top_scores, top_indices = torch.topk(corrected_scores, k=self.top_k + 1, dim=-1) # (B, T, K + 1), (B, T, K + 1)
			router_scores = router_scores.reshape(B * T, -1) # (BT, E)
			if self.training:
				self.quantile_balancer.update(top_scores=top_scores, router_scores=router_scores)
		top_indices = top_indices[..., :-1] # (B, T, K)
```

Consequently a training-mode forward under `torch.no_grad()` still changes QB, whereas a forward after `model.eval()` does not. Within gradient accumulation each microbatch refreshes QB, even if the final FP16 attempted optimizer update is skipped. Its latest buffer value belongs in the model state checkpoint. The order of *which experts are selected* is discrete; gradients to router weights arise from the raw selected sigmoid scores gathered later, not from differentiating through `topk` indices.

## Flatten routing and recover original token order

After discarding the extra threshold candidate, `top_indices` has shape `(BT,K)`. The original `x` is flattened to `(BT,M)` and compressed once per token into `z: (BT,L)` [source L93–L101](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L93-L101). The dispatch weights are gathered from the *raw* sigmoid `router_scores`, not from `corrected_scores`. With `moe_renormalize`, each token's selected weights have L1 sum one; otherwise the raw positive sigmoid weights remain unnormalized.

The token/expert assignments are flattened to length `BTK`, sorted by expert ID with `torch.argsort(..., stable=True)`, and recovered as `sorted_token_ids = order // top_k` [source L103–L112](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L103-L112). `torch.bincount` yields expert loads, while cumulative counts `ends` mark slices for each expert. Sorting groups rows for batched execution and preserves stable order among equal expert IDs. A training diagnostics context, if installed, accumulates integer per-expert assignment counts before expert computation [source L114–L118](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L114-L118).

### Dispatch is a stable permutation, not token duplication in model state

The exact flatten/sort/count sequence groups all token–expert assignments by expert ID ([source L103–L112](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L103-L112)):

```python
		expert_ids = top_indices.flatten() # (BTK)
		flat_weights = weights.flatten() # (BTK)

		order = torch.argsort(expert_ids, stable=True)

		sorted_token_ids = order // self.top_k
		sorted_weights = flat_weights[order]

		counts = torch.bincount(expert_ids, minlength=self.num_routed_experts)
		ends = torch.cumsum(counts, dim=0).tolist()
```

For a small illustration with `BT=3`, `K=2` and selected experts `[[2,0],[0,2],[2,1]]`, flattening yields `[2,0,0,2,2,1]`. The stable `order` is `[1,2,5,0,3,4]` and `sorted_token_ids` becomes `[0,1,2,0,1,2]`. The expert counts are `[2,1,3]`, so cumulative `ends=[2,3,6]`: the first two rows feed expert 0, the next row expert 1, and the last three rows expert 2. The permutation changes the **execution order of assignments**; final `index_add_` restores contributions to their original token rows.

## Grouped expert execution and sparse gradients

The normal path calls `_forward_grouped_experts`, **not** `_forward_naive_experts` [source L120–L129](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L120-L129). The latter remains a readable Python reference [source L132–L152](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L132-L152): for each expert, select assigned latent rows, apply its DenseFFN, multiply by dispatch weight and routed scaling, and `index_add_` into the original token positions. The grouped path implements the same structure without a separate Python FFN invocation for each nonempty expert.

At [source L155–L176](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L155-L176), `active_ids` excludes experts whose assignment range is empty. The selected rows `z_sorted` are in expert-sorted order; only active expert gate/up and down matrices are stacked. `F.grouped_mm` executes the packed expert matrix operations using cumulative `offsets` in integer `int32`. The intermediate SiTU activation may use the compiled callable in CUDA training with gradients, otherwise the eager form. The second grouped GEMM and `index_add_` scatter the weighted results back to `(BT,L)`.

```python
		routed_output = z.new_zeros(z.shape)
		routed_output.index_add_(0, sorted_token_ids, y_sorted * expert_weights)
```

Unselected experts are omitted from the stacks and remain outside autograd for that forward. With absent gradients, the optimizer can leave their weights, decay and state untouched. Selected expert contributions are summed when the same token reaches multiple experts.

### The actual grouped GEMM, and the gradient path

In the fused execution path the first operation drops zero-load experts. Only the remaining experts' stored trainable matrices participate in the `torch.stack`, preserving the autograd connection to those weights ([source L155–L175](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L155-L175)):

```python
		active_ids = [j for j, end in enumerate(ends) if end > (ends[j - 1] if j else 0)]
		offsets = torch.tensor([ends[j] for j in active_ids], device=z.device, dtype=torch.int32)
		z_sorted = z[sorted_token_ids] # (BTK, D_latent)

		# grouped_mm does not autocast its operands; only the temporary stacks change dtype.
		gate_up_weights = torch.stack([self.experts[j].gate_up_proj.weight for j in active_ids]).to(z.dtype)
		gate_up = F.grouped_mm(z_sorted, gate_up_weights.transpose(-2, -1), offs=offsets)
```

The second matrix multiplication stacks corresponding down-projection weights and uses the same assignment boundaries ([source L170–L175](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L170-L175)):

```python
		down_weights = torch.stack([self.experts[j].down_proj.weight for j in active_ids]).to(activated.dtype)
		y_sorted = F.grouped_mm(activated, down_weights.transpose(-2, -1), offs=offsets)
		expert_weights = (sorted_weights.unsqueeze(-1) * self.routed_scaling_factor).to(y_sorted.dtype)

		routed_output = z.new_zeros(z.shape)
		routed_output.index_add_(0, sorted_token_ids, y_sorted * expert_weights)
```

Backpropagation flows from `routed_output` through `index_add_`, the grouped GEMMs and the stacked active parameters. The scalar dispatch weights also feed gradients back to the **selected raw** router scores and thus to the router projection. Index selection itself remains nondifferentiable. Inactive experts do not enter the stack, so their parameter gradients remain `None` in that forward (rather than fabricated zero gradients); the optimizer's missing-gradient rule prevents an unintended decay/momentum update. This equivalence and sparsity are asserted by the pinned [grouped-versus-naive test](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py#L69-L119), not measured here.

## Return to model width

The `(BT,L)` routed output is reshaped back to `(B,T,L)`, optionally RMS-normalized, and up-projected to `(B,T,M)` [source L122–L129](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py#L122-L129). The returned tensor is `shared + routed`; there is no decoder residual addition in this class. The scaling factor is applied to each expert's weighted contribution **before** the latent post-norm and up-projection; consequently, the final magnitude depends also on the normalization flag.

## Verification and limits

[test test_moe_grouped_vs_naive.py L69–L119](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py#L69-L119) parameterizes execution over routing patterns, evaluation/training flags and precision/device choices, comparing outputs, gradients, quantile biases and expert loads with the naive reference. [test test_moe_grouped_vs_naive.py L135–L181](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py#L135-L181) covers sparse AdamW behavior for inactive and reactivated experts. [test test_moe_metrics.py L14–L52](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_moe_metrics.py#L14-L52) checks routed counts and bias snapshots. These tests were inspected but not executed. This implementation does not expose load-balancing as a differentiable auxiliary loss, does not infer throughput from grouped execution, and relies on PyTorch's `F.grouped_mm` support in the target environment.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py">stable_latent_moe.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py">test_moe_grouped_vs_naive.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_moe_metrics.py">test_moe_metrics.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
