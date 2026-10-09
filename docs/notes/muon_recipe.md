# Integrated Muon + QK-Clip recipe

The trainer has one optimizer recipe: hidden matrices use Muon; embeddings,
the vocabulary head, norm gains, biases, logical vectors, AttnRes pseudo-queries
and KDA depthwise filters use native AdamW. Shared parameters appear once.
Matrices and convolution filters receive configured weight decay; norms,
biases and logical vectors do not. Both algorithms share the LR schedule.

Muon uses raw momentum with Nesterov, five BF16 Newton–Schulz iterations and
the Moonlight scale `0.2 * sqrt(max(logical_rows, logical_columns))`. Norms are
independent for every logical matrix. Momentum and decay operate once on the
original stored parameter. `grad=None` skips decay and all state changes.

## Matrix geometry

| Projection | Logical Muon view | Treatment |
| --- | --- | --- |
| Dense/shared/routed FFN `gate_up_proj` | `(2, intermediate, input)` | Separate gate and up matrices |
| KDA `QKV` | `(3, heads, head_dim, model_dim)` | Separate Q/K/V and heads |
| MLA `W_UQ` | `(heads, key_dim + shared_dim, q_latent_dim)` | One independent matrix per head |
| MLA `W_UKV` | `(heads, key_dim + value_dim, kv_latent_dim)` | Separate K and V matrices inside each head; unequal dimensions supported |
| Shared latent down-projections, attention output projections, matrix gates | Original 2D matrix | Whole-matrix Muon |

The views describe the real projection layout without introducing aliased
parameters. Results are reassembled into the original storage shape. Checkpoints
validate names, shapes, views and K/V boundaries before restoring weights.

## QK weight clipping

MLA observers collect the maximum **scaled, causally visible attention logit**
per head during training. The statistic includes content and shared components
and accumulates the maximum across all microbatches of an optimizer attempt.
Chunked observation bounds temporary memory; it does not alter attention output.
Evaluation does not collect training statistics.

After a successful optimizer step, `gamma = min(1, threshold / max_logit)` for
each head whose maximum exceeds the threshold (default `100`). Content Q and K
rows are multiplied by `sqrt(gamma)`; shared Q rows by `gamma`. Shared K and
V rows are unchanged. This scales both attention-logit components by `gamma`.
KDA uses normalized Q/K and does not receive this MLA clipping.

QK-Clip is separate from global **gradient** clipping. The order is accumulation,
AMP unscaling, gradient clipping, optimizer update, then QK weight clipping.
An FP16 overflow skips both optimizer algorithms and QK-Clip and discards that
attempt's observations. Each completed update ends with no pending gradients or
statistics. Logs record maximum logits, observed head updates and clipped heads.

## Configurations and resume

`configs/training/acceptance.yaml` exercises 64 updates at peak LR `3e-4`, two
microbatches, weight decay `0.1`, eight warmup updates and a fixed 64-update cosine
horizon. It is a bounded integration check, not tuned pretraining hyperparameters.
`smoke`, `overfit` and `stability` use the same optimizer/clipping recipe.

Checkpoint format **7** saves the hybrid state, clipping policy/geometry,
precision, recipe, counters, loader positions and RNG. Legacy AdamW and
pre-clipping Muon checkpoints are rejected. Existing format-7 eager checkpoints
remain compatible: an absent `muon_compile`/`ns_compile` means `false`.
The requested compilation mode must match on resume. Compiled kernels are
process-local and rebuilt lazily; they are not checkpoint contents.

The shipped tiny regression checks next-token learning, evaluation isolation,
accumulated FP32/BF16 training and exact same-backend resume. CUDA cases run when
CUDA is available. CPU acceptance does not establish 4050 throughput.

```bash
uv run pytest -q tests/unit/training tests/integration/test_tiny_training.py \
  tests/integration/test_checkpoint_resume.py tests/integration/test_qk_clip_training.py
```

See [compilation and performance comparison](muon_compile.md) for the opt-in
compiled execution and GPU commands.
