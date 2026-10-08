# Muon optimizer core — deliverable 1

This delivery implements the numerical optimizer foundation for the agreed
Per-Head Muon + attention weight-clipping recipe. It was prepared against
repository commit [`9ceda92`](https://github.com/adriaflores/kimi-k3-from-scratch/commit/9ceda92de4b3928f3fc3f0c62c4e06185aa50c4f).

## Scope and installation

Extract the ZIP **at the repository root**, preserving its relative paths.
The ZIP contains only these added or modified files:

| Path | Change |
| --- | --- |
| `src/kimi_k3/training/optimizers.py` | New numerical kernel and unified optimizer. |
| `src/kimi_k3/training/__init__.py` | Public `MuonAdamW` export. |
| `tests/unit/training/test_optimizers.py` | Mathematical, state and PyTorch integration tests. |
| `docs/notes/muon_optimizer_core.md` | This delivery's contract and validation. |
| `references/muon/NOTICE.md` | Pinned references and upstream MIT notices. |

**The model parameter assignment and trainer connection belong to deliverable 2.**
`build_optimizer`, training YAML files, CLI flags and checkpoint-format version
are therefore unchanged in this delivery. Importing the new module does not
change an existing run. No optimizer-selection flag was introduced.

Per-head assignment belongs to deliverable 3, attention weight clipping to
deliverable 4, and the final recipe/CLI/training validation to deliverable 5.
Consequently, running `scripts/train.py` now still uses the existing trainer;
it does not yet exercise the new optimizer.

## References and the fixed update convention

The implementation follows the public MoonshotAI/Moonlight example for base Muon.
Kimi's [official K3 blog](https://www.kimi.com/en/blog/kimi-k3) establishes the
per-head extension, but does not give an executable optimizer recipe. The older
Moonlight code and paper supply the documented numerical foundation; this delivery
does not assume that all unpublished K3 optimizer details are identical.

For one matrix of shape `[rows, columns]` and gradient `G`:

```text
M <- momentum * M + G
direction <- G + momentum * M            # Nesterov
U <- NewtonSchulz(direction)
W <- (1 - lr * weight_decay) * W
W <- W - lr * 0.2 * sqrt(max(rows, columns)) * U
```

With `nesterov=False`, the direction is `M`. The momentum buffer follows the
upstream **unnormalized SGD** convention, rather than an exponential average.
The gradient tensor is not modified.

The shape factor scales the update; it **does not scale weight decay**. The
resulting learning-rate convention follows Moonlight Equation (4), so a Muon
learning rate from an implementation using another scaling convention cannot
be transferred without checking that convention.

Newton-Schulz uses coefficients `(3.4445, -4.7750, 2.0315)`, Frobenius
normalization with `1e-7`, and five iterations by default. It uses the smaller
Gram matrix by transposing tall inputs. This polynomial is an approximate
orthogonalization, so its output's singular values need not equal one.

The helper can process a batch of matrices on its final two axes. Each matrix
has its own normalization and polynomial iteration. This is a numerical building
block for later per-head handling; the optimizer itself currently accepts only
explicit **2D** Muon parameters and never silently flattens higher-rank tensors.

## One public optimizer

`MuonAdamW` is a `torch.optim.Optimizer` subclass. Each group explicitly sets
`algorithm="muon"` or `algorithm="adamw"`. That field assigns a parameter's
treatment inside the hybrid recipe; it is not a trainer option for choosing
between two recipes.

The auxiliary implementation is native `torch.optim.AdamW`, including its fused
path. It shares the parent optimizer's actual group dictionaries and state.
Changing a public group's learning rate reaches both algorithms, and one
`state_dict()` contains all momentum buffers, AdamW moments and step counters.
Loading that state rebinds the native AdamW component to the restored objects.

A model-independent example:

```python
import torch
from src.kimi_k3.training.optimizers import MuonAdamW

matrix = torch.nn.Parameter(torch.randn(4, 6))
bias = torch.nn.Parameter(torch.zeros(4))

optimizer = MuonAdamW([
    {"params": [matrix], "algorithm": "muon"},
    {"params": [bias], "algorithm": "adamw", "weight_decay": 0.0},
], lr=3e-4, weight_decay=0.1)

optimizer.zero_grad(set_to_none=True)
loss = matrix.square().mean() + (bias - 1).square().mean()
loss.backward()
optimizer.step()

checkpoint_state = optimizer.state_dict()
```

The model integration will build these groups using the model's structure and
deduplicate tied weights. The core rejects parameters appearing twice, including
duplicates inside one group.

## Numerical defaults and execution contract

| Setting | Default | Meaning |
| --- | --- | --- |
| `lr` | `3e-4` | Common base learning rate; groups may override it. |
| `weight_decay` | `0.1` | Decoupled decay with the unscaled group LR. |
| `momentum` | `0.95` | Muon momentum. |
| `nesterov` | `True` | Upstream Nesterov direction. |
| `ns_steps` | `5` | Quintic iterations. |
| `ns_dtype` | `torch.bfloat16` | Upstream Newton-Schulz working dtype. |
| `betas` | `(0.9, 0.95)` | Native AdamW moments. |
| `eps` | `1e-8` | Native AdamW epsilon. |
| `adamw_fused` | `True` | Native fused AdamW; configurable for its execution backend. |

These defaults are **not** a completed hyperparameter study. FP32 and FP64
Newton-Schulz working modes support numerical verification. FP16 Newton-Schulz
is not offered.

Construct the optimizer after moving the model to its training device. Parameter
storage is FP32, with FP64 supported for numerical tests. Mixed-precision
BF16/FP16 forward and backward use autocast with FP32 parameters, as in the current
trainer. The core rejects pure BF16/FP16 parameter storage; it does not maintain a
separate master-weight copy.

When `grad is None`, **both algorithms skip the parameter entirely**: no decay,
no momentum decay, no state creation, no AdamW step increment, and no synthesized
zero gradient. This is essential for inactive routed experts. A zero-valued
gradient is different: it is an active update, so momentum and decay still apply.

The standard parent optimizer API supports `zero_grad`, closures, learning-rate
schedulers and generic `GradScaler`. Overflow found in either group causes
`GradScaler` to skip the whole hybrid step. Gradient accumulation and global
gradient clipping remain the caller's responsibility, and will be connected to
the actual trainer in deliverable 2.

This is an eager, single-device reference. Distributed updates, CUDA graph
capture, differentiable optimizer steps and optimizer throughput tuning are not
implemented in this delivery.

## Validation performed

Environment: Python 3.12, PyTorch **2.11.0+cpu**, pytest 9.1.1, CPU with one
intra-op thread. No CUDA device was available.

- New optimizer suite: **55 passed, 1 skipped**. The skipped test requires CUDA
  and combines FP16 autocast, GradScaler and the hybrid update.
- Training regression selection: **137 passed, 15 skipped**, including the new
  optimizer tests, existing training unit tests, checkpoint/resume and persistent
  metrics integration tests.
- Static checks: Ruff passed for all changed Python files.
- Direct comparison to the pinned, eager Moonlight source: **3 kernel comparisons
  and 30 optimizer-step comparisons matched bit for bit**, including tall/wide/
  square matrices, both momentum directions, and a step with `grad=None`.

The unit suite independently checks the quintic output via SVD plus scalar
polynomial evaluation, rather than assuming an exact polar factor. It also
checks rank-deficient and zero matrices, batched independence, input/gradient
preservation, momentum and update scaling, native fused/unfused AdamW equivalence,
inactive parameters, per-parameter counters, exact checkpoint continuation,
scheduler LR changes, group additions, closures, CPU GradScaler overflow,
invalid assignments and learning on a small regression network.

The network test reaches less than 10% of its initial loss. It validates the
optimizer circuit; it is not a KimiK3 training run or a throughput measurement.

## Commands in the project environment

Run the focused suite first:

```bash
uv run python -m pytest tests/unit/training/test_optimizers.py -q
```

Run the training regression selection:

```bash
uv run python -m pytest tests/unit/training \
  tests/integration/test_checkpoint_resume.py \
  tests/integration/test_training_metrics.py -q
```

On the RTX 4050, the focused suite also runs its CUDA test. The 300M baseline and
KimiK3 tiny overfit with the final optimizer must wait for model/trainer assignment
and per-head integration; this delivery is the tested numerical foundation for
those steps.
