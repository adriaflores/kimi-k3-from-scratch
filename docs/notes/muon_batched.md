# Batched Muon

The second optimization phase groups compatible active parameters before
Newton–Schulz and uses `torch._foreach_*` for momentum, Nesterov, weight decay
and update application. It reduces the number of separate NS calls and small
pointwise launches. It does not change expert forward dispatch or AdamW.

Enable `muon_batched: true` in a complete training YAML or pass
`--muon-batched`. The default remains `false` until the target GPU measurement
establishes a benefit. Combine it with `--muon-compile` to batch the compiled
NS core. Batching also works with eager NS; CPU optimizers retain eager NS
regardless of the requested compilation setting.

## Geometry and state

Buckets contain parameters from the same optimizer group with matching stored
shape, device and dtype. Group settings supply the logical view, K/V boundaries,
NS settings and update hyperparameters. Only parameters with a gradient enter
the current step's chunks. `grad=None` creates no momentum state, applies no
decay and leaves any existing state untouched. A zero gradient is an ordinary
update, including momentum and decay.

Packing adds a leading parameter axis. NS normalizes and multiplies only the
last two matrix axes, independently for each parameter, FFN gate/up matrix,
Q/K/V projection and attention head. MLA K and V rows still receive separate
NS calls and scales, including unequal key/value dimensions. Results are
reassembled into each parameter's original shape. Single-parameter chunks use
the existing individual reference.

Momentum stays in the original per-parameter buffer. AdamW and post-step
QK-Clip retain the same order and state. Gradient accumulation, AMP unscaling,
global gradient clipping and the LR schedule are unchanged. BF16 batched and
individual matrix kernels can round differently; equivalence uses numerical
tolerances, while deterministic resume uses the same execution settings.

## Bounded chunks

| Training field / CLI override | Default | Meaning |
| --- | ---: | --- |
| `muon_batched` / `--muon-batched` | `false` | Enable parameter batching and foreach updates |
| `muon_batch_size` / `--muon-batch-size` | 32 | Maximum parameters per chunk |
| `muon_batch_max_mb` / `--muon-batch-max-mb` | 64 | MiB budget for the temporary-storage estimate |

Capacity is the largest power of two allowed by the parameter count, size limit
and storage estimate. The estimate accounts for packing, parameter-precision
updates, NS matrix temporaries and Gram matrices. Partial chunks round up to
the next power of two and pad temporary lanes with zeros. Padding receives no
parameter, momentum state or decay. Variable active-expert counts therefore
use a small fixed set of compiled shapes instead of a specialization per count.

The estimate is conservative, but it is **not a hard bound on total VRAM**:
model/gradient/optimizer state, the caching allocator and GEMM workspaces are
separate. A matrix that exceeds the budget still runs individually. If the
target GPU needs smaller chunks, reduce `--muon-batch-max-mb` for a fresh run.
No packed buffers survive a step or enter a checkpoint.

For the 300M parameter blueprint with every parameter active, the default
capacities reduce the expected NS calls per update from 1,680 to 193. Expert
gate/up chunks contain eight parameters and expert down-projection chunks
contain sixteen. This is a geometry-based call count, not a measured GPU
speedup; routed training may have fewer active parameters.

The Python optimizer API uses `ns_batched`, `ns_batch_size` and
`ns_batch_max_bytes` for these settings. They belong only to Muon groups.

## Checkpoints

Trainer format stays **7**. New checkpoints save batching and both limits in
the recipe and optimizer groups; all must match when resuming. Capacity affects
kernel selection and numerical rounding, so changing these settings requires a
fresh comparison run. Older format-7 checkpoints without batch fields resume
as individual execution with the default limits (32, 64 MiB). Their saved
compilation choice is preserved. Kernels and bucket caches are rebuilt after
loading; momentum and AdamW state retain their ordinary parameter layout.

All shipped training YAMLs include the three new fields. The YAML schema is
complete, so add them to any custom training configuration as well.

## Essential 4050 comparison

The existing individual compiled acceptance run is approximately 130 training
tokens/s after warmup. Run the same recipe once with batching, in a fresh folder:

```bash
uv run python scripts/train.py --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_300m.yaml \
  --training-config configs/training/acceptance.yaml --device cuda \
  --precision bf16 --muon-compile --muon-batched \
  --run-dir results/training/runs/300m_4050_muon_batched
```

Keep data, seed, model, precision, accumulation and power mode identical. Do
not use the profiler. Compare warmed logging windows (typically steps 24–64),
losses, finite gradient norms, skipped updates, QK clipping and peak CUDA memory.
Exclude any window still compiling a new layout. Initial compilation time is
reported by the run but does not establish steady throughput. The acceptance
recipe repeats a fixed batch; this is a circuit/performance check, not a test
of generalization. CPU correctness checks do not predict the GPU speedup.

For an optional isolated optimizer measurement:

```bash
uv run python scripts/bench_muon.py \
  --model-config configs/model/kimi_k3_300m.yaml --device cuda --compare batch \
  --warmup 5 --repeats 30 --output results/training/muon_batched_bench.json
```

This compares individual compiled NS with batched compiled NS, timing complete
hybrid steps with identical synthetic gradients for all parameters. The JSON
records first-step time separately, synchronized warm wall medians, CUDA-event
medians, peak memory, compile-cache counts, NS calls per update and speedup.
After timing it checks packed compiled NS against separate eager matrices for
each distinct logical layout, including noncontiguous K/V slices. Use
`--order candidate-first` to investigate order/thermal effects if needed.
Synthetic all-active experts can differ from a real batch, and the benchmark
excludes forward/backward and QK-Clip; training throughput is the final check.

## Verification

`tests/unit/training/test_muon_batched.py` compares multiple hybrid steps against
individual FP64 execution, changing active parameters and LR. It covers 2D,
tall matrices, gate/up, KDA heads, unequal MLA K/V, noncontiguous storage,
independence, padding, zero/missing gradients, limits, state and real CPU
Inductor for packed heads. CUDA tests additionally compare compiled BF16
updates and momentum when CUDA is available.

The integration tests check tiny learning and exact same-mode FP32/BF16 resume,
old format-7 compatibility, configuration mismatch rejection and fresh CLI
processes with accumulation and forced QK clipping. GPU cases remain skipped
on CPU-only hosts. Run the existing training suite on the target environment;
the new focused unit check is `uv run pytest -q tests/unit/training/test_muon_batched.py`.

Local validation used PyTorch 2.11.0 on CPU. The training suite produced 351
passes and 38 CUDA/FLA skips, with one existing failure in
`test_shipped_tiny_memorizes_next_tokens_and_evaluates_without_mutation`:
the final evaluation predicts the first token incorrectly. The identical
failure and loss history were reproduced from the unmodified main snapshot.
The batched tiny learning/resume cases passed. After expanding checkpoint
coverage, its entire file passed (56 passes, nine GPU skips), including old
compiled checkpoints and new batched CLI processes. CUDA correctness and
performance remain to be measured on the 4050.
