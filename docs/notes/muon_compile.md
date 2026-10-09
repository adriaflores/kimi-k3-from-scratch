# Compiled Newton–Schulz

Set `muon_compile: true` in a training YAML or pass `--muon-compile`. The default
is `false` until the target GPU comparison establishes a benefit. Both choices
implement the same Muon + AdamW + QK-Clip recipe. On CPU, the optimizer retains
the eager reference even when the requested setting is true; startup reports
the effective backend and the requested value.

Only the tensor arithmetic of Newton–Schulz is compiled with
`torch.compile(fullgraph=True, dynamic=False)`. The five iterations, coefficients,
per-matrix Frobenius normalization, BF16 working precision and logical matrix
views are retained. Momentum, Nesterov, shape scaling, weight decay, AdamW,
gradient handling and post-step QK-Clip remain outside that graph. This stage
does not group different parameters or change expert execution.

The per-call Inductor option `emulate_precision_casts=True` retains the BF16
rounding boundaries of eager pointwise operations inside fused kernels. This
avoids replacing the established BF16 recipe with higher-precision intermediate
arithmetic on the target CUDA route; global compiler configuration is unchanged.
CPU Inductor still optimizes some BF16 casts differently, so CPU verification
allows its measured rounding differences. Production CPU optimizers stay eager.

Each shape, stride, device, input dtype and NS arithmetic setting has a cached
specialization. Separate Python code objects avoid exhausting Dynamo's per-code
recompilation limit when a model has many matrix shapes. The cache is bounded
to 128 entries. LR changes, new tensor contents and `grad=None` do not create
new core specializations. No compiler cache is serialized.

Compiler failures propagate with PyTorch's diagnostics. There is no silent eager
fallback in this route; `TORCH_COMPILE_DISABLE=1` is rejected when the compiled
core is called. On Windows, use the repository's PyTorch CUDA/Triton setup and
the C++ build tools already used for fused KDA. The first run may spend minutes
compiling. Fusion can change floating-point rounding, so eager/compiled updates
are compared with dtype-specific tolerances rather than bitwise equality.
Same-backend deterministic resume is tested separately. Keep `muon_compile`
unchanged when resuming, including on CPU.

## Isolated optimizer comparison on the 4050

```bash
uv run python scripts/bench_muon.py \
  --model-config configs/model/kimi_k3_300m.yaml --device cuda \
  --warmup 5 --repeats 30 --output results/training/muon_bench_a.json

uv run python scripts/bench_muon.py \
  --model-config configs/model/kimi_k3_300m.yaml --device cuda \
  --warmup 5 --repeats 30 --order compiled-first \
  --output results/training/muon_bench_b.json
```

Commands use Bash line continuations; in PowerShell put each command on one line
or use PowerShell's continuation syntax. If checking tooling first, substitute
`configs/model/kimi_k3_tiny.yaml`.

The benchmark constructs the real parameter grouping and times complete
`MuonAdamW.step` calls with fixed synthetic gradients for **all parameters**.
Both modes start from identical generated weights/gradients and fresh state;
one mode occupies GPU memory at a time. First-step time (including lazy
compilation/state allocation) is reported separately from warm measurements.
The JSON includes synchronized wall medians, CUDA-event medians, peak memory,
compile-cache counts and the wall-time speedup. CUDA events also include gaps
between launches; synchronized wall time is the main optimizer comparison.
After timing it verifies eager/compiled BF16 outputs for every logical shape and
stride (including K/V slices) with `rtol=0.03, atol=0.03`, recording maximum
absolute and relative L2 differences. A numerical mismatch fails the benchmark.

This excludes forward/backward and QK-Clip and may activate more experts than
a real batch. It measures optimizer cost, not training throughput. Reversing
mode order helps reveal thermal or run-order effects. Keep power mode and other
GPU load consistent. Do not count the first compilation as steady training time.

## Complete unprofiled comparison

Repeat **the same command and data used for the original profiling**, once with
`--no-muon-compile` and once with `--muon-compile`, in different fresh run folders.
Keep model, seed, precision, accumulation, sequence length, LR and all clipping
settings identical. Remove `--profile`. This is the decisive throughput check.

For a bounded reproducible starting point:

```bash
uv run python scripts/train.py --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_300m.yaml \
  --training-config configs/training/acceptance.yaml --device cuda \
  --precision bf16 --no-muon-compile --run-dir results/training/muon_eager

uv run python scripts/train.py --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_300m.yaml \
  --training-config configs/training/acceptance.yaml --device cuda \
  --precision bf16 --muon-compile --run-dir results/training/muon_compiled

uv run python scripts/summarize_training_run.py results/training/muon_eager \
  --after-step 16 --expect-step 64
uv run python scripts/summarize_training_run.py results/training/muon_compiled \
  --after-step 16 --expect-step 64
```

Compare `steady_train_tokens_per_second`, losses, finite gradient norms, skips,
QK clipping metrics and peak CUDA memory. The summary excludes entire logging
windows touching warmup. If new specializations still appear after update 16,
extend both runs with the same fixed schedule horizon and exclude a longer
warmup. The acceptance command is an example; its settings may differ from the
original profile and cannot be compared directly with that capture.

## Verification

```bash
uv run pytest -q tests/unit/training/test_muon_compile.py \
  tests/unit/training/test_optimizers.py tests/unit/training/test_per_head_muon.py \
  tests/integration/test_tiny_training.py tests/integration/test_checkpoint_resume.py
```

Real CPU Inductor tests verify the compiled core, zero updates, input
immutability, independent heads, tall/noncontiguous views and dtype behavior.
A counting backend checks more than eight static layouts and warm reuse without
graph breaks. CUDA tests cover complete hybrid steps, split heads, LR changes,
missing gradients, and tiny learning/resume when a GPU is available. CPU-only
results leave CUDA performance and compiler compatibility for the 4050 run.

PyTorch references: [compiling optimizers](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html)
and [torch.compile](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html).
