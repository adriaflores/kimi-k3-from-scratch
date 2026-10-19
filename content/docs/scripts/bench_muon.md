---
title: "bench_muon.py — Muon optimizer microbenchmark"
description: "Compares synthetic CUDA optimizer-step modes and checks Newton–Schulz arithmetic across logical parameter layouts."
sourcePath: "scripts/bench_muon.py"
status: review
---

[`scripts/bench_muon.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py) is an isolated optimizer benchmark. It uses the **real model's parameter-group geometry**, but replaces every parameter and gradient with seeded synthetic values. It times `MuonAdamW.step()` under two implementation choices and checks the underlying Newton–Schulz arithmetic against the eager reference. There is **no forward pass, backward pass, token processing or end-to-end pretraining throughput measurement**. The numerical routines are documented separately in [optimizers.py](/docs/src/kimi_k3/training/optimizers/) and [parameter_groups.py](/docs/src/kimi_k3/training/parameter_groups/).

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## Blueprint: preserve model geometry without allocating weights

The script imports `KimiK3`, `KimiK3Config`, the hybrid [`MuonAdamW`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py) and [`build_parameter_groups()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py). Its [`_blueprint()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py#L30-L38) first loads a model config but deliberately switches KDA to `chunkwise` for graph construction:

```python
config = replace(load_config(path, KimiK3Config), kda_implementation="chunkwise")
with torch.device("meta"):
    model = KimiK3(config)
groups = build_parameter_groups(model, 0.1)
```

This preserves parameter dimensions and grouping even if the requested configuration uses fused KDA, without requiring that fused FLA/Triton implementation to be constructed. It then replaces each group's parameter objects with *shape tuples*, retaining metadata such as its `algorithm`, reshaping conventions and Muon split sizes. These metadata affect optimizer behavior but do not depend on numerical model weights. Weight decay `0.1` is a fixed blueprint setting in this script, **not** a CLI-controlled experiment variable.

[`_materialize()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py#L41-L51) creates fresh CUDA parameters and gradients with a seeded device generator:

```python
parameter = torch.nn.Parameter(torch.randn(shape, device=device, generator=generator) * 0.02)
parameter.grad = torch.randn(shape, device=device, generator=generator) * 0.01
```

Every parameter is active, unlike a sparse expert-routing training step. The synthetic initialization makes reference/candidate *starting arrays* comparable; it does not reproduce model-trained gradient distributions. Each timing mode gets its own new parameter values, momentum state and AdamW state with the same seed.

## Measured unit: an entire hybrid optimizer step

[`_measure()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py#L54-L97) constructs `MuonAdamW` with `ns_steps=5`, BF16 Newton–Schulz internal dtype, fused auxiliary AdamW, and the chosen compilation/batching controls. Batching is bounded by both `batch_size` and `batch_max_mb * 1024**2` estimated temporary bytes. When compiled mode is selected, it clears the Newton–Schulz compilation cache before measurement so the first step captures relevant first-use costs.

The timing procedure has **three distinct phases**. First it synchronizes CUDA and resets peak-memory statistics, then times one optimizer step including synchronization and stores `first_step_seconds`. Next it performs `warmup` additional steps without recording individual measurements. Finally it measures `repeats` further steps, using both a CPU `time.perf_counter()` wall interval and CUDA events:

```python
start = time.perf_counter()
begin.record()
optimizer.step()
end.record()
end.synchronize()
walls.append((time.perf_counter() - start) * 1000)
gpu_times.append(begin.elapsed_time(end))
```

`wall_ms_*` includes Python launch and waiting overhead; CUDA-event time measures elapsed event time on the GPU timeline. `cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes` are peaks from the first step through timing, not a persistent-weight-only model memory estimate. First compilation may be expensive and is explicitly kept separate from warm and measured updates. All steps repeatedly apply the **same fixed gradients** to evolving parameters. This isolates optimizer overhead, but means later parameter values differ from their original initialization. After each mode the optimizer/group objects are deleted, garbage collected, and the CUDA cache emptied so only one mode's model-sized storage is intended to be resident at once.

The result also records compilation cache statistics (`hits`, `misses`, etc.) and an NS call count per update when applicable. That count is inferred from cache use, not from profiler kernel launches. It is null for eager mode.

## Independent arithmetic check by layout

Timing alone cannot establish that compiled/batched Newton–Schulz handles all parameter layouts correctly. After timing, [`_verify_core()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py#L100-L133) iterates only `algorithm == "muon"` groups and deduplicates logical `(shape, stride)` pairs, including reshaped head views and splits such as gate/up or K/V blocks. It derives representative lane capacity from `_batch_capacity()` and the configured memory/size bounds. The arithmetic comparison is:

```python
eager = torch.stack([zeropower_via_newton_schulz(item) for item in part])
compiled = (zeropower_via_newton_schulz(part, compiled=True) if lanes > 1
    else torch.stack([zeropower_via_newton_schulz(part[0], compiled=True)]))
torch.testing.assert_close(compiled, eager, rtol=0.03, atol=0.03)
```

The same logical random matrices are passed to both implementations for each checked layout. The report stores maximum absolute and relative L2 differences, plus shape/stride and number of parameter lanes. The tolerance is explicitly nonzero; the comparison does **not** claim bitwise parity. The verification checks the Newton–Schulz core rather than proving full-optimizer equivalence across a training trajectory.

## CLI choices and execution order

The [argument parser](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py#L136-L158) requires `--model-config`. `--device` defaults to `cuda` and is rejected unless its type is CUDA and CUDA is available. Other defaults: `--warmup 5`, `--repeats 30`, `--seed 1337`, `--compare compile`, `--batch-size 32`, `--batch-max-mb 64`, and `--order reference-first`. The `--compare` choices are `compile` or `batch`. The `--order` choices are `eager-first`, `compiled-first`, `reference-first`, `candidate-first`; the first and third run the reference first, the other two run the candidate first. `--output` optionally writes a JSON report.

The [mode mapping](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py#L159-L175) differs by comparison:

| `--compare` | Reference | Candidate |
|---|---|---|
| `compile` | eager, individual NS | compiled, individual NS |
| `batch` | compiled, individual NS | compiled, batched NS |

The command refuses fewer than one warmup update, fewer than two repeats, nonpositive batch bounds, or `TORCH_COMPILE_DISABLE=1`. It prints phase messages to stderr, measures each mode in the requested order, runs core verification, and writes the report at the end. **Running the command compiles code, performs many optimizer updates and can use substantial GPU memory and time**; `--help` is the safe inspection command:

```bash
python scripts/bench_muon.py --help
```

## Output, interpretation and evidence

The [JSON report](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py#L176-L189) records PyTorch/CUDA versions, GPU name, config path, seed, comparison/order, batch constraints, `all_gradients_active: true`, NS steps and BF16 dtype, total parameter count, both mode measurements, verification tolerances/layout differences, and

$$\text{optimizer\_wall\_speedup}=\frac{\text{reference median wall-ms}}{\text{candidate median wall-ms}}.$$

This ratio is *relative microbenchmark optimizer-step latency*, not training tokens/s, iteration throughput or a measured speedup of a complete language-model run. The command prints JSON to stdout and optionally **overwrites** the requested `--output` file after creating parents. It does not modify model configs or training checkpoints.

The inspected optimizer tests exercise [batched Newton–Schulz and independent matrices](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_optimizers.py#L25-L80), [hybrid optimizer state and updates](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_optimizers.py#L81-L149), and [compiled/batch CLI recipe propagation](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_recipe.py#L163-L190). These provide supporting implementation coverage; they are not a stored benchmark run and do not assert a particular speedup. No GPU benchmark was executed for this documentation page.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/bench_muon.py">bench_muon.py · pinned revision</a></dd>
    <dt>Implementation</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/optimizers.py">optimizers.py</a> and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/parameter_groups.py">parameter_groups.py</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_optimizers.py">optimizer unit checks</a>; source inspected, not benchmarked</dd>
  </dl>
</aside>
