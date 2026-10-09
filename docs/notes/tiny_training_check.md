# Check the tiny training circuit

Before renting a GPU, check that the real KimiK3 can learn next-token targets,
evaluate without changing its state, and continue from a checkpoint.
Use `configs/model/kimi_k3_tiny.yaml`: vocabulary 49,152, hidden size 32,
KDA/KDA/KDA/MLA, one dense FFN followed by three MoE FFNs, two AttnRes blocks,
and context 128. `--smoke-test` selects a token-local toy model.

## Small automated check

From the repository root, without downloaded data or a GPU:

```bash
uv run python -m pytest -q -s tests/integration/test_tiny_training.py \
  tests/integration/test_train_cli.py \
  tests/integration/test_checkpoint_resume.py::test_cli_resume_in_fresh_process_matches_continuous_run \
  tests/integration/test_checkpoint_resume.py::test_evaluation_preserves_model_and_rng_on_success_and_failure
```

This selects 12 CPU cases. The new test loads the unmodified tiny model YAML
and the overfit recipe, then uses **64 updates at LR `1e-2`** to keep the
synthetic regression check short. This test-only override does not change the
presets' LR `3e-4` or propose a pretraining LR. It writes normal run artifacts
under pytest's temporary directory and prints a `tiny_next_token` summary.

The training stream repeats `[7, 13, 7, 29]`. Token 7 has two different
successors, so a token-local lookup cannot memorize all targets. The loader
reads 129 tokens into 128 inputs and their one-token-shifted targets.
Validation uses a separate stream with different token IDs.

| Check | Acceptance |
| --- | --- |
| Forward/backward | Finite logits, loss and produced gradients; nonzero gradients and changed weights in every attention/FFN family, embeddings and final AttnRes projection. |
| Next-token memorization | The trainer receives the exact shifted targets on every forward; fixed-batch loss `< 0.1` and all 128 next tokens predicted correctly. |
| Evaluation | Separate validation loader; finite loss; model parameters/buffers, QB biases, training cursor and RNG preserved. |
| Resume | Existing fresh-process tests compare continuous and interrupted training exactly: weights/QB, Muon momentum, AdamW, scaler, counters, both cursors and RNG. Metrics retain the previous history and add a session. |

The new learning test and CLI construction test use the shipped tiny YAML.
The integrated-recipe regression also checks exact resume with the shipped tiny
YAML. Additional resume regressions use a smaller mixed KDA/MLA/MoE architecture; they cover both fixed and successive batches, FP32 and an
accumulated BF16 recipe. The RNG isolation regression also uses a stochastic
model to test Python, NumPy and PyTorch RNG restoration on success and failure.

Low training loss here demonstrates **memorization of one synthetic batch**.
Validation loss only has to be finite; it may increase. This does not establish
generalization, FineWeb quality, GPU throughput or the feasibility of 300M/6B.

## Two short CLI segments on the prepared data

For Windows/Git Bash, run these commands from the repository root. Choose a
fresh run directory; use the same software, device, precision, thread settings
and recipe in both segments. Replace `cuda` with `cpu` for a CPU check.
Leave profiling disabled throughout this circuit.

First, stop at a completed checkpoint after 20 successful updates:

```bash
uv run python scripts/train.py \
  --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml \
  --device cuda --deterministic \
  --run-dir results/training/tiny-circuit
```

Then rebuild the model and continue to **40 total updates**, adding 20:

```bash
uv run python scripts/train.py \
  --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/smoke.yaml \
  --device cuda --deterministic --max-steps 40 \
  --run-dir results/training/tiny-circuit \
  --resume results/training/tiny-circuit/last.pt
```

The first segment should report step 20 and 2,560 tokens. Resume should start
there, finish at step 40 and 5,120 tokens, and append a second session to the
same `run.json` and `metrics.jsonl`. The training cursor should be 5,120 and
the validation cursor 128 on this large, unchanged dataset. This exercises
stopping at an update boundary; the exact trajectory comparison is automated
above. `--max-steps` always specifies the total target, including saved steps.

Inspect the result with this Git Bash snippet:

```bash
uv run python - <<'PY'
import json
import math
from pathlib import Path
import torch

directory = Path("results/training/tiny-circuit")
manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in
           (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
assert saved["format_version"] == 7
assert saved["step"] == 40 and saved["tokens_seen"] == 5120
assert saved["skipped_updates"] == 0
assert saved["loaders"]["train"]["position"] == 5120
assert saved["loaders"]["val"]["position"] == 128
assert len(manifest["sessions"]) == 2
assert manifest["sessions"][1]["start_state"]["step"] == 20
updates = [record for record in records if record["event"] == "train"]
assert [record["step"] for record in updates] == list(range(1, 41))
assert all(record["run_id"] == manifest["run_id"] for record in records)
assert all(math.isfinite(record[key]) for record in updates
           for key in ("train_loss", "grad_norm_mean", "grad_norm_max"))
assert all(record["nonfinite_grad_norms"] == 0 for record in updates)
validation = [record for record in records if record["event"] == "val"]
assert [record["step"] for record in validation] == [20, 40]
assert all(math.isfinite(record["val_loss"]) for record in validation)
print("PASS: 40 updates, 5120 tokens, two sessions, preserved metric history.")
PY
```

## Fixed-batch check on FineWeb

Reuse an existing overfit result if its model, recipe, data and logs establish
these checks. A 2,000-update run is not required again merely to close this
milestone. If that evidence is missing, start with a 200-update budget:

```bash
uv run python scripts/train.py \
  --data-dir C:/datasets/fineweb_edu_6b \
  --model-config configs/model/kimi_k3_tiny.yaml \
  --training-config configs/training/overfit.yaml \
  --device cuda --deterministic --max-steps 200 \
  --run-dir results/training/tiny-overfit
```

Look for a clear decrease from the starting loss and finite gradients.
For memorization, aim for fixed-batch train loss `< 0.1`; 200 updates are an
initial inspection budget, not a convergence guarantee. If more are needed,
resume the same recipe/checkpoint with a larger total `--max-steps`, up to
the preset's 2,000-update budget. Keep the batch dimensions, precision and
execution settings compatible. Validation loss need not improve.

Preserve `run.json`, `metrics.jsonl` and `last.pt` for each run. The
[monitoring guide](training_monitoring.md) defines the recorded fields and
checkpoint semantics. This milestone closes the tiny functional circuit;
candidate-size memory, speed and sustained stability belong to the next test.
