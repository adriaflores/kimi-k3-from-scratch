---
title: "evaluate.py — Exploratory checkpoint evaluation"
description: "Connects a trainer-format checkpoint to prepared token windows and prints or saves next-token evaluation metrics."
sourcePath: "scripts/evaluate.py"
status: review
---

[`scripts/evaluate.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/evaluate.py) is the thin, read-only entry point for **exploratory** next-token evaluation. It chooses a prepared split and disjoint windows, loads model weights from a compatible checkpoint, delegates scoring to `evaluate_windows()`, and serializes the result. The script explicitly does **not** reproduce the project's original pre-registered held-out evaluation protocol. Its delegated components have separate Docs references for [evaluation.py](/docs/src/kimi_k3/evaluation/) and [inference.py](/docs/src/kimi_k3/inference/).

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## CLI contract

The [parser](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/evaluate.py#L20-L33) requires two paths: `--checkpoint` (trainer-format model state) and `--data-dir` (prepared shards and metadata). The remaining options are:

| Argument | Default | Meaning |
|---|---|---|
| `--split {train,val}` | `val` | Selects which metadata split to score |
| `--device {auto,cpu,cuda}` | `auto` | Uses CUDA when available, otherwise CPU |
| `--backend {chunkwise,fused}` | `chunkwise` | Chooses the KDA implementation for inference |
| `--precision {fp32,bf16}` | `fp32` | Selects no autocast or CUDA BF16 autocast |
| `--seq-len` / `--num-windows` | `128` / `4` | Tokens scored per window / number of windows |
| `--start-offset` / `--stride` | `0` / `None` | First absolute token offset / spacing; `None` means `seq_len` |
| `--output` | omitted | Optional JSON report path |

From the repository root, an illustrative command is:

```bash
python scripts/evaluate.py --checkpoint path/to/last.pt --data-dir path/to/prepared-data --split val
```

The `path/to/...` values above are placeholders, not distributed checkpoint or dataset locations. The default request scores four nonoverlapping windows of 128 target tokens each **provided the split is long enough and the checkpoint supports that context**. No CLI option asserts checkpoint identity against a fixed research run.

## Build the evaluation context

The entire [orchestration path](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/evaluate.py#L36-L45) is explicit:

```python
device = resolve_device(args.device)
dataset = ShardedTokenDataset(args.data_dir, split=args.split)
model, metadata = load_checkpoint_model(args.checkpoint, device=device, backend=args.backend)
summary = evaluate_windows(
    model, dataset, device=device, seq_len=args.seq_len,
    num_windows=args.num_windows, start_offset=args.start_offset,
    stride=args.stride, precision=args.precision,
)
```

The dataset uses `metadata.json` and the ordered shard names to expose a single logical token stream. The [checkpoint loader](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L35-L80) accepts trainer checkpoint format 7, builds the model on the `meta` device, loads weights strictly, restores tied embedding/head weights, and returns architecture and checkpoint metadata. Selecting the fused backend requires CUDA; the actual fused attention implementation has its own installation constraints. A BF16 run requires CUDA hardware supporting BF16.

The script itself does not implement loss accumulation. In [`evaluation.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py#L16-L75), every window contributes `seq_len` next-token targets: input IDs from a window and their one-position-shifted targets. The service verifies positive lengths/counts, `start_offset >= 0`, `stride >= seq_len`, stream bounds, model/dataset vocabulary agreement, chunk alignment for reference KDA, and finite logits/loss. It uses evaluation mode and `torch.inference_mode()`, not a training update.

The metrics have precise meanings. With window length `T` and window count `W`, `scored_tokens = T W`, and the reported mean is

$$\mathrm{mean\_nll}=\frac{\sum_{i=1}^{W}\sum_{t=1}^{T}-\log p(y_{i,t}\mid x_{i,\le t})}{TW},\qquad \mathrm{PPL}=e^{\mathrm{mean\_nll}}.$$

`mean_nll_nats_per_token` is the cross-entropy in natural-log units. `perplexity_same_tokenizer` must be interpreted with the **same tokenizer** and comparable evaluation setup; `top1_accuracy` is the fraction of argmax predictions equal to the target. The JSON also records offset, stride, split, dataset metadata hash, precision, scored token count and an explicit exploratory-protocol note. This is not an independently held-out, pre-registered score.

## Output and failures

The [final block](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/evaluate.py#L46-L56) wraps the checkpoint metadata, device string and evaluation summary:

```python
result = {"checkpoint": metadata, "device": str(device), "summary": summary}
output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
print(output, end="")
```

It prints JSON even when `--output` is supplied. When an output path is supplied, it creates parent directories and **overwrites** that file; without the flag it writes no report. It does not change the dataset or checkpoint. `main()` returns zero on success; uncaught file, configuration, device, precision and window errors terminate the process rather than producing a synthetic result.

## Verification and interpretation limits

[`test_evaluate_windows_exact_scored_token_count`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py#L69-L89) uses a toy model and token stream to assert exactly eight scored targets for two windows of four tokens, uniform-logit NLL equal to `ln(5)`, a known accuracy and rejection of overlapping windows. The [checkpoint fixture and loader test](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py#L15-L34) inspect strict loading and weight tying. These tests cover the delegated services, not a full subprocess invocation of this particular CLI; neither a fresh test run nor evaluation of a real checkpoint is claimed here.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/evaluate.py">evaluate.py · pinned revision</a></dd>
    <dt>Implementation</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py">evaluation.py</a> and <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py">inference.py</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py">public inference/evaluation checks</a>; source inspection only</dd>
  </dl>
</aside>
