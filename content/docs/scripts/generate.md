---
title: "generate.py — Checkpoint text and token generation"
description: "Parses text or raw-token prompts, loads a trainer-format checkpoint and emits generated tokens as JSON."
sourcePath: "scripts/generate.py"
status: published
---

[`scripts/generate.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/generate.py) exposes generation from compatible Kimi K3 trainer checkpoints without imposing the original evaluation protocol. It is an interface around [`generate_tokens()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L131-L158), not a separate decoding engine. The implementation details are documented in [inference.py](/docs/src/kimi_k3/inference/). Two input paths matter: text prompts use a Hugging Face tokenizer; raw token IDs avoid loading a tokenizer, making an offline checkpoint smoke test possible.

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## CLI selection and defaults

The [parser](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/generate.py#L18-L38) requires `--checkpoint` and **exactly one** of `--prompt TEXT` and `--token-ids COMMA_SEPARATED_IDS` (an `argparse` mutually exclusive group).

| Argument | Default | Purpose |
|---|---|---|
| `--device {auto,cpu,cuda}` | `auto` | CUDA if available, else CPU |
| `--backend {chunkwise,fused}` | `chunkwise` | Attention backend for loaded weights |
| `--precision {fp32,bf16}` | `fp32` | Inference precision |
| `--max-new-tokens` | `16` | Maximum newly sampled/selected IDs |
| `--mode {greedy,sample}` | `greedy` | Argmax or stochastic selection |
| `--temperature` / `--top-k` / `--top-p` | `1.0` / `50` / `0.95` | Stochastic sampling settings |
| `--seed` | `1337` | CPU sampling RNG seed (sample mode) |
| `--eos-id` | absent | Explicit EOS override |
| `--tokenizer` | `HuggingFaceTB/SmolLM2-135M` | Text input tokenizer repository |
| `--tokenizer-revision` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | Tokenizer revision |
| `--local-files-only` | false | Prevent tokenizer download/remote resolution |

For an illustrative offline token-prompt run, replace the path with a locally available compatible checkpoint:

```bash
python scripts/generate.py --checkpoint path/to/last.pt --token-ids 7,13,29 --device cpu --mode greedy
```

`--token-ids` means *no tokenizer is loaded*, not that the model ignores the vocabulary and context constraints. Conversely, `--prompt "..."` uses the named tokenizer at the specified revision and may need cached files or network access unless `--local-files-only` is set.

## Convert the requested prompt to IDs

The [first branch of `main()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/generate.py#L41-L58) resolves the device and rejects BF16 on non-CUDA devices. With raw input, it splits on commas, strips whitespace, converts each field to `int`, and wraps a conversion `ValueError` with a flag-specific message:

```python
if args.token_ids is not None:
    try:
        ids = [int(x.strip()) for x in args.token_ids.split(",")]
    except ValueError as exc:
        raise ValueError("--token-ids must be comma-separated integers") from exc
```

Syntax parsing does not itself range-check the token IDs. That happens later in the inference service, where empty prompts, negative/out-of-vocabulary IDs and overlong contexts are rejected. In text mode, `AutoTokenizer.from_pretrained()` uses `use_fast=True` and `tokenizer.encode(..., add_special_tokens=False)`. In particular, the CLI does **not** prepend a BOS token on its own.

## Load, validate, then generate

The [checkpoint and EOS branch](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/generate.py#L59-L69) is intentionally after prompt encoding:

```python
model, metadata = load_checkpoint_model(args.checkpoint, device=device, backend=args.backend)
if tokenizer is not None and len(tokenizer) != model.config.vocab_size:
    raise ValueError("Tokenizer vocabulary does not match checkpoint")
eos_id = args.eos_id if args.eos_id is not None else (
    tokenizer.eos_token_id if tokenizer is not None and tokenizer.eos_token_id is not None else 0
)
```

In text mode, it checks tokenizer length against checkpoint vocabulary size. In raw-ID mode no tokenizer-vocabulary equality can be checked. By default, EOS comes from the tokenizer if available, otherwise ID zero. `--eos-id` takes precedence, including if set to zero; the inference service verifies that the chosen ID is in range. The checkpoint loader requires format-version-7 model/config state, checks strict weight assignment, restores embedding/LM-head tying and sets evaluation mode. The fused backend requires CUDA; BF16 also needs supported CUDA hardware, not just a CUDA device name.

The CLI passes `mode`, temperature, top-k, top-p, seed and maximum length to [`generate_tokens()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L131-L158). That routine recomputes next-token logits on the accumulated prefix on every step, pads a reference-KDA prefix to a full chunk if necessary, applies argmax or CPU-seeded sampling, and stops early at EOS. It has **no sliding context window or KV-cache reuse**. Even with early stopping, it rejects requests where `len(prompt_ids) + max_new_tokens` exceeds model `max_seq_len` (and chunk padding may impose another limit). Greedy mode ignores the sampling parameters; in sample mode invalid temperature/top-k/top-p raise `ValueError`.

## JSON contract and side effects

After generation, the [final block](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/generate.py#L70-L84) prints an indented JSON object containing checkpoint metadata, selected device, precision, `prompt_token_ids` and `generated_token_ids`. The latter contains **new tokens only**, not the entire prompt-plus-completion sequence. Text input adds `generated_text`, decoded **only from the new IDs**, with special tokens skipped.

```python
if tokenizer is not None:
    result["generated_text"] = tokenizer.decode(generated, skip_special_tokens=True)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

There is no `--output` option, persistent generation state, checkpoint modification or dataset access. Loading local weights is read-only; tokenizer retrieval may populate an external Hugging Face cache. Model device memory usage can be substantial, and sampling is seeded only at the CPU token-selection generator level; the CLI does not claim full-system deterministic execution.

## Verification and limits

The inspected [`test_public_inference.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py#L15-L52) constructs a tiny trainer-format checkpoint, checks weight tying and finite next-token logits, asserts that a one-token greedy continuation matches argmax, rejects an overlong request, and tests selection with a fixed top-k and invalid top-p. Those assertions verify core primitives, not the text-tokenizer network path or a subprocess run of this CLI. All descriptions here are based on pinned source and inspected tests, not a new generation trial.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/generate.py">generate.py · pinned revision</a></dd>
    <dt>Implementation</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py">inference.py</a> · loading, padding, selection and context limits</dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py">public inference regression tests</a> · inspected, not executed</dd>
  </dl>
</aside>
