---
title: "Checkpoint loading and token generation"
description: "Provides strict read-only model loading, next-token scoring, and greedy or filtered sampling."
sourcePath: "src/kimi_k3/inference.py"
status: review
---

The functions in `inference.py` operate on trainer-format model checkpoints **without** restoring optimizer state or training cursors. They are not tied to a particular saved model hash and should not be confused with the original pre-registered evaluation pipeline. The central interface is `load_checkpoint_model(...)` followed by `next_token_logits(...)` or `generate_tokens(...)`.

## Device and arithmetic policy

`resolve_device(name)` chooses CUDA if `name == "auto"` and CUDA is available, otherwise CPU. An explicit unavailable CUDA device raises `ValueError` [source L16–L22](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L16-L22). `inference_autocast(device,precision)` supports `"fp32"` as a null context and `"bf16"` only on a supported CUDA device; it rejects FP16 and BF16-on-CPU in this public inference API [source L25–L32](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L25-L32). This policy is narrower than the training module's mixed-precision options.

## Strict checkpoint materialization without random-weight duplication

`load_checkpoint_model(checkpoint_path,device,backend)` [source L35–L79](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L35-L79) accepts `backend="chunkwise"` or `"fused"`, with fused limited to CUDA. It loads a **trusted local** checkpoint using `torch.load(..., weights_only=True, mmap=True, map_location="cpu")`, requires trainer `format_version == 7`, and demands both `model_config` and `model` dictionaries. The call reconstructs a `KimiK3Config` for the training backend and a second effective config for the requested inference backend.

```python
with torch.device("meta"):
    model = KimiK3(effective)
model.load_state_dict(weights, strict=True, assign=True)
```

The `meta` constructor avoids allocating a redundant randomly initialized real model before reading a large checkpoint. `assign=True` fills it from stored weights. A subsequent check rejects any parameters still on `meta`; the code explicitly **re-ties** `lm_head.weight` to `embed_tokens.weight` after assignment, switches to eval mode, moves the model to `device` and returns model plus metadata. The metadata names the training and inference KDA backends separately. There is no optimizer restoration and no guarantee that an arbitrary checkpoint is scientifically the same as the project's pre-registered run.

## Next-token logits and reference-backend padding

`next_token_logits` [source L82–L110](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L82-L110) accepts a nonempty Python list of valid integer token IDs with length at most `max_seq_len`. It validates `pad_id` and, **only** for reference chunkwise KDA in models containing KDA layers, adds right-side padding until the length is divisible by `kda_chunk_size`. If that would exceed model context length, it errors instead of truncating. The model processes the padded tensor, but the function selects logits from `len(token_ids)-1`, the **last real prompt token**.

Right padding is safe for this readout because the model is causal; generated suffix slots cannot affect the earlier queried position. The method returns one `(V,)` FP32 tensor and rejects unexpected shape or nonfinite logits. It recomputes the **whole supplied prefix** on every call; there is no KV or recurrent-state cache.

## Greedy and seeded filtered sampling

`select_token(logits,mode,temperature,top_k,top_p,generator)` [source L113–L128](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L113-L128) returns `argmax` for greedy selection, without applying temperature/top-k/top-p. Sampling requires positive temperature and top-k, and `0 < top_p <= 1`. It detaches logits, converts them to FP32 on CPU, divides by temperature, and takes up to `top_k` candidates.

```python
excluded = (probs.cumsum(0) - probs) >= top_p
probs = probs.masked_fill(excluded, 0)
probs = probs / probs.sum()
```

After softmax on the selected scores, that mask retains the **smallest nonempty leading prefix** whose cumulative probability reaches `top_p`; it excludes subsequent candidates and renormalizes. `torch.multinomial` uses an optional CPU `Generator`, so a caller can obtain repeatable sampling from a fixed seed and fixed logits without depending on the model's compute device.

## Generation loop and stopping rules

`generate_tokens` [source L131–L158](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py#L131-L158) validates a positive `max_new_tokens`, allowed mode, EOS ID, sampling arguments and enough remaining context for the *entire requested maximum*. It creates a CPU RNG seeded only for sampling, copies the prompt IDs, and loops over `next_token_logits`, selection and append. It stops at EOS or after the requested token count. The returned list contains **new token IDs only**, not the prompt or decoded text. This implementation does not slide context forward if the limit is reached, even if early EOS might have occurred in a shorter run.

## Verification

[test test_public_inference.py L26–L50](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py#L26-L50) checks strict checkpoint restoration, shared embedding/head weights, next-token shapes, greedy results and invalid argument rejection. [test test_inference_causality.py L13–L35](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_inference_causality.py#L13-L35) checks causal-prefix equivalence and reproducibility under a fixed generation seed. These tests were inspected but not rerun, and they make no tokenizer-decoding guarantee: text conversion is the responsibility of the calling script.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/inference.py">inference.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py">test_public_inference.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_inference_causality.py">test_inference_causality.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
