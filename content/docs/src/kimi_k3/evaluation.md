---
title: "evaluate_windows"
description: "Computes exploratory next-token NLL, perplexity and top-1 accuracy over nonoverlapping prepared-token windows."
sourcePath: "src/kimi_k3/evaluation.py"
status: review
---

`evaluate_windows` is a **read-only exploratory** evaluation helper. Its module docstring explicitly states that it is *not* the original pre-registered 64-window protocol and does not inherit that protocol's exclusion or identity guarantees [source L1–L5](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py#L1-L5). The function operates on a `ShardedTokenDataset`-compatible token stream and a model that returns `(logits, loss)`.

## Window contract and boundary validation

The keyword-only options include `device`, `seq_len`, `num_windows`, `start_offset=0`, `stride=None` and `precision="fp32"` [source L16–L20](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py#L16-L20). `seq_len` must be positive and no longer than the model context; `num_windows` must be positive; the start offset must be nonnegative. A reference chunkwise KDA requires `seq_len` divisible by configured chunk size when the model contains KDA layers [source L21–L27](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py#L21-L27).

With no supplied stride, `stride=seq_len`; any smaller stride is rejected to avoid **overlapping scored windows**. The expression `start_offset + (num_windows - 1)*stride + seq_len + 1` ensures there is a final next-token target inside the dataset, and model/dataset vocabulary sizes must match [source L28–L35](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py#L28-L35). These checks ensure interval nonoverlap within the selected stream; they are not a proof that the selected tokens were unseen during pretraining.

## Inference-mode scoring and reductions

After `model.eval()`, the function loops under `torch.inference_mode()` and uses `dataset.get_sequence(offset,seq_len)` for each window [source L37–L58](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py#L37-L58). Each `x` is unsqueezed to `(1,T)`; the target `y` is kept as `(T,)`. Inside the chosen autocast context, the model returns logits expected to have `(1,T,V)` shape. Those logits are converted to FP32 and checked for finiteness.

```python
loss = F.cross_entropy(logits, y, reduction="sum")
if not bool(torch.isfinite(loss)):
    raise ValueError("Nonfinite NLL")
nll_sum += float(loss.item())
correct += int((logits.argmax(-1) == y).sum().item())
n += seq_len
```

Unlike a minibatch-mean loss, `reduction="sum"` makes aggregation explicitly token-weighted. After `num_windows` windows, `n = num_windows*seq_len`. The returned `mean_nll_nats_per_token` is `nll_sum/n`, perplexity is `exp(mean_nll)`, and `top1_accuracy` is the fraction of next-token argmax hits. Perplexity is labeled `perplexity_same_tokenizer`: it is interpretable for this tokenizer's tokenization, not automatically comparable across tokenizers.

## Returned provenance fields and limitations

The returned dictionary [source L59–L75](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py#L59-L75) includes a `kind` identifying exploratory contiguous validation windows, dataset split, canonical metadata digest, offset, stride, window length/count, scored tokens, all three scoring aggregates, precision and a note distinguishing the result from pre-registration. The code makes **no model-state restore** after switching to eval mode: the caller must not assume the previous training flag is restored. It does not save window token hashes, perform exclusions, or verify global provenance beyond the dataset interface.

[test test_public_inference.py L80–L86](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py#L80-L86) checks the exact scored-token count, NLL for a controlled toy distribution, top-1 fraction and rejection of overlapping windows. Those assertions were inspected without execution. This file cannot be used as evidence of the final project's original held-out evaluation by itself.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/evaluation.py">evaluation.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_public_inference.py">test_public_inference.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
