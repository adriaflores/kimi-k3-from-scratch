---
title: "inspect_config.py — Model parameter accounting"
description: "Instantiates a Kimi K3 configuration to report unique total parameters, estimated active parameters and component budgets."
sourcePath: "scripts/inspect_config.py"
status: published
---

[`scripts/inspect_config.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py) answers a static architecture-budget question: *how many unique stored parameters does this configured model have, and how many are on the selected-expert path for one token?* It builds the real module graph but performs no forward pass, optimization or data loading. By default, construction occurs on PyTorch's `meta` device so parameter tensors have shapes without allocating their numerical storage. The counted module graph belongs to [model.py](/docs/src/kimi_k3/model/) and [stable_latent_moe.py](/docs/src/kimi_k3/moe/stable_latent_moe/).

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## Invocation and configuration

The [CLI](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L38-L59) requires the positional model YAML `config`. `--device {meta,cpu}` defaults to `meta`; `--by-layer` adds one row for each decoder layer; `--show-config` prints every normalized `KimiK3Config` dataclass field before the report. There is no CUDA option and no checkpoint input.

A read-only example from the repo root uses the existing model configuration:

```bash
python scripts/inspect_config.py configs/model/kimi_k3_300m.yaml --by-layer
```

The YAML is parsed using [`load_config()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/config.py) and the normal model constructor. The script reports what *this implementation* would instantiate, rather than independently estimating dimensions from a paper table. `--device cpu` constructs real initialized parameters in RAM and can be expensive for very large configs; `meta` avoids weight storage but still builds Python module objects and shapes.

## Count unique parameter objects, not repeated references

[`unique_numel()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L62-L76) counts the number of elements of each distinct `Parameter` object once:

```python
for p in parameters:
    pid = id(p)
    if pid not in seen:
        seen.add(pid)
        total += p.numel()
```

This matters for tied token embeddings and LM head: the same parameter can be reachable through two module paths but should occupy one parameter budget. The routine deduplicates by Python identity, not by numerical value or by shared storage between distinct objects. It uses `numel()`, independent of dtype and device. `module_params()` applies it to any module's parameter iterator.

## Count routed experts as active only when selected

The [MoE helper](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L79-L126) expects the project's current `StableLatentMoE` representation: routed experts are a `nn.ModuleList` named `experts`, and `top_k` is defined on the MoE module. Nonmatching modules contribute zero routed-expert parameters. Equal expert sizes are an explicit invariant; unequal sizes raise rather than produce a misleading total:

```python
sizes = [module_params(expert) for expert in experts]
if len(set(sizes)) != 1:
    raise RuntimeError(
        "Routed experts have unequal parameter counts; "
        "the active-parameter estimate needs updating."
    )
```

With `E` routed experts, `k = top_k`, and `P` parameters per routed expert, the helper returns `(E*P, k*P)`. The total parameter count includes all experts; the *per-token active* estimate subtracts `(E-k)*P` for each MoE layer. Both router weights and shared experts **remain counted as active**. Dense FFNs, attention and embeddings are unchanged. This is a count of structurally active model parameters for a token, not an exact count of floating-point operations, unique tensors touched over a batch, or memory allocated during inference.

The [model-level count](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L118-L126) subtracts inactive routed experts directly from the unique model total. The [per-layer helper](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L105-L115) instead walks each layer's child modules and subtracts inactive routed-expert elements, so the same rule underlies both summaries.

## Categorize parameters without double-counting tying

[`classify_parameter()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L129-L159) assigns names to buckets using module paths and the loaded configuration. Its `layers.<index>.attn` entries are labeled KDA or Gated MLA according to `layer_types[index]`, while `layers.<index>.ffn` is Dense FFN for layers before `num_dense_layers` and Stable LatentMoE thereafter. Other decoder-layer parameters are classed as decoder norms/AttnRes; final norm and output AttnRes have their own bucket. Embedding/LM head tying is captured by the `embed_tokens` path. A defensive `lm_head` bucket exists if the implementation ever changes tying.

The [breakdown](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L162-L179) iterates `model.named_parameters(remove_duplicate=True)` for unique names, and subtracts *only* inactive MoE routed experts from the Stable LatentMoE active bucket. Bucket labels are a view over parameter names; they do not represent separately instantiated submodels. In particular, the output is sensitive to module naming and `num_dense_layers`. If model structure changes, the classification rules must be reviewed.

## Formatting and printed sections

[`print_report()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L215-L288) prints the architecture summary (vocabulary, width, layer mix, expert counts, max context and KDA settings), the parameter budget and component breakdown. It reports four different counts:

| Field | Exact interpretation |
|---|---|
| Total | Distinct model parameter elements |
| Active / token | Total minus all unselected routed experts, based on top-k |
| Trainable | Unique parameter elements with `requires_grad=True` |
| Buffers | Sum of registered buffer elements, separate from parameter elements |

The `Active / total` percentage uses the exact integer counts internally. Counts are displayed both as rounded K/M/B suffixes and exact comma-separated totals for main budget rows. The component `% total` column uses **total parameters in that bucket**, not active parameters. `--by-layer` prints index, attention type, Dense/MoE marker, total and active elements for each layer, but does not allocate or score a token.

The "Parameter-only memory" block multiplies **total** parameter elements by four bytes for FP32 or two for BF16/FP16 and presents MiB/GiB using powers of 1024. It is a storage-only estimate: no gradients, optimizer slots, activations, allocator overhead, temporary buffers or shared memory. It is not a claim that an entire model will fit in that amount of device RAM.

The [last entry point](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py#L291-L312) optionally prints normalized dataclass fields, constructs with `torch.device("meta")` or on CPU, prints the config path and invokes the report. There are no writes or generated artifacts; stdout is the only output. Invalid YAML/configuration or a model construction failure surfaces as an exception.

## Verification and boundaries

The source's correctness depends principally on [`KimiK3`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py) and [`StableLatentMoE`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py). The inspected [`test_recipe.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_recipe.py#L38-L55) asserts that optimizer groups cover tied weights once, a related deduplication invariant. The [`test_train_cli.py` real-model case](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_train_cli.py#L116-L124) checks the selected architecture configuration in a saved run. No direct `inspect_config.py` snapshot/count test was identified in the inspected suite; this page does not claim one or that any parameter report was executed.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/inspect_config.py">inspect_config.py · pinned revision</a></dd>
    <dt>Dependencies</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py">model.py</a>, <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/stable_latent_moe.py">stable_latent_moe.py</a></dd>
    <dt>Verification</dt><dd>Source-inspected tying/group tests; no dedicated parameter-report execution claimed</dd>
  </dl>
</aside>
