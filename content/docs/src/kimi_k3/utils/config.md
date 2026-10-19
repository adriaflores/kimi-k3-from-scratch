---
title: "load_config"
description: "Loads a YAML mapping into a dataclass-compatible configuration with exact field-name checking."
sourcePath: "src/kimi_k3/utils/config.py"
status: published
---

`load_config(path, config_cls)` reads a YAML document and constructs the requested configuration class. It is an intentionally strict **field-name gate**, not a generic nested-schema parser or automatic type coercer. Consumers include model and training CLI entry points that pass their dataclass definitions.

## Read and parse the mapping

The signature and file read appear at [source L11–L18](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/config.py#L11-L18).

```python
with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
```

Here the snippet is whitespace-normalized for display; the original uses tabs for indentation. `Path(path)` admits string or `Path` inputs, `safe_load` parses standard YAML values, and non-mapping top-level data raises a `ValueError`. File and YAML syntax errors are not silently caught or rewritten.

## Demand an exact dataclass field set

The decisive set operations are [source L20–L34](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/config.py#L20-L34).

```python
expected = {field.name for field in fields(config_cls)}
provided = set(data)

missing = expected - provided
unknown = provided - expected
```

`fields(config_cls)` enumerates the declared dataclass fields. If even one field is absent, the function raises with the sorted `missing` names. If extra keys are present, it separately reports the sorted `unknown` names. **Dataclass defaults are not treated as permission to omit keys**: the comparison requires exact equality of the two name sets. Python raises the underlying `TypeError` if `config_cls` is not a supported dataclass type.

## Construction is the final validation stage

Only after key checking does [source L36–L36](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/config.py#L36-L36) execute `return config_cls(**data)`. Values are passed through as parsed, without manual annotations inspection or coercion. Further range, membership and cross-field validation belongs to the dataclass's constructor or `__post_init__` (see `KimiK3Config` in `model.py` and `TrainConfig` in `trainer.py`). A provided YAML list can be normalized later by its target configuration class; `load_config` itself does not normalize it.

## Verification and limitations

The integration checks in [test test_moe_grouped_vs_naive.py L208–L223](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py#L208-L223) assert that a valid saved configuration round-trips and that unknown and missing keys are rejected. The configuration unit tests [test test_config.py L17–L38](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_config.py#L17-L38) exercise model-level constraints after construction, not YAML parsing per se. These assertions were inspected but not run in this documentation session. There is no recursive schema validation or protection against a caller passing an unsuitable class.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/config.py">config.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py">test_moe_grouped_vs_naive.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_config.py">test_config.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
