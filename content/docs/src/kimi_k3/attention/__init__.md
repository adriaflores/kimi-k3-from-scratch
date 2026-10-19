---
title: "Attention package exports"
description: "Re-exports the readable KDA implementations and GatedMLA from the attention package."
sourcePath: "src/kimi_k3/attention/__init__.py"
status: review
---

This package initializer exposes three named implementations through `kimi_k3.attention`: `KDARecurrent`, `KDAChunkwise`, and `GatedMLA`. Its full implementation consists of three relative imports [source L1–L3](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/__init__.py#L1-L3).

## Public import surface

```python
from .kda.recurrent import KDARecurrent
from .kda.chunkwise import KDAChunkwise
from .gated_mla import GatedMLA
```

A `from kimi_k3.attention import KDAChunkwise` resolves the reference KDA implementation from `attention/kda/chunkwise.py`, while `KDARecurrent` provides token-by-token reference behavior. `GatedMLA` is implemented in `attention/gated_mla.py`. The imports are eager: loading this subpackage loads those Python modules.

## What is not exported here

`FusedKDAChunkwise` is **not** a package-level export. The model factory imports it lazily when `kda_implementation == "fused"` [test test_kimi_k3.py L57–L62](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_kimi_k3.py#L57-L62); see `model.py` and `attention/kda/fused_chunk.py` for the selection and CUDA/FLA dependency. No `__all__` is declared here, and no attention computation or state management takes place in the initializer.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/attention/__init__.py">__init__.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_kimi_k3.py">test_kimi_k3.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
