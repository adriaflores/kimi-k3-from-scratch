---
title: "FFN package exports"
description: "Makes DenseFFN and SituGLU available from the feed-forward package."
sourcePath: "src/kimi_k3/ffn/__init__.py"
status: published
---

The module is a single import statement at [source L1–L1](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/ffn/__init__.py#L1-L1):

```python
from .dense_ffn import DenseFFN, SituGLU
```

`DenseFFN` builds the two-projection gated feed-forward path; `SituGLU` implements its elementwise activation and optional compiled call. Importing the package exposes these names without instantiating a network. It does import `dense_ffn.py`, where the shared compiled callable is constructed lazily for execution, so the export itself does not trigger a compiled kernel run. There is no standalone forward method or numerical state in this file.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/ffn/__init__.py">__init__.py · pinned source revision</a></dd>
  </dl>
</aside>
