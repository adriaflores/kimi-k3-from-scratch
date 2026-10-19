---
title: "Utility package export"
description: "Exports load_config as the utility package configuration reader."
sourcePath: "src/kimi_k3/utils/__init__.py"
status: review
---

This one-line initializer re-exports the exact function implemented in `utils/config.py` [source L1–L1](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/__init__.py#L1-L1):

```python
from .config import load_config
```

Importing the package does not read YAML. A file is opened only when a caller invokes `load_config(path, config_cls)`. There is no separate validation or defaulting policy in this initializer.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/__init__.py">__init__.py · pinned source revision</a></dd>
  </dl>
</aside>
