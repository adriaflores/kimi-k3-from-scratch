---
title: "Training package export"
description: "Exposes the unified MuonAdamW optimizer through the training package."
sourcePath: "src/kimi_k3/training/__init__.py"
status: review
---

The package declares its scope in a docstring and performs one explicit re-export [source L1–L2](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/__init__.py#L1-L2):

```python
from .optimizers import MuonAdamW as MuonAdamW
```

The alias is the same class object imported from `training/optimizers.py`, not a second optimizer implementation. It exposes neither `TrainConfig` nor the training loop, which are imported directly from `training/trainer.py`. There is no training configuration, I/O or mutation of model parameters in this initializer.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/__init__.py">__init__.py · pinned source revision</a></dd>
  </dl>
</aside>
