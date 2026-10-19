---
title: "MoE package export"
description: "Re-exports StableLatentMoE as the package-level mixture-of-experts module."
sourcePath: "src/kimi_k3/moe/__init__.py"
status: published
---

The entire file is [source L1–L1](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/__init__.py#L1-L1):

```python
from .stable_latent_moe import StableLatentMoE
```

This enables the decoder to import `StableLatentMoE` from `kimi_k3.moe`. The class, its `QuantileBalancer`, grouped expert dispatch and mutable balancing buffers are implemented in `moe/stable_latent_moe.py`. The initializer creates no experts, makes no routing choices and declares no `__all__`.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/moe/__init__.py">__init__.py · pinned source revision</a></dd>
  </dl>
</aside>
