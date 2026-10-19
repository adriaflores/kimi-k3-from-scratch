---
title: "Data package exports"
description: "Exposes prepared-shard readers without performing dataset preparation or tokenization."
sourcePath: "src/kimi_k3/data/__init__.py"
status: review
---

This package is the entry point for the prepared-token reader. The module docstring expressly limits its responsibility to reading already prepared shards, rather than downloading a corpus or tokenizing text [source L1–L5](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/__init__.py#L1-L5).

## Export contract

```python
from .dataset import ShardedTokenDataset, TokenDataLoader

__all__ = ["ShardedTokenDataset", "TokenDataLoader"]
```

`ShardedTokenDataset` validates the manifest and reads uint16 token ranges; `TokenDataLoader` iterates over causal batches with a resumable cursor. Their I/O, shape and state contracts belong to `data/dataset.py`, not to this export layer. The explicit `__all__` controls star imports; it does not construct a dataset, allocate token buffers or change a cursor.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/data/__init__.py">__init__.py · pinned source revision</a></dd>
  </dl>
</aside>
