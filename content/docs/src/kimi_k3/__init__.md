---
title: "kimi_k3 package"
description: "The root Python package marker, without imports, exports, or initialization logic."
sourcePath: "src/kimi_k3/__init__.py"
status: published
---

`src/kimi_k3/__init__.py` is deliberately empty at [source L1–L1](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/__init__.py). Importing the `kimi_k3` package executes no package-specific statements and does not automatically re-export `KimiK3`, configuration classes, trainers, or attention layers.

## Import boundary

Callers import functionality from the defining modules, such as `kimi_k3.model` or `kimi_k3.training.trainer`, or from subpackages that explicitly re-export classes. This file has no `__all__`, module-level configuration, side effects, runtime validation, or tensor contract. Python's ordinary package-import behavior still applies, but there is no custom initializer to document.

## Verification and scope

The identity of this module is established by inspection of the empty file at the pinned revision. Tests of the model and its subpackages concern their own implementations; they should not be interpreted as tests of logic in this file.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/__init__.py">__init__.py · pinned source revision</a></dd>
  </dl>
</aside>
