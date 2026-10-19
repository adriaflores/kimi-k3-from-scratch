---
title: "MixedPrecisionRMSNorm"
description: "Calls PyTorch RMSNorm with the gain cast to the current input dtype."
sourcePath: "src/kimi_k3/utils/rmsn.py"
status: published
---

`MixedPrecisionRMSNorm` subclasses `torch.nn.RMSNorm`. The change is intentionally narrow: preserve the superclass parameter/state layout while choosing the gain's arithmetic dtype from the tensor entering `forward`. This matters when model parameters remain FP32 but some activations are BF16.

## The entire forward path

The only method appears at [source L4–L6](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/rmsn.py#L4-L6):

```python
class MixedPrecisionRMSNorm(nn.RMSNorm):
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype) if self.weight is not None else None, self.eps)
```

The displayed excerpt normalizes tabs to spaces but preserves all operations. For an input `x` with final dimensions matching `normalized_shape`, the functional call computes RMS normalization over those dimensions, using `self.eps`. When the layer has an affine weight, `weight.to(x.dtype)` provides an activation-typed view of the FP32 parameter. When `self.weight` is `None`, no gain is supplied. The code neither casts `x` to FP32 nor defines a custom CUDA kernel. It relies on PyTorch's `F.rms_norm` behavior for the arithmetic and backward implementation.

The cast is differentiable: gradients flow back through it to the stored parameter. This layer is used by the decoder and attention modules, so it is a shared dtype boundary rather than a replacement for pre-norm or post-norm residual logic.

## Verification and limits

[test test_rmsnorm.py L19–L23](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_rmsnorm.py#L19-L23) compares the state dictionary and the parameter dtype against native RMSNorm. [test test_rmsnorm.py L26–L40](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_rmsnorm.py#L26-L40) checks FP32 output and backward equality, while [test test_rmsnorm.py L43–L57](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_rmsnorm.py#L43-L57) checks BF16 output, FP32 weight gradients and BF16 input gradients with numerical tolerance. These are source-inspected assertions, not freshly executed results. This file provides no fused implementation or performance claim.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/utils/rmsn.py">rmsn.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_rmsnorm.py">test_rmsnorm.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
