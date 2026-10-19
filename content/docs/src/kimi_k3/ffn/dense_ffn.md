---
title: "DenseFFN and SituGLU"
description: "Implements the FP32-internal SiTU-GLU gate and a two-linear feed-forward path with conditional compilation."
sourcePath: "src/kimi_k3/ffn/dense_ffn.py"
status: review
---

The file defines the elementwise SiTU-GLU activation and the dense/expert feed-forward block. `DenseFFN.forward(x)` maps a tensor with last dimension `hidden_size` back to the same width, preserving all leading dimensions. It is called directly by dense decoder layers and used as the expert network inside Stable LatentMoE.

## SiTU-GLU: one split, two softly bounded branches

The activation core is [source L5–L15](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/ffn/dense_ffn.py#L5-L15). If the incoming last dimension is `2I`, `.chunk(2, dim=-1)` creates `gate` and `values` of width `I`. The implementation explicitly converts this pair to FP32, even when the input was BF16, and casts the product back to the input dtype at the end.

```python
gate, values = x.float().chunk(2, dim=-1)

gate_branch = (beta * torch.tanh(gate / beta) * torch.sigmoid(gate))
value_branch = (linear_beta * torch.tanh(values / linear_beta))

return (gate_branch * value_branch).to(input_dtype)
```

Each `tanh` limits its branch smoothly instead of allowing unbounded linear growth. With $g$ the first half and $u$ the second, the implementation is $[\beta\tanh(g/\beta)\sigma(g)]\,[\beta_{lin}\tanh(u/\beta_{lin})]$. Both `beta` and `linear_beta` are passed from configuration, not trained as module parameters here. `SituGLU.forward` selects the plain core; `SituGLU.compiled` selects the shared `torch.compile` callable [source L18–L34](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/ffn/dense_ffn.py#L18-L34).

## Two linear maps and a guarded compiled branch

`DenseFFN.__init__` constructs `gate_up_proj: M → 2I` and `down_proj: I → M`, both without bias [source L36–L48](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/ffn/dense_ffn.py#L36-L48). This is a **packed storage layout** for the first two projections, not a mathematical coupling of their weights. The forward begins with the packed output and finishes with the down projection [source L50–L64](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/ffn/dense_ffn.py#L50-L64).

```python
gate_up = self.gate_up_proj(x)

if (
    self.compile_situ
    and gate_up.is_cuda
    and self.training
    and torch.is_grad_enabled()
):
```

Compilation is only selected for a CUDA training forward with autograd enabled and the module flag set. The compiled path calls `torch._dynamo.maybe_mark_dynamic(gate_up, 0)` before evaluating SiTU, for a varying number of expert-token rows. Evaluation, CPU, no-grad and ordinary dense decoder paths use eager `SituGLU`. The compilation promise concerns the activation, not all of `DenseFFN`, the expert dispatch or the optimizer.

## Verification

The FFN unit tests check output shapes, finite forward/backward values, BF16 round-tripping, the activation against a separate reference formula and the packed projection against a separate two-branch reference [test test_dense_ffn.py L23–L109](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/ffn/test_dense_ffn.py#L23-L109). Sparse expert gradient behavior is checked separately in [test test_moe_grouped_vs_naive.py L69–L119](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py#L69-L119). No kernel speed is asserted here.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/ffn/dense_ffn.py">dense_ffn.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/ffn/test_dense_ffn.py">test_dense_ffn.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/equivalence/test_moe_grouped_vs_naive.py">test_moe_grouped_vs_naive.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
