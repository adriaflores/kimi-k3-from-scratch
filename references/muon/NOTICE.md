# Muon source attribution

The Newton-Schulz polynomial, numerical kernel structure and momentum convention
in `src/kimi_k3/training/optimizers.py` are adapted from the following MIT-licensed
sources. The surrounding optimizer integration is specific to this repository.

- Keller Jordan, [Muon](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py).
  Pinned commit: `f98f1cacc0263b04290753e32be8d498c1efc806`.
- Moonshot AI, [Moonlight training example](https://github.com/MoonshotAI/Moonlight/blob/c2ad5b20c605086526a179d36901bfc41b52b44b/examples/toy_train.py).
  Pinned commit: `c2ad5b20c605086526a179d36901bfc41b52b44b`.
- Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982).
  Equation (4) specifies the update scale `0.2 * sqrt(max(rows, columns))`.
  This scale differs from the original Muon learning-rate convention.

The implementation retains the source kernel's quintic coefficients and five-step
BF16 default. An eager numerical reference remains available. The optional CUDA
route compiles the same Newton–Schulz core with TorchInductor, caching static
specializations by layout and arithmetic settings.
AdamW is provided by the installed PyTorch package, rather than copied from either
source.

## KellerJordan/Muon license

```text
MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MoonshotAI/Moonlight license

```text
The MIT License (MIT)
Copyright © 2025 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
