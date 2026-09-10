# Kimi K3 from Scratch

A from-scratch PyTorch implementation of the Kimi K3 language backbone,
derived from the papers and verified with reference tests and experiments.

## Philosophy

Every important mechanism should be traceable through:

**paper → derivation → implementation → test → experiment**

## Scope

This repository focuses on the Kimi K3 language backbone.

Planned components:

- Kimi Delta Attention
- Gated MLA
- Attention Residuals
- Stable LatentMoE
- SiTU-GLU
- Quantile Balancing
- Per-Head Muon
- Full language backbone
- KV/recurrent-state caching
- Tiny training and generation

The vision stack and production-optimized kernels are outside the initial scope.

## Status

Work in progress.