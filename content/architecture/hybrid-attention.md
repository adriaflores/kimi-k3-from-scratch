---
title: "Hybrid Attention"
description: "How recurrent memory and content-based attention complement one another in the scaled language backbone."
question: "Why combine two ways of mixing information across tokens?"
order: 1
status: draft
---

This is a **content-system smoke test**, not the finished KDA/MLA explanation. M1.5 will replace it with an evidence-reviewed chapter.

## A useful question

A recurrent state compresses previous tokens. Attention can instead address token representations directly. The architectural question is how these two approaches can complement each other within one decoder.

## From intuition to notation

For example, a single query-key score can be written as $q^\top k$. Here $q$ is a query vector and $k$ is a key vector. This small expression tests KaTeX rendering; it is not intended as a derivation of KDA.

## From the paper to code

The final chapter will identify the paper section, the canonical implementation and the corresponding numerical tests. For now, return to the [Architecture index](/architecture/).
