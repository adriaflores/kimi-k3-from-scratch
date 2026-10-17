# Kimi K3 — Project website

Astro site for the independent project at `kimi-k3.adriaflores.com`.

## Information architecture (v3)

- `/` — **gateway**, not the final Overview; introduces the project and links to its core sections.
- `/architecture/` — paper-grounded mechanisms, implementation map and eventual `model.py` assembly.
- `/training/` — six-phase engineering and experimentation journey, from first training loop to full pretraining.
- `/results/` — grouped evidence for the architecture and **every training phase** (not just final evaluation).
- `/docs/` — deliberately **deferred** until the source code cleanup. No file-level reference pages are published yet.

`/paper-to-code/` and `/engineering/` redirect to the new routes for compatibility with the previous prototype.

The KDA article draft in `docs/01_kimi_delta_attention.md` is preserved in the repository but is not represented as finished reference documentation. Deep technical tutorials will live in the author's personal blog at `adriaflores.com/blog`.

## Visual system

Near-black default theme, cool-blue accent, neutral card surfaces, optional persistent light mode. Shared styles are in `src/styles/site.css`, with the v3 token contract recorded in the root-level `ROADMAP_AND_CONTRACT.md` included with the patch.

## Local commands

```bash
cd website
npm ci
npm run dev
npm run build
```

The repo's existing `predev` and `prebuild` hooks sync images from `../assets/`.

See `DEPLOY.md` for Cloudflare Pages steps, and `ROADMAP_AND_CONTRACT.md` for parallel-content workstreams.

## Content development and production QA (M1.6)

The approved editorial references are `../content/architecture/hybrid-attention.md` and `../content/training/training-loop.md`. Their Markdown content is previewable in `npm run dev` and remains **draft-only** in production.

For the chapter ownership contract and publish checklist, read [`../content/AUTHORING_HANDOFF.md`](../content/AUTHORING_HANDOFF.md).

```bash
cd website
npm run check:content    # figures, provenance and answered question callouts
npm run test:content     # Node tests for editorial and production checks
npm run qa               # tests + asset sync + content check + Astro build + publication audit
npm run dev              # inspect draft chapters and mobile/dark/light rendering
```

`npm run verify:build` runs only **after** an Astro build and ensures every published chapter appears in its index and production output while draft/review chapters do not. It also catches missing emitted site figures and broken local anchors in published chapters. Link accuracy, source quality, claims and accessibility still require human review.
