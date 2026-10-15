# Kimi K3 — Project site

This directory contains the Astro site intended for `kimi-k3.adriaflores.com`.

Its purpose is to present the research engineering project and provide a polished rendering of the repository's **technical reference documentation**. It is intentionally separate from Adrià Flores' personal blog (`adriaflores.com/blog`), where independent educational deep dives will be published.

- `src/pages/index.astro`: project landing page.
- `src/pages/engineering.astro`: implementation and verification summary.
- `src/pages/results.astro`: evaluation status and, later, final results.
- `src/pages/docs/`: curated documentation, generated from `../docs/*.md`.
- `src/layouts/SiteLayout.astro` and `src/styles/site.css`: visual system.
- `DEPLOY.md`: local preview and Cloudflare Pages deployment instructions.

From this directory, run `npm ci` followed by `npm run dev` or `npm run build`.

The existing `docs/01_kimi_delta_attention.md` is an early, partially outdated long-form draft and is **not** included in the new technical-reference navigation until its editorial destination has been decided. The original Markdown is preserved.

## v2 site information architecture

- `/`: the independent-project overview.
- `/paper-to-code/`: architecture mechanisms, implementation map and validation (not a second mathematical blog).
- `/training/`: data, recipe, GPU qualification and recovery work.
- `/results/`: measured pilot evidence and final evaluation when available.
- `/docs/`: concise source-linked Markdown technical reference.

A persistent Light / Dark control follows the visitor's system preference until overridden. The selection is saved locally in the browser. `/engineering/` is kept as a backwards-compatible redirect to `/training/`.
