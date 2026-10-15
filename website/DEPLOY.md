# Kimi K3 project site — build and deploy

This Astro project lives in `website/` within `adriaflores/kimi-k3-from-scratch`. The repository Markdown in `docs/` is the canonical source of the site's technical reference, loaded by `website/src/content.config.ts`.

## Preview locally

At the repository root:

```bash
cd website
npm ci
npm run dev
```

Open the local URL printed by Astro. Production build:

```bash
npm run build
```

The existing `predev` / `prebuild` hook copies `../assets` into `public/assets`, so work from the complete repository clone rather than an isolated `website` folder.

## Cloudflare Pages via GitHub

1. In **Cloudflare → Workers & Pages → Create → Pages**, connect GitHub and choose `adriaflores/kimi-k3-from-scratch`.
2. Set project **Root directory** to `website`.
3. Set **Build command** to `npm run build` and **Build output directory** to `dist` (relative to the configured root).
4. Set environment variable `NODE_VERSION=22` (the Astro package requires Node >=22.12).
5. Confirm the Pages preview deployment is working before configuring the custom domain.
6. Under the Pages project → **Custom domains**, add `kimi-k3.adriaflores.com`. Because `adriaflores.com` is already on Cloudflare, the CNAME can normally be created automatically after you confirm the association. Do not create the DNS CNAME by itself before registering the domain on the Pages project.
7. Check `https://kimi-k3.adriaflores.com/`, `/engineering/`, `/results/`, `/docs/`, and `/docs/architecture/`.

Cloudflare documentation: https://developers.cloudflare.com/pages/configuration/build-configuration/ and https://developers.cloudflare.com/pages/configuration/custom-domains/.

## Editorial / technical boundaries

- Do not publish final evaluation scores before final checkpoint identity and protocol are verified.
- The results page is intentionally a placeholder; update it after M8/M9 evidence is available.
- `docs/01_kimi_delta_attention.md` is an *early tutorial draft* and is not currently published as technical reference because it contains outdated implementation placeholders. Decide whether to migrate the revised deep dive to `adriaflores.com/blog` separately.
- Technical reference: `docs/{architecture,data,training,reproducibility}.md`.
- Project website UI: `website/src/`.
- The repository retains the previous `website/src/layouts/ChapterLayout.astro` and `website/src/styles/article.css` for historical/reference use. Nothing is deleted by this patch.
- Changes in `docs/`, `website/` or build tooling can be deployed independently of model training; **never change the active training checkpoint or recipe for a documentation deployment**.
