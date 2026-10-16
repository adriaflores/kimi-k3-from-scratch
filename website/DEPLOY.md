# Build and deploy the Kimi K3 site

The Astro source is in `website/` inside `adriaflores/kimi-k3-from-scratch`.

## Local verification

From a complete clone of the repository:

```bash
cd website
npm ci
npm run build
npm run dev
```

The prebuild hook copies repo-owned figures from `../assets` to `public/assets`, so do not build from an isolated `website/` folder.

Verify dark default and toggle persistence, mobile navigation, and these paths: `/`, `/architecture/`, `/training/`, `/results/`. The old `/paper-to-code/` and `/engineering/` paths should redirect. `/docs/` should show only a deferred notice.

## Cloudflare Pages

1. Connect `adriaflores/kimi-k3-from-scratch` to a Cloudflare Pages project.
2. Project root directory: `website`.
3. Build command: `npm run build`; output directory: `dist` (relative to root).
4. Use Node >=22.12.0 (for example an environment variable pin to a supported 22.x release).
5. Test the Pages preview deployment before assigning the custom domain.
6. In the Pages project's **Custom domains**, add `kimi-k3.adriaflores.com` and follow Cloudflare's DNS instructions.
7. Check HTTPS, routes, redirect behavior and theme after the custom domain is active.

Cloudflare references: https://developers.cloudflare.com/pages/configuration/build-configuration/ and https://developers.cloudflare.com/pages/configuration/custom-domains/.

## Editorial safety

- Do not add or claim final-model scores without a verified checkpoint, protocol and source artifacts.
- The Results page currently contains the validated 80M pilot result as an example of sourced evidence.
- Overview and source-file Docs are deferred until Architecture/Training/Results and code cleanup permit reliable summaries.
- Site deploys must not modify active training code, model checkpoints or recipe files.
