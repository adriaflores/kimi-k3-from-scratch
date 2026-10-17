# Kimi K3 website — Figures, citations and evidence contract

**Version:** 1.0 · 16 October 2026  
**Applies to:** `content/architecture/*.md`, `content/training/*.md`, and future Results evidence records.  
**Read alongside:** [content contract](CONTENT_CONTRACT.md) and [reading style](READING_STYLE.md).

## 1. Guiding rule

**Every technical claim should be auditable at the level appropriate to that claim.** The prose explains; references make it possible to check. Figures help the reader reason; they do not provide scientific evidence unless they derive from documented measurements. Do not add citation machinery that interrupts every sentence. Prefer contextual inline links and one compact source trail per chapter.

Keep three different voices and evidence sources separate:

1. **The paper says** — a versioned paper citation to the *specific* section, equation or figure (confirmed by a human before publication).
2. **This project implements** — canonical code at a pinned Git commit, and a note about implementation-specific deviations (model scale, tokenizer, optimizations, original vs third-party kernels).
3. **The experiment shows** — a run/manifest/report, conditions, units, outcome and limitation. A test file's name alone does not establish a result.

Do not cite the old Docs pages as proof of code behavior without checking them against the referenced source revision.

## 2. Four reference types

| Kind | Link target | What to include in prose | Publish rule |
|---|---|---|---|
| Paper | Versioned arXiv/PDF or DOI | Verified paper § / equation / figure ID; identify claims from the authors | Preserve version (for example `https://arxiv.org/abs/2607.24653v2`); check the exact location |
| Code | GitHub `blob/<commit>/<path>#L...` | Canonical module/function and the project-specific decision | Pin Git commit; avoid links to mutable `main` for evidentiary claims |
| Test | GitHub `blob/<commit>/tests/...#L...` | Exact test/verification and **what it checks** | Pin Git commit; do not claim an execution unless logs/CI show it |
| Experiment | Stable run directory / artifact / manifest / report | Run ID, model/config, metric with unit, data split, hardware/precision where relevant, protocol and caveat | No number without traceable evidence and a comparable context |

Use normal Markdown links in sentences, for example: `See the [Kimi K3 paper (version 2)](https://arxiv.org/abs/2607.24653v2)`. That link identifies a version, not a verified equation reference. Write the exact §/equation/figure only **after inspecting the paper**. The repository root link is useful for orientation but **not** a substitute for a pinned file or test citation.

Do not re-host paper graphics unless their reuse is permitted; if we redraw a mechanism, name the source and explain what has been simplified or changed.

## 3. One figure contract: meaningful, accessible, attributable

Figures are stored under:

```text
assets/site/<architecture|training>/<chapter-id>/<descriptive-name>.svg
assets/site/<architecture|training>/<chapter-id>/<descriptive-name>.webp
assets/site/<architecture|training>/<chapter-id>/<descriptive-name>.png
```

The site already copies root `assets/` to `website/public/assets/`. Use paths beginning `/assets/site/...` in Markdown. Keep originals/source scripts used to generate plots with their experiment whenever possible; rendered figure assets are **not** a substitute for the source data.

Supported `data-kind` values:

- `conceptual`: an original teaching diagram, model sketch or schematic. Explicitly distinguish simplifications from actual Kimi K3 computation. **Never label it measured evidence.** No external `data-source` required.
- `adapted`: adapted from the paper or another source. Provide a `data-source` URL and matching clickable citation in the caption, crediting the origin and describing the adaptation. Confirm reuse rights where relevant.
- `measurement`: a plot drawn from genuine experiment evidence. Provide a `data-source` URL pointing to the stable report/manifest/artifact **and** matching citation in the caption. Caption/context must give metric and unit, run/config/split, conditions and limitations. If different runs are compared, justify comparability.

### Conceptual diagram example (replace placeholders before shipping)

```html
<figure id="fig-update-intuition" class="reading-figure" data-kind="conceptual">
  <a href="/assets/site/architecture/CHAPTER/diagram.svg"
     target="_blank" rel="noopener noreferrer" aria-label="Open diagram at full size">
    <img src="/assets/site/architecture/CHAPTER/diagram.svg"
         width="900" height="420" loading="lazy" decoding="async"
         alt="Describe the states, arrows, and the central contrast without relying on color." />
  </a>
  <figcaption>
    <strong>Figure 1.</strong> Explain the single idea this diagram makes visible.
    <span class="figure-provenance">Original conceptual illustration; simplified and not experimental evidence.</span>
  </figcaption>
</figure>
```

Refer to the figure in prose with `[Figure 1](#fig-update-intuition)`; keep `id` stable and unique within the chapter. Number figures sequentially within each chapter. If you remove a figure, correct all references. A full-size link is helpful, but not mandatory.

### Adapted or experimental figure

Use the same markup with `data-kind="adapted"` or `data-kind="measurement"` and a **real**, checked `data-source="https://..."`. Include the **same URL** as a visible `<a href="https://...">Source</a>` in `figcaption`. For measurements also include a nearby evidence panel with method, run identity, axes/units, observed result and caveats. Do not add a fake source, placeholder metric, or demo plot just to satisfy the format.

**Figure preparation:** prefer SVG for crisp conceptual figures and WebP/PNG for raster plots; preserve meaningful `viewBox`/width/height, readable labels at phone size, adequate contrast in both themes (or a deliberate neutral canvas), non-color-dependent semantics and a concise alt describing the take-away. Avoid raster screenshots of formulas or code. Name files descriptively; do not use `fig1-final-v7.png`. Embed via local paths only; use the caption for attribution rather than adding tiny watermarks. Interactivity belongs to integration-owned components and must have a noninteractive fallback.

## 4. Chapter source trail

Collect the chapter's primary verification links in one light, unobtrusive block (not a bibliography of everything mentioned). This is especially useful to readers who skim directly to code and experiments:

```html
<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Paper</dt>
    <dd><a href="https://arxiv.org/abs/2607.24653v2">Kimi K3 (v2)</a> — cite the precise, checked section in the final text.</dd>
    <dt>Implementation</dt>
    <dd>Link to a <em>real, pinned</em> code file and describe the specific design choice.</dd>
    <dt>Verification</dt>
    <dd>Link to the applicable test and execution evidence; distinguish inspected tests from recorded passing tests.</dd>
  </dl>
</aside>
```

For Training, replace Paper with `Run/experiment` when the question is chiefly operational. Do not force three entries if only two are meaningful. Inline links are still welcome next to the exact claims they support. Avoid hiding links behind unexplained icons or elaborate citation widgets.

## 5. Experimental evidence — Results policy

**Decision:** use **structured JSON evidence records** as the source of numerical claims, with Markdown for interpretation, links and narrative. This avoids manual duplication of values across Results and Training. Records must be linked to real, reviewed artifacts before their numerical claims are published.

A publishable evidence record must have: stable `id`, phase (`architecture-validation`, `training-reliability`, `optimization`, `scaling`, `qualification`, `pilot`, `final`), metric **and unit**, value(s), model and config revision, dataset/split, run ID/date, method, hardware/precision when material, provenance artifact/manifest URL, limitations and a conservative interpretation. Unknown values must remain unknown, not invented or silently filled. Label `pilot`, `proxy`, `qualification` and `final` explicitly. Charts may be generated from audited JSON records when the Results pipeline exists.

To prevent contradictions, a chapter may summarize a vetted record but must not independently type a conflicting number. Before claiming a final result, check the experimental report and audit receipts from the pilot, full pretraining run and held-out evaluations.

## 6. Static checks and their limits

From `website/`:

```bash
npm run check:content
npm run test:content
npm run build  # prebuild automatically runs the content checker
```

`check:content` checks both Markdown collections for figure IDs, type, caption, alt text, owned asset existence, figure cross-links, and (for published chapters) source trails, versioned Kimi K3 paper links, and pinned links into the canonical repository. It checks **without additional packages or HTTP requests**.

Incomplete `draft` / `review` chapters emit **warnings**, not build failures. Defects in `published` chapters fail the build. The check is deliberately lightweight; it does **not** verify scientific accuracy, run commands, resolve external URLs, validate licenses, prove experiment provenance, or parse all Markdown/HTML cases. A reviewer must still inspect equations, paper IDs, code and tests, factual captions, performance comparisons and sources before approving publication.

**Editorial acceptance:** the figure answers a real question; the reader can use it on a phone and with keyboard/screen reader; its source status is unambiguous; the claim is supported at the right granularity; and documentation depth stays appropriate for the project site rather than duplicating the future articles.
