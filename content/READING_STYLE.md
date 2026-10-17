# Kimi K3 website — Reading and visual style

**Version:** 1.2 · 16 October 2026  
**Status:** Integration-owned patterns, prepared for review  
**Applies to:** Markdown chapters under `content/architecture/` and `content/training/`

This document complements [CONTENT_CONTRACT.md](CONTENT_CONTRACT.md). The site shell and chapter renderer remain shared infrastructure, not chapter-specific code.

## 1. Reader experience

The narrative should feel like a thoughtful technical explanation, not a lecture: one meaningful question, one idea at a time, clear transitions, and enough mathematical precision to verify the reasoning. Use visuals only when they resolve a real cognitive obstacle. Resist long callout chains, motivational filler, gimmicky animations, and oversized decorative headlines.

Chapter hierarchy:

- Frontmatter provides the **only H1**, the central question, and the concise description. Don't add an H1 in the Markdown body.
- `##` separates major turns in reasoning. `###` is for a genuine subproblem. The sidebar is automatically generated from H2 and H3.
- Prefer 2–5-sentence paragraphs and roughly 65–72 characters per text line on desktop (dependent on font metrics).
- Center the **reading column itself** in the page, not just the combined sidebar-and-article grid. Prose remains left-aligned for comfortable reading; the table of contents occupies the balanced left rail on wide screens and becomes a horizontal/on-page index at narrow widths.
- The chapter H1 is an editorial display element: it may extend beyond the reading-column width while remaining visually centered. The lead, summary, body, and footer navigation follow the same centered reading axis.
- Main explanations use sans-serif prose (about 17–18px desktop, 16.75px narrow mobile), editorial serif titles, and monospace for code, measurements and labels. Contrast must work in the default `#090b10` dark theme and optional light theme.
- A short question or an interactive check may enrich a chapter. Neither is mandatory or a substitute for explanation.

## 2. Basic Markdown patterns

Use these as the default: paragraphs, lists, tables, links, KaTeX-compatible `$inline$` or `$$display$$` math, and fenced code with a language. Source snippets must be short and linked to their canonical implementation when editorially verified. The renderer provides editor-like Shiki syntax colors (One Dark Pro / One Light) that follow the site's manual theme toggle. Token highlights use foreground colors only, with no per-token backgrounds, glows, shadows, or letter effects. Fenced snippets should always declare their language (for example, `python`).

A normal blockquote is treated as an understated reading highlight or reader-facing thought experiment:

```md
> **Think about the next step.** Would the result still be equivalent if the data position changed?
```

Avoid using blockquotes to suggest an actual quote from another author without attribution.

## 3. Figure with a meaningful caption

Plain Markdown does not need a custom Astro component. Use the semantic HTML pattern below inside the Markdown chapter. HTML inside the `figure` must be written as HTML (not Markdown syntax), and image assets live under the chapter's own directory.

```html
<figure id="fig-example-contrast" class="reading-figure" data-kind="conceptual">
  <a href="/assets/site/architecture/hybrid-attention/figure.svg"
     target="_blank" rel="noopener noreferrer"
     aria-label="Open the diagram at full size">
    <img src="/assets/site/architecture/hybrid-attention/figure.svg"
         width="900" height="380" loading="lazy"
         alt="Describe the actual information flow or contrast shown in this figure." />
  </a>
  <figcaption><strong>Figure 1.</strong> What the diagram shows and why it matters. Original conceptual illustration; not experimental evidence.</figcaption>
</figure>
```

Refer to it with `[Figure 1](#fig-example-contrast)`. Figures are **not required** to be links, but opening complex diagrams full size is helpful on phones. Do not rely on color to encode meaning. An experimental plot requires units, provenance and data context, described in the shared figure and evidence guide; the demo figure is explicitly conceptual, not experimental. Do not copy figures from the paper without attribution.

## 4. Three optional reading interventions

Use only when a paragraph or figure alone would be weaker. **These are native HTML patterns, not Astro components**, so they work directly inside `.md`. Use ordinary HTML tags inside these blocks rather than nested Markdown.

### Guiding question

```html
<aside class="reading-question" aria-label="Question to consider">
  <span class="reading-note-label">A question worth asking</span>
  <p>What changes if new information contradicts the state we have already stored?</p>
</aside>
```

This is the most teacherly pattern. It should prompt the next step of reasoning, not ask an obvious rhetorical question.

### Implementation note

```html
<aside class="reading-note" aria-label="Implementation note">
  <span class="reading-note-label">Implementation note</span>
  <p>Explain one project-specific implementation decision and link to the actual source.</p>
</aside>
```

### Evidence or methodological limitation

```html
<aside class="reading-evidence" aria-label="Evidence and limitations">
  <span class="reading-note-label">Evidence &amp; limitations</span>
  <p><strong>Measured:</strong> State the metric and conditions.</p>
  <p><strong>Limit:</strong> Say what the evidence does not establish.</p>
</aside>
```

Every factual measurement requires genuine records and provenance. An evidence panel is not a license to make up a plausible number.

## 5. Optional interactive self-check

A native `<details>`/`<summary>` disclosure provides a small prediction-and-reveal without a new JS framework. The explanation must remain complete without opening it.

```html
<details class="reading-reveal">
  <summary>Check your intuition: what should happen in this case?</summary>
  <p>Here is a concise answer that connects back to the preceding mechanism.</p>
</details>
```

Do not overuse disclosures, hide crucial derivations inside them, or introduce interactions just for novelty. Keyboard, touch and reduced-motion users should receive equivalent information.

## 6. Figure, code, math and layout rules

- Prose stays within a restrained reading measure. Code and display math scroll horizontally when intrinsically wider; never force the page itself wider than the viewport.
- Tables scroll horizontally on narrow screens, with enough heading context to interpret them.
- Equations should define the symbols near first use and explain how they change the reader's mental model. Avoid a wall of unlabeled formulas.
- For diagrams, include `alt`, `figcaption`, provenance (conceptual / adapted / measured) and convenient larger viewing for intricate drawings.
- Keep the index navigable without JavaScript; active-section highlighting is optional progressive enhancement.
- No per-chapter CSS, palette overrides, or one-off layouts. A new pattern goes through the integration owner.

## 7. Reference fixtures and publication gate

The two **draft** chapters are deliberately not finished content:

- `content/architecture/hybrid-attention.md`: figure, a guided question, a math expression, and a native self-check.
- `content/training/training-loop.md`: code, data table, note, blockquote, and an illustrative evidence block.

They demonstrate the design system only. No paper-accurate mechanism exposition or quantitative experiment is claimed; They must be revised and reviewed before publication. **Never change `status: draft` to `published` merely to make them visible in production.**

In `website/`, run `npm ci`, `npm run build`, then `npm run dev`; verify both previews in development at `/architecture/hybrid-attention/` and `/training/training-loop/`, dark/light modes, narrow/mobile screens, anchor navigation, scroll behavior, native self-check, code readability, figure zoom and keyboard focus.

## Verified source and figure patterns

Figure provenance, stable figure anchors, evidence citations and compact chapter source trails are specified in [FIGURES_AND_SOURCES.md](FIGURES_AND_SOURCES.md). Use that guide instead of inventing per-chapter citation widgets or figure styles. The checker runs before the production Astro build.
