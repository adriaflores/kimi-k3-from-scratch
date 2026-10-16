# Kimi K3 website — M1.1 Content and editorial contract

**Version:** 1.2 — 16 October 2026  
**Status:** Design contract; not integrated into GitHub.  
**Basis:** Kimi K3 site v3 overlay and its roadmap.  
**Scope:** Architecture and Training chapters. Results use the same evidence principles, but their data manifest is intentionally deferred to M1.4/R1.

## 1. Purpose and boundaries

The independent site at `kimi-k3.adriaflores.com` presents the *language backbone* and the engineering path to pretraining. A chapter is a Markdown file, not a custom Astro page. The site renderer owns page layout, navigation, typography and reusable visual components. Chapter authors own content and uniquely named figures.

- **Architecture:** research question → intuition/diagram → mechanism → paper-to-code choices → validation → relation to full model.
- **Training:** engineering obstacle → considered approach → implementation → evidence → decision/trade-off → what this enabled next.
- **Results:** aggregate independently verifiable evidence across stages. Claims must identify experiment type, method and limitations; this contract does not decide Results storage yet.
- **Not in scope:** final Overview; file-by-file Docs; personal-blog tutorials; code refactoring; new experiments; redesign of the global dark theme.

## 2. Canonical paths and URLs

```text
content/
  CONTENT_CONTRACT.md
  architecture/
    hybrid-attention.md
    attention-residuals.md
    feedforward-and-moe.md
    backbone-assembly.md
  training/
    training-loop.md
    reliability.md
    optimization.md
    scaling-experiments.md
    gpu-qualification.md
    pilot-to-pretraining.md
assets/
  site/
    architecture/<chapter-id>/...
    training/<chapter-id>/...
website/
  src/content.config.ts        # integration-owned Astro collection definitions
  src/pages/architecture/      # integration-owned generic chapter route
  src/pages/training/          # integration-owned generic chapter route
  src/layouts/                # integration-owned shared layout
  src/styles/                 # integration-owned shared styles
```

The filenames above are *initial chapter groupings*, not a permanent limit on chapter count. Names use descriptive kebab-case IDs **without numeric prefixes**; `order` determines display sequence. IDs and URLs should remain stable once public. Root `docs/` continues to serve repository-oriented documentation independently. `website/scripts/sync-assets.mjs` already copies root `assets/` to `website/public/assets/`; M1 will reuse it.

Routes are `/architecture/<chapter-id>/` and `/training/<chapter-id>/`. The top-level `/architecture/`, `/training/` and `/results/` pages remain the section gateways established by the v3 overlay. The site must not duplicate the chapter inventory in hardcoded arrays once collections are connected.

## 3. Minimal frontmatter schema (both collections)

| Field | Type | Required | Purpose |
|---|---|---|---|
| `title` | nonempty string | yes | Chapter title and SEO heading (renderer owns H1) |
| `description` | nonempty string | yes | Concise summary for cards, metadata and social previews |
| `question` | nonempty string | yes | Central motivating question at the beginning of the article |
| `order` | positive integer | yes | Position within its section; no duplicated position in a section |
| `status` | `draft`, `review`, `published` | yes | Editorial gate; only the integrator may set `published` after verification |

No manual `slug` or redundant `section` field. The collection is determined by folder; the route ID is the Markdown filename. Do not add new frontmatter fields ad hoc: request a shared schema change. Later fields, if useful (e.g. a per-chapter cover image), require one integration-owned schema revision.

### Example — Architecture

File: `content/architecture/hybrid-attention.md`

```md
---
title: "Hybrid Attention"
description: "How KDA and gated MLA provide complementary token-mixing mechanisms in the scaled language backbone."
question: "Why combine recurrent memory with content-based attention?"
order: 1
status: draft
---

Short opening paragraph that answers the question at a high level.

## The problem
...

## How the mechanism works
...

## From paper to code
...

## How we validated it
...

## Role in the complete backbone
...
```

### Example — Training

File: `content/training/training-loop.md`

```md
---
title: "The Training Loop"
description: "From tokenized data to a verified forward, backward and optimizer-update circuit."
question: "What must work before scaling a training run?"
order: 1
status: draft
---

Explain what we built and why it matters in a short opening paragraph.

## The engineering problem
...

## The approach
...

## What we implemented
...

## Evidence that it works
...

## Decisions and trade-offs
...

## What this enabled next
...
```

These headings are a **narrative guide**, not an exact heading-name API. Clear equivalent headings are allowed. Never include a duplicate `# H1` inside Markdown: `title` supplies it. Use `##` for main sections, and `###` for subsections. Contents indexes should be generated from these headings in M1.2/1.3.

## 4. Editorial voice and pedagogy

**Editorial identity: _Distill clarity, with a Chris Piech touch_.** This names an aspiration, not an instruction to imitate a particular person's wording. Write rigorous, highly connected technical explanations with a natural, warm, curious teaching voice. The reader should feel invited to work out the idea, not lectured at. Use small moments of genuine delight when an insight warrants them; never manufacture excitement, jokes, exclamation marks or empty encouragement.

### 4.1 Narrative voice

- **Theory: clear, mostly impersonal prose.** Explain a mechanism as precisely as a thoughtful technical textbook would. We can use direct address ("Consider...", "What changes if...?") sparingly when it actively improves intuition.
- **Engineering: first-person singular for real author decisions.** "I chose...", "I tested...", or "I found..." only when supported by the project's actual history and evidence. Do not use "we" to imply a team built the system. An occasional inclusive teaching "we can see" is acceptable if it plainly means author and reader reasoning together.
- **Results: measured, exact and restrained.** Separate observations, interpretations and hypotheses; report methods and limitations without hype.
- **English: fluent, conversational but technically exact.** Prefer active verbs, short or medium sentences, concise paragraphs and consistent terms. Avoid marketing language, inflated claims, bureaucratic prose and a contrived "AI assistant" tone.

### 4.2 How to teach a concept

A typical conceptual passage should create this progression **when it helps the idea**; it is *not* a compulsory checklist of headings:

1. **Why care?** Establish the real problem or limitation in a few sentences.
2. **Give the reader a handle.** Introduce a mental model, tiny example or visual before notation if it reduces cognitive load.
3. **Invite a prediction.** Pose a genuine question at a turning point (e.g. "What happens when new information contradicts the state?"). Let the question guide the next paragraph and answer it.
4. **Make the mechanism precise.** Introduce the smallest equation, algorithmic step or code excerpt needed; define symbols near first use and connect each term to intuition.
5. **Close the loop.** State what the mechanism buys us, what it does not solve, and how it connects to the next section or to our implementation.

**Questions are a teaching tool, not decoration.** One or two well-placed questions may be more effective than a string of rhetorical questions. Do not ask something obvious, then spend a paragraph congratulating the reader for knowing it. Likewise, use re-explanation only to address a likely misconception or reveal a different perspective; avoid repetitive recaps.

Respect what the reader already knows, but never introduce an unexplained leap that blocks understanding of the chapter. Define unusual notation and distinguish closely related terms. When an elaborate foundational derivation is valuable but not necessary for the project chapter, give the essential intuition and link to a future deep-dive article instead.

### 4.3 Depth and length budget (editorial targets, not hard limits)

The website is a **technical map of what was built and learned**, not the place for every derivation. Sections should feel substantial but easy to scan. The primary measure is whether a paragraph advances understanding, not whether a word count has been met.

| Content unit | Suggested prose budget | What it should accomplish |
|---|---:|---|
| Chapter opening | 80–140 words | Motivate one compelling question and give the main intuition; avoid repeating the index |
| Conceptual/mechanism subsection | 150–300 words | Make one important idea intelligible, normally helped by a figure, equation or worked example |
| Implementation/decision subsection | 120–250 words | Explain the meaningful design choice and show where it lives in the code |
| Validation/evidence subsection | 100–200 words | Say what was checked, how, what the evidence supports and its limits |
| Closing transition | 50–100 words | Synthesize the contribution and connect to the next chapter; no generic summary |
| **Architecture chapter** | **900–1,600 words typical** | A connected explanation of a component or tightly related group of mechanisms |
| **Training chapter** | **700–1,300 words typical** | A causal engineering story with enough evidence to justify decisions |
| **Results evidence page** | **~400–900 words typical, excluding charts/tables** | Let plots and methodological context do most of the work |

Complex chapters may go beyond these ranges, especially when merging KDA/MLA or several design decisions. Above roughly 1,800–2,000 words, review whether a chapter should be split, a derivation linked out to an article, or a lengthy section replaced with a better visual. **Do not compress away a key inferential step merely to meet the budget.** Reading time is inherently longer for equations and diagrams than word count suggests.

Keep paragraphs reasonably short (usually 2–5 sentences) and avoid unbroken technical walls. Subsections should generally serve **one core idea**; prefer a clear transition or a figure over adding more headings or bulleted lists. The chapter should read as a continuous explanation, not a checklist.

### 4.4 Visual-first, but not visual-for-visual's-sake

Text is only one teaching medium. When a picture can turn an abstract statement into an intelligible mechanism, the picture should do that work.

- **Diagrams:** illustrate states, information flows, tensor shapes, attention paths or before/after contrasts. Keep labels readable and connect the explanation to visible elements.
- **Equations:** reveal precisely *what changes* in a mechanism. Prefer a small number of carefully interpreted equations over dumping all the paper's formulas. Annotate terms when useful.
- **Code excerpts:** show one decision or meaningful operation, never a whole source file. Point back to the canonical implementation and the relevant test.
- **Mini-examples:** use tiny numbers/shapes or one short thought experiment if they prevent conceptual ambiguity.
- **Interactive elements:** consider a slider, toggle, step-through, or hover explanation only where user control makes the concept noticeably easier to understand. They are optional; their core explanation must remain accessible without interaction and they must not distract from reading.
- **Experimental charts:** visualize measured evidence and make axes, units, provenance and context clear. Do not disguise conceptual illustrations as measured results.

A strong chapter often has **one signature explanatory figure** and a few supporting mathematical/code/evidence elements; there is no quota. Every visual must answer a real reader question, be accompanied by explanatory prose and alt text/caption, and work on mobile. The visual system should invite exploration without demanding it.

### 4.5 Distinct narrative arcs by section

- **Architecture:** question → intuition/visual → precise mechanism → implementation-specific decision → test → relation to assembled model. Aim for the natural progression of a technical chapter, not a compressed retelling of the paper.
- **Training:** obstacle → meaningful alternatives → what I implemented and why → measured outcome → trade-off → next capability unlocked. The author's voice belongs here when recounting genuine work.
- **Results:** question/metric → method and conditions → figure/table → interpretation → uncertainty and limits. Keep the reader able to audit the claim.

Throughout the site, distinguish Kimi K3's published methods from this reduced implementation, its tokenizer/scale choices and optimizations. Attribution matters: FLA's fused KDA must not be presented as independently implemented from scratch. Avoid generic textbook detours, filler, repeated content across chapters and unsupported claims. Deep conceptual articles live on `adriaflores.com/blog`, with optional contextual links from the site.

### 4.6 Two short calibration examples

**Architecture (mostly impersonal):**

> A recurrent model keeps the past in a fixed-size state. That is attractive for long sequences, but it raises a question: what happens when the model needs to correct information already stored? A delta update gives us a way to describe that correction. The equation below makes the idea precise.

**Training (author's own voice, subject to evidence):**

> Before scaling the run, I needed to know whether checkpoint recovery actually resumed the same training state. Saving model weights alone would not answer that question: the optimizer, data cursor and random-number generators matter too. So I tested the continuation explicitly and compared the resulting state.

These examples calibrate **voice and structure**, not evidence for any specific historical experiment. Replace their claims with verified chapter-specific facts before publication.

## 5. References, equations, code, and figures

- **Paper:** reference the specific Kimi K3 paper version (the v3 overlay points to `https://arxiv.org/abs/2607.24653v2`) and label the relevant §/equation/figure in the sentence. The author must check the precise numbering when writing the chapter.
- **Code/tests:** link to repo-relative canonical paths, ideally with a pinned Git commit once content is reviewed. For tests cite a specific assertion/test name and explain what it establishes; avoid implying tests were executed if only inspected.
- **Experiments:** link to stable run ID, manifest, report, measured artifacts or plotted source; name the context and methodology. Keep original measurements separate from interpretations.
- **Equations:** use Markdown with KaTeX-compatible math (`$...$` and `$$...$$`); define symbols near their first use. Do not handcode custom equation CSS within a chapter.
- **Code samples:** use fenced blocks with a language, short excerpts, clear explanations and a link to the canonical implementation. Do not keep a second full implementation in a doc.
- **Figures:** store under `assets/site/<collection>/<chapter-id>/`; link via `/assets/site/<collection>/<chapter-id>/<filename>`. Add descriptive alt text and a caption that identifies whether the figure is conceptual, adapted from the paper or experimental. Graphs need axes/units and evidence provenance. The root-assets copying mechanism is already in place.
- **Interaction and accessibility:** an interactive illustration is an optional enhancement, not a precondition for understanding. Its baseline meaning must survive on touchscreens, without hover, with reduced motion, and without client-side interaction where feasible. Any reusable interactive component belongs to the integration chat, not an individual chapter author.

Use ordinary Markdown links first. Specialized paper/code/evidence citation components should only be added in M1.4 if they materially improve readers' experience; the content must not depend on chapter-specific Astro code.

## 6. Parallel ownership contract

- A chapter chat modifies only its assigned `content/<collection>/<chapter-id>.md` and assets under `assets/site/<collection>/<chapter-id>/`.
- The integration chat alone modifies shared Astro collection schemas, route components, layout, styles, figure components and navigation.
- No documentation chat may modify model `src/`, experiment scripts, tests, checkpoints or training configs simply to support a page.
- Each chat delivers a patch/ZIP limited to owned files and a short list of evidence sources, unchecked claims and known limitations. No independent commits/deployments without approval.

## 7. Editorial states and acceptance

- `draft`: work-in-progress; not exposed publicly.
- `review`: ready for source/technical/editorial validation; not yet public.
- `published`: integrator accepted the sources, links, maths/figures and copy. Publication is an **editorial decision**, not a claim of formal scientific certification.

M1.2 must specify and implement the dev/production rendering rules for these states. A chapter is publishable when it (a) has a coherent, progressive and inviting story, with length proportional to its ideas, (b) meets metadata validation, (c) clearly distinguishes paper and our implementation, (d) links to appropriate source/tests/evidence, (e) has no unverified quantitative claims, (f) uses figures/equations/examples intentionally and accessibly, and (g) renders and navigates correctly on desktop and mobile.

## 8. M1.1 exit gate and what is deferred

**M1.1 delivers**: one schema shared by two collections, stable filenames/routes, a unified editorial voice, depth/length/visual guidelines, distinct Architecture/Training narrative standards, reference/evidence principles, and ownership boundaries. This contract is ready for review but is not a tested implementation.

**M1.2:** implement Astro collections, filtering, chapter routes and navigation from metadata.  
**M1.3:** reusable visual reading components.  
**M1.4:** richer references/figure strategy and Results storage choice.  
**M1.5:** two representative Markdown chapters and rendered checks.  
**M1.6:** build, links, responsive QA and reusable handoff.

**Integration prerequisite:** apply/reconcile the v3 overlay before treating the `main` branch as the v3 site. As inspected on 16 October 2026, `main` still has `paper-to-code.astro` and `engineering.astro` and no Architecture/Training content collections. Nothing in this contract was committed or deployed.

## M1.3 addition — shared reading patterns

The **reading/visual vocabulary** is defined in [`READING_STYLE.md`](READING_STYLE.md). It is mandatory for chapter authors and shared across Architecture and Training: semantic figure/caption markup, Math/KaTeX, code blocks with dual syntax themes, restrained callouts, an optional native self-check, and accessible behavior on desktop and mobile. These are generic Markdown/HTML conventions, not custom Astro pages. M1.4 still owns final paper/code/test citation patterns and experimental figure provenance.

## M1.4 addition — citation and figure provenance system

See [FIGURES_AND_SOURCES.md](FIGURES_AND_SOURCES.md) for the final M1.4 authoring patterns, evidence requirements, three figure kinds, and the editorial/build checks. The site uses ordinary Markdown links plus semantic figures/source trails rather than adding a new citation syntax or dependencies. Results will use structured JSON evidence records with Markdown narrative **when R1 is implemented**; this milestone does not create unaudited measurements.
