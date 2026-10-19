---
title: "convert_assets_to_webp.py — Lossless asset conversion"
description: "Converts eligible assets to verified lossless WebP and updates matching Markdown asset paths."
sourcePath: "scripts/convert_assets_to_webp.py"
status: published
---

[`scripts/convert_assets_to_webp.py`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/convert_assets_to_webp.py) is a repository-maintenance command, not a model or training utility. It searches the repository's `assets/` subtree for raster images, converts them to lossless WebP, verifies that the new file can be decoded, **deletes the original**, then replaces matching asset-path strings inside Markdown under `content/`. The order matters: a failed conversion must not remove an original source image, while a completed conversion can require subsequent content-link rewriting.

*Code excerpts below are dedented from their enclosing functions for readability; the displayed statements and ordering are preserved.*

## Scope and safe inspection

[`ROOT`, `ASSETS` and `CONTENT`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/convert_assets_to_webp.py#L13-L16) are derived from the script's own path, not from the invoking working directory. This lets the command run from elsewhere and still operate on **this** repository. Eligible file extensions, case-insensitively, are `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tif`, `.tiff`; existing `.webp` and `.svg` files are untouched.

The only [CLI flag](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/convert_assets_to_webp.py#L51-L59) is `--dry-run`. From the repository root:

```bash
python scripts/convert_assets_to_webp.py --dry-run
```

This is the recommended first invocation. It sorts matching paths, prints `would convert ...` for each one, and does **not** write, delete, re-encode or rewrite links. If nothing matches, the script reports `No raster images to convert.`. Running **without** `--dry-run` is destructive with respect to the source files.

## Convert one image, then remove its source

[`_convert()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/convert_assets_to_webp.py#L19-L29) selects a sibling destination with the same basename and `.webp` extension. It refuses to overwrite an existing destination before opening the source:

```python
target = source.with_suffix(".webp")
if target.exists():
    raise FileExistsError(f"Refusing to overwrite {target.relative_to(ROOT)}")
with Image.open(source) as image:
    image.save(target, "WEBP", lossless=True, method=6)
with Image.open(target) as check:
    check.verify()
source.unlink()
```

Pillow's lossless WebP encoder is explicitly requested (`lossless=True`, `method=6`); no target quality or resize flag exists. `check.verify()` checks that the produced file is structurally readable, **not** pixel-for-pixel equality, animation/frame equivalence, metadata preservation or visual appearance. The original file is unlinked only after verification succeeds. The tool does not write to a temporary destination and atomically rename it: a failure during save/verification may leave an incomplete `.webp` requiring manual cleanup. It also does not rollback an earlier successful conversion if a later source fails.

For every successful image it prints original relative path, original decimal-megabyte size and target decimal-megabyte size. These figures are file sizes computed from `stat()`, not measured visual quality or a promise that WebP is always smaller.

## Rewrite Markdown references after conversion

The script builds an old-path/new-path map for **successfully converted files only**. [`_rewrite_links()`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/convert_assets_to_webp.py#L32-L48) constructs string pairs for raw repository-relative POSIX paths and URL-encoded variants when `quote(old) != old`. For example, paths containing spaces can have an encoded spelling as well as a raw spelling. It scans every `content/**/*.md` recursively, and for any changed document rewrites the **entire UTF-8 file**:

```python
updated = text
for old, new in pairs:
    updated = updated.replace(old, new)
if updated != text:
    document.write_text(updated, encoding="utf-8")
    changed.append(document)
```

This is literal substring replacement, not Markdown parsing. It can change matching text outside a link, including examples or prose; it can also miss references written with a different path convention. It does not update Markdown outside `content/`, source code, styles, generated website assets or external documents. The conversion order (all eligible images first, then Markdown links) means a failure mid-run can leave converted/deleted images with old Markdown references. `--dry-run` passes an empty rename map, so even though the rewrite function is called, it changes nothing.

## Operational limits and verification

There is no checkpoint, transaction, force-overwrite, backup or automatic rollback. Existing WebP targets abort the corresponding conversion rather than being replaced. A GIF may have animation or color/metadata subtleties not captured by the single-image save path; inspect outputs where those matter. When evaluating a real conversion, check the source images and version-control diff first, then inspect every changed Markdown document. Do not execute the destructive mode merely to verify documentation.

The [source entry point](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/convert_assets_to_webp.py#L51-L77) is the evidence for supported flags and side effects. No dedicated automated test of this entry point was identified in the inspected test sources; this is an explicit coverage limitation, not a test result. The site's separate [`sync-assets.mjs`](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/website/scripts/sync-assets.mjs) handles website asset copying and is outside this script's responsibility.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/scripts/convert_assets_to_webp.py">convert_assets_to_webp.py · pinned revision</a></dd>
    <dt>Related</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/website/scripts/sync-assets.mjs">website asset synchronization</a> · separate responsibility</dd>
    <dt>Verification</dt><dd>Static source inspection only; no destructive conversion performed</dd>
  </dl>
</aside>
