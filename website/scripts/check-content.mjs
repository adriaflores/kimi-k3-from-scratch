/**
 * M1.4: lightweight editorial source/figure checker (no extra dependencies).
 * Human verification of scientific claims, links to external sites and
 * experiment provenance is still required before setting status: published.
 */
import { readFile, readdir, access } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const chapterDirs = ["architecture", "training"];
const knownFigureKinds = new Set(["conceptual", "adapted", "measurement"]);

function attr(markup, name) {
  const found = markup.match(new RegExp(`\\b${name}\\s*=\\s*(["'])(.*?)\\1`, "is"));
  return found?.[2] ?? "";
}

function resolveAsset(src) {
  if (!src.startsWith("/assets/")) return null;
  let relative;
  try { relative = decodeURIComponent(src.slice(1).split(/[?#]/, 1)[0]); }
  catch { return null; }
  const fullPath = path.resolve(repoRoot, relative);
  if (!fullPath.startsWith(path.join(repoRoot, "assets") + path.sep)) return null;
  return fullPath;
}

/** Returns findings without accessing external services or external URLs. */
export async function inspectChapter({ chapter, chapterId, text, exists = async file => {
  try { await access(file); return true; } catch { return false; }
} }) {
  const findings = [];
  const status = text.match(/^status:\s*["']?(draft|review|published)["']?\s*$/m)?.[1] ?? "draft";
  const severity = status === "published" ? "error" : "warning";
  const add = (message, level = severity) => findings.push({ chapter: `${chapter}/${chapterId}`, level, message });
  const figureIds = new Set();
  const figureMatches = [...text.matchAll(/<figure\b([^>]*)>([\s\S]*?)<\/figure>/gi)];

  for (const [, attrs, body] of figureMatches) {
    const id = attr(attrs, "id");
    const kind = attr(attrs, "data-kind");
    const source = attr(attrs, "data-source");
    if (!id || !/^fig-[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)) add("Each figure needs a stable kebab-case id beginning with fig-.");
    else if (figureIds.has(id)) add(`Duplicate figure id: ${id}`);
    else figureIds.add(id);
    if (!knownFigureKinds.has(kind)) add(`Figure ${id || "(unnamed)"} needs data-kind=conceptual|adapted|measurement.`);
    const caption = body.match(/<figcaption\b[^>]*>([\s\S]*?)<\/figcaption>/i)?.[1] ?? "";
    if (!caption.replace(/<[^>]+>/g, "").trim()) add(`Figure ${id} needs a meaningful figcaption.`);
    if (status === "published" && !/<strong>Figure \d+[.:]<\/strong>/i.test(caption)) add(`Figure ${id} caption should begin with a numbered Figure label.`);
    const img = body.match(/<img\b([^>]*?)\/?\s*>/i);
    if (!img) add(`Figure ${id} needs an <img> (M1.4 supports static image figures).`);
    else {
      const src = attr(img[1], "src");
      const alt = attr(img[1], "alt");
      if (!alt.trim() || /^(image|figure|diagram|chart)$/i.test(alt.trim())) add(`Figure ${id} needs descriptive alt text.`);
      if (!src.startsWith(`/assets/site/${chapter}/${chapterId}/`)) add(`Figure ${id} must use an owned asset path under /assets/site/${chapter}/${chapterId}/.`);
      const fullPath = resolveAsset(src);
      if (!fullPath || !(await exists(fullPath))) add(`Figure ${id} asset not found: ${src}`);
    }
    if ((kind === "adapted" || kind === "measurement") && !/^https?:\/\//.test(source)) {
      add(`Figure ${id} (${kind}) needs a full data-source URL pointing to the original source or evidence artifact.`);
    }
    if ((kind === "adapted" || kind === "measurement") && !(source && caption.includes(`href="${source}"`) || source && caption.includes(`href='${source}'`))) {
      add(`Figure ${id} (${kind}) caption must link the exact data-source URL.`);
    }
  }

  // Resolve figure links in prose: [Figure 1](#fig-example).
  for (const match of text.matchAll(/\[[^\]]*\]\(#(fig-[a-z0-9-]+)\)/gi)) {
    if (!figureIds.has(match[1])) add(`Broken in-page figure link: #${match[1]}`);
  }

  // Local Markdown images (outside HTML figures) are supported, but not a
  // substitute for a captioned figure when making research claims.
  for (const match of text.matchAll(/!\[([^\]]*)\]\((\/assets\/[^)\s]+)(?:\s+"[^"]*")?\)/g)) {
    if (!match[1].trim()) add(`Markdown image needs descriptive alt text: ${match[2]}`);
    const fullPath = resolveAsset(match[2]);
    if (!fullPath || !(await exists(fullPath))) add(`Markdown image asset not found: ${match[2]}`);
  }

  if (status === "published") {
    if (!/class=["'][^"']*\breading-sources\b/.test(text)) add("Published chapters need a compact source trail (reading-sources).");
    // We only guard the Kimi K3 paper and canonical repository; the reviewer
    // must validate section/equation IDs and verify links actually resolve.
    for (const match of text.matchAll(/https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\/(2607\.24653)(?!v\d+)/g)) {
      add(`Kimi K3 paper link must include a checked arXiv version: ${match[0]}`);
    }
    for (const match of text.matchAll(/https?:\/\/github\.com\/adriaflores\/kimi-k3-from-scratch\/(?:blob|tree)\/([^/\s)"']+)/g)) {
      if (!/^[a-f\d]{7,40}$/i.test(match[1])) add(`Canonical code/test link must use a pinned commit, not '${match[1]}'.`);
    }
  }
  return findings;
}

export async function inspectAll({ root = repoRoot, exists } = {}) {
  const findings = [];
  for (const chapter of chapterDirs) {
    const folder = path.join(root, "content", chapter);
    const files = (await readdir(folder)).filter(file => file.endsWith(".md"));
    for (const file of files) {
      const text = await readFile(path.join(folder, file), "utf8");
      findings.push(...await inspectChapter({chapter, chapterId: file.slice(0, -3), text, exists}));
    }
  }
  return findings;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const findings = await inspectAll();
  for (const { chapter, level, message } of findings) {
    process.stderr.write(`[${level}] ${chapter}: ${message}\n`);
  }
  const errors = findings.filter(finding => finding.level === "error");
  process.stdout.write(`M1.4 content check: ${errors.length} errors, ${findings.length - errors.length} draft/review warnings.\n`);
  if (errors.length) process.exitCode = 1;
}
