import { getCollection } from "astro:content";
import { DOC_FILES, docId } from "../data/docs-files";

export interface TreeNode {
  name: string;
  fullPath: string;
  children: TreeNode[];
  description?: string;
}

/** Single tree from the frozen manifest; no invented conceptual categories. */
export function createDocsTree(): TreeNode[] {
  const roots: TreeNode[] = [];
  for (const [file, description] of DOC_FILES) {
    const parts = file.split("/");
    let level = roots;
    let prefix = "";
    for (const [index, part] of parts.entries()) {
      prefix = prefix ? `${prefix}/${part}` : part;
      let node = level.find(item => item.name === part);
      if (!node) {
        node = { name: part, fullPath: prefix, children: [] };
        level.push(node);
      }
      if (index === parts.length - 1) node.description = description;
      level = node.children;
    }
  }
  return roots;
}

/** Visible routes only: dev previews review/draft; production requires published. */
export async function getDocsEntries() {
  const entries = await getCollection("docs");
  const allowed = new Set<string>(DOC_FILES.map(([path]) => path));
  const bySource = new Map<string, (typeof entries)[number]>();
  for (const entry of entries) {
    const source = entry.data.sourcePath;
    if (!allowed.has(source)) throw new Error(`[docs] Unknown source path: ${source}`);
    if (docId(source) !== entry.id) {
      throw new Error(`[docs] ${entry.id} does not match source ${source}`);
    }
    if (bySource.has(source)) throw new Error(`[docs] Duplicate page for ${source}`);
    bySource.set(source, entry);
  }
  return entries.filter(entry => import.meta.env.DEV || entry.data.status === "published");
}
