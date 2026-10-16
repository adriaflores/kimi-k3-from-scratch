import { getCollection, type CollectionEntry } from "astro:content";

export type ChapterCollection = "architecture" | "training";
export type ChapterEntry = CollectionEntry<"architecture"> | CollectionEntry<"training">;

/** Absolute chapter links must not depend on the currently open section. */
export function chapterHref(section: ChapterCollection, id: string): string {
  return `/${section}/${id}/`;
}

/**
 * Validate the entire inventory, including drafts, before deciding what can
 * be shown. Two authors accidentally choosing the same order is an error.
 * During astro dev, review/draft chapters are previewable; static builds
 * publish ONLY chapters whose editorial status is `published`.
 */
export async function getVisibleChapters<C extends ChapterCollection>(section: C) {
  const entries = await getCollection(section);
  const orders = new Map<number, string>();

  for (const entry of entries) {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(entry.id)) {
      throw new Error(`[${section}] Invalid chapter ID "${entry.id}". Use a kebab-case Markdown filename.`);
    }
    const previous = orders.get(entry.data.order);
    if (previous) {
      throw new Error(`[${section}] Duplicate chapter order ${entry.data.order}: "${previous}" and "${entry.id}".`);
    }
    orders.set(entry.data.order, entry.id);
  }

  const visible = import.meta.env.DEV
    ? entries
    : entries.filter((entry) => entry.data.status === "published");

  return visible.sort((a, b) => a.data.order - b.data.order || a.id.localeCompare(b.id));
}
