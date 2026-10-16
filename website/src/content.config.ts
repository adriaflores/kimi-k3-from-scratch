import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

// One editorial contract for both kinds of chapter. The section is determined
// by the collection, and the stable URL identifier comes from the filename.
const chapterSchema = z.object({
  title: z.string().trim().min(1),
  description: z.string().trim().min(1),
  question: z.string().trim().min(1),
  order: z.number().int().positive(),
  status: z.enum(["draft", "review", "published"]),
}).strict();

const architecture = defineCollection({
  loader: glob({ pattern: "*.md", base: "../content/architecture" }),
  schema: chapterSchema,
});

const training = defineCollection({
  loader: glob({ pattern: "*.md", base: "../content/training" }),
  schema: chapterSchema,
});

// Existing repository documentation is still maintained independently.
// Its web routes remain intentionally deferred until W3.
const docs = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "../docs" }),
});

export const collections = { architecture, training, docs };
