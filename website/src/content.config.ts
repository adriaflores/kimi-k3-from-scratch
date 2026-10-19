import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

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

// Docs is a versioned file-by-file reference, not the legacy repository docs/ folder.
const docs = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "../content/docs" }),
  schema: z.object({
    title: z.string().trim().min(1),
    description: z.string().trim().min(1),
    sourcePath: z.string().regex(/^(src\/kimi_k3\/|scripts\/).+\.py$/),
    status: z.enum(["draft", "review", "published"]),
  }).strict(),
});
export const collections = { architecture, training, docs };
