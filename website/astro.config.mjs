import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHeadings from "./src/markdown/rehype-headings.mjs";
import rehypeStripPreamble from "./src/markdown/rehype-strip-preamble.mjs";

export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeStripPreamble, rehypeHeadings],
  },
});