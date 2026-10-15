import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHeadings from "./src/markdown/rehype-headings.mjs";

export default defineConfig({
  site: "https://kimi-k3.adriaflores.com",
  output: "static",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeHeadings],
  },
});
