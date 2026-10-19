import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHeadings from "./src/markdown/rehype-headings.mjs";

export default defineConfig({
  site: "https://kimi-k3.adriaflores.com",
  output: "static",
  markdown: {
    shikiConfig: {
      themes: { light: "one-light", dark: "one-dark-pro" },
    },
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, rehypeHeadings],
    }),
  },
});