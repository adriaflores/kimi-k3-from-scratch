import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import sitemap from "@astrojs/sitemap";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHeadings from "./src/markdown/rehype-headings.mjs";

const excludedPaths = new Set([
  "/404.html",
  "/engineering/",
  "/paper-to-code/",
]);

export default defineConfig({
  site: "https://kimi-k3.adriaflores.com",
  output: "static",
  integrations: [
    sitemap({
      filter: (page) => !excludedPaths.has(new URL(page).pathname),
    }),
  ],
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