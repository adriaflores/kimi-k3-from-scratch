import test from "node:test";
import assert from "node:assert/strict";
import { inspectChapter } from "./check-content.mjs";

const base = `---
title: Test
description: Demonstration
question: Why?
order: 1
status: published
---
`;
const sources = `<aside class="reading-sources"><a href="https://arxiv.org/abs/2607.24653v2">Paper v2</a></aside>`;
const figure = (attrs = `id="fig-memory" class="reading-figure" data-kind="conceptual"`, extra = "") => `<figure ${attrs}><a href="/assets/site/architecture/demo/memory.svg"><img src="/assets/site/architecture/demo/memory.svg" alt="Tokens reach a recurrent state" width="600" height="300" /></a><figcaption><strong>Figure 1.</strong> Conceptual memory sketch. ${extra}</figcaption></figure>`;
const inspect = async (text, exists = async () => true) => inspectChapter({chapter: "architecture",chapterId:"demo",text,exists});

test("well-formed conceptual figure and source trail", async () => {
  assert.deepEqual(await inspect(base + figure() + "\n" + sources + "\n[Figure 1](#fig-memory)"), []);
});
test("unpublished drafts report warnings but do not block", async () => {
  const findings = await inspect(base.replace("published", "draft") + figure().replace("alt=\"Tokens reach a recurrent state\"", "alt=\"\""));
  assert.ok(findings.some(f=>f.level === "warning" && f.message.includes("alt text")));
});
test("missing image, caption and source are detected", async () => {
  const findings = await inspect(base + figure(`id="fig-memory" data-kind="measurement"`).replace(/<figcaption>[\s\S]*?<\/figcaption>/, "") + "\n" + sources, async()=>false);
  assert.ok(findings.some(f=>f.message.includes("asset not found")));
  assert.ok(findings.some(f=>f.message.includes("figcaption")));
  assert.ok(findings.some(f=>f.message.includes("data-source")));
});
test("adapted figure carries provenance", async () => {
  const attrs=`id="fig-memory" data-kind="adapted" data-source="https://arxiv.org/abs/2607.24653v2"`;
  const findings=await inspect(base + figure(attrs, `<a href="https://arxiv.org/abs/2607.24653v2">Source</a>`) + sources);
  assert.deepEqual(findings, []);
});
test("unversioned paper, mutable code link and broken figure ref are detected", async () => {
  const text=base + sources + figure() + `\n[Paper](https://arxiv.org/abs/2607.24653) [Code](https://github.com/adriaflores/kimi-k3-from-scratch/blob/main/src/kimi_k3/model.py) [Figure 2](#fig-unknown)`;
  const messages=(await inspect(text)).map(f=>f.message).join("\n");
  for (const snippet of ["arXiv version", "pinned commit", "Broken in-page figure link"]) assert.ok(messages.includes(snippet));
});
test("an unrelated zoom link cannot masquerade as provenance", async () => {
  const source = `https://arxiv.org/abs/2607.24653v2`;
  const findings = await inspect(base + figure(`id="fig-memory" data-kind="adapted" data-source="${source}"`) + sources);
  assert.ok(findings.some(f => f.message.includes("caption must link the exact data-source")));
});
test("duplicate figure ids and path traversal are detected", async () => {
  const malformed=figure().replaceAll("/assets/site/architecture/demo/memory.svg","/assets/../private/memory.svg");
  const findings=await inspect(base + sources + figure() + malformed);
  assert.ok(findings.some(f=>f.message.includes("Duplicate figure id")));
  assert.ok(findings.some(f=>f.message.includes("owned asset path")));
});

test("reader questions require a visible answer", async () => {
  const question='<aside class="reading-question"><p>Why?</p></aside>';
  const findings=await inspect(base+figure()+sources+question);
  assert.ok(findings.some(f=>f.level==='error' && f.message.includes('visible .reading-answer')));
});
test("an answered reader question is valid", async () => {
  const question='<aside class="reading-question"><p>Why?</p><p class="reading-answer">Because the model needs the next target.</p></aside>';
  assert.deepEqual(await inspect(base+figure()+sources+question),[]);
});
