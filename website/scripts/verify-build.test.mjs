import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, writeFile, rm} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {inspectBuiltSite} from './verify-build.mjs';
import { execFileSync } from "node:child_process";

async function fixture(fn){
 const root=await mkdtemp(path.join(os.tmpdir(),'kimi-build-'));
 const put=async (name,text)=>{const p=path.join(root,name);await mkdir(path.dirname(p),{recursive:true});await writeFile(p,text);};
 try{
  await put('website/dist/index.html','<main>home</main>');
  await put('website/dist/architecture/index.html','<a href="/architecture/published/">Read</a>');
  await put('website/dist/training/index.html','<main>training</main>');
  await put('content/architecture/published.md','---\nstatus: published\norder: 1\n---\n');
  await put('content/architecture/draft.md','---\nstatus: draft\norder: 2\n---\n');
  await put('content/training/example.md','---\nstatus: review\norder: 1\n---\n');
  await put('website/dist/architecture/published/index.html','<article class="chapter-body doc-prose"><h2 id="understanding">Hi</h2><a href="#understanding">Jump</a><aside class="reading-sources">Sources</aside><img src="/assets/site/architecture/published/fig.svg"></article>');
  await put('website/dist/assets/site/architecture/published/fig.svg','<svg/>');
  return await fn({root,put});
 } finally {await rm(root,{recursive:true,force:true});}
}

test('a published chapter renders while draft and review stay private', async()=>{
 await fixture(async ({root})=>assert.deepEqual(await inspectBuiltSite({root}),[]));
});
test('a draft chapter accidentally emitted is blocked', async()=>{
 await fixture(async({root,put})=>{
  await put('website/dist/architecture/draft/index.html','draft');
  assert.match((await inspectBuiltSite({root})).join('\n'),/leaked into production dist/);
 });
});
test('a published chapter missing from the section index is blocked', async()=>{
 await fixture(async({root,put})=>{
  await put('website/dist/architecture/index.html','<a>no link</a>');
  assert.match((await inspectBuiltSite({root})).join('\n'),/not linked/);
 });
});
test('a missing built figure and broken in-page anchor are reported', async()=>{
 await fixture(async({root,put})=>{
  await put('website/dist/architecture/published/index.html','<article class="chapter-body doc-prose"><aside class="reading-sources"></aside><a href="#lost">Lost</a><img src="/assets/site/architecture/published/lost.svg"></article>');
  const problems=(await inspectBuiltSite({root})).join('\n');
  assert.match(problems,/broken in-page anchor/);
  assert.match(problems,/built asset is missing/);
 });
});
test('duplicate chapter orders are detected before parallel integration', async()=>{
 await fixture(async({root,put})=>{
  await put('content/architecture/duplicate.md','---\nstatus: draft\norder: 1\n---\n');
  assert.match((await inspectBuiltSite({root})).join('\n'),/duplicate order/);
 });
});

test("Results images and downloads must exist in the build", async () => {
  await fixture(async ({ root, put }) => {
    await put(
      "website/dist/results/index.html",
      '<img src="/assets/site/results/training-monitoring.png">' +
      '<a href="/assets/site/results/verified-evidence.zip" download>Evidence</a>'
    );

    const missing = (await inspectBuiltSite({ root })).join("\n");

    assert.match(
      missing,
      /built asset is missing: \/assets\/site\/results\/training-monitoring\.png/
    );

    assert.match(
      missing,
      /built asset is missing: \/assets\/site\/results\/verified-evidence\.zip/
    );

    await put(
      "website/dist/assets/site/results/training-monitoring.png",
      "image"
    );

    await put(
      "website/dist/assets/site/results/verified-evidence.zip",
      "archive"
    );

    assert.deepEqual(await inspectBuiltSite({ root }), []);
  });
});
test("Untracked Results assets fail even when present in dist", async () => {
  await fixture(async ({ root, put }) => {
    await put(
      "assets/site/architecture/published/fig.svg",
      "<svg/>"
    );

    await put(
      "assets/site/results/training-monitoring.png",
      "image"
    );

    await put(
      "website/dist/assets/site/results/training-monitoring.png",
      "image"
    );

    await put(
      "website/dist/results/index.html",
      '<img src="/assets/site/results/training-monitoring.png">'
    );

    execFileSync("git", ["init", "-q"], { cwd: root });

    execFileSync(
      "git",
      ["add", "assets/site/architecture/published/fig.svg"],
      { cwd: root }
    );

    const missing = (await inspectBuiltSite({ root })).join("\n");

    assert.match(
      missing,
      /asset is not tracked by Git: \/assets\/site\/results\/training-monitoring\.png/
    );

    execFileSync(
      "git",
      ["add", "assets/site/results/training-monitoring.png"],
      { cwd: root }
    );

    assert.deepEqual(await inspectBuiltSite({ root }), []);
  });
});