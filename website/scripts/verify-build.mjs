/**
 * M1.6: verify the *emitted production site*, not just Markdown source.
 * Runs after astro build; uses only Node's standard library.
 * The content schema and editorial figure/source rules are checked elsewhere.
 */
import { readFile, readdir, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const defaultRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const sections = ['architecture', 'training'];
const exists = async (p) => { try { await access(p); return true; } catch { return false; } };

/** No filesystem changes, no network requests. Returns actionable failures. */
export async function inspectBuiltSite({root=defaultRoot, dist=path.join(root, 'website/dist')}={}) {
  const issues=[];
  const check=(ok, message)=>{if(!ok)issues.push(message);};
  check(await exists(path.join(dist,'index.html')), 'Missing built home page. Run npm run build first.');
  for (const section of sections) {
    const listing=await readdir(path.join(root, 'content', section), {withFileTypes:true});
    const indexPath=path.join(dist, section, 'index.html');
    const index=await exists(indexPath) ? await readFile(indexPath,'utf8') : '';
    check(Boolean(index), `Missing /${section}/ index in production build.`);
    const seenOrders=new Map();
    for (const entry of listing.filter(x=>x.isFile() && x.name.endsWith('.md'))) {
      const id=entry.name.slice(0,-3);
      const text=await readFile(path.join(root,'content',section,entry.name),'utf8');
      const fm=text.match(/^---\s*\n([\s\S]*?)\n---(?:\s*\n|$)/);
      if (!fm) {issues.push(`${section}/${id}: missing frontmatter`);continue;}
      const status=fm[1].match(/^status:\s*["']?(draft|review|published)["']?\s*$/m)?.[1];
      const order=Number(fm[1].match(/^order:\s*(\d+)\s*$/m)?.[1] ?? NaN);
      if (!status) issues.push(`${section}/${id}: missing/invalid status`);
      if (!Number.isInteger(order) || order<1) issues.push(`${section}/${id}: invalid order`);
      else if (seenOrders.has(order)) issues.push(`${section}: duplicate order ${order}: ${seenOrders.get(order)} and ${id}`);
      else seenOrders.set(order,id);
      const chapterPath=path.join(dist,section,id,'index.html');
      const generated=await exists(chapterPath);
      const canonical=`/${section}/${id}/`;
      if (status==='published') {
        check(generated,`${canonical} is published in Markdown but absent from dist.`);
        check(index.includes(`href="${canonical}"`), `${canonical} is published but not linked by its index.`);
        if (generated) {
          const html=await readFile(chapterPath,'utf8');
          check(/<article\b[^>]*\bchapter-body\b/.test(html),`${canonical} lacks its article reading layout.`);
          check(/class="[^"]*reading-sources/.test(html), `${canonical} lacks a rendered Sources & verification block.`);
          const ids=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]));
          for (const [,fragment] of html.matchAll(/\bhref="#([a-zA-Z0-9-]+)"/g)) {
            if (!ids.has(fragment)) issues.push(`${canonical}: broken in-page anchor #${fragment}`);
          }
          for (const [,src] of html.matchAll(/\bsrc="(\/assets\/site\/[^"?#]+)"/g)) {
            let relative; try {relative=decodeURIComponent(src.slice(1));}catch{issues.push(`${canonical}: invalid asset encoding ${src}`);continue;}
            const full=path.resolve(dist,relative);
            if(!full.startsWith(path.join(dist,'assets')+path.sep) || !(await exists(full))) issues.push(`${canonical}: built asset is missing: ${src}`);
          }
        }
      } else if(status) {
        check(!generated,`${canonical} has status ${status}, but leaked into production dist.`);
        check(!index.includes(`href="${canonical}"`), `${canonical} has status ${status}, but leaked into the production index.`);
      }
    }
  }
  return issues;
}

if(process.argv[1] && fileURLToPath(import.meta.url)===path.resolve(process.argv[1])) {
  const issues=await inspectBuiltSite();
  for(const issue of issues) process.stderr.write(`M1.6 production QA: ${issue}\n`);
  process.stdout.write(`M1.6 production QA: ${issues.length ? `${issues.length} issue(s)` : 'PASS (published chapters present, drafts excluded, links/assets checked)'}.\n`);
  if (issues.length) process.exitCode=1;
}
