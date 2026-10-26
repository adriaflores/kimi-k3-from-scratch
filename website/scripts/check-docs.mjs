/** Docs gate. Runs on an integrated full tree; content is still in editorial review. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const website = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(website, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(website,'src/data/docs-files.json'),'utf8'));
const docsDir = path.join(root,'content/docs');
const wanted = new Set(manifest.files.map(x=>x.sourcePath));
// Release utilities added after the frozen 37-file documentation snapshot.
// They are documented in the repository README and Hugging Face Model Card.
const postSnapshotScripts = new Set([
  'scripts/export_hf.py',
  'scripts/generate_hf.py',
]);
const issues = [];
const errors = x => issues.push(x);
const sha = manifest.sourceCommit;
if (!/^[0-9a-f]{40}$/.test(sha)) errors('Invalid source commit hash');
if (wanted.size!==37 || wanted.size!==manifest.files.length) errors('Manifest must contain 37 distinct Python files');
function walk(d) { return fs.existsSync(d) ? fs.readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(d,e.name)):(e.isFile() && e.name.endsWith('.md') ? [path.join(d,e.name)] : [])) : []; }
const entries = walk(docsDir);
const seen = new Set();
const lineCounts = new Map();
function sourceLines(file) {
 if (lineCounts.has(file)) return lineCounts.get(file);
 const resolved=path.join(root,file);
 if (!fs.existsSync(resolved)) return null;
 const text=fs.readFileSync(resolved,'utf8');
 const n=text===''?0:text.split(/\r?\n/).length;
 lineCounts.set(file,n); return n;
}
const github = /^https:\/\/github\.com\/adriaflores\/kimi-k3-from-scratch\/blob\/([0-9a-f]{40})\/([^)#\s"<>]+)(?:#L(\d+)(?:-L(\d+))?)?$/;
let checkedRefs=0;
for (const f of entries) {
 const t=fs.readFileSync(f,'utf8');
 const rel=path.relative(docsDir,f).split(path.sep).join('/');
 const fm=t.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/)?.[1];
 if (!fm) { errors(`Missing frontmatter: ${rel}`); continue; }
 const source=fm.match(/^sourcePath:\s*["']([^"']+)["']\s*$/m)?.[1];
 const status=fm.match(/^status:\s*(draft|review|published)\s*$/m)?.[1];
 if (!status) errors(`Invalid status: ${rel}`);
 if (source!==rel.replace(/\.md$/,'.py')) errors(`Mismatched sourcePath: ${rel}`);
 if (!wanted.has(source)) errors(`Unlisted source: ${source}`);
 if (seen.has(source)) errors(`Duplicate documentation: ${source}`);
 seen.add(source);
 if (/^# /m.test(t.slice(fm.length+8))) errors(`Duplicate H1 in ${rel}`);
 if ((t.match(/^```/gm)||[]).length%2!==0) errors(`Unbalanced fenced code in ${rel}`);
 const links=[...t.matchAll(/https:\/\/github\.com\/adriaflores\/kimi-k3-from-scratch\/blob\/[^\s)"<>]+/g)];
 if(!links.length) errors(`Missing GitHub source reference: ${rel}`);
 if(!links.some(x=>x[0].includes('/'+source))) errors(`No canonical source link for ${rel}`);
 for(const m of links){
   const url=m[0].replace(/[.,;]$/,'');
   const r=url.match(github);
   if(!r) { errors(`Malformed GitHub source URL: ${rel} => ${url}`);continue; }
   if(r[1]!==sha) errors(`Unpinned GitHub URL in ${rel}`);
   if(r[3]){
     checkedRefs++;
     const a=Number(r[3]),b=Number(r[4]||r[3]);const max=sourceLines(r[2]);
     if(a<1 || b<a || (max!==null && b>max)) errors(`Invalid line range ${r[2]}#L${a}-L${b} from ${rel}`);
   }
 }
 const internal=[...t.matchAll(/(?:\]\(|href=["'])(\/docs\/(?:src|scripts)\/[A-Za-z0-9_/-]+\/)(?:[)"'])/g)];
 for(const m of internal){
   const target=m[1].slice('/docs/'.length,-1)+'.md';
   if(!fs.existsSync(path.join(docsDir,target))) errors(`Broken Docs link ${m[1]} in ${rel}`);
 }
}
for(const file of wanted) if(!seen.has(file)) errors(`Missing Docs: ${file}`);
const actual=[];
const srcDir=path.join(root,'src/kimi_k3');
const scriptsDir=path.join(root,'scripts');
if(fs.existsSync(srcDir)&&fs.existsSync(scriptsDir)){
 for(const name of walkPy(srcDir)) actual.push(path.relative(root,name).split(path.sep).join('/'));
 for(const name of fs.readdirSync(scriptsDir)) if(name.endsWith('.py')&&fs.statSync(path.join(scriptsDir,name)).isFile()) actual.push(`scripts/${name}`);
 for (const file of actual) {
    if (!wanted.has(file) && !postSnapshotScripts.has(file)) {
      errors(`Unexpected source not documented: ${file}`);
    }
  }

  for (const file of postSnapshotScripts) {
    if (!actual.includes(file)) {
      errors(`Expected post-snapshot script missing: ${file}`);
    }
  }
 for(const file of wanted) if(!actual.includes(file)) errors(`Documented source absent: ${file}`);
}else console.warn('Source tree absent: offline source-content and inventory check skipped.');
function walkPy(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walkPy(path.join(dir,e.name)):e.isFile()&&e.name.endsWith('.py')?[path.join(dir,e.name)]:[]);}
console.log(`Docs: ${seen.size}/${wanted.size} authored, ${checkedRefs} pinned line references checked structurally.`);
if(issues.length){for(const e of issues)console.error('ERROR:',e);process.exitCode=1;} else console.log('Docs structural QA passed; scientific and browser review remain separate requirements.');
