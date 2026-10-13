#!/usr/bin/env python3
"""Make a single M7.1 handoff ZIP with local receipts and verified remote ZIP."""
import argparse
import hashlib
import json
import pathlib
import zipfile
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'results/m7'
LOCAL=OUT/'m71'
REFERENCE=ROOT/'inputs/m7'
FILES=['local_preflight.json','remote_pre.json','helpers.json','upload_full.json','upload_summary.json','remote_post.json']
p=argparse.ArgumentParser();p.add_argument('--remote-zip',type=pathlib.Path,required=True);p.add_argument('--run-id',required=True)
a=p.parse_args()
assert all(c.isalnum() or c in '_-' for c in a.run_id)
with zipfile.ZipFile(a.remote_zip) as f:
    code=json.loads(f.read('data/summary.json'))
    if code.get('status') != 'passed': raise RuntimeError('Remote acceptance failed; use remote ZIP for diagnosis')
for name in FILES:
    path=LOCAL/name
    if not path.is_file():raise FileNotFoundError(f'Missing local report {path}')
    report=json.loads(path.read_text(encoding='utf-8'))
    if report.get('status') != 'passed':raise RuntimeError(f'Failed local stage: {name}')
target=OUT/(a.run_id+'_m71_handoff.zip')
if target.exists():raise FileExistsError('Refusing to overwrite handoff ZIP')
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
    out.write(a.remote_zip,'remote_results.zip')
    for name in FILES:out.write(LOCAL/name,f'local/{name}')
    for path in [REFERENCE/'reference_metadata.json',REFERENCE/'origin_inspection/verified_data_manifest.json']:
        out.write(path,'reference/'+path.name)
sha=hashlib.sha256(target.read_bytes()).hexdigest()
print(f'HANDOFF: {target} ({target.stat().st_size} bytes) SHA-256 {sha}')
print('Share the handoff ZIP; no shards, checkpoint, credentials or secret SSH config inside.')
