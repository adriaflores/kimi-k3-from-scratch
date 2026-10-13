#!/usr/bin/env python3
"""Produce portable handoff ZIP with local+remote M7.2 proof. No binary training assets."""
import argparse
import hashlib
import json
import pathlib
import zipfile
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'results/m7'
ASSETS=ROOT/'experiments/m7/current'
p=argparse.ArgumentParser()
p.add_argument('--remote-zip',type=pathlib.Path,required=True)
p.add_argument('--run-id',required=True)
a=p.parse_args()
if not all(c.isalnum() or c in '_-' for c in a.run_id):raise SystemExit('Bad run ID')
local=OUT/'m72/local_preflight.json'
if json.loads(local.read_text(encoding='utf-8')).get('status')!='passed':raise SystemExit('M72 local preflight not approved')
with zipfile.ZipFile(a.remote_zip) as z:
    if json.loads(z.read('summary.json'))['status']!='passed':
        raise SystemExit('M72 remote audit failed')
previous=list(OUT.glob('*_m71_handoff.zip'))
if not previous:raise SystemExit('Missing earlier M7.1 handoff; run scripts/m7/download.sh first')
target=OUT/(a.run_id+'_m72_handoff.zip')
if target.exists():raise SystemExit('Handoff already exists')
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
    z.write(a.remote_zip,'m72_remote_results.zip')
    z.write(local,'m72_local_preflight.json')
    for file in ('pilot_80m.yaml','pilot_contract.json','m6_baseline_train_config.json'):
        z.write(ASSETS/file,'recipe/'+file)
    evidence=sorted(previous,key=lambda f:f.stat().st_mtime)[-1]
    z.write(evidence,'m71_handoff.zip')
print(f'M7.2 HANDOFF {target} SHA-256 {hashlib.sha256(target.read_bytes()).hexdigest()}')
