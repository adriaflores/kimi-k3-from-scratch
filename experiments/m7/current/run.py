#!/usr/bin/env python3
"""Remote M7.1 acceptance; verifies corpus and M6 runtime, NO training."""
from __future__ import annotations
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=pathlib.Path('/workspace/kimi')
DATA=ROOT/'data/full'
HELPER=ROOT/'ops/m7/branch4_remote_ops.py'
REF=ROOT/'ops/m7/reference_metadata.json'
PINNED=ROOT/'control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python'

def check() -> dict:
    start=time.monotonic()
    a={"time_utc":datetime.now(timezone.utc).isoformat(),"phase":"m71-acceptance","status":"failed","training_started":False}
    if not ROOT.joinpath('reference').is_dir(): raise RuntimeError('Missing immutable M6 source reference')
    if not PINNED.is_file(): raise RuntimeError('Missing pinned M6 training runtime')
    if not HELPER.is_file() or not REF.is_file(): raise RuntimeError('Missing staged dataset validation helpers')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True)
    if gpu.returncode or 'H100 NVL' not in gpu.stdout: raise RuntimeError('Expected H100 NVL is not accessible')
    a['gpu']=gpu.stdout.strip()
    py=subprocess.run([str(PINNED),'-c','import sys, torch; print(sys.version.split()[0],torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)'],capture_output=True,text=True)
    if py.returncode: raise RuntimeError('Pinned runtime import failed: '+py.stderr[-800:])
    a['runtime']=py.stdout.strip()
    if 'True' not in py.stdout or '(9, 0)' not in py.stdout: raise RuntimeError('H100 capability unavailable to pinned PyTorch')
    verify=subprocess.run([sys.executable,str(HELPER),'verify-dataset','--data-dir',str(DATA),'--reference-metadata',str(REF),'--scope','full'],capture_output=True,text=True)
    if verify.returncode:
        raise RuntimeError('Full-corpus verification failed: '+(verify.stdout+verify.stderr)[-2500:])
    proof=json.loads(verify.stdout)
    if not proof.get('verified') or proof.get('files')!=254 or proof.get('bytes')!=12_633_283_564:
        raise RuntimeError(f'Unexpected complete corpus validation: {proof}')
    a['dataset_verification']=proof
    free=shutil.disk_usage(ROOT).free
    a['free_after_bytes']=free
    if free<20*1024**3: raise RuntimeError('Less than 20 GiB available after deployment')
    a['duration_seconds']=round(time.monotonic()-start,3)
    a['status']='passed'
    return a

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=pathlib.Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        report=check()
    except (OSError,ValueError,KeyError,RuntimeError,TypeError) as exc:
        report={"phase":"m71-acceptance","status":"failed","training_started":False,"error":f'{type(exc).__name__}: {exc}',"time_utc":datetime.now(timezone.utc).isoformat()}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
