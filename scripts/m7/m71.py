#!/usr/bin/env python3
"""M7.1: local/remote preflight, verified full-corpus upload, evidence bundle.

Read-only by default; upload writes only managed /workspace/kimi/data/full and helpers.
Does not rent, start, stop, destroy, train or modify source/optimizers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/m7/vendor'))
from branch4_data_common import canonical_hash, validate_metadata, inspect_shard, read_json, sha256_file, write_json
from branch4_transport import Connection

DEFAULT_DATA=Path('C:/datasets/fineweb_edu_6b')
INSPECTION=ROOT/'inputs/m7/origin_inspection'
REF=ROOT/'inputs/m7/reference_metadata.json'
MANIFEST=INSPECTION/'verified_data_manifest.json'
OUT=ROOT/'results/m7/m71'
EXPECTED_SHARDS=254
EXPECTED_BYTES=12_633_283_564
EXPECTED_TRAIN=6_000_001_675
EXPECTED_VAL=316_623_851
MIN_FREE_AFTER_UPLOAD=20*1024**3


def stamp(): return datetime.now(timezone.utc).isoformat()

def load_report(path:Path):
    if not path.is_file(): raise FileNotFoundError(f'Missing previous report: {path}')
    return read_json(path)

def save_report(name, report):
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/name
    report.setdefault('created_at_utc',stamp())
    write_json(target,report)
    print(f'[{report.get("status","unknown")}] {target}',flush=True)
    return target

def validate_reference():
    inspection=read_json(INSPECTION/'inspection.json')
    manifest=read_json(MANIFEST)
    reference=read_json(REF)
    if not inspection.get('dataset_verified') or inspection.get('errors'):
        raise RuntimeError('M5 origin inspection was not successful')
    records=validate_metadata(reference)
    if len(records)!=EXPECTED_SHARDS: raise ValueError('Unexpected shard count in metadata')
    if len(manifest['files'])!=EXPECTED_SHARDS: raise ValueError('Unexpected manifest count')
    if reference['splits']['train']['tokens']!=EXPECTED_TRAIN or reference['splits']['val']['tokens']!=EXPECTED_VAL:
        raise ValueError('Unexpected token counts')
    if manifest['metadata_canonical_sha256']!=canonical_hash(reference):
        raise ValueError('Manifest/metadata canonical digest mismatch')
    pairs={(r['file'],r['sha256'],r['tokens']) for r in records}
    expected={(r['file'],r['sha256'],r['tokens']) for r in manifest['files']}
    if pairs!=expected: raise ValueError('Manifest/metadata shard mismatch')
    if sum(r['file_bytes'] for r in manifest['files'])!=EXPECTED_BYTES:
        raise ValueError('Unexpected data payload total')
    return reference, manifest


def local_preflight(data:Path):
    started=time.monotonic()
    report={'phase':'local-preflight','status':'failed','data_dir':str(data),'full_hash_verification':True}
    try:
        reference,manifest=validate_reference()
        if not data.is_dir(): raise FileNotFoundError(f'Local data folder not found: {data}')
        metadata=data/'metadata.json'
        if sha256_file(metadata)!=manifest['metadata_file_sha256']:
            raise ValueError('Local metadata.json hash differs from the M5 verified manifest')
        if canonical_hash(read_json(metadata))!=canonical_hash(reference):
            raise ValueError('Local metadata content differs from verified source')
        listed={p.name for p in data.glob('*.npy')}
        expected={e['file'] for e in manifest['files']}
        if listed!=expected:
            raise ValueError(f'Unexpected missing or extra NPY shards (missing={len(expected-listed)}, extra={len(listed-expected)})')
        by_split=Counter()
        for idx,entry in enumerate(manifest['files'],1):
            row=inspect_shard(data,entry)
            if not (row['exists'] and row['header_ok'] and row['hash_ok']):
                raise ValueError(f'{entry["file"]}: {row.get("error")}')
            by_split[entry['split']]+=1
            if idx%25==0 or idx==EXPECTED_SHARDS:print(f'Local SHA-256: {idx}/{EXPECTED_SHARDS} shards',flush=True)
        if dict(by_split)!={'train':241,'val':13}: raise ValueError(f'Unexpected split counts: {by_split}')
        report.update(status='passed',files=EXPECTED_SHARDS,split_files=dict(by_split),bytes=EXPECTED_BYTES,
                      train_tokens=EXPECTED_TRAIN,val_tokens=EXPECTED_VAL,
                      metadata_sha256=manifest['metadata_file_sha256'],
                      manifest_sha256=sha256_file(MANIFEST),
                      duration_seconds=round(time.monotonic()-started,3))
    except (ValueError,OSError,KeyError,TypeError,SyntaxError) as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        report['duration_seconds']=round(time.monotonic()-started,3)
    save_report('local_preflight.json',report)
    return report['status']=='passed'


def make_connection(alias:str,output:Path):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',alias): raise ValueError('Invalid SSH alias')
    proc=subprocess.run(['ssh','-G',alias],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr)
    settings={}
    for l in proc.stdout.splitlines():
        x=l.split(' ',1)
        if len(x)==2 and x[0] not in settings:settings[x[0]]=x[1]
    port=int(settings['port'])
    user=settings['user']
    # Use the alias as Host so any ProxyJump/IdentityFile in ~/.ssh/config still applies.
    connection={'host':alias,'port':port,'user':user,'remote_root':'/workspace/kimi','remote_python':'python3'}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(f'Refusing to overwrite connection file: {output}')
    write_json(output,connection)
    try:os.chmod(output,0o600)
    except OSError:pass
    print(f'Connection file: {output} (alias {alias}; user {user}; port {port}; no passwords/keys included)')

REMOTE_CODE='''
import json,os,shutil,subprocess
from pathlib import Path
root=Path('/workspace/kimi'); data=root/'data/full'
usage=shutil.disk_usage(root if root.exists() else Path('/workspace'))
files=[p for p in data.glob('*.npy') if p.is_file() and not p.is_symlink()] if data.is_dir() else []
existing=sum(p.stat().st_size for p in files)
proc=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True)
result={'remote_root_exists':root.exists(),'data_full_exists':data.exists(),
 'data_existing_shards_count':len(files),'data_existing_shards_bytes_unverified':existing,
 'disk_free_bytes':usage.free,'disk_total_bytes':usage.total,'disk_used_bytes':usage.used,
 'reference_exists':(root/'reference').exists(),'venv_exists':(root/'control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python').is_file(),
 'gpu_query_ok':proc.returncode==0,'gpu_query':proc.stdout.strip().splitlines() if proc.returncode==0 else [],
 'python_version':subprocess.run(['python3','--version'],capture_output=True,text=True).stdout.strip()}
print(json.dumps(result))
'''


def remote_inventory(alias:str,post:bool):
    _,manifest=validate_reference()
    result={'phase':'remote-inventory-post' if post else 'remote-inventory-pre','status':'failed','ssh_alias':alias}
    try:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',alias): raise ValueError('Invalid SSH alias')
        cmd=['ssh','-T',alias,'python3 -c '+shlex.quote(REMOTE_CODE)]
        proc=subprocess.run(cmd,capture_output=True,text=True)
        if proc.returncode: raise RuntimeError(proc.stderr.strip() or f'SSH command failed {proc.returncode}')
        details=json.loads(proc.stdout.strip())
        result.update(details)
        if not details['reference_exists'] or not details['venv_exists']:
            raise RuntimeError('M6 frozen reference and pinned runtime must exist')
        if not details['gpu_query_ok'] or not any('H100 NVL' in s for s in details['gpu_query']):
            raise RuntimeError('Expected accessible H100 NVL')
        if post:
            if details['disk_free_bytes']<MIN_FREE_AFTER_UPLOAD:
                raise RuntimeError('Less than 20 GiB remaining after transfer')
        else:
            # Conservative before any verified remote receipt: require full footprint.
            # If retrying an almost-complete transfer, this may overestimate needs.
            remaining=EXPECTED_BYTES
            required=remaining+MIN_FREE_AFTER_UPLOAD
            result['estimated_new_bytes_needed']=remaining
            result['estimated_required_free_bytes']=required
            if details['disk_free_bytes']<required:raise RuntimeError('Not enough free disk for full corpus + 20 GiB reserve')
        result['status']='passed'
    except (OSError,ValueError,RuntimeError,KeyError,json.JSONDecodeError) as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    save_report('remote_post.json' if post else 'remote_pre.json',result)
    return result['status']=='passed'


def verified_upload(connection_file:Path,data:Path):
    existing=load_report(OUT/'local_preflight.json')
    if existing.get('status')!='passed':raise RuntimeError('Local preflight must pass first')
    if Path(existing['data_dir']).resolve()!=data.resolve():
        raise RuntimeError('Data path differs from the preflighted origin')
    pre=load_report(OUT/'remote_pre.json')
    if pre.get('status')!='passed':raise RuntimeError('Remote precheck must pass first')
    conn=Connection(read_json(connection_file))
    # Reuses the M5 protocol and writes a durable report even on failure.
    script=ROOT/'scripts/m7/vendor/branch4_transport.py'
    commands=[['install-helpers'],['upload-data','--scope','full','--data-dir',str(data),'--inspection-dir',str(INSPECTION)]]
    for label,args in zip(('helpers','upload_full'),commands):
        command=[sys.executable,str(script),'--connection',str(connection_file),
                 '--output',str(OUT/f'{label}.json'),*args]
        print('Running verified M7 transfer:',label,flush=True)
        code=subprocess.call(command)
        if code!=0:raise RuntimeError(f'{label} failed (exit code={code}). Re-running upload resumes safe partials.')
    upload=load_report(OUT/'upload_full.json')
    remote=upload.get('result',{}).get('destination_verification',{})
    if upload['status']!='passed' or not remote.get('verified') or remote.get('files')!=EXPECTED_SHARDS:
        raise RuntimeError('Server-side dataset validation did not pass')
    report={'phase':'m71-upload','status':'passed','destination':upload['result']['destination'],
            'shards_verified_at_destination':remote['files'],'remote_verified_bytes':remote['bytes'],
            'note':'Full dataset only; pilot not started; qualification checkpoint not used'}
    save_report('upload_summary.json',report)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    local=sub.add_parser('local-preflight');local.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    connect=sub.add_parser('make-connection');connect.add_argument('--ssh-alias',default='vast');connect.add_argument('--output',type=Path,default=ROOT/'configs/storage/m71_connection.local.json')
    pre=sub.add_parser('remote-precheck');pre.add_argument('--ssh-alias',default='vast')
    upload=sub.add_parser('upload');upload.add_argument('--data-dir',type=Path,default=DEFAULT_DATA);upload.add_argument('--connection',type=Path,default=ROOT/'configs/storage/m71_connection.local.json')
    post=sub.add_parser('remote-postcheck');post.add_argument('--ssh-alias',default='vast')
    args=p.parse_args()
    try:
        if args.command=='local-preflight':return 0 if local_preflight(args.data_dir.resolve()) else 1
        if args.command=='make-connection':make_connection(args.ssh_alias,args.output);return 0
        if args.command=='remote-precheck':return 0 if remote_inventory(args.ssh_alias,False) else 1
        if args.command=='upload':verified_upload(args.connection,args.data_dir.resolve());return 0
        if args.command=='remote-postcheck':return 0 if remote_inventory(args.ssh_alias,True) else 1
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
    return 1

if __name__=='__main__':raise SystemExit(cli())
