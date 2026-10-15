#!/usr/bin/env python3
"""Off-Vast, pull-only snapshot receiver for Windows Git Bash/Linux, via ssh/scp.

Keep this process alive on an independent PC while the Vast training runs.
An ACK means bytes have been hashed and committed locally; never send ACK first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN_ID='m8_full_6b_v1'
REMOTE_PY='/workspace/kimi/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python'
REMOTE_SCRIPT='/workspace/kimi/scripts/m8/m82.py'
REMOTE_ART='/workspace/kimi/artifacts/m8/m82/'+RUN_ID+'/snapshots'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path,data):
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(data,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def exec_check(args,**kwargs):
    return subprocess.run(args,check=True,**kwargs)


def request(host, runner=exec_check):
    proc=runner(['ssh','-o','BatchMode=yes',host,REMOTE_PY,REMOTE_SCRIPT,'request'],capture_output=True,text=True)
    return json.loads(proc.stdout)


def receive(host,local_dir,record,runner=exec_check):
    if record is None:return 'NO_BACKUP_DUE'
    if not isinstance(record,dict): raise ValueError('Invalid remote request')
    step=record.get('step');checksum=record.get('sha256');size=record.get('bytes')
    if type(step) is not int or not (4096<=step<=183106) or (step%4096!=0 and step!=183106):
        raise ValueError('Invalid step from remote')
    if not isinstance(checksum,str) or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Invalid checksum from remote')
    if type(size) is not int or size<2_000_000_000:
        raise ValueError('Suspicious backup size')
    name=f'snapshot_step_{step}.pt'
    if record.get('file')!=name:raise ValueError('Untrusted remote filename')
    local_dir=Path(local_dir).expanduser().resolve()/RUN_ID
    local_dir.mkdir(parents=True,exist_ok=True)
    completed=local_dir/name
    partial=local_dir/(name+'.partial')
    if completed.exists():
        if completed.stat().st_size!=size or sha(completed)!=checksum:
            raise ValueError('Existing local backup differs; do not overwrite')
    else:
        if partial.exists():
            # Never overwrite an earlier incomplete operation silently.
            raise FileExistsError(f'Remove/investigate incomplete download: {partial}')
        try:
            runner(['scp','-o','BatchMode=yes',f'{host}:{REMOTE_ART}/{name}',str(partial)])
            if partial.stat().st_size!=size or sha(partial)!=checksum:
                raise ValueError('Downloaded size/hash mismatch; no ACK')
            with partial.open('r+b') as f:os.fsync(f.fileno())
            os.replace(partial,completed)
        except BaseException:
            # Keep failed partial for forensics; no silent overwrite on retry.
            raise
    receipt=local_dir/(name+'.verified.json')
    if not receipt.exists():
        atomic_json(receipt,{'step':step,'sha256':checksum,'bytes':size,'verified':True,
                             'path':str(completed),'remote_host':host})
    runner(['ssh','-o','BatchMode=yes',host,REMOTE_PY,REMOTE_SCRIPT,
            'ack','--step',str(step),'--sha256',checksum],capture_output=True,text=True)
    return f'VERIFIED_AND_ACKNOWLEDGED step={step} sha256={checksum}'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='vast')
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--interval',type=int,default=60)
    p.add_argument('--once',action='store_true')
    args=p.parse_args(argv)
    if args.interval<10: p.error('Minimum polling interval is 10 seconds')
    while True:
        try:
            print(receive(args.host,args.directory,request(args.host)),flush=True)
        except (OSError,ValueError,subprocess.CalledProcessError) as exc:
            print(f'BACKUP NOT VERIFIED: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
            if args.once:return 2
        if args.once:return 0
        time.sleep(args.interval)

if __name__=='__main__':sys.exit(main())
