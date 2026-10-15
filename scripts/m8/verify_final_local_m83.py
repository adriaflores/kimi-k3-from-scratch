#!/usr/bin/env python3
"""Independent off-Vast final checkpoint audit before manual Vast shutdown."""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RUN_ID='m8_full_6b_v1';STEP=183106

def require(condition, reason):
    if not condition:raise ValueError(reason)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(backups, evidence):
    root=Path(backups).expanduser().resolve()/RUN_ID
    name=f'snapshot_step_{STEP}.pt'
    checkpoint=root/name
    sidecar=root/(name+'.verified.json')
    require(checkpoint.is_file() and sidecar.is_file(),'Final local checkpoint or verification sidecar missing')
    record=json.loads(sidecar.read_text(encoding='utf-8'))
    require(record.get('verified') is True and record.get('step')==STEP,'Local backup receipt not verified')
    require(record.get('bytes')==checkpoint.stat().st_size,'Final backup size mismatch')
    h=sha(checkpoint)
    require(h==record.get('sha256'),'Final backup SHA-256 mismatch')
    evidence=Path(evidence).resolve(strict=True)
    with zipfile.ZipFile(evidence) as z:
        names=set(z.namelist())
        required={'final_audit.json','final_snapshot_receipt.json','offsite_ack.json',
                  'metrics.jsonl','run.json','runner_state.json','probe_receipt.json','recovery_gate.json'}
        require(required.issubset(names),'Incomplete remote final evidence ZIP')
        require(not any(n.endswith('.pt') or n.endswith('.safetensors') for n in names),'Unexpected weight file in evidence ZIP')
        audit=json.loads(z.read('final_audit.json'))
        remote=json.loads(z.read('final_snapshot_receipt.json'))
        ack=json.loads(z.read('offsite_ack.json'))
        state=json.loads(z.read('runner_state.json'))
        require(audit.get('status')=='PASS' and audit.get('final_step')==STEP,'Remote final audit did not pass')
        require(state.get('state')=='COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP','Remote training not complete')
        require(audit.get('checkpoint_sha256')==remote.get('sha256')==ack.get('sha256')==h,
                'Local file, remote receipt, offsite ACK and final audit differ')
        require(remote.get('step')==ack.get('step')==STEP and ack.get('verified_external') is True,
                'Wrong final backup step/verification')
        require(remote.get('bytes')==checkpoint.stat().st_size,'Remote snapshot differs in size')
        metrics=z.read('metrics.jsonl').splitlines()
        require(any(json.loads(line).get('event')=='checkpoint' and json.loads(line).get('step')==STEP
                    for line in metrics),'Final trainer checkpoint event missing')
    return {'status':'PASS','decision':'SAFE_TO_CONSIDER_MANUAL_SHUTDOWN',
            'step':STEP,'sha256':h,'bytes':checkpoint.stat().st_size,
            'local_file':str(checkpoint),'evidence_zip_sha256':sha(evidence),
            'checked_at_utc':datetime.now(timezone.utc).isoformat()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup-directory',type=Path,required=True)
    p.add_argument('--evidence-zip',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        result=verify(args.backup_directory,args.evidence_zip)
        path=args.receipt
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.partial')
        with tmp.open('x',encoding='utf-8') as out:
            json.dump(result,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(tmp,path)
        print(json.dumps(result,indent=2));return 0
    except Exception as exc:
        print(f'M8.3 FINAL LOCAL NO-GO: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':sys.exit(main())
