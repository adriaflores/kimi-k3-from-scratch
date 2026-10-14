#!/usr/bin/env python3
"""Short, separate paid-instance watchdog for a *restarted* Vast H100 qualification.

Does not reuse/reset the historical two-hour qualification budget. Limits this
follow-up session only. Keep the historical billing and receipts unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
from datetime import datetime,timezone

API='https://console.vast.ai/api/v0/instances/{id}/'


def api(instance_id,method='GET',body=None):
    key=os.environ.get('VAST_API_KEY')
    if not key:raise RuntimeError('Set VAST_API_KEY locally; never paste it into chat')
    req=urllib.request.Request(API.format(id=instance_id),
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method=method)
    with urllib.request.urlopen(req,timeout=12) as res:return json.load(res)


def identity(id):
    o=api(id)
    obj=o.get('instances',o.get('instance'))
    if not isinstance(obj,dict):raise RuntimeError('Unexpected Vast instance response')
    if obj.get('id')!=id or obj.get('label')!='kimi-k3-m5-qualification' or 'H100' not in obj.get('gpu_name',''):
        raise RuntimeError('Instance ID/label/GPU mismatch; refusing stop action')
    return obj


def record(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance-id',required=True,type=int)
    p.add_argument('--max-minutes',type=float,default=20)
    p.add_argument('--output',type=Path,default=Path('profile_followup_watch.json'))
    args=p.parse_args()
    if not 5<=args.max_minutes<=30:p.error('max-minutes must be between 5 and 30')
    obj=identity(args.instance_id)
    if obj.get('cur_state')!='running' or obj.get('actual_status') not in ('running','loading'):
        p.error('Start/verify the instance first; it is not currently running')
    started=time.monotonic()
    deadline=started+args.max_minutes*60
    print(f'Followup watchdog active for H100 instance {args.instance_id}; stop by {args.max_minutes:g} minutes.',flush=True)
    while time.monotonic()<deadline:
        time.sleep(min(12,max(0,deadline-time.monotonic())))
        try:
            current=identity(args.instance_id)
            if current.get('cur_state')=='stopped' or current.get('actual_status')=='stopped':
                record(args.output,{'status':'already_stopped','instance_id':args.instance_id})
                print('Already stopped in Vast. No stop request required.',flush=True)
                return 0
        except Exception as exc:
            print('Watch verification retry needed:',type(exc).__name__,flush=True)
    for i in range(3):
        try:
            identity(args.instance_id)
            ans=api(args.instance_id,'PUT',{'state':'stopped'})
            if ans.get('success') is not True:raise RuntimeError('No Vast stop acknowledgement')
            end={'status':'stop_acknowledged','instance_id':args.instance_id,'utc':datetime.now(timezone.utc).isoformat(),
                 'provider_stop_acknowledged':True,'storage_billing_continues':True}
            record(args.output,end)
            print('Stop acknowledged by Vast; confirm Stopped in web UI.',flush=True)
            return 0
        except Exception as exc:
            print(f'STOP ATTEMPT {i+1} FAILED ({type(exc).__name__}). Stop manually in Vast if needed!',flush=True)
            time.sleep(3)
    record(args.output,{'status':'manual_stop_required','instance_id':args.instance_id})
    return 1

if __name__=='__main__':raise SystemExit(main())
