"""M9.4 qualitative evaluation runner (run at repository root using uv)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.m9.m94_core import (
    CHECKPOINT, CONTRACT, PROTOCOL, OUT, now, atomic_json,
    verify_preflight, load_session, generate_case, run_fixed,
)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight-only', action='store_true',help='No GPU or PyTorch checkpoint load')
    modes.add_argument('--smoke', action='store_true',help='One 8-token continuation; NOT scientific evidence')
    modes.add_argument('--fixed', action='store_true',help='Twelve frozen continuations (6 greedy + 6 sample)')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--contract',type=Path,default=CONTRACT)
    p.add_argument('--protocol',type=Path,default=PROTOCOL)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--resume',action='store_true',help='Only for existing frozen fixed run')
    p.add_argument('--local-files-only',action='store_true',help='Use cached pinned tokenizer; do not download')
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.resume and not args.fixed:
        raise ValueError('--resume requires --fixed')
    started=time.perf_counter()
    status='FAIL'
    report_path = args.output / ('preflight/m94_preflight.json' if args.preflight_only else 'smoke/m94_smoke.json' if args.smoke else 'fixed/m94_status.json')
    report={'schema_version':'m94_qualitative_1','started_at_utc':now(),
            'mode':'PREFLIGHT' if args.preflight_only else 'SMOKE_NOT_BENCHMARK' if args.smoke else 'FROZEN_FIXED_12',
            'status':'FAIL','device_requested':args.device,
            'checkpoint':str(args.checkpoint),'fused_parity':'NOT_ESTABLISHED'}
    try:
        _, protocol=verify_preflight(args.checkpoint,args.contract,args.protocol)
        report['checks']={'checkpoint_size_m91':'PASS','contract_sha256':'PASS','protocol_sha256':'PASS',
                          'fixed_protocol_valid':'PASS'}
        if args.preflight_only:
            status='PASS_PREFLIGHT_ONLY'
            report['cases']=len(protocol['cases'])
        else:
            session=load_session(device=args.device,local_files_only=args.local_files_only,
                                 checkpoint=args.checkpoint,contract_path=args.contract,
                                 protocol_path=args.protocol)
            if args.fixed:
                # Fixed runs require default frozen contract/protocol paths for identity.
                if Path(args.contract).resolve()!=CONTRACT.resolve() or Path(args.protocol).resolve()!=PROTOCOL.resolve():
                    raise ValueError('Fixed evidence requires canonical frozen contract and protocol paths')
                summary=run_fixed(session,out=args.output/'fixed',resume=args.resume)
                report['run_identity_sha256']=summary['run_identity_sha256']
                report['case_count']=summary['case_count']
                status=summary['status']
            else:
                g=generate_case(session, prompt='Once upon a time, in a small village',
                    mode='greedy',max_new_tokens=8)
                report['smoke'] = g
                status='PASS_SMOKE_NOT_BENCHMARK'
                print('SMOKE CONTINUATION:',repr(g['continuation_text']), flush=True)
        report['device']=str(session['device']) if not args.preflight_only else 'not loaded'
    except Exception as e:
        status='FAIL'
        report['error']=f'{type(e).__name__}: {e}'
        report['traceback']=traceback.format_exc(limit=10)
        print('M9.4 FAILED:',report['error'],flush=True)
    finally:
        report['status']=status
        report['seconds_total']=round(time.perf_counter()-started,3)
        report['finished_at_utc']=now()
        atomic_json(report_path,report)
        print('M9.4 STATUS:',status,flush=True)
        print('REPORT:',report_path,flush=True)
    return 0 if status.startswith('PASS_') else 1


if __name__=='__main__':
    sys.exit(main())
