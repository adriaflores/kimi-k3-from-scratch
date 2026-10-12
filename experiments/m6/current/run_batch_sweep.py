#!/usr/bin/env python3
"""Bounded, sequential microbatch sweep; no Vast API access, no instance controls."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

DEFAULT_CONTROL=Path('/workspace/kimi/control/kimi_k3_milestone5_branch5')
DEFAULT_REPO=Path('/workspace/kimi/reference')
DEFAULT_DATA=Path('/workspace/kimi/data/qualification')


def atomic_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def package(out:Path):
    zpath=out/'batch_sweep_results.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path==zpath:continue
            rel=path.relative_to(out)
            if rel.name in ('last.pt',) or any(p.startswith('profiles') for p in rel.parts):continue
            if rel.name=='run.json':continue  # may contain full optimizer layout, no value here
            z.write(path,arcname=rel.as_posix())
    return zpath


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control',type=Path,default=DEFAULT_CONTROL)
    ap.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    ap.add_argument('--data',type=Path,default=DEFAULT_DATA)
    ap.add_argument('--out',type=Path,default=Path('/workspace/kimi/runs/m6_batch_sweep_01'))
    ap.add_argument('--batches',nargs='+',type=int,default=[2,4,8])
    ap.add_argument('--steps',type=int,default=320)
    ap.add_argument('--burn',type=int,default=200)
    ap.add_argument('--per-case-wall',type=int,default=420)
    ap.add_argument('--total-wall',type=int,default=1200)
    ap.add_argument('--effective-eur-per-hour',type=float,default=None,
                    help='Optional, current ALL-IN active GPU hourly cost in EUR. Otherwise report hours only.')
    args=ap.parse_args()
    if args.batches != [2,4,8] and args.batches != [16]:
        ap.error('This package authorizes batches 2 4 8, or a later isolated 16 rerun.')
    if args.effective_eur_per_hour is not None and not 0<args.effective_eur_per_hour<10:
        ap.error('Invalid hourly price')
    for path in (args.repo/'scripts/train.py', args.control/'scripts/milestone5/environment_common.py'):
        if not path.is_file():ap.error(f'Missing prerequisite: {path}')
    if args.out.exists():ap.error('Output directory exists. Choose a fresh --out; do not overwrite historic data.')
    args.out.mkdir(parents=True)
    started=time.monotonic()
    rows=[]
    abort_reason=None
    for batch in args.batches:
        if time.monotonic()-started > args.total_wall-args.per_case_wall:
            abort_reason='Insufficient time budget for next case';break
        name=f'b{batch}_t1024'
        case_dir=args.out/name
        command=[sys.executable,str(Path(__file__).with_name('bench_one.py')),
                 '--control',str(args.control),'--repo',str(args.repo),'--data',str(args.data),
                 '--batch',str(batch),'--steps',str(args.steps),'--burn',str(args.burn),'--out',str(case_dir)]
        print(f'\n== RUNNING {name} ==',flush=True)
        timeout=min(args.per_case_wall, max(1,args.total_wall-(time.monotonic()-started)))
        started_case=time.monotonic()
        with (args.out/f'{name}.log').open('w',encoding='utf-8') as log:
            try:
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                rc=p.returncode
            except subprocess.TimeoutExpired:
                rc=124
                log.write(f'\nTIMEOUT: case exceeded {timeout:.1f} seconds.\n')
        duration=time.monotonic()-started_case
        summary_file=case_dir/'summary.json'
        if rc==0 and summary_file.is_file():
            report=json.loads(summary_file.read_text(encoding='utf-8'))
            print(f'  {report["throughput_tokens_per_second"]:.0f} tok/s '
                  f'({report["speedup_vs_historical"]:.2f}x); '
                  f'{report["compute_only_6b_hours"]:.1f} compute h',flush=True)
            row=dict(batch=batch,status='success',wall_seconds=round(duration,2),
                     tok_s=round(report['throughput_tokens_per_second'],1),
                     speedup=round(report['speedup_vs_historical'],3),
                     compute_hours_6b=round(report['compute_only_6b_hours'],2),
                     peak_reserved_gib=round(report['peak_cuda_reserved_bytes']/1024**3,2),
                     windows=report['post_burn_windows'],
                     stability_ratio=round(report['window_ratio_max_min'],3))
            if args.effective_eur_per_hour:
                row['compute_cost_eur_6b']=round(report['compute_only_6b_hours']*args.effective_eur_per_hour,2)
        else:
            print(f'  FAILED (returncode {rc}). Review {name}.log; not continuing.',flush=True)
            row=dict(batch=batch,status=f'failed_{rc}',wall_seconds=round(duration,2))
            abort_reason=f'{name} failed (rc {rc})'
        rows.append(row)
        atomic_json(args.out/'progress.json',{'cases':rows,'abort_reason':abort_reason,
                                             'elapsed_wall_s':round(time.monotonic()-started,2)})
        package(args.out)
        if abort_reason:break
    csv_path=args.out/'comparison.csv'
    keys=['batch','status','wall_seconds','tok_s','speedup','compute_hours_6b','compute_cost_eur_6b',
          'peak_reserved_gib','windows','stability_ratio']
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    atomic_json(args.out/'summary.json',{
        'description':'Systems benchmark, unchanged model, no profiler/checkpoints; batches differ from M4 recipe',
        'baseline_qualified_t1024_tok_s':12023.30,
        'baseline_qualified_total_6b_hours':144.35,
        'baseline_qualified_total_6b_cost_eur':373.97,
        'actual_eur_hour_assumption':args.effective_eur_per_hour,
        'target_tokens':6_000_001_024,
        'cases':rows,'abort_reason':abort_reason,
        'elapsed_wall_seconds':round(time.monotonic()-started,2),
        'not_production_training':True,
        'limitations':'Cost column, when present, is compute-only; does not include fixed evaluation/checkpoint/storage/transfer/tax unless all-in hourly rate accounts for taxes.'})
    z=package(args.out)
    print(f'\nZIP: {z}\nWall: {time.monotonic()-started:.1f} sec; abort: {abort_reason}',flush=True)
    return 1 if abort_reason else 0

if __name__=='__main__':
    sys.exit(main())
