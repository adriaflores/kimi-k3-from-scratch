#!/usr/bin/env python3
"""Sweep v2 on the existing verified H100 source, with time/memory/plateau gates.

Plan: b8 control, b16, b24, b32; b40 only if b32 improves >=5% over
b24 and projected memory <=90% of GPU memory. No model/optimizer changes.
All cases remain systems-only: no scheduled evaluations or checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

CONTROL = Path('/workspace/kimi/control/kimi_k3_milestone5_branch5')
REPO = Path('/workspace/kimi/reference')
DATA = Path('/workspace/kimi/data/qualification')
HISTORIC_B8_RATE = 39064.30583842132
HISTORIC_B4_RESERVED_GIB = 11.79
TARGET_TOKENS = 6_000_001_024
CASES = (8, 16, 24, 32, 40)
CSV_COLUMNS = ['batch','status','wall_seconds','tok_s','speedup_vs_2048_baseline',
               'speedup_vs_v1_b8','compute_hours_6b','compute_cost_eur_6b',
               'peak_reserved_gib','peak_allocated_gib','gpu_total_gib',
               'windows','stability_ratio']


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def package(out: Path) -> Path:
    path = out/'batch_sweep_results.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if not f.is_file() or f == path: continue
            rel = f.relative_to(out)
            if f.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors','.zip'):continue
            if any(x.startswith('profiles') for x in rel.parts):continue
            if f.stat().st_size > 50_000_000: continue
            z.write(f, rel.as_posix())
    return path


def predict_reserved_gib(history: list[tuple[int,float]], batch: int) -> float:
    """Conservative linear extrapolation; *not* a proof batch will fit."""
    if len(history) < 2: return float('inf')
    (b0, m0), (b1, m1) = history[-2:]
    slope = max(0.6, (m1 - m0) / (b1 - b0))
    return m1 + (batch - b1) * slope


def should_run_optional(rate32:float, rate24:float, predicted:float, gpu_total:float) -> tuple[bool,str]:
    if rate32 < 1.05 * rate24:
        return False, 'Skipping b40: b32 gained less than 5% versus b24'
    if predicted > 0.90*gpu_total or (gpu_total - predicted) < 7.0:
        return False, 'Skipping b40: insufficient predicted VRAM headroom'
    return True, 'b40 authorized by throughput and memory trend'


def is_oom(log_file:Path) -> bool:
    try: tail = log_file.read_text(encoding='utf-8', errors='replace')[-20000:].lower()
    except OSError: return False
    return any(token in tail for token in ('outofmemoryerror','cuda out of memory','out of memory',
                                           'cudnn_status_alloc_failed','cublas_status_alloc_failed'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--total-wall', type=float, default=1080.0)
    parser.add_argument('--per-case-wall', type=float, default=390.0)
    parser.add_argument('--effective-eur-per-hour', type=float, default=None,
                        help='Optional current all-in hourly price in euros (compute-only projection)')
    args = parser.parse_args()
    if args.total_wall > 1150 or args.total_wall < 240 or args.per_case_wall > 450 or args.per_case_wall < 120:
        parser.error('Unsafe wall limits: total <=1150 and per-case <=450 s')
    if args.effective_eur_per_hour is not None and not (0 < args.effective_eur_per_hour < 10):
        parser.error('Expected a realistic, positive EUR/hour price')
    for p in (args.repo/'scripts/train.py', args.control/'scripts/milestone5/environment_common.py'):
        if not p.is_file():parser.error(f'Missing prerequisite: {p}')
    if args.out.exists():parser.error(f'Output directory exists: {args.out}')
    args.out.mkdir(parents=True)
    started=time.monotonic()
    history=[(4, HISTORIC_B4_RESERVED_GIB)]
    rows=[]
    stop_reason=None
    outcome='completed'

    def save_progress():
        elapsed=time.monotonic()-started
        atomic_json(args.out/'progress.json', dict(cases=rows,stop_reason=stop_reason,
                     outcome=outcome,elapsed_wall_seconds=round(elapsed,2),
                     memory_history=history))
        package(args.out)

    for batch in CASES:
        remaining = args.total_wall - (time.monotonic()-started)
        if remaining < min(args.per_case_wall, 300.0):
            stop_reason=f'Insufficient wall budget for b{batch} ({remaining:.1f}s left)'
            outcome='partial_time_limit'
            print('STOP:',stop_reason,flush=True)
            break
        if batch > 8:
            predicted=predict_reserved_gib(history,batch)
            lastrow=rows[-1]
            total=float(lastrow['gpu_total_gib'])
            if predicted > 0.90*total or total-predicted < 7.0:
                stop_reason=f'Memory guard before b{batch}: predicted {predicted:.1f} GiB, total {total:.1f} GiB'
                outcome='completed_memory_limit'
                print('STOP:',stop_reason,flush=True)
                break
            if batch == 40:
                ok,reason=should_run_optional(rows[-1]['tok_s'], rows[-2]['tok_s'], predicted, total)
                print(reason,flush=True)
                if not ok:
                    stop_reason=reason
                    outcome='completed_plateau_or_memory_guard'
                    break
        case_name=f'b{batch}_t1024'
        out=args.out/case_name
        log=args.out/f'{case_name}.log'
        command=[sys.executable,str(Path(__file__).with_name('bench_one.py')),
                 '--control',str(args.control),'--repo',str(args.repo),
                 '--data',str(args.data),'--batch',str(batch),
                 '--steps','320','--burn','200','--out',str(out)]
        deadline=min(args.per_case_wall, max(1.0, args.total_wall-(time.monotonic()-started)))
        print(f'\n== RUNNING {case_name} == (timeout {deadline:.0f}s)',flush=True)
        t0=time.monotonic()
        with log.open('w',encoding='utf-8') as stream:
            try:
                process=subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=deadline)
                rc=process.returncode
            except subprocess.TimeoutExpired:
                rc=124
                stream.write(f'\nCASE TIMEOUT after {deadline:.1f}s\n')
        duration=time.monotonic()-t0
        report_file=out/'summary.json'
        if rc==0 and report_file.is_file():
            report=json.loads(report_file.read_text(encoding='utf-8'))
            if report['status']!='success' or report['batch']!=batch:raise RuntimeError('Case result mismatch')
            rate=float(report['throughput_tokens_per_second'])
            reserved=float(report['peak_cuda_reserved_bytes'])/1024**3
            allocated=float(report['peak_cuda_allocated_bytes'])/1024**3
            total=float(report['gpu']['total_memory_bytes'])/1024**3
            windows=int(report['post_burn_windows'])
            variability=float(report['window_ratio_max_min'])
            row=dict(batch=batch,status='success',wall_seconds=round(duration,2),
                     tok_s=round(rate,2),speedup_vs_2048_baseline=round(rate/12023.3,3),
                     speedup_vs_v1_b8=round(rate/HISTORIC_B8_RATE,3),
                     compute_hours_6b=round(TARGET_TOKENS/rate/3600,2),
                     peak_reserved_gib=round(reserved,3),peak_allocated_gib=round(allocated,3),
                     gpu_total_gib=round(total,3),windows=windows,stability_ratio=round(variability,4))
            if args.effective_eur_per_hour is not None:
                row['compute_cost_eur_6b']=round(row['compute_hours_6b']*args.effective_eur_per_hour,2)
            rows.append(row)
            history.append((batch,reserved))
            print(f'  {rate:,.0f} tok/s; {rate/HISTORIC_B8_RATE:.2f}× vs v1 b8; '
                  f'{row["compute_hours_6b"]:.1f} compute h; {reserved:.1f}/{total:.1f} GiB reserved',flush=True)
            if batch==8 and rate < 0.85 * HISTORIC_B8_RATE:
                stop_reason=f'Control b8 only {rate:.0f} tok/s vs previous {HISTORIC_B8_RATE:.0f}: environment changed'
                outcome='failed_control'
            elif variability > 1.15:
                stop_reason=f'Unstable timing on b{batch}: max/min {variability:.3f}'
                outcome='partial_unstable'
            elif batch>=16 and (rate/rows[-2]['tok_s']) < 1.02:
                stop_reason=f'Performance plateau: b{batch} improved <2% over b{rows[-2]["batch"]}'
                outcome='completed_plateau'
        else:
            oom=is_oom(log)
            row=dict(batch=batch,status='oom' if oom else f'failed_{rc}',wall_seconds=round(duration,2),
                     error='See '+log.name)
            rows.append(row)
            stop_reason=f'b{batch} OOM: preceding successful configs preserved' if oom else f'b{batch} failed (rc {rc})'
            outcome='completed_hardware_limit' if oom and len(rows)>1 else 'failed_case'
            print('STOP:',stop_reason,flush=True)
        save_progress()
        if stop_reason:break

    keys=CSV_COLUMNS
    with (args.out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary=dict(description='M6 systems-only microbatch sweep v2; model/optimizer unchanged',
        previous_sweep=dict(batch8_tok_s=HISTORIC_B8_RATE,batch8_reserved_gib=19.58,
                            batch4_reserved_gib=HISTORIC_B4_RESERVED_GIB),
        cases=rows,stop_reason=stop_reason,outcome=outcome,
        target_tokens=TARGET_TOKENS,effective_eur_hour=args.effective_eur_per_hour,
        total_wall_limit_seconds=args.total_wall,case_wall_limit_seconds=args.per_case_wall,
        elapsed_wall_seconds=round(time.monotonic()-started,2),
        systems_only=True,not_approved_training_recipe=True,
        limitations='Compute-only extrapolation; schedule, eval, checkpoint, storage, transfer and real price remain unverified. Batch >2 changes effective batch.')
    atomic_json(args.out/'summary.json',summary)
    save_progress()
    z=package(args.out)
    print(f'\nZIP: {z}\nOutcome: {outcome}; stop_reason: {stop_reason}; '
          f'wall: {time.monotonic()-started:.1f}s',flush=True)
    return 1 if outcome.startswith('failed') else 0

if __name__=='__main__':
    sys.exit(main())
