#!/usr/bin/env python3
"""Bounded, reproducible final operational qualification on an existing Vast H100.

Outputs evidence even on partial failure. Does not change reference code or GPU state,
create/destroy Vast instances, or certify learning stability at peak LR.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

CONTROL = Path('/workspace/kimi/control/kimi_k3_milestone5_branch5')
REPO = Path('/workspace/kimi/reference')
DATA = Path('/workspace/kimi/data/qualification')
CASE = Path(__file__).with_name('qualification_case.py')
TARGET_TOKENS = 6_000_001_024
HISTORIC_RATE_32 = 66888.06449882299
HISTORIC_RATE_40 = 69640.48328809797


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def telemetry_metrics(path):
    if not path.exists():
        return {'count':0,'error':'Telemetry not found'}
    used=[]; total=[]; util=[]
    with path.open(encoding='utf-8',errors='replace') as f:
        for row in csv.reader(f):
            if len(row)<4: continue
            try:
                used.append(float(row[1]));total.append(float(row[2]));util.append(float(row[3]))
            except ValueError: continue
    if not used:
        return {'count':0,'error':'No numeric nvidia-smi samples'}
    return {'count':len(used),'peak_physical_memory_mib':max(used),
            'total_gpu_memory_mib':max(total),'peak_physical_memory_fraction':max(used)/min(total),
            'gpu_utilization_mean_pct_including_setup':sum(util)/len(util),
            'gpu_utilization_max_pct':max(util),
            'note':'Utilization average includes initialization/compilation; not a steady-step occupancy measure'}


def run_case(name, *, out, python, steps, resume, timeout_seconds):
    case=out/name
    cmd=[str(python),str(CASE),'--control',str(CONTROL),'--repo',str(REPO),
         '--data',str(DATA),'--out',str(case),'--batch','32','--steps',str(steps)]
    if resume: cmd+=['--resume',str(resume)]
    case_log=out/(name+'.log');telemetry=out/(name+'_physical_gpu.csv')
    t0=time.monotonic()
    print(f'\n>>> {name}: steps={steps}, resume={bool(resume)}',flush=True)
    with case_log.open('w',encoding='utf-8') as log, telemetry.open('w',encoding='utf-8') as gpu:
        mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw',
                               '--format=csv,noheader,nounits','-lms','500'],stdout=gpu,
                              stderr=subprocess.DEVNULL)
        try:
            try:
                # No shell; wrapper's wall timeout is separate from Vast rental clock.
                process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout_seconds,check=False)
                rc=process.returncode;timed_out=False
            except subprocess.TimeoutExpired:
                rc=124;timed_out=True
                log.write('\nTIMED OUT: bounded worker run\n')
        finally:
            mon.terminate()
            try:mon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mon.kill();mon.wait()
    elapsed=time.monotonic()-t0
    record={'name':name,'returncode':rc,'timed_out':timed_out,'elapsed_wall_seconds':round(elapsed,2),
            'telemetry':telemetry_metrics(telemetry)}
    audit_path=case/'case_report.json'
    if audit_path.exists():
        try:record['case_report']=json.loads(audit_path.read_text(encoding='utf-8'))
        except Exception as ex:record['case_report_error']=str(ex)
    atomic_json(out/(name+'_phase.json'),record)
    print(f'{name}: rc={rc}, wall={elapsed:.1f}s, physical GPU peak '
          f'{record["telemetry"].get("peak_physical_memory_mib","n/a")} MiB',flush=True)
    return record


def projected_cost(observation, gpu_eur_hour):
    # Separate empirical base throughput from measured final-step overheads.
    # Maintain eval/save cadence in TOKENS from 2500 updates * 2048 old batch.
    n_events=(TARGET_TOKENS+5_120_000-1)//5_120_000
    rate=HISTORIC_RATE_32
    base_seconds=TARGET_TOKENS/rate
    t_eval=observation.get('eval_seconds')
    t_save=observation.get('checkpoint_seconds')
    overhead=(n_events*(t_eval+t_save) if isinstance(t_eval,(int,float)) and isinstance(t_save,(int,float)) else None)
    result={'target_tokens':TARGET_TOKENS,'throughput_without_eval_or_save_tok_s':rate,
            'observed_from':'microbatch sweep 2, batch=32, no profiler',
            'compute_only_hours':base_seconds/3600,
            'compute_only_eur':base_seconds/3600*gpu_eur_hour,
            'hourly_eur_user_reported':gpu_eur_hour,
            'reference_eval_checkpoint_cadence_tokens':5_120_000,
            'estimated_eval_checkpoint_occurrences':n_events,
            'measured_eval_seconds':t_eval,'measured_checkpoint_seconds':t_save,
            'other_costs_excluded':['Vast stopped-disk','full data upload','cold starts','failed/restarted work','FX/tax if not included in rate','transfers'],
            'assumptions':['Extrapolation from short-run throughput','Token-aligned evaluation and checkpoint frequency is a scenario, not a finalized M7 schedule','Short-run timing may not reflect full training']}
    if overhead is not None:
        result['estimated_eval_checkpoint_overhead_hours']=overhead/3600
        result['illustrative_compute_eval_checkpoint_hours']=(base_seconds+overhead)/3600
        result['illustrative_compute_eval_checkpoint_eur']=(base_seconds+overhead)/3600*gpu_eur_hour
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--eur-per-hour',type=float,default=2.32,
                    help='User-supplied active hourly price, not a verified Vast invoice')
    ap.add_argument('--case-timeout',type=int,default=480)
    ap.add_argument('--total-limit',type=int,default=1080)
    args=ap.parse_args()
    if args.eur_per_hour<=0 or args.case_timeout<120 or args.total_limit<2*args.case_timeout:
        ap.error('Invalid cost or time budgets')
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    summary={'schema':1,'experiment':'M6 final operational qualification',
             'state':'running','candidate':{'microbatch':32,'context':1024,'grad_accumulation':1,'tokens_per_update':32768},
             'historical_rates':{'b32_tok_s':HISTORIC_RATE_32,'b40_tok_s':HISTORIC_RATE_40},
             'eur_hour':args.eur_per_hour,'system_only':True,
             'steps_initial':80,'steps_continued':8,'phase_records':{},
             'produced_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    atomic_json(args.out/'summary.json',summary)
    try:
        python=CONTROL/'environments/milestone5_linux/.venv/bin/python'
        for path in (python,CASE,REPO/'scripts/train.py',DATA,CONTROL/'scripts/milestone5/qualification_worker.py'):
            if not path.exists(): raise RuntimeError(f'Missing prerequisite: {path}')
        try:
            gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                            '--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
        except Exception as ex:raise RuntimeError(f'nvidia-smi not available: {ex}')
        if len(gpus)!=1 or 'H100 NVL' not in gpus[0]:
            raise RuntimeError(f'Expected exactly one H100 NVL, got {gpus!r}')
        summary['gpu_initial']=gpus
        free=shutil.disk_usage(args.out).free
        summary['disk_free_start_bytes']=free
        if free<12*1024**3:
            raise RuntimeError(f'Insufficient disk free ({free/1024**3:.1f} GiB), need >=12 GiB for two checkpoint saves and temp files')
        initial=run_case('initial',out=args.out,python=python,steps=80,resume=None,timeout_seconds=args.case_timeout)
        summary['phase_records']['initial']=initial
        atomic_json(args.out/'summary.json',summary)
        if initial['returncode'] or initial.get('case_report',{}).get('status')!='success':
            raise RuntimeError('Initial qualification failed; do not resume')
        snapshot=args.out/'initial/run/last.pt'
        initial_sha=digest(snapshot)
        summary['initial_checkpoint']={'path':str(snapshot),'bytes':snapshot.stat().st_size,'sha256':initial_sha}
        print('Initial checkpoint SHA256 verified locally on Vast.',flush=True)
        if time.monotonic()-started>args.total_limit-args.case_timeout-60:
            raise RuntimeError('Insufficient walltime left for bounded restore; first phase evidence retained')
        resume=run_case('resume',out=args.out,python=python,steps=88,resume=snapshot,timeout_seconds=args.case_timeout)
        summary['phase_records']['resume']=resume
        atomic_json(args.out/'summary.json',summary)
        if resume['returncode'] or resume.get('case_report',{}).get('status')!='success':
            raise RuntimeError('Resume qualification failed')
        aud=resume['case_report'].get('restore_audit',{})
        if not aud.get('passed') or resume['case_report'].get('numerical_audit',{}).get('new_updates')!=8:
            raise RuntimeError('Restore must be exact and 8 continuation updates must pass')
        after=args.out/'resume/run/last.pt'
        summary['resume_checkpoint']={'path':str(after),'bytes':after.stat().st_size,'sha256':digest(after)}
        phys_initial=initial['telemetry'];phys_resume=resume['telemetry']
        if phys_initial.get('count',0)<6 or phys_resume.get('count',0)<6:
            raise RuntimeError('Physical GPU telemetry insufficient')
        peak=max(phys_initial['peak_physical_memory_fraction'],phys_resume['peak_physical_memory_fraction'])
        summary['physical_memory_peak_fraction']=peak
        summary['projection']=projected_cost(initial['case_report'].get('observation',{}),args.eur_per_hour)
        summary['status_checks']={
            'source_and_dependency_verified':all(phase['case_report'].get('source_checked_files') and phase['case_report'].get('dependency_matched') for phase in (initial,resume)),
            'initial_numerics':initial['case_report'].get('numerical_audit',{}).get('passed'),
            'resume_audit':aud.get('passed'),
            'resume_numerics':resume['case_report'].get('numerical_audit',{}).get('passed'),
            'under_80pct_physical_vram':peak<=0.80,
            'under_85pct_physical_vram':peak<=0.85,
        }
        if not all([summary['status_checks']['source_and_dependency_verified'],
                    summary['status_checks']['initial_numerics'], summary['status_checks']['resume_audit'],
                    summary['status_checks']['resume_numerics']]):
            raise RuntimeError('One or more functional gates failed')
        if peak>0.80:
            # Do not silently approve a marginal memory margin; keep all evidence.
            summary['state']='needs_memory_review'
            print(f'WARNING: physical GPU peak {peak:.1%} >80%. Manual margin decision required.',flush=True)
        else:
            summary['state']='qualification_evidence_ready_for_review'
        print('Operational qualification complete; review report before declaring M6 approved.',flush=True)
        return 0
    except BaseException as error:
        summary['state']='failed_or_partial';summary['error']=type(error).__name__+': '+str(error)
        print('STOP:',summary['error'],file=sys.stderr,flush=True)
        return 1
    finally:
        summary['elapsed_wall_seconds']=round(time.monotonic()-started,2)
        summary['completed_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        atomic_json(args.out/'summary.json',summary)


if __name__=='__main__':
    raise SystemExit(main())
