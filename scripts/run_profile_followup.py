#!/usr/bin/env python3
"""Isolated post-qualification H100 profile + optional restore check; no model edits.

Run with the pinned M5 Linux environment on an already active, paid Vast instance.
Does NOT create, stop, or destroy instances. Always use a separate external watchdog.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

BASE = Path('/workspace/kimi/control/kimi_k3_milestone5_branch5')
OPS = BASE / 'scripts/milestone5'
sys.path.insert(0, str(OPS))
from branch4_remote_ops import snapshot_run, restore_snapshot


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def phase(name, cmd, root, timeout, deadline):
    log = root / (name + '.log')
    telem = root / (name + '_gpu.csv')
    terr = root / (name + '_gpu.stderr.txt')
    start = time.monotonic()
    max_end = min(start + timeout, deadline - 30)
    if max_end <= start:
        raise TimeoutError('Insufficient diagnostic wall-clock budget')
    print('Starting', name, 'max seconds', int(max_end - start), flush=True)
    with log.open('wb') as out, telem.open('wb') as tel, terr.open('wb') as err:
        sampler = subprocess.Popen([
            'nvidia-smi', '-i', '0',
            '--query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits', '-lms', '250'
        ], stdout=tel, stderr=err)
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            while p.poll() is None:
                if time.monotonic() >= max_end:
                    timed_out = True
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=5)
                    break
                time.sleep(.2)
        finally:
            if sampler.poll() is None:
                sampler.terminate()
            try: sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait(timeout=5)
    rows = []
    with telem.open(newline='') as f:
        for line in csv.reader(f):
            try:
                total, used, util = float(line[4]), float(line[5]), float(line[6])
                if 0 <= used <= total: rows.append((total,used,util))
            except (ValueError,IndexError):
                continue
    result = {'name': name, 'returncode': p.returncode, 'timed_out': timed_out,
              'wall_seconds': time.monotonic()-start,
              'gpu_samples': len(rows),
              'gpu_peak_used_mib': max((r[1] for r in rows), default=None),
              'gpu_utilization_max_pct': max((r[2] for r in rows), default=None),
              'gpu_utilization_mean_pct': sum(r[2] for r in rows)/len(rows) if rows else None,
              'telemetry_returncode': sampler.returncode,
              'log': log.name, 'telemetry': telem.name}
    print(name, result, flush=True)
    write_json(root / (name + '_phase.json'), result)
    return result


def package(root):
    output = root / 'profile_followup_results.zip'
    # Explicit allowlist by directory; no checkpoint or saved optimizer tensors in report.
    items = []
    for f in root.rglob('*'):
        if not f.is_file() or f == output or 'snapshots' in f.parts or 'runs' in f.parts:
            continue
        if f.suffix in ('.log', '.json', '.txt', '.csv', '.gz'):
            items.append(f)
    # Include profiler artifacts stored inside profile run but never checkpoints.
    for f in (root/'runs'/'profile'/'profiles').rglob('*') if (root/'runs'/'profile'/'profiles').exists() else []:
        if f.is_file() and (f.suffix in ('.gz', '.txt', '.json')): items.append(f)
    with zipfile.ZipFile(output, 'x') as z:
        for f in sorted(set(items)):
            z.write(f, f.relative_to(root), compress_type=zipfile.ZIP_STORED if f.suffix == '.gz' else zipfile.ZIP_DEFLATED)
    return {'path': str(output), 'bytes': output.stat().st_size, 'sha256':sha(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=Path('/workspace/kimi/runs/qualification_01/runs/t1024'))
    p.add_argument('--output-dir', type=Path, default=Path('/workspace/kimi/runs/qualification_profile_followup_01'))
    p.add_argument('--max-wall-seconds', type=int, default=900)
    p.add_argument('--with-resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not 120 <= args.max_wall_seconds <= 1800:
        p.error('max-wall-seconds must be in [120,1800]')
    python = BASE/'environments/milestone5_linux/.venv/bin/python'
    if Path(sys.executable).resolve() != python.resolve() and not args.dry_run:
        p.error('Use the M5 pinned .venv/bin/python')
    src = args.source_run.resolve()
    out = args.output_dir.resolve()
    if str(src) != '/workspace/kimi/runs/qualification_01/runs/t1024':
        p.error('Expected the preserved immutable-source t1024 run')
    if str(out) == str(src) or src in out.parents or out in src.parents:
        p.error('Output directory must not overlap the source run')
    for f in ('last.pt','run.json','metrics.jsonl'):
        if not (src/f).is_file():
            p.error(f'Source is missing {f}; do not restart original training')
    state = json.loads((src/'run.json').read_text())
    print('source run:', src, 'run_id:', state.get('run_id'))
    if args.dry_run:
        print('DRY RUN: no directory created, no GPU run, no instance management')
        return 0
    if out.exists():
        p.error('Output already exists; choose another name, never overwrite results')
    out.mkdir(parents=True)
    report = {'status':'running', 'original_source_run':str(src),
              'source_run_id':state.get('run_id'),
              'source_checkpoint_sha256': sha(src/'last.pt'), 'phases': [],
              'start_utc':datetime.now(timezone.utc).isoformat(), 'max_wall_seconds':args.max_wall_seconds}
    write_json(out/'followup.json', report)
    start = time.monotonic()
    deadline = start + args.max_wall_seconds
    try:
        snap = snapshot_run(src, out/'snapshots')
        report['snapshot'] = snap['snapshot']['snapshot_id']
        obs = json.loads((Path('/workspace/kimi/runs/qualification_01/observations/t1024.json')).read_text())
        old_step = int(obs['final_state']['step'])
        if old_step <= 0: raise ValueError('Original checkpoint step is invalid')
        worker = OPS/'qualification_worker.py'
        cfg = BASE/'configs/training/milestone5_branch5/t1024.yaml'
        base = [str(python), '-u', str(worker), '--repo-root', '/workspace/kimi/reference',
                '--target', 'h100', '--data-dir', '/workspace/kimi/data/qualification',
                '--training-config', str(cfg)]
        for mode, new in [('profile', 5), *([('resume', 8)] if args.with_resume else [])]:
            run = out/'runs'/mode
            restore_snapshot(Path(snap['snapshot_dir']), run)
            command = base + ['--mode', mode, '--run-dir', str(run), '--resume',str(run/'last.pt'),
                              '--max-steps', str(old_step+new),'--output',str(out/(mode+'.json'))]
            result = phase(mode, command, out, 600 if mode == 'profile' else 400, deadline)
            report['phases'].append(result)
            if result['returncode'] != 0:
                raise RuntimeError(f'{mode} failed; examine {result["log"]}')
            observation = json.loads((out/(mode+'.json')).read_text())
            if not observation.get('numerics_passed') or observation.get('new_updates') != new:
                raise RuntimeError(f'{mode} numerical/update checks failed')
            if mode == 'resume' and not observation.get('resume_audit',{}).get('passed'):
                raise RuntimeError('restore audit did not pass')
            if mode == 'profile' and not observation.get('routes'):
                raise RuntimeError('profiler did not export kernel routes')
        report['status'] = 'completed_for_review'
    except Exception as exc:
        report['status'] = 'failed_or_budget_stopped'
        report['error'] = repr(exc)
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out/'followup.json', report)
        artifact = package(out)
        print('Diagnostic status:', report['status'])
        print('DOWNLOAD:', artifact, flush=True)
        print('STOP THE PAID INSTANCE MANUALLY; this script never manages billing.', flush=True)
    return 0 if report['status']=='completed_for_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
