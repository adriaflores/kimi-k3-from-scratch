#!/usr/bin/env python3
"""M8.2 operational control plane; stdlib-only until optional checkpoint validation.

M8.1 and M8.3 are independent gates. Nothing here changes trainer or recipe.
Production start is blocked without a verified M8.1 remote receipt and explicit
phase authorization; continuous mode additionally requires the M8.3 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/workspace/kimi')
SOURCE = BASE / 'runs/m73/m73_pilot_80m_v1_20261014T065121Z/last.pt'
APPROVED_YAML_SHA = '2917190c3db675d18f0e439696e77222c87a8e577c5eba96aadaaf4708516ca6'
SOURCE_SHA = 'cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12'
RUN_ID = 'm8_full_6b_v1'
START_STEP, TARGET_STEP, TOKENS_PER_STEP = 2442, 183106, 32768
FIRST_CP, PERIOD = 4096, 4096
RUN_ID_RE = re.compile(r'^m8_full_6b_v1$')
SNAP_RE = re.compile(r'^snapshot_step_(\d+)\.json$')
MIN_SNAPSHOT_BYTES = 2_000_000_000  # realistic 2.5 GB format-7 production checkpoints


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, record, *, create_only=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.partial')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(record, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if create_only and p.exists():
        tmp.unlink(); raise FileExistsError(p)
    os.replace(tmp, p)


def production_paths(root=BASE, run_id=RUN_ID):
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError('Unapproved M8 run id')
    root = Path(root)
    return (root / 'runs/m8' / run_id, root / 'artifacts/m8/m82' / run_id)


def step_allowed(step):
    return isinstance(step, int) and not isinstance(step, bool) and (
        FIRST_CP <= step <= TARGET_STEP and (step % PERIOD == 0 or step == TARGET_STEP))


def verify_gate(root, phase, *, new_run, worker=False):
    root = Path(root)
    receipt = load_json(root / 'artifacts/m8/m81/preflight.json')
    if receipt.get('stage') != 'remote' or receipt.get('status') != 'PASS' or receipt.get('training_started') is not False:
        raise ValueError('M8.1 remote gate not PASS; no training allowed')
    checks = receipt.get('checks', {})
    if checks.get('recipe', {}).get('sha256') != APPROVED_YAML_SHA or checks.get('checkpoint', {}).get('sha256') != SOURCE_SHA:
        raise ValueError('M8.1 immutable recipe/source hashes absent or invalid')
    if not all(x in checks for x in ('cost', 'storage', 'reference', 'runtime', 'corpus')):
        raise ValueError('M8.1 remote receipt incomplete')
    run, art = production_paths(root)
    target = checks.get('paths', {}).get('proposed_run_dir')
    if target != str(run):
        raise ValueError('M8.1 approved a different run directory')
    if checks.get('checkpoint', {}).get('step') != START_STEP:
        raise ValueError('M8.1 checkpoint step is not source step 2442')
    if sha(root / 'experiments/m7/m75_gate/recipe/m8_full_6b.yaml') != APPROVED_YAML_SHA:
        raise ValueError('Current M8 recipe differs from approval')
    if new_run:
        if run.exists() or (art.exists() and not worker) or (worker and (art/'runner_state.json').exists()):
            raise FileExistsError('A run/artifact directory already exists; do not start again')
    else:
        if not (run / 'run.json').is_file() or not (run / 'last.pt').is_file():
            raise ValueError('Existing M8 run/checkpoint missing for continuation')
    if phase == 'continuous':
        if new_run:
            raise ValueError('Continuous mode cannot skip the first recovery drill')
        # This gate checks the *real* recovery evidence and byte-for-byte
        # identity of the resume source. Never accept a manually written PASS.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.m8 import m83
        m83.verify_continuous_authorization(root)
    return run, art


def validate_snapshot_header(path, step):
    # Pinned H100 Python; mmap avoids materializing 2.5GB of CPU tensors.
    import torch
    cp = torch.load(str(path), weights_only=True, map_location='cpu', mmap=True)
    if (cp.get('format_version'), cp.get('step'), cp.get('tokens_seen'), cp.get('skipped_updates')) != (7, step, step * TOKENS_PER_STEP, 0):
        raise ValueError(f'Checkpoint header mismatch for step {step}')
    for key in ('model', 'optimizer', 'optimizer_layout', 'loaders', 'rng_state', 'runtime', 'qk_clip'):
        if key not in cp:
            raise ValueError(f'Missing checkpoint key {key}')
    if cp['loaders']['val']['position'] != 8 * TOKENS_PER_STEP:
        raise ValueError('Validation cursor unexpectedly changed')
    if cp['loaders']['train']['position'] != (TOKENS_PER_STEP if step == TARGET_STEP else step * TOKENS_PER_STEP):
        raise ValueError('Training cursor mismatch')
    del cp


def copy_snapshot(source, snapshot, step, *, min_bytes=MIN_SNAPSHOT_BYTES, validator=validate_snapshot_header):
    """Copy open inode, refuse mutation, validate format-7 step, and commit atomically."""
    source, snapshot = Path(source), Path(snapshot)
    if not step_allowed(step):
        raise ValueError('Not a legal M8 checkpoint step')
    if snapshot.exists() or snapshot.with_suffix('.pt.partial').exists():
        raise FileExistsError('Never replace a preserved snapshot or partial')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    tmp = snapshot.with_name(snapshot.name + '.partial')
    digest = hashlib.sha256(); count = 0
    try:
        with source.open('rb') as src, tmp.open('xb') as dst:
            before = os.fstat(src.fileno())
            if before.st_size < min_bytes:
                raise ValueError('Checkpoint too small')
            while True:
                block = src.read(8 * 1024 * 1024)
                if not block: break
                dst.write(block); digest.update(block); count += len(block)
            dst.flush(); os.fsync(dst.fileno())
            after = os.fstat(src.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or count != before.st_size:
                raise RuntimeError('Source changed during snapshot')
        validator(tmp, step)
        os.replace(tmp, snapshot)
        record = {'step': step, 'sha256': digest.hexdigest(), 'bytes': count,
                  'file': snapshot.name, 'verified_format7': True, 'created_at_utc': utc()}
        try:
            atomic_json(snapshot.with_suffix('.json'), record, create_only=True)
        except Exception:
            # Never leave unreceipted snapshots in the official rotation.
            snapshot.unlink(missing_ok=True)
            raise
        return record
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def snapshot_receipts(art):
    snapshots = Path(art) / 'snapshots'
    records = []
    for p in snapshots.glob('snapshot_step_*.json'):
        m = SNAP_RE.fullmatch(p.name)
        if not m: continue
        row = load_json(p)
        if row.get('step') != int(m.group(1)) or not step_allowed(row['step']):
            raise ValueError('Invalid snapshot receipt')
        f = snapshots / row.get('file', '')
        if f.name != f'snapshot_step_{row["step"]}.pt' or not f.is_file() or f.stat().st_size != row.get('bytes'):
            raise ValueError('Snapshot missing or size mismatch')
        records.append(row)
    return sorted(records, key=lambda r:r['step'])


def offsite_ack(art):
    path = Path(art)/'offsite_ack.json'
    return load_json(path) if path.is_file() else None


def ack_backup(art, step, checksum):
    records = {x['step']:x for x in snapshot_receipts(art)}
    row = records.get(step)
    if row is None or row['sha256'] != checksum:
        raise ValueError('Backup ACK does not match an existing verified remote snapshot')
    old = offsite_ack(art)
    if old is not None and old['step'] > step:
        raise ValueError('Cannot downgrade an acknowledged external backup')
    record = {'step':step,'sha256':checksum,'bytes':row['bytes'],
              'verified_external':True,'acknowledged_at_utc':utc(),
              'attestation':'Client asserts independent SHA-256 verification; not remote proof of possession'}
    atomic_json(Path(art)/'offsite_ack.json', record)
    return record


def backup_due(art):
    rows = snapshot_receipts(art)
    if not rows: return None
    newest = rows[-1]
    ack = offsite_ack(art)
    if ack is None:
        return newest
    if newest['step'] == TARGET_STEP and ack['step'] != TARGET_STEP:
        return newest
    # Request in advance: backup every <=4 periodic checkpoints.
    if newest['step'] - ack['step'] >= 3 * PERIOD:
        return newest
    return None


def retention(art, *, keep=3):
    """Never prune without >=3 current snapshots AND a verified off-Vast ACK."""
    records = snapshot_receipts(art)
    if len(records) <= keep: return []
    ack = offsite_ack(art)
    if ack is None:
        raise ValueError('Cannot rotate snapshots without an off-Vast ACK')
    old = records[:-keep]
    if ack['step'] < old[-1]['step']:
        raise ValueError('External backup too old for safe remote snapshot rotation')
    deleted = []
    for record in old:
        pt = Path(art)/'snapshots'/record['file']
        sidecar = pt.with_suffix('.json')
        pt.unlink(); sidecar.unlink(); deleted.append(record['step'])
    return deleted


def checkpoint_coverage(art):
    rows = snapshot_receipts(art)
    ack = offsite_ack(art)
    if not rows: return {'status':'PENDING','latest_step':None,'offsite_step':None,'backup_due':False}
    latest = rows[-1]['step']
    last = ack['step'] if ack else 0
    status = 'OK'
    if latest == TARGET_STEP and last != TARGET_STEP: status = 'FINAL_BACKUP_PENDING'
    elif last == 0: status = 'BACKUP_REQUIRED'
    elif latest - last >= 4*PERIOD: status = 'BACKUP_LAG_LIMIT'
    return {'status':status,'latest_step':latest,'offsite_step':last,
            'backup_due':backup_due(art) is not None,'remote_snapshots':len(rows)}


class MetricTail:
    """Exactly newline-terminated JSONL records; ignore partial last line."""
    def __init__(self, path):self.path=Path(path);self.offset=0
    def read(self):
        if not self.path.exists(): return []
        if self.path.stat().st_size < self.offset:
            raise ValueError('metrics.jsonl shrank unexpectedly')
        records=[]
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                before=stream.tell()
                line=stream.readline()
                if not line or not line.endswith(b'\n'):
                    break
                if len(line)>2_000_000: raise ValueError('Oversized metric event')
                self.offset=stream.tell()
                try: record=json.loads(line)
                except (UnicodeError, ValueError) as exc:
                    raise ValueError(f'Corrupt committed JSONL at byte {before}') from exc
                records.append(record)
        return records


def moe_diversity(moe):
    summaries=[]
    for layer, metrics in (moe or {}).items():
        counts=metrics.get('expert_counts', [])
        if len(counts)!=64 or any(type(n) is not int or n<0 for n in counts) or not sum(counts):
            raise ValueError(f'Invalid MoE counts for layer {layer}')
        total=sum(counts); ps=[n/total for n in counts if n]
        shannon=math.exp(-sum(p*math.log(p) for p in ps))
        simpson=1/sum(p*p for p in ps)
        biases=metrics.get('qb_biases', [])
        biases_valid=len(biases)==64 and all(isinstance(b,(int,float)) and math.isfinite(b) for b in biases)
        summaries.append({'layer':layer,'shannon':shannon,'simpson':simpson,
                          'qb_min':min(biases) if biases_valid else None,
                          'qb_max':max(biases) if biases_valid else None,
                          'max_load_over_mean':max(counts)*64/total,
                          'zero_load_experts':counts.count(0),
                          'bias_nonfinite_or_missing':not biases_valid})
    return summaries


class Monitor:
    def __init__(self):self.best_val=None;self.bad_val=0;self.latest={};self.warnings=[]
    def feed(self, rec):
        if rec.get('schema_version') !=1 or rec.get('event') not in ('session_start','train','val','checkpoint','overflow','session_end'):
            raise ValueError('Unexpected metric schema/event')
        step=rec.get('step')
        if not isinstance(step,int) or step < START_STEP or step > TARGET_STEP:
            raise ValueError('Invalid step in metrics')
        event=rec['event'];self.latest['step']=step;self.latest['event']=event
        if rec.get('skipped_updates',0)!=0 or event=='overflow':
            raise RuntimeError('Optimizer skipped updates, investigate numerics')
        if event=='train':
            for key in ('train_loss','grad_norm_max','lr'):
                value=rec.get(key)
                if not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise RuntimeError(f'Nonfinite/missing train diagnostic: {key}')
            if rec.get('nonfinite_grad_norms',0):raise RuntimeError('Nonfinite gradient detected')
            if rec.get('skipped_updates_in_window',0):raise RuntimeError('Training skip detected')
            qk=rec.get('qk_clip', {})
            if qk.get('max_logit') is not None and not math.isfinite(qk['max_logit']):
                raise RuntimeError('Nonfinite QK logit')
            moe=moe_diversity(rec.get('moe'))
            self.latest.update({'train_loss':rec['train_loss'], 'lr':rec['lr'],
                'grad_norm_max':rec['grad_norm_max'], 'moe_min_shannon':min((x['shannon'] for x in moe),default=None),
                'moe_max_load_ratio':max((x['max_load_over_mean'] for x in moe),default=None),
                'tokens_per_second':rec.get('train_tokens_per_second'),
                'grad_norm_mean':rec.get('grad_norm_mean'),
                'clipped_fraction':rec.get('clipped_fraction'),
                'moe_min_simpson':min((x['simpson'] for x in moe),default=None),
                'moe_max_zero_experts':max((x['zero_load_experts'] for x in moe),default=None),
                'moe_qb_min':min((x['qb_min'] for x in moe if x['qb_min'] is not None),default=None),
                'moe_qb_max':max((x['qb_max'] for x in moe if x['qb_max'] is not None),default=None),
                'qk_max_logit':qk.get('max_logit'),
                'qk_clipped_head_updates':qk.get('clipped_head_updates'),
                'cuda_peak_allocated_bytes':(rec.get('memory') or {}).get('cuda_peak_allocated_bytes'),
                'cuda_peak_reserved_bytes':(rec.get('memory') or {}).get('cuda_peak_reserved_bytes')})
            if any(row['bias_nonfinite_or_missing'] for row in moe):
                raise RuntimeError('Invalid MoE QB diagnostics')
            if any(row['zero_load_experts'] or row['shannon']<32 for row in moe):
                self.warnings.append({'step':step, 'reason':'MoE diversity/routing deterioration; not automatic collapse verdict'})
        elif event=='val':
            v=rec.get('val_loss')
            if not isinstance(v,(int,float)) or not math.isfinite(v):raise RuntimeError('Nonfinite/missing validation loss')
            self.best_val=v if self.best_val is None else min(self.best_val,v)
            self.bad_val=self.bad_val+1 if v>self.best_val*1.12 else 0
            self.latest['val_loss']=v
            if self.bad_val>=3:
                self.warnings.append({'step':step,'reason':'3 persistent high validation losses; investigate'})
                if self.latest.get('moe_min_shannon') is not None and self.latest['moe_min_shannon'] < 32:
                    raise RuntimeError('Persistent val deterioration alongside severe MoE diversity loss')


def process_metrics(tail, monitor, run, art, *, snapshot_validator=validate_snapshot_header,
                    min_snapshot_bytes=MIN_SNAPSHOT_BYTES):
    checkpoint_steps=[]
    previous = snapshot_receipts(art)
    last_snapshot = previous[-1]['step'] if previous else 0
    for event in tail.read():
        monitor.feed(event)
        if event['event']!='checkpoint':continue
        step=event['step']
        if not step_allowed(step): raise RuntimeError('Unexpected checkpoint cadence')
        if step<=last_snapshot:continue
        if step<=START_STEP:raise RuntimeError('Refuse source checkpoint event')
        if step != FIRST_CP and step != TARGET_STEP and step%PERIOD != 0:
            raise RuntimeError('Unexpected checkpoint boundary')
        record=copy_snapshot(Path(run)/'last.pt',Path(art)/'snapshots'/f'snapshot_step_{step}.pt',step,
                             min_bytes=min_snapshot_bytes,validator=snapshot_validator)
        last_snapshot=step
        checkpoint_steps.append(step)
        # Protect required external backup coverage before pruning.
        retention(art)
        if checkpoint_coverage(art)['status']=='BACKUP_LAG_LIMIT':
            raise RuntimeError('Off-Vast backup lag reached 4 checkpoints: pause')
    return checkpoint_steps


def enforce_disk(root, reserve_gib=8):
    free=shutil.disk_usage(root).free
    if free < reserve_gib * 1024**3:
        raise RuntimeError(f'Free disk below {reserve_gib} GiB during production')


def supervisor(root, phase, *, python, command=None, interval=5, stall_seconds=900,
               validator=validate_snapshot_header, min_snapshot_bytes=MIN_SNAPSHOT_BYTES):
    root=Path(root)
    run,art=production_paths(root)
    log=art/'training.log'
    art.mkdir(parents=True,exist_ok=True)
    monitor=Monitor();tail=MetricTail(run/'metrics.jsonl')
    stopped_for_drill=False
    training_started=False
    proc=None;rc=None;incident=None
    if command is None:
        command=[str(python),'-u',str(root/'reference/scripts/train.py'),
            '--data-dir',str(root/'data/full'), '--model-config',str(root/'reference/configs/model/kimi_k3_300m.yaml'),
            '--training-config',str(root/'experiments/m7/m75_gate/recipe/m8_full_6b.yaml'),
            '--device','cuda','--run-dir',str(run),
            '--resume',str(SOURCE if phase=='drill' else run/'last.pt'),
            '--checkpoint',str(run/'last.pt')]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1')
    try:
        with log.open('ab') as out:
            proc=subprocess.Popen(command,cwd=str(root/'reference'),env=env,stdout=out,stderr=subprocess.STDOUT)
            training_started=True
            atomic_json(art/'runner_state.json',{'state':'RUNNING','phase':phase,'pid':proc.pid,'started_at_utc':utc()})
            last_activity=time.monotonic()
            # A normal training invocation should emit a session_start once model construction completes.
            while True:
                previous_offset=tail.offset
                snapshots=process_metrics(tail,monitor,run,art,snapshot_validator=validator,
                                          min_snapshot_bytes=min_snapshot_bytes)
                if tail.offset>previous_offset:
                    last_activity=time.monotonic()
                    atomic_json(art/'heartbeat.json',{'at_utc':utc(),'phase':phase,'monitor':monitor.latest,
                                                     'warnings':monitor.warnings[-15:]})
                if snapshots:enforce_disk(root)
                if phase=='drill' and FIRST_CP in snapshots:
                    stopped_for_drill=True
                    proc.terminate()
                if proc.poll() is not None:
                    rc=proc.returncode
                    # Flush final committed metrics after child exit.
                    final_steps=process_metrics(tail,monitor,run,art,snapshot_validator=validator,
                                                min_snapshot_bytes=min_snapshot_bytes)
                    if phase=='drill' and FIRST_CP in final_steps:stopped_for_drill=True
                    break
                if time.monotonic()-last_activity>stall_seconds:
                    raise RuntimeError('No committed metrics for too long; training may be stalled')
                if checkpoint_coverage(art)['status']=='BACKUP_LAG_LIMIT':
                    raise RuntimeError('Off-Vast backup lag reached maximum')
                time.sleep(interval)
        if phase=='drill':
            if not stopped_for_drill or not any(x['step']==FIRST_CP for x in snapshot_receipts(art)):
                raise RuntimeError('Drill did not produce verified step 4096 snapshot')
            state='AWAITING_M83_RECOVERY_GATE'
        elif rc == 0 and any(x['step']==TARGET_STEP for x in snapshot_receipts(art)):
            state='COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP'
        else:
            raise RuntimeError(f'Trainer exited {rc}, or no final snapshot')
    except BaseException as exc:
        incident=f'{type(exc).__name__}: {exc}'
        state='PAUSED_INCIDENT'
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if proc:rc=proc.returncode
    finally:
        atomic_json(art/'runner_state.json',{'state':state,'phase':phase,'pid':proc.pid if proc else None,
                'train_exit_code':rc,'training_started':training_started,'incident':incident,
                'monitor':monitor.latest,'warnings':monitor.warnings[-20:], 'ended_at_utc':utc()})
    return 0 if state in ('AWAITING_M83_RECOVERY_GATE','COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP') else 2


def status(root):
    run, art=production_paths(root)
    state=load_json(art/'runner_state.json') if (art/'runner_state.json').is_file() else {'state':'NOT_STARTED'}
    cover=checkpoint_coverage(art)
    if state['state']=='COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP' and cover['latest_step']==TARGET_STEP and cover['offsite_step']==TARGET_STEP:
        state['state']='READY_FOR_FINAL_VERIFICATION_AND_SHUTDOWN'
    heartbeat=(load_json(art/'heartbeat.json') if (art/'heartbeat.json').is_file() else None)
    result={'run_id':RUN_ID,'runner':state,'heartbeat':heartbeat,'backup':cover,
            'request':backup_due(art),
            'snapshot_directory':str(art/'snapshots'),
            'run_directory':str(run)}
    return result


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    for name in ('status','request','ack','guard','supervise'):
        s=sub.add_parser(name);s.add_argument('--root',type=Path,default=BASE)
        if name=='ack':s.add_argument('--step',type=int,required=True);s.add_argument('--sha256',required=True)
        if name=='guard':s.add_argument('--phase',choices=('drill','continuous'),required=True)
        if name=='supervise':
            s.add_argument('--phase',choices=('drill','continuous'),required=True)
            s.add_argument('--python',type=Path,required=True)
    return p.parse_args(argv)


def main(argv=None):
    args=parse(argv)
    try:
        if args.action=='guard':
            run,art=verify_gate(args.root,args.phase,new_run=args.phase=='drill')
            print(json.dumps({'status':'PASS','phase':args.phase,'run':str(run),'art':str(art)}))
        elif args.action=='supervise':
            verify_gate(args.root,args.phase,new_run=args.phase=='drill',worker=True)
            if args.phase=='continuous':
                s=load_json(production_paths(args.root)[1]/'runner_state.json')
                if s.get('state') not in ('AWAITING_M83_RECOVERY_GATE','RECOVERY_APPROVED'):
                    raise ValueError('Cannot enter continuous phase without completed recovery gate')
            return supervisor(args.root,args.phase,python=args.python)
        elif args.action=='ack':
            _,art=production_paths(args.root)
            print(json.dumps(ack_backup(art,args.step,args.sha256),indent=2))
        else:
            result=status(args.root)
            print(json.dumps(result if args.action=='status' else result['request'],indent=2))
        return 0
    except Exception as exc:
        print(f'M8.2 {args.action} NO-GO: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
