#!/usr/bin/env python3
"""M8.3 fail-closed recovery authorization and production-close audit.

All GPU/tensor loading happens in probe_resume_m83.py on the qualified H100.
Local checks and injection-tested incident simulations use Python stdlib only.
Never executes optimizer steps; only M8.2 guarded worker launches the trainer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.m8 import m82
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.m8 import m82

ROOT = Path('/workspace/kimi')
ART = Path('artifacts/m8/m83')
PROBE_NAME = 'probe_receipt.json'
GATE_NAME = 'recovery_gate.json'
PYTHON = Path('control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python')
SCRIPT = Path('scripts/m8/probe_resume_m83.py')
STEP = m82.FIRST_CP


def digest(path):
    return m82.sha(path)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_string(value):
    return type(value) is str and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


def paths(root):
    root = Path(root)
    run, art = m82.production_paths(root)
    return run, art, root / ART


def step_receipt(art, step):
    rows = {row['step']: row for row in m82.snapshot_receipts(art)}
    _require(step in rows, 'Verified remote snapshot receipt missing')
    row = rows[step]
    _require(row.get('verified_format7') is True and _sha_string(row.get('sha256')),
             'Snapshot format/header/SHA attestation missing')
    _require(type(row.get('bytes')) is int and row['bytes'] >= m82.MIN_SNAPSHOT_BYTES,
             'Checkpoint size unexpectedly small')
    return row


def _runner(art):
    return m82.load_json(art / 'runner_state.json')


def _check_manifest(run):
    manifest = m82.load_json(run / 'run.json')
    _require(manifest.get('schema_version') == 1 and isinstance(manifest.get('sessions'), list)
             and len(manifest['sessions']) >= 1, 'Run manifest absent/invalid')
    _require(manifest['sessions'][0].get('checkpoint_path') == str((run/'last.pt').resolve()),
             'M8 trainer output path was changed')
    _require(manifest['sessions'][0].get('start_state', {}).get('step') == m82.START_STEP,
             'M8 must originate at step 2442')
    _require(manifest['sessions'][0].get('resume_checkpoint') == str(m82.SOURCE),
             'M8 initial run does not resume approved pilot checkpoint')
    return manifest


def _check_probe(probe, *, expected_step, expected_sha, snapshot):
    _require(probe.get('status') == 'PASS' and probe.get('no_optimizer_updates_executed') is True,
             'Real restore proof is not PASS/no-updates')
    _require(probe.get('checkpoint_step') == expected_step and probe.get('restored_step') == expected_step,
             'Probe restored wrong step')
    _require(probe.get('checkpoint_sha256_before') == expected_sha and
             probe.get('checkpoint_sha256_after') == expected_sha, 'Probe checkpoint hash mismatch')
    _require(probe.get('checkpoint_path') == str(snapshot), 'Probe did not load approved snapshot')
    _require(probe.get('restored_train_position') ==
             (m82.TOKENS_PER_STEP if expected_step == m82.TARGET_STEP else expected_step*m82.TOKENS_PER_STEP),
             'Restored training cursor mismatch')
    _require(probe.get('restored_val_position') == 8*m82.TOKENS_PER_STEP,
             'Restored validation cursor mismatch')
    _require(probe.get('rng_verified') == ['python','numpy','torch_cpu','torch_cuda'],
             'Missing real RNG restore evidence')
    _require(probe.get('optimizer_state_entries') == 1830 and
             probe.get('model_parameters') == 294676808, 'Restored optimizer/model mismatch')
    _require(probe.get('original_source_sha256') == m82.SOURCE_SHA and
             probe.get('source_files_sha256_verified') == 35, 'M7 immutable source evidence mismatch')
    _require(probe.get('recipe_sha256') == m82.APPROVED_YAML_SHA,
             'Training recipe differs from approved M8 contract')
    _require(probe.get('checkpoint_format_version') == 7 and
             probe.get('skipped_updates') == 0, 'Incompatible format or skipped updates')
    _require(probe.get('torch') == '2.11.0+cu128' and probe.get('gpu') == 'NVIDIA H100 NVL',
             'Restore probe not run with approved H100 runtime')
    _require(probe.get('checkpoint_filesize') == snapshot.stat().st_size,
             'Checkpoint bytes differ from probe evidence')
    return True


def drill_ready(root, *, verify_hash=True):
    run, art, m83art = paths(root)
    # M8.1 persisted gate must remain approved. No new run: M8.2 checks run.json.
    pre = m82.load_json(Path(root)/'artifacts/m8/m81/preflight.json')
    _require(pre.get('status') == 'PASS' and pre.get('stage') == 'remote' and
             pre.get('checks',{}).get('checkpoint',{}).get('sha256') == m82.SOURCE_SHA,
             'M8.1 remote preflight not approved')
    state = _runner(art)
    _require(state.get('state') == 'AWAITING_M83_RECOVERY_GATE' and state.get('phase') == 'drill'
             and state.get('training_started') is True, 'Drill worker did not finish in required state')
    _require(not (art/'training.lock').exists(), 'Training lock still present')
    _check_manifest(run)
    row = step_receipt(art, STEP)
    snap = art / 'snapshots' / row['file']
    _require(snap.is_file() and (run/'last.pt').is_file(), 'Checkpoint missing')
    _require((run/'last.pt').stat().st_size == row['bytes'], 'Trainer checkpoint size differs from snapshot')
    if verify_hash:
        _require(digest(snap) == row['sha256'] and digest(run/'last.pt') == row['sha256'],
                 'Step4096 snapshot/current checkpoint integrity failed')
        _require(digest(m82.SOURCE if Path(root)==ROOT else Path(root)/m82.SOURCE.relative_to(ROOT)) == m82.SOURCE_SHA,
                 'M7.3 source checkpoint has changed')
    ack = m82.offsite_ack(art)
    _require(ack and ack.get('step') == STEP and ack.get('sha256') == row['sha256'] and
             ack.get('verified_external') is True, 'Off-Vast verified step4096 backup ACK required')
    return {'run':run, 'art':art, 'm83art':m83art, 'row':row, 'snapshot':snap, 'checkpoint_step':STEP}


def _probe_command(root, snap, receipt):
    root = Path(root)
    python = root/PYTHON
    _require(python.is_file(), f'Pinned Python missing: {python}')
    _require((root/SCRIPT).is_file(), 'Real restore probe script missing')
    return [str(python),str(root/SCRIPT),'--root',str(root),'--checkpoint',str(snap),
            '--receipt',str(receipt)]


def _no_active_training():
    # Must be executed on Vast; tmux can be dead even when a child process survives,
    # so production gate also requires the worker's terminal runner_state.
    found = subprocess.run(['tmux','has-session','-t','kimi_m82'],capture_output=True)
    _require(found.returncode != 0, 'M8 training tmux is still active')


def run_probe(root, *, step=STEP, runner=subprocess.run, check_live=True):
    root=Path(root)
    run, art, m83art=paths(root)
    if step == STEP and _runner(art).get('state') == 'AWAITING_M83_RECOVERY_GATE':
        candidate = drill_ready(root)
        snap=candidate['snapshot']
    else:
        _require(_runner(art).get('state') == 'PAUSED_INCIDENT', 'Not a paused-incident recovery')
        _check_manifest(run)
        row=step_receipt(art,step)
        _require(step==m82.snapshot_receipts(art)[-1]['step'],
                 'Recovery must use latest preserved snapshot; review newer receipts manually')
        snap=art/'snapshots'/row['file']
        _require(digest(snap)==row['sha256'], 'Recovery snapshot checksum invalid')
        backup=m82.offsite_ack(art)
        _require(backup is not None and backup.get('verified_external') is True,
                 'Cannot recover without an independent offsite backup')
        _require(backup['step'] <= step, 'Offsite backup step must not exceed selected recovery point')
        _require(digest(root/m82.SOURCE.relative_to(ROOT)) == m82.SOURCE_SHA,'Original pilot checkpoint modified')
    if check_live: _no_active_training()
    m83art.mkdir(parents=True,exist_ok=True)
    # Preserve earlier proof/gate evidence when a failed/repeated probe or
    # later incident requires a new proof. Never silently discard provenance.
    history=m83art/'history'
    for filename in (PROBE_NAME,GATE_NAME):
        previous=m83art/filename
        if previous.is_file():
            history.mkdir(exist_ok=True)
            backup=history/f'{previous.stem}_{digest(previous)[:16]}.json'
            if not backup.exists():
                shutil.copy2(previous,backup)
    receipt=m83art/PROBE_NAME
    # Probe itself writes PASS/FAIL; no supplied/hand-edited receipt ever becomes a GO.
    command=_probe_command(root,snap,receipt)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
    proc=runner(command,cwd=str(root/'reference'),env=env,check=False)
    _require(proc.returncode==0,'Real checkpoint restore probe failed; see receipt')
    probe=m82.load_json(receipt)
    _check_probe(probe,expected_step=step,expected_sha=digest(snap),snapshot=snap)
    return probe


def _commit_gate(root, *, step, phase, probe_path=None):
    run,art,m83art=paths(root)
    probe_path=m83art/PROBE_NAME if probe_path is None else Path(probe_path)
    _require(probe_path == m83art/PROBE_NAME, 'Probe must be generated in canonical artifact path')
    row=step_receipt(art,step)
    snap=art/'snapshots'/row['file']
    _require(digest(snap)==row['sha256'], 'Snapshot changed since its SHA receipt')
    probe=m82.load_json(probe_path)
    _check_probe(probe,expected_step=step,expected_sha=row['sha256'],snapshot=snap)
    state=_runner(art)
    expected='AWAITING_M83_RECOVERY_GATE' if phase=='drill' else 'PAUSED_INCIDENT'
    _require(state.get('state')==expected, f'Expected state {expected}, got {state.get("state")}')
    if phase=='drill':
        drill_ready(root,verify_hash=True)
    else:
        _require(m82.offsite_ack(art) and m82.offsite_ack(art).get('verified_external') is True,
                 'Offsite backup missing')
        _require(step>=STEP and step<m82.TARGET_STEP,'Invalid incident recovery step')
    _no_active_training()
    _require((run/'last.pt').is_file(), 'M8 trainer last.pt missing')
    # For incident restoration, replace current last.pt only after a fresh genuine
    # model/optimizer/RNG probe from the snapshot. Never overwrite prior source.
    if phase=='incident':
        old=run/'last.pt'
        archived=run/f'incident_last_before_restore_{step}_{int(time.time())}.pt'
        _require(not archived.exists(), 'Incident checkpoint archive collision')
        _require(shutil.disk_usage(run).free >= row['bytes'] + 8*1024**3,
                 'Not enough free space for safe incident checkpoint restoration plus 8 GiB reserve')
        tmp=run/'last.pt.m83_restore.partial'
        _require(not tmp.exists(), 'Incomplete restore requires manual inspection')
        try:
            with snap.open('rb') as src, tmp.open('xb') as out:
                shutil.copyfileobj(src,out,length=8*1024*1024)
                out.flush(); os.fsync(out.fileno())
            _require(digest(tmp)==row['sha256'], 'Staged restore copy changed')
            old.replace(archived)
            try: tmp.replace(old)
            except BaseException:
                archived.replace(old)
                raise
        finally:
            tmp.unlink(missing_ok=True)
    _require(digest(run/'last.pt')==row['sha256'],'Current last.pt differs from proven restore point')
    gate={'schema_version':1,'status':'PASS','run_id':m82.RUN_ID,'phase':phase,
          'checkpoint_step':step,'checkpoint_sha256':row['sha256'],
          'probe_receipt_sha256':digest(probe_path),'source_checkpoint_sha256':m82.SOURCE_SHA,
          'approved_recipe_sha256':m82.APPROVED_YAML_SHA,
          'ready_for_continuous':True,'no_training_during_probe':True,
          'approved_at_utc':m82.utc()}
    m82.atomic_json(m83art/GATE_NAME,gate,create_only=(phase=='drill'))
    if phase=='incident':
        m82.atomic_json(art/'runner_state.json',{
            'state':'RECOVERY_APPROVED','phase':'incident','recovery_step':step,
            'backup_attested':True,'at_utc':m82.utc()})
    return gate


def verify_continuous_authorization(root, *, rehash=True):
    run,art,m83art=paths(root)
    gate=m82.load_json(m83art/GATE_NAME)
    _require(gate.get('schema_version')==1 and gate.get('status')=='PASS' and
             gate.get('run_id')==m82.RUN_ID and gate.get('ready_for_continuous') is True and
             gate.get('no_training_during_probe') is True,'M8.3 recovery gate incomplete')
    phase=gate.get('phase')
    _require(phase in ('drill','incident'),'Invalid gate phase')
    step=gate.get('checkpoint_step')
    row=step_receipt(art,step)
    _require(gate.get('checkpoint_sha256')==row['sha256'] and
             gate.get('source_checkpoint_sha256')==m82.SOURCE_SHA and
             gate.get('approved_recipe_sha256')==m82.APPROVED_YAML_SHA,'Recovery gate hashes inconsistent')
    _require(gate.get('probe_receipt_sha256')==digest(m83art/PROBE_NAME),'Probe receipt was modified')
    _check_probe(m82.load_json(m83art/PROBE_NAME),expected_step=step,expected_sha=row['sha256'],
                 snapshot=art/'snapshots'/row['file'])
    ack=m82.offsite_ack(art)
    _require(ack and ack.get('verified_external') is True and ack.get('step')<=step,
             'Offsite recovery coverage required')
    if phase=='drill':
        _require(step==STEP and ack['step']==STEP and ack['sha256']==row['sha256'],
                 'First checkpoint offsite ACK not verified')
    _require((run/'last.pt').stat().st_size==row['bytes'],'Current M8 checkpoint size changed')
    if rehash:
        _require(digest(run/'last.pt')==row['sha256'] and
                 digest(art/'snapshots'/row['file'])==row['sha256'],
                 'M8 resume checkpoint differs from restored snapshot')
    return gate


def final_audit(root, *, independent_backup=False):
    root=Path(root)
    run,art,m83art=paths(root)
    state=m82.load_json(art/'runner_state.json')
    _require(state.get('state')=='COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP',
             'Trainer did not finish successfully')
    row=step_receipt(art,m82.TARGET_STEP)
    _require(digest(art/'snapshots'/row['file'])==row['sha256'],'Final remote snapshot SHA mismatch')
    _require(digest(run/'last.pt')==row['sha256'],'Final trainer last.pt differs from snapshot')
    backup=m82.offsite_ack(art)
    _require(backup is not None and backup.get('step')==m82.TARGET_STEP and
             backup.get('sha256')==row['sha256'] and backup.get('verified_external') is True,
             'Final off-Vast backup is not acknowledged')
    _check_manifest(run)
    tail=run/'metrics.jsonl'
    _require(tail.is_file(),'Training metrics missing')
    latest_step=0; checkpoint_seen=False; final_val=False; count=0
    with tail.open('rb') as stream:
        for line in stream:
            _require(line.endswith(b'\n'),'Incomplete metrics tail at final audit')
            r=json.loads(line)
            if r.get('schema_version')!=1: raise ValueError('Invalid event schema')
            _require(r.get('skipped_updates',0)==0,'Skipped updates in M8 metrics')
            if r.get('event')=='checkpoint' and r.get('step')==m82.TARGET_STEP:checkpoint_seen=True
            if r.get('event')=='val' and r.get('step')==m82.TARGET_STEP:final_val=True
            latest_step=max(latest_step,r.get('step',0));count+=1
    _require(count>0 and latest_step==m82.TARGET_STEP and checkpoint_seen and final_val,
             'Final checkpoint/validation events missing')
    report={'status':'PASS','final_step':m82.TARGET_STEP,'tokens_seen':m82.TARGET_STEP*m82.TOKENS_PER_STEP,
            'checkpoint_sha256':row['sha256'],'checkpoint_bytes':row['bytes'],
            'offsite_acknowledged':True,'offsite_cryptographic_recheck':independent_backup,
            'events':count,'completed_at_utc':m82.utc(),
            'decision':'Remote evidence PASS; verify local checkpoint bytes independently before shutdown'}
    m82.atomic_json(m83art/'final_audit.json',report)
    return report


def package_evidence(root):
    root=Path(root)
    run,art,m83art=paths(root)
    dest=m83art/'m8_final_evidence.zip'
    if dest.exists():
        expected=(m83art/'m8_final_evidence.zip.sha256').read_text(encoding='utf-8').split()[0]
        _require(digest(dest)==expected,'Existing final evidence package checksum changed')
        # Idempotent recovery after an interrupted SCP download: do not
        # regenerate an archive or silently change the published evidence.
        report=m82.load_json(m83art/'final_audit.json')
        _require(report.get('status')=='PASS','Existing packaged audit invalid')
        return {'status':'ALREADY_PACKAGED','file':str(dest),'sha256':expected,'audit':report}
    report=final_audit(root)
    files=[(m83art/'final_audit.json','final_audit.json'),(m83art/GATE_NAME,'recovery_gate.json'),
           (m83art/PROBE_NAME,'probe_receipt.json'),
           (art/'runner_state.json','runner_state.json'),(art/'offsite_ack.json','offsite_ack.json'),
           (run/'run.json','run.json'),(run/'metrics.jsonl','metrics.jsonl'),
           (art/'heartbeat.json','heartbeat.json'),(art/'training.log','training.log'),
           (art/'snapshots'/f'snapshot_step_{m82.TARGET_STEP}.json','final_snapshot_receipt.json')]
    for p,_ in files: _require(p.is_file(),f'Missing evidence file: {p}')
    temp=dest.with_suffix('.partial')
    with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p,name in files:z.write(p,arcname=name)
    temp.replace(dest)
    (m83art/'m8_final_evidence.zip.sha256').write_text(digest(dest)+'  '+dest.name+'\n',encoding='utf-8')
    return {'status':'PACKAGED','file':str(dest),'sha256':digest(dest),'audit':report}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    for name in ('drill-ready','probe','approve','verify-go','final-audit','final-package'):
        s=sub.add_parser(name)
        s.add_argument('--root',type=Path,default=ROOT)
        if name in ('probe','approve'):s.add_argument('--step',type=int,default=STEP)
        if name=='approve':s.add_argument('--phase',choices=('drill','incident'),default='drill')
    args=p.parse_args(argv)
    try:
        if args.action=='drill-ready':result=drill_ready(args.root)
        elif args.action=='probe':result=run_probe(args.root,step=args.step)
        elif args.action=='approve':result=_commit_gate(args.root,step=args.step,phase=args.phase)
        elif args.action=='verify-go':result=verify_continuous_authorization(args.root)
        elif args.action=='final-audit':result=final_audit(args.root)
        else:result=package_evidence(args.root)
        print(json.dumps(result,indent=2,default=str))
        return 0
    except Exception as e:
        print(f'M8.3 NO-GO: {type(e).__name__}: {e}',file=sys.stderr)
        return 2


if __name__=='__main__':sys.exit(main())
