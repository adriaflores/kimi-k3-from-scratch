#!/usr/bin/env python3
"""M7.2: offline contract audit + remote read-only runtime/data readiness.

Never invokes training, writes checkpoints, starts or stops GPU instances.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'experiments/m7/current'
CONFIG = ASSETS / 'pilot_80m.yaml'
BASELINE = ASSETS / 'm6_baseline_train_config.json'
CONTRACT = ASSETS / 'pilot_contract.json'
LOCAL_M71_REPORT = ROOT / 'results/m7/m71/local_preflight.json'
REMOTE_ROOT = Path('/workspace/kimi')


def atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scalar(value):
    if value in ('true','false'):
        return value == 'true'
    try:
        if value.startswith(('0x', '0o', '0b')):
            raise ValueError('Not allowed numeric representation')
        return int(value) if '.' not in value and 'e' not in value.lower() else float(value)
    except ValueError:
        return value


def simple_yaml(path):
    """Restricted mapping-only parser for preparing the YAML without third-party dependencies.

    The remote gate additionally parses it through the REAL repository load_config.
    """
    values = {}
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.count(':') != 1:
            raise ValueError(f'Unsupported YAML syntax at line {index}')
        key, raw = (x.strip() for x in line.split(':', 1))
        if not key.isidentifier() or not raw or key in values:
            raise ValueError(f'Invalid or duplicate YAML key at line {index}')
        values[key] = scalar(raw)
    return values


def validate_static(require_m71=True):
    candidate = simple_yaml(CONFIG)
    baseline = read_json(BASELINE)
    c = read_json(CONTRACT)
    expected = dict(baseline)
    expected.update(max_steps=2442, log_interval=32, eval_interval=128, eval_batches=8,
                    checkpoint_interval=1832)
    if candidate != expected:
        discrepancies = [k for k in sorted(set(candidate)|set(expected)) if candidate.get(k) != expected.get(k)]
        raise ValueError(f'Pilot YAML differs from frozen M6 recipe+approved observation fields: {discrepancies}')
    changes = [key for key in candidate if candidate[key] != baseline[key]]
    if sorted(changes) != sorted(c['allowed_m6_recipe_changes']):
        raise ValueError(f'Unexpected M6 recipe changes: {changes}')
    tokens = candidate['batch_size']*candidate['seq_len']*candidate['gradient_accumulation_steps']
    if tokens != c['tokens_per_update'] or candidate['max_steps']*tokens != c['pilot_nominal_tokens']:
        raise ValueError('Pilot token math mismatch')
    if candidate['warmup_steps']*tokens < 60_000_000 or (candidate['warmup_steps']-1)*tokens >= 60_000_000:
        raise ValueError('Warmup conversion not ceil(60M/update)')
    if candidate['decay_steps']*tokens < 6_000_001_024 or (candidate['decay_steps']-1)*tokens >= 6_000_001_024:
        raise ValueError('Decay conversion not ceil(6B/update)')
    es = [i for i in range(1,candidate['max_steps']+1) if i%candidate['eval_interval']==0 or i==candidate['max_steps']]
    cs = [i for i in range(1,candidate['max_steps']+1) if i%candidate['checkpoint_interval']==0 or i==candidate['max_steps']]
    if es!=c['eval_steps'] or cs!=c['checkpoint_steps']:
        raise ValueError('Event schedule mismatch')
    if candidate['eval_batches']*tokens != c['eval_tokens_per_eval']:
        raise ValueError('Evaluation token volume mismatch')
    if require_m71:
        previous = read_json(LOCAL_M71_REPORT)
        if previous.get('status')!='passed' or previous.get('files')!=254 or previous.get('bytes')!=12_633_283_564:
            raise ValueError('M7.1 local preflight not accepted')
    return {'configuration':candidate, 'changes_from_m6':changes,
            'token_arithmetic':{'tokens_per_step':tokens,'warmup_tokens':candidate['warmup_steps']*tokens,
                                'pilot_tokens':candidate['max_steps']*tokens,
                                'decay_horizon_tokens':candidate['decay_steps']*tokens},
            'events':{'validation_steps':es,'validation_count':len(es),'checkpoint_steps':cs,
                      'validation_tokens_each':candidate['eval_batches']*tokens}}


def remote_runtime(audit):
    c = read_json(CONTRACT)
    m71_status = REMOTE_ROOT / 'artifacts/m7/latest_status.txt'
    if not m71_status.is_file() or m71_status.read_text().strip() != 'success':
        raise RuntimeError('M7.1 acceptance is not successful on this Vast instance')
    runtime=Path(c['python_runtime'])
    if Path(sys.executable).resolve()!=runtime.resolve():
        raise RuntimeError(f'M7.2 must run in pinned M6 Python, found {sys.executable}')
    repo=Path(c['source_reference'])
    control=REMOTE_ROOT / 'control/kimi_k3_milestone5_branch5'
    data=Path(c['data_path'])
    if not repo.joinpath('scripts/train.py').is_file() or not data.joinpath('metadata.json').is_file():
        raise RuntimeError('Missing frozen source tree or full dataset')
    sys.path.insert(0,str((control/'scripts/milestone5').resolve()))
    from environment_common import source_identity, dependency_identity, apply_numerical_profile
    source=source_identity(repo,scope='all')
    dependency=dependency_identity()
    if not source.get('ok') or not dependency.get('ok'):
        raise RuntimeError(f'Immutable M6 source or dependency validation failed: {source.get("ok")}, {dependency.get("ok")}')
    import torch
    num_profile=apply_numerical_profile(torch)
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one H100 expected')
    gpu=torch.cuda.get_device_properties(0)
    if (gpu.major,gpu.minor)!=(9,0) or 'H100 NVL' not in gpu.name:
        raise RuntimeError(f'Expected H100 NVL SM90, got {gpu.name}')
    audit['snapshot']={'source_checked':source.get('checked',source.get('matched')),
                       'dependencies_checked':dependency.get('matched'),
                       'python':sys.version.split()[0], 'torch':str(torch.__version__),
                       'cuda':torch.version.cuda, 'gpu':gpu.name,
                       'numerical_profile':num_profile}
    import shutil
    if shutil.disk_usage(REMOTE_ROOT).free < 20*1024**3:
        raise RuntimeError('Less than 20 GiB disk free; M7.3 checkpoint staging unsafe')
    sys.path.insert(0,str(repo.resolve()))
    os.chdir(repo)
    from src.kimi_k3.training.trainer import TrainConfig, learning_rate_at_step
    from src.kimi_k3.data.dataset import ShardedTokenDataset
    from src.kimi_k3.utils import load_config
    from src.kimi_k3.model import KimiK3Config
    cfg=load_config(CONFIG,TrainConfig)
    expected=audit['configuration']
    if asdict(cfg)!=expected:
        raise RuntimeError('Real load_config TrainConfig YAML differs from local static parser')
    model_cfg=load_config(repo/'configs/model/kimi_k3_300m.yaml',KimiK3Config)
    if model_cfg.vocab_size != 49_152 or model_cfg.max_seq_len < cfg.seq_len:
        raise RuntimeError('Model config not compatible with pilot')
    train=ShardedTokenDataset(data,split='train')
    val=ShardedTokenDataset(data,split='val')
    if train.total_tokens!=c['expected_train_tokens'] or val.total_tokens!=c['expected_val_tokens']:
        raise RuntimeError('Full-corpus token count mismatch')
    if train.metadata_sha256 != c['expected_metadata_canonical_sha256'] or val.metadata_sha256 != train.metadata_sha256:
        raise RuntimeError('Data manifest canonical SHA mismatch')
    import hashlib
    if hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest()!=c['expected_dataset_metadata_file_sha256']:
        raise RuntimeError('Dataset metadata file checksum differs from M7.1 reference')
    audit['dataset']={'train_tokens':train.total_tokens,'val_tokens':val.total_tokens,
                      'train_shards':len(train.shards),'val_shards':len(val.shards),
                      'canonical_metadata_sha256':train.metadata_sha256}
    targets=[1,cfg.warmup_steps,cfg.warmup_steps+1,cfg.max_steps,cfg.decay_steps]
    rates={str(s):learning_rate_at_step(cfg,s) for s in targets}
    if not math.isclose(rates[str(cfg.warmup_steps)],cfg.learning_rate,rel_tol=1e-13):
        raise RuntimeError('LR does not peak at warmup boundary')
    if not math.isclose(rates[str(cfg.decay_steps)],cfg.learning_rate*cfg.min_lr_ratio,rel_tol=1e-13):
        raise RuntimeError('Cosine horizon changes before 6B')
    future=replace(cfg,max_steps=cfg.decay_steps)
    for step in targets:
        if learning_rate_at_step(cfg,step)!=learning_rate_at_step(future,step):
            raise RuntimeError('Schedule differs when max_steps expanded; cannot resume M8')
    audit['scheduler']={'lr_at_step':rates,'m8_max_steps':future.max_steps,'independent_of_max_steps':True}
    audit['checkpoint_policy']={'checkpoint_steps':audit['events']['checkpoint_steps'],
        'note':'Both checkpoints overwrite last.pt. M7.3 must preserve final and optionally warmup snapshots before GPU shutdown.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('local','remote'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    t0=time.monotonic()
    report={'phase':'m72-'+args.mode+'-acceptance','status':'failed','training_started':False,
            'created_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        report.update(validate_static(require_m71=(args.mode=='local')))
        if args.mode=='remote':remote_runtime(report)
        report['status']='passed'
        print(f'{args.mode.upper()} M7.2 PASS — no training started.',flush=True)
    except BaseException as e:
        report['error']=type(e).__name__+': '+str(e)
        report['traceback']=traceback.format_exc()
        print('M7.2 FAIL: '+report['error'],file=sys.stderr,flush=True)
    finally:
        report['duration_seconds']=round(time.monotonic()-t0,3)
        atomic_json(args.out,report)
        print('Evidence:',args.out,flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
