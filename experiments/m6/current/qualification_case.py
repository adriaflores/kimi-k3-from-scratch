#!/usr/bin/env python3
"""Bounded actual Kimi K3 run; no changes to checked production source.

Run with pinned Python from /workspace/kimi/control, never system Python.
The M5 train_linux recipe gate fixes 2048 tokens/update, so this *systems-only*
qualification explicitly bypasses THAT recipe gate while independently verifying
ALL M5 source and dependency hashes, GPU, and numerical execution policy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def scan_nonfinite(model, optimizer, torch):
    failures = []
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                if not bool(torch.isfinite(value).all().item()):
                    failures.append('model.' + name)
        for key, value in optimizer.state.items():
            for name, item in value.items():
                if torch.is_tensor(item) and (item.is_floating_point() or item.is_complex()):
                    if not bool(torch.isfinite(item).all().item()):
                        failures.append('optimizer.' + name)
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--batch', type=int, choices=(32,), default=32)
    args = p.parse_args()
    if args.steps not in (80, 88) or (args.resume is None) != (args.steps == 80):
        p.error('Expected initial 80 updates or resume 80->88 updates')
    args.out.mkdir(parents=True, exist_ok=False)
    status = {'schema': 1, 'mode': 'resume_8' if args.resume else 'initial_80',
              'status': 'running', 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'systems_only': True, 'learning_recipe_approved': False}
    atomic(args.out/'case_report.json', status)
    started = time.monotonic()
    try:
        assert (args.control / 'configs/training/milestone5_branch5/t1024.yaml').is_file()
        assert (args.repo / 'scripts/train.py').is_file()
        assert args.data.is_dir()
        sys.path.insert(0, str((args.control / 'scripts/milestone5').resolve()))
        from environment_common import source_identity, dependency_identity, apply_numerical_profile
        from qualification_worker import compare_tree
        import torch
        sid = source_identity(args.repo, scope='all')
        did = dependency_identity()
        if not sid['ok'] or not did['ok']:
            raise RuntimeError(f'Pinned source/dependency identity mismatch: {sid["ok"]}, {did["ok"]}')
        status['source_checked_files'] = sid.get('checked', sid.get('matched'))
        status['dependency_matched'] = did.get('matched')
        status['numerical_flags'] = apply_numerical_profile(torch)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one GPU')
        dev = torch.cuda.get_device_properties(0)
        if (dev.major, dev.minor) != (9, 0) or 'H100' not in dev.name:
            raise RuntimeError(f'Expected H100 SM90, found {dev.name}, SM{dev.major}{dev.minor}')
        status['gpu'] = {'name': dev.name, 'sm': [dev.major,dev.minor], 'memory_bytes':dev.total_memory}
        sys.path.insert(0, str(args.repo.resolve()))
        os.chdir(args.repo.resolve())
        import src.kimi_k3.training.trainer as trainer
        original_load = trainer.load_checkpoint
        original_train = trainer.train

        def audit_restore(path, model, optimizer, config, **kwargs):
            t0 = time.perf_counter()
            state = original_load(path, model, optimizer, config, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-t0
            snapshot = torch.load(path, map_location='cpu', weights_only=True)
            failures=[]
            for key, actual, expected in (
                ('model',model.state_dict(),snapshot['model']),
                ('optimizer',optimizer.state_dict(),snapshot['optimizer']),
                ('rng',trainer._capture_rng_state('cuda:0'),snapshot['rng_state']),
                ('loaders.train',kwargs['train_loader'].state_dict(),snapshot['loaders']['train']),
                ('loaders.val',kwargs['val_loader'].state_dict(),snapshot['loaders']['val']),
            ):
                compare_tree(actual, expected, torch, key, failures)
            for name in ('step','tokens_seen','skipped_updates','overfit_batch_start','grad_scaler_state'):
                compare_tree(getattr(state,name),snapshot.get(name),torch,name,failures)
            if optimizer.param_groups[0]['lr'] != snapshot['optimizer']['param_groups'][0]['lr']:
                failures.append('optimizer.lr')
            config_now=asdict(config)
            for k, expected in snapshot['train_config'].items():
                if k not in ('max_steps','log_interval','eval_interval','eval_batches','checkpoint_interval') and config_now[k]!=expected:
                    failures.append('recipe.'+k)
            qb=[k for k in snapshot['model'] if 'bias' in k.lower() and any(s in k.lower() for s in ('router','expert','balance'))]
            status['restore_audit']={
                'passed':not failures,'differences':failures[:80],
                'original_step':state.step,'original_tokens_seen':state.tokens_seen,
                'checkpoint_format_version':snapshot['format_version'],
                'model_buffers_compared':True,'optimizer_compared':True,
                'rng_compared':True,'data_cursors_compared':True,
                'qb_buffer_names':qb,'load_seconds':round(elapsed,3),
                'bitwise_same_trajectory_unverified':True,
            }
            atomic(args.out/'case_report.json', status)
            del snapshot
            if failures:
                raise RuntimeError('Exact restore mismatch: '+', '.join(failures[:10]))
            return state

        def observed_train(model, optimizer, train_loader, val_loader, config, device, **kwargs):
            initial=kwargs['state'].step
            result=original_train(model, optimizer, train_loader, val_loader, config, device, **kwargs)
            failures=scan_nonfinite(model,optimizer,torch)
            status['numerical_audit']={
                'passed':not failures and result.skipped_updates==0,
                'failures':failures[:80], 'skipped_updates':result.skipped_updates,
                'initial_step':initial,'end_step':result.step,
                'new_updates':result.step-initial,'tokens_seen':result.tokens_seen,
            }
            atomic(args.out/'case_report.json', status)
            if failures or result.skipped_updates or result.step!=args.steps or result.step-initial!=(8 if args.resume else 80):
                raise RuntimeError('Numerical/step audit failed')
            return result

        trainer.load_checkpoint = audit_restore
        trainer.train = observed_train
        tokens_update = args.batch*1024
        warmup_steps = math.ceil(60_000_000/tokens_update)
        decay_steps = math.ceil(6_000_001_024/tokens_update)
        config_path = args.control/'configs/training/milestone5_branch5/t1024.yaml'
        cli = [
            '--data-dir',str(args.data.resolve()),
            '--training-config',str(config_path.resolve()),
            '--model-config',str((args.repo/'configs/model/kimi_k3_300m.yaml').resolve()),
            '--device','cuda:0','--run-dir',str((args.out/'run').resolve()),
            '--batch-size',str(args.batch),'--seq-len','1024',
            '--gradient-accumulation-steps','1','--max-steps',str(args.steps),
            '--warmup-steps',str(warmup_steps),'--decay-steps',str(decay_steps),
            '--eval-batches','2','--eval-interval','1000000',
            '--checkpoint-interval','1000000','--log-interval','8',
        ]
        if args.resume:
            if not args.resume.is_file():
                raise FileNotFoundError(args.resume)
            cli.extend(['--resume',str(args.resume.resolve())])
        status['training']={'batch_size':args.batch,'seq_len':1024,'gradient_accumulation_steps':1,
            'effective_tokens_update':tokens_update,'warmup_steps':warmup_steps,
            'decay_steps':decay_steps,'eval_tokens':65536,
            'eval_at_last_step':True,'checkpoint_at_last_step':True,
            'optimizer_step_count':args.steps,'config_path':str(config_path)}
        atomic(args.out/'case_report.json',status)
        sys.argv=[str(args.repo/'scripts/train.py'),*cli]
        print('Operational qualification: unchanged reference, numerical/identity guards, real eval/checkpoint.',flush=True)
        runpy.run_path(str(args.repo/'scripts/train.py'),run_name='__main__')
        ckpt=args.out/'run/last.pt'
        if not ckpt.is_file() or ckpt.stat().st_size<500_000_000:
            raise RuntimeError('Missing full checkpoint')
        rows=[json.loads(line) for line in (args.out/'run/metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        vals=[r for r in rows if r.get('event')=='val']
        saves=[r for r in rows if r.get('event')=='checkpoint']
        if len(vals)!=1 or len(saves)!=1:
            raise RuntimeError(f'Expected exactly 1 validation and 1 checkpoint; got val={len(vals)} checkpoint={len(saves)}')
        if args.resume and not status.get('restore_audit',{}).get('passed'):
            raise RuntimeError('No exact checkpoint restoration audit')
        if not status.get('numerical_audit',{}).get('passed'):
            raise RuntimeError('Numerical audit missing or failed')
        status['observation']={
            'eval_seconds':vals[0].get('eval_seconds'),
            'val_loss':vals[0].get('val_loss'),
            'checkpoint_seconds':saves[0].get('checkpoint_seconds'),
            'checkpoint_bytes':ckpt.stat().st_size,
            'checkpoint_path':str(ckpt),
            'allocator_reserved_peak_bytes':max((r.get('memory') or {}).get('cuda_peak_reserved_bytes',0) or 0 for r in rows),
            'allocator_allocated_peak_bytes':max((r.get('memory') or {}).get('cuda_peak_allocated_bytes',0) or 0 for r in rows),
        }
        if not isinstance(status['observation']['eval_seconds'],(int,float)) or not isinstance(status['observation']['checkpoint_seconds'],(int,float)):
            raise RuntimeError('Missing eval/checkpoint timings')
        status['status']='success'
        return 0
    except BaseException as error:
        status['status']='failed';status['error']=type(error).__name__+': '+str(error)
        status['traceback']=traceback.format_exc()
        print(status['traceback'],file=sys.stderr,flush=True)
        return 1
    finally:
        status['wall_seconds']=round(time.monotonic()-started,3)
        status['completed_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        atomic(args.out/'case_report.json',status)


if __name__=='__main__':
    raise SystemExit(main())
