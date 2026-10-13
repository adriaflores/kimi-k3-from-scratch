#!/usr/bin/env python3
"""One isolated *systems-only* microbatch benchmark against the verified M5 source.

Deliberately bypasses the qualification launcher's 2048-token recipe guard,
not its source/dependency/numerical identity guards. The model source is unchanged.
No checkpoints are written; this is not an approved training run.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--steps', type=int, default=320)
    ap.add_argument('--burn', type=int, default=200)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config_path = args.control / 'configs/training/milestone5_branch5/t1024.yaml'
    assert args.batch in (8, 16, 24, 32, 40), 'Only sweep v2 batches are allowed'
    assert args.steps == 320 and args.burn == 200, "Sweep v2 uses the same 320/200 windows as sweep v1"
    assert config_path.is_file() and (args.repo / 'scripts/train.py').is_file()
    assert (args.data / 'metadata.json').exists() or any(args.data.glob('*.json')), 'Missing qualification data metadata'
    sys.path.insert(0, str((args.control / 'scripts/milestone5').resolve()))
    from environment_common import apply_numerical_profile, dependency_identity, source_identity
    import torch
    src = source_identity(args.repo, scope='all')
    dep = dependency_identity()
    if not src['ok'] or not dep['ok']:
        raise RuntimeError(f'Source/dependency mismatch: source={src["ok"]} runtime={dep["ok"]}')
    flags = apply_numerical_profile(torch)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU')
    gpu = torch.cuda.get_device_properties(0)
    if gpu.major != 9 or gpu.minor != 0:
        raise RuntimeError(f'Expected Hopper SM90; got SM{gpu.major}{gpu.minor} ({gpu.name})')
    # Keep learning-rate schedule approximately fixed in *tokens*, not steps,
    # while explicitly altering the effective batch ONLY for system benchmarks.
    tokens_update = args.batch * 1024
    warmup_steps = math.ceil(60_000_000 / tokens_update)
    decay_steps = math.ceil(6_000_001_024 / tokens_update)
    metadata = dict(batch=args.batch, seq_len=1024, accumulation=1,
                    tokens_update=tokens_update, burn_steps=args.burn,
                    steps=args.steps, warmup_steps=warmup_steps,
                    decay_steps=decay_steps, no_profiler=True,
                    checkpoint_disabled=True, systems_only=True,
                    source_verified=src['checked'], dependency_verified=dep['matched'],
                    source_ok=src['ok'], dependency_ok=dep['ok'], numerical_flags=flags,
                    gpu={'name':gpu.name, 'sm':[gpu.major,gpu.minor],
                         'total_memory_bytes':gpu.total_memory},
                    timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(args.out / 'experiment.json', metadata)
    os.chdir(args.repo.resolve())
    sys.path.insert(0, str(args.repo.resolve()))
    import src.kimi_k3.training.trainer as trainer
    original_train = trainer.train
    def no_checkpoint_train(model, optimizer, train_loader, val_loader, config, device, **kwargs):
        kwargs['checkpoint_path'] = None  # only disable expensive final save
        return original_train(model, optimizer, train_loader, val_loader, config, device, **kwargs)
    trainer.train = no_checkpoint_train
    cli = [
        '--data-dir', str(args.data.resolve()),
        '--model-config', str((args.repo / 'configs/model/kimi_k3_300m.yaml').resolve()),
        '--training-config', str(config_path.resolve()),
        '--device', 'cuda:0', '--run-dir', str((args.out / 'run').resolve()),
        '--batch-size',str(args.batch), '--seq-len','1024',
        '--gradient-accumulation-steps','1',
        '--max-steps',str(args.steps), '--log-interval','20',
        '--eval-batches','1', '--eval-interval','1000000000',
        '--checkpoint-interval','1000000000',
        '--warmup-steps',str(warmup_steps), '--decay-steps',str(decay_steps),
    ]
    sys.argv = [str(args.repo / 'scripts/train.py'), *cli]
    print('SYSTEMS BENCHMARK, NOT PRETRAINING. Model source unchanged; no checkpoint.',flush=True)
    print(f'Batch {args.batch} × 1024; tokens/update={tokens_update}; burn={args.burn}',flush=True)
    runpy.run_path(str(args.repo / 'scripts/train.py'), run_name='__main__')
    rows = []
    with (args.out / 'run/metrics.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['event'] == 'train':
                rows.append(row)
    good = [r for r in rows if r['step'] > args.burn and r['step'] <= args.steps]
    if len(good) < 4:
        raise RuntimeError(f'Only {len(good)} post-burn windows; expected >=4')
    skipped = max(r['skipped_updates'] for r in rows)
    nonfinite_norms = sum(r.get('nonfinite_grad_norms',0) for r in rows)
    if skipped or nonfinite_norms:
        raise RuntimeError(f'Skipped updates {skipped}, nonfinite grad_norms {nonfinite_norms}')
    elapsed = sum(float(r['train_seconds']) for r in good)
    tokens = sum(int(r['tokens']) for r in good)
    rates = [float(r['train_tokens_per_second']) for r in good]
    memory = [r.get('memory') or {} for r in rows]
    summary = {**metadata,
        'status':'success',
        'post_burn_windows':len(good),
        'post_burn_elapsed_seconds':elapsed,
        'post_burn_tokens':tokens,
        'throughput_tokens_per_second':tokens / elapsed,
        'window_rates':rates,
        'min_window_rate':min(rates),
        'max_window_rate':max(rates),
        'window_ratio_max_min':max(rates)/min(rates),
        'skipped_updates':skipped,
        'nonfinite_grad_norm_windows':nonfinite_norms,
        'peak_cuda_reserved_bytes':max((m.get('cuda_peak_reserved_bytes') or 0 for m in memory),default=0),
        'peak_cuda_allocated_bytes':max((m.get('cuda_peak_allocated_bytes') or 0 for m in memory),default=0),
        'baseline_t1024_rate':12023.30,
        'previous_sweep_batch8_rate':39064.30583842132,
        'speedup_vs_historical':(tokens / elapsed)/12023.30,
        'compute_only_6b_hours':6_000_001_024/(tokens/elapsed)/3600,
        'note':'Throughput measured without profiler; excludes evaluation, checkpoints, upload, setup, taxes and storage.',
    }
    write_json(args.out / 'summary.json',summary)
    print('RESULT',json.dumps({k:summary[k] for k in ('batch','throughput_tokens_per_second','speedup_vs_historical','compute_only_6b_hours','window_ratio_max_min')}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
