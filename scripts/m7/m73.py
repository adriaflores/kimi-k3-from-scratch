#!/usr/bin/env python3
"""M7.3 operational helpers. No training API is modified; GPU use only in 'preflight'."""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/kimi')
REFERENCE = ROOT / 'reference'
DATA = ROOT / 'data/full'
RECIPE = ROOT / 'experiments/m7/current/pilot_80m.yaml'
BUNDLE = ROOT / 'experiments/m7/m73'
M73_ROOT = ROOT / 'artifacts/m7/m73'
REQUIRED_YAML_SHA = 'c7c84909ce4f8e759f11ca07b607fad18c387aca905659126a6274b7e4027835'
REQUIRED_CONTRACT_SHA = '24b8a04f5a5d168d302d1ceb6b69b89a149284a07145f23addf2066be667e4be'
REQUIRED_BASELINE_SHA = '548c8f1b592fefb40173041555dd08ef8311443acf003cd9834ccd750cc3e4f0'
META_FILE_SHA = '15d7fedd033e919aceb76f79b002935bd7cb2e0dcee3ec598f46b5f42fd2805f'
META_CANON_SHA = '06709535cc3067f61fc1b76847f7d4b3798dbfe3bb5b4cde3dc1755832fd98a9'
PILOT_STEPS = 2442
TOKENS_PER_STEP = 32768
WARMUP_STEP = 1832
EXPECTED_EVENTS = list(range(128, 2443, 128)) + [2442]
EXPECTED_SOURCE_PATHS = ['scripts/train.py', 'src/kimi_k3/training/trainer.py',
                         'src/kimi_k3/training/run.py', 'src/kimi_k3/training/metrics.py',
                         'src/kimi_k3/training/monitoring.py',
                         'configs/model/kimi_k3_300m.yaml']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temp, p)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def invariant_config(config, baseline):
    allowed = {'max_steps', 'log_interval', 'eval_interval', 'eval_batches', 'checkpoint_interval'}
    if set(config) != set(baseline):
        raise ValueError(f'Config keys differ from M6: {set(config) ^ set(baseline)}')
    changes = {k for k in config if config[k] != baseline[k]}
    if changes != allowed:
        raise ValueError(f'M6-to-M7.3 changed keys {sorted(changes)} vs allowed {sorted(allowed)}')
    expected = {'batch_size': 32, 'seq_len': 1024, 'gradient_accumulation_steps': 1,
                'max_steps': PILOT_STEPS, 'learning_rate': 0.003, 'warmup_steps': WARMUP_STEP,
                'decay_steps': 183106, 'checkpoint_interval': WARMUP_STEP,
                'eval_interval': 128, 'eval_batches': 8, 'log_interval': 32,
                'adam_eps': 1e-8, 'precision': 'bf16', 'adam_fused': True,
                'muon_compile': True, 'muon_batched': True, 'muon_ns_dtype': 'bf16',
                'seed': 1337, 'overfit_batch': False}
    for k, v in expected.items():
        if config.get(k) != v or type(config.get(k)) is not type(v):
            # Float equality across YAML and JSON is intentional for the immutable recipe.
            raise ValueError(f'Invalid recipe {k}: {config.get(k)!r} expected {v!r}')
    return sorted(changes)


def validate_recipe(recipe, bundle, remote=False):
    import yaml
    if sha(recipe) != REQUIRED_YAML_SHA:
        raise ValueError('Pilot YAML byte hash differs from approved M7.2 YAML')
    if sha(bundle / 'pilot_80m.yaml') != REQUIRED_YAML_SHA:
        raise ValueError('Bundled YAML differs from approved M7.2 YAML')
    if sha(bundle / 'pilot_contract.json') != REQUIRED_CONTRACT_SHA:
        raise ValueError('M7.2 pilot contract altered')
    if sha(bundle / 'm6_baseline_train_config.json') != REQUIRED_BASELINE_SHA:
        raise ValueError('M6 baseline comparison configuration altered')
    contract = load(bundle / 'pilot_contract.json')
    if (contract.get('pilot_steps'), contract.get('pilot_nominal_tokens'),
        contract.get('warmup_steps'), contract.get('cosine_decay_end_steps')) != (2442,80019456,1832,183106):
        raise ValueError('Unexpected M7.2 contract arithmetic')
    config = yaml.safe_load(recipe.read_text(encoding='utf-8'))
    baseline = load(bundle / 'm6_baseline_train_config.json')
    changes = invariant_config(config, baseline)
    schedule = {}
    if remote:
        sys.path.insert(0, str(REFERENCE))
        from dataclasses import asdict
        from src.kimi_k3.training.trainer import TrainConfig, learning_rate_at_step
        from src.kimi_k3.utils import load_config
        cfg = load_config(recipe, TrainConfig)
        if asdict(cfg) != config:
            raise ValueError('TrainConfig normalization differs from YAML')
        expected = {1: 1.6375545851528385e-06, 1832: 0.003,
                    1833: 0.0029999999997972634,
                    2442: 0.002999924562357562, 183106: 0.00030000000000000003}
        schedule = {str(k): learning_rate_at_step(cfg, k) for k in expected}
        for k, v in expected.items():
            if not math.isclose(schedule[str(k)], v, abs_tol=1e-12, rel_tol=1e-11):
                raise ValueError(f'Unexpected actual trainer scheduler at {k}')
    return {'recipe_sha256': sha(recipe), 'changes_from_m6': changes, 'scheduler': schedule,
            'steps': config['max_steps'], 'tokens': config['max_steps'] * TOKENS_PER_STEP}


def source_manifest():
    # The M7.2 gate already matched this M6 reference against its snapshot.
    # Capture full reference Python/model configs now to detect any M7.3 drift.
    paths = sorted(set([p for p in REFERENCE.rglob('*.py') if '.venv' not in p.parts] +
                       [p for p in (REFERENCE / 'configs').rglob('*.yaml')]))
    records = {p.relative_to(REFERENCE).as_posix(): sha(p) for p in paths}
    for p in EXPECTED_SOURCE_PATHS:
        if p not in records:
            raise ValueError(f'Missing M6 reference source {p}')
    return records


def preflight(args):
    path = Path(args.output)
    out = {'phase': 'm73_preflight', 'at_utc': utc(), 'status': 'failed', 'checks': {}}
    try:
        if args.local:
            # Local test: no CUDA imports, dataset I/O or remote paths.
            import yaml
            b = Path(args.bundle)
            out['checks']['recipe'] = validate_recipe(b / 'pilot_80m.yaml', b, remote=False)
        else:
            if not RECIPE.is_file() or not DATA.is_dir():
                raise FileNotFoundError('M7.2 approved current recipe or full dataset missing')
            out['checks']['recipe'] = validate_recipe(RECIPE, BUNDLE, remote=True)
            import torch
            if platform.python_version() != '3.12.13' or str(torch.__version__) != '2.11.0+cu128' or torch.version.cuda != '12.8':
                raise ValueError('Pinned runtime differs from M7.2')
            if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA H100 NVL':
                raise ValueError('Expected H100 NVL GPU is unavailable')
            if torch.get_num_threads() != 8:
                raise ValueError(f'Expected torch threads=8, got {torch.get_num_threads()}')
            numerical = {
                'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                'matmul_precision': torch.get_float32_matmul_precision(),
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'deterministic': torch.are_deterministic_algorithms_enabled(),
            }
            if numerical != {'tf32_matmul': False, 'matmul_precision': 'highest',
                             'cudnn_allow_tf32': True, 'cudnn_benchmark': False,
                             'cudnn_deterministic': False, 'deterministic': False}:
                raise ValueError(f'Numerical profile differs from M6: {numerical}')
            out['checks']['runtime'] = {'python': platform.python_version(), 'torch': str(torch.__version__),
                                         'gpu': torch.cuda.get_device_name(0), 'numerical': numerical}
            used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True).splitlines()
            if int(used[0].strip()) > 2048:
                raise ValueError(f'GPU already occupied: {used[0]} MiB')
            free = shutil.disk_usage(ROOT).free
            if free < 15 * 1024 ** 3:
                raise ValueError(f'Insufficient free workspace disk: {free} bytes, require 15 GiB')
            out['checks']['free_bytes'] = free
            if sha(DATA / 'metadata.json') != META_FILE_SHA:
                raise ValueError('Full corpus metadata byte hash mismatch')
            from src.kimi_k3.data import ShardedTokenDataset
            totals = {}
            for split, exp_tokens, exp_shards in [('train', 6000001675, 241), ('val', 316623851, 13)]:
                ds = ShardedTokenDataset(DATA, split=split)
                total = ds.total_tokens
                count = len(list(DATA.glob(split + '_*.npy')))
                if total != exp_tokens or count != exp_shards or ds.metadata_sha256 != META_CANON_SHA:
                    raise ValueError(f'Corpus {split} mismatch: {total} tokens, {count} shards, {ds.metadata_sha256}')
                totals[split] = {'tokens': total, 'shards': count}
            out['checks']['dataset'] = totals
            src = source_manifest()
            dump(path.with_name('source_manifest.json'), src)
            out['checks']['source_files'] = len(src)
            out['checks']['source_manifest_sha256'] = sha(path.with_name('source_manifest.json'))
            out['checks']['training_started'] = False
            if args.run_dir and Path(args.run_dir).exists():
                raise FileExistsError(f'Run directory must not pre-exist: {args.run_dir}')
        out['status'] = 'passed'
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
    dump(path, out)
    print(json.dumps(out, indent=2))
    return 0 if out['status'] == 'passed' else 2


def atomic_snapshot(src, dest, *, min_bytes=100_000_000):
    """Copy one opened inode; avoid partial destination even if source is renamed."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Never overwrite preserved checkpoint: {dest}')
    temporary = dest.with_name(dest.name + '.partial')
    if temporary.exists():
        raise FileExistsError(f'Partial exists; manual inspection required: {temporary}')
    h = hashlib.sha256()
    n = 0
    try:
        with src.open('rb') as in_file, temporary.open('xb') as out_file:
            before = os.fstat(in_file.fileno())
            if before.st_size < min_bytes:
                raise ValueError('Checkpoint unexpectedly small')
            while True:
                chunk = in_file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                h.update(chunk)
                n += len(chunk)
            out_file.flush()
            os.fsync(out_file.fileno())
            after = os.fstat(in_file.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError('Source checkpoint mutated during copy')
        if n != before.st_size:
            raise RuntimeError('Source size changed during copy')
        os.replace(temporary, dest)
        return {'file': str(dest), 'bytes': n, 'sha256': h.hexdigest(), 'copied_at_utc': utc()}
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def inspect_checkpoint(path, expected_step):
    import torch
    # mmap avoids eagerly materializing ~2.5GB of tensor storage in CPU memory.
    obj = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
    if obj.get('format_version') != 7 or obj.get('step') != expected_step:
        raise ValueError(f'Checkpoint header does not match step={expected_step}')
    if obj.get('tokens_seen') != expected_step * TOKENS_PER_STEP or obj.get('skipped_updates') != 0:
        raise ValueError('Checkpoint token count / skips mismatch')
    c = obj.get('train_config', {})
    if (c.get('max_steps'), c.get('warmup_steps'), c.get('decay_steps'), c.get('learning_rate')) != (2442, 1832, 183106, 0.003):
        raise ValueError('Checkpoint training recipe inconsistent with M7.2')
    for key in ('model', 'optimizer', 'optimizer_layout', 'loaders', 'rng_state', 'runtime', 'qk_clip'):
        if key not in obj:
            raise ValueError(f'Checkpoint missing resume-required key: {key}')
    return {'format_version': 7, 'step': expected_step,
            'tokens_seen': obj['tokens_seen'], 'skipped_updates': obj['skipped_updates'],
            'resume_state_keys_present': True}


def watch(args):
    run_dir, logdir = Path(args.run_dir), Path(args.log_dir)
    metric_path = run_dir / 'metrics.jsonl'
    stop_path = logdir / 'stop_watcher'
    dest = logdir / 'warmup_step1832.pt'
    pos = 0
    deadline = time.monotonic() + 6 * 3600
    handled = False
    print('Watcher armed; waiting for committed checkpoint step 1832', flush=True)
    while time.monotonic() < deadline:
        if metric_path.is_file():
            with metric_path.open('rb') as f:
                f.seek(pos)
                while True:
                    line = f.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    pos = f.tell()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get('event') == 'checkpoint' and event.get('step') == WARMUP_STEP and not handled:
                        receipt = atomic_snapshot(run_dir / 'last.pt', dest)
                        receipt.update(inspect_checkpoint(dest, WARMUP_STEP))
                        dump(logdir / 'warmup_receipt.json', receipt)
                        print('Warmup snapshot preserved and inspected: ' + json.dumps(receipt), flush=True)
                        handled = True
                    if event.get('event') == 'checkpoint' and event.get('step') == PILOT_STEPS:
                        print('Final checkpoint event detected; watcher exits', flush=True)
                        return 0
        if stop_path.exists():
            print('Stop signal detected; watcher exits', flush=True)
            return 0
        time.sleep(1)
    raise TimeoutError('Watcher exceeded 6 hours')


def metrics_events(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summary(args):
    logdir, run_dir = Path(args.log_dir), Path(args.run_dir)
    report = {'phase': 'm73_training_end', 'at_utc': utc(), 'status': 'failed', 'run_id': logdir.name}
    try:
        events = list(metrics_events(run_dir / 'metrics.jsonl'))
        start = [x for x in events if x.get('event') == 'session_start']
        end = [x for x in events if x.get('event') == 'session_end']
        val = [x for x in events if x.get('event') == 'val']
        train = [x for x in events if x.get('event') == 'train']
        ck = [x for x in events if x.get('event') == 'checkpoint']
        if len(start) != 1 or len(end) != 1 or start[0]['step'] != 0 or start[0]['tokens_seen'] != 0:
            raise ValueError('Training not confirmed as single fresh-from-scratch run')
        if [(x['step']) for x in val] != EXPECTED_EVENTS:
            raise ValueError('Expected 20 validation records through final step')
        if [x['step'] for x in ck] != [WARMUP_STEP, PILOT_STEPS]:
            raise ValueError('Expected warmup and final checkpoint records')
        if end[0]['step'] != PILOT_STEPS or end[0]['tokens_seen'] != PILOT_STEPS * TOKENS_PER_STEP:
            raise ValueError('Incorrect final step/tokens')
        if end[0].get('skipped_updates') != 0:
            raise ValueError('Unexpected skipped updates')
        for rec in val + train:
            key = 'val_loss' if rec['event'] == 'val' else 'train_loss'
            if not isinstance(rec.get(key), (int, float)) or not math.isfinite(rec[key]):
                raise ValueError('Nonfinite or missing loss')
        manifest = load(run_dir / 'run.json')
        s = manifest['sessions'][0]
        if len(manifest['sessions']) != 1 or s['resume_checkpoint'] is not None or s['rng_source'] != 'seed' or s['start_state']['step'] != 0:
            raise ValueError('Resume/initialization invariant violated')
        if s['data']['train']['metadata_sha256'] != META_CANON_SHA:
            raise ValueError('Wrong training dataset')
        if source_manifest() != load(logdir / 'source_manifest.json'):
            raise ValueError('M6 source code drifted during M7.3')
        checkpoint = run_dir / 'last.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError('Final checkpoint missing')
        header = inspect_checkpoint(checkpoint, PILOT_STEPS)
        receipt = {'file': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                   'sha256': sha(checkpoint), 'inspected': header, 'verified_at_utc': utc()}
        dump(logdir / 'final_receipt.json', receipt)
        report.update({'status': 'training_complete_pending_external_backup',
                       'step': PILOT_STEPS, 'tokens_seen': end[0]['tokens_seen'],
                       'validation_events': len(val), 'checkpoint_events': len(ck),
                       'train_loss_first': train[0]['train_loss'], 'train_loss_last': train[-1]['train_loss'],
                       'val_loss_first': val[0]['val_loss'], 'val_loss_last': val[-1]['val_loss'],
                       'warmup_preserved': (logdir / 'warmup_receipt.json').is_file(),
                       'final_checkpoint': receipt})
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
    dump(logdir / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'training_complete_pending_external_backup' else 3


def package(args):
    logdir, run_dir = Path(args.log_dir), Path(args.run_dir)
    result = logdir / 'm73_results.zip'
    if result.exists():
        raise FileExistsError('Evidence ZIP already exists; do not overwrite')
    names = ['preflight.json', 'source_manifest.json', 'current.json', 'summary.json',
             'final_receipt.json', 'warmup_receipt.json', 'training.log',
             'watcher.log', 'gpu.csv', 'runner_exit_code.txt', 'runner_state.json', 'summary.log']
    with zipfile.ZipFile(result, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in names:
            f = logdir / name
            if f.is_file():
                z.write(f, name)
        for name in ['run.json', 'metrics.jsonl']:
            f = run_dir / name
            if f.is_file():
                z.write(f, 'trainer/' + name)
        for name in ['pilot_80m.yaml', 'pilot_contract.json', 'm6_baseline_train_config.json']:
            z.write(BUNDLE / name, 'recipe/' + name)
    (logdir / 'm73_results.zip.sha256').write_text(f'{sha(result)}  {result.name}\n', encoding='utf-8')
    print(f'Evidence ZIP: {result} ({result.stat().st_size} bytes, SHA256 {sha(result)})')


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('preflight')
    p.add_argument('--output', required=True)
    p.add_argument('--run-dir')
    p.add_argument('--local', action='store_true')
    p.add_argument('--bundle', default=str(BUNDLE))
    p.set_defaults(fn=preflight)
    p = subs.add_parser('watch')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--log-dir', required=True)
    p.set_defaults(fn=watch)
    p = subs.add_parser('summary')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--log-dir', required=True)
    p.set_defaults(fn=summary)
    p = subs.add_parser('package')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--log-dir', required=True)
    p.set_defaults(fn=package)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
