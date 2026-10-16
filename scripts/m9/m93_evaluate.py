#!/usr/bin/env python3
"""M9.3: frozen validation-stream NLL/PPL with FP32 local reference.

Run from repo root: uv run python scripts/m9/m93_evaluate.py --device cuda
No weight export, optimizer restoration, network, or write to input data.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.m9.m92_core import (
    EXPECTED_CKPT_SHA, EXPECTED_PARAMS, MODEL_CONFIG_HASH,
    canonical_hash, require_baseline_metadata, verify_contract,
)
from scripts.m9.m93_core import (
    DATA_METADATA_FILE_SHA256, FROZEN_CONTRACT_FILE_SHA256, WINDOW_COUNT,
    WINDOW_LEN, RUN_SCHEMA, check_checkpoint_identity, ensure_dataset_identity,
    file_sha256, input_target, make_run_identity, read_finished_windows,
    score_window, summarize_windows, validate_evaluation_plan,
    validate_existing_identity, write_atomic, write_window_csv,
)

DEFAULT_CKPT = Path('C:/datasets/kimi_k3_backups/m8/m8_full_6b_v1/snapshot_step_183106.pt')
DEFAULT_DATA = Path('C:/datasets/fineweb_edu_6b')
DEFAULT_CONTRACT = ROOT / 'scripts/m9/reference/m91_evaluation_contract_frozen.json'
DEFAULT_OUT = ROOT / 'results/m9/m93_quantitative'
SOURCE_MANIFEST = ROOT / 'experiments/m7/m75_gate/evidence/m73_source_manifest.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='M9.3 / frozen held-out validation NLL (FP32, no training)')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    p.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='cuda')
    p.add_argument('--verify-checkpoint-sha', action='store_true', help='Recompute 2.5GB SHA-256 (already PASS M9.1)')
    p.add_argument('--preflight-only', action='store_true', help='Check data/contract/checkpoint without model weights')
    p.add_argument('--smoke', action='store_true', help='Two pre-registered windows only, NON-BENCHMARK dry run')
    p.add_argument('--resume', action='store_true', help='Resume only if contract, device and code hashes unchanged')
    return p.parse_args(argv)


def source_hashes():
    """Verify inference implementation matches M7/M8 frozen Python source paths."""
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or len(manifest) != 35:
        raise ValueError('Wrong historical source manifest')
    source_entries = {path: digest for path, digest in manifest.items()
                      if path.startswith('src/kimi_k3/') and path.endswith('.py')}
    if len(source_entries) < 5:
        raise ValueError('Manifest lacks model implementation files')
    failures = []
    for rel, expected in source_entries.items():
        p = ROOT / rel
        if not p.is_file() or file_sha256(p) != expected:
            failures.append(rel)
    if failures:
        raise ValueError('Model source differs from production source manifest: ' + ', '.join(failures))
    # Record the evaluation source too: any change blocks resuming old windows.
    paths = ['scripts/m9/m93_core.py', 'scripts/m9/m93_evaluate.py',
             'scripts/m9/m92_core.py', 'src/kimi_k3/data/dataset.py']
    return {path: file_sha256(ROOT / path) for path in paths}


def readout(report):
    lines = ['# M9.3 — Independent validation-stream evaluation', '',
             f"**Status:** `{report['status']}`", '',
             '## Contract and results', '',
             f"- Data: `val`, total `{report.get('val_total_tokens', 'not read')}` tokens; 13 .npy shards",
             f"- Model: checkpoint SHA `{EXPECTED_CKPT_SHA}` (local M9.1 verified)",
             '- Backend/precision: KDA chunkwise, FP32 (training used fused KDA/BF16)',
             '- Windows: 64 × 1024; offsets exactly frozen in M9.1, no overlap',
             '- The first 262,144 validation tokens used for M8 monitoring are excluded',
             '- Each window resets model recurrent state; x[t] predicts x[t+1], including EOS',
             '', '## Checks', '']
    for k, v in report.get('checks', {}).items():
        lines.append(f'- `{k}`: `{v}`')
    s = report.get('summary')
    if s:
        lines.extend(['', '## Measured values', '',
                      f"- Completed windows: **{s.get('completed_windows')} / {s.get('expected_windows')}**",
                      f"- Scored tokens: **{s.get('scored_tokens', 0)}**",
                      f"- Mean NLL (nats/token): **{s.get('mean_nll_nats_per_token', '—')}**",
                      f"- Perplexity (same tokenizer): **{s.get('perplexity_same_tokenizer', '—')}**",
                      f"- Next-token top-1 accuracy: **{s.get('top1_next_token_accuracy', '—')}**"])
        if 'bootstrap' in s:
            b = s['bootstrap']
            lines.extend([f"- Approximate 95% window-bootstrap NLL interval: `{b['nll_95pct_bootstrap_approx']}`",
                          '- Bootstrap resamples windows, not independent documents: do not read this as formal confidence bounds'])
    lines.extend(['', '## Context and limitations', '',
                  '- M8 monitoring final NLL = 3.1824890673, measured on repeatedly observed fixed val prefix. '
                  '**It is not a directly matched baseline for this separately sampled evaluation.**',
                  '- No model-selection, hyperparameter-tuning, or early stopping was performed using these 64 windows.',
                  '- A PASS here validates independent predictive NLL under the FP32 chunkwise local backend. '
                  'Numeric parity with the trained fused backend has not been established.',
                  '- Different model tokenizers cannot be fairly ranked by raw token-level perplexity.',
                  '- EOS/doc boundaries follow the exact contiguous stream training convention; no cross-document resets.',
                  '- The data origin was verified in M7; M9.3 rechecks metadata SHA and shard headers, not every 12.6 GB shard SHA.',
                  '- FP32 correctness was established in M9.2; BF16 suffix sensitivity remains documented.',
                  '', f"Runtime (seconds): `{report.get('seconds_total')}`", ''])
    if report.get('error'):
        lines.extend(['## Error', '', f"`{report['error']}`", ''])
    return '\n'.join(lines)


def save_report(outdir, report):
    write_atomic(outdir / 'm93_results.json', report)
    (outdir / 'M93_READOUT.md').write_text(readout(report), encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    outdir = args.output / 'smoke' if args.smoke else args.output
    output_exists = (outdir / 'run_manifest.json').exists()
    report = {'schema_version': RUN_SCHEMA, 'overlay_revision': 'm93_v2_raw_and_canonical_metadata_gate',
              'started_at_utc': utc(), 'status': 'INCOMPLETE',
              'checks': {}, 'data_dir': str(args.data_dir), 'checkpoint': str(args.checkpoint),
              'device_requested': args.device, 'mode': 'SMOKE_NOT_BENCHMARK' if args.smoke else 'FROZEN_FULL_64',
              'fused_parity': 'NOT_ESTABLISHED', 'historical_val_prefix_nll': 3.1824890673160553}
    try:
        if file_sha256(args.contract) != FROZEN_CONTRACT_FILE_SHA256:
            raise ValueError('M9.1 contract file SHA-256 mismatch: do not edit offsets')
        contract = json.loads(args.contract.read_text(encoding='utf-8'))
        verify_contract(contract)
        offsets = validate_evaluation_plan(contract)
        report['checks']['frozen_m91_contract_and_64_offsets'] = 'PASS'
        check_checkpoint_identity(args.checkpoint)
        report['checks']['checkpoint_size_m91'] = 'PASS'
        if args.verify_checkpoint_sha:
            if file_sha256(args.checkpoint) != EXPECTED_CKPT_SHA:
                raise ValueError('Checkpoint SHA-256 no longer matches M9.1')
            report['checks']['checkpoint_sha256'] = 'PASS_REHASHED'
        else:
            report['checks']['checkpoint_sha256'] = 'REUSED_M91_PASS'
        metadata_digest = ensure_dataset_identity(args.data_dir, contract)
        report['checks']['metadata_file_sha256_m7'] = 'PASS'
        report['checks']['metadata_canonical_sha256_m91'] = 'PASS'
        report['checks']['metadata_tokenizer'] = 'PASS'
        # Dataset reader checks uint16 NPY headers for all 13 val shards and ordered offsets.
        from src.kimi_k3.data.dataset import ShardedTokenDataset
        dataset = ShardedTokenDataset(args.data_dir, split='val')
        if (dataset.total_tokens != contract['data']['val']['total_tokens'] or dataset.vocab_size != 49152
                or dataset.metadata_sha256 != contract['data']['val']['metadata_sha256']):
            raise ValueError('Dataset shape/vocab do not match contract')
        report['val_total_tokens'] = dataset.total_tokens
        report['checks']['val_shard_headers_and_token_count'] = 'PASS'
        code_hashes = source_hashes()
        report['checks']['source_implementation_matches_frozen_manifest'] = 'PASS'
        # Make sure every offset's target is present, without reading the full dataset.
        selected_ids = list(range(2)) if args.smoke else list(range(len(offsets)))
        hashes = {}
        for idx in selected_ids:
            _, _, digest = input_target(dataset, offsets[idx])
            hashes[idx] = digest
        report['checks']['window_offsets_and_token_alignment'] = 'PASS'
        if args.preflight_only:
            report['status'] = 'PASS_PREFLIGHT_ONLY'
            report['checks']['no_checkpoint_deserialization'] = 'PASS'
            return_code = 0
        else:
            import torch
            report['torch'] = str(torch.__version__)
            report['python'] = platform.python_version()
            torch.manual_seed(int(contract['seed']))
            if args.device == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(args.device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; change --device cpu explicitly')
            report['device'] = str(device)
            report['device_name'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()
            identity = make_run_identity(contract_sha=FROZEN_CONTRACT_FILE_SHA256,
                data_sha=metadata_digest, code_hashes=code_hashes, checkpoint_sha=EXPECTED_CKPT_SHA,
                offsets=offsets, device=str(device), torch_version=str(torch.__version__))
            fingerprint = canonical_hash(identity)
            report['run_identity_sha256'] = fingerprint
            manifest_file = outdir / 'run_manifest.json'
            if output_exists:
                if not args.resume:
                    raise FileExistsError(f'Existing M9.3 run; use --resume to preserve completed windows: {manifest_file}')
                old = json.loads(manifest_file.read_text(encoding='utf-8'))
                validate_existing_identity(old, identity)
            else:
                if args.resume and (outdir / 'windows').is_dir():
                    raise ValueError('Orphaned windows directory without run manifest; refusing to resume')
                if (outdir / 'windows').exists():
                    raise ValueError('Unowned windows directory exists without a run manifest')
                write_atomic(manifest_file, identity)
            windows_dir = outdir / 'windows'
            windows_dir.mkdir(parents=True, exist_ok=True)
            completed = read_finished_windows(windows_dir, identity, selected_ids)
            for idx in completed:
                if completed[idx]['input_tokens_sha256'] != hashes[idx]:
                    raise ValueError(f'Window {idx} tokens changed since previous evaluation')
            report['checks']['run_identity_and_resume_receipts'] = 'PASS'
            report['windows_reused'] = len(completed)
            if len(completed) < len(selected_ids):
                tload = time.perf_counter()
                checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu', mmap=True)
                cfg = require_baseline_metadata(checkpoint, contract)
                from src.kimi_k3.model import KimiK3, KimiK3Config
                with torch.device('meta'):
                    model = KimiK3(KimiK3Config(**{**cfg, 'kda_implementation': 'chunkwise'}))
                nparams = sum(p.numel() for p in model.parameters())
                if nparams != EXPECTED_PARAMS:
                    raise ValueError(f'Parameter count mismatch: {nparams}')
                model.load_state_dict(checkpoint['model'], strict=True, assign=True)
                model.lm_head.weight = model.embed_tokens.weight
                if model.lm_head.weight is not model.embed_tokens.weight:
                    raise ValueError('Lost model embedding/head tying')
                model.eval().to(device)
                del checkpoint
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                report['seconds_model_load'] = round(time.perf_counter()-tload, 3)
                report['checks']['strict_checkpoint_load'] = 'PASS'
                for idx in selected_ids:
                    if idx in completed:
                        continue
                    offset = offsets[idx]
                    print(f'M9.3 scoring window {idx+1}/{len(selected_ids)} (val offset {offset})', flush=True)
                    t0 = time.perf_counter()
                    x, y, digest = input_target(dataset, offset)
                    if digest != hashes[idx]:
                        raise ValueError(f'Token content modified during evaluation at offset {offset}')
                    nll_sum, correct = score_window(model, x, y, device)
                    receipt = {'schema_version': RUN_SCHEMA, 'run_identity_sha256': fingerprint,
                        'window_id': idx, 'offset': offset, 'scored_tokens': WINDOW_LEN,
                        'input_tokens_sha256': digest, 'nll_sum': nll_sum,
                        'mean_nll_nats_per_token': nll_sum/WINDOW_LEN,
                        'top1_correct': correct, 'seconds': round(time.perf_counter()-t0, 3),
                        'completed_at_utc': utc()}
                    write_atomic(windows_dir / f'window_{idx:03d}.json', receipt)
                    completed[idx] = receipt
                    print(f'  nll={nll_sum/WINDOW_LEN:.6f}  top1={correct}/{WINDOW_LEN}  '
                          f'seconds={receipt["seconds"]}', flush=True)
            ordered = [completed[i] for i in selected_ids]
            summary = summarize_windows(ordered, len(selected_ids))
            write_window_csv(outdir / 'per_window.csv', ordered)
            report['summary'] = summary
            report['checks']['all_windows_scored_once'] = ('PASS' if summary['status'] == 'PASS_EVALUATION' else 'INCOMPLETE')
            if device.type == 'cuda':
                report['peak_cuda_allocated_mib'] = round(torch.cuda.max_memory_allocated(device)/2**20, 2)
                report['peak_cuda_reserved_mib'] = round(torch.cuda.max_memory_reserved(device)/2**20, 2)
            report['status'] = 'PASS_SMOKE_NOT_BENCHMARK' if args.smoke else 'PASS_FROZEN_HELDOUT'
            return_code = 0
    except Exception as exc:
        return_code = 1
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=10)
        print('M9.3 FAILED:', report['error'], flush=True)
        # Interrupted runs retain per-window receipts; --resume safely continues.
        try:
            if (outdir / 'run_manifest.json').is_file():
                manifest = json.loads((outdir / 'run_manifest.json').read_text(encoding='utf-8'))
                selected = range(2) if args.smoke else range(WINDOW_COUNT)
                existing = read_finished_windows(outdir / 'windows', manifest, selected)
                report['summary'] = summarize_windows(list(existing.values()), 2 if args.smoke else WINDOW_COUNT)
        except Exception:
            pass
    finally:
        report['seconds_total'] = round(time.perf_counter()-started, 3)
        report['finished_at_utc'] = utc()
        # Preflight writes separately and never erases a successful evaluation.
        save_to = outdir / 'preflight' if args.preflight_only else outdir
        # Refuse to overwrite existing PASS when no --resume was specified.
        if not (output_exists and not args.resume and not args.preflight_only):
            save_report(save_to, report)
        print(f'M9.3 STATUS: {report["status"]}')
        for k, v in report['checks'].items():
            print(f'  {k}: {v}')
        if 'summary' in report:
            print('  mean_nll:', report['summary'].get('mean_nll_nats_per_token'))
            print('  perplexity:', report['summary'].get('perplexity_same_tokenizer'))
            print('  completed_windows:', report['summary'].get('completed_windows'))
        print('REPORT:', save_to / 'M93_READOUT.md')
        print('JSON:', save_to / 'm93_results.json')
    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
