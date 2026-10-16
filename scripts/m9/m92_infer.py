"""M9.2 / read-only full-checkpoint inference. Run from repo root with uv run.

Never restores an optimizer, loader cursors, or RNG from the training run.
Original fused backend is opt-in. The default reference PyTorch backend is
not claimed numerically identical until a separate fused parity test.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.m9.m92_core import (
    EXPECTED_CKPT_BYTES, EXPECTED_CKPT_SHA, EXPECTED_PARAMS,
    EXPECTED_STEP, EXPECTED_TOKENS, MODEL_CONFIG_HASH,
    atomic_json, generate, last_logits, require_baseline_metadata,
    sha256_file, verify_contract,
)

DEFAULT_CHECKPOINT = Path('C:/datasets/kimi_k3_backups/m8/m8_full_6b_v1/snapshot_step_183106.pt')
DEFAULT_CONTRACT = REPO / 'scripts/m9/reference/m91_evaluation_contract_frozen.json'
DEFAULT_OUTPUT = REPO / 'results/m9/m92_inference'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='M9.2 checkpoint inference/causality smoke (read only)')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--backend', choices=('chunkwise', 'fused'), default='chunkwise')
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32',
                   help='FP32 is the validated M9.2 causality reference; BF16 is experimental (v3 failed suffix-invariance)')
    p.add_argument('--verify-sha', action='store_true', help='Rehash ~2.5 GB; already verified M9.1')
    p.add_argument('--prompt', default='Once upon a time, in a small village')
    p.add_argument('--max-new-tokens', type=int, default=4)
    p.add_argument('--mode', choices=('greedy', 'sample'), default='greedy')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--local-files-only', action='store_true', help='Do not contact Hugging Face for tokenizer')
    p.add_argument('--skip-tokenizer', action='store_true', help='Only synthetic-token smoke (not a text-quality result)')
    p.add_argument('--preflight-only', action='store_true', help='No torch import/checkpoint deserialization')
    return p.parse_args(argv)


def make_report_markdown(result):
    lines = [
        '# M9.2 — Local inference report', '',
        f"**Status:** `{result.get('status')}`", '',
        f"Checkpoint: `{result.get('checkpoint')}`",
        f"Execution device: `{result.get('device', 'not attempted')}`",
        f"KDA backend: `{result.get('backend')}`",
        '', '## Checks', '',
    ]
    for k,v in result.get('checks',{}).items():
        lines.append(f'- `{k}`: **{v}**')
    lines += ['', '## Results', '']
    for k in ('torch', 'precision', 'parameters', 'prompt_token_count',
              'first_next_token_id', 'greedy_first_id', 'generated_token_ids',
              'generated_text', 'seconds_model_load', 'seconds_forward_and_checks',
              'seconds_generation', 'max_cuda_allocated_mib',
              'suffix_max_abs_logit_delta', 'suffix_argmax_changed'):
        if k in result:
            lines.append(f'- **{k}:** `{result[k]}`')
    lines.extend(['', '## Interpretation', ''])
    lines.append('FP32 is the reference precision for the M9.2 causal suffix-invariance gate. '
                 'The BF16 run on 2026-10-15 showed max abs logit delta 0.078125, '
                 'while its FP32 diagnostic showed max delta ~8.1e-6. '
                 'This indicates numerical sensitivity; the particular operator causing it is not proven. '
                 'Do not interpret a FP32 PASS as a BF16 PASS. ')
    lines.append('A PASS_LOCAL_REFERENCE means that checkpoint loading, finite logits, '
                 'repeatable forward, causal suffix-invariance, and generation passed '
                 'on the PyTorch chunkwise backend. It does NOT establish numeric parity '
                 'with the fused KDA used during training, nor benchmark capability.')
    if result.get('backend') == 'fused':
        lines.append('Running the fused backend alone still does not prove '
                     'full-backend parity: paired measurements are required.')
    if result.get('causality_diagnostics'):
        lines.extend(['', '## Causality diagnostics', '',
                      'In each precision, compare first-token-to-prefix representations after',
                      'changing only tokens strictly beyond the prediction position.',
                      'These comparisons diagnose numerical variation; they do not permit',
                      'automatically waiving a causal-invariance failure.', ''])
        for prec, item in result['causality_diagnostics'].items():
            lines += [f'### {prec}', '', f"- Diagnostic status: `{item.get('status')}`",
                      f"- First module over tolerance: `{item.get('first_module_over_tolerance')}`",
                      f"- Last logits max difference: `{item.get('last_logit_max_abs')}`",
                      f"- Different logits: `{item.get('last_logit_elements_above_tolerance')}`",
                      f"- Argmax IDs: `{item.get('last_argmax_original')}`, `{item.get('last_argmax_changed_suffix')}`", '']
            if item.get('exception'):
                lines += [f"- Error: `{item['exception']}`", '']
            else:
                lines += ['| Module | Max abs delta | Elements over tolerance |',
                          '|---|---:|---:|']
                lines += [f"| {m['module']} | {m['max_abs']:.6g} | {m['elements_above_tolerance']} / {m['elements_total']} |" for m in item['modules']]
                lines += ['']
    if result.get('error'):
        lines += ['', '## Error', '', '```', str(result['error']), '```']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    args = parse_args(argv)
    report = {
        'schema_version': 'm92_inference_1',
        'overlay_revision': 'm92_v4_fp32_reference_gate',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FAIL',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256_m91_verified': EXPECTED_CKPT_SHA,
        'contract': str(args.contract),
        'backend': args.backend,
        'checks': {},
        'inference_is_not_a_benchmark': True,
        'fused_reference_parity': 'NOT_ESTABLISHED',
        'source_provenance': 'M9.1: 32/35 frozen source entries matched; 3 historical YAML absent',
    }
    exit_code = 1
    start = time.perf_counter()
    try:
        if not (1 <= args.max_new_tokens <= 64):
            raise ValueError('max_new_tokens must be in [1, 64] for M9.2 generation smoke')
        contract = json.loads(args.contract.read_text(encoding='utf-8'))
        original_config = verify_contract(contract)
        report['checks']['frozen_contract'] = 'PASS'
        if args.checkpoint.stat().st_size != EXPECTED_CKPT_BYTES:
            raise ValueError('Checkpoint file size differs from verified M9.1 baseline')
        report['checks']['checkpoint_file_size'] = 'PASS'
        if args.verify_sha:
            if sha256_file(args.checkpoint) != EXPECTED_CKPT_SHA:
                raise ValueError('Checkpoint SHA mismatch')
            report['checks']['checkpoint_sha256_recomputed'] = 'PASS'
        else:
            report['checks']['checkpoint_sha256_recomputed'] = 'REUSED_M91_PASS'
        if args.preflight_only:
            report['status'] = 'PASS_PREFLIGHT_ONLY'
            exit_code = 0
        else:
            import torch
            report['torch'] = str(torch.__version__)
            report['python'] = platform.python_version()
            torch.manual_seed(args.seed)
            if args.device == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(args.device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable')
            if args.backend == 'fused' and device.type != 'cuda':
                raise ValueError('Fused KDA requires CUDA')
            report['device'] = str(device)
            report['device_name'] = torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()
            precision = args.precision
            if precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
                raise ValueError('BF16 requires a CUDA device supporting BF16')
            report['precision'] = precision
            report['precision_policy'] = ('FP32_REFERENCE' if precision == 'fp32' else 'BF16_EXPERIMENTAL')
            # Avoid accidentally touching optimizer, restoring RNG, or building the trainer.
            t_load = time.perf_counter()
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
            cfg = require_baseline_metadata(checkpoint, contract)
            state = checkpoint['model']
            report['checks']['checkpoint_metadata'] = 'PASS'
            report['model_config_sha256'] = MODEL_CONFIG_HASH
            from src.kimi_k3.model import KimiK3, KimiK3Config
            with torch.device('meta'):
                model = KimiK3(KimiK3Config(**{**cfg, 'kda_implementation': args.backend}))
            n_params = sum(p.numel() for p in model.parameters())
            if n_params != EXPECTED_PARAMS:
                raise ValueError(f'Unexpected parameter count: {n_params}')
            report['parameters'] = n_params
            report['checks']['parameters'] = 'PASS'
            # Strict names and shapes. assign=True materializes directly from mmap CPU storage.
            model.load_state_dict(state, strict=True, assign=True)
            if any(p.is_meta for p in model.parameters()):
                raise ValueError('Model still contains meta parameters')
            # Assign loading can break embedding/head aliasing: restore the training tie.
            model.lm_head.weight = model.embed_tokens.weight
            model.eval()
            if device.type == 'cuda':
                model = model.to(device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                model = model.to('cpu')
            # Don't retain references to optimizer buffers after strict loading.
            del state, checkpoint
            report['checks']['model_weights_loaded_strict'] = 'PASS'
            report['checks']['tied_embedding_head'] = ('PASS' if model.lm_head.weight is model.embed_tokens.weight
                                                      else 'FAIL')
            if report['checks']['tied_embedding_head'] != 'PASS':
                raise ValueError('Embedding/head weight tying lost')
            report['seconds_model_load'] = round(time.perf_counter()-t_load, 3)
            t_forward = time.perf_counter()
            # Always execute a tokenizer-independent test with 17 real tokens.
            synthetic = [17, 44, 192, 291, 82, 617, 93, 8, 765, 67, 43, 9, 21, 92, 187, 314, 888]
            logits_a = last_logits(model, synthetic, device=device, precision=precision, torch_module=torch)
            logits_b = last_logits(model, synthetic, device=device, precision=precision, torch_module=torch)
            delta_repeat = float((logits_a-logits_b).abs().max().item())
            report['repeat_max_abs_logit_delta'] = delta_repeat
            if not torch.allclose(logits_a, logits_b, atol=1e-3, rtol=1e-3):
                raise ValueError('Repeated forward is not deterministic within tolerance')
            report['checks']['repeatable_finite_logits'] = 'PASS'
            # Change tokens strictly AFTER the last observed position.
            # Different suffix cannot affect earlier logits in a causal LM.
            logits_alt = last_logits(model, synthetic, device=device, precision=precision,
                                     torch_module=torch, suffix_token=151)
            delta_suffix = float((logits_a-logits_alt).abs().max().item())
            report['suffix_max_abs_logit_delta'] = delta_suffix
            report['suffix_argmax_changed'] = int(torch.argmax(logits_a)) != int(torch.argmax(logits_alt))
            report['suffix_logits_above_tolerance'] = int(((logits_a-logits_alt).abs() >
                (5e-3 + 1e-3 * logits_alt.abs())).sum().item())
            if not torch.allclose(logits_a, logits_alt, atol=5e-3, rtol=1e-3):
                report['checks']['causal_suffix_invariance'] = 'FAIL_UNDER_INVESTIGATION'
                from scripts.m9.m92_causality import suffix_diagnosis
                from scripts.m9.m92_core import inference_context
                report['causality_diagnostics'] = suffix_diagnosis(
                    model, synthetic, device=device, precision=precision,
                    inference_context=inference_context)
                raise ValueError('Causal suffix-invariance differs above tolerance; per-layer traces recorded')
            report['checks']['causal_suffix_invariance'] = 'PASS'
            report['first_next_token_id'] = int(torch.argmax(logits_a).item())
            report['seconds_forward_and_checks'] = round(time.perf_counter()-t_forward, 3)
            if args.skip_tokenizer:
                report['checks']['tokenizer_identity'] = 'SKIPPED'
                prompt_ids = synthetic
                tokenizer = None
            else:
                from transformers import AutoTokenizer
                t_tok = time.perf_counter()
                tokenizer = AutoTokenizer.from_pretrained(
                    contract['tokenizer']['repository'], revision=contract['tokenizer']['revision'],
                    use_fast=True, trust_remote_code=False, local_files_only=args.local_files_only)
                report['seconds_tokenizer_load'] = round(time.perf_counter()-t_tok, 3)
                if tokenizer.vocab_size != 49152 or tokenizer.eos_token_id != 0:
                    raise ValueError('Tokenizer vocab/EOS does not match training')
                prompt_ids = list(tokenizer.encode(args.prompt, add_special_tokens=False))
                report['checks']['tokenizer_identity'] = 'PASS'
            if not prompt_ids:
                raise ValueError('Empty tokenizer result')
            report['prompt'] = args.prompt if tokenizer is not None else 'synthetic IDs (tokenizer bypassed)'
            report['prompt_token_count'] = len(prompt_ids)
            # Explicit next-token alignment check for greedy mode, not merely end-to-end text.
            step_logits = last_logits(model, prompt_ids, device=device, precision=precision, torch_module=torch)
            greedy_id = int(torch.argmax(step_logits).item())
            report['greedy_first_id'] = greedy_id
            t_gen = time.perf_counter()
            ids = generate(model, prompt_ids, device=device, torch_module=torch,
                           max_new_tokens=args.max_new_tokens, mode=args.mode,
                           temperature=args.temperature, top_k=args.top_k,
                           top_p=args.top_p, seed=args.seed, eos_id=0, precision=precision)
            if args.mode == 'greedy' and ids and ids[0] != greedy_id:
                raise ValueError('Generation first token does not match argmax of final prompt logit')
            report['checks']['next_token_alignment'] = 'PASS' if args.mode == 'greedy' else 'NOT_APPLICABLE_SAMPLE'
            report['generated_token_ids'] = ids
            report['seconds_generation'] = round(time.perf_counter()-t_gen, 3)
            report['generated_text'] = (tokenizer.decode(ids, skip_special_tokens=False)
                                        if tokenizer is not None else None)
            report['checks']['autoregressive_generation'] = 'PASS'
            if device.type == 'cuda':
                report['max_cuda_allocated_mib'] = round(torch.cuda.max_memory_allocated(device)/2**20, 2)
                report['max_cuda_reserved_mib'] = round(torch.cuda.max_memory_reserved(device)/2**20, 2)
            report['checks']['backend_numeric_parity'] = 'NOT_TESTED'
            report['status'] = 'PASS_LOCAL_REFERENCE' if args.backend == 'chunkwise' else 'PASS_FUSED_ONLY'
            if tokenizer is None:
                report['status'] = 'PASS_SYNTHETIC_ONLY'
            exit_code = 0
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc(limit=16)
        print('M9.2 FAILED:', report['error'])
        exit_code = 1
    finally:
        report['seconds_total'] = round(time.perf_counter()-start, 3)
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output / 'm92_inference.json', report)
        (args.output/'M92_READOUT.md').write_text(make_report_markdown(report),encoding='utf-8')
        print('M9.2 STATUS:', report['status'])
        for label,value in report['checks'].items():
            print(f'  {label}: {value}')
        if 'generated_text' in report and report['generated_text'] is not None:
            print('Completion:', repr(report['generated_text']))
        print('REPORT:', args.output / 'M92_READOUT.md')
        print('JSON:', args.output / 'm92_inference.json')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
