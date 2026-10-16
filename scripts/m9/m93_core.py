"""M9.3 protocol, streaming reductions, and crash-safe evidence primitives.

No trainer and no network. Importing this module does not touch the checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

from scripts.m9.m92_core import EXPECTED_CKPT_SHA, canonical_hash

FROZEN_CONTRACT_FILE_SHA256 = 'd32b5fc7518977093811f38ed424ea3d3c52913fc8334188c311116abb603a64'
DATA_METADATA_FILE_SHA256 = '15d7fedd033e919aceb76f79b002935bd7cb2e0dcee3ec598f46b5f42fd2805f'
DATA_METADATA_CANONICAL_SHA256 = '06709535cc3067f61fc1b76847f7d4b3798dbfe3bb5b4cde3dc1755832fd98a9'
N_TOKENS_VAL = 316_623_851
MONITOR_PREFIX = 262_144
WINDOW_LEN = 1024
WINDOW_COUNT = 64
RUN_SCHEMA = 'm93_eval_1'


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8') + b'\n'


def write_atomic(path, payload):
    """Atomic replace in destination folder, no partial JSON on interruptions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        f.write(json_bytes(payload))
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_evaluation_plan(contract):
    """Pre-registration is immutable. Validation errors never truncate silently."""
    e = contract['evaluation']
    if (e.get('kind') != 'causal-language-model-contiguous-stream-NLL-per-token'
            or e.get('split') != 'val' or e.get('n_windows') != WINDOW_COUNT
            or e.get('window_length') != WINDOW_LEN
            or e.get('total_scored_tokens') != WINDOW_COUNT * WINDOW_LEN
            or e.get('data_total_tokens') != N_TOKENS_VAL
            or contract['data']['val']['total_tokens'] != N_TOKENS_VAL
            or contract['data']['val']['metadata_sha256'] != DATA_METADATA_CANONICAL_SHA256):
        raise ValueError('Frozen M9.1 evaluation plan/validation source differs')
    offsets = e.get('window_offsets')
    if (not isinstance(offsets, list) or len(offsets) != WINDOW_COUNT
            or any(type(o) is not int for o in offsets)
            or offsets != sorted(set(offsets))):
        raise ValueError('Evaluation window offsets invalid')
    for i, off in enumerate(offsets):
        if off < MONITOR_PREFIX + 1 or off + WINDOW_LEN + 1 > N_TOKENS_VAL:
            raise ValueError(f'Evaluation window {i} crosses monitoring prefix or corpus end')
        if i and off - offsets[i - 1] <= WINDOW_LEN:
            raise ValueError('Evaluation windows overlap on scored input/target stream')
    if (e.get('input_alignment') != 'x = tokens[offset : offset+1024]; y = tokens[offset+1 : offset+1025]; every target counted exactly once'
            or e.get('reset_state_each_window') is not True):
        raise ValueError('Input alignment / reset semantics drifted')
    return offsets


def dataset_metadata_canonical_sha256(metadata):
    """Match ShardedTokenDataset's canonical JSON hash (not file bytes)."""
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def ensure_dataset_identity(data, contract):
    """Verify BOTH original M7 raw bytes and M7 canonical metadata identity.

    The frozen M9.1 contract stores the canonical digest. The M7.2 provenance
    separately pinned the exact raw file SHA-256. These cannot be interchanged.
    Return the raw-file digest for the strict M9.3 resume identity.
    """
    path = Path(data) / 'metadata.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing prepared data: {path} (expected C:/datasets/fineweb_edu_6b)')
    file_digest = file_sha256(path)
    if file_digest != DATA_METADATA_FILE_SHA256:
        raise ValueError(f'Dataset metadata.json raw file SHA256 mismatch: {file_digest}')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    canonical_digest = dataset_metadata_canonical_sha256(metadata)
    if canonical_digest != DATA_METADATA_CANONICAL_SHA256:
        raise ValueError(f'Dataset metadata.json canonical SHA256 mismatch: {canonical_digest}')
    if canonical_digest != contract['data']['val']['metadata_sha256']:
        raise ValueError('Frozen M9.1 contract metadata canonical SHA256 mismatch')
    if metadata.get('format_version') != 2:
        raise ValueError('Unexpected dataset metadata version')
    if metadata.get('splits', {}).get('val', {}).get('tokens') != N_TOKENS_VAL:
        raise ValueError('Validation total token count drifted')
    if len(metadata['splits']['val']['shards']) != 13:
        raise ValueError('Expected 13 validation shards')
    expected_tok = contract['tokenizer']
    for key in ('repository', 'revision', 'vocab_size', 'eos_id', 'add_special_tokens', 'append_eos'):
        if metadata.get('tokenizer', {}).get(key) != expected_tok[key]:
            raise ValueError(f'Dataset tokenizer metadata differs: {key}')
    return file_digest


def check_checkpoint_identity(checkpoint):
    p = Path(checkpoint)
    if not p.is_file():
        raise FileNotFoundError(f'Missing M8 checkpoint: {p}')
    if p.stat().st_size != 2_485_700_551:
        raise ValueError('Checkpoint size differs from M9.1 verification')
    return p


def input_target(dataset, offset, length=WINDOW_LEN):
    """Deliberately use the exact trainer reader alignment, no EOS masking."""
    if type(offset) is not int or offset < 0 or offset + length + 1 > dataset.total_tokens:
        raise ValueError('Window offset out of bounds')
    x, y = dataset.get_sequence(offset, length)
    if x.shape != (length,) or y.shape != (length,):
        raise ValueError('Unexpected x/y sequence shapes')
    if not bool((x[1:] == y[:-1]).all().item()):
        raise ValueError('Causal shift misaligned')
    if not bool(((x >= 0) & (x < dataset.vocab_size) & (y >= 0) & (y < dataset.vocab_size)).all().item()):
        raise ValueError('Out-of-vocabulary tokens in dataset')
    # Stable fingerprint of all L+1 tokens including the last lookahead target.
    import numpy as np
    stream = dataset.read_tokens(offset, length + 1)
    fingerprint = hashlib.sha256(np.asarray(stream, dtype='<u2').tobytes()).hexdigest()
    return x, y, fingerprint


def score_window(model, x, y, device, *, expected_vocab=49152):
    """Full 1024 x 49152 logits, one causal window. FP32 reference ONLY.

    Avoid autocast and do not call model(x, targets) to ensure sum-reduction and
    denominator are explicit. Gradients disabled. Independent forward resets KDA.
    """
    import torch
    import torch.nn.functional as F
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError('Scoring requires matching rank-one input and target sequences')
    with torch.inference_mode():
        logits, _ = model(x.to(device).unsqueeze(0))
        if logits.shape != (1, x.numel(), expected_vocab):
            raise ValueError(f'Wrong logits shape: {tuple(logits.shape)}')
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError('Non-finite logits')
        # F.cross_entropy reduction=sum scores all target positions once.
        targets = y.to(device)
        nll = F.cross_entropy(logits[0].float(), targets, reduction='sum')
        if not bool(torch.isfinite(nll).item()):
            raise ValueError('Non-finite NLL')
        top1 = int((logits[0].argmax(-1) == targets).sum().item())
        return float(nll.item()), top1


def bootstrap_window_ci(windows, *, seed=1337, n=4000):
    """Approximate resampling over windows, not a formal i.i.d. CI."""
    if not windows or n < 1:
        raise ValueError('Bootstrap requires complete windows and iterations')
    weighted = [(w['nll_sum'], w['scored_tokens']) for w in windows]
    if any(not math.isfinite(a) or a < 0 or type(b) is not int or b <= 0 for a,b in weighted):
        raise ValueError('Invalid NLL/window tokens')
    rng = random.Random(seed)
    k = len(weighted)
    samples = []
    for _ in range(n):
        draw = [weighted[rng.randrange(k)] for _ in range(k)]
        samples.append(sum(a for a, _ in draw) / sum(b for _, b in draw))
    samples.sort()
    # Resampling interval is approximate and windows may share corpus structure.
    return {'nll_95pct_bootstrap_approx': [samples[int((n - 1) * .025)], samples[int((n - 1) * .975)]],
            'iterations': n, 'seed': seed, 'unit': '1024-token windows',
            'warning': 'Exploratory window bootstrap; not independent-document uncertainty'}


def make_run_identity(*, contract_sha, data_sha, code_hashes, checkpoint_sha, offsets, device, torch_version):
    """An interrupted evaluation may resume only with identical inputs and code."""
    return {'schema_version': RUN_SCHEMA, 'contract_file_sha256': contract_sha,
            'dataset_metadata_file_sha256': data_sha, 'checkpoint_sha256_verified_m91': checkpoint_sha,
            'code_sha256': code_hashes, 'window_offsets': list(offsets),
            'seq_len': WINDOW_LEN, 'precision': 'fp32', 'backend': 'chunkwise',
            'device': device, 'torch': torch_version,
            'scored_target_alignment': 'x[t] predicts y[t]=tokens[offset+t+1]',
            'training_eval_prefix_excluded': MONITOR_PREFIX}


def validate_existing_identity(existing, requested):
    if canonical_hash(existing) != canonical_hash(requested):
        raise ValueError('Cannot resume: run manifest differs in data, code, device, or contract')


def read_finished_windows(folder, identity, indices):
    completed = {}
    for i in indices:
        file = Path(folder) / f'window_{i:03d}.json'
        if not file.is_file():
            continue
        w = json.loads(file.read_text(encoding='utf-8'))
        if (w.get('schema_version') != RUN_SCHEMA or w.get('run_identity_sha256') != canonical_hash(identity)
                or w.get('window_id') != i or w.get('offset') != identity['window_offsets'][i]
                or w.get('scored_tokens') != WINDOW_LEN or type(w.get('top1_correct')) is not int
                or not 0 <= w['top1_correct'] <= WINDOW_LEN
                or type(w.get('nll_sum')) not in (float, int)
                or not math.isfinite(w['nll_sum']) or w['nll_sum'] < 0
                or not isinstance(w.get('input_tokens_sha256'), str)
                or len(w['input_tokens_sha256']) != 64):
            raise ValueError(f'Invalid existing window receipt {file}')
        completed[i] = w
    return completed


def summarize_windows(windows, expected_total):
    if not windows:
        return {'completed_windows': 0, 'expected_windows': expected_total, 'status': 'INCOMPLETE'}
    scored = sum(int(w['scored_tokens']) for w in windows)
    loss = math.fsum(w['nll_sum'] for w in windows) / scored
    if not math.isfinite(loss):
        raise ValueError('Invalid aggregated NLL')
    result = {'completed_windows': len(windows), 'expected_windows': expected_total,
              'scored_tokens': scored, 'nll_sum': math.fsum(w['nll_sum'] for w in windows),
              'mean_nll_nats_per_token': loss, 'perplexity_same_tokenizer': math.exp(loss),
              'top1_next_token_accuracy': sum(w['top1_correct'] for w in windows)/scored,
              'uniform_nll_nats_per_token': math.log(49152),
              'status': 'PASS_EVALUATION' if len(windows) == expected_total else 'INCOMPLETE'}
    if len(windows) == expected_total:
        result['bootstrap'] = bootstrap_window_ci(windows)
    return result


def write_window_csv(path, windows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['window_id', 'offset', 'scored_tokens', 'nll_sum', 'mean_nll', 'perplexity', 'top1_correct', 'top1_accuracy', 'input_tokens_sha256', 'seconds'])
        for w in windows:
            cnt = w['scored_tokens']
            avg = w['nll_sum']/cnt
            writer.writerow([w['window_id'], w['offset'], cnt, repr(w['nll_sum']), repr(avg), repr(math.exp(avg)),
                             w['top1_correct'], repr(w['top1_correct']/cnt), w['input_tokens_sha256'], w.get('seconds')])
    tmp.replace(path)
