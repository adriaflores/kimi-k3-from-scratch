"""M9.2 inference primitives: no optimizer, no trainer, no network.

Only model logits at genuine (unpadded) token positions are used. Right
padding to the KDA chunk multiple does not require attention masks in a
properly causal network, and a suffix-interference test verifies this.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

EXPECTED_CKPT_SHA = 'e332bf4d34ea5d392bf0bd6d13a87b90ddc799e737275d972672671492c67dd1'
EXPECTED_CKPT_BYTES = 2_485_700_551
EXPECTED_STEP = 183106
EXPECTED_TOKENS = 6_000_017_408
EXPECTED_PARAMS = 294_676_808
CHUNK_SIZE = 64
PREFERRED_CONTEXT = 1024
MODEL_CONFIG_HASH = '9d455836820541befff1b4fda57b42ec2f4fc4dd5b497e86f79aea4a7b6746c9'


def canonical_hash(obj):
    """Stable semantic hash independent of JSON formatting (not a file hash)."""
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def normalize_json_sequences(obj):
    """Compare dataclass tuples with JSON arrays without changing any values.

    ``asdict(KimiK3Config)`` preserves ``layer_types`` as a tuple in trainer
    format-7 checkpoints, while JSON serialization converts it to a list.
    Only tuple/list representation is normalized; keys and scalar values stay
    exact.  Unknown data types are left unchanged (and will fail comparison).
    """
    if isinstance(obj, (tuple, list)):
        return [normalize_json_sequences(item) for item in obj]
    if isinstance(obj, dict):
        return {key: normalize_json_sequences(value) for key, value in obj.items()}
    return obj


def require_baseline_metadata(checkpoint, contract):
    if checkpoint.get('format_version') != 7:
        raise ValueError('Expected format_version=7')
    if checkpoint.get('step') != EXPECTED_STEP or checkpoint.get('tokens_seen') != EXPECTED_TOKENS:
        raise ValueError('Wrong checkpoint step/tokens')
    if checkpoint.get('vocab_size') != 49152:
        raise ValueError('Wrong checkpoint vocab')
    model_config = checkpoint.get('model_config')
    expected = contract['model_config']
    if not isinstance(model_config, dict):
        raise ValueError('Checkpoint model_config must be a dict')
    normalized = normalize_json_sequences(model_config)
    # JSON encoding is intentionally type-aware: e.g. a bool is not an int.
    if canonical_hash(normalized) != canonical_hash(expected):
        missing = object()
        diff = sorted(k for k in model_config.keys() | expected.keys()
                      if k not in normalized or k not in expected or
                      canonical_hash(normalized.get(k, missing)) != canonical_hash(expected.get(k, missing)))
        raise ValueError('Checkpoint model_config differs from M9.1 frozen contract: '
                         + ', '.join(diff))
    # Check all keys/values. The original checkpoint dict is returned unchanged.
    if 'model' not in checkpoint or not isinstance(checkpoint['model'], dict):
        raise ValueError('Checkpoint missing model state_dict')
    return model_config


def verify_contract(contract):
    if contract.get('schema_version') != 'm91_eval_contract_1':
        raise ValueError('Not the frozen M9.1 evaluation contract')
    if contract.get('baseline_checkpoint_sha256') != EXPECTED_CKPT_SHA:
        raise ValueError('Unexpected frozen checkpoint identity')
    cfg = contract.get('model_config')
    if not isinstance(cfg, dict) or cfg.get('kda_implementation') != 'fused':
        raise ValueError('Unexpected original model backend')
    if canonical_hash(cfg) != MODEL_CONFIG_HASH or contract.get('model_config_json_sha256') != MODEL_CONFIG_HASH:
        raise ValueError('Frozen config identity mismatch')
    tok = contract.get('tokenizer', {})
    if (tok.get('repository'), tok.get('revision'), tok.get('eos_id'), tok.get('vocab_size')) != (
        'HuggingFaceTB/SmolLM2-135M',
        '93efa2f097d58c2a74874c7e644dbc9b0cee75a2', 0, 49152):
        raise ValueError('Tokenizer identity differs from frozen contract')
    return cfg


def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pad_suffix(ids, *, chunk_size=CHUNK_SIZE, eos_id=0, max_context=PREFERRED_CONTEXT):
    if not ids or len(ids) > max_context:
        raise ValueError(f'Input token count must be in [1, {max_context}]')
    if any(not isinstance(t, int) or t < 0 or t >= 49152 for t in ids):
        raise ValueError('Invalid input token id')
    pad_len = (-len(ids)) % chunk_size
    if len(ids) + pad_len > max_context:
        raise ValueError('Padding would exceed the allowed context')
    return list(ids) + [eos_id] * pad_len


def inference_context(device, precision, torch_module):
    if device.type == 'cuda' and precision == 'bf16':
        return torch_module.autocast('cuda', dtype=torch_module.bfloat16)
    return nullcontext()


def last_logits(model, ids, *, device, precision='fp32', torch_module=None, suffix_token=0):
    if torch_module is None:
        import torch as torch_module
    padded = pad_suffix(ids, eos_id=suffix_token)
    tokens = torch_module.tensor([padded], dtype=torch_module.long, device=device)
    with torch_module.inference_mode(), inference_context(device, precision, torch_module):
        out = model(tokens)
        logits = out[0] if isinstance(out, tuple) else out
        last = logits[0, len(ids) - 1, :].float()
    if not bool(torch_module.isfinite(last).all().item()):
        raise ValueError('Non-finite next-token logits')
    if last.numel() != 49152:
        raise ValueError('Wrong next-token vocabulary dimension')
    return last


def choose_next(logits, *, mode, temperature=1.0, top_k=50, top_p=0.95, generator=None, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    if mode == 'greedy':
        return int(torch_module.argmax(logits).item())
    if mode != 'sample' or temperature <= 0 or not 0 < top_p <= 1 or top_k < 1:
        raise ValueError('Bad sampling arguments')
    logits = logits.float() / temperature
    k = min(top_k, logits.numel())
    scores, indices = torch_module.topk(logits, k)
    probs = torch_module.softmax(scores, dim=-1)
    order = torch_module.argsort(probs, descending=True)
    sorted_prob = probs[order]
    csum = torch_module.cumsum(sorted_prob, dim=-1)
    cutoff = (csum - sorted_prob) >= top_p
    sorted_prob = sorted_prob.masked_fill(cutoff, 0)
    sorted_prob = sorted_prob / sorted_prob.sum()
    selected = torch_module.multinomial(sorted_prob, 1, generator=generator)
    return int(indices[order[selected]].item())


def generate(model, prompt_ids, *, device, torch_module, max_new_tokens=4,
             mode='greedy', temperature=1.0, top_k=50, top_p=0.95,
             seed=1337, eos_id=0, precision='fp32'):
    if max_new_tokens < 0 or max_new_tokens > 64:
        raise ValueError('M9.2 allows at most 64 generated tokens')
    if not prompt_ids or len(prompt_ids) + max_new_tokens > PREFERRED_CONTEXT:
        raise ValueError('Would exceed the 1024-token evaluation context (no silent sliding window)')
    gen = torch_module.Generator(device=device.type).manual_seed(seed) if mode == 'sample' else None
    sequence = list(prompt_ids)
    generated = []
    for _ in range(max_new_tokens):
        logits = last_logits(model, sequence, device=device, precision=precision, torch_module=torch_module)
        next_id = choose_next(logits, mode=mode, temperature=temperature, top_k=top_k,
                              top_p=top_p, generator=gen, torch_module=torch_module)
        sequence.append(next_id)
        generated.append(next_id)
        if next_id == eos_id:
            break
    return generated


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)
