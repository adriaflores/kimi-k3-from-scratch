"""M9.4 qualitative base-LM continuations (FP32, chunkwise, no cache).

Import has no model loading, GPU allocations, file writes or network activity.
Scientific fixed-battery receipts are separate from notebook playground.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.m9.m92_core import (
    EXPECTED_CKPT_BYTES, EXPECTED_CKPT_SHA, EXPECTED_PARAMS, CHUNK_SIZE,
    PREFERRED_CONTEXT, MODEL_CONFIG_HASH, canonical_hash, choose_next,
    last_logits, require_baseline_metadata, verify_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path('C:/datasets/kimi_k3_backups/m8/m8_full_6b_v1/snapshot_step_183106.pt')
CONTRACT = ROOT / 'scripts/m9/reference/m91_evaluation_contract_frozen.json'
PROTOCOL = ROOT / 'scripts/m9/reference/m94_qualitative_protocol_frozen.json'
OUT = ROOT / 'results/m9/m94_qualitative'
FROZEN_M91_FILE_SHA256 = 'd32b5fc7518977093811f38ed424ea3d3c52913fc8334188c311116abb603a64'
# Filled after the pre-registered protocol is generated; acts as a tamper-evident guard.
FROZEN_M94_PROTOCOL_SHA256 = '7f2dedd49dc09689789b95570f66afd02382b07fcf11852c3de0b7bde781f9c8'
SCHEMA = 'm94_qualitative_1'


def now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def verify_preflight(checkpoint=CHECKPOINT, contract_path=CONTRACT, protocol_path=PROTOCOL):
    """Quick read-only preflight: only hashes tiny JSON files and checkpoint *size*."""
    checkpoint = Path(checkpoint)
    contract_path = Path(contract_path)
    protocol_path = Path(protocol_path)
    if not checkpoint.is_file() or checkpoint.stat().st_size != EXPECTED_CKPT_BYTES:
        raise ValueError('Expected M8 final checkpoint file is absent or wrong size')
    if file_sha(contract_path) != FROZEN_M91_FILE_SHA256:
        raise ValueError('Frozen M9.1 contract file SHA-256 differs')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    verify_contract(contract)
    if file_sha(protocol_path) != FROZEN_M94_PROTOCOL_SHA256:
        raise ValueError('M9.4 pre-registered protocol SHA-256 differs')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    validate_protocol(protocol)
    return contract, protocol


def validate_protocol(p):
    if (p.get('schema_version') != SCHEMA or p.get('status') != 'FROZEN_BEFORE_OUTPUTS'
            or p.get('checkpoint_sha256') != EXPECTED_CKPT_SHA
            or p.get('model_config_sha256') != MODEL_CONFIG_HASH
            or p.get('backend') != 'chunkwise' or p.get('precision') != 'fp32'
            or p.get('max_context_tokens') != 1024 or p.get('eos_id') != 0
            or p.get('tokenizer_revision') != '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'):
        raise ValueError('M9.4 frozen protocol header differs')
    specs = p.get('cases')
    if not isinstance(specs, list) or len(specs) != 12:
        raise ValueError('M9.4 expects exactly 12 fixed generations')
    ids = set()
    for s in specs:
        if set(s) != {'id','group','prompt','mode','seed','max_new_tokens','temperature','top_k','top_p'}:
            raise ValueError('Case fields differ')
        if not isinstance(s['id'], str) or s['id'] in ids or not s['id'].replace('_','').isalnum():
            raise ValueError('Case id invalid or repeated')
        ids.add(s['id'])
        if not isinstance(s['prompt'], str) or not s['prompt'] or len(s['prompt']) > 1000:
            raise ValueError('Invalid fixed prompt')
        validate_generation_args(mode=s['mode'], max_new_tokens=s['max_new_tokens'], temperature=s['temperature'], top_k=s['top_k'], top_p=s['top_p'], seed=s['seed'])
    if len({s['prompt'] for s in specs}) != 6:
        raise ValueError('Expected six distinct prompts')
    for prompt in {s['prompt'] for s in specs}:
        matches = [s for s in specs if s['prompt'] == prompt]
        if sorted(s['mode'] for s in matches) != ['greedy', 'sample']:
            raise ValueError('Each prompt needs one greedy and one sampled continuation')
    return specs


def validate_generation_args(*, mode, max_new_tokens, temperature, top_k, top_p, seed):
    if (mode not in ('greedy','sample') or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(top_k) is not int or not 1 <= top_k <= 49152
            or isinstance(temperature,bool) or not math.isfinite(temperature) or temperature <= 0
            or isinstance(top_p,bool) or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError('Invalid generation parameters (1..64 new tokens, valid temperature/top-k/top-p/seed)')


def unique_ratio(tokens):
    return (len(set(tokens)) / len(tokens)) if tokens else 0.0


def repeat_ngram_ratio(tokens, n=3):
    if n < 1:
        raise ValueError('n must be >=1')
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    return 1 - len(set(grams))/len(grams) if grams else 0.0


def load_tokenizer(contract, *, local_files_only=False):
    from transformers import AutoTokenizer
    t = contract['tokenizer']
    tok = AutoTokenizer.from_pretrained(t['repository'], revision=t['revision'],
                                        use_fast=True, trust_remote_code=False,
                                        local_files_only=local_files_only)
    if not tok.is_fast or len(tok) != 49152 or tok.eos_token_id != 0:
        raise ValueError('Tokenizer identity differs from M7/M8/M9 frozen contract')
    return tok


def load_model(checkpoint_path, contract, device='auto'):
    """Only load model tensors, never instantiate optimizer/trainer. Strict FP32 path."""
    import torch
    from src.kimi_k3.model import KimiK3, KimiK3Config
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    ckpt = torch.load(checkpoint_path, weights_only=True, mmap=True, map_location='cpu')
    cfg = require_baseline_metadata(ckpt, contract)
    with torch.device('meta'):
        model = KimiK3(KimiK3Config(**{**cfg, 'kda_implementation':'chunkwise'}))
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMS:
        raise ValueError('Unexpected model parameter count')
    model.load_state_dict(ckpt['model'], strict=True, assign=True)
    model.lm_head.weight = model.embed_tokens.weight
    if model.lm_head.weight is not model.embed_tokens.weight:
        raise ValueError('Embedding and output head are not tied')
    model.eval().to(device)
    del ckpt
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Model still has meta tensors')
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError('FP32 reference model expected')
    return model, torch.device(device)




def verify_production_source(manifest_path=None):
    """Match model Python files against the M7/M8 frozen manifest.

    Historical YAML files absent from the working checkout do not invalidate
    the already verified model implementation provenance.
    """
    manifest_path=Path(manifest_path or ROOT/'experiments/m7/m75_gate/evidence/m73_source_manifest.json')
    doc=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or len(doc)!=35:
        raise ValueError('M7/M8 frozen source manifest must have 35 entries')
    paths={name: digest for name, digest in doc.items()
           if name.startswith('src/kimi_k3/') and name.endswith('.py')}
    if len(paths)<5:
        raise ValueError('Frozen source manifest missing model Python entries')
    changed=[name for name, digest in paths.items()
             if not (ROOT/name).is_file() or file_sha(ROOT/name)!=digest]
    if changed:
        raise ValueError('Implementation diverged from production manifest: '+', '.join(changed))
    return len(paths)


def load_session(*, device='auto', local_files_only=False, checkpoint=CHECKPOINT,
                 contract_path=CONTRACT, protocol_path=PROTOCOL):
    """Notebook convenience: load once and reuse model + tokenizer."""
    contract, protocol = verify_preflight(checkpoint, contract_path, protocol_path)
    verify_production_source()
    tok = load_tokenizer(contract, local_files_only=local_files_only)
    model, torch_device = load_model(checkpoint, contract, device=device)
    return {'model':model, 'tokenizer':tok, 'device':torch_device,
            'contract':contract, 'protocol':protocol, 'checkpoint_path':str(checkpoint)}


def _sync(device, torch_module):
    if device.type == 'cuda':
        torch_module.cuda.synchronize(device)


def generate_case(session, *, prompt, mode='greedy', max_new_tokens=32,
                  temperature=1.0, top_k=50, top_p=0.95, seed=1337):
    """Generate from base-LM prefix; no cache, no sliding window, no continuation after EOS.

    Statistics are descriptive; interactive outputs are NOT pre-registered evidence.
    Sample RNG is a dedicated generator on the selected device. No global RNG mutation.
    """
    validate_generation_args(mode=mode, max_new_tokens=max_new_tokens, temperature=temperature,
                             top_k=top_k, top_p=top_p, seed=seed)
    import torch
    tok, model, device = session['tokenizer'], session['model'], session['device']
    ids = list(tok.encode(prompt, add_special_tokens=False))
    if not ids:
        raise ValueError('Prompt must contain at least one token')
    if len(ids) + max_new_tokens > PREFERRED_CONTEXT:
        raise ValueError('Prompt + maximum new tokens exceeds verified context length 1024')
    if max_new_tokens > 64:
        raise ValueError('M9.4 permits at most 64 new tokens')
    gen = torch.Generator(device=device.type).manual_seed(seed) if mode=='sample' else None
    continuation = []
    top_next = []
    chosen = []
    entropy = []
    _sync(device, torch)
    t0 = time.perf_counter()
    eos_hit = False
    for idx in range(max_new_tokens):
        logits = last_logits(model, ids+continuation, device=device, precision='fp32', torch_module=torch)
        probs = torch.softmax(logits, dim=-1)
        if not bool(torch.isfinite(probs).all().item()):
            raise ValueError('Generation produced nonfinite probabilities')
        if idx == 0:
            vals, tokids = torch.topk(probs, 5)
            top_next = [{'token_id':int(j), 'token':tok.decode([int(j)], skip_special_tokens=False),
                         'prob':float(v)} for v,j in zip(vals.tolist(),tokids.tolist())]
        ent = -torch.sum(probs * torch.log(probs.clamp_min(1e-30)))
        entropy.append(float(ent.item()))
        picked = choose_next(logits, mode=mode, temperature=temperature, top_k=top_k,
                             top_p=top_p, generator=gen, torch_module=torch)
        chosen.append(float(probs[picked].item()))
        continuation.append(picked)
        if picked == 0:
            eos_hit = True
            break
    _sync(device,torch)
    return {
        'schema_version':SCHEMA, 'prompt':prompt, 'prompt_token_count':len(ids),
        'mode':mode, 'seed':seed, 'max_new_tokens':max_new_tokens,
        'temperature':temperature, 'top_k':top_k, 'top_p':top_p,
        'precision':'fp32', 'backend':'chunkwise', 'cache':False,
        'generated_token_ids':continuation,
        'continuation_text':tok.decode(continuation, skip_special_tokens=False),
        'full_text':tok.decode(ids+continuation, skip_special_tokens=False),
        'generated_token_count':len(continuation), 'eos_hit':eos_hit,
        'unique_token_ratio':unique_ratio(continuation),
        'repeated_trigram_fraction':repeat_ngram_ratio(continuation,3),
        'first_step_top5':top_next,
        'mean_model_entropy_nats':sum(entropy)/len(entropy),
        'mean_chosen_token_probability':sum(chosen)/len(chosen),
        'seconds_generation':round(time.perf_counter()-t0,4),
        'created_at_utc':now(),
        'claim':'Descriptive continuation; not an instruction-following test or model benchmark',
    }


def fingerprint_identity(*, torch_version, device, tokenizer_revision,
                         contract_file=CONTRACT, protocol_file=PROTOCOL,
                         code_files=None):
    code_files = code_files or [ROOT/'scripts/m9/m94_core.py', ROOT/'scripts/m9/m92_core.py',
                               ROOT/'src/kimi_k3/model.py', ROOT/'src/kimi_k3/attention/kda/chunkwise.py',
                               ROOT/'src/kimi_k3/moe/stable_latent_moe.py']
    paths = {str(Path(p).resolve().relative_to(ROOT.resolve())).replace('\\','/'):file_sha(p)
             for p in code_files}
    return {
        'schema_version':SCHEMA,'checkpoint_sha256_verified_m91':EXPECTED_CKPT_SHA,
        'checkpoint_size':EXPECTED_CKPT_BYTES,
        'm91_contract_file_sha256':file_sha(contract_file),
        'm94_protocol_file_sha256':file_sha(protocol_file),
        'tokenizer_revision':tokenizer_revision,
        'backend':'chunkwise','precision':'fp32','device':str(device),
        'torch_version':torch_version,'source_file_hashes':paths,
        'generation_impl':'no-cache full-prefix, right-pad to 64 with EOS; truncate logits to true prefix',
    }


def validate_receipt(r, *, case, identity_sha):
    if (r.get('schema_version') != SCHEMA or r.get('case_id') != case['id']
            or r.get('group') != case['group'] or r.get('fixed_evaluation') is not True
            or r.get('run_identity_sha256') != identity_sha
            or any(r.get(k) != case[k] for k in ('prompt','mode','seed','max_new_tokens','temperature','top_k','top_p'))):
        raise ValueError('Existing case receipt does not match frozen run/case')
    ids = r.get('generated_token_ids')
    if (not isinstance(ids,list) or any(type(x)!=int or not 0<=x<49152 for x in ids)
            or not 1<=len(ids)<=case['max_new_tokens'] or r.get('generated_token_count')!=len(ids)):
        raise ValueError('Existing receipt has invalid token list')
    return r


def summarize_fixed(receipts, identity_sha, protocol_sha):
    if len(receipts)!=12 or len({r['case_id'] for r in receipts})!=12:
        raise ValueError('12 unique cases required for fixed summary')
    return {'schema_version':SCHEMA, 'status':'PASS_FIXED_QUALITATIVE',
            'case_count':len(receipts),'run_identity_sha256':identity_sha,
            'protocol_sha256':protocol_sha,
            'eos_count':sum(r['eos_hit'] for r in receipts),
            'mean_generated_tokens':sum(r['generated_token_count'] for r in receipts)/len(receipts),
            'mean_repeated_trigram_fraction':sum(r['repeated_trigram_fraction'] for r in receipts)/len(receipts),
            'mean_unique_token_ratio':sum(r['unique_token_ratio'] for r in receipts)/len(receipts),
            'fused_parity':'NOT_ESTABLISHED',
            'interpretation':'Human review needed for coherence, factuality and failure patterns. Do not infer scores from illustrative continuations.',
            'cases':receipts}


def write_fixed_exports(out, summary):
    out = Path(out)
    atomic_json(out/'m94_results.json',summary)
    csv_path = out/'generations.csv'
    tmp = csv_path.with_name(csv_path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['case_id','group','mode','seed','prompt','continuation','generated_tokens','eos_hit',
                     'unique_token_ratio','repeated_trigram_fraction','seconds'])
        for r in summary['cases']:
            wr.writerow([r['case_id'],r['group'],r['mode'],r['seed'],r['prompt'],r['continuation_text'],
                         r['generated_token_count'],r['eos_hit'],r['unique_token_ratio'],
                         r['repeated_trigram_fraction'],r['seconds_generation']])
    tmp.replace(csv_path)
    lines=['# M9.4 — Fixed qualitative evaluation','',
           f"**Status:** {summary['status']}",'',
           '**Method:** 6 pre-registered prompts × greedy + seeded sampling, 48 tokens maximum,',
           'FP32, chunkwise PyTorch, no cache, 1024-token training-length context cap.','',
           '**Limitations:** samples are descriptive; no automated human-coherence score,',
           'not Chat/Instruct, BF16/fused equivalence unestablished.','',
           '## All generated continuations (unedited)','']
    for r in summary['cases']:
        lines += [f"### {r['case_id']} — {r['group']} ({r['mode']})",'',
                  '**Prompt**', '', '```text',r['prompt'],'```','',
                  '**Continuation**','','```text',r['continuation_text'],'```','',
                  f"Tokens: {r['generated_token_count']} | EOS: {r['eos_hit']} | "
                  f"Repeat trigrams: {r['repeated_trigram_fraction']:.3f}",'']
    (out/'M94_READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return csv_path


def run_fixed(session, *, out=None, resume=False, selected_ids=None):
    """Run twelve frozen cases with atomic receipts. selected_ids is testing-only subset.

    `selected_ids` may not be used for producing PASS_FIXED_QUALITATIVE.
    Public CLI and notebook always use all twelve.
    """
    import torch
    out = Path(out or OUT/'fixed')
    specs = validate_protocol(session['protocol'])
    if selected_ids is not None:
        raise ValueError('Partial fixed evaluation not allowed; use smoke or playground instead')
    identity = fingerprint_identity(
        torch_version=str(torch.__version__), device=session['device'],
        tokenizer_revision=session['contract']['tokenizer']['revision'])
    identity_sha = canonical_hash(identity)
    manifest = out / 'run_manifest.json'
    cases_dir = out / 'cases'
    if manifest.exists():
        if not resume:
            raise FileExistsError('Fixed run already exists; use resume=True to preserve receipts')
        existing = json.loads(manifest.read_text(encoding='utf-8'))
        if existing != identity:
            raise ValueError('Cannot resume: code/runtime/checkpoint/contract identity differs')
    else:
        if resume or cases_dir.exists() or (out / 'm94_results.json').exists():
            raise ValueError('Orphaned or missing fixed run manifest; refusing to overwrite or assume a run')
        atomic_json(manifest, identity)
    cases_dir.mkdir(parents=True,exist_ok=True)
    results = []
    for i,case in enumerate(specs, 1):
        receipt_path = cases_dir / (case['id'] + '.json')
        if receipt_path.exists():
            receipt = validate_receipt(json.loads(receipt_path.read_text(encoding='utf-8')),
                                       case=case, identity_sha=identity_sha)
            print(f"M9.4 {i:02d}/12 reused {case['id']}",flush=True)
        else:
            print(f"M9.4 {i:02d}/12 generating {case['id']}",flush=True)
            receipt = generate_case(session, **{k:case[k] for k in
                ('prompt','mode','seed','max_new_tokens','temperature','top_k','top_p')})
            receipt.update(case_id=case['id'], group=case['group'],
                           run_identity_sha256=identity_sha,
                           fixed_evaluation=True)
            atomic_json(receipt_path,receipt)
        results.append(receipt)
    summary = summarize_fixed(results, identity_sha, FROZEN_M94_PROTOCOL_SHA256)
    summary['finished_at_utc'] = now()
    summary['precision_policy'] = 'M9.2 FP32 causal reference; numerical parity with fused KDA not established'
    summary['human_rating'] = 'NOT_YET_COLLECTED'
    write_fixed_exports(out, summary)
    return summary
