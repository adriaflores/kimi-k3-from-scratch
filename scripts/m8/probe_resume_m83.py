#!/usr/bin/env python3
"""True format-7 M8 restore probe. No train/optimizer.step/save_checkpoint calls.

Use only the pinned H100 NVL Python. This deliberately recreates the real
KimiK3 model, Muon+AdamW, loader cursors and RNG state, and calls the trainer's
unaltered load_checkpoint. Never use a mock to authorize a GPU run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

START_SHA = 'cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12'
RECIPE_SHA = '2917190c3db675d18f0e439696e77222c87a8e577c5eba96aadaaf4708516ca6'
META_CANON_SHA = '06709535cc3067f61fc1b76847f7d4b3798dbfe3bb5b4cde3dc1755832fd98a9'
ORIGINAL = Path('/workspace/kimi/runs/m73/m73_pilot_80m_v1_20261014T065121Z/last.pt')


def sha(path):
    dig = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            dig.update(chunk)
    return dig.hexdigest()


def verify_reference(reference, manifest_path):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if len(manifest)!=35 or any(not isinstance(p,str) or not isinstance(h,str) or len(h)!=64
                                  for p,h in manifest.items()):
        raise ValueError('Original 35-file source manifest incomplete')
    for name,expected in manifest.items():
        file=(reference/name).resolve()
        if not file.is_relative_to(reference) or not file.is_file() or sha(file)!=expected:
            raise ValueError(f'Immutable M7.3 reference differs: {name}')
    return len(manifest)


def execute(args):
    root=args.root.resolve(strict=True)
    if root != Path('/workspace/kimi'):
        raise ValueError('Production proof requires the approved /workspace/kimi root')
    cp=args.checkpoint.resolve(strict=True)
    snapshot_root=root/'artifacts/m8/m82/m8_full_6b_v1/snapshots'
    if cp.parent != snapshot_root or not cp.name.startswith('snapshot_step_') or cp.suffix!='.pt':
        raise ValueError('Only the preserved M8.2 snapshot can be probed')
    step=int(cp.stem.split('_')[-1])
    if step<4096 or step>=183106 or step%4096:
        raise ValueError('Recovery checkpoint must be an actual M8 periodic checkpoint')
    receipt=json.loads(cp.with_suffix('.json').read_text(encoding='utf-8'))
    digest_before=sha(cp)
    if receipt.get('step')!=step or receipt.get('bytes')!=cp.stat().st_size or receipt.get('sha256')!=digest_before:
        raise ValueError('Snapshot receipt differs from actual bytes')
    original=root/ORIGINAL.relative_to(root)
    original_sha=sha(original)
    if original_sha!=START_SHA:
        raise ValueError('M7.3 source checkpoint modified')
    reference=(root/'reference').resolve(strict=True)
    source_count=verify_reference(reference,root/'experiments/m7/m75_gate/evidence/m73_source_manifest.json')
    recipe=root/'experiments/m7/m75_gate/recipe/m8_full_6b.yaml'
    if sha(recipe)!=RECIPE_SHA:
        raise ValueError('Approved recipe hash changed')
    os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    sys.path.insert(0,str(reference))
    import numpy as np
    import torch
    from src.kimi_k3.data import ShardedTokenDataset,TokenDataLoader
    from src.kimi_k3.model import KimiK3,KimiK3Config
    from src.kimi_k3.training.trainer import (TrainConfig,build_optimizer,load_checkpoint,
                                             learning_rate_at_step,_runtime_signature)
    from src.kimi_k3.utils import load_config
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    if str(torch.__version__)!='2.11.0+cu128' or torch.version.cuda!='12.8':
        raise ValueError('Pinned PyTorch/CUDA runtime differs from M7')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='NVIDIA H100 NVL':
        raise ValueError('Restore must be performed on qualified H100 NVL')
    device=torch.device('cuda:0')
    config=load_config(recipe,TrainConfig)
    if (config.max_steps,config.batch_size,config.seq_len,config.gradient_accumulation_steps,
        config.learning_rate,config.precision,config.muon_compile,config.muon_batched,
        config.checkpoint_interval)!=(183106,32,1024,1,.003,'bf16',True,True,4096):
        raise ValueError('Training config unexpectedly changed')
    data=root/'data/full'
    train_data=ShardedTokenDataset(data,split='train')
    val_data=ShardedTokenDataset(data,split='val')
    if (train_data.total_tokens,val_data.total_tokens)!=(6000001675,316623851) or (
        train_data.metadata_sha256!=META_CANON_SHA or val_data.metadata_sha256!=META_CANON_SHA):
        raise ValueError('Corpus/tokenizer metadata changed')
    train_loader=TokenDataLoader(train_data,config.batch_size,config.seq_len)
    val_loader=TokenDataLoader(val_data,config.batch_size,config.seq_len)
    model_config=load_config(reference/'configs/model/kimi_k3_300m.yaml',KimiK3Config)
    if model_config.vocab_size!=train_data.vocab_size:
        raise ValueError('Data/model tokenizer mismatch')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    model=KimiK3(model_config).to(device)
    optimizer=build_optimizer(model,config)
    count=sum(p.numel() for p in model.parameters())
    if count!=294676808:raise ValueError('Parameter count mismatch')
    saved=torch.load(str(cp),map_location='cpu',weights_only=True,mmap=True)
    if saved.get('format_version')!=7 or saved.get('step')!=step or saved.get('skipped_updates')!=0:
        raise ValueError('Saved checkpoint header mismatch')
    if saved.get('runtime')!=_runtime_signature(model):
        raise ValueError('Runtime signature mismatch')
    if saved['train_config'] != asdict(config):
        raise ValueError('Checkpoint recipe differs from approved M8 recipe')
    if len(saved['optimizer']['state']) != 1830:
        raise ValueError('Unexpected optimizer state coverage')
    result=load_checkpoint(cp,model,optimizer,config,vocab_size=train_data.vocab_size,
                           train_loader=train_loader,val_loader=val_loader)
    if (result.step,result.tokens_seen,result.skipped_updates)!=(step,step*32768,0):
        raise ValueError('load_checkpoint did not restore expected step/tokens/skips')
    expected_train_position=step*32768
    if train_loader.position!=saved['loaders']['train']['position'] or train_loader.position!=expected_train_position:
        raise ValueError('Train cursor was not restored')
    if val_loader.position!=saved['loaders']['val']['position'] or val_loader.position!=8*32768:
        raise ValueError('Val cursor was not restored')
    if train_loader.state_dict()!=saved['loaders']['train'] or val_loader.state_dict()!=saved['loaders']['val']:
        raise ValueError('Dataloader state mismatch')
    if random.getstate()!=saved['rng_state']['python']:
        raise ValueError('Python RNG differs')
    npa=np.random.get_state();nps=saved['rng_state']['numpy']
    if npa[0]!=nps[0] or not np.array_equal(npa[1],np.asarray(nps[1],dtype=np.uint32)) or tuple(npa[2:])!=tuple(nps[2:]):
        raise ValueError('NumPy RNG differs')
    if not torch.equal(torch.get_rng_state(),saved['rng_state']['torch_cpu']):
        raise ValueError('CPU RNG differs')
    if not torch.equal(torch.cuda.get_rng_state(device).cpu(),saved['rng_state']['torch_cuda'].cpu()):
        raise ValueError('CUDA RNG differs')
    if len(optimizer.state_dict()['state'])!=len(saved['optimizer']['state']):
        raise ValueError('Optimizer state missing after restore')
    if len(optimizer.param_groups)!=len(saved['optimizer']['param_groups']) or any(
        now['lr']!=before['lr'] for now,before in zip(optimizer.param_groups,saved['optimizer']['param_groups'])):
        raise ValueError('Optimizer groups/LR not restored')
    # Verify next batch is readable without consuming the checkpoint cursor.
    x,y=train_data.get_batch(train_loader.position,config.batch_size,config.seq_len)
    if x.shape!=y.shape!=(32,1024) or train_loader.position!=expected_train_position:
        raise ValueError('Next training batch or loader cursor differs')
    next_lr=learning_rate_at_step(config,result.step+1)
    if not math.isfinite(next_lr) or next_lr<=0:
        raise ValueError('Next scheduled LR is invalid')
    digest_after=sha(cp)
    if digest_after!=digest_before:
        raise ValueError('Checkpoint mutated during recovery gate')
    return {'status':'PASS','no_optimizer_updates_executed':True,
            'checkpoint_path':str(cp),'checkpoint_step':step,'restored_step':result.step,
            'checkpoint_format_version':7,'checkpoint_filesize':cp.stat().st_size,
            'checkpoint_sha256_before':digest_before,'checkpoint_sha256_after':digest_after,
            'restored_train_position':train_loader.position,'restored_val_position':val_loader.position,
            'rng_verified':['python','numpy','torch_cpu','torch_cuda'],
            'optimizer_state_entries':len(optimizer.state_dict()['state']),
            'optimizer_groups':len(optimizer.param_groups),'model_parameters':count,
            'source_files_sha256_verified':source_count,'original_source_sha256':original_sha,
            'recipe_sha256':sha(recipe),'skipped_updates':result.skipped_updates,
            'next_step':result.step+1,'next_lr':next_lr,
            'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace/kimi'))
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    args=p.parse_args(argv)
    result={'status':'FAIL'}
    try:result=execute(args)
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        result['traceback']=traceback.format_exc()
    result['at_utc']=datetime.now(timezone.utc).isoformat()
    allowed=args.root/'artifacts/m8/m83'
    if args.receipt.resolve(strict=False) != (allowed/'probe_receipt.json').resolve(strict=False):
        print('Refuse to write restore receipt outside dedicated artifact path',file=sys.stderr)
        return 2
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    temp=args.receipt.with_name(args.receipt.name+'.partial')
    with temp.open('x',encoding='utf-8') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n');out.flush();os.fsync(out.fileno())
    os.replace(temp,args.receipt)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__':sys.exit(main())
