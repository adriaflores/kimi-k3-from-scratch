#!/usr/bin/env python3
"""Pure-stdlib local static gate, no torch, YAML installation or GPU needed."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def scalar_yaml(path):
    """Read only the intentionally flat scalar YAML of TrainConfig; fail closed."""
    result = {}
    for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition(':')
        if not sep or not key.strip() or not value.strip() or key.strip() in result:
            raise ValueError(f'Unexpected recipe syntax at {path}:{n}')
        result[key.strip()] = value.strip()
    return result

pilot_path = BASE / 'recipe/m73_pilot_original.yaml'
m8_path = BASE / 'recipe/m8_full_6b.yaml'
pilot = scalar_yaml(pilot_path)
m8 = scalar_yaml(m8_path)
assert sha256(pilot_path) == 'c7c84909ce4f8e759f11ca07b607fad18c387aca905659126a6274b7e4027835'
assert pilot.keys() == m8.keys()
changed = {key for key in pilot if pilot[key] != m8[key]}
assert changed == {'max_steps','log_interval','eval_interval','checkpoint_interval'}, changed
assert int(pilot['max_steps']) == 2442 and int(m8['max_steps']) == 183106
assert int(m8['warmup_steps']) == int(pilot['warmup_steps']) == 1832
assert int(m8['decay_steps']) == int(pilot['decay_steps']) == 183106
batch = int(m8['batch_size']) * int(m8['seq_len']) * int(m8['gradient_accumulation_steps'])
assert batch == 32768
assert batch * int(pilot['max_steps']) == 80019456
assert (int(m8['max_steps']) - int(pilot['max_steps'])) * batch == 5919997952
assert int(m8['max_steps']) * batch == 6000017408
assert 6000001675 - (int(m8['max_steps'])-1) * batch == 17035
assert int(m8['max_steps']) * batch - 6000001675 == 15733
manifest = json.loads((BASE / 'evidence/m73_source_manifest.json').read_text(encoding='utf-8'))
assert len(manifest) == 35
assert all(isinstance(v, str) and len(v) == 64 for v in manifest.values())
receipt = json.loads((BASE / 'evidence/m73_final_receipt.json').read_text(encoding='utf-8'))
assert receipt['sha256'] == 'cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12'
assert receipt['bytes'] == 2485700423
for path in BASE.rglob('*.py'):
    compile(path.read_bytes(), str(path), 'exec')
print('PASS: M7.5 local static contract and Python syntax; original pilot YAML SHA matched.')
print('PASS: full 6B target; 35 recorded source hashes; no recipe change except M8 cadence/horizon.')
print('NOTE: final batch restarts from corpus offset 0 after dropping 17,035-token tail.')
