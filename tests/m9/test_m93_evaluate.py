"""M9.3 protocol and scoring tests, no actual M8 checkpoint or dataset."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.m9.m93_core import (
    FROZEN_CONTRACT_FILE_SHA256, DATA_METADATA_FILE_SHA256, DATA_METADATA_CANONICAL_SHA256, WINDOW_COUNT, WINDOW_LEN,
    bootstrap_window_ci, dataset_metadata_canonical_sha256, file_sha256, input_target, make_run_identity,
    read_finished_windows, score_window, summarize_windows,
    validate_evaluation_plan, validate_existing_identity, write_atomic,
)
from scripts.m9.m93_evaluate import parse_args

CONTRACT = ROOT / 'scripts/m9/reference/m91_evaluation_contract_frozen.json'


class ToySequence:
    total_tokens = 5000
    vocab_size = 64
    def __init__(self):
        self.a = np.arange(self.total_tokens, dtype=np.uint16) % self.vocab_size
    def read_tokens(self, start, count):
        return self.a[start:start+count].copy()
    def get_sequence(self, start, length):
        v = torch.from_numpy(self.read_tokens(start, length+1).astype('int64'))
        return v[:-1], v[1:]


class ToyCausalModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        B,T=x.shape
        logits = torch.full((B,T,self.vocab), -2.0)
        targets = (x + 1) % self.vocab
        logits.scatter_(2, targets.unsqueeze(-1), 2.0)
        return logits, None


class M93Tests(unittest.TestCase):
    def test_v4_contract_file_unchanged(self):
        self.assertEqual(file_sha256(CONTRACT), FROZEN_CONTRACT_FILE_SHA256)

    def test_exact_preregistered_offsets_and_no_monitor_prefix_overlap(self):
        contract = json.loads(CONTRACT.read_text(encoding='utf8'))
        offsets = validate_evaluation_plan(contract)
        self.assertEqual(len(offsets), 64)
        self.assertEqual(offsets[0], 1_000_000)
        self.assertEqual(offsets[-1], 303_400_000)
        self.assertEqual(len(set(offsets)), 64)
        self.assertTrue(all(b-a > WINDOW_LEN for a,b in zip(offsets, offsets[1:])))

    def test_reject_modified_window_offsets_and_lengths(self):
        c = json.loads(CONTRACT.read_text(encoding='utf8'))
        for bad in (['changed'], ['duplicate'], ['prefix'], ['too_far'], ['alignment']):
            import copy
            clone = copy.deepcopy(c)
            if bad == ['changed']:
                clone['evaluation']['n_windows'] = 63
            elif bad == ['duplicate']:
                clone['evaluation']['window_offsets'][1] = clone['evaluation']['window_offsets'][0]
            elif bad == ['prefix']:
                clone['evaluation']['window_offsets'][0] = 0
            elif bad == ['too_far']:
                clone['evaluation']['window_offsets'][-1] = 316623850
            else:
                clone['evaluation']['reset_state_each_window'] = False
            with self.assertRaises(ValueError):
                validate_evaluation_plan(clone)

    def test_contiguous_shift_and_last_target_counted(self):
        seq = ToySequence()
        x,y,digest = input_target(seq, 57, 64)
        self.assertEqual(x.tolist(), seq.a[57:121].tolist())
        self.assertEqual(y.tolist(), seq.a[58:122].tolist())
        self.assertEqual(y[-1].item(), int(seq.a[121]))
        self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            input_target(seq, 4990, 64)

    def test_nll_sum_exact_count_vs_crossentropy(self):
        ds = ToySequence()
        model = ToyCausalModel(64).eval()
        x,y,_ = input_target(ds, 100, 64)
        actual, top1 = score_window(model, x, y, torch.device('cpu'), expected_vocab=64)
        self.assertEqual(model.calls, 1)
        self.assertEqual(top1, 64)
        expected = torch.nn.functional.cross_entropy(
            model(x.unsqueeze(0))[0][0],y,reduction='sum').item()
        self.assertAlmostEqual(actual, expected, places=5)

    def test_score_rejects_wrong_shape_nan(self):
        x=torch.tensor([1,2,3]);y=torch.tensor([2,3,4])
        model=ToyCausalModel(64)
        with self.assertRaises(ValueError):
            score_window(model,x,y[:2],torch.device('cpu'),expected_vocab=64)
        with self.assertRaises(ValueError):
            score_window(model,x,y,torch.device('cpu'),expected_vocab=65)
        class Bad(torch.nn.Module):
            def forward(self,x):
                return torch.full((1,x.size(1),64),float('nan')),None
        with self.assertRaises(ValueError):
            score_window(Bad(),x,y,torch.device('cpu'),expected_vocab=64)

    def test_mean_aggregation_uses_tokens_not_windows(self):
        def window(i,nll,n):
            return {'nll_sum':nll,'scored_tokens':n,'top1_correct':n//2}
        r=summarize_windows([window(0,100,10),window(1,100,20)],2)
        self.assertAlmostEqual(r['mean_nll_nats_per_token'],200/30)
        self.assertEqual(r['status'],'PASS_EVALUATION')
        self.assertAlmostEqual(r['perplexity_same_tokenizer'],math.exp(200/30))
        r2=summarize_windows([window(0,10,10)],2)
        self.assertEqual(r2['status'],'INCOMPLETE')
        self.assertNotIn('bootstrap',r2)

    def test_deterministic_bootstrap(self):
        w=[{'nll_sum':float(i*1024),'scored_tokens':1024} for i in range(3,10)]
        a=bootstrap_window_ci(w,seed=1337,n=100)
        b=bootstrap_window_ci(w,seed=1337,n=100)
        self.assertEqual(a,b)
        self.assertLessEqual(a['nll_95pct_bootstrap_approx'][0],a['nll_95pct_bootstrap_approx'][1])

    def test_resume_guard_and_atomic_window_receipts(self):
        identity=make_run_identity(contract_sha='abc',data_sha='x',code_hashes={'model':'y'},
            checkpoint_sha='z',offsets=[300000,310000],device='cuda',torch_version='2.11')
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'run_manifest.json'
            write_atomic(f,identity)
            self.assertEqual(json.loads(f.read_text()),identity)
            validate_existing_identity(json.loads(f.read_text()),identity)
            bad={**identity,'device':'cpu'}
            with self.assertRaises(ValueError):
                validate_existing_identity(json.loads(f.read_text()),bad)
            from scripts.m9.m92_core import canonical_hash
            entry={'schema_version':'m93_eval_1','run_identity_sha256':canonical_hash(identity),
                   'window_id':0,'offset':300000,'scored_tokens':1024,'top1_correct':20,
                   'nll_sum':4800.0,'input_tokens_sha256':'a'*64}
            d=Path(td)/'windows'
            write_atomic(d/'window_000.json',entry)
            read=read_finished_windows(d,identity,[0,1])
            self.assertEqual(len(read),1)
            entry['offset']=55
            write_atomic(d/'window_000.json',entry)
            with self.assertRaises(ValueError):
                read_finished_windows(d,identity,[0])


    def test_m7_provenance_raw_and_canonical_digests_are_distinct(self):
        # Raw-file digest from original M7.2 pilot_contract.json; canonical
        # digest from ShardedTokenDataset and M9.1 frozen contract.
        self.assertEqual(DATA_METADATA_FILE_SHA256,
            '15d7fedd033e919aceb76f79b002935bd7cb2e0dcee3ec598f46b5f42fd2805f')
        self.assertEqual(DATA_METADATA_CANONICAL_SHA256,
            '06709535cc3067f61fc1b76847f7d4b3798dbfe3bb5b4cde3dc1755832fd98a9')
        self.assertNotEqual(DATA_METADATA_FILE_SHA256, DATA_METADATA_CANONICAL_SHA256)
        contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
        self.assertEqual(contract['data']['val']['metadata_sha256'],
                         DATA_METADATA_CANONICAL_SHA256)

    def test_dataset_metadata_rejects_bad_identity_before_reading_shards(self):
        from scripts.m9.m93_core import ensure_dataset_identity
        from unittest.mock import patch
        contract=json.loads(CONTRACT.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            metadata={'format_version':2,'tokenizer':dict(contract['tokenizer']),
                      'splits':{'val':{'tokens':316_623_851,
                              'shards':[{'file':f'val_{j:05d}.npy','tokens':10} for j in range(13)]}}}
            mp=d/'metadata.json'
            mp.write_text(json.dumps(metadata),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'raw file SHA256'):
                ensure_dataset_identity(d,contract)
            raw = file_sha256(mp)
            canonical = dataset_metadata_canonical_sha256(metadata)
            with patch('scripts.m9.m93_core.DATA_METADATA_FILE_SHA256', raw):
                with self.assertRaisesRegex(ValueError,'canonical SHA256'):
                    ensure_dataset_identity(d,contract)
                with patch('scripts.m9.m93_core.DATA_METADATA_CANONICAL_SHA256', canonical):
                    toy_contract = json.loads(json.dumps(contract))
                    toy_contract['data']['val']['metadata_sha256'] = canonical
                    self.assertEqual(ensure_dataset_identity(d,toy_contract),raw)
                    metadata['tokenizer']['eos_id'] = 1
                    mp.write_text(json.dumps(metadata),encoding='utf-8')
                    with patch('scripts.m9.m93_core.DATA_METADATA_FILE_SHA256',file_sha256(mp)):
                        with self.assertRaisesRegex(ValueError,'canonical SHA256'):
                            ensure_dataset_identity(d,toy_contract)
                    with patch('scripts.m9.m93_core.DATA_METADATA_FILE_SHA256',file_sha256(mp)):
                        with patch('scripts.m9.m93_core.DATA_METADATA_CANONICAL_SHA256',dataset_metadata_canonical_sha256(metadata)):
                            changed_contract = json.loads(json.dumps(toy_contract))
                            changed_contract['data']['val']['metadata_sha256'] = dataset_metadata_canonical_sha256(metadata)
                            with self.assertRaisesRegex(ValueError,'Dataset tokenizer metadata differs: eos_id'):
                                ensure_dataset_identity(d,changed_contract)

    def test_dataset_shift_detects_bad_reader_and_rejects_vocab(self):
        class Misaligned(ToySequence):
            def get_sequence(self, start, length):
                x,y=super().get_sequence(start,length)
                return x,y.roll(1)
        with self.assertRaisesRegex(ValueError,'misaligned'):
            input_target(Misaligned(),27,64)
        bad = ToySequence()
        bad.vocab_size = 10
        with self.assertRaisesRegex(ValueError,'Out-of-vocabulary'):
            input_target(bad,0,64)

    def test_cli_defaults_and_safety_flags(self):
        a=parse_args([])
        self.assertEqual(a.device,'cuda')
        self.assertFalse(a.smoke)
        self.assertFalse(a.resume)
        self.assertFalse(a.verify_checkpoint_sha)
        self.assertEqual(a.data_dir.as_posix().lower(),'c:/datasets/fineweb_edu_6b')

if __name__ == '__main__':
    unittest.main()
