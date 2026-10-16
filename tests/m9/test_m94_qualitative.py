"""M9.4 protocol/IO/sampling tests; no 2.5GB checkpoint or HuggingFace network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.m9 import m94_core as c


class ProtocolTests(unittest.TestCase):
    def test_m94_protocol_pinned_and_frozen(self):
        self.assertEqual(c.file_sha(c.PROTOCOL), c.FROZEN_M94_PROTOCOL_SHA256)
        p=json.loads(c.PROTOCOL.read_text(encoding='utf-8'))
        self.assertEqual(len(c.validate_protocol(p)),12)
        self.assertEqual(p['status'],'FROZEN_BEFORE_OUTPUTS')

    def test_m91_contract_pinned(self):
        self.assertEqual(c.file_sha(c.CONTRACT), c.FROZEN_M91_FILE_SHA256)
        doc=json.loads(c.CONTRACT.read_text(encoding='utf-8'))
        c.verify_contract(doc)

    def test_reject_modified_protocol(self):
        p=json.loads(c.PROTOCOL.read_text(encoding='utf-8'))
        p['cases'][0]['mode']='sample'
        with self.assertRaises(ValueError):
            c.validate_protocol(p)

    def test_reject_duplicate_fixed_case_id(self):
        p=json.loads(c.PROTOCOL.read_text(encoding='utf-8'))
        p['cases'][1]['id']=p['cases'][0]['id']
        with self.assertRaises(ValueError):
            c.validate_protocol(p)

    def test_reject_invalid_generation_parameters(self):
        valid=dict(mode='sample',max_new_tokens=32,temperature=1.,top_k=50,top_p=.95,seed=1337)
        for k,v in [('max_new_tokens',65),('top_p',0),('top_k',0),('temperature',0),('seed',-1),('mode','instruction')]:
            bad=dict(valid);bad[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):
                c.validate_generation_args(**bad)

    def test_context_cap_matches_m92_reference(self):
        self.assertEqual(c.PREFERRED_CONTEXT,1024)
        self.assertEqual(c.CHUNK_SIZE,64)

    def test_stats_empty_and_repeated(self):
        self.assertEqual(c.unique_ratio([]),0.0)
        self.assertEqual(c.repeat_ngram_ratio([],3),0.0)
        self.assertEqual(c.unique_ratio([1,2,1,2]),0.5)
        self.assertAlmostEqual(c.repeat_ngram_ratio([1,2,1,2,1],2),0.5)
        with self.assertRaises(ValueError):
            c.repeat_ngram_ratio([1],0)

    def test_check_checkpoint_size_and_contract_without_loading(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'large.pt'
            f.write_bytes(b'synthetic-test-file')
            with patch.object(c, 'EXPECTED_CKPT_BYTES', len(b'synthetic-test-file')):
                contract, protocol = c.verify_preflight(checkpoint=f)
            self.assertEqual(contract['model_config']['hidden_size'],640)
            self.assertEqual(len(protocol['cases']),12)

    def test_modify_protocol_fails_sha_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'modified.json'
            p.write_text(c.PROTOCOL.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            ck=Path(d)/'large.pt'
            ck.write_bytes(b'synthetic-test-file')
            with patch.object(c, 'EXPECTED_CKPT_BYTES', len(b'synthetic-test-file')):
                with self.assertRaisesRegex(ValueError,'protocol SHA-256'):
                    c.verify_preflight(ck, c.CONTRACT, p)

    def test_receipt_validation_catches_mutations(self):
        p=json.loads(c.PROTOCOL.read_text(encoding='utf-8'))
        spec=p['cases'][0]
        r=dict(spec, schema_version=c.SCHEMA, case_id=spec['id'],run_identity_sha256='abc',
               generated_token_ids=[55],generated_token_count=1,fixed_evaluation=True)
        c.validate_receipt(r,case=spec,identity_sha='abc')
        for changes in ({'seed':55},{'run_identity_sha256':'xyz'},{'generated_token_count':2},{'generated_token_ids':[49152]}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                c.validate_receipt({**r,**changes},case=spec,identity_sha='abc')

    def test_atomic_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'nested'/'out.json'
            c.atomic_json(path, {'hola':'áéí','list':[1,2]})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['hola'],'áéí')
            self.assertFalse(path.with_name('out.json.tmp').exists())

    def test_sample_is_seeded_and_deterministic(self):
        import torch
        logits=torch.tensor([2.,1.5,0.,-1.,-2.])
        def draw(seed):
            g=torch.Generator(device='cpu').manual_seed(seed)
            return [c.choose_next(logits,mode='sample',temperature=.9,top_k=4,top_p=.95,
                         generator=g,torch_module=torch) for _ in range(16)]
        self.assertEqual(draw(99),draw(99))
        self.assertEqual(c.choose_next(logits,mode='greedy',torch_module=torch),0)

    def test_generate_smoke_uses_actual_prefix_and_no_cache(self):
        import torch
        class Tok:
            def encode(self,x,add_special_tokens=False):
                assert add_special_tokens is False
                return [2,3]
            def decode(self,x,skip_special_tokens=False):
                return ' '.join(map(str,x))
        class Model:
            def __call__(self,tokens):
                assert tokens.shape[1] % 64 == 0
                logits=torch.full((1,tokens.shape[1],49152),-10.,device=tokens.device)
                logits[...,13]=7.
                return logits,None
        session={'tokenizer':Tok(),'model':Model(),'device':torch.device('cpu')}
        x=c.generate_case(session,prompt='hello',max_new_tokens=4,mode='greedy')
        self.assertEqual(x['generated_token_ids'],[13]*4)
        self.assertEqual(x['prompt_token_count'],2)
        self.assertEqual(x['generated_token_count'],4)
        self.assertFalse(x['eos_hit'])
        self.assertEqual(x['first_step_top5'][0]['token_id'],13)

    def test_generate_rejects_excessive_context_before_model(self):
        import torch
        class Tok:
            def encode(self,prompt,add_special_tokens=False):return [1]*1000
        session={'tokenizer':Tok(),'model':None,'device':torch.device('cpu')}
        with self.assertRaisesRegex(ValueError,'context'):
            c.generate_case(session,prompt='foo',max_new_tokens=32)

    def test_fixed_resume_preserves_receipts_and_detects_code_drift(self):
        import torch
        p=json.loads(c.PROTOCOL.read_text(encoding='utf-8'))
        contract=json.loads(c.CONTRACT.read_text(encoding='utf-8'))
        session={'protocol':p,'contract':contract,'device':torch.device('cpu')}
        from unittest.mock import Mock
        def fake_gen(session,**kwargs):
            return dict(schema_version=c.SCHEMA,**kwargs,generated_token_ids=[41],
                        generated_token_count=1, continuation_text=' hello', eos_hit=False,
                        unique_token_ratio=1., repeated_trigram_fraction=0.,seconds_generation=0.1)
        identity = {'example':'hash'}
        with tempfile.TemporaryDirectory() as d, patch.object(c,'generate_case',side_effect=fake_gen) as generator, \
             patch.object(c,'fingerprint_identity',return_value=identity):
            result=c.run_fixed(session,out=d)
            self.assertEqual(result['status'],'PASS_FIXED_QUALITATIVE')
            self.assertEqual(generator.call_count,12)
            self.assertTrue((Path(d)/'generations.csv').is_file())
            with self.assertRaises(FileExistsError):c.run_fixed(session,out=d)
            again=c.run_fixed(session,out=d,resume=True)
            self.assertEqual(again['case_count'],12)
            self.assertEqual(generator.call_count,12)
            (Path(d)/'run_manifest.json').write_text('{"example":"changed"}')
            with self.assertRaisesRegex(ValueError,'Cannot resume'):
                c.run_fixed(session,out=d,resume=True)


if __name__=='__main__':unittest.main()
