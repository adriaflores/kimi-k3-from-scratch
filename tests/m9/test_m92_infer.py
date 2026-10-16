"""M9.2 unit contract and generation tests; never load the 2.5 GB checkpoint."""
import json
import sys
import unittest
from pathlib import Path
from contextlib import nullcontext

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.m9.m92_core import (
    MODEL_CONFIG_HASH, EXPECTED_STEP, EXPECTED_TOKENS,
    canonical_hash, choose_next, generate, last_logits, pad_suffix,
    require_baseline_metadata, verify_contract,
)

from scripts.m9.m92_causality import capture_prefix, compare_snapshots, suffix_diagnosis

CONTRACT = ROOT / 'scripts/m9/reference/m91_evaluation_contract_frozen.json'


class DummyCausalLM(torch.nn.Module):
    """Instant test-only causal logits, no real Kimi model or network."""
    def forward(self, x):
        b,t = x.shape
        l = torch.full((b,t,49152), -10.0, device=x.device)
        expected = (x.cumsum(1) % 1024).long() + 30
        l.scatter_(-1, expected.unsqueeze(-1), 10.0)
        return l, None


class ToyBlock(torch.nn.Module):
    def __init__(self, bad=False):
        super().__init__()
        self.bad = bad
        self.attn = torch.nn.Linear(12, 12, bias=False)
        self.ffn = torch.nn.Linear(12, 12, bias=False)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        if self.bad:
            # Deliberately illegal, future-dependent operation for regression.
            x = x + x.mean(dim=1, keepdim=True)
        return x


class ToyProbeModel(torch.nn.Module):
    vocab_size = 49152
    def __init__(self, bad=False):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(49152, 12)
        self.layers = torch.nn.ModuleList([ToyBlock(bad=bad)])
        self.norm = torch.nn.LayerNorm(12)
        self.head = torch.nn.Linear(12, 49152, bias=False)

    def forward(self, tokens):
        h = self.embed_tokens(tokens)
        for block in self.layers:
            h = block(h)
        h = self.norm(h)
        return self.head(h), None


class M92UnitTests(unittest.TestCase):
    def test_cli_precision_defaults_to_fp32_reference(self):
        from scripts.m9.m92_infer import parse_args
        args = parse_args(['--device', 'cuda', '--backend', 'chunkwise'])
        self.assertEqual(args.precision, 'fp32')

    def test_cli_precision_can_explicitly_request_bf16(self):
        from scripts.m9.m92_infer import parse_args
        args = parse_args(['--precision', 'bf16'])
        self.assertEqual(args.precision, 'bf16')

    def test_cli_precision_rejects_unrecognized_values(self):
        from scripts.m9.m92_infer import parse_args
        with self.assertRaises(SystemExit):
            parse_args(['--precision', 'fp16'])

    def test_layerwise_diagnostic_passes_truly_causal_toy(self):
        model = ToyProbeModel().eval()
        ctx = lambda *a, **k: nullcontext()
        result = suffix_diagnosis(model, [23, 42, 17], device=torch.device('cpu'),
                                  precision='fp32', inference_context=ctx)
        self.assertEqual(result['fp32']['status'], 'TRACE_CAPTURED')
        self.assertIsNone(result['fp32']['first_module_over_tolerance'])
        self.assertEqual(result['fp32']['last_logit_elements_above_tolerance'], 0)
        self.assertEqual(result['fp32']['last_logit_max_abs'], 0.0)

    def test_layerwise_diagnostic_flags_intentionally_noncausal_toy(self):
        model = ToyProbeModel(bad=True).eval()
        result = suffix_diagnosis(model, [23, 42, 17], device=torch.device('cpu'),
                                  precision='fp32', inference_context=lambda *a, **k: nullcontext())
        trace = result['fp32']
        self.assertEqual(trace['status'], 'TRACE_CAPTURED')
        self.assertEqual(trace['first_module_over_tolerance'], 'layer_00_residual')
        self.assertGreater(trace['last_logit_max_abs'], 0)
        self.assertGreater(trace['last_logit_elements_above_tolerance'], 0)

    def test_prefix_capture_unregisters_hooks(self):
        model = ToyProbeModel().eval()
        before = [len(m._forward_hooks) for m in model.modules()]
        _ = capture_prefix(model, [1, 2, 3], device=torch.device('cpu'), precision='fp32',
                           suffix_token=0, inference_context=lambda *a, **k: nullcontext())
        after = [len(m._forward_hooks) for m in model.modules()]
        self.assertEqual(before, after)

    def test_frozen_contract(self):
        c = json.loads(CONTRACT.read_text(encoding='utf-8'))
        self.assertEqual(canonical_hash(c['model_config']), MODEL_CONFIG_HASH)
        self.assertEqual(verify_contract(c)['kda_implementation'], 'fused')

    def test_reject_wrong_checkpoint_metadata(self):
        c = json.loads(CONTRACT.read_text(encoding='utf-8'))
        payload = {'format_version': 7, 'step': EXPECTED_STEP,
                   'tokens_seen': EXPECTED_TOKENS, 'vocab_size':49152,
                   'model_config':c['model_config'], 'model': {'weight': torch.ones(1)}}
        self.assertEqual(require_baseline_metadata(payload,c), c['model_config'])
        payload['step'] -= 1
        with self.assertRaisesRegex(ValueError, 'step/tokens'):
            require_baseline_metadata(payload,c)

    def test_checkpoint_dataclass_tuple_vs_frozen_json_list(self):
        # The trainer stores KimiK3Config with asdict(), which retains tuples.
        # M9.1 writes a JSON contract, which cannot represent tuples.
        c = json.loads(CONTRACT.read_text(encoding='utf-8'))
        cfg = {**c['model_config'], 'layer_types': tuple(c['model_config']['layer_types'])}
        payload = {'format_version': 7, 'step': EXPECTED_STEP,
                   'tokens_seen': EXPECTED_TOKENS, 'vocab_size': 49152,
                   'model_config': cfg, 'model': {'weight': torch.ones(1)}}
        self.assertEqual(require_baseline_metadata(payload, c), cfg)
        self.assertIsInstance(payload['model_config']['layer_types'], tuple)

    def test_checkpoint_real_config_drift_is_rejected(self):
        c = json.loads(CONTRACT.read_text(encoding='utf-8'))
        cfg = {**c['model_config'], 'layer_types': tuple(c['model_config']['layer_types'])}
        payload = {'format_version': 7, 'step': EXPECTED_STEP,
                   'tokens_seen': EXPECTED_TOKENS, 'vocab_size': 49152,
                   'model_config': cfg, 'model': {'weight': torch.ones(1)}}
        for key, value in [('num_layers', 12), ('hidden_size', 512),
                           ('kda_implementation', 'chunkwise'), ('mla_output_gate', 1)]:
            modified = {**payload, 'model_config': {**cfg, key: value}}
            with self.assertRaisesRegex(ValueError, key):
                require_baseline_metadata(modified, c)
        modified = {**payload, 'model_config': {**cfg, 'unexpected_key': 'not permitted'}}
        with self.assertRaisesRegex(ValueError, 'unexpected_key'):
            require_baseline_metadata(modified, c)
        removed = {**payload, 'model_config': {k: v for k, v in cfg.items() if k != 'num_layers'}}
        with self.assertRaisesRegex(ValueError, 'num_layers'):
            require_baseline_metadata(removed, c)

    def test_padding_end_of_chunk(self):
        self.assertEqual(pad_suffix([7]), [7]+[0]*63)
        self.assertEqual(pad_suffix(list(range(64))), list(range(64)))
        self.assertEqual(len(pad_suffix([9]*65)), 128)
        self.assertEqual(len(pad_suffix([9]*1024)), 1024)

    def test_reject_bad_inputs_and_no_silent_context_sliding(self):
        for value in ([], [-1], [49152], list(range(1025))):
            with self.assertRaises(ValueError):
                pad_suffix(value)
        with self.assertRaises(ValueError):
            generate(DummyCausalLM(), [1]*1024, device=torch.device('cpu'),
                     torch_module=torch, max_new_tokens=1)

    def test_suffix_is_not_used_for_next_token_alignment(self):
        model = DummyCausalLM().eval()
        ids = [19, 23, 7]
        a = last_logits(model,ids,device=torch.device('cpu'),torch_module=torch)
        b = last_logits(model,ids,device=torch.device('cpu'),torch_module=torch,suffix_token=444)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(int(torch.argmax(a)), (sum(ids)%1024)+30)

    def test_greedy_matches_full_prefix_next_logit(self):
        model=DummyCausalLM().eval()
        ids=[2,3,5]
        next_id=int(torch.argmax(last_logits(model,ids,device=torch.device('cpu'),torch_module=torch)))
        generated=generate(model,ids,device=torch.device('cpu'),torch_module=torch,max_new_tokens=3)
        self.assertEqual(generated[0],next_id)
        self.assertEqual(len(generated),3)

    def test_seeded_sampling(self):
        logits=torch.full((49152,),-100.)
        logits[22] = 3.0
        logits[15] = 2.9
        tokens=[]
        for _ in range(2):
            rng=torch.Generator().manual_seed(212)
            tokens.append([choose_next(logits,mode='sample',top_k=2,top_p=0.95,
                         generator=rng,torch_module=torch) for i in range(15)])
        self.assertEqual(tokens[0],tokens[1])
        self.assertTrue(set(tokens[0]).issubset({22,15}))


if __name__ == '__main__':
    unittest.main()
