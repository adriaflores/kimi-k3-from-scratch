"""Dependency-free unit tests for the M9.1 read-only evidence audit.

Run from repository root: python -m unittest discover -s tests/m9 -p 'test_m91_audit.py' -v
"""
import hashlib
import json
import tempfile
import unittest
import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'm9'))
import m91_audit as m91


class M91UnitTests(unittest.TestCase):
    def test_expected_sha256_of_bundled_recipe(self):
        recipe = ROOT / 'scripts' / 'm9' / 'reference' / 'm8_full_6b.yaml'
        self.assertTrue(recipe.is_file())
        self.assertEqual(m91.sha256_file(recipe), m91.EXPECTED_RECIPE_SHA)

    def test_contract_offsets_fit_val_and_skip_monitor_prefix(self):
        session = {
            'model_config': {'max_seq_len': 2048, 'kda_implementation': 'fused'},
            'train_config': {'seq_len': 1024},
            'data': {'val': {'total_tokens': 316623851, 'tokenizer': {'vocab_size': 49152}}},
        }
        obj = m91.frozen_contract({'sessions': [session]}, {'checkpoint_sha256': m91.EXPECTED_CHECKPOINT_SHA}, m91.EXPECTED_EVIDENCE_SHA)
        e = obj['evaluation']
        self.assertEqual(e['total_scored_tokens'], 65536)
        self.assertEqual(len(e['window_offsets']), 64)
        self.assertEqual(e['window_offsets'][0], 1_000_000)
        self.assertEqual(e['window_offsets'][-1], 303_400_000)
        self.assertTrue(all(b - a == 4_800_000 for a, b in zip(e['window_offsets'], e['window_offsets'][1:])))
        self.assertTrue(all(o >= 262_144 and o + 1024 < 316623851 for o in e['window_offsets']))

    def test_source_manifest_pass_then_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = {}
            for i in range(35):
                rel = f'src/file_{i:02d}.py'
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f'# source file {i}\n', encoding='utf-8')
                manifest[rel] = m91.sha256_file(p)
            manifest_path = root / 'original_manifest.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(m91.verify_source(root, manifest_path)['status'], 'PASS')
            (root / 'src/file_00.py').write_text('# changed\n', encoding='utf-8')
            report = m91.verify_source(root, manifest_path)
            self.assertEqual(report['status'], 'MISMATCH')
            self.assertEqual(len(report['mismatches']), 1)

    def test_source_manifest_rejects_paths_escaping_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = {f'src/f{i:02d}.py': '0' * 64 for i in range(34)}
            manifest['../outside.py'] = '0' * 64
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            report = m91.verify_source(root, manifest_path)
            self.assertEqual(report['status'], 'MISMATCH')
            self.assertIn('unsafe_path', [x['issue'] for x in report['mismatches']])

    def test_zip_evidence_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'evidence.zip'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('final_audit.json', json.dumps({'status': 'PASS'}))
            original = m91.sha256_file(p)
            with zipfile.ZipFile(p) as z:
                self.assertEqual(m91.load_json(z, 'final_audit.json')['status'], 'PASS')
            self.assertEqual(m91.sha256_file(p), original)


if __name__ == '__main__':
    unittest.main()
