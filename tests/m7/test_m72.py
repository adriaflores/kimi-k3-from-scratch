"""Offline M7.2 contract and compatibility tests: no PyTorch/GPU required."""
from __future__ import annotations
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/m7'))
import scripts.m7.m72_validate as m72


class M72Tests(unittest.TestCase):
    def test_m6_baseline_is_preserved_except_observation(self):
        r=m72.validate_static(require_m71=False)
        self.assertEqual(sorted(r['changes_from_m6']),sorted(m72.read_json(m72.CONTRACT)['allowed_m6_recipe_changes']))
        self.assertEqual(r['configuration']['muon_batched'],True)
        self.assertEqual(r['configuration']['muon_compile'],True)
        self.assertEqual(r['configuration']['precision'],'bf16')
        self.assertEqual(r['configuration']['learning_rate'],0.003)
        self.assertFalse(r['configuration']['overfit_batch'])
    def test_schedule_token_math(self):
        c=m72.validate_static(require_m71=False)
        self.assertEqual(c['token_arithmetic']['tokens_per_step'],32768)
        self.assertEqual(c['token_arithmetic']['warmup_tokens'],60_030_976)
        self.assertEqual(c['token_arithmetic']['pilot_tokens'],80_019_456)
        self.assertEqual(c['token_arithmetic']['decay_horizon_tokens'],6_000_017_408)
    def test_eval_and_checkpoint_steps(self):
        v=m72.validate_static(require_m71=False)['events']
        self.assertEqual(v['validation_steps'][:2],[128,256])
        self.assertEqual(v['validation_steps'][-2:],[2432,2442])
        self.assertEqual(v['validation_count'],20)
        self.assertEqual(v['checkpoint_steps'],[1832,2442])
        self.assertEqual(v['validation_tokens_each'],262144)
    def test_m71_local_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/'local_preflight.json'
            with patch.object(m72,'LOCAL_M71_REPORT',report):
                with self.assertRaises(FileNotFoundError):m72.validate_static()
                report.write_text(json.dumps({'status':'passed','files':254,'bytes':12_633_283_564}))
                self.assertEqual(m72.validate_static()['configuration']['max_steps'],2442)
    def test_no_training_in_m72_driver(self):
        source=(ROOT/'scripts/m7/m72_validate.py').read_text()
        self.assertNotIn('trainer.train(',source)
        self.assertNotIn('runpy.run_path(',source)
        self.assertNotIn('torch.save(',source)
        self.assertIn('training_started\':False',source)
    def test_yaml_modification_rejected(self):
        original=m72.CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'pilot.yaml'
            cfg.write_text(original.read_text().replace('learning_rate: 0.003','learning_rate: 0.01'))
            with patch.object(m72,'CONFIG',cfg):
                with self.assertRaisesRegex(ValueError,'Pilot YAML differs'):m72.validate_static(False)
    def test_scientific_float_literal_is_pyyaml_compatible(self):
        # PyYAML (used by the real trainer) treats 1e-08 as a string.
        config = m72.CONFIG.read_text(encoding='utf-8')
        self.assertIn('adam_eps: 1.0e-08', config)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'pilot_bad.yaml'
            bad.write_text(config.replace('adam_eps: 1.0e-08', 'adam_eps: 1e-08'), encoding='utf-8')
            with patch.object(m72, 'CONFIG', bad):
                with self.assertRaisesRegex(ValueError, 'PyYAML would load'):
                    m72.validate_static(require_m71=False)
        # Run the same semantic YAML parse as the frozen trainer when installed.
        try:
            import yaml
        except ImportError:
            return
        loaded = yaml.safe_load(config)
        self.assertIsInstance(loaded['adam_eps'], float)
        self.assertEqual(loaded['adam_eps'], 1.0e-8)

    def test_m72_shell_scripts_are_present(self):
        for f in ['local_preflight_m72.sh','launch_m72.sh','run_m72.sh','status_m72.sh','download_m72.sh']:
            self.assertTrue((ROOT/'scripts/m7'/f).is_file())

if __name__=='__main__':unittest.main()
