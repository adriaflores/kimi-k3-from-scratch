"""CPU-only smoke/unit tests; no CUDA, no remote access, no trainer invocation."""
import importlib.util
import json
import math
import pathlib
import tempfile
import unittest
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('m73_ops', ROOT / 'scripts/m7/m73.py')
m73 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m73)
BUNDLE = ROOT / 'experiments/m7/m73'


class M73Tests(unittest.TestCase):
    def test_approved_yaml_and_m6_change_contract(self):
        r = m73.validate_recipe(BUNDLE / 'pilot_80m.yaml', BUNDLE)
        self.assertEqual(r['tokens'], 80019456)
        self.assertEqual(r['recipe_sha256'], m73.REQUIRED_YAML_SHA)
        self.assertEqual(r['changes_from_m6'], sorted(['max_steps', 'log_interval',
                       'eval_interval', 'eval_batches', 'checkpoint_interval']))
        self.assertEqual(len(m73.EXPECTED_EVENTS), 20)
        self.assertEqual(m73.EXPECTED_EVENTS[-2:], [2432,2442])

    def test_any_learning_change_blocked(self):
        original=json.loads((BUNDLE / 'm6_baseline_train_config.json').read_text())
        c = original.copy()
        for name in ['max_steps', 'log_interval','eval_interval','eval_batches','checkpoint_interval']:
            c[name] = {'max_steps':2442, 'log_interval':32, 'eval_interval':128,
                       'eval_batches':8, 'checkpoint_interval':1832}[name]
        c['learning_rate'] = 0.001
        with self.assertRaises(ValueError):
            m73.invariant_config(c, original)

    def test_scheduler_reference_is_global_horizon(self):
        import yaml
        c = yaml.safe_load((BUNDLE / 'pilot_80m.yaml').read_text())
        def lr(step):
            if step <= c['warmup_steps']:
                return c['learning_rate'] * step / c['warmup_steps']
            p = min(1.0, (step - c['warmup_steps']) / (c['decay_steps'] - c['warmup_steps']))
            return c['learning_rate'] * (c['min_lr_ratio'] +
                   (1 - c['min_lr_ratio']) * 0.5 * (1 + math.cos(math.pi * p)))
        for step, expected in [(1,1.6375545851528385e-6),(1832,0.003),
                               (2442,0.002999924562357562),(183106,0.0003)]:
            self.assertAlmostEqual(lr(step),expected,places=11)

    def test_atomic_copy_checksum_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = pathlib.Path(tmp)/'last.pt',pathlib.Path(tmp)/'warmup.pt'
            src.write_bytes(b'A' * 12001)
            receipt=m73.atomic_snapshot(src,dst,min_bytes=1)
            self.assertEqual(receipt['sha256'],m73.sha(src))
            self.assertEqual(receipt['bytes'],12001)
            with self.assertRaises(FileExistsError):
                m73.atomic_snapshot(src,dst,min_bytes=1)
            self.assertFalse(dst.with_name('warmup.pt.partial').exists())

    def test_no_resume_invocation_and_unique_run_dir(self):
        worker=(ROOT / 'scripts/m7/worker_m73.sh').read_text()
        run=(ROOT / 'scripts/m7/run_m73.sh').read_text()
        self.assertIn('--run-dir "$RUN"',worker)
        self.assertNotIn('--resume', '\n'.join(x for x in worker.splitlines() if not x.lstrip().startswith('#')))
        self.assertIn('test ! -e "$OUT" && test ! -e "$RUN"',run)
        self.assertIn('M73_START',run)

    def test_watcher_triggers_only_after_committed_warmup_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=pathlib.Path(tmp)
            rd, out=t/'run', t/'logs'
            rd.mkdir(); out.mkdir()
            (rd/'last.pt').write_bytes(b'fake-but-post-commit')
            rows=[{'event':'session_start','step':0},
                  {'event':'checkpoint','step':1832},
                  {'event':'checkpoint','step':2442}]
            (rd/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            def fake_snapshot(src,dst):
                dst.write_bytes(pathlib.Path(src).read_bytes())
                return {'bytes': len(dst.read_bytes()),'sha256':m73.sha(dst)}
            with mock.patch.object(m73,'atomic_snapshot',side_effect=fake_snapshot), \
                 mock.patch.object(m73,'inspect_checkpoint',return_value={'step':1832}):
                rc=m73.watch(type('A',(),{'run_dir':str(rd),'log_dir':str(out)})())
            self.assertEqual(rc,0)
            self.assertEqual(m73.load(out/'warmup_receipt.json')['step'],1832)
            self.assertEqual((out/'warmup_step1832.pt').read_bytes(),b'fake-but-post-commit')

    def test_missing_events_cannot_report_training_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=pathlib.Path(tmp)
            rd, out=t/'run',t/'logs'
            rd.mkdir();out.mkdir()
            (rd/'metrics.jsonl').write_text(json.dumps({'event':'session_start','step':0,'tokens_seen':0})+'\n')
            rc=m73.summary(type('A',(),{'run_dir':str(rd),'log_dir':str(out)})())
            self.assertEqual(rc,3)
            self.assertEqual(m73.load(out/'summary.json')['status'],'failed')

    def test_package_excludes_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=pathlib.Path(tmp)
            logdir, rundir=t/'art',t/'run'
            logdir.mkdir();rundir.mkdir()
            (logdir/'training.log').write_text('hello')
            (rundir/'metrics.jsonl').write_text('{"event":"checkpoint"}\n')
            (rundir/'last.pt').write_bytes(b'fake')
            with mock.patch.object(m73,'BUNDLE',BUNDLE):
                m73.package(type('A',(),{'log_dir':str(logdir),'run_dir':str(rundir)})())
            with zipfile.ZipFile(logdir/'m73_results.zip') as z:
                self.assertIn('trainer/metrics.jsonl', z.namelist())
                self.assertNotIn('trainer/last.pt',z.namelist())
                self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))
            self.assertEqual(m73.sha(logdir/'m73_results.zip'),
                             (logdir/'m73_results.zip.sha256').read_text().split()[0])


if __name__ == '__main__': unittest.main()
