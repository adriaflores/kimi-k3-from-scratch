"""CPU-only tests; generated trajectories are synthetic, not 300M measurements."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import  scripts.analyze_pretraining_validation as analysis
import scripts.run_pretraining_validation as runner
TEMPLATES = json.loads((ROOT/'tests/fixtures/pretraining_validation/templates.json').read_text())


def oracle():
	module = types.ModuleType('branch6_trainer_oracle')
	module.__dict__.update({'dataclass':dataclass, 'math':math})
	sys.modules[module.__name__] = module
	nodes = [n for n in ast.parse((ROOT/'src/kimi_k3/training/qk_clip.py').read_text()).body
			 if isinstance(n,ast.FunctionDef) and n.name in ('_positive','_chunk_size')]
	nodes += [n for n in ast.parse((ROOT/'src/kimi_k3/training/trainer.py').read_text()).body
			  if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('TrainConfig','learning_rate_at_step')]
	exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_trainer_definitions','exec'), module.__dict__)
	return module


ORACLE = oracle()


def dump(path, records):
	path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records), encoding='utf-8')


def candidate(study, plan, entry, until=1000):
	manifest = deepcopy(TEMPLATES['run_manifest'])
	manifest['sessions'] = manifest['sessions'][:1]
	manifest['run_id'] = entry['id']
	session = manifest['sessions'][0]
	session['session_id'] = entry['id']+'_synthetic'
	session['model_config'] = TEMPLATES['model']
	session['parameter_count'] = 294676808
	config = analysis.expected_recipe(plan['common'],entry)
	session['train_config'] = config
	records = []
	templates = {r['event']:r for r in TEMPLATES['records']}
	for step in range(10,until+1,10):
		r = deepcopy(templates['train'])
		r.update({'step':step,'tokens_seen':step*2048,'lr':ORACLE.learning_rate_at_step(ORACLE.TrainConfig(**config),step),
				  'train_loss':9-step/1000,'train_seconds':20.48,'train_tokens_per_second':1000.0})
		r['qk_clip']['observed_head_updates'] = 320
		for layer in r['qk_clip']['layers'].values():
			for field in ['max_logits','minimum_gamma','clipped_head_updates']:
				layer[field] = [layer[field][0]]*8
		records.append(r)
	for step in range(125,until+1,125):
		r=deepcopy(templates['val'])
		r.update({'step':step,'tokens_seen':step*2048,'val_loss':6+(0 if entry['learning_rate']==.003 else .2)+2*(1000-step)/1000})
		records.append(r)
	records.sort(key=lambda r:(r['step'],r['event']=='val'))
	records.insert(0,deepcopy(templates['session_start']))
	if until==1000:
		records.append(deepcopy(templates['session_end']))
	for r in records:
		r['run_id']=manifest['run_id'];r['session_id']=session['session_id']
	directory=study/'runs'/entry['id'];directory.mkdir(parents=True)
	analysis.write_json(directory/'run.json',manifest)
	dump(directory/'metrics.jsonl',records)
	return manifest,records


class Branch6Tests(unittest.TestCase):
	def setUp(self):
		self.tmp=tempfile.TemporaryDirectory()
		self.path=Path(self.tmp.name)
		self.study=self.path/'study';self.study.mkdir()
		self.data=self.path/'data';self.data.mkdir()
		analysis.write_json(self.data/'metadata.json',TEMPLATES['metadata'])
		self.plan=runner.load_plan(ROOT)
		self.state=runner.create_state(ROOT,self.study,self.data,self.plan)
		self.state['environment']=self.plan['reference']['environment']
		for entry in self.plan['branch6_validation_runs']:
			self.state['runs'][entry['id']]={'status':'completed','error':None,'attempts':[{'wall_seconds':2400,'returncode':0}]}
			candidate(self.study,self.plan,entry)
		runner.save_state(self.study,self.state)
		self.entry=self.plan['branch6_validation_runs'][0]
		self.metrics=self.study/'runs'/self.entry['id']/'metrics.jsonl'

	def tearDown(self): self.tmp.cleanup()
	def summary(self):
		e=analysis.Evidence(self.study)
		try:return analysis.summarize_study(e)
		finally:e.close()
	def mutate_record(self,fn):
		records=[json.loads(l) for l in self.metrics.read_text().splitlines()]
		fn(records);dump(self.metrics,records)

	def test_configs_match_actual_trainer_schedule(self):
		for entry in self.plan['branch6_validation_runs']:
			config=yaml.safe_load((ROOT/entry['training_config']).read_text())
			cfg=ORACLE.TrainConfig(**config)
			for step in [1,10,11,125,500,1000]:
				self.assertAlmostEqual(analysis.lr_at_step(config,step),ORACLE.learning_rate_at_step(cfg,step))
			self.assertEqual(config['max_steps']*2048,2048000)

	def test_complete_pair(self):
		result=self.summary()
		self.assertTrue(result['selection']['grid_complete_and_valid'])
		self.assertEqual(result['selection']['best_observed_run'],self.entry['id'])
		self.assertEqual(len(result['candidates'][0]['validation']),8)

	def test_missing_final_evaluation(self):
		self.mutate_record(lambda rs:rs.remove(next(r for r in rs if r['event']=='val' and r['step']==1000)))
		self.assertFalse(self.summary()['candidates'][0]['eligible'])

	def test_skipped_update(self):
		self.mutate_record(lambda rs:next(r for r in rs if r['event']=='train').update({'skipped_updates_in_window':1}))
		self.assertFalse(self.summary()['candidates'][0]['eligible'])

	def test_nonfinite_gradient(self):
		self.mutate_record(lambda rs:next(r for r in rs if r['event']=='train').update({'nonfinite_grad_norms':1}))
		self.assertFalse(self.summary()['candidates'][0]['eligible'])

	def test_wrong_head_count(self):
		self.mutate_record(lambda rs:next(r for r in rs if r['event']=='train')['qk_clip'].update({'observed_head_updates':200}))
		self.assertFalse(self.summary()['candidates'][0]['eligible'])

	def test_source_tampering_blocks_selection(self):
		p=self.study/'source_snapshot/scripts/train.py';p.write_text(p.read_text()+'\n# changed\n')
		result=self.summary()
		self.assertTrue(result['evidence_issues'])
		self.assertFalse(result['selection']['grid_complete_and_valid'])

	def test_recipe_mismatch(self):
		p=self.study/'runs'/self.entry['id']/'run.json'
		m=json.loads(p.read_text());m['sessions'][0]['train_config']['gradient_accumulation_steps']=2
		analysis.write_json(p,m)
		self.assertFalse(self.summary()['candidates'][0]['eligible'])

	def test_partial_has_no_winner_for_pair(self):
		self.mutate_record(lambda rs:rs.remove(next(r for r in rs if r['event']=='session_end')))
		r=self.summary()
		b=json.loads((ROOT/runner.REFERENCE).read_text())
		self.assertEqual(analysis.validation_review(r,b)['status'],'awaiting_valid_300m_pair')

	def test_resume_rolls_back_old_future(self):
		p=self.study/'runs'/self.entry['id']/'run.json'
		m=json.loads(p.read_text());old=m['sessions'][0]
		new=deepcopy(old);new['session_id']='synthetic_resume';new['rng_source']='checkpoint'
		new['start_state'].update({'step':500,'tokens_seen':1024000})
		new['start_state']['loaders']['train']['position']=1024000
		new['start_state']['loaders']['val']['position']=65536
		m['sessions'].append(new)
		records=[json.loads(l) for l in self.metrics.read_text().splitlines()]
		later=[deepcopy(r) for r in records if r['step']>500 or r['event']=='session_end']
		for r in later:r['session_id']=new['session_id']
		records+=later
		analysis.write_json(p,m);dump(self.metrics,records)
		row=self.summary()['candidates'][0]
		self.assertTrue(row['eligible'],row['issues'])
		self.assertEqual(row['discarded_logged_tokens'],1024000)
		self.assertEqual(len(row['train']),100)

	def test_pending_restart_preserves_attempt(self):
		run=self.study/'runs'/self.entry['id']
		argv,log=runner.prepare_attempt(ROOT,self.study,self.data,self.entry,self.state['runs'][self.entry['id']])
		self.assertFalse(run.exists())
		self.assertTrue((self.study/'abandoned_attempts'/self.entry['id']/'attempt_01/run.json').exists())
		self.assertNotIn('--resume',argv)

	def test_checkpoint_is_resumed_and_excluded_from_zip(self):
		run=self.study/'runs'/self.entry['id'];(run/'last.pt').write_bytes(b'synthetic placeholder')
		argv,_=runner.prepare_attempt(ROOT,self.study,self.data,self.entry,self.state['runs'][self.entry['id']])
		self.assertIn('--resume',argv)
		z=runner.archive_results(self.study)
		with ZipFile(z) as archive:
			self.assertFalse(any(n.endswith('.pt') for n in archive.namelist()))
			self.assertIn('study_state.json',archive.namelist())

	def test_full_analysis_and_figures(self):
		result=analysis.analyze(self.study)
		self.assertEqual(result['validation_review']['lr_ranking_transfer'],'3e-3_lower_loss')
		for name in ['loss_curves','proxy_to_300m','gradients','qk_clip','moe_health','efficiency']:
			for extension in ['png','pdf']:
				self.assertTrue((self.study/'analysis/figures'/f'{name}.{extension}').is_file())
		self.assertTrue((self.study/'analysis/report.md').is_file())

	def test_runner_stops_on_failure_and_saves_partial_zip(self):
		target=self.path/'failed_study'
		with patch.object(runner,'environment',return_value=self.plan['reference']['environment']), \
			 patch.object(runner,'run_logged',return_value=(1,2.0,False)), \
			 patch.object(runner,'analyze',return_value=None), redirect_stdout(io.StringIO()):
			code=runner.main(['--data-dir',str(self.data),'--output-dir',str(target)])
		self.assertEqual(code,1)
		self.assertTrue((target/'branch6_results.zip').is_file())
		state=json.loads((target/'study_state.json').read_text())
		self.assertEqual(state['runs'][self.entry['id']]['status'],'failed')
		second=self.plan['branch6_validation_runs'][1]['id']
		self.assertEqual(state['runs'][second]['status'],'pending')

	def test_runner_cancel_exports_interrupted_state(self):
		target=self.path/'interrupted_study'
		with patch.object(runner,'environment',return_value=self.plan['reference']['environment']), \
			 patch.object(runner,'run_logged',return_value=(130,2.0,True)), \
			 patch.object(runner,'analyze',return_value=None), redirect_stdout(io.StringIO()):
			code=runner.main(['--data-dir',str(self.data),'--output-dir',str(target)])
		self.assertEqual(code,130)
		with ZipFile(target/'branch6_results.zip') as z:
			state=json.loads(z.read('study_state.json'))
		self.assertEqual(state['runs'][self.entry['id']]['status'],'interrupted')

	def test_runner_resume_skips_completed_candidates(self):
		with patch.object(runner,'environment',return_value=self.plan['reference']['environment']), \
			 patch.object(runner,'run_logged',side_effect=AssertionError('completed candidate rerun')), \
			 patch.object(runner,'analyze',return_value=None), redirect_stdout(io.StringIO()):
			code=runner.main(['--output-dir',str(self.study),'--resume'])
		self.assertEqual(code,0)

	def test_dry_run_never_probes_cuda(self):
		target=self.path/'new_study'
		with patch.object(runner,'ROOT',ROOT), patch.object(runner,'environment',side_effect=AssertionError('CUDA probe forbidden')), redirect_stdout(io.StringIO()):
			self.assertEqual(runner.main(['--data-dir',str(self.data),'--output-dir',str(target),'--dry-run']),0)
		self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
