"""M8.2 local operational tests: no GPU, torch, vast, weights or real training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.m8 import m82, backup_client_m82 as client


def cp_event(step):
    return {'schema_version':1,'event':'checkpoint','step':step,'tokens_seen':step*32768,'skipped_updates':0}


def train_event(step=2496,moe=None):
    return {'schema_version':1,'event':'train','step':step,'skipped_updates':0,
            'train_loss':4.0,'grad_norm_max':1.5,'nonfinite_grad_norms':0,
            'skipped_updates_in_window':0,'lr':0.003,'qk_clip':{'max_logit':44},'moe':moe or {}}


class M82Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.run,self.art=m82.production_paths(self.root)
        self.run.mkdir(parents=True); (self.art/'snapshots').mkdir(parents=True)
        (self.root/'reference').mkdir()

    def register(self,step,data=None):
        if data is None:data=f'step{step}'.encode()
        dest=self.art/'snapshots'/f'snapshot_step_{step}.pt'
        src=self.run/'last.pt';src.write_bytes(data)
        return m82.copy_snapshot(src,dest,step,min_bytes=1,validator=lambda p,s:self.assertEqual(p.read_bytes(),data))

    def test_production_paths_are_fixed(self):
        with self.assertRaises(ValueError):m82.production_paths(self.root,'../../m73')

    def test_step_cadence(self):
        for s in (4096,8192,183106):self.assertTrue(m82.step_allowed(s))
        for s in (2442,4095,4100,183107,True):self.assertFalse(m82.step_allowed(s))

    def test_atomic_snapshot_and_header_hook(self):
        row=self.register(4096)
        self.assertEqual(row['step'],4096)
        self.assertEqual(row['sha256'],m82.sha(self.run/'last.pt'))
        self.assertTrue((self.art/'snapshots/snapshot_step_4096.json').is_file())
        with self.assertRaises(FileExistsError):self.register(4096)

    def test_bad_snapshot_never_published(self):
        p=self.run/'last.pt';p.write_bytes(b'invalid')
        with self.assertRaisesRegex(ValueError,'bad checkpoint'):
            m82.copy_snapshot(p,self.art/'snapshots/snapshot_step_4096.pt',4096,
                              min_bytes=1,validator=lambda p,s:(_ for _ in ()).throw(ValueError('bad checkpoint')))
        self.assertEqual(list((self.art/'snapshots').iterdir()),[])

    def test_incomplete_checkpoint_too_small(self):
        p=self.run/'last.pt';p.write_bytes(b'abc')
        with self.assertRaises(ValueError):m82.copy_snapshot(p,self.art/'snapshots/snapshot_step_4096.pt',4096,min_bytes=4,
                                      validator=lambda *_:None)

    def test_missing_checkpoint_receipt_fails(self):
        (self.art/'snapshots/snapshot_step_4096.json').write_text(json.dumps({'step':4096,'file':'snapshot_step_4096.pt','bytes':100}))
        with self.assertRaises(ValueError):m82.snapshot_receipts(self.art)

    def test_external_ack_rejects_wrong_sha(self):
        self.register(4096)
        with self.assertRaises(ValueError):m82.ack_backup(self.art,4096,'0'*64)
        self.assertFalse((self.art/'offsite_ack.json').exists())

    def test_ack_monotonic(self):
        a=self.register(4096);m82.ack_backup(self.art,4096,a['sha256'])
        b=self.register(8192);m82.ack_backup(self.art,8192,b['sha256'])
        with self.assertRaises(ValueError):m82.ack_backup(self.art,4096,a['sha256'])

    def test_offsite_due_first_and_third_checkpoint(self):
        a=self.register(4096)
        self.assertEqual(m82.backup_due(self.art)['step'],4096)
        m82.ack_backup(self.art,4096,a['sha256'])
        for s in (8192,12288):
            self.register(s);self.assertIsNone(m82.backup_due(self.art))
        self.register(16384)
        self.assertEqual(m82.backup_due(self.art)['step'],16384)

    def test_retention_requires_external_backup(self):
        for s in (4096,8192,12288,16384):self.register(s)
        with self.assertRaises(ValueError):m82.retention(self.art)
        self.assertEqual(len(m82.snapshot_receipts(self.art)),4)

    def test_retention_only_removes_verified_oldest(self):
        a=self.register(4096);m82.ack_backup(self.art,4096,a['sha256'])
        for s in (8192,12288,16384):self.register(s)
        self.assertEqual(m82.retention(self.art),[4096])
        self.assertEqual([r['step'] for r in m82.snapshot_receipts(self.art)],[8192,12288,16384])

    def test_tail_incomplete_line_defers(self):
        target=self.run/'metrics.jsonl';target.write_bytes(json.dumps(cp_event(4096)).encode()+b'\n'+b'{"event":"train"')
        tail=m82.MetricTail(target)
        self.assertEqual(len(tail.read()),1)
        self.assertEqual(tail.read(),[])
        with target.open('ab') as f:f.write(b',"schema_version":1,"step":4096,"skipped_updates":0}\n')
        self.assertEqual(len(tail.read()),1)

    def test_tail_committed_corruption_fails(self):
        p=self.run/'metrics.jsonl';p.write_bytes(b'{garbage}\n')
        with self.assertRaises(ValueError):m82.MetricTail(p).read()

    def test_corrupted_metrics_no_snapshot(self):
        p=self.run/'metrics.jsonl';p.write_bytes(b'{garbage}\n')
        with self.assertRaises(ValueError):m82.process_metrics(m82.MetricTail(p),m82.Monitor(),self.run,self.art,
                                                snapshot_validator=lambda *_:None,min_snapshot_bytes=1)
        self.assertFalse(list((self.art/'snapshots').iterdir()))

    def test_checkpoint_event_required_for_snapshot(self):
        (self.run/'last.pt').write_bytes(b'step4096')
        p=self.run/'metrics.jsonl';p.write_text(json.dumps(train_event())+'\n')
        self.assertEqual(m82.process_metrics(m82.MetricTail(p),m82.Monitor(),self.run,self.art,
                         snapshot_validator=lambda *_:None,min_snapshot_bytes=1),[])

    def test_metrics_checkpoint_triggers_snapshot_once(self):
        (self.run/'last.pt').write_bytes(b'step4096')
        p=self.run/'metrics.jsonl';p.write_text(json.dumps(cp_event(4096))+'\n')
        tail=m82.MetricTail(p)
        self.assertEqual(m82.process_metrics(tail,m82.Monitor(),self.run,self.art,
                                           snapshot_validator=lambda *_:None,min_snapshot_bytes=1),[4096])
        self.assertEqual(m82.process_metrics(tail,m82.Monitor(),self.run,self.art,
                                           snapshot_validator=lambda *_:None,min_snapshot_bytes=1),[])

    def test_rejects_wrong_step_in_snapshot(self):
        (self.run/'last.pt').write_bytes(b'step8192')
        p=self.run/'metrics.jsonl';p.write_text(json.dumps(cp_event(4096))+'\n')
        with self.assertRaisesRegex(ValueError,'wrong step'):
            m82.process_metrics(m82.MetricTail(p),m82.Monitor(),self.run,self.art,
                                snapshot_validator=lambda p,s:(_ for _ in ()).throw(ValueError('wrong step')),
                                min_snapshot_bytes=1)
        self.assertEqual(list((self.art/'snapshots').iterdir()),[])

    def test_monitor_nonfinite_grad_reject(self):
        e=train_event();e['nonfinite_grad_norms']=1
        with self.assertRaises(RuntimeError):m82.Monitor().feed(e)

    def test_monitor_overflow_reject(self):
        e={'schema_version':1,'event':'overflow','step':2490,'skipped_updates':1}
        with self.assertRaises(RuntimeError):m82.Monitor().feed(e)

    def test_moe_diversity_balanced(self):
        metrics={'layer':{'expert_counts':[3]*64,'qb_biases':[0.0]*64}}
        rows=m82.moe_diversity(metrics)
        self.assertAlmostEqual(rows[0]['shannon'],64)
        self.assertAlmostEqual(rows[0]['simpson'],64)
        self.assertEqual(rows[0]['zero_load_experts'],0)

    def test_moe_diversity_concentrated_warning_not_stop(self):
        moe={'layer':{'expert_counts':[100]+[0]*63,'qb_biases':[0]*64}}
        monitor=m82.Monitor();monitor.feed(train_event(moe=moe))
        self.assertTrue(monitor.warnings)

    def test_persistent_validation_deterioration_warns(self):
        monitor=m82.Monitor()
        for n,val in enumerate([3.0,4.0,4.1,4.2]):
            monitor.feed({'schema_version':1,'event':'val','step':2500+n,'skipped_updates':0,'val_loss':val})
        self.assertTrue(monitor.warnings)

    def test_backup_client_rejects_untrusted_remote_request(self):
        with self.assertRaises(ValueError):client.receive('vast',self.root,{'step':4096,'sha256':'x','bytes':2_500_000_000,'file':'../passwd'})

    def test_backup_client_bad_download_never_acks(self):
        rec={'step':4096,'sha256':'0'*64,'bytes':2_000_000_001,'file':'snapshot_step_4096.pt'}
        invocations=[]
        def fake_runner(cmd,**kwargs):
            invocations.append(cmd)
            if cmd[0]=='scp':Path(cmd[-1]).write_bytes(b'fake')
        with self.assertRaises(ValueError):client.receive('vast',self.root,rec,runner=fake_runner)
        self.assertEqual([x[0] for x in invocations],['scp'])

    def test_backup_client_idempotent_local_protection(self):
        data=b'fake-snapshot'; local=self.root/client.RUN_ID;local.mkdir()
        p=local/'snapshot_step_4096.pt';p.write_bytes(data)
        rec={'step':4096,'sha256':m82.sha(p),'bytes':2_000_000_001,'file':p.name}
        with self.assertRaises(ValueError):client.receive('vast',self.root,rec,runner=lambda *_a,**_kw:None)

    def test_gate_requires_remote_preflight(self):
        with self.assertRaises(FileNotFoundError):m82.verify_gate(self.root,'drill',new_run=True)

    def test_gate_blocks_continuous_before_recovery(self):
        receipt={'stage':'remote','status':'PASS','training_started':False,
                 'checks':{'recipe':{'sha256':m82.APPROVED_YAML_SHA},'checkpoint':{'sha256':m82.SOURCE_SHA,'step':m82.START_STEP},
                           'cost':{},'storage':{},'reference':{},'runtime':{},'corpus':{},
                           'paths':{'proposed_run_dir':str(self.run)}}}
        p=self.root/'artifacts/m8/m81/preflight.json';m82.atomic_json(p,receipt)
        c=self.root/'experiments/m7/m75_gate/recipe/m8_full_6b.yaml';c.parent.mkdir(parents=True)
        # Keep real recipe bytes: the bundled file is reused, not mutated.
        c.write_bytes((ROOT/'experiments/m7/m75_gate/recipe/m8_full_6b.yaml').read_bytes())
        (self.run/'run.json').write_text('{}');(self.run/'last.pt').write_bytes(b'test')
        with self.assertRaises(FileNotFoundError):m82.verify_gate(self.root,'continuous',new_run=False)

    def test_supervisor_fails_closed_when_training_process_fails(self):
        # Pure fake process that emits no checkpoints; no torch/GPU involved.
        command=[sys.executable,'-c','import sys; print("fake failure");sys.exit(13)']
        rc=m82.supervisor(self.root,'drill',python=sys.executable,command=command,
                          interval=.02,stall_seconds=2,validator=lambda *_:None,min_snapshot_bytes=1)
        self.assertEqual(rc,2)
        self.assertEqual(m82.load_json(self.art/'runner_state.json')['state'],'PAUSED_INCIDENT')

    def test_supervisor_stops_after_first_checkpoint(self):
        # The only training executed is a small fake child that writes metrics and fake checkpoint.
        program=("import pathlib,json,time,sys\n"
                 "r=pathlib.Path(sys.argv[1]);r.mkdir(parents=True,exist_ok=True)\n"
                 "(r/'last.pt').write_bytes(b'fake-data-step4096')\n"
                 "e={'schema_version':1,'event':'checkpoint','step':4096,'tokens_seen':4096*32768,'skipped_updates':0}\n"
                 "(r/'metrics.jsonl').write_text(json.dumps(e)+'\\n')\n"
                 "time.sleep(8)\n")
        command=[sys.executable,'-c',program,str(self.run)]
        rc=m82.supervisor(self.root,'drill',python=sys.executable,command=command,
                          interval=.05,stall_seconds=3,validator=lambda *_:None,min_snapshot_bytes=1)
        self.assertEqual(rc,0)
        self.assertEqual(m82.load_json(self.art/'runner_state.json')['state'],'AWAITING_M83_RECOVERY_GATE')
        self.assertEqual(m82.snapshot_receipts(self.art)[0]['step'],4096)

    def test_backup_due_at_final_step_even_with_recent_ack(self):
        a=self.register(180224);m82.ack_backup(self.art,180224,a['sha256'])
        self.register(183106)
        self.assertEqual(m82.backup_due(self.art)['step'],183106)

    def test_backup_client_success_ack_after_verified_bytes(self):
        import hashlib
        # Sparse 2 GB fixture: tests file-size contract without allocating 2 GB.
        size=2_000_000_001
        good='a'*64
        row={'step':4096,'sha256':good,'bytes':size,'file':'snapshot_step_4096.pt'}
        calls=[]
        def fake_runner(cmd,**kwargs):
            calls.append(cmd[0])
            if cmd[0]=='scp':
                with open(cmd[-1],'wb') as f:f.truncate(size)
            if cmd[0]=='ssh':
                self.assertTrue((self.root/client.RUN_ID/row['file']).exists())
                self.assertTrue((self.root/client.RUN_ID/(row['file']+'.verified.json')).exists())
        with mock.patch.object(client,'sha',return_value=good):
            result=client.receive('vast',self.root,row,runner=fake_runner)
        self.assertIn('VERIFIED_AND_ACKNOWLEDGED',result)
        self.assertEqual(calls,['scp','ssh'])

    def test_backup_lag_limit_status(self):
        first=self.register(4096);m82.ack_backup(self.art,4096,first['sha256'])
        for step in (8192,12288,16384,20480): self.register(step)
        self.assertEqual(m82.checkpoint_coverage(self.art)['status'],'BACKUP_LAG_LIMIT')

    def test_finished_run_requires_final_offsite_ack(self):
        row=self.register(183106)
        m82.atomic_json(self.art/'runner_state.json',{'state':'COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP'})
        self.assertEqual(m82.status(self.root)['runner']['state'],'COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP')
        m82.ack_backup(self.art,183106,row['sha256'])
        self.assertEqual(m82.status(self.root)['runner']['state'],'READY_FOR_FINAL_VERIFICATION_AND_SHUTDOWN')

    def test_continuous_command_line_guard_rejects_no_m83(self):
        args=['guard','--root',str(self.root),'--phase','continuous']
        self.assertEqual(m82.main(args),2)

    def test_missing_moe_bias_is_hard_failure(self):
        m=m82.Monitor()
        with self.assertRaises(RuntimeError):
            m.feed(train_event(moe={'layer':{'expert_counts':[1]*64,'qb_biases':[0]*63}}))

    def test_no_unchanged_tar_recipe_required_in_this_overlay(self):
        self.assertEqual(m82.sha(ROOT/'experiments/m7/m75_gate/recipe/m8_full_6b.yaml'),m82.APPROVED_YAML_SHA)


if __name__=='__main__':unittest.main()
