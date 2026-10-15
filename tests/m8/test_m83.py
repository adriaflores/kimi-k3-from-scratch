"""M8.3 offline evidence and incident tests. NO Vast, GPU or optimizer updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.m8 import m82,m83


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.run,self.art,self.m83art=m83.paths(self.root)
        self.run.mkdir(parents=True)
        (self.root/'reference').mkdir(parents=True)
        (self.art/'snapshots').mkdir(parents=True)
        self.m83art.mkdir(parents=True)
        self.original=self.root/m82.SOURCE.relative_to(m82.BASE)
        self.original.parent.mkdir(parents=True)
        self.original.write_bytes(b'source-original-immutable-simulated')
        recipe=self.root/'experiments/m7/m75_gate/recipe/m8_full_6b.yaml'
        recipe.parent.mkdir(parents=True)
        recipe.write_bytes((ROOT/'experiments/m7/m75_gate/recipe/m8_full_6b.yaml').read_bytes())
        pre={'stage':'remote','status':'PASS','training_started':False,
             'checks':{'recipe':{'sha256':m82.APPROVED_YAML_SHA},
                       'checkpoint':{'sha256':m82.SOURCE_SHA,'step':m82.START_STEP},
                       'paths':{'proposed_run_dir':str(self.run)},
                       'cost':{},'storage':{},'reference':{},'runtime':{},'corpus':{}}}
        m82.atomic_json(self.root/'artifacts/m8/m81/preflight.json',pre)
        self.run.joinpath('run.json').write_text(json.dumps({'schema_version':1,
            'run_id':m82.RUN_ID,'sessions':[{'checkpoint_path':str((self.run/'last.pt').resolve()),
             'start_state':{'step':2442},'resume_checkpoint':str(m82.SOURCE)}]}))
        self.run.joinpath('metrics.jsonl').write_text('')
        self.snapshot(4096)
        self.state('AWAITING_M83_RECOVERY_GATE','drill')
        self.ack(4096)
        self.patch_min=mock.patch.object(m82,'MIN_SNAPSHOT_BYTES',1);self.patch_min.start();self.addCleanup(self.patch_min.stop)
        # The source is deliberately a tiny fixture, not the real 2.5GB pilot.
        real_digest=m83.digest
        def fixture_digest(path):
            if Path(path)==self.original:return m82.SOURCE_SHA
            return real_digest(path)
        self.patch_digest=mock.patch.object(m83,'digest',side_effect=fixture_digest)
        self.patch_digest.start();self.addCleanup(self.patch_digest.stop)
        self.patch_tmux=mock.patch.object(m83,'_no_active_training')
        self.patch_tmux.start();self.addCleanup(self.patch_tmux.stop)

    def state(self,status,phase):
        m82.atomic_json(self.art/'runner_state.json',{'state':status,'phase':phase,'training_started':True})

    def snapshot(self,step):
        payload=(f'fake checkpoint {step}').encode()
        (self.run/'last.pt').write_bytes(payload)
        name=f'snapshot_step_{step}.pt'
        return m82.copy_snapshot(self.run/'last.pt',self.art/'snapshots'/name,step,
                                 min_bytes=1,validator=lambda *_:None)

    def ack(self,step):
        row=m83.step_receipt(self.art,step) if m82.MIN_SNAPSHOT_BYTES==1 else m82.snapshot_receipts(self.art)[-1]
        return m82.ack_backup(self.art,step,row['sha256'])

    def fake_probe(self,step=4096):
        row=m83.step_receipt(self.art,step)
        snap=self.art/'snapshots'/row['file']
        return {'status':'PASS','no_optimizer_updates_executed':True,
                'checkpoint_path':str(snap),'checkpoint_step':step,'restored_step':step,
                'checkpoint_format_version':7,'checkpoint_filesize':row['bytes'],
                'checkpoint_sha256_before':row['sha256'],'checkpoint_sha256_after':row['sha256'],
                'restored_train_position':step*32768,'restored_val_position':8*32768,
                'rng_verified':['python','numpy','torch_cpu','torch_cuda'],
                'optimizer_state_entries':1830,'model_parameters':294676808,
                'original_source_sha256':m82.SOURCE_SHA,
                'source_files_sha256_verified':35,'recipe_sha256':m82.APPROVED_YAML_SHA,
                'skipped_updates':0,'torch':'2.11.0+cu128','gpu':'NVIDIA H100 NVL'}

    def put_probe(self,step=4096):
        m82.atomic_json(self.m83art/'probe_receipt.json',self.fake_probe(step))

    def approve_first(self):
        self.put_probe()
        return m83._commit_gate(self.root,step=4096,phase='drill')


class M83Tests(Fixture):
    def test_local_drill_recovers_required_context(self):
        self.assertEqual(m83.drill_ready(self.root)['checkpoint_step'],4096)

    def test_drill_fails_before_offsite_ack(self):
        (self.art/'offsite_ack.json').unlink()
        with self.assertRaisesRegex(ValueError,'Off-Vast'):m83.drill_ready(self.root)

    def test_drill_fails_if_runner_not_stopped(self):
        self.state('RUNNING','drill')
        with self.assertRaisesRegex(ValueError,'Drill worker'):m83.drill_ready(self.root)

    def test_drill_fails_if_manifest_from_fresh_training(self):
        p=self.run/'run.json';data=json.loads(p.read_text());data['sessions'][0]['start_state']['step']=0
        p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'originate'):m83.drill_ready(self.root)

    def test_drill_fails_checkpoint_last_mismatch(self):
        (self.run/'last.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):m83.drill_ready(self.root)

    def test_drill_detects_modified_m7_source(self):
        self.original.write_bytes(b'modified')
        with mock.patch.object(m83,'digest',wraps=m82.sha):
            with self.assertRaises(ValueError):m83.drill_ready(self.root)

    def test_probe_wrong_rng_rejected(self):
        proof=self.fake_probe();proof['rng_verified']=['python','torch_cpu']
        with self.assertRaisesRegex(ValueError,'RNG'):m83._check_probe(
            proof,expected_step=4096,expected_sha=m83.step_receipt(self.art,4096)['sha256'],
            snapshot=self.art/'snapshots/snapshot_step_4096.pt')

    def test_probe_wrong_optimizer_rejected(self):
        proof=self.fake_probe();proof['optimizer_state_entries']=1829
        with self.assertRaisesRegex(ValueError,'optimizer'):m83._check_probe(
            proof,expected_step=4096,expected_sha=m83.step_receipt(self.art,4096)['sha256'],
            snapshot=self.art/'snapshots/snapshot_step_4096.pt')

    def test_probe_wrong_gpu_rejected(self):
        proof=self.fake_probe();proof['gpu']='NVIDIA A100'
        with self.assertRaisesRegex(ValueError,'H100'):m83._check_probe(
            proof,expected_step=4096,expected_sha=m83.step_receipt(self.art,4096)['sha256'],
            snapshot=self.art/'snapshots/snapshot_step_4096.pt')

    def test_drill_approval_requires_probe_file(self):
        with self.assertRaises(FileNotFoundError):m83._commit_gate(self.root,step=4096,phase='drill')
        self.assertFalse((self.m83art/'recovery_gate.json').exists())

    def test_drill_go_authorized_after_proof(self):
        gate=self.approve_first()
        self.assertEqual(gate['checkpoint_step'],4096)
        self.assertTrue(m83.verify_continuous_authorization(self.root)['ready_for_continuous'])
        # Same guard that M8.2 real launcher uses, no mocks on the guard itself.
        self.assertEqual(m82.verify_gate(self.root,'continuous',new_run=False)[0],self.run)

    def test_forged_simple_pass_rejected(self):
        m82.atomic_json(self.m83art/'recovery_gate.json',{'status':'PASS','checkpoint_step':4096,'run_id':m82.RUN_ID})
        with self.assertRaisesRegex(ValueError,'incomplete'):m83.verify_continuous_authorization(self.root)

    def test_probe_tampering_invalidates_go(self):
        self.approve_first();self.put_probe();data=m82.load_json(self.m83art/'probe_receipt.json')
        data['restored_step']=8192;m82.atomic_json(self.m83art/'probe_receipt.json',data)
        with self.assertRaisesRegex(ValueError,'Probe receipt was modified'):
            m83.verify_continuous_authorization(self.root)

    def test_last_checkpoint_corruption_blocks_start(self):
        self.approve_first()
        (self.run/'last.pt').write_bytes(b'fake checkpoint 4096CORRUPT')
        with self.assertRaises(ValueError):m82.verify_gate(self.root,'continuous',new_run=False)

    def test_simple_replay_of_approval_refused(self):
        self.approve_first()
        with self.assertRaises(FileExistsError):m83._commit_gate(self.root,step=4096,phase='drill')

    def test_missing_snapshot_blocks_gate(self):
        self.approve_first()
        (self.art/'snapshots/snapshot_step_4096.pt').unlink()
        with self.assertRaises(ValueError):m83.verify_continuous_authorization(self.root)

    def test_run_probe_must_be_true_process_not_fixture(self):
        # Fake runner used only to ensure orchestration fails closed on a nonzero probe exit.
        self.root.joinpath(m83.PYTHON).parent.mkdir(parents=True,exist_ok=True)
        self.root.joinpath(m83.PYTHON).write_text('fake interpreter')
        self.root.joinpath(m83.SCRIPT).parent.mkdir(parents=True,exist_ok=True)
        self.root.joinpath(m83.SCRIPT).write_text('# fake probe')
        with self.assertRaisesRegex(ValueError,'Real checkpoint restore probe failed'):
            m83.run_probe(self.root,runner=lambda *a,**kw:subprocess.CompletedProcess(a,12),check_live=False)

    def test_probe_wrong_checkpoint_metadata_rejected(self):
        proof=self.fake_probe();proof['checkpoint_step']=4100
        with self.assertRaisesRegex(ValueError,'wrong step'):m83._check_probe(
            proof,expected_step=4096,expected_sha=m83.step_receipt(self.art,4096)['sha256'],
            snapshot=self.art/'snapshots/snapshot_step_4096.pt')

    def test_incident_resume_from_verified_snapshot(self):
        self.snapshot(8192);self.state('PAUSED_INCIDENT','continuous')
        self.put_probe(8192)
        (self.run/'last.pt').write_bytes(b'bad incomplete active last')
        gate=m83._commit_gate(self.root,step=8192,phase='incident')
        self.assertEqual(gate['checkpoint_step'],8192)
        self.assertTrue(m83.verify_continuous_authorization(self.root)['ready_for_continuous'])
        self.assertEqual(m82.load_json(self.art/'runner_state.json')['state'],'RECOVERY_APPROVED')
        self.assertEqual((self.run/'last.pt').read_bytes(),(self.art/'snapshots/snapshot_step_8192.pt').read_bytes())
        self.assertEqual(len(list(self.run.glob('incident_last_before_restore_*'))),1)

    def test_incident_refuses_without_offsite_copy(self):
        self.snapshot(8192);self.state('PAUSED_INCIDENT','continuous')
        (self.art/'offsite_ack.json').unlink()
        self.put_probe(8192)
        with self.assertRaisesRegex(ValueError,'Offsite'):m83._commit_gate(self.root,step=8192,phase='incident')

    def test_incident_refuses_unapproved_prior_state(self):
        self.snapshot(8192);self.state('RUNNING','continuous')
        self.put_probe(8192)
        with self.assertRaisesRegex(ValueError,'Expected state'):m83._commit_gate(self.root,step=8192,phase='incident')

    def test_final_audit_requires_final_backup(self):
        self.snapshot(183106);self.state('COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP','continuous')
        with self.assertRaisesRegex(ValueError,'backup'):m83.final_audit(self.root)

    def test_final_audit_happy_path(self):
        self.snapshot(183106);self.state('COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP','continuous')
        self.ack(183106)
        def event(ev,step):return json.dumps({'schema_version':1,'event':ev,'step':step,'skipped_updates':0})+'\n'
        (self.run/'metrics.jsonl').write_text(event('val',183106)+event('checkpoint',183106))
        out=m83.final_audit(self.root)
        self.assertEqual(out['status'],'PASS')
        self.assertTrue((self.m83art/'final_audit.json').exists())

    def test_final_audit_rejects_missing_validation(self):
        self.snapshot(183106);self.state('COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP','continuous')
        self.ack(183106)
        (self.run/'metrics.jsonl').write_text(json.dumps({'schema_version':1,'event':'checkpoint',
                                                           'step':183106,'skipped_updates':0})+'\n')
        with self.assertRaisesRegex(ValueError,'validation'):m83.final_audit(self.root)

    def test_end_to_end_mock_drill_backup_proof_continuation_and_final(self):
        # A real OS child exercises the M8.2 supervisor and its append-only
        # committed events. All bytes are disposable fixtures; the only model
        # restore represented here is mock evidence, NEVER a GPU acceptance.
        import time
        (self.art/'snapshots/snapshot_step_4096.pt').unlink()
        (self.art/'snapshots/snapshot_step_4096.json').unlink()
        (self.art/'offsite_ack.json').unlink()
        script=("import pathlib,json,time,sys\n"
                "r=pathlib.Path(sys.argv[1]);r.mkdir(parents=True,exist_ok=True)\n"
                "(r/'last.pt').write_bytes(b'fake checkpoint first')\n"
                "e={'schema_version':1,'event':'checkpoint','step':4096,'skipped_updates':0}\n"
                "(r/'metrics.jsonl').write_text(json.dumps(e)+'\\n')\n"
                "time.sleep(6)\n")
        cmd=[sys.executable,'-c',script,str(self.run)]
        self.assertEqual(m82.supervisor(self.root,'drill',python=sys.executable,command=cmd,
                         interval=.02,stall_seconds=2,validator=lambda *_:None,min_snapshot_bytes=1),0)
        self.assertEqual(m82.load_json(self.art/'runner_state.json')['state'],'AWAITING_M83_RECOVERY_GATE')
        self.ack(4096)
        self.put_probe()
        self.assertEqual(m83._commit_gate(self.root,step=4096,phase='drill')['status'],'PASS')
        self.assertEqual(m82.verify_gate(self.root,'continuous',new_run=False)[0],self.run)
        continue_script=("import pathlib,json,sys\n"
                         "r=pathlib.Path(sys.argv[1])\n"
                         "(r/'last.pt').write_bytes(b'fake checkpoint final')\n"
                         "with (r/'metrics.jsonl').open('a') as f:\n"
                         " for event in ('val','checkpoint'):\n"
                         "  e={'schema_version':1,'event':event,'step':183106,'skipped_updates':0,'val_loss':4.0}\n"
                         "  f.write(json.dumps(e)+'\\n')\n")
        command=[sys.executable,'-c',continue_script,str(self.run)]
        self.assertEqual(m82.supervisor(self.root,'continuous',python=sys.executable,command=command,
                         interval=.02,stall_seconds=2,validator=lambda *_:None,min_snapshot_bytes=1),0)
        self.assertEqual(m82.load_json(self.art/'runner_state.json')['state'],
                         'COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP')
        self.ack(183106)
        self.assertEqual(m83.final_audit(self.root)['status'],'PASS')
        (self.m83art/'recovery_gate.json').is_file()
        self.assertEqual(m83.package_evidence(self.root)['status'],'PACKAGED')
        self.assertEqual(m83.package_evidence(self.root)['status'],'ALREADY_PACKAGED')

    def test_final_evidence_zip_excludes_weights(self):
        self.snapshot(183106);self.state('COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP','continuous')
        self.ack(183106)
        self.put_probe();self.approve_first() if False else None
        (self.m83art/'recovery_gate.json').write_text('{}')
        (self.art/'heartbeat.json').write_text('{}');(self.art/'training.log').write_text('test')
        (self.run/'metrics.jsonl').write_text(''.join(json.dumps({'schema_version':1,'event':e,'step':183106,
                                                    'skipped_updates':0})+'\n' for e in ('val','checkpoint')))
        self.assertIn('PACKAGED',m83.package_evidence(self.root)['status'])
        import zipfile
        with zipfile.ZipFile(self.m83art/'m8_final_evidence.zip') as z:
            self.assertFalse(any(x.endswith('.pt') for x in z.namelist()))

if __name__=='__main__':unittest.main()


class FinalReceiverTests(unittest.TestCase):
    """Local final verification uses real hashes on small fixtures, no GPU or networking."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.out=self.root/'backups'/m82.RUN_ID
        self.out.mkdir(parents=True)
        self.pt=self.out/'snapshot_step_183106.pt'
        self.pt.write_bytes(b'fake checkpoint final bytes')
        self.h=m82.sha(self.pt)
        self.sidecar=self.out/(self.pt.name+'.verified.json')
        self.sidecar.write_text(json.dumps({'verified':True,'step':183106,'bytes':self.pt.stat().st_size,
                                             'sha256':self.h}))
        self.evidence=self.root/'evidence.zip'
        rows={
            'final_audit.json':{'status':'PASS','final_step':183106,'checkpoint_sha256':self.h},
            'final_snapshot_receipt.json':{'step':183106,'bytes':self.pt.stat().st_size,'sha256':self.h},
            'offsite_ack.json':{'step':183106,'verified_external':True,'sha256':self.h},
            'runner_state.json':{'state':'COMPLETED_PENDING_EXTERNAL_FINAL_BACKUP'},
            'run.json':{},'probe_receipt.json':{},'recovery_gate.json':{},
        }
        import zipfile
        with zipfile.ZipFile(self.evidence,'w') as z:
            for n,v in rows.items():z.writestr(n,json.dumps(v))
            z.writestr('metrics.jsonl',json.dumps({'schema_version':1,'event':'checkpoint',
                                                  'step':183106,'skipped_updates':0})+'\n')

    def call(self):
        from scripts.m8 import verify_final_local_m83 as check
        return check.verify(self.root/'backups',self.evidence)

    def test_final_offsite_sha_verified(self):
        self.assertEqual(self.call()['status'],'PASS')

    def test_final_offsite_corruption_blocks_shutdown(self):
        self.pt.write_bytes(b'bad')
        with self.assertRaises(ValueError):self.call()

    def test_final_missing_offsite_file_blocks_shutdown(self):
        self.pt.unlink()
        with self.assertRaises(ValueError):self.call()

    def test_final_mismatched_remote_receipt_blocks_shutdown(self):
        import zipfile
        d={}
        with zipfile.ZipFile(self.evidence) as z:
            d={n:z.read(n) for n in z.namelist()}
        obj=json.loads(d['offsite_ack.json']);obj['sha256']='a'*64
        d['offsite_ack.json']=json.dumps(obj).encode()
        with zipfile.ZipFile(self.evidence,'w') as z:
            for n,b in d.items():z.writestr(n,b)
        with self.assertRaises(ValueError):self.call()


class StaticDeliveryTests(unittest.TestCase):
    def test_no_unapproved_training_entrypoint(self):
        text=(ROOT/'scripts/m8/run_gate_m83.sh').read_text()
        self.assertNotIn('train.py',text)
        self.assertNotIn('M82_START=YES_M8_CONTINUOUS bash',text)
        self.assertIn('probe',text)

    def test_full_deployment_has_all_phase_scripts(self):
        text=(ROOT/'scripts/m8/deploy_m83.sh').read_text()
        for filename in ('m81.py','m82.py','m83.py','probe_resume_m83.py',
                         'run_m82.sh','run_gate_m83.sh','backup_client_m82.py'):
            self.assertIn(filename,text)
        self.assertNotIn('last.pt',text)

    def test_windows_fsync_patch_packaged(self):
        text=(ROOT/'scripts/m8/backup_client_m82.py').read_text()
        self.assertIn("partial.open('r+b')",text)
        self.assertNotIn("partial.open('rb') as f:os.fsync",text)

    def test_incident_may_restore_4096_if_new_run_ends_early(self):
        text=(ROOT/'scripts/m8/recover_incident_m83.sh').read_text()
        self.assertIn('"$STEP" -ge 4096',text)
