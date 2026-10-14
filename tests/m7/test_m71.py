"""Offline checks: reference integrity and SFTP first/resume behavior."""
from __future__ import annotations
import hashlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/m7/vendor'))
sys.path.insert(0,str(ROOT/'scripts/m7'))
import scripts.m7.vendor.branch4_transport as transport
from scripts.m7 import m71

class FakeConnection:
    def __init__(self,part_exists=False,part_bytes=0):
        self.part_exists=part_exists;self.part_bytes=part_bytes;self.commands=[];self.rpcs=[]
    def rpc(self,args):
        self.rpcs.append(args)
        if args[0]=='file-status':
            if '.b4part.' in args[2]:
                return {'valid':False,'exists':self.part_exists,'bytes':self.part_bytes}
            return {'valid':False,'exists':False,'bytes':0}
        if args[0]=='promote':return {'published':True}
        raise AssertionError(args)
    def batch(self,s):self.commands.append(s)

class BasicTests(unittest.TestCase):
    def test_reference_manifest(self):
        reference,manifest=m71.validate_reference()
        self.assertEqual(len(manifest['files']),254)
        self.assertEqual(reference['splits']['train']['tokens'],6_000_001_675)
        self.assertEqual(reference['splits']['val']['tokens'],316_623_851)
    def test_isolation_from_m6(self):
        self.assertTrue((ROOT/'scripts/m7/launch.sh').is_file())
        self.assertTrue((ROOT/'scripts/m7/vendor/branch4_remote_ops.py').is_file())
        with patch.object(transport.shutil,'which',return_value='/usr/bin/ssh'):
            c=transport.Connection({'host':'vast','port':22,'user':'root'})
        self.assertEqual(c.ops,'/workspace/kimi/ops/m7/branch4_remote_ops.py')
    def test_first_upload_uses_put(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'train_00000.npy';p.write_bytes(b'hello')
            c=FakeConnection(False,0)
            transport.upload_one(c,p,'/workspace/kimi/data/full/train_00000.npy',5,hashlib.sha256(b'hello').hexdigest())
            self.assertEqual(len(c.commands),1)
            self.assertTrue(c.commands[0].startswith('put '))
    def test_resume_upload_uses_reput(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'train_00000.npy';p.write_bytes(b'hello')
            c=FakeConnection(True,2)
            transport.upload_one(c,p,'/workspace/kimi/data/full/train_00000.npy',5,hashlib.sha256(b'hello').hexdigest())
            self.assertTrue(c.commands[0].startswith('reput '))

if __name__=='__main__':unittest.main()
