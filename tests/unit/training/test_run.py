import copy
import json
import random
import subprocess
from dataclasses import replace

import numpy as np
import pytest
import torch

from kimi_k3.data import ShardedTokenDataset, TokenDataLoader
from kimi_k3.model import KimiK3
from kimi_k3.training.run import _git_metadata, start_run
from kimi_k3.training.trainer import TrainConfig, TrainState, build_optimizer


@pytest.fixture
def run_arguments(tmp_path, kda_config_factory):
	directory = tmp_path / "data"
	directory.mkdir()
	metadata = {
		"format_version": 2, "storage": {"format": "npy", "dtype": "uint16"},
		"source": {"dataset": "test/corpus", "revision": "pinned-data"},
		"tokenizer": {"repository": "test/tokenizer", "revision": "pinned-tokenizer",
					  "vocab_size": 32, "eos_id": 0}, "splits": {},
	}
	for split in ("train", "val"):
		filename = f"{split}_000.npy"
		np.save(directory / filename, np.arange(65, dtype=np.uint16) % 32)
		metadata["splits"][split] = {"tokens": 65, "shards": [{"file": filename, "tokens": 65}]}
	(directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
	config = TrainConfig(batch_size=1, seq_len=8, max_steps=2)
	model = KimiK3(kda_config_factory(num_layers=2, layer_types=("kda", "mla"), num_dense_layers=1))
	loaders = [TokenDataLoader(ShardedTokenDataset(directory, split), 1, 8)
			   for split in ("train", "val")]
	return {"model": model, "optimizer": build_optimizer(model, config), "config": config,
			"state": TrainState(), "train_loader": loaders[0], "val_loader": loaders[1],
			"repo_root": tmp_path, "runs_root": tmp_path / "runs"}


def test_run_records_provenance_without_changing_training_state(run_arguments):
	model = run_arguments["model"]
	weights = copy.deepcopy(model.state_dict())
	optimizer_state = copy.deepcopy(run_arguments["optimizer"].state_dict())
	cursors = [run_arguments[name].state_dict() for name in ("train_loader", "val_loader")]
	random.seed(17)
	np.random.seed(18)
	torch.manual_seed(19)
	rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
	run = start_run(**run_arguments)
	manifest = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
	session = manifest["sessions"][0]
	assert manifest["run_id"] == run.run_id
	assert session["session_id"] == run.session_id
	assert run.checkpoint_path == run.directory / "last.pt"
	assert session["model_config"]["layer_types"] == ["kda", "mla"]
	assert session["data"]["train"]["metadata_sha256"] == run_arguments["train_loader"].dataset.metadata_sha256
	assert session["data"]["train"]["source"]["revision"] == "pinned-data"
	assert session["data"]["val"]["tokenizer"]["revision"] == "pinned-tokenizer"
	assert session["train_config"]["seed"] == run_arguments["config"].seed
	groups = session["optimizer"]["param_groups"]
	assert {group["algorithm"] for group in groups} == {"muon", "adamw"}
	assert all(group["ns_dtype"] == "torch.bfloat16" for group in groups if group["algorithm"] == "muon")
	assert sum(group["element_count"] for group in groups) == session["parameter_count"]
	assert sum(group["parameter_count"] for group in groups) == len(list(model.parameters()))
	ukv = next(group for group in groups if "layers.1.attn.W_UKV.weight" in group["param_names"])
	assert ukv["muon_shape"] == [2, 8, 8]
	assert ukv["muon_split_sizes"] == [4, 4]
	layout = session["optimizer"]["layout"]
	assert next(group for group in layout if "muon_split_sizes" in group)["muon_split_sizes"] == [4, 4]
	assert session["environment"]["torch"] == str(torch.__version__)
	assert session["environment"]["hardware"]["device"] == "cpu"
	assert session["environment"]["hardware"]["gpu"] is None
	assert session["rng_source"] == "seed"
	assert session["qk_clip"]["policy"] == "mla_shared_key_v1"
	assert session["qk_clip"]["threshold"] == 100.0
	assert session["qk_clip"]["layers"][0]["name"] == "layers.1.attn"
	assert session["resume_checkpoint"] is None
	for name, value in weights.items():
		torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
	assert model.training
	assert run_arguments["optimizer"].state_dict() == optimizer_state
	assert [run_arguments[name].state_dict() for name in ("train_loader", "val_loader")] == cursors
	assert random.getstate() == rng[0]
	np.testing.assert_equal(np.random.get_state(), rng[1])
	torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0, atol=0)
	assert start_run(**run_arguments).directory != run.directory


def test_resume_preserves_run_identity_and_appends_effective_settings(run_arguments, tmp_path):
	first = start_run(**run_arguments)
	original = json.loads((first.directory / "run.json").read_text(encoding="utf-8"))
	config = replace(run_arguments["config"], max_steps=4, learning_rate=0.002, seed=999)
	run_arguments["config"] = config
	run_arguments["state"] = TrainState(step=2, tokens_seen=16)
	run_arguments["train_loader"].reset(16)
	run_arguments["optimizer"].param_groups[0]["lr"] = 0.001
	resumed = start_run(**run_arguments, resume_path=first.checkpoint_path,
						checkpoint_path=tmp_path / "external.pt")
	manifest = json.loads((first.directory / "run.json").read_text(encoding="utf-8"))
	assert resumed.directory == first.directory
	assert resumed.run_id == first.run_id
	assert resumed.session_id != first.session_id
	assert manifest["created_at"] == original["created_at"]
	assert manifest["sessions"][0] == original["sessions"][0]
	session = manifest["sessions"][1]
	assert session["train_config"]["max_steps"] == 4
	assert session["train_config"]["learning_rate"] == 0.002
	assert session["optimizer"]["param_groups"][0]["lr"] == 0.001
	assert session["start_state"]["step"] == 2
	assert session["start_state"]["loaders"]["train"]["position"] == 16
	assert session["rng_source"] == "checkpoint"
	assert session["resume_checkpoint"] == str(first.checkpoint_path)
	# An external checkpoint can reuse the same run when its folder is explicit.
	again = start_run(**run_arguments, resume_path=resumed.checkpoint_path, run_dir=first.directory)
	assert again.run_id == first.run_id
	assert again.checkpoint_path == resumed.checkpoint_path


def test_existing_run_or_unrelated_checkpoint_cannot_overwrite_manifest(run_arguments, tmp_path):
	run = start_run(**run_arguments)
	path = run.directory / "run.json"
	before = path.read_bytes()
	with pytest.raises(FileExistsError, match="already exists"):
		start_run(**run_arguments, run_dir=run.directory)
	with pytest.raises(ValueError, match="not associated"):
		start_run(**run_arguments, run_dir=run.directory, resume_path=tmp_path / "unrelated.pt")
	with pytest.raises(ValueError, match="overwrite run.json"):
		start_run(**run_arguments, resume_path=run.checkpoint_path, checkpoint_path=path)
	with pytest.raises(ValueError, match="overwrite metrics.jsonl"):
		start_run(**run_arguments, resume_path=run.checkpoint_path,
				  checkpoint_path=run.directory / "metrics.jsonl")
	assert path.read_bytes() == before


def test_checkpoint_without_manifest_starts_a_linked_run(run_arguments, tmp_path):
	checkpoint = tmp_path / "legacy.pt"
	run_arguments["state"] = TrainState(step=1, tokens_seen=8)
	run = start_run(**run_arguments, resume_path=checkpoint)
	session = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))["sessions"][0]
	assert run.directory != checkpoint.parent
	assert run.checkpoint_path == run.directory / "last.pt"
	assert session["start_state"]["step"] == 1
	assert session["rng_source"] == "checkpoint"
	assert session["resume_checkpoint"] == str(checkpoint)


def test_run_folder_can_move_without_losing_identity(run_arguments, tmp_path):
	first = start_run(**run_arguments)
	moved = tmp_path / "moved_run"
	first.directory.rename(moved)
	resumed = start_run(**run_arguments, resume_path=moved / "last.pt")
	assert resumed.directory == moved
	assert resumed.run_id == first.run_id
	manifest = json.loads((moved / "run.json").read_text(encoding="utf-8"))
	assert len(manifest["sessions"]) == 2
	assert manifest["sessions"][-1]["checkpoint_relative_path"] == "last.pt"


@pytest.mark.parametrize("changed", ["model", "data"])
def test_resume_rejects_inconsistent_manifest_without_rewriting_it(run_arguments, changed):
	run = start_run(**run_arguments)
	path = run.directory / "run.json"
	before = path.read_bytes()
	if changed == "model":
		model = run_arguments["model"]
		model.config = replace(model.config, initializer_range=0.03)
	else:
		run_arguments["train_loader"].dataset.metadata_sha256 = "different-manifest"
	with pytest.raises(ValueError, match="Model or data differs"):
		start_run(**run_arguments, resume_path=run.checkpoint_path)
	assert path.read_bytes() == before


def test_git_provenance_reports_clean_dirty_and_unavailable(tmp_path):
	assert _git_metadata(tmp_path) == {"commit": None, "dirty": None}
	repository = tmp_path / "repository"
	repository.mkdir()
	def git(*args):
		return subprocess.run(["git", "-C", str(repository), *args], check=True,
							  capture_output=True, text=True).stdout.strip()
	git("init", "-q")
	source = repository / "source.py"
	source.write_text("original\n", encoding="utf-8")
	git("add", "source.py")
	git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "Initial")
	commit = git("rev-parse", "HEAD")
	assert _git_metadata(repository) == {"commit": commit, "dirty": False}
	source.write_text("modified\n", encoding="utf-8")
	assert _git_metadata(repository) == {"commit": commit, "dirty": True}
