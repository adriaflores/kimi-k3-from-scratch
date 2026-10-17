"""Offline regression tests for the checkpoint-agnostic public CLIs/primitives."""
from dataclasses import asdict

import pytest
import torch

from kimi_k3.evaluation import evaluate_windows
from kimi_k3.inference import (
    generate_tokens, load_checkpoint_model, next_token_logits, select_token,
)
from kimi_k3.model import KimiK3, KimiK3Config
from kimi_k3.utils import load_config


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    torch.manual_seed(11)
    config = load_config("configs/model/kimi_k3_tiny.yaml", KimiK3Config)
    model = KimiK3(config)
    path = tmp_path_factory.mktemp("public-inference") / "tiny.pt"
    torch.save({"format_version": 7, "model_config": asdict(config),
                "model": model.state_dict(), "step": 0, "tokens_seen": 0}, path)
    return path


def test_strict_tiny_checkpoint_and_weight_tying(tiny_checkpoint):
    model, meta = load_checkpoint_model(tiny_checkpoint, device=torch.device("cpu"))
    assert meta["backend_inference"] == "chunkwise"
    assert model.embed_tokens.weight is model.lm_head.weight
    assert not model.training
    logits = next_token_logits(model, [7, 13, 29], device=torch.device("cpu"))
    assert logits.shape == (model.config.vocab_size,)
    assert torch.isfinite(logits).all()


def test_greedy_generation_matches_argmax(tiny_checkpoint):
    model, _ = load_checkpoint_model(tiny_checkpoint, device=torch.device("cpu"))
    start = [7, 13, 29]
    expected = next_token_logits(model, start, device=torch.device("cpu")).argmax().item()
    assert generate_tokens(model, start, device=torch.device("cpu"),
                           max_new_tokens=1) == [expected]
    with pytest.raises(ValueError, match="exceed max_seq_len"):
        generate_tokens(model, [1] * 128, device=torch.device("cpu"), max_new_tokens=1)


def test_sampling_seed_and_limits():
    logits = torch.tensor([0.5, 3.0, 0.25])
    assert select_token(logits, mode="greedy") == 1
    assert select_token(logits, mode="sample", top_k=1, top_p=1) == 1
    with pytest.raises(ValueError):
        select_token(logits, mode="sample", top_p=0)


class _FakeConfig:
    max_seq_len = 4
    vocab_size = 5
    kda_implementation = "fused"
    layer_types = ("mla",)


class _ToyModel(torch.nn.Module):
    config = _FakeConfig()

    def forward(self, ids):
        out = torch.zeros((1, ids.shape[1], 5), device=ids.device)
        return out, None


class _TinyDataset:
    vocab_size = 5
    split = "val"
    metadata_sha256 = "example"
    total_tokens = 10

    def get_sequence(self, offset, seq_len):
        stream = torch.tensor([1, 2, 3, 4, 0, 1, 2, 3, 4, 0])
        return stream[offset:offset+seq_len], stream[offset+1:offset+seq_len+1]


def test_evaluate_windows_exact_scored_token_count():
    results = evaluate_windows(_ToyModel(), _TinyDataset(), device=torch.device("cpu"),
                               seq_len=4, num_windows=2)
    assert results["scored_tokens"] == 8
    assert results["mean_nll_nats_per_token"] == pytest.approx(torch.tensor(5.0).log().item())
    assert results["top1_accuracy"] == pytest.approx(1/8)
    with pytest.raises(ValueError, match="overlapping"):
        evaluate_windows(_ToyModel(), _TinyDataset(), device=torch.device("cpu"),
                         seq_len=4, num_windows=2, stride=3)
