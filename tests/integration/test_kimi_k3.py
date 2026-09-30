import math

import pytest
import torch

from kimi_k3.model import KimiK3

# Layer 0 has no previous AttnRes block to mix before its attention sublayer, so these are never used.
UNUSED_PARAMS = {"layers.0.attn_res_norm.weight", "layers.0.attn_res_proj.weight"}


@pytest.fixture
def model_config(kda_config_factory):
	# KDA and MLA layers, one dense FFN followed by MoE layers, and two AttnRes blocks.
	return kda_config_factory(
		num_layers=4,
		layer_types=("kda", "kda", "kda", "mla"),
		num_dense_layers=1,
		attn_res_block_size=2,
		max_seq_len=16,
		kda_chunk_size=4,
		num_experts=4,
		num_experts_per_token=2,
	)


def test_forward_shapes_and_initial_loss(model_config):
	model = KimiK3(model_config)
	idx = torch.randint(0, model_config.vocab_size, (2, 16))

	logits, loss = model(idx, targets=idx)

	assert logits.shape == (2, 16, model_config.vocab_size)
	assert torch.isfinite(logits).all()
	# With small initial weights the prediction is close to uniform, so the loss is near ln(V).
	assert abs(loss.item() - math.log(model_config.vocab_size)) < 0.5


def test_forward_without_targets_returns_no_loss(model_config):
	model = KimiK3(model_config)
	idx = torch.randint(0, model_config.vocab_size, (2, 8))

	logits, loss = model(idx)

	assert logits.shape == (2, 8, model_config.vocab_size)
	assert loss is None


def test_sequence_longer_than_max_seq_len_raises(model_config):
	model = KimiK3(model_config)
	idx = torch.randint(0, model_config.vocab_size, (1, model_config.max_seq_len + 4))

	with pytest.raises(ValueError, match="exceeds max_seq_len"):
		model(idx)


def test_layer_schedule(model_config):
	model = KimiK3(model_config)

	assert [type(layer.attn).__name__ for layer in model.layers] == ["KDAChunkwise", "KDAChunkwise", "KDAChunkwise", "GatedMLA"]
	assert [type(layer.ffn).__name__ for layer in model.layers] == ["DenseFFN", "StableLatentMoE", "StableLatentMoE", "StableLatentMoE"]
	assert model.lm_head.weight is model.embed_tokens.weight


def test_backward_reaches_every_used_parameter(model_config):
	model = KimiK3(model_config)
	idx = torch.randint(0, model_config.vocab_size, (2, 16))

	_, loss = model(idx, targets=idx)
	loss.backward()

	for name, p in model.named_parameters():
		if name in UNUSED_PARAMS:
			assert p.grad is None, f"{name} was expected to be unused"
			continue
		assert p.grad is not None, f"no gradient for {name}"
		assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_training_step_reduces_loss(model_config):
	model = KimiK3(model_config)
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
	idx = torch.randint(0, model_config.vocab_size, (2, 16))

	losses = []
	for _ in range(5):
		_, loss = model(idx, targets=idx)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(loss.item())

	assert losses[-1] < losses[0]


@pytest.mark.parametrize("t", [0, 5, 11])
def test_causality_future_tokens_do_not_affect_past_logits(model_config, t):
	model = KimiK3(model_config)
	model.eval()

	idx_a = torch.randint(0, model_config.vocab_size, (2, 16))
	idx_b = idx_a.clone()
	idx_b[:, t + 1:] = torch.randint(0, model_config.vocab_size, (2, 16 - (t + 1)))

	# Every forward updates the MoE quantile-balancer biases, so run both inputs from the same buffer state.
	buffers = {name: b.clone() for name, b in model.named_buffers()}
	with torch.no_grad():
		logits_a, _ = model(idx_a)
		for name, b in model.named_buffers():
			b.copy_(buffers[name])
		logits_b, _ = model(idx_b)

	torch.testing.assert_close(logits_a[:, :t + 1], logits_b[:, :t + 1])
