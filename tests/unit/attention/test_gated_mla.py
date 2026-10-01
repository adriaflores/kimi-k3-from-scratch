import math

import pytest
import torch
import torch.nn.functional as F

from kimi_k3.attention.gated_mla import GatedMLA

# Distinct Q/K content, shared and value dims so a mix-up between them changes shapes or scale.
ASYMMETRIC_DIMS = dict(mla_qk_nope_head_dim=4, mla_qk_shared_head_dim=2, mla_v_head_dim=6, mla_num_heads=3)
OUTPUT_GATES = [True, False]


def _make_mla(kda_config_factory, **overrides):
	config = kda_config_factory(**overrides)
	return config, GatedMLA(config)


def _reference_forward(mla, x):
	"""Per-head reference built from the module's weights with explicit tensor operations."""
	B, T, _ = x.shape
	H, D, D_s, D_v = mla.num_heads, mla.d_head, mla.d_shared, mla.dv_head

	c_q = F.rms_norm(x @ mla.W_DQ.weight.T, (mla.q_latent_dim,), mla.q_rmsnorm.weight, mla.q_rmsnorm.eps)
	kv = x @ mla.W_DKV.weight.T
	c_kv = F.rms_norm(kv[..., :mla.kv_latent_dim], (mla.kv_latent_dim,), mla.kv_rmsnorm.weight, mla.kv_rmsnorm.eps)
	k_shared = kv[..., mla.kv_latent_dim:] # (B, T, D_s)

	W_UQ = mla.W_UQ.weight.view(H, D + D_s, mla.q_latent_dim)
	W_UKV = mla.W_UKV.weight.view(H, D + D_v, mla.kv_latent_dim)
	causal = torch.tril(torch.ones(T, T, dtype=torch.bool))

	scores, heads = [], []
	for h in range(H):
		q_h = c_q @ W_UQ[h].T # (B, T, D + D_s)
		k_c_h = c_kv @ W_UKV[h, :D].T # (B, T, D)
		v_h = c_kv @ W_UKV[h, D:].T # (B, T, D_v)
		k_h = torch.cat([k_c_h, k_shared], dim=-1) # (B, T, D + D_s)

		s_h = (q_h @ k_h.transpose(-1, -2)) / math.sqrt(D + D_s)
		s_h = s_h.masked_fill(~causal, float("-inf"))
		scores.append(s_h)
		heads.append(torch.softmax(s_h, dim=-1) @ v_h) # (B, T, D_v)

	values = torch.cat(heads, dim=-1) # (B, T, H * D_v)
	if mla.output_gate:
		values = torch.sigmoid(x @ mla.Wg.weight.T) * values
	out = values @ mla.Wo.weight.T

	return out, torch.stack(scores, dim=1) # (B, T, C), (B, H, T, T)


@pytest.mark.parametrize("B, T", [(1, 1), (2, 5), (3, 16), (4, 32)])
def test_output_shape(kda_config_factory, B, T):
	config, mla = _make_mla(kda_config_factory)
	x = torch.randn(B, T, config.hidden_size)

	out = mla(x)

	assert out.shape == (B, T, config.hidden_size)


@pytest.mark.parametrize("output_gate", OUTPUT_GATES)
def test_forward_is_finite(kda_config_factory, output_gate):
	config, mla = _make_mla(kda_config_factory, mla_output_gate=output_gate)
	x = torch.randn(3, 12, config.hidden_size)

	out = mla(x)

	assert torch.isfinite(out).all()


@pytest.mark.parametrize("output_gate", OUTPUT_GATES)
def test_backward_produces_finite_gradients_for_all_parameters(kda_config_factory, output_gate):
	config, mla = _make_mla(kda_config_factory, mla_output_gate=output_gate)
	x = torch.randn(2, 8, config.hidden_size, requires_grad=True)

	mla(x).square().mean().backward()

	assert torch.isfinite(x.grad).all()
	for name, p in mla.named_parameters():
		assert p.grad is not None, f"no gradient for {name}"
		assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_sequence_length_equal_to_max_seq_len_works(kda_config_factory):
	config, mla = _make_mla(kda_config_factory)
	x = torch.randn(2, config.max_seq_len, config.hidden_size)

	out = mla(x)

	assert out.shape == (2, config.max_seq_len, config.hidden_size)
	assert torch.isfinite(out).all()


def test_sequence_length_one(kda_config_factory):
	config, mla = _make_mla(kda_config_factory)
	x = torch.randn(2, 1, config.hidden_size)

	out = mla(x)

	assert out.shape == (2, 1, config.hidden_size)
	assert torch.isfinite(out).all()
	torch.testing.assert_close(out, _reference_forward(mla, x)[0])


@pytest.mark.parametrize("t", [0, 5, 14])
def test_causality_future_positions_do_not_affect_past_outputs(kda_config_factory, t):
	config, mla = _make_mla(kda_config_factory)
	mla.eval()

	B, T, C = 2, 16, config.hidden_size
	x_a = torch.randn(B, T, C)
	x_b = x_a.clone()
	x_b[:, t + 1:, :] = torch.randn(B, T - (t + 1), C)

	with torch.no_grad():
		out_a = mla(x_a)
		out_b = mla(x_b)

	torch.testing.assert_close(out_a[:, :t + 1], out_b[:, :t + 1])
	assert not torch.allclose(out_a[:, t + 1:], out_b[:, t + 1:])


def test_output_gate_enabled(kda_config_factory):
	config, mla = _make_mla(kda_config_factory, mla_output_gate=True)
	x = torch.randn(2, 8, config.hidden_size)

	out = mla(x)
	out.square().mean().backward()

	assert hasattr(mla, "Wg")
	assert mla.Wg.weight.shape == (mla.num_heads * mla.dv_head, config.hidden_size)
	assert mla.Wg.weight.grad is not None
	assert torch.isfinite(mla.Wg.weight.grad).all()
	assert mla.Wg.weight.grad.abs().sum() > 0


def test_output_gate_disabled(kda_config_factory):
	config, mla = _make_mla(kda_config_factory, mla_output_gate=False)
	x = torch.randn(2, 8, config.hidden_size)

	out = mla(x)
	out.square().mean().backward()

	assert torch.isfinite(out).all()
	assert not hasattr(mla, "Wg")
	assert not any(name.startswith("Wg") for name, _ in mla.named_parameters())


def test_shared_key_is_identical_across_heads_and_content_key_is_head_specific(kda_config_factory, monkeypatch):
	config, mla = _make_mla(kda_config_factory, **ASYMMETRIC_DIMS)
	x = torch.randn(2, 6, config.hidden_size)

	captured = []
	original_cat = torch.cat

	def spy_cat(tensors, *args, **kwargs):
		captured.append([t.detach().clone() for t in tensors])
		return original_cat(tensors, *args, **kwargs)

	monkeypatch.setattr(torch, "cat", spy_cat)
	with torch.no_grad():
		mla(x)
	monkeypatch.undo()

	assert len(captured) == 1, "expected exactly one torch.cat call building K"
	K_c, K_shared = captured[0]
	H = mla.num_heads
	assert K_c.shape == (2, H, 6, mla.d_head)
	assert K_shared.shape == (2, H, 6, mla.d_shared)

	# Shared part: every head sees the same tensor, taken straight from the W_DKV output.
	with torch.no_grad():
		expected_shared = mla.W_DKV(x)[..., mla.kv_latent_dim:]
	for h in range(H):
		torch.testing.assert_close(K_shared[:, h], expected_shared)

	# Content part: each head has its own projection.
	for h in range(1, H):
		assert not torch.allclose(K_c[:, 0], K_c[:, h])


def test_qk_head_dim_is_content_plus_shared(kda_config_factory):
	config, mla = _make_mla(kda_config_factory, **ASYMMETRIC_DIMS)
	d_qk = ASYMMETRIC_DIMS["mla_qk_nope_head_dim"] + ASYMMETRIC_DIMS["mla_qk_shared_head_dim"]

	assert mla.W_UQ.in_features == config.mla_q_lora_rank
	assert mla.W_UQ.out_features == mla.num_heads * d_qk
	assert mla.d_head + mla.d_shared == d_qk


def test_kv_projection_decomposition(kda_config_factory):
	config, mla = _make_mla(kda_config_factory, **ASYMMETRIC_DIMS)
	D, D_v = ASYMMETRIC_DIMS["mla_qk_nope_head_dim"], ASYMMETRIC_DIMS["mla_v_head_dim"]

	assert mla.W_DKV.in_features == config.hidden_size
	assert mla.W_DKV.out_features == config.mla_kv_lora_rank + ASYMMETRIC_DIMS["mla_qk_shared_head_dim"]
	assert mla.W_UKV.in_features == config.mla_kv_lora_rank
	assert mla.W_UKV.out_features == mla.num_heads * (D + D_v)
	assert mla.Wo.in_features == mla.num_heads * D_v


@pytest.mark.parametrize("output_gate", OUTPUT_GATES)
def test_forward_and_backward_match_reference(kda_config_factory, output_gate):
	config, mla = _make_mla(kda_config_factory, mla_output_gate=output_gate, **ASYMMETRIC_DIMS)
	x = torch.randn(2, 10, config.hidden_size)
	x_module = x.clone().requires_grad_(True)
	x_ref = x.clone().requires_grad_(True)

	out_module = mla(x_module)
	out_module.square().mean().backward()
	module_grads = {name: p.grad.clone() for name, p in mla.named_parameters()}
	mla.zero_grad()

	out_ref, _ = _reference_forward(mla, x_ref)
	out_ref.square().mean().backward()

	torch.testing.assert_close(out_module, out_ref)
	torch.testing.assert_close(x_module.grad, x_ref.grad)
	for name, p in mla.named_parameters():
		torch.testing.assert_close(module_grads[name], p.grad, msg=lambda m, name=name: f"{name}: {m}")


def test_attention_scores_are_scaled_by_full_qk_head_dim(kda_config_factory, monkeypatch):
	config, mla = _make_mla(kda_config_factory, **ASYMMETRIC_DIMS)
	x = torch.randn(2, 6, config.hidden_size)

	captured = []
	original_sdpa = F.scaled_dot_product_attention

	def spy_sdpa(query, key, value, *args, **kwargs):
		captured.append((query.detach().clone(), key.detach().clone(), kwargs.get("scale")))
		return original_sdpa(query, key, value, *args, **kwargs)

	monkeypatch.setattr(F, "scaled_dot_product_attention", spy_sdpa)
	with torch.no_grad():
		mla(x)
	monkeypatch.undo()

	assert len(captured) == 1
	Q, K, scale = captured[0]
	# SDPA falls back to 1/sqrt(Q.size(-1)) when no scale is given.
	if scale is None:
		scale = 1.0 / math.sqrt(Q.size(-1))

	T = x.shape[1]
	causal_mask = torch.ones(T, T, dtype=torch.bool).triu(1)
	module_scores = (Q @ K.mT * scale).masked_fill(causal_mask, float("-inf"))
	with torch.no_grad():
		_, ref_scores = _reference_forward(mla, x) # scaled by 1/sqrt(d_head + d_shared)

	torch.testing.assert_close(module_scores, ref_scores)

	# Scaling by d_head alone would give scores larger by sqrt((d_head + d_shared) / d_head).
	wrong_scores = ref_scores * math.sqrt((mla.d_head + mla.d_shared) / mla.d_head)
	finite = torch.isfinite(ref_scores)
	assert not torch.allclose(module_scores[finite], wrong_scores[finite])


def test_bf16_dtype_round_trip(kda_config_factory):
	config, mla = _make_mla(kda_config_factory)
	mla = mla.to(torch.bfloat16)
	x = torch.randn(2, 8, config.hidden_size, dtype=torch.bfloat16)

	out = mla(x)

	assert out.dtype == torch.bfloat16
	assert torch.isfinite(out).all()


def test_forward_is_deterministic(kda_config_factory):
	config, mla = _make_mla(kda_config_factory)
	x = torch.randn(2, 12, config.hidden_size)

	with torch.no_grad():
		out_1 = mla(x)
		out_2 = mla(x)

	assert torch.equal(out_1, out_2)
