import math

import pytest
import torch
import torch.nn.functional as F

from kimi_k3.moe.stable_latent_moe import QuantileBalancer, StableLatentMoE

NUM_EXPERTS = 4
TINY_MOE = dict(
	num_experts=NUM_EXPERTS,
	num_experts_per_token=2,
	num_shared_experts=1,
	latent_moe_dim=8,
	expert_intermediate_size=12,
	situ_beta=0.5,
	situ_linear_beta=2.0,
)

# Router logits for 4 tokens over 4 experts. With top-2 every token picks a different pair, every expert gets
# exactly two tokens, and every token is served by two experts:
# t0 -> {0, 1}, t1 -> {1, 2}, t2 -> {2, 3}, t3 -> {0, 3}.
DISPATCH_LOGITS = torch.tensor([
	[3.0, 2.0, -1.0, -2.0],
	[-2.0, 3.0, 2.0, -1.0],
	[-1.0, -2.0, 3.0, 2.0],
	[3.0, -2.0, -1.0, 2.0],
])
DISPATCH_SELECTION = [{0, 1}, {1, 2}, {2, 3}, {0, 3}]


def _make_moe(kda_config_factory, **overrides):
	config = kda_config_factory(**{**TINY_MOE, **overrides})
	return config, StableLatentMoE(config)


def _route_on_leading_features(moe, x, logits):
	"""Make router logits equal to the first E features of x, and write the requested logits there."""
	E = moe.num_routed_experts
	with torch.no_grad():
		moe.router.weight.zero_()
		moe.router.weight[:, :E] = torch.eye(E)
	x = x.clone()
	x[..., :E] = logits
	return x


def _manual_expert(expert, z, beta, linear_beta):
	# gate_up_proj stacks the gate rows first and the up rows second, matching SituGLU's chunk order.
	gate_weight, up_weight = expert.gate_up_proj.weight.chunk(2, dim=0)
	g = z @ gate_weight.T
	u = z @ up_weight.T
	activated = (beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (linear_beta * torch.tanh(u / linear_beta))
	return activated @ expert.down_proj.weight.T


def _manual_routed_latent(moe, config, x, biases):
	"""Token-by-token routed latent (before the post-router norm) using the given QB biases."""
	B, T, C = x.shape
	x_flat = x.reshape(B * T, C)
	scores = torch.sigmoid(x_flat.float() @ moe.router.weight.float().T) # (BT, E)
	z = x_flat @ moe.down_latent_proj.weight.T

	routed = torch.zeros(B * T, moe.latent_moe_dim)
	selections = []
	for t in range(B * T):
		selected = torch.topk(scores[t] + biases, k=moe.top_k).indices.tolist()
		selections.append(set(selected))
		w = scores[t, selected]
		if moe.moe_renormalize:
			w = w / w.sum()
		w = w * moe.routed_scaling_factor
		for weight, e in zip(w, selected):
			routed[t] += weight * _manual_expert(moe.experts[e], z[t], config.situ_beta, config.situ_linear_beta)
	return routed, selections


def _manual_forward(moe, config, x, biases):
	B, T, C = x.shape
	shared = _manual_expert(moe.shared_experts, x, config.situ_beta, config.situ_linear_beta)
	routed, _ = _manual_routed_latent(moe, config, x, biases)
	routed = routed.view(B, T, -1)
	if moe.latent_moe_use_norm:
		norm = moe.post_router_rmsnorms
		routed = F.rms_norm(routed, (moe.latent_moe_dim,), norm.weight, norm.eps)
	return shared + routed @ moe.up_latent_proj.weight.T


def _capture_up_latent_input(moe):
	"""Record the routed latent right before up_latent_proj (equal to the index_add result when the norm is off)."""
	captured = []
	moe.up_latent_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
	return captured


# QuantileBalancer

def test_quantile_balancer_update_uses_higher_quantile_and_is_mean_centered():
	E, k = 4, 2
	qb = QuantileBalancer(num_experts=E, k=k)
	# 6 tokens so the p = k/E = 0.5 quantile sits between two samples (position 2.5) and "higher" matters.
	router_scores = torch.tensor([
		[0.90, 0.80, 0.10, 0.20],
		[0.30, 0.70, 0.60, 0.05],
		[0.55, 0.15, 0.85, 0.40],
		[0.20, 0.95, 0.35, 0.65],
		[0.75, 0.25, 0.45, 0.50],
		[0.10, 0.60, 0.30, 0.90],
	])
	top_scores, _ = torch.topk(router_scores, k=k + 1, dim=-1) # biases are zero, so corrected == router scores

	qb.update(top_scores=top_scores, router_scores=router_scores)

	thresh = top_scores[:, -1:] # (BT, 1), the (k+1)-th largest score per token
	relative = thresh - router_scores # (BT, E)
	n = relative.shape[0]
	index = math.ceil((k / E) * (n - 1)) # "higher" interpolation picks the upper neighbour
	expected = relative.sort(dim=0).values[index]
	expected = expected - expected.mean()

	torch.testing.assert_close(qb.biases, expected)
	torch.testing.assert_close(qb.biases.mean(), torch.tensor(0.0), atol=1e-7, rtol=0)

	linear = torch.quantile(relative, k / E, dim=0, interpolation="linear")
	assert not torch.allclose(qb.biases, linear - linear.mean()), "test data does not distinguish higher from linear"


def test_quantile_balancer_biases_are_a_non_trainable_buffer():
	qb = QuantileBalancer(num_experts=4, k=2)

	assert "biases" in dict(qb.named_buffers())
	assert "biases" not in dict(qb.named_parameters())
	assert not qb.biases.requires_grad


def test_quantile_balancer_updates_only_in_training_mode(kda_config_factory):
	config, moe = _make_moe(kda_config_factory)
	x = torch.randn(2, 8, config.hidden_size)

	moe.eval()
	with torch.no_grad():
		moe(x)
	torch.testing.assert_close(moe.quantile_balancer.biases, torch.zeros(NUM_EXPERTS), rtol=0, atol=0)

	moe.train()
	with torch.no_grad():
		moe(x)
	assert not torch.equal(moe.quantile_balancer.biases, torch.zeros(NUM_EXPERTS))


# End-to-end equivalence

@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("use_norm", [True, False])
def test_forward_matches_manual_computation(kda_config_factory, renormalize, use_norm):
	config, moe = _make_moe(
		kda_config_factory, moe_renormalize=renormalize, latent_moe_use_norm=use_norm, routed_scaling_factor=1.7,
	)
	with torch.no_grad():
		moe.quantile_balancer.biases.copy_(torch.tensor([0.05, -0.1, 0.2, -0.15]))
	biases = moe.quantile_balancer.biases.clone() # forward updates the biases after routing
	x = torch.randn(2, 5, config.hidden_size)

	with torch.no_grad():
		out = moe(x)
		expected = _manual_forward(moe, config, x, biases)

	torch.testing.assert_close(out, expected)


# Routing vs weighting

def test_biases_change_selection_but_not_mixture_weights(kda_config_factory):
	config, moe = _make_moe(
		kda_config_factory, num_experts_per_token=1, moe_renormalize=False, latent_moe_use_norm=False,
		routed_scaling_factor=1.5,
	)
	logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]]) # unbiased top-1 is expert 0
	x = _route_on_leading_features(moe, torch.randn(1, 1, config.hidden_size), logits)
	with torch.no_grad():
		moe.quantile_balancer.biases.copy_(torch.tensor([0.0, 0.0, 0.0, 5.0])) # biased top-1 is expert 3

	calls = {e: 0 for e in range(NUM_EXPERTS)}
	for e, expert in enumerate(moe.experts):
		expert.register_forward_hook(lambda module, args, out, e=e: calls.__setitem__(e, calls[e] + 1))
	routed_latent = _capture_up_latent_input(moe)

	with torch.no_grad():
		moe(x)

	# Selection used router_scores + biases.
	assert calls == {0: 0, 1: 0, 2: 0, 3: 1}

	# The weight is the unbiased sigmoid score of expert 3, not sigmoid + bias.
	with torch.no_grad():
		z = moe.down_latent_proj(x.view(1, -1))
		y3 = moe.experts[3](z)
	unbiased_weight = torch.sigmoid(torch.tensor(-1.0))
	torch.testing.assert_close(routed_latent[0].view(1, -1), unbiased_weight * 1.5 * y3)
	assert not torch.allclose(routed_latent[0].view(1, -1), (unbiased_weight + 5.0) * 1.5 * y3)


# Sparse dispatch and accumulation

def test_each_expert_processes_only_its_assigned_tokens(kda_config_factory):
	config, moe = _make_moe(kda_config_factory)
	x = _route_on_leading_features(moe, torch.randn(1, 4, config.hidden_size), DISPATCH_LOGITS)

	seen = {}
	for e, expert in enumerate(moe.experts):
		expert.register_forward_hook(lambda module, args, out, e=e: seen.__setitem__(e, args[0].detach().clone()))

	with torch.no_grad():
		moe(x)
		z = moe.down_latent_proj(x.view(4, -1))

	for e in range(NUM_EXPERTS):
		expected_tokens = [t for t, sel in enumerate(DISPATCH_SELECTION) if e in sel]
		assert e in seen, f"expert {e} was never called"
		torch.testing.assert_close(seen[e], z[expected_tokens], msg=lambda m, e=e: f"expert {e}: {m}")


def test_index_add_accumulates_multiple_expert_contributions(kda_config_factory):
	config, moe = _make_moe(kda_config_factory, latent_moe_use_norm=False, routed_scaling_factor=1.3)
	x = _route_on_leading_features(moe, torch.randn(1, 4, config.hidden_size), DISPATCH_LOGITS)
	routed_latent = _capture_up_latent_input(moe)

	with torch.no_grad():
		moe(x)
		expected, selections = _manual_routed_latent(moe, config, x, torch.zeros(NUM_EXPERTS))

	assert selections == DISPATCH_SELECTION
	torch.testing.assert_close(routed_latent[0].view(4, -1), expected)

	# Each token's latent is the sum of two non-zero expert contributions, not just one of them.
	with torch.no_grad():
		z = moe.down_latent_proj(x.view(4, -1))
		scores = torch.sigmoid(DISPATCH_LOGITS)
		only_first = scores[0, 0] / scores[0, :2].sum() * 1.3 * moe.experts[0](z[0])
	assert not torch.allclose(routed_latent[0][0, 0], only_first)


# Renormalization and routed scaling

def test_renormalization_normalizes_top_k_weights_before_scaling(kda_config_factory):
	config, moe = _make_moe(kda_config_factory, moe_renormalize=True, latent_moe_use_norm=False, routed_scaling_factor=2.0)
	x = _route_on_leading_features(moe, torch.randn(1, 4, config.hidden_size), DISPATCH_LOGITS)
	routed_latent = _capture_up_latent_input(moe)

	with torch.no_grad():
		moe(x)
		z = moe.down_latent_proj(x.view(4, -1))

	scores = torch.sigmoid(DISPATCH_LOGITS)
	for t, selected in enumerate(DISPATCH_SELECTION):
		selected = sorted(selected)
		w = scores[t, selected] / scores[t, selected].sum()
		assert torch.isclose(w.sum(), torch.tensor(1.0))
		with torch.no_grad():
			expected = sum(2.0 * w_i * moe.experts[e](z[t]) for w_i, e in zip(w, selected))
			unnormalized = sum(2.0 * scores[t, e] * moe.experts[e](z[t]) for e in selected)
		torch.testing.assert_close(routed_latent[0][0, t], expected)
		assert not torch.allclose(routed_latent[0][0, t], unnormalized)


@pytest.mark.parametrize("renormalize", [True, False])
def test_routed_scaling_factor_scales_routed_contribution(kda_config_factory, renormalize):
	config, moe = _make_moe(kda_config_factory, moe_renormalize=renormalize, latent_moe_use_norm=False)
	x = torch.randn(2, 5, config.hidden_size)
	routed_latent = _capture_up_latent_input(moe)

	outputs = {}
	for factor in [1.0, 2.5]:
		moe.routed_scaling_factor = factor
		with torch.no_grad():
			moe.quantile_balancer.biases.zero_() # same routing on both runs
			outputs[factor] = moe(x)

	torch.testing.assert_close(routed_latent[1], 2.5 * routed_latent[0])
	with torch.no_grad():
		shared = moe.shared_experts(x)
	torch.testing.assert_close(outputs[2.5] - shared, 2.5 * (outputs[1.0] - shared))


# Backward

@pytest.mark.parametrize("use_norm", [True, False])
def test_backward_reaches_router_latents_shared_and_selected_experts(kda_config_factory, use_norm):
	config, moe = _make_moe(kda_config_factory, latent_moe_use_norm=use_norm)
	# Every expert is selected by exactly two tokens.
	x = _route_on_leading_features(moe, torch.randn(1, 4, config.hidden_size), DISPATCH_LOGITS)
	x.requires_grad_(True)

	moe(x).square().mean().backward()

	assert torch.isfinite(x.grad).all()
	expected = ["router.weight", "down_latent_proj.weight", "up_latent_proj.weight"]
	expected += [n for n, _ in moe.shared_experts.named_parameters(prefix="shared_experts")]
	expected += [n for n, _ in moe.experts.named_parameters(prefix="experts")]
	if use_norm:
		expected.append("post_router_rmsnorms.weight")
	params = dict(moe.named_parameters())
	for name in expected:
		assert params[name].grad is not None, f"no gradient for {name}"
		assert torch.isfinite(params[name].grad).all(), f"non-finite gradient for {name}"
		assert params[name].grad.abs().sum() > 0, f"zero gradient for {name}"

	biases = moe.quantile_balancer.biases
	assert not biases.requires_grad
	assert biases.grad is None


def test_unselected_expert_receives_no_gradient(kda_config_factory):
	config, moe = _make_moe(kda_config_factory, num_experts_per_token=1)
	logits = torch.tensor([[3.0, 0.0, -1.0, -2.0], [0.0, 3.0, -1.0, -2.0]]) # only experts 0 and 1 are used
	x = _route_on_leading_features(moe, torch.randn(1, 2, config.hidden_size), logits)

	moe(x).square().mean().backward()

	for e in [0, 1]:
		assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in moe.experts[e].parameters())
	for e in [2, 3]:
		assert all(p.grad is None for p in moe.experts[e].parameters())


# Output shape and dtype

@pytest.mark.parametrize("B, T", [(1, 1), (2, 5), (3, 8)])
def test_output_shape(kda_config_factory, B, T):
	config, moe = _make_moe(kda_config_factory)
	x = torch.randn(B, T, config.hidden_size)

	out = moe(x)

	assert out.shape == (B, T, config.hidden_size)
	assert torch.isfinite(out).all()


def test_bf16_output_dtype_with_fp32_router_scores(kda_config_factory, monkeypatch):
	config, moe = _make_moe(kda_config_factory)
	moe = moe.to(torch.bfloat16)
	x = torch.randn(2, 5, config.hidden_size, dtype=torch.bfloat16)

	captured = {}
	original_update = moe.quantile_balancer.update

	def spy_update(top_scores, router_scores):
		captured["top_scores"] = top_scores.dtype
		captured["router_scores"] = router_scores.dtype
		return original_update(top_scores=top_scores, router_scores=router_scores)

	monkeypatch.setattr(moe.quantile_balancer, "update", spy_update)

	out = moe(x)

	assert out.dtype == torch.bfloat16
	assert torch.isfinite(out).all()
	assert captured == {"top_scores": torch.float32, "router_scores": torch.float32}
