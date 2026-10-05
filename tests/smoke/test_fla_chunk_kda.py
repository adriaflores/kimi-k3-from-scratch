import pytest
import torch
import torch.nn.functional as F

fla_kda = pytest.importorskip("fla.ops.kda")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fla Triton kernels need a CUDA device")

B, T, H, K, V = 2, 128, 2, 64, 64


def _make_inputs(dtype):
	device = "cuda"
	q = torch.randn(B, T, H, K, device=device, dtype=dtype)
	k = torch.randn(B, T, H, K, device=device, dtype=dtype)
	v = torch.randn(B, T, H, V, device=device, dtype=dtype)
	# g is the per-channel decay in log space, so it must be <= 0.
	g = F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=torch.float32)).to(dtype)
	beta = torch.rand(B, T, H, device=device, dtype=dtype)
	return q, k, v, g, beta


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunk_kda_forward_runs(dtype):
	q, k, v, g, beta = _make_inputs(dtype)
	h0 = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

	o, ht = fla_kda.chunk_kda(
		q, k, v, g, beta,
		initial_state=h0,
		output_final_state=True,
		use_qk_l2norm_in_kernel=True,
	)

	assert o.shape == (B, T, H, V)
	assert ht.shape == (B, H, K, V)
	assert torch.isfinite(o).all()
	assert torch.isfinite(ht).all()


def test_chunk_kda_backward_runs():
	inputs = [t.requires_grad_() for t in _make_inputs(torch.bfloat16)]

	o, _ = fla_kda.chunk_kda(*inputs, use_qk_l2norm_in_kernel=True)
	o.float().pow(2).mean().backward()

	for name, t in zip(["q", "k", "v", "g", "beta"], inputs):
		assert t.grad is not None, name
		assert t.grad.shape == t.shape, name
		assert torch.isfinite(t.grad).all(), name


def test_chunk_kda_fused_kimi_k3_path_runs():
	# Raw logits: the kernel applies the gate activation, sigmoid(beta) and q/k L2 norm itself.
	q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
	k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
	v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
	g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
	beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
	A_log = torch.randn(H, device="cuda", dtype=torch.float32, requires_grad=True)
	dt_bias = torch.randn(H * K, device="cuda", dtype=torch.float32, requires_grad=True)

	o, _ = fla_kda.chunk_kda(
		q, k, v, g, beta,
		A_log=A_log,
		dt_bias=dt_bias,
		use_qk_l2norm_in_kernel=True,
		use_gate_in_kernel=True,
		use_beta_sigmoid_in_kernel=True,
		safe_gate=True,
		lower_bound=-5.0,
	)

	assert o.shape == (B, T, H, V)
	assert torch.isfinite(o).all()

	o.float().pow(2).mean().backward()

	params = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "A_log": A_log, "dt_bias": dt_bias}
	for name, t in params.items():
		assert t.grad is not None, name
		assert t.grad.shape == t.shape, name
		assert torch.isfinite(t.grad).all(), name
