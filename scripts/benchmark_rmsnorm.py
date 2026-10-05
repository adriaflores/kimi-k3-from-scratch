import argparse
import time
import warnings

import torch
import torch.nn.functional as F


def rmsnorm_baseline(x, weight, eps):
	# Comportamiento actual: activación BF16 + weight FP32
	return F.rms_norm(x, (x.shape[-1],), weight, eps)


def rmsnorm_cast_weight(x, weight, eps):
	# Candidato: mantener parámetro maestro FP32,
	# pero hacer la operación con weight en el dtype de x.
	return F.rms_norm(
		x,
		(x.shape[-1],),
		weight.to(x.dtype),
		eps,
	)


def check_correctness(b, t, c, eps):
	torch.manual_seed(0)

	x1 = torch.randn(b, t, c, device="cuda", dtype=torch.bfloat16, requires_grad=True)
	x2 = x1.detach().clone().requires_grad_(True)

	w1 = torch.randn(c, device="cuda", dtype=torch.float32, requires_grad=True)
	w2 = w1.detach().clone().requires_grad_(True)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		y1 = rmsnorm_baseline(x1, w1, eps)

	y2 = rmsnorm_cast_weight(x2, w2, eps)

	loss1 = y1.float().square().mean()
	loss2 = y2.float().square().mean()

	loss1.backward()
	loss2.backward()

	torch.cuda.synchronize()

	print("\nCorrectness")
	print("-----------")
	print(f"baseline output dtype : {y1.dtype}")
	print(f"candidate output dtype: {y2.dtype}")
	print(f"master weight dtype   : {w2.dtype}")
	print(f"master grad dtype     : {w2.grad.dtype}")

	print(
		f"output max abs diff   : "
		f"{(y1.float() - y2.float()).abs().max().item():.6g}"
	)
	print(
		f"x grad max abs diff   : "
		f"{(x1.grad.float() - x2.grad.float()).abs().max().item():.6g}"
	)
	print(
		f"weight grad max diff  : "
		f"{(w1.grad - w2.grad).abs().max().item():.6g}"
	)


def benchmark(fn, b, t, c, eps, warmup, iterations):
	torch.manual_seed(0)

	x = torch.randn(
		b, t, c,
		device="cuda",
		dtype=torch.bfloat16,
		requires_grad=True,
	)
	weight = torch.randn(
		c,
		device="cuda",
		dtype=torch.float32,
		requires_grad=True,
	)

	# Warmup
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		for _ in range(warmup):
			x.grad = None
			weight.grad = None

			y = fn(x, weight, eps)
			loss = y.float().square().mean()
			loss.backward()

	torch.cuda.synchronize()

	start = time.perf_counter()

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		for _ in range(iterations):
			x.grad = None
			weight.grad = None

			y = fn(x, weight, eps)
			loss = y.float().square().mean()
			loss.backward()

	torch.cuda.synchronize()

	elapsed = time.perf_counter() - start

	return elapsed / iterations * 1e6


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--seq-len", type=int, default=128)
	parser.add_argument("--hidden-size", type=int, default=640)
	parser.add_argument("--eps", type=float, default=1e-5)
	parser.add_argument("--warmup", type=int, default=100)
	parser.add_argument("--iterations", type=int, default=1000)
	args = parser.parse_args()

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required")

	if not torch.cuda.is_bf16_supported():
		raise RuntimeError("Native BF16 support is required")

	print(f"GPU: {torch.cuda.get_device_name()}")
	print(
		f"shape: ({args.batch_size}, {args.seq_len}, {args.hidden_size})"
		f" | dtype=bf16 | weight=fp32"
	)

	check_correctness(
		args.batch_size,
		args.seq_len,
		args.hidden_size,
		args.eps,
	)

	baseline = benchmark(
		rmsnorm_baseline,
		args.batch_size,
		args.seq_len,
		args.hidden_size,
		args.eps,
		args.warmup,
		args.iterations,
	)

	candidate = benchmark(
		rmsnorm_cast_weight,
		args.batch_size,
		args.seq_len,
		args.hidden_size,
		args.eps,
		args.warmup,
		args.iterations,
	)

	speedup = baseline / candidate
	improvement = (baseline - candidate) / baseline * 100

	print("\nForward + backward benchmark")
	print("----------------------------")
	print(f"baseline : {baseline:.2f} us/iter")
	print(f"candidate: {candidate:.2f} us/iter")
	print(f"speedup  : {speedup:.3f}x")
	print(f"improvement: {improvement:+.2f}%")


if __name__ == "__main__":
	main()