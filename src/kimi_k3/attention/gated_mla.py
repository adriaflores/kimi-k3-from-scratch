import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMLA(nn.Module):
	"""
	Kimi Team .26 - KIMI K3: OPEN FRONTIER INTELLIGENCE

		Main differences from the official implementation:

		- Simplified attention dimensions: 
		This implementation assumes d_head = d_model / num_heads and uses the same head dimension for Q, K and V. Kimi K3 uses independent qk and value head dimensions, 
		so num_heads * head_dim is not constrained to equal d_model.

		- Simplified Q/K structure:
		Kimi K3 uses NoPE, but its implementation retains separate qk_nope_head_dim and qk_rope_head_dim branches inherited from MLA. This implementation collapses both into 
		a single per-head Q/K space and applies no RoPE.

		- Explicit inference-time matrix absorption:
		This implementation absorbs the Q and K up-projections into W_QK and performs attention directly against the cached KV latent. It also delays the per-head W_UV 
		projection until after the latent attention readout. These operations are mathematically equivalent to the original MLA projections, but differ from the straightforward 
		public reference forward implementation.

		- Simplified KV cache:
		Only the compressed KV latent is cached. The full K3 formulation also has the additional Q/K branch mentioned above, whose key component must be preserved alongside 
		the compressed latent in an optimized MLA cache.

		- Training precision:
		Kimi K3 keeps the FlashAttention output in FP32 during training to avoid biased low-precision rounding. This implementation converts the softmax probabilities back to 
		the model dtype and therefore does not reproduce that FP32 attention-output path.

		- Attention kernel:
		Kimi K3 uses a custom optimized FlashAttention training kernel whose on-chip memory layout overlaps the FP32 output tile with KV staging buffers, enabling a deeper 
		KV pipeline. This implementation uses ordinary PyTorch tensor operations and does not reproduce those kernel-level optimizations.

	"""

	def __init__(self, config):
		super().__init__()
		
		self.d_model = config.d_model
		self.num_heads = config.num_heads
		self.q_latent_dim = config.q_latent_dim
		self.kv_latent_dim = config.kv_latent_dim
		self.T = config.block_size
		self.B = config.max_batch_size

		self.q_rmsnorm = nn.RMSNorm(normalized_shape=self.q_latent_dim, eps=config.norm_eps)
		self.kv_rmsnorm = nn.RMSNorm(normalized_shape=self.kv_latent_dim, eps=config.norm_eps)

		if self.d_model % self.num_heads != 0:
			raise ValueError("d_model must be divisible by num_heads")

		self.d_head = self.d_model // self.num_heads

		if self.q_latent_dim >= self.d_model:
			raise ValueError("q_latent_dim must be smaller than d_model")

		if self.kv_latent_dim >= self.d_model:
			raise ValueError("kv_latent_dim must be smaller than d_model")

		# Down Projections		
		self.W_DQ = nn.Linear(self.d_model, self.q_latent_dim, bias = False)
		self.W_DKV = nn.Linear(self.d_model, self.kv_latent_dim, bias = False)

		# Up Projections
		self.W_UQ = nn.Linear(self.q_latent_dim, self.d_model, bias = False)
		self.W_UK = nn.Linear(self.kv_latent_dim, self.d_model, bias = False)
		self.W_UV = nn.Linear(self.kv_latent_dim, self.d_model, bias = False)

		#Output Gate
		self.Wg = nn.Linear(self.d_model, self.d_model, bias = False)

		# Output Matrix
		self.Wo = nn.Linear(self.d_model, self.d_model, bias = False)

		# Caching
		self.register_buffer("cache_kv", torch.zeros(self.B, self.T, self.kv_latent_dim), persistent=False,)

		mask = torch.triu(torch.ones(self.T, self.T, dtype=torch.bool,),diagonal=1)
		self.register_buffer("mask", mask, persistent=False)

		# Matrix absortion
		self.register_buffer("W_QK", None, persistent=False)

	@torch.no_grad()
	def absorb_weights(self):
		if self.training:
			raise RuntimeError("Weights should only be absorbed for inference")

		# W_UQ + W_UK
		W_UQ = self.W_UQ.weight.view(self.num_heads, self.d_head, self.q_latent_dim)
		W_UK = self.W_UK.weight.view(self.num_heads, self.d_head, self.kv_latent_dim)
		self.W_QK = W_UQ.transpose(-2, -1) @ W_UK # (H, q_latent_dim, kv_latent_dim)

	def forward(self, x: torch.Tensor, start_pos = 0, use_cache = False):
		B, T, C = x.shape

		if B > self.B:
			raise ValueError("Batch size exceeds maximum batch size")

		if start_pos + T > self.T:
			raise ValueError("Sequence exceeds maximum context length")

		if not use_cache and start_pos != 0:
			raise ValueError("start_pos must be 0 when KV cache is disabled")

		if use_cache and self.training:
			raise RuntimeError("KV cache should only be used during inference")

		C_q = self.q_rmsnorm(self.W_DQ(x)) # (B, T, q_latent_dim)
		C_kv = self.kv_rmsnorm(self.W_DKV(x)) # (B, T, kv_latent_dim)

		if use_cache:

			self.cache_kv = self.cache_kv.to(C_kv)
			self.cache_kv[:B, start_pos:start_pos + T] = C_kv
			C_kv = self.cache_kv[:B, :start_pos + T] # (B, S, kv_latent_dim)

		S = C_kv.size(1) # S represents T + cache

		if self.W_QK is not None and not self.training:

			Q_latent = torch.einsum("btq,hqk->bhtk", C_q, self.W_QK) # (B, H, T, k)
			content_scores = (Q_latent @ C_kv.mT.unsqueeze(1)) # (B, H, T, S)

		else:

			Q = self.W_UQ(C_q) # (B, T, C)
			K = self.W_UK(C_kv) # (B, S, C)

			Q = Q.view(B,T, self.num_heads, self.d_head).transpose(1,2) # (B, H, T, D)
			K = K.view(B,S, self.num_heads, self.d_head).transpose(1,2) # (B, H, S, D)

			content_scores = (Q @ K.mT) # (B, H, T, S)

		raw_scores = (content_scores) / math.sqrt(self.d_head)

		causal_mask = self.mask[start_pos:start_pos + T, :S,]
		raw_scores = raw_scores.masked_fill(causal_mask, float("-inf"))

		attention_weights = F.softmax(raw_scores.float(), dim=-1).to(C_kv.dtype)

		attention_latent = attention_weights @ C_kv.unsqueeze(1) # (B, H, T, kv_latent_dim)
		W_UV = self.W_UV.weight.view(self.num_heads, self.d_head, self.kv_latent_dim) # (H, D, kv_latent_dim)
		head_outputs = torch.einsum("bhtk,hdk->bhtd", attention_latent, W_UV) # (B, H, T, D)
		values = head_outputs.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, C)

		gate = torch.sigmoid(self.Wg(x)) # (B, T, C)

		x_new = self.Wo(gate * values) # (B, T, C)

		return x_new