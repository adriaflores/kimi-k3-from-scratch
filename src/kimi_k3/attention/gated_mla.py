import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMLA(nn.Module):

	def __init__(self, config):
		super().__init__()
		
		self.d_model = config.hidden_size
		self.num_heads = config.mla_num_heads
		self.q_latent_dim = config.mla_q_lora_rank
		self.kv_latent_dim = config.mla_kv_lora_rank
		self.d_head = config.mla_qk_nope_head_dim
		self.d_shared = config.mla_qk_shared_head_dim
		self.dv_head = config.mla_v_head_dim
		self.output_gate = config.mla_output_gate

		self.q_rmsnorm = nn.RMSNorm(normalized_shape=self.q_latent_dim, eps=config.rms_norm_eps)
		self.kv_rmsnorm = nn.RMSNorm(normalized_shape=self.kv_latent_dim, eps=config.rms_norm_eps)

		if self.q_latent_dim >= self.d_model:
			raise ValueError("q_latent_dim must be smaller than d_model")

		if self.kv_latent_dim >= self.d_model:
			raise ValueError("kv_latent_dim must be smaller than d_model")

		# Down Projections		
		self.W_DQ = nn.Linear(self.d_model, self.q_latent_dim, bias = False)
		self.W_DKV = nn.Linear(self.d_model, self.kv_latent_dim + self.d_shared, bias = False)

		# Up Projections
		self.W_UQ = nn.Linear(self.q_latent_dim, self.num_heads * (self.d_head + self.d_shared), bias = False)
		self.W_UKV = nn.Linear(self.kv_latent_dim, self.num_heads * (self.d_head + self.dv_head), bias = False)

		#Output Gate
		if self.output_gate:
			self.Wg = nn.Linear(self.d_model, self.num_heads * self.dv_head, bias = False)

		# Output Matrix
		self.Wo = nn.Linear(self.num_heads * self.dv_head, self.d_model, bias = False)

	def forward(self, x: torch.Tensor):
		B, T, _ = x.shape

		C_q = self.q_rmsnorm(self.W_DQ(x)) # (B, T, q_latent_dim)
		kv = self.W_DKV(x) # (B, T, kv_latent_dim + D_s)
		C_kv, K_shared = kv.split([self.kv_latent_dim, self.d_shared], dim=-1) # (B, T, kv_latent_dim), # (B, T, D_s)
		C_kv = self.kv_rmsnorm(C_kv)

		Q = self.W_UQ(C_q) # (B, T, H * (D + D_s))
		Q = Q.view(B,T, self.num_heads, self.d_head + self.d_shared).transpose(1,2) # (B, H, T, D + D_s)

		KV = self.W_UKV(C_kv) # (B, T, H * (D + D_v))
		KV = KV.view(B, T, self.num_heads, self.d_head + self.dv_head) # (B, T, H, D + D_v)
		K_c, V = KV.split([self.d_head, self.dv_head], dim=-1) # (B, T, H, D), (B, T, H, D_v)
		K_c = K_c.transpose(1,2) # (B, H, T, D)
		V = V.transpose(1,2) # (B, H, T, D_v)
		K_shared = K_shared.unsqueeze(1) # (B, 1, T, D_s)
		K_shared = K_shared.expand(B, self.num_heads, T, self.d_shared) # (B, H, T, D_s)
		K = torch.cat([K_c, K_shared], dim=-1) # (B, H, T, D + D_s)

		# scores = (Q @ K.mT) # (B, H, T, T)
		# raw_scores = (scores) / math.sqrt(self.d_head + self.d_shared)
		# causal_mask = self.mask[:T, :T,]
		# raw_scores = raw_scores.masked_fill(causal_mask, float("-inf"))
		# attention_weights = F.softmax(raw_scores.float(), dim=-1).to(C_kv.dtype)
		# head_outputs = attention_weights @ V # (B, H, T, D_v)
		head_outputs = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=True)

		values = head_outputs.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.dv_head)  # (B, T, H * D_v)

		if self.output_gate:
			gate = torch.sigmoid(self.Wg(x)) # (B, T, H * D_v)
			x_new = self.Wo(gate * values) # (B, T, C)

		else:
			x_new = self.Wo(values) # (B, T, C)

		return x_new