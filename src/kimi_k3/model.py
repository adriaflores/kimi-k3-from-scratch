from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kimi_k3.attention import KDARecurrent as KDA
from src.kimi_k3.attention import GatedMLA

from src.kimi_k3.moe import StableLatentMoE
from src.kimi_k3.ffn import DenseFFN


def apply_attn_res(block_residual: torch.Tensor, prefix_sum: torch.Tensor, norm: nn.RMSNorm, proj: nn.Linear):
	# We use tensors in their reshaped form

	V = torch.cat([
					block_residual, # (BT, previous_blocks, C)
					prefix_sum.unsqueeze(1), # (BT, 1, C)
				], dim=1) # (BT, K, C), where K = previous blocks + current prefix
	
	scores = proj(norm(V)).squeeze(-1) # (BT, K)
	weights = scores.softmax(dim=-1) # (BT, K)

	h = (weights.unsqueeze(-1) * V).sum(dim=1) # ((BT, K, 1) * (BT, K, C)).sum(dim=1) -> (BT, C)

	return h


class KimiK3DecoderBlock(nn.Module):
	def __init__(self, config, layer_idx):
		super().__init__()

		C = config.hidden_size

		self.layer_idx = layer_idx
		self.attn_res_block_size = config.attn_res_block_size

		# Attention sublayer
		layer_type = config.layer_types[layer_idx]

		if layer_type == "kda":
			self.attn = KDA(config)

		elif layer_type == "mla":
			self.attn = GatedMLA(config)

		else:
			raise ValueError(f"Unknown layer type: {layer_type}")

		# FFN sublayer 
		if layer_idx < config.num_dense_layers:
			self.ffn = DenseFFN(config)
		else:
			self.ffn = StableLatentMoE(config)

		# PreNorms
		self.attn_norm = nn.RMSNorm(C, eps=config.rms_norm_eps)
		self.ffn_norm = nn.RMSNorm(C, eps=config.rms_norm_eps)

		# Attention Residual
		self.attn_res_norm = nn.RMSNorm(C, eps=config.rms_norm_eps)
		self.attn_res_proj = nn.Linear(C, 1, bias=False)

		self.mlp_res_norm = nn.RMSNorm(C, eps=config.rms_norm_eps)
		self.mlp_res_proj = nn.Linear(C, 1, bias=False)


	def forward(self, hidden_states, block_residual):
		B, T, C = hidden_states.shape

		prefix_sum = hidden_states

		# Attention Residual for the attention sublayer
		if block_residual.shape[1] > 0:
			hidden_states = apply_attn_res(block_residual, prefix_sum.view(-1, C), norm=self.attn_res_norm, proj=self.attn_res_proj).view(B, T, C)

		# Are we starting a new AttnRes block?
		if self.layer_idx % self.attn_res_block_size == 0:

			block_residual = torch.cat([
					block_residual, # (BT, previous_blocks, C)
					prefix_sum.reshape(B * T, C).unsqueeze(1), # (BT, 1, C)
				], dim=1) # -> (BT, K, C)

			prefix_sum = None

		h = self.attn_norm(hidden_states)
		attn_output = self.attn(h)

		if prefix_sum is None:
			prefix_sum = attn_output
		else:
			prefix_sum = prefix_sum + attn_output

		hidden_states = apply_attn_res(block_residual, prefix_sum.view(-1, C), norm=self.mlp_res_norm, proj=self.mlp_res_proj).view(B, T, C)

		h = self.ffn_norm(hidden_states)
		mlp_output = self.ffn(h)

		prefix_sum = prefix_sum + mlp_output

		return prefix_sum, block_residual 

class KimiK3(nn.Module):
	"""
	Kimi Team - Kimi K3: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
	"""

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.vocab_size = config.vocab_size
		self.hidden_size = config.hidden_size
		self.num_layers = config.num_layers
		self.max_seq_len = config.max_seq_len

		# Token embeddings
		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

		# Kimi K3 decoder stack
		self.layers = nn.ModuleList([
			KimiK3DecoderBlock(config, layer_idx) for layer_idx in range(config.num_layers)
		])

		# Final normalization
		self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

		# Language-model head
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		# Attention Residual output mixing
		self.output_attn_res_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

		self.apply(self._init_weights)

		# Weight tying: input embeddings and output vocabulary projection. 
		# Not present in the original Kimi K3 implementation, but we add it here for efficiency and simplicity.
		self.lm_head.weight = self.embed_tokens.weight

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			nn.init.normal_(
				module.weight,
				mean=0.0,
				std=self.config.initializer_range,
			)

			if isinstance(module, nn.Linear) and module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(self, idx, targets=None):
		B, T = idx.shape

		if T > self.max_seq_len:
			raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}")

		x = self.embed_tokens(idx) # (B, T) -> (B, T, C)

		block_residual = x.new_zeros(B * T, 0, self.hidden_size)

		for layer in self.layers:
			x, block_residual = layer(x, block_residual=block_residual)

		x = apply_attn_res(block_residual, x.view(-1, self.hidden_size), norm=self.output_attn_res_norm, proj=self.output_attn_res_proj).view(B, T, self.hidden_size)

		x = self.norm(x)

		logits = self.lm_head(x) # (B, T, C) -> (B, T, V)

		loss = None
		if targets is not None:
			loss = F.cross_entropy(
				logits.view(-1, self.vocab_size),
				targets.view(-1),
			)

		return logits, loss

@dataclass
class KimiK3Config:

	# Backbone
	vocab_size: int # Number of tokens in the tokenizer vocabulary.
	hidden_size: int # Main model dimension (d_model); size of each token representation.
	num_layers: int # Total number of Transformer/Kimi decoder layers.
	rms_norm_eps: float # Numerical-stability epsilon used by RMSNorm layers.
	initializer_range: float # Standard deviation used when initializing model weights.

	# Context
	max_seq_len: int # Maximum sequence/context length supported by the model.

	# Layer schedule
	num_dense_layers: int # Number of initial layers using a dense FFN instead of MoE.
	attn_res_block_size: int # Number of layers grouped into each Attention Residual block.
	layer_types: tuple[str, ...] # Attention type used by every layer, e.g. ("kda", "kda", "mla", ...).

	# KDA
	kda_num_heads: int # Number of parallel Kimi Delta Attention heads.
	kda_head_dim: int # Dimension of each KDA head.
	kda_short_conv_kernel_size: int # Kernel size of the short causal convolutions applied to Q, K and V.
	kda_gate_lower_bound: float # Lower bound applied to KDA's decay/update gate for numerical stability.
	kda_full_rank_output_gate: bool # Whether KDA's output gate uses a full-rank projection instead of the factorized low-rank projection.
	kda_chunk_size: int # Number of tokens per chunk in the chunkwise KDA; the sequence length must be divisible by it.

	# Gated MLA
	mla_num_heads: int # Number of Multi-Head Latent Attention heads.
	mla_q_lora_rank: int # Rank of the low-rank query (Q) projection.
	mla_kv_lora_rank: int # Rank of the compressed latent representation shared by K and V.
	mla_qk_nope_head_dim: int # Per-head Q/K dimension that does not receive positional (RoPE) encoding.
	mla_qk_shared_head_dim: int # Dimension of the additional Q/K subspace whose K representation is shared across all attention heads.
	mla_v_head_dim: int # Dimension of the value representation for each attention head.
	mla_output_gate: bool # Whether to apply a learned sigmoid gate to MLA outputs before o_proj.

	# Dense FFN
	dense_intermediate_size: int # Hidden/intermediate dimension of the dense feed-forward network.

	# Stable LatentMoE
	num_experts: int # Total number of routed experts available in each MoE layer.
	num_experts_per_token: int # Number of routed experts selected for each token (top-k).
	num_shared_experts: int # Number of experts evaluated for every token, independently of routing.
	latent_moe_dim: int # Dimension of the shared latent space in which routed experts operate; corresponds to routed_expert_hidden_size in the official implementation.
	expert_intermediate_size: int # Intermediate/hidden dimension inside each expert MLP.
	latent_moe_use_norm: bool # Whether to RMS-normalize the routed-expert output in latent space before projecting it back to hidden_size.
	moe_renormalize: bool # Whether routing weights of the selected top-k experts are renormalized.

	# SiTU-GLU
	situ_beta: float # Beta parameter controlling the nonlinear SiTU activation component.
	situ_linear_beta: float # Beta parameter controlling SiTU's linear-transition component.