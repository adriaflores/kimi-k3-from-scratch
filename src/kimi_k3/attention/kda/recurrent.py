import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KDARecurrent(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.d_model = config.hidden_size
		self.num_heads = config.kda_num_heads
		self.d_head = config.kda_head_dim
		self.eps = 1e-6

		self.kda_dim = self.num_heads * self.d_head
		self.scale = self.d_head ** -0.5

		self.QKV = nn.Linear(self.d_model, 3*self.kda_dim, bias = False)

		self.kernel_size = config.kda_short_conv_kernel_size
		self.q_conv = nn.Conv1d(in_channels=self.kda_dim, out_channels=self.kda_dim, kernel_size=self.kernel_size, groups=self.kda_dim, bias=False)
		self.k_conv = nn.Conv1d(in_channels=self.kda_dim, out_channels=self.kda_dim, kernel_size=self.kernel_size, groups=self.kda_dim, bias=False)
		self.v_conv = nn.Conv1d(in_channels=self.kda_dim, out_channels=self.kda_dim, kernel_size=self.kernel_size, groups=self.kda_dim, bias=False)

		self.Wb = nn.Linear(self.d_model, self.num_heads, bias = False)
		self.Wad = nn.Linear(self.d_model, self.d_head, bias = False)
		self.Wau = nn.Linear(self.d_head, self.num_heads * self.d_head, bias = False)
		self.out_norm = nn.RMSNorm(self.d_head, eps=config.rms_norm_eps)

		self.full_rank_output_gate = config.kda_full_rank_output_gate
		if self.full_rank_output_gate:
			self.Wg = nn.Linear(self.d_model, self.kda_dim, bias=False)
		else:
			self.Wgd = nn.Linear(self.d_model, self.d_head, bias=False)
			self.Wgu = nn.Linear(self.d_head, self.kda_dim, bias=False)

		self.Wo = nn.Linear(self.kda_dim, self.d_model, bias = False)

		A_init = torch.empty(self.num_heads, dtype=torch.float).uniform_(1, 16)
		self.A_log = nn.Parameter(A_init.log())
		dt_init = torch.empty(self.num_heads * self.d_head, dtype=torch.float).uniform_(math.log(0.001), math.log(0.1)).exp()
		self.dt_bias = nn.Parameter(dt_init + torch.log(-torch.expm1(-dt_init)))
		self.gate_lower_bound = config.kda_gate_lower_bound

	def forward(self, x):
				
		B, T, C = x.shape

		q, k, v = self.QKV(x).split(self.kda_dim, dim=-1) # (B, T, KDA) each matrix

		q = q.transpose(1, 2) # (B, KDA, T)
		k = k.transpose(1, 2) # (B, KDA, T)
		v = v.transpose(1, 2) # (B, KDA, T)

		padding = self.kernel_size - 1
		q = F.pad(q, (padding, 0))  # (B, KDA, T+3)
		k = F.pad(k, (padding, 0)) # (B, KDA, T + 3)
		v = F.pad(v, (padding, 0)) # (B, KDA, T + 3)

		q = F.silu(self.q_conv(q))  # (B, KDA, T)
		q = q.contiguous().view(B, self.num_heads, self.d_head, T).transpose(-1, -2) # (B, H, T, D)
		
		k = F.silu(self.k_conv(k))  # (B, KDA, T)
		k = k.contiguous().view(B, self.num_heads, self.d_head, T).transpose(-1, -2) # (B, H, T, D)

		v = F.silu(self.v_conv(v))  # (B, KDA, T)
		v = v.contiguous().view(B, self.num_heads, self.d_head, T).transpose(-1, -2) # (B, H, T, D)

		out_dtype = v.dtype

		q, k, v = map(lambda x: x.to(torch.float), [q, k, v])

		q = F.normalize(q, p=2, dim=-1, eps=self.eps)  # (B, H, T, D)
		k = F.normalize(k, p=2, dim=-1, eps=self.eps)  # (B, H, T, D)

		beta = torch.sigmoid(self.Wb(x).float()) # (B, T, H)
		beta = beta.transpose(1,2).unsqueeze(-1)  # (B, H, T, 1)

		a = self.Wau(self.Wad(x)).float() # (B, T, H*D)

		dt = a + self.dt_bias # (B, T, H*D)
		dt = dt.view(B, T, self.num_heads, self.d_head) # (B, T, H, D)

		A = torch.exp(self.A_log)  # (H)
		A = A.view(1, 1, self.num_heads, 1) # (1, 1, H, 1)

		alpha = torch.exp(self.gate_lower_bound * torch.sigmoid(A * dt)) # (B, T, H, D)

		
		S = k.new_zeros((B, self.num_heads, self.d_head, self.d_head))

		outputs = []

		for t in range(T):
			q_t = q[:, :, t]      # (B, H, D_dot)
			k_t = k[:, :, t]      # (B, H, D_dot)
			v_t = v[:, :, t]      # (B, H, D_v)
			alpha_t = alpha[:, t].unsqueeze(-1)  # (B, H, D_k, 1)

			S = alpha_t * S # (B, H, D_k, D_v)

			v_old = torch.einsum("bhkv,bhk->bhv", S, k_t)
			delta_v = v_t - v_old
			update = k_t.unsqueeze(-1) * delta_v.unsqueeze(-2) # (B, H, D_k, D_v)
			beta_t = beta[:, :, t, :].unsqueeze(-1) # (B, H, 1, 1)
			S = S + beta_t * update
			y_t = self.scale * torch.einsum("bhkv,bhk->bhv", S, q_t) # (B, H, D_v)
			outputs.append(y_t)

		o = torch.stack(outputs, dim=1)  # (B, T, H, D)
		o = o.to(out_dtype)
		o = self.out_norm(o)            # (B, T, H, D)

		if self.full_rank_output_gate:
			g = self.Wg(x) # (B, T, KDA)  
		else:
			g = self.Wgu(self.Wgd(x)) # (B, T, KDA) 
		g = torch.sigmoid(g)       
		g = g.view(B, T, self.num_heads, self.d_head) # (B, T, H, D)

		o = o * g                      
		o = o.contiguous().view(B,T, self.kda_dim)
		x_new = self.Wo(o)

		return x_new