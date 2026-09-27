import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KDAChunkwise(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.d_model = config.hidden_size
		self.num_heads = config.kda_num_heads
		self.d_head = config.kda_head_dim
		self.eps = 1e-6
		self.scale = self.d_head ** -0.5
		self.chunk_size = config.kda_chunk_size

		self.kda_dim = self.num_heads * self.d_head

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
		B, T, _ = x.shape

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

		log_alpha = (self.gate_lower_bound * torch.sigmoid(A * dt)) # (B, T, H, D), we want to store it in log space to avoid numerical issues
		log_alpha = log_alpha.transpose(1, 2) # (B, H, T, D)

		assert T % self.chunk_size == 0, f"Sequence length {T} must be divisible by chunk_size={self.chunk_size}"

		num_chunks = T // self.chunk_size
		q = q.contiguous().view(B, self.num_heads, num_chunks, self.chunk_size, self.d_head) # (B, H, N, C, D)
		k = k.contiguous().view(B, self.num_heads, num_chunks, self.chunk_size, self.d_head) # (B, H, N, C, D)
		v = v.contiguous().view(B, self.num_heads, num_chunks, self.chunk_size, self.d_head) # (B, H, N, C, D)
		log_alpha = log_alpha.contiguous().view(B, self.num_heads, num_chunks, self.chunk_size, self.d_head) # (B, H, N, C, D)
		beta = beta.contiguous().view(B, self.num_heads, num_chunks, self.chunk_size, 1) # (B, H, N, C, 1)

		log_gamma = log_alpha.cumsum(dim=-2)

		A = k.new_zeros(B, self.num_heads, num_chunks, self.chunk_size, self.chunk_size)
		beta = beta.squeeze(-1) # (B, H, N, C)

		for j in range(self.chunk_size):
			k_j = k[:, :, :, j, :].unsqueeze(-2) # (B, H, N, 1, D)
			g_j = log_gamma[:, :, : ,j, :].unsqueeze(-2) # (B, H, N, 1, D)
			g_future = log_gamma[..., j+1:, :] # (B, H, N, C-j-1, D)
			k_future = k[..., j+1:, :] # (B, H, N, C-j-1, D)
			beta_future = beta[..., j+1:] # (B, H, N, C-j-1)
			log_delta_gamma = g_future - g_j # (B, H, N, C-j-1, D)
			coeff = (k_future * torch.exp(log_delta_gamma) * k_j).sum(-1) # (B, H, N, C-j-1)
			coeff = -beta_future * coeff
			A[..., j+1:, j] = coeff

		I = torch.eye(self.chunk_size, device=A.device, dtype=A.dtype)
		T_ut = torch.linalg.solve_triangular(I - A, I, upper=False) # (B, H, N, C, C)
		L = torch.exp(log_gamma) * k # (B, H, N, C, D)
		beta_scalar = beta.unsqueeze(-2) # (B, H, N, 1, C)
		
		T_b = T_ut * beta_scalar
		R = T_b @ L # (B, H, N, C, D)
		U = T_b @ v # (B, H, N, C, D)

		S = k.new_zeros(B, self.num_heads, self.d_head, self.d_head)

		outputs = []

		for chunk in range(num_chunks):
			U_n = U[:, :, chunk, :, :] # (B, H, C, D)
			R_n = R[:, :, chunk, :, :] # (B, H, C, D)
			Z = U_n - R_n @ S
			q_n = q[:, :, chunk, :, :] # (B, H, C, D)
			log_gamma_n = log_gamma[:, :, chunk, :, :] # (B, H, C, D)
			Q_n = torch.exp(log_gamma_n) * q_n

			Aqk = k.new_zeros(B, self.num_heads, self.chunk_size, self.chunk_size)

			for j in range(self.chunk_size):
				k_j = k[:, :, chunk, j, :].unsqueeze(-2) # (B, H, 1, D)
				g_j = log_gamma[:, :, chunk ,j, :].unsqueeze(-2) # (B, H, 1, D)
				g_future = log_gamma[:, :, chunk, j:, :] # (B, H, C-j, D)
				log_delta_gamma = g_future - g_j # (B, H, C-j, D)
				q_t = q[:, : ,chunk, j:, :]
				coeff = (q_t * torch.exp(log_delta_gamma) * k_j).sum(-1) # (B, H, C-j)
				Aqk[..., j:, j] = coeff

			log_gamma_c = log_gamma[:, : , chunk, self.chunk_size-1, :]# (B, H, D)
			S_decay = torch.exp(log_gamma_c.unsqueeze(-1)) * S
			g_j = log_gamma[:, :, chunk , :, :] # (B, H, C, D)
			K_decay = torch.exp(log_gamma_c.unsqueeze(-2) - g_j) * k[:, :, chunk, :, :] # (B, H, C, D)
			S_new = S_decay + K_decay.mT @ Z
			O_n = self.scale * (Q_n @ S + Aqk @ Z) # (B, H, C, D)
			outputs.append(O_n)
			S = S_new

		o = torch.cat(outputs, dim=2) # (B, H, T, D)
		o = o.transpose(1, 2)  # (B, T, H, D)
		o = o.to(out_dtype)
		o = self.out_norm(o) # (B, T, H, D)

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