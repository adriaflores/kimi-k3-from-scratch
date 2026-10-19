---
title: "KimiK3, decoder blocks and configuration"
description: "Assembles the hybrid-attention decoder, attention residuals, tied embeddings, and language-model loss."
sourcePath: "src/kimi_k3/model.py"
status: review
---

`model.py` is the model's composition layer. It selects an attention implementation for each decoder block, decides whether that block uses a dense FFN or Stable LatentMoE, manages **Attention Residuals (AttnRes)** across blocks, and exposes the vocabulary projection and optional next-token cross-entropy. The internal KDA/MLA/MoE algorithms are documented in their own source files.

## Select KDA without importing FLA on CPU

`build_kda(config)` chooses the fused wrapper only when `config.kda_implementation == "fused"` [source L15–L20](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L15-L20). The FLA-backed module is imported inside that branch so constructing the ordinary PyTorch reference model does not import Triton/FLA as an incidental dependency. Any other supported setting returns `KDAChunkwise`. The dataclass validator later restricts valid implementations to `"chunkwise"` and `"fused"`.

## AttnRes mixes old block endpoints and the live prefix

`apply_attn_res(block_residual, prefix_sum, norm, proj)` [source L23–L36](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L23-L36) receives a flat batch-time axis `BT`, stored block endpoints `(BT,P,M)` and the current prefix `(BT,M)`. It concatenates the prefix as one more candidate, giving `V` of shape `(BT,P+1,M)`:

```python
	scores = proj(norm(V)).squeeze(-1) # (BT, K)
	weights = scores.softmax(dim=-1) # (BT, K)

	h = (weights.unsqueeze(-1) * V).sum(dim=1) # ((BT, K, 1) * (BT, K, C)).sum(dim=1) -> (BT, C)
```

`norm` is applied **to candidate states before scoring**. The learned scalar projection is shared across candidates, softmax normalizes across the candidate axis, and the weighted sum uses the *unnormalized* candidate tensor `V`. No attention over token positions occurs here: `BT` indexes independent token positions, while `K=P+1` indexes residual candidates from network depth.

## One decoder block: the two sublayer boundaries

`KimiK3DecoderBlock.__init__` [source L39–L78](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L39-L78) reads the layer type from `config.layer_types[layer_idx]`. For `"kda"` it calls the factory; for `"mla"` it constructs `GatedMLA`; unexpected values raise. Early layers `layer_idx < num_dense_layers` get a `DenseFFN` with `SituGLU`; later layers use `StableLatentMoE`. The block stores separate pre-norms and separate learned AttnRes scorer parameters for its attention and FFN inputs.

`forward(hidden_states, block_residual)` [source L80–L114](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L80-L114) first names the incoming hidden state `prefix_sum`. If earlier block endpoints exist, an AttnRes mixture becomes the attention input. On layers where `layer_idx % attn_res_block_size == 0`, that incoming prefix is appended to stored endpoints and the live prefix accumulator is reset to `None`.

Next, the attention branch receives a **pre-normalized** hidden state. Its contribution either initializes the current prefix (new AttnRes block) or is added to the still-live prefix. The FFN input is a *new* AttnRes mixture over stored endpoints and this updated prefix. Its own pre-norm precedes the FFN computation. Finally the FFN output adds to `prefix_sum`, and the block returns both the new prefix and the endpoint tensor. This ordering explains why AttnRes in this file is not merely a post-layer residual addition.

### Trace the block's actual state machine

The function carries two objects between layers: `hidden_states` has shape `(B,T,M)`, and `block_residual` has shape `(B*T,P,M)` where `P` counts previously completed AttnRes block endpoints. It names the incoming layer value `prefix_sum`. Before opening a new block, any existing endpoints participate in the **attention-input** mixture, but that mixture is only a temporary `hidden_states` value; it does not replace `prefix_sum` ([source L80–L96](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L80-L96)):

```python
		B, T, C = hidden_states.shape

		prefix_sum = hidden_states

		# Attention Residual for the attention sublayer
		if block_residual.shape[1] > 0:
			hidden_states = apply_attn_res(block_residual, prefix_sum.view(-1, C), norm=self.attn_res_norm, proj=self.attn_res_proj).view(B, T, C)
```

At a boundary (`layer_idx % attn_res_block_size == 0`), the **incoming prefix, not that just-mixed attention input**, becomes a stored endpoint. The candidate count increases from `P` to `P+1`, and setting `prefix_sum = None` marks the beginning of a fresh in-block accumulator ([source L90–L102](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L90-L102)):

```python
		if self.layer_idx % self.attn_res_block_size == 0:

			block_residual = torch.cat([
					block_residual, # (BT, previous_blocks, C)
					prefix_sum.reshape(B * T, C).unsqueeze(1), # (BT, 1, C)
				], dim=1) # -> (BT, K, C)

			prefix_sum = None
```

That `None` sentinel changes the attention residual update: the first attention output *starts* the new prefix; later layers *add* attention to their existing prefix. The FFN does not consume that raw sum. It receives a newly scored AttnRes mixture including the updated prefix, then contributes back to the accumulator ([source L100–L114](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L100-L114)):

```python
		if prefix_sum is None:
			prefix_sum = attn_output
		else:
			prefix_sum = prefix_sum + attn_output

		hidden_states = apply_attn_res(block_residual, prefix_sum.view(-1, C), norm=self.mlp_res_norm, proj=self.mlp_res_proj).view(B, T, C)

		h = self.ffn_norm(hidden_states)
		mlp_output = self.ffn(h)

		prefix_sum = prefix_sum + mlp_output
```

For a concrete **block size of 2**, consider the first four layers. At layer 0 the endpoint history starts empty (`P=0`), so the attention input is just the embedding. The boundary stores that embedding (`P=1`) and resets the accumulator; the FFN then scores `[embedding, attention output]`. At layer 1, the attention scorer sees `[embedding, incoming prefix]`, while the FFN scorer sees `[embedding, incoming prefix + current attention output]`. No endpoint is added there. At layer 2, attention *first* scores `[embedding, completed block-0 prefix]`. Only afterward does the boundary store that completed prefix (`P=2`) and reset `prefix_sum=None`; the FFN can now score `[embedding, completed block-0 prefix, current attention output]`. Layer 3 retains those two stored endpoints and uses the evolving block-1 prefix as the third candidate. These are **depth candidates for one token**, not preceding tokens in the sequence.

This ordering also explains the final model-level call: the last layer returns its current prefix and stored endpoints, and `KimiK3.forward` makes one last AttnRes selection over both ([source L166–L183](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L166-L183)). No endpoint or prefix state survives between separate calls to `KimiK3.forward`.

### Endpoint lifetime

The endpoint tensor's axis is `(BT,P,M)`, rather than a list of tensors in Python. It grows only at an AttnRes block boundary, not at every sublayer. Each decoder layer receives the updated endpoint tensor and current prefix from the previous layer. An empty candidate history is valid at the first layer because `prefix_sum` still provides a candidate. No state is cached across independent model forward calls.

## Embeddings, final mixture, and loss

`KimiK3.__init__` [source L121–L153](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L121-L153) creates token embeddings `Embedding(V,M)`, a `ModuleList` of `num_layers` decoder blocks, final RMSNorm, a vocabulary `Linear(M,V)` head, and final output AttnRes scorer. After initializing linear and embedding weights from the configured normal distribution, it performs the explicit tying:

```python
		self.lm_head.weight = self.embed_tokens.weight
```

This deliberately shares one `Parameter` object between input embeddings and the classifier head; its constructor comment says it is a project-specific efficiency deviation, not a claim about the original Kimi implementation. The optimizer's assignment logic deduplicates this tied parameter.

The forward at [source L166–L192](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L166-L192) accepts `idx` of shape `(B,T)` and optional `targets` of matching token shape. It rejects `T > max_seq_len`, embeds IDs to `(B,T,M)`, initializes an empty endpoint tensor, and runs the decoder stack. It then applies final AttnRes, RMSNorm and the tied head to produce logits `(B,T,V)`. If targets are provided, `F.cross_entropy(logits.view(-1,V), targets.view(-1))` averages categorical next-token loss over all `B*T` positions; otherwise `loss is None`.

The code does not implement a KV cache, padding mask, sliding context window, or an inference-only path. Those differences matter when interpreting `inference.py` generation, which recomputes the prefix.

## KimiK3Config enforces structural invariants

The `@dataclass KimiK3Config` fields occupy [source L194–L245](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L194-L245), with `__post_init__` validation at [source L247–L372](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L247-L372). It normalizes `layer_types` to a tuple and checks that the schedule length equals `num_layers`, each attention type is known, `num_dense_layers` lies in `[0,num_layers]`, and AttnRes block size is positive. It validates positive dimensions, ranks below hidden width, MoE top-k strictly smaller than the number of experts, positive SiTU hyperparameters and supported backend settings.

For `"fused"` KDA, the validator additionally requires a chunk size of 32 or 64 and decay bound in `[-5,0)` [source L310–L320](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L310-L320). These backend restrictions do not apply to the ordinary `"chunkwise"` configuration; the latter's forward still requires a length divisible by chunk size. The config does **not** prescribe a fixed 13-layer or 300M-parameter model: those values come from a particular YAML configuration, not the class definition.

### Examples of rejected model configurations

`KimiK3Config.__post_init__` catches structural errors when constructing the config, before a decoder is built ([source L247–L320](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L247-L320)):

| Configuration change | Exact validation consequence |
| --- | --- |
| `num_layers=4`, but `layer_types` has three entries | `len(layer_types) must equal num_layers` |
| A layer type of `"recurrent"` rather than `"kda"` or `"mla"` | `Unknown layer types` |
| `num_dense_layers` outside `[0, num_layers]` | Rejected before FFN assignment |
| `attn_res_block_size=0` | Division-by-zero is prevented before the forward |
| `kda_implementation="fused"` with `kda_chunk_size=16` | The fused backend permits only 32 or 64 |
| `num_experts_per_token == num_experts` | Top-k must be strictly smaller than expert count; the extra threshold candidate is needed for QB |

For the last example, the explicit branch is ([source L349–L353](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py#L349-L353)):

```python
		if not 1 <= self.num_experts_per_token < self.num_experts:
			raise ValueError("num_experts_per_token must be in [1, num_experts)")
```

These are constructor-time checks. For example, whether a particular *input sequence* is an exact multiple of the reference KDA chunk size is enforced by that attention implementation at forward time, not by `KimiK3Config` alone.

## Verification and boundaries

[test test_kimi_k3.py L27–L62](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_kimi_k3.py#L27-L62) checks logits/loss, target-free forward, context-length rejection, attention/FFN layer schedule and tied parameters. [test test_kimi_k3.py L65–L113](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_kimi_k3.py#L65-L113) checks parameter gradients, a small learning update, and causal-prefix invariance. [test test_config.py L4–L38](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_config.py#L4-L38) exercises top-k and fused KDA validation. These assertions were inspected in the pinned test sources, not rerun. AttnRes is implemented and tested as part of this model, but no independent benchmark or paper-scale architectural identity follows from these source checks.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/model.py">model.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_kimi_k3.py">test_kimi_k3.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/test_config.py">test_config.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
