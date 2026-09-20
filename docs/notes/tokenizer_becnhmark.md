# SmolLM2 Tokenizer Compression Check on FineWeb-Edu

## Purpose

Before generating the first persistent FineWeb-Edu training shards for the small Kimi K3 reimplementation, we ran a lightweight tokenizer sanity check.

The goal was **not** to prove that a tokenizer with better compression necessarily produces a better language model. Instead, the test was intended to answer a narrower engineering question:

> Does the selected SmolLM2 tokenizer represent our FineWeb-Edu data at a reasonable compression rate, without any obvious sign of tokenizer/data mismatch?

This check follows the tokenizer-selection research summarized separately in `TOKENIZER_RESEARCH.md`.

---

## Candidate tokenizer

We tested the tokenizer distributed with:

```text
HuggingFaceTB/SmolLM2-135M
```

The benchmark pinned the exact tokenizer revision:

```text
93efa2f097d58c2a74874c7e644dbc9b0cee75a2
```

Relevant tokenizer properties:

```text
Vocabulary size:       49,152
EOS token ID:          0
Fast tokenizer:        yes
Special tokens added:  no
EOS appended manually: yes
```

The released SmolLM2 tokenizer has a vocabulary size of 49,152 and uses `<|endoftext|>` as its base-model EOS token. Hugging Face also documents that the SmolLM tokenizer was trained on the **SmolLM Corpus**, making it particularly relevant to our setting because that corpus ecosystem contains substantial FineWeb-Edu data.

---

## Benchmark data

The benchmark used:

```text
Dataset:   HuggingFaceFW/fineweb-edu
Config:    sample-10BT
Split:     train
Revision:  87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
```

Sampling configuration:

```text
Maximum documents: 1,000
Seed:              1337
Shuffle buffer:    1,000
Sampling method:   buffer shuffle, then take
```

This is an **approximate streaming sample**, not a uniformly random sample of the complete FineWeb-Edu corpus. It is therefore appropriate for a cheap sanity check, but not for making precise corpus-wide statistical claims.

All 1,000 visited documents contained non-empty text.

---

## Results

| Metric | Result |
|---|---:|
| Source rows visited | 1,000 |
| Documents tokenized | 1,000 |
| Empty documents skipped | 0 |
| UTF-8 bytes | 4,839,652 |
| Unicode code points | 4,812,416 |
| Text tokens | 1,056,706 |
| Tokens including appended EOS | 1,057,706 |
| **Bytes/token** | **4.57994** |
| **Characters/token** | **4.55417** |
| Bytes/token including EOS | 4.57561 |
| Characters/token including EOS | 4.54986 |

The benchmark therefore gives approximately:

$$
\boxed{4.58\ \text{bytes/token}}
$$
and:

$$
\boxed{4.55\ \text{characters/token}}
$$

for SmolLM2 on this FineWeb-Edu sample.

The manually appended EOS token adds exactly 1,000 tokens, one per document. Relative to the 1,056,706 text tokens, this is only about **0.095% additional tokens**, so document-boundary markers have negligible impact on the overall token budget.

---

## Interpretation

### 1. No obvious compression mismatch

The observed **4.58 bytes/token** is a healthy result for a modern subword tokenizer on predominantly English web text.

As a useful external point of reference, Limisiewicz et al. (2026), in *Compute Optimal Tokenization*, report **4.57 bytes/token** for a popular BPE tokenizer used as a reference point in their scaling experiments.

Our measured value:

```text
SmolLM2 on sampled FineWeb-Edu: 4.57994 bytes/token
Reference BPE in Limisiewicz et al.: 4.57 bytes/token
```

The near equality should **not** be interpreted as proof that SmolLM2 is optimal. It does, however, indicate that our data are not being fragmented at an obviously pathological rate.

This is exactly what this benchmark was designed to establish.

### 2. The result is consistent with the expected domain match

SmolLM is a particularly low-risk existing tokenizer for this project because Hugging Face states that its 49,152-token tokenizer was trained on the **SmolLM Corpus**.

The SmolLM data ecosystem is strongly connected to FineWeb-Edu and related modern pretraining data. Since FineWeb-Edu is also our main pretraining corpus, we have substantially less reason to expect a severe tokenizer/domain mismatch than we would with an older tokenizer trained on a substantially different distribution.

This matters because tokenizer/data mismatch is not merely cosmetic. Ali et al. (2024) show that tokenizer choice can affect both downstream performance and training cost, while also warning that simple intrinsic metrics such as fertility are not always reliable predictors of downstream model quality.

### 3. Compression is a sanity check, not a model-quality metric

The result should **not** be interpreted as:

> Higher bytes/token is always better.

Schmidt et al. (2024), *Tokenization Is More Than Compression*, directly test this intuition and find that minimizing the number of tokens does not automatically improve downstream language-model performance.

Similarly, *Compute Optimal Tokenization* shows that the compute-optimal compression rate depends on the training regime.

Therefore the purpose of measuring `bytes/token` and `chars/token` here is to detect an obvious inefficiency or mismatch, not to optimize the tokenizer solely for maximum compression.

### 4. Token counts are tokenizer-dependent

The experiment also reinforces an important budgeting point.

A statement such as:

```text
train on 6B tokens
```

does not identify a tokenizer-independent amount of text. Different tokenizers can map the same raw corpus to different token counts.

For this project we should therefore record, whenever practical:

- total tokens processed,
- raw UTF-8 bytes represented,
- average bytes/token,
- tokenizer repository and exact revision.

This makes later compute and scaling comparisons easier to interpret.

---

## Decision

The benchmark reveals **no reason to reject the SmolLM2 tokenizer**.

For the first serious Kimi K3 training pipeline, we therefore freeze the current default as:

```text
Tokenizer:          HuggingFaceTB/SmolLM2-135M
Tokenizer revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
Vocabulary size:    49,152
EOS token ID:       0
Token storage:      uint16
```

The current evidence supports proceeding to the next stage:

> Generate and inspect the first FineWeb-Edu train/validation shards using this frozen tokenizer.

A custom tokenizer is **not justified at this stage**. Tokenizer design should be revisited only if:

1. later profiling uncovers a concrete weakness;
2. the training-data distribution changes substantially, for example toward code, mathematics, multilingual data, or a specialized domain;
3. lexical embedding/output matrices become unacceptable relative to the final model budget; or
4. tokenization itself becomes an explicit research question of the project.

---

## Limitations

This benchmark is intentionally small.

The 1,000-document sample is sufficient to detect gross tokenization problems, but it cannot establish:

- corpus-wide compression statistics with high precision;
- vocabulary utilization over the complete dataset;
- downstream model quality;
- whether another tokenizer would produce a statistically significant improvement;
- whether a custom tokenizer would be compute-optimal for the final Kimi K3 architecture.

A scientifically stronger tokenizer comparison would require training proxy language models under matched compute or raw-byte budgets. That experiment is outside the current scope because tokenization is not the primary research contribution of this repository.

---

## Reproducibility

Benchmark environment:

```text
Python:             3.12.13
datasets:           5.0.1
huggingface-hub:    1.32.0
numpy:              2.5.3
pyarrow:            25.0.1
transformers:       5.17.0
tokenizers:         0.23.2
```

Exact remote revisions:

```text
FineWeb-Edu:
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9

SmolLM2-135M:
93efa2f097d58c2a74874c7e644dbc9b0cee75a2
```

---

## References

1. Tomasz Limisiewicz, Artidoro Pagnoni, Srini Iyer, Mike Lewis, Sachin Mehta, Alisa Liu, Margaret Li, Gargi Ghosh, and Luke Zettlemoyer. **Compute Optimal Tokenization.** 2026.  
   https://arxiv.org/abs/2605.01188

2. Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, and Chris Tanner. **Tokenization Is More Than Compression.** EMNLP 2024.  
   https://aclanthology.org/2024.emnlp-main.40/

3. Mehdi Ali et al. **Tokenizer Choice For LLM Training: Negligible or Crucial?** Findings of NAACL 2024.  
   https://aclanthology.org/2024.findings-naacl.247/

4. Hugging Face. **SmolLM — blazingly fast and remarkably powerful.**  
   Documents the 49,152-token tokenizer trained on the SmolLM Corpus.  
   https://huggingface.co/blog/smollm

5. Hugging Face. **HuggingFaceTB/SmolLM2-135M tokenizer configuration.**  
   https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/tokenizer_config.json
