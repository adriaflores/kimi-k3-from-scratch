# Tokenizer Choice for a Small Kimi K3 Reimplementation

## Purpose

This note records the research used to decide which tokenizer to use for the small, trainable Kimi K3 reimplementation in this repository.

The current training setting is approximately:

- **Pretraining corpus:** FineWeb-Edu
- **Training budget:** roughly 6B tokens, subject to tokenizer-dependent measurement
- **Model scale:** a few hundred million active parameters
- **Goal:** faithfully study Kimi K3's architectural contributions (KDA, Gated MLA, MoE, etc.) while making the model practical to train on a limited compute budget
- **Constraint:** the tokenizer should not consume an unnecessarily large fraction of the parameter or compute budget

The key question is therefore not simply *"Which tokenizer compresses text the most?"* or *"Which tokenizer has the smallest vocabulary?"*. The relevant question is:

> Which tokenizer gives a strong, well-tested trade-off between vocabulary cost, sequence length, data efficiency, and engineering risk at our scale?

---

## 1. Schmidt et al. (2024) — *Tokenization Is More Than Compression*

**Reference**

Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, and Chris Tanner.  
*Tokenization Is More Than Compression.* EMNLP 2024.

- Paper: https://aclanthology.org/2024.emnlp-main.40/
- DOI: https://doi.org/10.18653/v1/2024.emnlp-main.40

### What they studied

The paper investigates the common assumption that a tokenizer is better when it represents text using fewer tokens.

The authors introduce **PathPiece**, which explicitly searches for segmentations that minimize the number of tokens for a given vocabulary. They then compare tokenizer design choices across:

1. pre-tokenization,
2. vocabulary construction,
3. segmentation/inference.

They train **64 language models ranging from 350M to 2.4B parameters**.

### Main conclusions

The central result is that:

> Better compression does not automatically produce a better language model.

A tokenizer that represents a document with fewer tokens can still lead to worse downstream performance.

The study also finds that choices such as **pre-tokenization and vocabulary construction matter materially**, and reports benefits from using BPE during vocabulary construction.

### Consequence for our project

This weakens the argument:

> "We should train our own 32K BPE because it produces fewer embedding parameters."

A 32K vocabulary can reduce the embedding/output-head parameter count, but that alone does **not** establish that it is the best tokenizer for the model.

It also means that evaluating candidate tokenizers purely by compression ratio would be insufficient.

---

## 2. Tao et al. (2024) — *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*

**Reference**

Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, and Ngai Wong.  
*Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* 2024.

- Paper: https://arxiv.org/abs/2407.13623
- Code: https://github.com/sail-sg/scaling-with-vocab

### What they studied

The authors explicitly add **vocabulary size** to language-model scaling-law analysis.

Their experiments cover:

- models from **33M to 3B parameters**,
- multiple vocabulary sizes,
- up to **500B characters** of training data,
- fixed-compute comparisons.

They propose several methods for estimating a compute-optimal vocabulary size.

### Main conclusions

The optimal vocabulary size depends on the model and compute budget.

In particular:

- vocabulary size should be treated as a scaling variable rather than a fixed implementation detail;
- larger models generally justify larger vocabularies;
- some existing LLMs may use vocabularies that are smaller than compute-optimal;
- increasing vocabulary can improve performance even though it increases embedding and output-head costs.

One reported example increases a 3B model's vocabulary from **32K to 43K** under the same FLOP budget and improves ARC-Challenge performance.

### Consequence for our project

This directly contradicts the simplistic idea that:

> smaller vocabulary = more efficient model.

A smaller vocabulary reduces lexical parameters, but it can also produce longer sequences and require more transformer computation to process the same amount of raw text.

Therefore, **32K is not an obviously optimal choice just because our model is small**.

For a model in the hundreds-of-millions parameter range, a vocabulary around **40K–50K is entirely plausible** and should not be rejected purely because it costs more embedding parameters.

---

## 3. Limisiewicz et al. (2026) — *Compute Optimal Tokenization*

**Reference**

Tomasz Limisiewicz, Artidoro Pagnoni, Srini Iyer, Mike Lewis, Sachin Mehta, Alisa Liu, Margaret Li, Gargi Ghosh, and Luke Zettlemoyer.  
*Compute Optimal Tokenization.* 2026.

- Paper: https://arxiv.org/abs/2605.01188
- Project page: https://co-tok.github.io/

### What they studied

This work asks how the **compression rate of a tokenizer**, measured in average bytes per token, interacts with compute-optimal language-model scaling.

The study trains **988 models ranging from 50M to 7B parameters** while varying token compression.

This is particularly relevant because conventional scaling laws normally assume a fixed tokenizer.

### Main conclusions

Three findings are especially important.

#### 1. Raw bytes are a more stable data unit than tokens

When tokenizer compression changes, the compute-optimal relationship is approximately stable in terms of:

> **bytes of training data per model parameter**

rather than:

> tokens per model parameter.

This matters because "6B tokens" does not describe the same amount of underlying text for two different tokenizers.

For example, if one tokenizer averages 3.5 bytes/token and another 4.5 bytes/token, then 6B tokens represent very different quantities of raw text.

#### 2. There is a compute-dependent optimal compression rate

More compression is not always better.

At a given compute budget there is a compression range that minimizes loss, and this optimum changes with scale.

#### 3. Popular BPE tokenizers are not necessarily compression-optimal

The optimal compression rate found experimentally does not simply coincide with the compression rate of existing BPE tokenizers.

### Consequence for our project

We should stop treating **"6B tokens" as a tokenizer-independent training budget**.

Once the tokenizer is selected, we should record at least:

- total tokens,
- total raw bytes represented,
- average bytes per token.

For reproducibility, **raw bytes processed** is a useful tokenizer-independent quantity to keep in the dataset manifest.

This paper also reinforces that choosing a tokenizer solely because it minimizes sequence length or vocabulary size is not theoretically justified.

---

## 4. Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models*

**Reference**

Jordan Hoffmann et al.  
*Training Compute-Optimal Large Language Models.* 2022.

- Paper: https://arxiv.org/abs/2203.15556

### What they studied

The Chinchilla work studies the relationship between:

- model parameter count,
- training-data quantity,
- total compute.

The authors train more than 400 models from roughly **70M to 16B parameters**, using between **5B and 500B tokens**.

### Main conclusions

For compute-optimal dense language-model training, model size and training-data quantity should scale together much more aggressively than earlier large-model recipes assumed.

The well-known approximate heuristic derived from this work is on the order of **20 training tokens per parameter**, although this should not be treated as a universal constant.

### Consequence for our project

This work influenced the broader model-size decision: with a training budget around 6B tokens, blindly building a 1B+ active-parameter model would be difficult to justify from a compute-efficiency perspective.

However, the later *Compute Optimal Tokenization* work adds an important qualification:

> Chinchilla-style token counts implicitly assume a fixed tokenizer.

Therefore we should use Chinchilla as a **rough model-scale prior**, not as a reason to optimize a tokenizer around exactly 6B arbitrary token units.

For Kimi K3's MoE architecture, we must also distinguish between **total parameters** and **parameters activated per token**.

---

## 5. Penedo et al. (2024) — *The FineWeb Datasets*

**Reference**

Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, and Thomas Wolf.  
*The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* 2024.

- Paper: https://arxiv.org/abs/2406.17557
- Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

### What they studied

The work introduces FineWeb and FineWeb-Edu.

FineWeb-Edu is a **1.3T-token educational subset of FineWeb**, produced by filtering web documents for educational quality.

The paper reports that models pretrained on FineWeb-Edu substantially improve on knowledge- and reasoning-heavy benchmarks such as MMLU and ARC compared with less targeted web corpora.

### Main conclusions relevant here

FineWeb-Edu is a serious language-model pretraining corpus rather than a toy dataset, and its distribution is sufficiently specific that tokenizer/domain compatibility is worth considering.

### Consequence for our project

Because FineWeb-Edu will dominate our pretraining data, a tokenizer already exercised on similar modern web/educational data has a practical advantage over an older tokenizer such as GPT-2's tokenizer.

This is not proof that such a tokenizer is mathematically optimal, but it reduces distribution mismatch and engineering uncertainty.

---

## 6. Ben Allal et al. (2025) — *SmolLM2: When Smol Goes Big*

**Reference**

Loubna Ben Allal et al.  
*SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model.* 2025.

- Paper: https://arxiv.org/abs/2502.02737
- Model: https://huggingface.co/HuggingFaceTB/SmolLM2-135M

### What they studied

SmolLM2 is an unusually relevant reference because it studies **small language models** rather than extrapolating only from multi-billion-parameter systems.

The project trains models at small scale using large amounts of curated web, math, code, and specialized data, and documents extensive data-mixture ablations.

The released tokenizer uses a vocabulary of **49,152 tokens**.

### Why it matters for our project

The scale is much closer to our intended mini-Kimi K3 than the original Kimi K3 tokenizer.

The public SmolLM documentation states that its tokenizer was trained on the **SmolLM Corpus** with a vocabulary size of 49,152.

The SmolLM model-training mixture is documented as using FineWeb-Edu-dedup as its dominant component (70% in the published SmolLM1 training configuration), alongside Cosmopedia, OpenWebMath, code, and StackOverflow data.

### Important qualification

The sources reviewed do **not** establish that the tokenizer-training sample itself consisted of exactly 70% FineWeb-Edu.

Therefore the defensible statement is:

> The SmolLM tokenizer was trained on the SmolLM Corpus, a corpus ecosystem strongly centered on FineWeb-Edu and related modern pretraining data.

It would be too strong to claim:

> "The tokenizer itself was trained on exactly 70% FineWeb-Edu."

### Consequence for our project

SmolLM's tokenizer gives us an already-tested **49K-class BPE/GPT-2-style tokenizer** designed in the context of small modern language models and a data ecosystem close to ours.

That makes it a lower-risk default than training an unvalidated custom 32K tokenizer.

---

## 7. Ali et al. (2024) — *Tokenizer Choice For LLM Training: Negligible or Crucial?*

**Reference**

Mehdi Ali et al.  
*Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL 2024.

- Paper: https://aclanthology.org/2024.findings-naacl.247/
- DOI: https://doi.org/10.18653/v1/2024.findings-naacl.247

### What they studied

The authors train **24 mono- and multilingual 2.6B-parameter language models** while varying tokenizer algorithms and parameterizations.

This makes the paper especially useful for separating two questions that are often conflated:

1. whether an intrinsic tokenizer metric looks good, and
2. whether that tokenizer actually improves the trained language model.

### Main conclusions

Tokenizer choice can materially affect both **downstream performance and training cost**. However, common intrinsic metrics such as **fertility** and **parity** are not always reliable predictors of downstream model quality.

The strongest mismatch result is multilingual: using an English-centric tokenizer for multilingual training can create severe inefficiency, with the paper reporting **up to 68% additional training cost** in the studied setting.

### Important qualification

That 68% figure should **not** be transferred directly to our mostly English FineWeb-Edu setup. It is evidence that tokenizer/data mismatch can be expensive, not an estimate of the gain we would get from a custom tokenizer.

### Consequence for our project

A tokenizer benchmark on our corpus is useful as a **sanity check**, but metrics such as bytes/token or fertility should not be treated as proof that one tokenizer will train a better model.

This supports reusing a tokenizer whose training ecosystem is already close to FineWeb-Edu unless profiling reveals a concrete mismatch.

---

## 8. Dagan et al. (2024) — *Getting the most out of your tokenizer for pre-training and domain adaptation*

**Reference**

Gautier Dagan, Gabriel Synnaeve, and Baptiste Rozière.  
*Getting the most out of your tokenizer for pre-training and domain adaptation.* 2024.

- Paper: https://arxiv.org/abs/2402.01035

### What they studied

The paper studies how tokenizer **vocabulary size, pre-tokenization rules, and tokenizer-training data** affect language-model efficiency and quality, with extensive experiments on code models and domain adaptation.

### Main conclusions

Tokenizer design can change:

- generation speed,
- effective context length,
- memory usage,
- and downstream performance.

The paper also shows that a domain-specialized tokenizer can be worthwhile when the target distribution is sufficiently different and enough continued-training data is available.

### Important qualification

The strongest experiments are in **code**, where domain vocabulary differs substantially from ordinary web text. Therefore this paper supports the principle of domain matching, but it does not imply that a custom tokenizer will beat SmolLM2 on FineWeb-Edu.

### Consequence for our project

The right question is not *"custom or reused?"* in the abstract. It is:

> Is our target corpus sufficiently different from the corpus behind the reused tokenizer to justify adding tokenizer training and validation as another research variable?

For FineWeb-Edu versus SmolLM's data ecosystem, the expected mismatch is relatively small. If we later move heavily toward code, mathematics, another language, or a specialized scientific domain, this decision should be revisited.

---

## 9. Takase et al. (2025) — *Large Vocabulary Size Improves Large Language Models*

**Reference**

Sho Takase, Ryokan Ri, Shun Kiyono, and Takuya Kato.  
*Large Vocabulary Size Improves Large Language Models.* Findings of ACL 2025.

- Paper: https://aclanthology.org/2025.findings-acl.57/
- DOI: https://doi.org/10.18653/v1/2025.findings-acl.57

### What they studied

The authors investigate how **subword vocabulary size** affects language-model performance and also study replacing the vocabulary when continually training a model on a different target language.

### Main conclusions

Their experiments find that larger vocabularies can improve LLM performance, reinforcing the idea that vocabulary size should not be minimized solely to save embedding parameters.

The paper also notes that vocabulary sizes around **30K–60K are common for monolingual LLMs**, placing SmolLM2's 49,152-token vocabulary in an entirely conventional range.

### Consequence for our project

This provides independent support for not treating 32K as inherently superior to 49K. The additional lexical parameters of SmolLM2's tokenizer must be weighed against shorter sequences and potentially better lexical coverage.

Combined with Tao et al. (2024), it strengthens the case that **49,152 is a defensible engineering choice**, even though it is not proven compute-optimal for our exact architecture.

---

# Synthesis

The research changes the decision from the naive formulation:

> "Use a custom 32K tokenizer because it saves embedding parameters."

to:

> "Treat vocabulary size and tokenization as part of the compute/data trade-off, and avoid introducing a custom tokenizer unless we validate that it is actually advantageous."

The important lessons are:

| Research result | Implication |
|---|---|
| Compression alone does not predict LM quality | Do not select a tokenizer only by tokens/byte |
| Vocabulary size participates in scaling laws | 32K is not automatically optimal for a small model |
| Larger vocabularies can reduce sequence-side compute | Embedding cost must be balanced against transformer compute |
| Bytes are more tokenizer-independent than token counts | Track raw bytes as well as tokens |
| Tokenizer construction details matter | A custom tokenizer needs validation, not just a chosen vocab size |
| SmolLM provides a tested ~49K tokenizer at small-model scale | It is a strong low-risk default for our project |
| FineWeb-Edu dominates our data distribution | Compatibility with modern web/educational text is valuable |
| Tokenizer/data mismatch can materially increase training cost | Reuse is safest when the tokenizer was trained on a similar distribution |
| Fertility/compression are imperfect quality proxies | Use intrinsic profiling as a sanity check, not as final evidence |
| 30K–60K vocabularies remain conventional for monolingual LMs | SmolLM2's 49,152-token vocabulary is not unusually large |

---

# Current Recommendation

For the first serious training run of the mini-Kimi K3 implementation, the most defensible default is:

```text
Tokenizer: HuggingFaceTB/SmolLM2-135M tokenizer
Vocabulary size: 49,152
Storage dtype: uint16
```

This recommendation is **not** based on the claim that 49,152 is theoretically optimal for our exact model.

It is based on risk-adjusted engineering evidence:

1. the vocabulary size is in a range supported by modern scaling experiments;
2. the tokenizer comes from a small-LM project, much closer to our scale than Kimi K3 itself;
3. its training ecosystem is closely related to FineWeb-Edu;
4. it avoids spending project scope on tokenizer research;
5. it still fits inside `uint16`;
6. it avoids the extreme lexical parameter cost of Kimi K3's 163,840-token vocabulary.

At `d_model = 768`, a 49,152-token embedding matrix contains:

```text
49,152 × 768 = 37,748,736 parameters
```

If input embeddings and the output LM head are tied, that lexical parameter cost remains approximately **37.7M parameters**.

If they are untied, it becomes approximately **75.5M parameters**.

Whether to tie those weights should be decided separately from tokenizer selection.

---

# What Would Justify a Custom Tokenizer?

A custom tokenizer becomes worthwhile if we are willing to validate it experimentally.

Before tokenizing the full training corpus, a cheap tokenizer benchmark could compare:

```text
SmolLM2 49K
GPT-2 50K
Custom BPE 32K
Custom BPE ~40K
Custom BPE ~49K
```

on a held-out FineWeb-Edu sample.

At minimum, we should measure:

- average bytes/token,
- average characters/token,
- average tokens/document,
- vocabulary utilization,
- tokenization throughput,
- frequency distribution of token IDs,
- behavior on code, math, punctuation, URLs, numbers, and non-ASCII text.

However, intrinsic tokenizer metrics would still not prove downstream superiority. Ali et al. (2024) specifically caution that common metrics such as fertility are not always predictive of downstream performance.

A scientifically stronger decision would require training small proxy language models under the **same FLOP or byte budget** and comparing validation loss/downstream performance.

Unless we want tokenization itself to become part of the research contribution, this extra experimental branch is probably not worth the complexity for the first Kimi K3 training run.

---

# Final Decision Rule

For this repository:

> **Use SmolLM2's 49,152-token tokenizer as the default pretraining tokenizer. Do not train a custom tokenizer merely to reduce vocabulary size.**

Revisit this choice only if:

- tokenizer profiling on our FineWeb-Edu sample reveals a concrete weakness,
- the lexical matrices become an unacceptable fraction of the final parameter budget,
- or tokenizer design becomes an explicit research question of the project.

The dataset preparation manifest should record the exact tokenizer revision and tokenizer-independent corpus statistics so that the preprocessing run is reproducible.

Suggested manifest fields:

```json
{
  "tokenizer": "HuggingFaceTB/SmolLM2-135M",
  "tokenizer_revision": "<commit-hash>",
  "vocab_size": 49152,
  "token_dtype": "uint16",
  "num_tokens": 6000000000,
  "num_raw_bytes": "<measured>",
  "bytes_per_token": "<measured>"
}
```

The `num_tokens` target should be treated as provisional until the tokenizer is frozen and its compression rate on FineWeb-Edu has been measured.
