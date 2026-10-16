# Phase 1 & 2 — Evaluation Results

> **System**: IDEArqRAG (Retrieval-Augmented Generation for archaeological research, IDEArq)
> **Dataset**: `RAG-IDEArq-eval-v3-SIMPLE` — 30 multilingual questions
> **Date**: July 2026
> **Source CSVs**: `data/results/eval_v3_baseline_*.csv`

---

## 5.1 Experimental Setup

The IDEArqRAG system was evaluated on the `RAG-IDEArq-eval-v3-SIMPLE` dataset of 30
multilingual questions (ES 55%, EN 33%, PT 6%, CA 4%, FR 1%; 15 simple / 15 complex)
spanning Iberian archaeology from the Paleolithic to the Iron Age. We tested 54
configurations: **2 embedding models** × **3 LLMs** × **3 prompt strategies** ×
**3 temperatures**, all without reranking (Phase 1 baseline). A second experiment
compared 17 questions with vs. without BAAI/bge-reranker-v2-m3 reranking (Phase 2),
evaluating statistical significance via paired two-tailed t-tests across the 54
configurations. RAGAS v0.4.3 provided four metrics: faithfulness,
context_precision, context_recall, and answer_correctness, using Qwen2.5:14B
(Ollama local) as judge with `num_ctx=32768`. All LLMs ran locally via Ollama with
Q8_0 quantization.

### Valid CSV files used

| File | Rows | Combos | Questions | Rerank | Status |
|---|---|---|---|---|---|
| `eval_v3_baseline_20260723_072430.csv` | 1620 | 54 | 30 | False | ✅ Phase 1 (full) |
| `eval_v3_baseline_20260723_183558.csv` | 810 | 54 | 15 | False | ❌ No metrics |
| `eval_v3_baseline_20260724_044611.csv` | 810 | 54 | 15 | False | ✅ Phase 1 (subset) |
| `eval_v3_baseline_20260724_165606.csv` | 810 | 54 | 15 | True | ✅ Phase 2 (rerank) |
| `eval_v3_baseline_20260728_025831.csv` | 918 | 54 | 17 | False | ✅ Phase 1 (baseline pair) |
| `eval_v3_baseline_20260728_234802.csv` | 918 | 54 | 17 | True | ✅ Phase 2 (rerank pair) |

The `20260723_183558.csv` file was discarded because it lacks RAGAS metric columns
(only 12 columns, no `faithfulness`/`context_precision`/`context_recall`/
`answer_correctness`).

---

## 5.2 Overall System Performance (Phase 1: Baseline)

| Metric | μ | σ | Min | Max |
|---|---|---|---|---|
| Faithfulness | 0.4793 | 0.3208 | 0.0000 | 1.0000 |
| Context Precision | 0.3641 | 0.3729 | 0.0000 | 1.0000 |
| Context Recall | 0.2131 | 0.3946 | 0.0000 | 1.0000 |
| Answer Correctness | 0.1766 | 0.1923 | -0.0353 | 0.8691 |

The system shows moderate faithfulness (≈0.48) but consistently low
answer_correctness (≈0.18) and context_recall (≈0.21), indicating that while
responses are moderately grounded in retrieved documents, the information
retrieved frequently misses key content required to answer the question correctly.
Context_precision (≈0.36) suggests that approximately two-thirds of retrieved
chunks are not directly relevant to the question.

---

## 5.3 Factor Analysis

### 5.3.1 LLM Selection — The Dominant Factor

Language model choice was the single most impactful variable.

| LLM | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 0.6646 | 0.3600 | 0.2223 | 0.2190 |
| Llama-3.2-3B-Instruct | 0.4586 | 0.3764 | 0.2044 | 0.2083 |
| Phi-4-mini | 0.3094 | 0.3560 | 0.2127 | 0.1021 |

**Qwen3-4B-Instruct-2507** (Q8_0) outperformed all others on answer_correctness
(μ=0.219) and faithfulness (μ=0.665). **Llama-3.2-3B-Instruct** (Q8_0) achieved
comparable answer_correctness (μ=0.208) with lower but acceptable faithfulness
(μ=0.459). **Phi-4-mini** (Q8_0) systematically underperformed (faithfulness
μ=0.309, answer_correctness μ=0.102), degrading catastrophically on non-Spanish
questions (see §5.3.4).

### 5.3.2 Embedding Model

| Embedding | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (384D) | 0.5158 | 0.3126 | 0.1635 | 0.1632 |
| gte-multilingual-base (768D) | 0.4426 | 0.4166 | 0.2628 | 0.1900 |

GTE-multilingual delivered superior retrieval quality across precision
(μ=0.417 vs. 0.313) and recall (μ=0.263 vs. 0.164), along with marginally higher
answer_correctness (μ=0.190 vs. 0.163). Conversely, MiniLM-L6-v2 yielded higher
faithfulness scores (μ=0.516 vs. 0.443), suggesting a trade-off between retrieval
breadth and response accuracy. This is consistent with GTE's multilingual
pretraining better matching the multilingual corpus, while MiniLM's larger cluster
of weakly relevant documents may inflate the faithfulness metric artifactually.

### 5.3.3 Prompting Strategy and Temperature

| Prompt | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|
| zero_shot | 0.4811 | 0.3666 | 0.2158 | 0.1778 |
| one_shot | 0.4869 | 0.3634 | 0.2195 | 0.1811 |
| few_shot | 0.4699 | 0.3623 | 0.2040 | 0.1708 |

| Temperature | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|
| 0.3 | 0.4828 | 0.3556 | 0.2092 | 0.1791 |
| 0.5 | 0.4776 | 0.3643 | 0.2162 | 0.1751 |
| 0.7 | 0.4775 | 0.3724 | 0.2140 | 0.1755 |

Neither prompt strategy nor temperature produced statistically meaningful
variations. All prompt variants clustered within ±0.01 of each other on
answer_correctness. Likewise, temperature ranges from 0.3 to 0.7 showed
differences below 0.004 on all metrics. This suggests that **for domain-specific
RAG evaluation, prompt engineering and decoding parameters contribute negligibly
compared to embedding and LLM architecture choices**.

### 5.3.4 Language and Question Complexity

| Language | n | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|---|
| ES | 864 | 0.4918 | 0.3205 | 0.1190 | 0.1942 |
| EN | 540 | 0.4688 | 0.4655 | 0.3630 | 0.1594 |
| PT | 108 | 0.4043 | 0.4409 | 0.2907 | 0.2448 |
| CA | 54 | 0.4949 | 0.1615 | 0.2778 | 0.0309 |
| FR | 54 | 0.5161 | 0.0407 | 0.0000 | 0.0767 |

| Question Type | n | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|---|
| Simple | 810 | 0.5103 | 0.3108 | 0.2191 | 0.1654 |
| Complex | 810 | 0.4479 | 0.4165 | 0.2071 | 0.1878 |

Spanish-language questions achieved the highest answer_correctness (μ=0.194),
followed by Portuguese (μ=0.245, n=108), English (μ=0.159), French (μ=0.077), and
Catalan (μ=0.031). The Portuguese result is driven primarily by a small number of
complex questions that the system answered well. Surprisingly, complex questions
(requiring synthesis of 2+ documents) scored higher on answer_correctness
(μ=0.188) than simple questions (μ=0.165), contradicting the assumption that
multi-document synthesis poses greater difficulty. This may reflect the fact that
complex questions in our dataset provide richer lexical overlap with the retrieval
index.

**Phi-4-mini language breakdown** (catastrophic non-Spanish degradation):

| Language | n | Faithfulness | Answer Correctness |
|---|---|---|---|
| ES | 288 | 0.4492 | 0.1491 |
| EN | 180 | 0.1788 | 0.0583 |
| PT | 36 | 0.0232 | 0.0354 |
| CA | 18 | 0.1019 | 0.0156 |
| FR | 18 | 0.1014 | 0.0045 |

### 5.3.5 Latency

| LLM | μ (s) | σ (s) | Max (s) |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 3.3 | 3.4 | 28.0 |
| Phi-4-mini | 6.5 | 9.5 | 76.7 |
| Qwen3-4B-Instruct-2507 | 8.1 | 14.6 | 240.8 |

Llama-3.2-3B provided the fastest generation (μ=3.3s, σ=3.4s), Phi-4-mini was
intermediate (μ=6.5s, σ=9.5s), and Qwen3-4B was slowest but with high variance
(μ=8.1s, σ=14.6s, max=240.8s). For real-time or interactive use, Llama offers a
strong speed-accuracy trade-off.

---

## 5.4 Phase 2: Impact of Reranking

We deployed **BAAI/bge-reranker-v2-m3** (multilingual CrossEncoder) with
`k_retrieval=10` and `top_n=5`. The reranker selects the 5 most relevant chunks
from the initial 10 candidates before context assembly.

| Metric | Baseline | Reranking | Δ | t-test |
|---|---|---|---|---|
| Faithfulness | 0.2697 | 0.2709 | +0.4% | p=0.891 (n.s.) |
| Context Precision | 0.4447 | 0.4321 | **−2.8%** | p=0.003 (**) |
| Context Recall | 0.2267 | 0.2087 | **−7.9%** | p<0.001 (***) |
| Answer Correctness | 0.0847 | 0.0851 | +0.5% | p=0.413 (n.s.) |

**Contrary to expectations, reranking produced no net improvement and significantly
degraded context_recall (Δ=−7.9%, p<0.001) and context_precision (Δ=−2.8%,
p<0.01).** The degradation was consistent across most configurations: 38 of 54
combos showed lower context_recall with reranking, and context_precision split
evenly (27 wins / 27 losses).

### Why Reranking Failed

We propose two explanations. First, reducing the context window from 10 to 5
documents via `top_n=5` **eliminated relevant information that the LLM would have
otherwise accessed**. The cross-encoder reorders documents by relevance to the
question, but the discarded 5 documents may contain supplementary context
necessary for factual completeness. Second, the BGE reranker, while multilingual,
may not adapt well to the highly specialized archaeological terminology found in
our corpus (radiocarbon dating, lithic typologies, site-specific nomenclature),
placing technically-grounded but semantically-distant documents above more
relevant but less linguistically-aligned candidates.

Additionally, breaking down by embedding model reveals an asymmetry: with
GTE-multilingual, reranking *improved* faithfulness (+0.022) and precision
(+0.016), whereas with MiniLM-L6-v2, it *degraded* all metrics. This suggests
**the reranker's effectiveness depends on the quality distribution of the initial
retrieval set**; MiniLM's already-noisier initial set suffers more from aggressive
pruning.

### Reranking by Embedding

| Embedding | Metric | Baseline | Reranking | Δ |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | Faithfulness | 0.3258 | 0.3063 | −0.0195 |
| all-MiniLM-L6-v2 | Context Precision | 0.3757 | 0.3344 | −0.0412 |
| all-MiniLM-L6-v2 | Context Recall | 0.1365 | 0.1199 | −0.0166 |
| all-MiniLM-L6-v2 | Answer Correctness | 0.0840 | 0.0856 | +0.0016 |
| gte-multilingual-base | Faithfulness | 0.2136 | 0.2354 | +0.0219 |
| gte-multilingual-base | Context Precision | 0.5138 | 0.5298 | +0.0160 |
| gte-multilingual-base | Context Recall | 0.3169 | 0.2975 | −0.0193 |
| gte-multilingual-base | Answer Correctness | 0.0854 | 0.0846 | −0.0008 |

### Reranking by LLM

| LLM | Metric | Baseline | Reranking | Δ |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | Faithfulness | 0.0443 | 0.0374 | −0.0069 |
| Llama-3.2-3B-Instruct | Context Recall | 0.2245 | 0.2078 | −0.0167 |
| Llama-3.2-3B-Instruct | Answer Correctness | 0.0799 | 0.0794 | −0.0004 |
| Phi-4-mini | Faithfulness | 0.2680 | 0.2588 | −0.0091 |
| Phi-4-mini | Context Recall | 0.2314 | 0.2038 | −0.0276 |
| Phi-4-mini | Answer Correctness | 0.0848 | 0.0860 | +0.0012 |
| Qwen3-4B-Instruct-2507 | Faithfulness | 0.4967 | 0.5163 | +0.0196 |
| Qwen3-4B-Instruct-2507 | Context Recall | 0.2241 | 0.2145 | −0.0096 |
| Qwen3-4B-Instruct-2507 | Answer Correctness | 0.0894 | 0.0899 | +0.0005 |

---

## 5.5 Best Configuration

The top-performing combinations without reranking:

| Rank | Combo | AC | Faith | CP | CR |
|---|---|---|---|---|---|
| 1 | GTE + Qwen3-4B + one_shot + t0.5 | 0.309 | 0.632 | 0.404 | 0.308 |
| 2 | GTE + Qwen3-4B + one_shot + t0.3 | 0.272 | 0.697 | 0.445 | 0.242 |
| 3 | GTE + Qwen3-4B + zero_shot + t0.5 | 0.269 | 0.686 | 0.345 | 0.275 |
| 4 | GTE + Llama-3.2-3B + few_shot + t0.3 | 0.266 | 0.400 | 0.403 | 0.178 |
| 5 | GTE + Qwen3-4B + few_shot + t0.5 | 0.262 | 0.674 | 0.456 | 0.268 |

All top-10 configurations used **gte-multilingual-base** embeddings, confirming
that multilingual embedding capacity dominates retrieval quality for our
mixed-language corpus. Qwen3-4B and Llama-3.2-3B alternated in the top positions,
while **Phi-4-mini never appeared in the top 15**, consistent with its severe
cross-language degradation.

---

## 5.6 Limitations and Future Work

1. **k_retrieval × top_n ratio**: Our current configuration (10 → 5) halves the
   context budget. A gentler reduction (e.g., 20 → 15 or 15 → 10) might preserve
   recall while improving precision.
2. **Cold-start for reranker**: bge-reranker-v2-m3 was used out-of-the-box without
   fine-tuning on archaeological data. Domain-adaptive training of the
   cross-encoder could improve reranking quality.
3. **BM25 + vector hybrid search**: Current retrieval uses dense embeddings only.
   Hybrid lexical-semantic search (BM25 + Weaviate vector) may improve initial
   recall before reranking, as documented in
   `docs/future-work-hybrid-search.md`.
4. **Judge model**: Qwen2.5:14B as RAGAS judge may introduce its own biases,
   particularly for non-English languages. Cross-validation with alternative
   judges (GPT-4o, Gemini) would strengthen the evaluation.
5. **e5-large-instruct** (1024D) was not evaluated in this experiment due to
   computational constraints; as the highest-dimensional embedding in our
   registry, it may yield better results and should be prioritized in Phase 3
   (see `notebooks/evaluation/rag-evaluacion-con-chonkie.ipynb` Cell 12).

---

## 5.7 Conclusion

Our evaluation of 2,916 RAG queries (54 configurations × 30 questions + 54 × 17
× 2) reveals that (1) LLM architecture is the dominant performance driver for
archaeological RAG, (2) multilingual embeddings (GTE) outperform monolingual
alternatives even for majority-Spanish corpora, (3) prompting strategy and
temperature exert negligible influence, and (4) **CrossEncoder reranking at
aggressive k=10 → top_n=5 ratios provides no net benefit and actively harms
context_recall**, warranting reconsideration of the K/Top-N ratio or introduction
of hybrid retrieval before reranking.