# AGENTS.md — RAG-IDEArq

## Project
RAG system for archaeological research (IDEArq — http://www.idearqueologia.org).
Uses small LLMs (Phi-4-mini, Qwen3-4B, Llama-3.2-3B) + embeddings
(MiniLM-L6-v2, GTE-multilingual-base, E5-large-instruct). Evaluated with RAGAS
using qwen2.5:14b (Ollama local) as judge, monitored via Langfuse (:4000).

## Commands
```bash
source ../env_rag/bin/activate
pip install -r requirements.txt

# 1. Index (creates new collections per embedding)
jupyter notebook notebooks/indexing/rag-indexacion-mejorado.ipynb

# 2. Index GeoJSON (adds yacimientos to existing collections)
python src/index_geojson.py

# 3. Evaluate (uses Langfuse dataset + prompts)
jupyter notebook notebooks/evaluation/rag-evaluacion-con-chonkie.ipynb

# 4. Weaviate (Langfuse assumed running on :4000)
docker compose -f deploy/docker/docker-compose.yml up -d
```

## Architecture
- `src/config.py` — chunking params (800/50) + embedding registry + reranking config
- `src/langfuse_monitor.py` — connects to Langfuse :4000 (CallbackHandler + REST API)
- `src/reranker.py` — BAAI/bge-reranker-v2-m3 (multilingual, disabled by default)
- `src/index_geojson.py` — indexes c14_v2.geojson yacimientos into existing collections
- `data/eval_questions.py` — 30 eval questions (15 simple + 15 complex, multilingual)
- `notebooks/indexing/` — indexing pipelines (recursive, parametrizable)
- `notebooks/evaluation/rag-evaluacion-con-chonkie.ipynb` — main evaluation notebook
- `notebooks/graph/` — bibliographic graph via Morph-KGC (BIBO ontology → RDF)
- `docs/future-work-hybrid-search.md` — justification for hybrid BM25+vector search
- Collections: `IdearqMiniLM_800_50_v2`, `IdearqGTE_800_50_v2`, `Idearqe5largeinstruct_800_50_v2`

## Models
- **LLMs (Ollama)**: Phi-4-mini (Q8_0), Qwen3-4B-Instruct-2507 (Q8_0), Llama-3.2-3B-Instruct (Q8_0)
- **Temperatures**: 0.3, 0.5, 0.7
- **Embeddings**: all-MiniLM-L6-v2 (384D), gte-multilingual-base (768D, trust_remote_code), e5-large-instruct (1024D, custom E5InstructEmbeddings class)
- **RAGAS judge**: qwen2.5:14b (Ollama local, format="json", num_ctx=32768)
- **Reranker**: BAAI/bge-reranker-v2-m3 (disabled by default, enabled in phase 2)

## Dataset
- **Langfuse dataset**: `RAG-IDEArq-eval-v3-SIMPLE`
- 30 questions (15 simple + 15 complex)
- Distribution: ES 55% (16), EN 33% (10), PT 6% (2), CA 4% (1), FR 1% (1)
- Sources: PDF filenames (fill manually in Langfuse UI)

## Prompts
- **Loaded from Langfuse**: `prompt_zero_shot`, `prompt_one_shot`, `prompt_few_shot`
- Read via `langfuse_client.get_prompt(name=..., type="chat")`
- Do NOT modify prompts in code — edit in Langfuse UI

## Evaluation Grid
- 3 LLMs × 3 embeddings × 3 prompts × 3 temperatures = 81 combos
- 81 × 30 questions = 2,430 evaluations per phase
- **Phase 1 (current)**: No reranking, k=10, 10 docs to context
- **Phase 2 (future)**: With reranking (BAAI/bge-reranker-v2-m3), k=10, top_n=5
- **Phase 3 (future)**: Hybrid search (BM25 + vector), see `docs/future-work-hybrid-search.md`

## Langfuse Integration
- **Pattern**: `start_as_current_observation()` → `get_current_trace_id()` → `CallbackHandler` → `create_score()` → `flush()`
- **Per question**: trace with input (question), output (full answer), retrieval_breakdown event (10 docs with filenames + metadata), 4 RAGAS scores
- **Per combo**: `flush()` + `sleep(10)` after RAGAS evaluation
- **CSV output**: `data/results/eval_v3_baseline_<timestamp>.csv`

## Weaviate Collections
| Collection | Embedding | PDFs | Yacimientos | Total |
|---|---|---|---|---|
| `IdearqMiniLM_800_50_v2` | all-MiniLM-L6-v2 (384D) | 53,758 | 4,478 | 58,236 |
| `IdearqGTE_800_50_v2` | gte-multilingual-base (768D) | 53,758 | 4,478 | 58,236 |
| `Idearqe5largeinstruct_800_50_v2` | e5-large-instruct (1024D) | 53,758 | 4,478 | 58,236 |

## Critical Gotchas
1. **Langfuse must be running on :4000** before evaluation
2. **Prompts are READ from Langfuse** — do not modify in code
3. **Dataset is `RAG-IDEArq-eval-v3-SIMPLE`** in Langfuse (not local file)
4. **GTE requires `trust_remote_code=True`** when loading embeddings
5. **E5 uses custom class `E5InstructEmbeddings`** with passage/query prefixes
6. **Phi-4-mini (Q8_0) may generate garbage** — use Llama-3.2-3B as default
7. **Qwen3-4B may generate repetition loops** with long context — limit to 10 docs
8. **k=10, 10 docs to context** (no reranking in phase 1)
9. **Chunking 800/50** — change in `src/config.py` to compare (creates new collection)
10. **Do NOT delete existing Weaviate collections** — new ones use `_v2` suffix
11. **All notebooks preserved** — do not delete any without confirmation
12. **morph-kgc downgrades rdflib** from 7.6.0 → 7.2.1 (compatible, no issues observed)
13. **RAGAS v0.4.3**: use `evaluate(dataset=...)` not `eval_ds.evaluate()`, scores in `result.scores[0]`
14. **Langfuse v4.x**: use `from langfuse.langchain import CallbackHandler` (not `from langfuse import`)
