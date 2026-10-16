# AGENTS.md — RAG-IDEArq

## Project
RAG system for archaeological research (IDEArq — http://www.idearqueologia.org).
Uses small LLMs (Phi-4-mini, Qwen3-4B, Llama-3.2-3B) + embeddings
(MiniLM-L6-v2, GTE-multilingual-base, E5-large-instruct). Evaluated with RAGAS
using **Qwen3-14B Q6_K** (Ollama local) as judge.

**Langfuse is NOT running** — do not start that stack. Prompts are frozen in
`data/prompts.json` and loaded locally (`PROMPT_SOURCE=local`, the default).

Canonical dataset: **`data/eval_questions.py`** — 32 questions (17 simple +
15 complex). The old Langfuse dataset (`RAG-IDEArq-eval-v3-SIMPLE`, 30 q) is
superseded.

## Quick start (cluster: CESGA Slurm)

```bash
# 0. Environment (Lustre = execution tree)
source /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/env_rag/bin/activate

# 1. Persistent services (Weaviate :8080 + Ollama :11434) — start ONCE, 7-day job
sbatch /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/rag-services.sbatch
#    → writes services.env (node, URLs, PIDs). Verify:
#    curl -s $(grep WEAVIATE_URL .../services.env | cut -d'"' -f2)/v1/.well-known/ready
#    → also AUTO-SUBMITS the smoke (start_rag.sh §11) once services are
#      ready. Opt out: export RAG_NO_AUTO_SMOKE=1 before starting.
#      Never run two smoke-notebook jobs at once (guarded by squeue check).
#    → if the job pends with (AssocGrpGRES) the account GPU quota is full
#      (csic: gres/gpu=32); start the CPU variant instead — same services,
#      Ollama falls back to CPU inference:
#      sbatch /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/rag-services-cpu.sbatch
#      (a queued GPU job keeps waiting for the old Weaviate to die ⇒ handoff)

# 2. Smoke the evaluation notebook (3 questions, SMOKE_ONLY=True in cell 1)
#    Usually automatic (see above); run manually only as a fallback/retry:
sbatch /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/smoke-notebook.sbatch
#    NOTE: it executes the STORE notebook — src/ exists only in Store, so the
#    Lustre copy dies with `ModuleNotFoundError: No module named 'src'`.
#    NOTE: never use --dependency=parallel:<jobid> here — CESGA's Slurm
#    rejects it ("Job dependency problem"); afterok/afterany work.

# 3. Evaluation phases (run_fase.py wrapper; NEVER starts services)
FASE=A SMOKE=1 RESUME=1 sbatch scripts/eval-fase.sbatch

# Indexing / notebooks (manual)
jupyter notebook notebooks/indexing/rag-indexacion-mejorado.ipynb
python src/index_geojson.py
```

- `scripts/run_fase.py --fase {A,B,C,D,G} [--smoke] [--resume] [--workers N]
  [--base-strategy vector|hybrid_0.5] [--graph|--no-graph] [--kill-residual-weaviate]`
- CSV output: `data/results/fase{X}_{smoke|full}_{ts}.csv` (30-column schema,
  see `docs/eval-strategy.md` §8).
- Full phase design, grids, estimations: **`docs/eval-strategy.md`**.

## Architecture
- `src/config.py` — chunking (800/50), embedding registry, reranking config,
  `RAGAS_JUDGE` (Qwen3-14B Q6_K), model registry
- `src/langfuse_monitor.py` — Langfuse callback code (stack currently OFF;
  safe to import only when Langfuse runs)
- `src/prompts_loader.py` — frozen prompts from `data/prompts.json` (single
  source of truth for the three prompts)
- `src/weaviate_conn.py` — host-agnostic Weaviate connection with fallback
- `src/reranker.py` — BAAI/bge-reranker-v2-m3 (disabled by default)
- `src/index_geojson.py` — indexes `c14_v2.geojson` yacimientos
- `data/eval_questions.py` — **32 canonical eval questions** (v4). Every item
  is a 7-tuple: `(id, question, ground_truth, fuente, tipo, idioma, n_articulos)`
- `scripts/run_fase.py` — phase runner (health guard, CSV, token capture)
- `scripts/eval-fase.sbatch` — Slurm template for one phase (no services)
- `smoke-notebook.sbatch` (Lustre) / `scripts/smoke-notebook.sbatch` (Store) —
  executes the eval notebook headless via `jupyter nbconvert --execute`
- `rag-services.sbatch` + `start_rag.sh` (Lustre) — persistent Weaviate+Ollama
- `notebooks/evaluation/rag-evaluacion-hibrida-multiquery.ipynb` — **main**
  evaluation notebook (26 cells; cell 17 = graph smoke, `SMOKE_ONLY=True`)
- `notebooks/graph/` — bibliographic graph via Morph-KGC (BIBO → RDF)
- `docs/future-work-hybrid-search.md` — hybrid BM25+vector justification

## Models
- **LLMs (Ollama)**: Phi-4-mini (Q8_0), Qwen3-4B-Instruct-2507 (Q8_0),
  Llama-3.2-3B-Instruct (Q8_0); temps 0.3 / 0.5 / 0.7
- **Phase D large models**: Qwen3-14B Q6_K, Qwen3-30B-A3B Q8_0 (GPU-bound)
- **Embeddings**: all-MiniLM-L6-v2 (384D), gte-multilingual-base (768D,
  `trust_remote_code=True`), e5-large-instruct (1024D, custom
  `E5InstructEmbeddings` class with passage/query prefixes)
- **RAGAS judge**: `hf.co/unsloth/Qwen3-14B-GGUF:Q6_K` (`format="json"`)
- **Multiquery generator**: Qwen3-4B-Instruct-2507 (`MQ_GENERATOR_*`)
- **Reranker**: BAAI/bge-reranker-v2-m3 (off by default; `k=10`, `top_n=5`)

## Dataset
- `data/eval_questions.py` — 32 items, v4 (27/07/26)
- Split: 17 simple + 15 complex; ES 18, EN 6, PT 4, CA 2, FR 2
- Accessors: `get_eval_items()`, `get_metadata()`, `get_questions()`
- Sources: PDF filenames or `geojson` / `varias entradas del geojson`

## Prompts
- Frozen in `data/prompts.json`: `prompt_zero_shot`, `prompt_one_shot`,
  `prompt_few_shot` — loaded via `src/prompts_loader.py`
- To change a prompt: edit `data/prompts.json` (then the notebooks pick it up)
- `PROMPT_SOURCE=langfuse` exists but the stack is off; default `local`

## Evaluation (phased ablation)
Sequential/fractional ablation over the 32-question dataset, ~50–60 combos
total instead of the 486-combo full factorial. Details: `docs/eval-strategy.md`.

| Phase | What varies | Combos |
|-------|-------------|--------|
| A | 3 embeddings × 6 retrieval strategies (LLM/prompt/temp frozen) | 18 |
| B | 3 LLM × 3 prompts × 3 temps (A winner frozen) | 27 |
| C | rerank on/off over A/B finalists | 4 |
| D | large models (14B/30B) on B finalists — **2×A100 candidate** | ≈4 |
| E | multiquery generator ablation (optional, skip if budget tight) | — |
| G | `{vector, hybrid_0.5} × {graph off, on}` — **context-only graph** | 4 |

- **No `graph_only` strategy exists** — the graph only appends RDF docs to the
  retrieval context (`data/biblio-graph/idearq-graph-instances-dedup-v2.nt`).
- Smoke mode (`--smoke` / `SMOKE_ONLY=True`): 3 questions, minutes per phase.

## Slurm / CESGA rules (IMPORTANT)
1. **`--mem` is mandatory** (else accounting policy rejects the job).
2. **Never pass `--qos` by hand** — `JobSubmitPlugins=require_timelimit,lua`
   derives QOS/partition from `--time` (short≤6h, medium≤3d, long≤7d). An
   explicit `--qos` fails with *Invalid qos specification*.
3. **GPU rule**: CPUs must equal `nodes × gpus × 32`. With 1 GPU:
   `--gres=gpu:a100:1` + `--cpus-per-task=32`. (8 CPUs + GPU ⇒ submission
   error *CPU count specification invalid*.)
4. GPU request syntax: `#SBATCH --gres=gpu:a100:1`. Partitions `short`,
   `medium`, `long` all have A100 nodes.
5. Monitor: `squeue -u $USER` (job disappears ⇒ finished), results via
   `sacct -j <id>`, logs under `$LUSTRE/idearq-rag/logs/`.
6. Resource baseline used: services = 32 CPU + 1 GPU + 32G (GPU rule),
   smoke = 4 CPU + 8G (no GPU — it only orchestrates over HTTP).
7. **Account quotas (this is what really blocks)**: account `csic` carries
   `GrpTRES=cpu=6144,gres/gpu=32` and `csusi` inherits it. Saturated ⇒ jobs
   pend with `(AssocGrpGRES)` (GPU) or `(AssocGrpCpuLimit)` (CPU) even while
   A100 nodes sit idle — quota, NOT hardware. Holders are invisible
   (`PrivateData`): `squeue -a` / `sacct -a` show only your own jobs; use
   `sreport cluster AccountUtilizationByUser` for per-account totals.
8. `module load cuda` (12.8.0) works on CPU-only nodes; a job without
   `--gres` just runs Ollama on CPU (slower judge, same results).
9. Account `csic_viz` accepts only `qos=viz` ⇒ only partition `viz`
   (T4 nodes, MaxTime 8h). It has no own GrpTRES, but it is a separate
   allocation — do not route RAG eval jobs there by default.

## Workspace / Storage Convention (IMPORTANT)
Two copies with different roles:

- **STORE (persistence)**: `/mnt/netapp1/Store_CSIC/home/csic/usi/eag/idearq-rag`
  — finished work: notebooks, data, results, docs, `scripts/`.
- **LUSTRE (execution)**: `/mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag`
  — what runs: `env_rag/`, `ollama/`, `rag-services.sbatch`,
  `smoke-notebook.sbatch`, `start_*.sh`, `services.env`, logs.

Rule: **store to save, lustre to run.** Persist artifacts in Store; copy to
Lustre only what is about to execute. Lustre holds no `notebooks/` folder.

## Critical Gotchas
1. **Exactly one Weaviate ever** — two instances on the same `vectordb`
   corrupt it. Phase jobs must NOT call `start_rag.sh`; only
   `rag-services.sbatch` starts services.
2. **`services.env` is the source of truth** for node/URLs/PIDs. If services
   restart on another node, consumers must re-read it (a smoke job that read
   it before the rewrite waits on a dead host and times out).
3. **Residual Weaviate remediation order**: `scancel` the owning job →
   `pkill -f weaviate` only on the services node with a stale
   `WEAVIATE_PID` → recheck ready endpoint. Never guess.
4. **Dataset tuples are 7 fields** — a missing comma glues the next string
   into the previous one (adjacent literal concatenation) and
   `get_eval_items()` throws `IndexError`. Item 19 had exactly this bug.
5. **GTE** requires `trust_remote_code=True`; **E5** uses the custom
   `E5InstructEmbeddings` class with prefixes.
6. **Phi-4-mini (Q8_0) may generate garbage** — prefer Llama-3.2-3B as
   fallback; Qwen3-4B can loop with long context — keep 10 docs.
7. **Chunking 800/50** — changing it requires a new collection
   (`src/config.py`); collections use the `_v2` suffix.
8. **Do NOT delete Weaviate collections** (7 live: MiniLM/GTE/E5 `_v2` plus
   `BgeE5`, `BgeE5Chunk1000Sim05`, `IdearqLinq`, `IdearqQwen3`).
9. **All notebooks preserved** — do not delete any without confirmation.
10. **morph-kgc downgrades rdflib** 7.6.0 → 7.2.1 (compatible; `rdflib>=7`
    is declared in `requirements.txt` and installed in `env_rag`).
11. **RAGAS v0.4.3**: `evaluate(dataset=...)`, scores in `result.scores[0]`.
12. **Langfuse v4.x imports**: `from langfuse.langchain import CallbackHandler`
    (not `from langfuse import`).
13. **Jupyter kernels** may hold a stale `OLLAMA_HOST` — restart the kernel
    (user action) after `services.env` changes.
14. **`run_fase.py` needs `env_rag` python** (system python lacks `dotenv`).
15. **Smoke must execute the STORE notebook** — `src/` exists only in Store;
    cell 1 derives `PROJECT_ROOT` from the kernel cwd, so the stale Lustre
    copy fails with `ModuleNotFoundError: No module named 'src'`.
    `smoke-notebook.sbatch` points at `$STORE/.../rag-evaluacion-hibrida-multiquery.ipynb`.

## SDD (Spec-Driven Development)
Gentle AI SDD is available for this project. Artifact store: **openspec**
(under this store copy).

- `openspec/config.yaml` — SDD config: project context, phase rules, strict
  TDD flag, testing capability.
- `.atl/skill-registry.md` — index of available skills by trigger and path.
- SDD phases: `init → explore → research → proposal → spec → design → tasks
  → apply → verify → sync → archive`.
- Launch with `/gentle-sdd-init` or `/gentle-sdd-continue`, or a
  natural-language request.
- **Testing capability:** no automated test runner yet (only
  `lda/scripts/smoke_test.py`). Strict TDD is disabled until a runner exists;
  verification is notebook- or manual-based.
- This store copy is **not a git repository**; delivery/PR policy does not
  apply here directly.
