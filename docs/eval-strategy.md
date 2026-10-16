# Estrategia de evaluación por fases — RAG-IDEArq

## 1. Objetivo y alcance

Evaluar de forma reproducible y **barata** la cadena RAG del proyecto (3 embeddings ×
estrategias de retrieval × LLMs × prompts × temperaturas × rerank × contexto de grafo)
sobre el dataset canónico de 32 preguntas (`data/eval_questions.py`,
`get_eval_items()`), con juez RAGAS local (Ollama `Qwen3-14B Q6_K`).

Alcance de este documento:

- Definir las fases (A, B, C, D, E opcional, G), sus grids congelados y sus métricas.
- Definir la infraestructura: **servicios persistentes** (Weaviate + Ollama) + **jobs
  de Slurm por fase** que no arrancan servicios.
- Definar el runner CLI (`scripts/run_fase.py`), el job plantilla
  (`scripts/eval-fase.sbatch`) y el esquema del CSV de resultados.

Fuera de alcance: ejecutar la evaluación, tocar cuotas, levantar Langfuse, mover
MinIO, añadir estrategias nuevas al índice.

## 2. Marco metodológico vs decisión de este proyecto

La literatura de ablaciones en NLP (Rodman et al., 2021) recomienda *controlled
experiments* y *fractional / sequential ablations*: cambiar un factor a la vez sobre
una línea base congelada, o usar diseños fraccionados cuando el grid completo es
inviable. Montgomery cubre el diseño factorial completo y sus fracciones.

**Honestidad:** lo que sigue **no** es "el estándar RAGAS" ni un diseño de experimentos
publicado como tal. Es **nuestra instancia** — una ablación secuencial/fractional
adaptada a un clúster con GPU limitada, un juez local lento y 32 preguntas. Cada fase
congela lo que la fase anterior eligió; es una decisión de ingeniería justificada por
presupuesto de cómputo, no un canon metodológico.

## 3. Grid completo 

Grid completo (full factorial), sin rerank:

```
3 embeddings × 6 strategies × 3 LLM × 3 prompts × 3 temps = 486 combos
486 × 32 preguntas ≈ 15.552 ejecuciones pipeline + juez
```

Con rerank on/off (×2): **≈ 31k ejecuciones**. A eso súmale el juez RAGAS
(Qwen3-14B en CPU/GPU compartida) que domina el coste por muestra, y el grid completo
queda en **semanas** de un solo A100. Inviable.

Plan por fases: **≈ 50–60 combinaciones** en total (A=18, B=27, C=4, D≈4, G=4;
E opcional), cada una sobre 32 preguntas → órdenes de magnitud menos cómputo, al precio
de no observar interacciones de alto orden (asunción estándar de ablación secuencial:
los efectos de interacción fuerte entre factores lejanos son poco probables o quedan
cubiertos por las fases C/D).

## 4. Fases

### A — Retrieval (base de todo lo demás)

- Grid: **3 embeddings** (`all-MiniLM-L6-v2`, `gte-multilingual-base`,
  `e5-large-instruct` → `collection_name()`) **× 6 estrategias**
  (`vector`, `hybrid_0.3`, `hybrid_0.5`, `hybrid_0.7`, `mq_vector`, `mq_hybrid_0.5`)
  = 18 combos.
- Congelado: LLM = `Qwen3-4B-Instruct-2507`, prompt = `one_shot`, temp = `0.5`,
  rerank **OFF**, graph **OFF**.
- Métricas de decisión: `context_precision`, `context_recall`
  (`faithfulness` opcional solo sobre los ganadores de A).
- Salva: el embedding + estrategia ganadores pasan congelados a B/C/D.

### B — Generación

- Congela el ganador de A (embedding + estrategia; fallback:
  `gte-multilingual-base` + `hybrid_0.5` si A no existe).
- Grid: **3 LLM** (`Phi-4-mini`, `Qwen3-4B-Instruct-2507`,
  `Llama-3.2-3B-Instruct`) **× 3 prompts** (`zero_shot`, `one_shot`, `few_shot`)
  **× 3 temps** `[0.3, 0.5, 0.7]` = 27 combos.
- Rerank OFF, graph OFF.

### C — Rerank on/off

- Sobre los finalistas de A/B (estrategias ganadoras; fallback
  `vector` + `hybrid_0.5`): `use_rerank ∈ {False, True}` → 4 combos.
- Reranker fijo: `BAAI/bge-reranker-v2-m3`, `k_retrieval=10`, `top_n=5`
  (`src/reranker.py`).
- Resto congelado (mejor LLM/prompt/temp de B; fallback `Qwen3-4B`/`one_shot`/`0.5`).

### D — Modelos grandes (14B / 30B)

- Solo sobre finalistas de B: 2 LLMs grandes (`Qwen3-14B`, `Qwen3-30B-A3B`;
  tags ajustables en `LARGE_LLMS` del runner) × finalistas de retrieval
  ≈ 4 combos.
- **Justificación GPU:** 14B Q6_K ≈ 10 GB de pesos y 30B ≈ 17–20 GB; en A100 40 GB
  caben ambos + KV cache para `num_ctx=16384`, pero el juez RAGAS (14B) compite por
  la misma GPU. Por eso D es la única fase candidata a **2×A100**; A/B/C/G van con
  **1 GPU + workers=1**.

### E — Ablación del generador multiquery (opcional)

- Cambiar `MQ_GENERATOR_MODEL`/`MQ_GENERATOR_TEMP` sobre `mq_*` de A.
- **Se salta si el presupuesto aprieta** — no cambia la decisión principal.

### G — Grafo bibliográfico (context-only)

- 4 combos: `{vector | hybrid_0.5} × {graph off, graph on}`.
- El grafo **solo añade contexto**: los docs RDF de `data/biblio-graph/
  idearq-graph-instances-dedup-v2.nt` que coinciden por título/keywords con la
  pregunta se **appendan al final** del contexto de retrieval
  (SPARQL + `Document(metadata.source="biblio_graph")`).
- **NO existe `graph_only`** como estrategia; `STRATEGIES` no se toca.

## 5. Infraestructura: servicios persistentes + jobs por fase

- **Servicios**: `rag-services.sbatch` (QOS `long`, 7d) ejecuta `start_rag.sh` y
  mantiene Weaviate (`:8080`) + Ollama (`:11434`) vivos. **No se arrancan desde
  los jobs de evaluación.**
- **Guard de fase**: antes de cada corrida, `run_fase.py` comprueba
  `GET $WEAVIATE_URL/v1/.well-known/ready` (fallback `/v1/schema`).
  - Exactamente **una** instancia de Weaviate esperada (host de `services.env`).
  - Si el health-check falla → el runner **no arranca** y printa remediation:
    1. `squeue -u $USER` → localizar el job `rag-services` (debe estar RUNNING) y
       cualquier job residual no-persistente que tenga un Weaviate encima;
    2. `scancel <job_residual>` (preferido: matar el **job**, no el proceso);
    3. alternativa solo si no hay job: en el nodo, `pkill -f weaviate` **tras**
       verificar que `WEAVIATE_PID` de `services.env` está muerto (PID estancado);
    4. reintentar `curl -s $WEAVIATE_URL/v1/.well-known/ready` → 200.
  - `--kill-residual-weaviate`: opt-in del runner. Solo hace `pkill` local de
    weaviate si `WEAVIATE_PID` de `services.env` está estancado **y** el hostname
    actual es el nodo de servicios. **Peligro:** matar el PID equivocado corrompe
    el vectordb si quedan dos instancias; por eso el runner **nunca hace scancel**
    y este flag es explícito.
  - Nunca dos Weaviates sobre el mismo `vectordb`.
- Campos de `services.env` usados: `WEAVIATE_URL`, `WEAVIATE_HOST`, `OLLAMA_URL`,
  `OLLAMA_HOST`, `WEAVIATE_PID`, `IDEARQ_SERVICES_NODE`. El runner los lee con el
  mismo parser que la celda 1 del notebook (`_load_ollama_base`).

## 6. Slurm (CESGA)

- `--mem` **obligatorio** (si no: *Job violates accounting/QOS policy*). Plantilla
  `eval-fase.sbatch` con `--mem=32G`, `-N 1`.
- QOS/partición **no** se pasa a mano: `JobSubmitPlugins=require_timelimit,lua`
  deriva QOS de `--time` (3 min→short/6h, 1d→medium/3d, 7d→long/7d). Pasar
  `--qos` a mano falla con *Invalid qos specification*.
- `--partition`/`--qos` en la plantilla son **informativos**, no autoritativos.

Estimaciones (orden de magnitud, **incertidumbre ×2–×3** — el juez manda):

| Fase | Combos × 32 | 1 GPU (A100) | Notas |
|------|-------------|--------------|-------|
| A    | 18 × 32 = 576 | 6–18 h     | retrieval barato; juez en batch por combo |
| B    | 27 × 32 = 864 | 8–24 h     | 3 LLMs × 3 × 3 |
| C    | 4 × 32 = 128  | 1–4 h      | rerank cross-encoder en CPU/GPU |
| D    | ≈4 × 32       | 12–36 h    | **candidato a 2×A100**: modelo grande + juez 14B |
| G    | 4 × 32 = 128  | 1–4 h      | + rdflib (segundos) |
| E    | opcional      | 2–6 h      | solo si sobra presupuesto |

Smoke (`--smoke`, 3 preguntas): minutos por fase.

## 7. Runner CLI

```bash
# Fase A en smoke (3 preguntas) y reanudable:
python scripts/run_fase.py --fase A --smoke --resume

# Fase G con baseline híbrido, solo graph-on (por defecto corre los 4 combos):
python scripts/run_fase.py --fase G --smoke --base-strategy hybrid_0.5 --graph

# Workers >1 es EXPERIMENTAL (Ollama serializa internamente):
python scripts/run_fase.py --fase B --workers 1

# Solo si hay residual confirmado y PID estancado en services.env:
python scripts/run_fase.py --fase A --kill-residual-weaviate
```

Flags: `--fase {A,B,C,D,G}`, `--smoke`, `--resume`, `--workers N` (default 1),
`--kill-residual-weaviate`, `--base-strategy {vector,hybrid_0.5}`,
`--graph` / `--no-graph` (G), `--skip-health`.

- `--resume`: reanuda el CSV `fase{X}_{smoke|full}_*.csv` más reciente; salta claves
  `(combo, question)` ya presentes y **appendea**.
- Salida: `data/results/fase{X}_{smoke|full}_{ts}.csv`.

## 8. Esquema CSV (columnas)

```
fase, combo, embedding, llm, prompt, temperature, strategy, use_rerank,
graph_context, question, ground_truth, tipo, idioma, answer, n_docs,
filenames, context_chars, latency_retrieval_s, latency_gen_s, latency_judge_s,
prompt_tokens_gen, completion_tokens_gen, prompt_tokens_judge,
completion_tokens_judge, total_tokens, faithfulness, context_precision,
context_recall, answer_correctness, error
```

Tokens: se capturan del `prompt_eval_count`/`eval_count` de Ollama (o
`response.usage` / `response_metadata['token_usage']` de LangChain) para la
generación y, cuando está disponible, para el juez; si no, `None`. **SPARQL no
consume tokens**: el grafo solo aporta `ms_graph` (latencia de query+merge dentro
de `latency_retrieval_s`).

## 9. Fundamentación y referencias

- Es, S. et al. — *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
  (arXiv:2309.15217; EACL 2025 System Demonstrations). Métricas
  `faithfulness`, `context_precision`, `context_recall`, `answer_correctness`.
- Rodman, E. et al. — *A Best Practices Guide to Ablation Studies for NLP*
  (arXiv:2010.07292). Fundamenta la ablación controlada/secuencial de la §2.
- Montgomery, D. C. — *Design and Analysis of Experiments*. Factorial completo y
  diseños fraccionados (§3).
- Cormack, G. et al. — *Reciprocal Rank Fusion* (SIGIR 2009). Base de `rrf_fuse`
  en las estrategias `mq_*`.
- Documentos internos (trabajo futuro): `docs/future-work-hybrid-search.md`,
  `docs/future-roadmap.md`.

## 10. Qué NO se hace

- **No** se arranca Langfuse (stack caído; prompts congelados en
  `data/prompts.json`).
- **No** existe estrategia `graph_only` (solo append de contexto).
- **No** se mueve MinIO ni se reindexa el vectordb para esta evaluación.
- **No** se pide cuota GPU adicional: se dimensiona con lo que ya hay
  (1 GPU por defecto; 2×A100 solo como opción para D).
- **No** se arranca Weaviate/Ollama desde los jobs de fase (servicios persistentes).
