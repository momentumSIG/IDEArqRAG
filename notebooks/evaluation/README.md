# Notebooks de evaluación

Mapa de `notebooks/evaluation/`. El estado se deriva de evidencia (referencias en
`scripts/`, `*.sbatch` y otros notebooks), no de suposiciones:

- **activo** — referenciado por un script/sbatch o declarado principal.
- **dudoso** — sin referencias de ejecución, pero con uso documentado o único.
- **muerto** — sin referencias y superado; movido a `notebooks/_archive/`.

| notebook | qué pregunta responde | estado | nota |
|---|---|---|---|
| `rag-evaluacion-hibrida-multiquery.ipynb` | Compara estrategias de recuperación (vector, híbrida, multiquery) con RAGAS + hit@k/MRR | **activo** | **Punto de entrada de la evaluación actual.** `AGENTS.md:74` lo declara «main»; `scripts/smoke-notebook.sbatch:46` lo ejecuta por `jupyter nbconvert --execute`. |
| `viz-geo-check.ipynb` | Smoke de la ruta geográfica (Route B) en la partición `viz` | **activo** | Réplica de las celdas 13/15 del maestro; `scripts/viz-geo-check.sbatch:45` lo ejecuta. Incluye `hit_at_k`/`reciprocal_rank` (celdas 25-27). |
| `rag-chonkie-v2-langfuse.ipynb` | Flujo de autoría de prompts en la UI de Langfuse | **dudoso** | Sin referencias en `scripts/`, `.sbatch` ni otros notebooks. Los prompts están congelados en `data/prompts.json` (`odd/tasks/prompts-freeze-weaviate.md`). No se archiva por ser la única herramienta de autoría. |
| `rag-evaluacion-con-chonkie.ipynb` | Evaluación de `e5-large-instruct` (1024D) con Langfuse + Ollama local | **dudoso** | Sin referencias de ejecución, pero `docs/future-roadmap.md:342` sigue apuntando aquí para evaluar e5. No se archiva. |
| `micro_jaen_mqes.ipynb` | Micro-experimento de Jaén con multiquery en español (mqes) | **muerto** | Sin referencias en `scripts/`, `.sbatch` ni otros notebooks; superado por `rag-evaluacion-hibrida-multiquery` (el sufijo `mqes` acabó integrado en el maestro, ver backup `…-mqes`). Movido a `notebooks/_archive/`. |
| `rag-evaluacion-langgraph.ipynb` | Evaluación con LangGraph + RAGAS | **muerto** | Sin referencias en `scripts/`, `.sbatch` ni otros notebooks; la evaluación RAGAS quedó absorbida por el maestro. Movido a `notebooks/_archive/`. |
| `rag-evaluacion-ragas.ipynb` | Evaluación RAGAS pura con Langfuse | **muerto** | Sin referencias de ejecución; absorbido por el maestro. `odd/tasks/prompts-freeze-weaviate.md` registra un `NameError` (`w_client`). Movido a `notebooks/_archive/`. |
| `rag-idearq-langgraph-weaviate.ipynb` | Orquestación LangGraph + Weaviate previa a `src/` | **muerto** | Sin referencias en `scripts/`, `.sbatch` ni otros notebooks; su función vive ahora en `src/backend_flask.py` + `langgraph.json`. Movido a `notebooks/_archive/`. |

## Punto de entrada actual

`rag-evaluacion-hibrida-multiquery.ipynb` es el notebook maestro de evaluación:
`AGENTS.md` lo marca como «main» y `scripts/smoke-notebook.sbatch` lo ejecuta
headless (con `SMOKE_ONLY=True` en la celda 1). Es el único con métricas de
recuperación deterministas (`hit_at_k`, `reciprocal_rank`/MRR) y referencias a
`scripts/run_fase.py`. `viz-geo-check.ipynb` es el smoke específico de la ruta
geográfica, no la entrada de la evaluación general.

Los scripts `scripts/run_full_evaluation.py` (rejilla completa) y
`scripts/run_fase.py` (fases A/B/C/D/G) son ejecutables independientes, no
notebooks.
