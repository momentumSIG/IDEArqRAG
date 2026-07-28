# RAG de recursos arqueológicos 

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C?logo=langchain)
![Weaviate](https://img.shields.io/badge/Vector_DB-Weaviate-00C7B7)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)
![LLMs](https://img.shields.io/badge/LLMs-3%20models-purple)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?logo=huggingface&logoColor=yellow)
![Archaeology](https://img.shields.io/badge/Domain-Archaeology-brown)

---
Retrieval-Augmented Generation sobre documentación arqueológica de IDEArq
(http://www.idearqueologia.org). Evalúa modelos pequeños (Phi-4-mini,
Qwen3-4B, Llama-3.2-3B) con embeddings pequeños (AllMiniLM-L6-v2,
GTE-multilingual-base y e5-multilingual) y RAGAS.

---

## Dataset de evaluación

30 preguntas de evaluación sobre arqueología ibéricadivididas en **15 simples** (1 artículo) y **15 complejas** (2+ artículos).


### Definición de complejidad
- **Simple**: la respuesta se encuentra en un único paper/artículo del corpus.
- **Compleja**: la respuesta requiere sintetizar información de 2 o más papers
  (listas de yacimientos, comparaciones regionales, periodizaciones).

---

## Cómo usar el dataset

1. Ejecutar el notebook de evaluación para crear `RAG-IDEArq-eval-v3` en Langfuse.
2. Rellenar la columna "Fuente" con el nombre del PDF exacto (ej. `1000_oms_2017.pdf`) en Langfuse UI.
3. Las preguntas se evalúan con RAGAS (faithfulness, context_precision, context_recall, answer_correctness).
4. El reporte se segmenta por `tipo` (simple/compleja), `idioma` y `temperatura`.

## Modelos evaluados
- **LLMs**: Phi-4-mini, Qwen3-4B-Instruct-2507, Llama-3.2-3B-Instruct
- **Temperaturas**: 0.3, 0.5, 0.7
- **Embeddings**: all-MiniLM-L6-v2 (384D), gte-multilingual-base (768D), e5-large-instruct (1024D)
- **Judge RAGAS**: qwen-7B

## Estructura del proyecto
```
RAG/
├── src/                          ← Código Python
│   ├── config.py                 ← Parámetros de chunking y embeddings
│   ├── langfuse_monitor.py       ← Conexión a Langfuse
├── notebooks/
│   ├── indexing/                 ← Notebooks de indexación
│   └── evaluation/               ← Notebooks de evaluación
├── deploy/docker/                ← Docker Compose (Weaviate)
└── weaviate_data/                ← Datos de Weaviate
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
