"""Configuration for RAG indexing and evaluation."""

import os

# ── Chunking (parametrizable) ────────────────────────────────────────────────
# Edit these to compare different chunking strategies.
# Each change creates a NEW Weaviate collection (no overwrite).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
MIN_CHUNK_LENGTH = 100
MAX_CHUNK_LENGTH = 5000
MAX_DIGIT_RATIO = 0.5

# ── Weaviate ─────────────────────────────────────────────────────────────────
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# ── Embeddings (one collection per model) ────────────────────────────────────
EMBEDDINGS = {
    "all-MiniLM-L6-v2": {
        "model_class": "HuggingFaceEmbeddings",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    },
    "gte-multilingual-base": {
        "model_class": "HuggingFaceEmbeddings",
        "model_name": "Alibaba-NLP/gte-multilingual-base",
        "dim": 768,
        "trust_remote_code": True,
    },
    "e5-large-instruct": {
        "model_class": "E5InstructEmbeddings",  # clase custom
        "model_name": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
    },
}


def collection_name(embedding_key: str) -> str:
    """Build collection name: Idearq{Key}_{chunk}_{overlap}_v2"""
    key_map = {
        "all-MiniLM-L6-v2": "MiniLM",
        "gte-multilingual-base": "GTE",
        #"e5-large-instruct": "e5largeinstruct",
    }
    key = key_map.get(embedding_key, embedding_key.replace("-", ""))
    return f"Idearq{key}_{CHUNK_SIZE}_{CHUNK_OVERLAP}_v2"


# ── Indexing schema (12 props for PDFs, shared across all collections) ───────
INDEX_PROPERTIES = [
    ("content", "TEXT"),
    ("filename", "TEXT"),
    ("title", "TEXT"),
    ("source", "TEXT"),
    ("chunk_index", "INT"),
    ("doc_index", "INT"),
    ("year", "INT"),
    ("language", "TEXT"),
    ("doi", "TEXT"),
    ("authors", "TEXT"),
    ("periodo", "TEXT"),
    ("region", "TEXT"),
]

# ── Full schema (PDFs + GeoJSON in same collection) ──────────────────────────
# GeoJSON docs have these extra props; PDFs leave them null.
#
# bp_values / dataciones_c_14 close the "filter by a concrete C14 date" gap
# (e.g. "does Valencina hold 1377±23?"). bp_max alone cannot answer that: it
# holds only the OLDEST date of the site.
#
# Re-vectorization risk: none. IdearqMiniLM_800_50_v2 (and every other Idearq
# collection) runs with `vectorizer: none`, so Weaviate never embeds anything;
# vectors are supplied by the client and a property PATCH leaves them untouched
# (verified before/after backfill, see docs/dataciones-property.md).
#
# BM25 risk: dataciones_c_14 is declared with indexSearchable=False on purpose,
# so the extra text never enters the lexical index and hybrid/BM25 ranking is
# bit-for-bit unchanged. It stays filterable (Equal/Like) via indexFilterable.
INDEX_PROPERTIES_FULL = INDEX_PROPERTIES + [
    ("lat", "NUMBER"),
    ("lon", "NUMBER"),
    ("tipologia_crono", "TEXT"),
    ("bp_mean", "NUMBER"),
    ("bp_min", "NUMBER"),
    ("bp_max", "NUMBER"),
    ("c14_count", "INT"),
    ("doc_type", "TEXT"),
    ("yacimiento_id", "INT"),
    ("yacimiento_nombre", "TEXT"),
    ("unidad_territorial", "TEXT"),
    ("bp_values", "NUMBER_ARRAY"),       # every B.P. age of the site (filterable)
    ("dataciones_c_14", "TEXT"),         # raw date strings (filterable, not BM25)
]

# ── LLMs (3 modelos small via Ollama) ────────────────────────────────────────
LLMS = {
    "Phi-4-mini": "hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q8_0",
    "Qwen3-4B-Instruct-2507": "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0",
    "Llama-3.2-3B-Instruct": "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q8_0",
}
LLM_TEMPERATURES = [0.3, 0.5, 0.7]

# ── Langfuse ─────────────────────────────────────────────────────────────────
LANGFUSE_DATASET_NAME = "RAG-IDEArq-eval-v3-COMPLEX"
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://localhost:4000")
PROMPT_NAMES = {
    "zero_shot": "prompt_zero_shot",
    "one_shot": "prompt_one_shot",
    "few_shot": "prompt_few_shot",
}

# ── RAGAS ────────────────────────────────────────────────────────────────────
RAGAS_JUDGE = "hf.co/unsloth/Qwen3-14B-GGUF:Q6_K"  # local Ollama judge (Qwen3-14B Q6_K)
RAGAS_METRICS = ["faithfulness", "context_precision", "context_recall", "answer_correctness"]

# ── Reranking ─────────────────────────────────────────────────────────────────
RERANKING_CONFIG = {
    "enabled": True,  # Phase 2: reranking activado
    "model": "BAAI/bge-reranker-v2-m3",  # Multilingüe (100+ idiomas)
    "k_retrieval": 10,  # Docs a traer antes de rerank
    "top_n": 5,  # Docs a quedarse después de rerank
}

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INGESTA_DIR = os.path.join(BASE_DIR, "data", "ingesta")
GEOJSON_DIR = os.path.join(BASE_DIR, "data", "ingesta", "geojson", "c14_v2.geojson")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
