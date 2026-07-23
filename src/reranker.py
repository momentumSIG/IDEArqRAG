"""Reranking module for RAG-IDEArq.

Uses BAAI/bge-reranker-v2-m3 (multilingual) for cross-encoder reranking.
"""
from typing import Any, Dict, List
from sentence_transformers import CrossEncoder

# Cache del modelo (singleton a nivel de módulo)
_RERANKER_CACHE: Dict[str, Any] = {"model_name": None, "model": None}


def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Carga el modelo de reranking con cache.
    
    Args:
        model_name: Nombre del modelo CrossEncoder
        
    Returns:
        Instancia de CrossEncoder cacheada
    """
    if _RERANKER_CACHE["model"] is None or _RERANKER_CACHE["model_name"] != model_name:
        _RERANKER_CACHE["model"] = CrossEncoder(model_name)
        _RERANKER_CACHE["model_name"] = model_name
    return _RERANKER_CACHE["model"]


def rerank_documents(
    question: str,
    docs: List[Any],
    top_n: int = 5,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
) -> List[Dict[str, Any]]:
    """Rerank documents y devuelve los top_n mejores.
    
    Args:
        question: Pregunta del usuario
        docs: Lista de Documents de LangChain
        top_n: Número de documentos a devolver
        reranker_model: Nombre del modelo CrossEncoder
        
    Returns:
        Lista de dicts con {'doc': Document, 'score': float}
    """
    if not docs:
        return []
    
    model = get_reranker(reranker_model)
    scores = model.predict([(question, d.page_content or "") for d in docs])
    ranked = sorted(
        [{"doc": d, "score": float(s)} for d, s in zip(docs, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_n]


def clear_cache():
    """Limpia el cache del modelo (útil para tests)."""
    global _RERANKER_CACHE
    _RERANKER_CACHE = {"model_name": None, "model": None}
