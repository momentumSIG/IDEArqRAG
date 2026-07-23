"""
LangGraph evaluation graph for RAG-IDEArq.

Retrieve → Rerank (optional) → Generate → Evaluate (RAGAS)

Usage with LangGraph Studio:
    cd /home/raglinux/RAG
    source ../env_rag/bin/activate
    langgraph dev --port 8123

Usage from notebook:
    from src.graph_eval import graph
    result = graph.invoke({
        "question": "...",
        "embedding_key": "all-MiniLM-L6-v2",
        "llm_name": "Phi-3.5-mini",
        "prompt_key": "zero_shot",
        ...
    })
"""
from __future__ import annotations

import os
import sys
import time
import logging
import gc
from pathlib import Path
from typing import TypedDict, List, Optional, Dict, Any

# Setup paths
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.config import (
    EMBEDDINGS, LLMS, LLM_TEMPERATURES, WEAVIATE_URL,
    collection_name, RERANKING_CONFIG, PROMPT_NAMES,
    RAGAS_JUDGE, RAGAS_METRICS,
)
from src.reranker import rerank_documents
from src.langfuse_monitor import get_callback, get_prompt, score_trace

import torch
import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── State ─────────────────────────────────────────────────────────────────────
class EvalState(TypedDict, total=False):
    # Inputs (parámetros dinámicos)
    question: str
    embedding_key: str        # "all-MiniLM-L6-v2", "gte-multilingual-base", "e5-large-instruct"
    llm_name: str             # "Phi-3.5-mini", "Qwen3-4B-Instruct-2507", "Llama-3.2-3B-Instruct"
    prompt_key: str           # "zero_shot", "one_shot", "few_shot"
    temperature: float        # 0.3, 0.5, 0.7
    use_rerank: bool          # True/False
    ground_truth: str         # Para evaluación RAGAS

    # Outputs (intermedios)
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    context: str
    answer: str
    latency_s: float

    # Output final (RAGAS)
    ragas_scores: Dict[str, float]


# ── Recursos Lazy ─────────────────────────────────────────────────────────────
_RESOURCES: Dict[str, Any] = {}
_RESOURCES_LOADED = False


def _ensure_resources():
    """Carga recursos la primera vez que se invocan."""
    global _RESOURCES, _RESOURCES_LOADED
    if _RESOURCES_LOADED:
        return _RESOURCES
    _RESOURCES_LOADED = True

    # Weaviate client
    host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
    _RESOURCES["w_client"] = weaviate.connect_to_local(
        host=host, port=8080, grpc_port=50051,
    )
    logging.info("Weaviate client connected")

    # Embedding models
    _RESOURCES["embedding_models"] = {}
    for emb_key, emb_cfg in EMBEDDINGS.items():
        try:
            emb = HuggingFaceEmbeddings(
                model_name=emb_cfg["model_name"],
                model_kwargs={"device": "cuda"},
                encode_kwargs={"device": "cuda"},
            )
            _RESOURCES["embedding_models"][emb_key] = emb
            logging.info(f"  [OK] {emb_key} on CUDA")
        except Exception as e:
            emb = HuggingFaceEmbeddings(
                model_name=emb_cfg["model_name"],
                model_kwargs={"device": "cpu"},
                encode_kwargs={"device": "cpu"},
            )
            _RESOURCES["embedding_models"][emb_key] = emb
            logging.info(f"  [CPU] {emb_key}")

    # LLM models
    _RESOURCES["llm_models"] = {}
    for llm_name, llm_model in LLMS.items():
        for temp in LLM_TEMPERATURES:
            key = f"{llm_name}_t{temp}"
            _RESOURCES["llm_models"][key] = OllamaLLM(
                model=llm_model, temperature=temp,
            )
            logging.info(f"  [OK] {key}")

    # RAGAS judge (Mistral)
    from langchain_mistralai import ChatMistralAI
    _RESOURCES["ragas_judge"] = ChatMistralAI(
        model=RAGAS_JUDGE, temperature=0.1, max_retries=3, timeout=180,
    )

    return _RESOURCES


def _safe_empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Nodos ─────────────────────────────────────────────────────────────────────
def node_retrieve(state: EvalState) -> dict:
    """Retrieve from Weaviate."""
    res = _ensure_resources()
    w_client = res["w_client"]
    emb_key = state.get("embedding_key", "all-MiniLM-L6-v2")
    coll_name = collection_name(emb_key)

    if not w_client.collections.exists(coll_name):
        return {"retrieved_docs": [], "context": ""}

    embedding = res["embedding_models"][emb_key]
    vs = WeaviateVectorStore(
        client=w_client,
        index_name=coll_name,
        text_key="content",
        embedding=embedding,
        attributes=["filename", "source", "chunk_index", "year", "language",
                    "doi", "authors", "periodo", "region"],
    )

    k = RERANKING_CONFIG["k_retrieval"] if state.get("use_rerank", False) else RERANKING_CONFIG["top_n"]
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(state["question"])

    return {"retrieved_docs": docs}


def node_rerank(state: EvalState) -> dict:
    """Rerank with CrossEncoder (optional)."""
    docs = state.get("retrieved_docs", [])
    if not state.get("use_rerank", False):
        return {"reranked_docs": docs[:RERANKING_CONFIG["top_n"]]}

    if not docs:
        return {"reranked_docs": []}

    ranked = rerank_documents(
        state["question"], docs,
        top_n=RERANKING_CONFIG["top_n"],
        reranker_model=RERANKING_CONFIG["model"],
    )
    return {"reranked_docs": [r["doc"] for r in ranked]}


def node_generate(state: EvalState) -> dict:
    """Generate answer with LLM."""
    docs = state.get("reranked_docs", state.get("retrieved_docs", []))
    context = "\n\n".join([d.page_content for d in docs[:RERANKING_CONFIG["top_n"]]])

    prompt_key = state.get("prompt_key", "zero_shot")
    prompt_template = get_prompt(prompt_key)

    llm_name = state.get("llm_name", "Phi-3.5-mini")
    temp = state.get("temperature", 0.3)
    llm_key = f"{llm_name}_t{temp}"

    res = _ensure_resources()
    llm = res["llm_models"][llm_key]

    t0 = time.time()
    if hasattr(prompt_template, 'format'):
        prompt_text = prompt_template.format(context=context, question=state["question"])
    else:
        prompt_text = prompt_template.compile(context=context, question=state["question"])

    answer = llm.invoke(prompt_text)
    latency = time.time() - t0

    return {"answer": answer, "context": context, "latency_s": latency}


def node_evaluate(state: EvalState) -> dict:
    """Evaluate answer with RAGAS (Mistral judge)."""
    from ragas import EvaluationDataset
    from ragas.metrics import (
        Faithfulness, ContextPrecision, ContextRecall, AnswerCorrectness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample
    from ragas.run_config import RunConfig

    res = _ensure_resources()
    ragas_llm = LangchainLLMWrapper(res["ragas_judge"])
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    ragas_metrics = [
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    ragas_run_config = RunConfig(
        max_workers=1, max_retries=10, max_wait=120, timeout=900,
    )

    docs = state.get("reranked_docs", state.get("retrieved_docs", []))
    contexts = [d.page_content[:800] for d in docs[:RERANKING_CONFIG["top_n"]]]

    sample = SingleTurnSample(
        user_input=state["question"],
        response=state["answer"],
        reference=state.get("ground_truth", ""),
        retrieved_contexts=contexts,
    )

    eval_ds = EvaluationDataset(samples=[sample])

    # Retry on rate limit
    ragas_result = None
    for attempt in range(5):
        try:
            ragas_result = eval_ds.evaluate(
                metrics=ragas_metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=ragas_run_config,
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (2 ** attempt)
                logging.warning(f"Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"RAGAS error: {e}")
                break

    if ragas_result:
        scores = {k: float(v) for k, v in ragas_result.items()}
        # Score in Langfuse
        for metric_name, metric_value in scores.items():
            try:
                score_trace(trace_id=None, name=metric_name, value=metric_value)
            except Exception:
                pass
        return {"ragas_scores": scores}

    return {"ragas_scores": {m: 0.0 for m in RAGAS_METRICS}}


# ── Construcción del grafo ───────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

builder = StateGraph(EvalState)

builder.add_node("retrieve", node_retrieve)
builder.add_node("rerank", node_rerank)
builder.add_node("generate", node_generate)
builder.add_node("evaluate", node_evaluate)

builder.set_entry_point("retrieve")

# Conditional edge: rerank or skip
def _should_rerank(state: EvalState) -> str:
    return "rerank" if state.get("use_rerank", False) else "generate"

builder.add_conditional_edges("retrieve", _should_rerank, {
    "rerank": "rerank",
    "generate": "generate",
})

builder.add_edge("rerank", "generate")
builder.add_edge("generate", "evaluate")
builder.add_edge("evaluate", END)

graph = builder.compile()
