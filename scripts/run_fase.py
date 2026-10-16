#!/usr/bin/env python3
"""Run one evaluation phase (A, B, C, D or G) for RAG-IDEArq.

Closed design decisions reflected here:
- Infrastructure is *persistent services*: Weaviate + Ollama stay alive under
  rag-services.sbatch / start_rag.sh. This runner NEVER starts them and phase
  sbatch jobs must NOT call start_rag.sh (two Weaviates on one vectordb
  corrupt it).
- Before each phase a health guard requires exactly one ready Weaviate at
  WEAVIATE_URL (services.env). On failure it prints remediation instructions;
  it does not scancel anything unless --kill-residual-weaviate is passed AND
  services.env records a stale WEAVIATE_PID on THIS node.
- Phase G is context-only: biblio RDF docs are appended to the retrieval
  context. There is no graph_only strategy.

Usage:
    python scripts/run_fase.py --fase A --smoke
    python scripts/run_fase.py --fase A --smoke --resume
    python scripts/run_fase.py --fase B --workers 1
    python scripts/run_fase.py --fase G --smoke --base-strategy hybrid_0.5 --graph

CSV output: data/results/fase{X}_{smoke|full}_{ts}.csv
Append/resume: --resume reuses the newest CSV for the same fase+mode and skips
rows whose (combo, question) key already exists.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError

# PROJECT_ROOT from this file (scripts/run_fase.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Constants (mirror src/config.py + the evaluation notebook cell 1)
# ---------------------------------------------------------------------------
SMOKE_N = int(os.environ.get("SMOKE_N", "3"))

EMBEDDING_KEYS = [
    "all-MiniLM-L6-v2",
    "gte-multilingual-base",
    "e5-large-instruct",
]
PROMPTS = ["zero_shot", "one_shot", "few_shot"]
TEMPERATURES = [0.3, 0.5, 0.7]
STRATEGIES_A = [
    "vector",
    "hybrid_0.3",
    "hybrid_0.5",
    "hybrid_0.7",
    "mq_vector",
    "mq_hybrid_0.5",
]
BASE_STRATEGIES_G = ["vector", "hybrid_0.5"]

# Phase D large models (GPU-heavy). Tags are adjustable here to whatever
# `ollama list` actually shows on the services node.
LARGE_LLMS = {
    "Qwen3-14B": "hf.co/unsloth/Qwen3-14B-GGUF:Q6_K",
    "Qwen3-30B-A3B": "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q8_0",
}

DEFAULT_EMBEDDING = "gte-multilingual-base"
DEFAULT_LLM = "Qwen3-4B-Instruct-2507"
DEFAULT_PROMPT = "one_shot"
DEFAULT_TEMP = 0.5

MQ_GENERATOR_MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0"
MQ_GENERATOR_TEMP = 0.2
RAGAS_JUDGE = "hf.co/unsloth/Qwen3-14B-GGUF:Q6_K"
N_QUERIES = 4
RRF_K = 60
K_CONTEXT = 10  # final docs in the LLM context
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_K_RETRIEVAL = 10
RERANK_TOP_N = 5

BM25_PROPS = ["content", "tipologia_crono", "unidad_territorial"]
RETURN_PROPS = [
    "content", "filename", "doc_type", "yacimiento_nombre",
    "tipologia_crono", "unidad_territorial", "year", "language", "chunk_index",
]

GRAPH_PATH = (
    PROJECT_ROOT / "data" / "biblio-graph" / "idearq-graph-instances-dedup-v2.nt"
)
GRAPH_TOP_N = 5

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "fase", "combo", "embedding", "llm", "prompt", "temperature", "strategy",
    "use_rerank", "graph_context", "question", "ground_truth", "tipo",
    "idioma", "answer", "n_docs", "filenames", "context_chars",
    "latency_retrieval_s", "latency_gen_s", "latency_judge_s",
    "prompt_tokens_gen", "completion_tokens_gen", "prompt_tokens_judge",
    "completion_tokens_judge", "total_tokens", "faithfulness",
    "context_precision", "context_recall", "answer_correctness", "error",
]
METRIC_KEYS = [
    "faithfulness", "context_precision", "context_recall", "answer_correctness",
]


# ---------------------------------------------------------------------------
# Env / services.env loading (same parse as notebook _load_ollama_base)
# ---------------------------------------------------------------------------
def load_services_env() -> Dict[str, str]:
    """Parse services.env (export KEY="value" lines) from the known locations.

    Never raises; returns {} when no file is readable.
    """
    candidates = [
        Path("/mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/services.env"),
    ]
    lustre = os.environ.get("LUSTRE")
    if lustre:
        candidates.append(Path(lustre) / "idearq-rag" / "services.env")
    candidates.append(PROJECT_ROOT / "services.env")
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        vals = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip().strip('"').strip("'")
        if vals:
            return vals
    return {}


def resolve_weaviate_url(services: Dict[str, str]) -> str:
    raw = (
        services.get("WEAVIATE_URL")
        or os.environ.get("WEAVIATE_URL")
        or "http://localhost:8080"
    )
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def resolve_ollama_base(services: Dict[str, str]) -> str:
    raw = (
        services.get("OLLAMA_URL")
        or services.get("OLLAMA_HOST")
        or os.environ.get("OLLAMA_URL")
        or os.environ.get("OLLAMA_HOST")
        or ""
    ).strip()
    if not raw:
        raise RuntimeError(
            "Cannot resolve OLLAMA base URL: no OLLAMA_URL/OLLAMA_HOST in "
            "services.env or process env"
        )
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/").removesuffix("/v1").rstrip("/")


def parse_host_port(url: str, default_port: int) -> Tuple[str, int]:
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    return host, port


# ---------------------------------------------------------------------------
# Health guard: exactly one Weaviate, remediation on failure
# ---------------------------------------------------------------------------
def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (URLError, OSError, ValueError):
        return False


def weaviate_ready(base_url: str) -> bool:
    ready = base_url.rstrip("/") + "/v1/.well-known/ready"
    if http_ok(ready):
        return True
    # Fallback when the ready endpoint is absent (older proxies).
    return http_ok(base_url.rstrip("/") + "/v1/schema")


def print_remediation(base_url: str) -> None:
    print("ERROR: Weaviate is not ready at " + base_url, file=sys.stderr)
    print(
        "Persistent services (rag-services.sbatch / start_rag.sh) own Weaviate "
        "and Ollama; phase jobs must NOT start them.",
        file=sys.stderr,
    )
    print("Remediation:", file=sys.stderr)
    print("  1. squeue -u $USER   # rag-services must be RUNNING", file=sys.stderr)
    print(
        "  2. If a residual NON-persistent job holds a Weaviate on the same "
        "vectordb: scancel <that job> (kill the JOB, not the process).",
        file=sys.stderr,
    )
    print(
        "  3. Fallback on the services node only, after confirming the "
        "WEAVIATE_PID in services.env is stale: pkill -f weaviate",
        file=sys.stderr,
    )
    print(
        "  4. curl -s " + base_url + "/v1/.well-known/ready   # expect HTTP 200",
        file=sys.stderr,
    )
    print(
        "Never run two Weaviate instances against the same vectordb. "
        "Optional: re-run with --kill-residual-weaviate only when "
        "WEAVIATE_PID in services.env is stale on THIS node.",
        file=sys.stderr,
    )


def kill_residual_weaviate(services: Dict[str, str]) -> bool:
    """Best-effort kill of a LOCAL residual Weaviate.

    DANGER: pkill -f weaviate can take down the persistent service if the
    guards below are wrong. This runner NEVER runs scancel; killing a residual
    Slurm job remains a manual step (prefer scancel of that job).

    Guards (all must hold):
      - services.env records WEAVIATE_PID;
      - WEAVIATE_PID is stale (process not alive) on THIS node;
      - current hostname matches IDEARQ_SERVICES_NODE / WEAVIATE_HOST.
    """
    pid_raw = (services.get("WEAVIATE_PID") or "").strip()
    if not pid_raw:
        print(
            "kill-residual: WEAVIATE_PID missing in services.env — refusing "
            "to pkill without a clear residual PID.",
            file=sys.stderr,
        )
        return False
    try:
        pid = int(pid_raw)
    except ValueError:
        print("kill-residual: WEAVIATE_PID is not an integer — refusing.", file=sys.stderr)
        return False

    this_host = socket.gethostname().split(".")[0]
    services_node = (
        services.get("IDEARQ_SERVICES_NODE")
        or (services.get("WEAVIATE_HOST") or "").split(":")[0]
        or ""
    ).split(".")[0]
    if services_node and services_node != this_host:
        print(
            "kill-residual: this node (" + this_host + ") is not the services "
            "node (" + services_node + ") — refusing local pkill.",
            file=sys.stderr,
        )
        return False

    # PID alive => that is the persistent service; never kill it.
    if _pid_alive(pid):
        print(
            "kill-residual: WEAVIATE_PID " + str(pid) + " is alive (persistent "
            "service) — refusing to kill.",
            file=sys.stderr,
        )
        return False

    print(
        "kill-residual: WEAVIATE_PID " + str(pid) + " is stale on " + this_host
        + " — attempting pkill -f weaviate (DANGEROUS, local only).",
        file=sys.stderr,
    )
    try:
        subprocess.run(["pkill", "-f", "weaviate"], check=False)
        time.sleep(3.0)
    except OSError as exc:
        print("kill-residual: pkill failed: " + str(exc), file=sys.stderr)
        return False
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Weaviate connection (host from WEAVIATE_URL, not hardcoded localhost)
# ---------------------------------------------------------------------------
def connect_weaviate_client(weaviate_url: str, headers: Optional[dict] = None):
    try:
        from src.weaviate_conn import connect_weaviate  # type: ignore

        return connect_weaviate(headers=headers)
    except Exception:
        pass
    import weaviate

    host, port = parse_host_port(weaviate_url, 8080)
    # gRPC port convention on this cluster: HTTP 8080 -> gRPC 50051.
    return weaviate.connect_to_local(
        host=host, port=port, grpc_port=50051, headers=headers
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_items(smoke: bool) -> List[Dict[str, Any]]:
    from data.eval_questions import get_eval_items

    raw = get_eval_items()
    items = []
    for entry in raw:
        inp = entry.get("input") or {}
        out = entry.get("expected_output") or {}
        items.append(
            {
                "question": inp.get("question"),
                "ground_truth": out.get("ground_truth"),
                "tipo": inp.get("tipo"),
                "idioma": inp.get("idioma"),
            }
        )
    if smoke:
        items = items[:SMOKE_N]
    return items


# ---------------------------------------------------------------------------
# Embeddings / retrieval (ported from the evaluation notebook)
# ---------------------------------------------------------------------------
_EMB_CACHE: Dict[str, Any] = {}
_LLM_CACHE: Dict[Tuple[str, float], Any] = {}
_MQ_LLMS: Dict[str, Any] = {}


def load_embeddings(key: str):
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]
    import torch
    from langchain_core.embeddings import Embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import SentenceTransformer

    from src.config import EMBEDDINGS

    cfg = EMBEDDINGS[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.get("model_class") == "E5InstructEmbeddings":

        class E5InstructEmbeddings(Embeddings):
            def __init__(self, model_name, device=None):
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.model = SentenceTransformer(model_name, device=self.device)

            def embed_documents(self, texts):
                return self.model.encode(
                    ["passage: " + t for t in texts], device=self.device
                ).tolist()

            def embed_query(self, text):
                return self.model.encode(["query: " + text], device=self.device)[0].tolist()

        emb = E5InstructEmbeddings(model_name=cfg["model_name"], device=device)
    else:
        mk = {"device": device}
        if cfg.get("trust_remote_code"):
            mk["trust_remote_code"] = True
        emb = HuggingFaceEmbeddings(
            model_name=cfg["model_name"],
            model_kwargs=mk,
            encode_kwargs={"device": device},
        )
    _EMB_CACHE[key] = emb
    return emb


def get_gen_llm(llm_key: str, temperature: float, ollama_base: str):
    from langchain_ollama import ChatOllama

    from src.config import LLMS

    cache_key = (llm_key, float(temperature))
    if cache_key not in _LLM_CACHE:
        model = LLMS.get(llm_key, llm_key)
        _LLM_CACHE[cache_key] = ChatOllama(
            model=model,
            temperature=float(temperature),
            num_ctx=16384,
            num_predict=-1,
            base_url=ollama_base,
        )
    return _LLM_CACHE[cache_key]


def get_mq_llm(ollama_base: str):
    if "default" not in _MQ_LLMS:
        from langchain_ollama import ChatOllama

        _MQ_LLMS["default"] = ChatOllama(
            model=MQ_GENERATOR_MODEL,
            temperature=MQ_GENERATOR_TEMP,
            base_url=ollama_base,
        )
    return _MQ_LLMS["default"]


def to_docs(objects) -> List[Any]:
    from langchain_core.documents import Document

    out = []
    for o in objects:
        props = dict(o.properties)
        content = props.pop("content", "") or ""
        md = {k: v for k, v in props.items() if v is not None}
        md["uuid"] = str(o.uuid)
        md["score"] = getattr(o.metadata, "score", None)
        out.append(Document(page_content=content, metadata=md))
    return out


def vector_search(col, embedding, query: str, k: int) -> List[Any]:
    emb = embedding.embed_query(query)
    res = col.query.near_vector(
        near_vector=emb,
        limit=k,
        return_metadata=["score"],
        return_properties=RETURN_PROPS,
    )
    return to_docs(res.objects)


def hybrid_search(col, embedding, query: str, alpha: float, k: int) -> List[Any]:
    emb = embedding.embed_query(query)
    res = col.query.hybrid(
        query=query,
        vector=emb,
        alpha=alpha,
        limit=k,
        query_properties=BM25_PROPS,
        return_metadata=["score"],
        return_properties=RETURN_PROPS,
    )
    return to_docs(res.objects)


def generate_query_variants(mq_llm, question: str, n: int = N_QUERIES) -> List[str]:
    prompt = (
        "Genera {n} reformulaciones distintas de la siguiente pregunta de "
        "investigacion arqueologica.\n"
        "Reglas:\n"
        "- Conserva la intencion y los nombres propios (yacimientos, autores, "
        "regiones, dataciones).\n"
        "- Varia el vocabulario y el enfoque; incluye al menos una version en "
        "ingles.\n"
        "- Devuelve SOLO las preguntas, una por linea, sin numeracion ni texto "
        "adicional.\n\n"
        "Pregunta: {question}\n"
    )
    try:
        out = mq_llm.invoke(prompt.format(n=n, question=question))
        text = out.content if hasattr(out, "content") else str(out)
        lines = [
            re.sub(r"^\s*\d+[\.\)\-:\s]+", "", ln).strip()
            for ln in text.splitlines()
            if ln.strip()
        ]
        variants = [ln for ln in lines if len(ln) > 10][:n]
        return variants or [question]
    except Exception as exc:
        print("  MQ gen error: " + str(exc))
        return [question]


def rrf_fuse(lists_of_docs: List[List[Any]], k: int = RRF_K, top_k: int = K_CONTEXT):
    scores: Dict[Any, float] = {}
    docmap: Dict[Any, Any] = {}
    for docs in lists_of_docs:
        for rank, d in enumerate(docs, start=1):
            uid = d.metadata.get("uuid") or hash(d.page_content)
            docmap[uid] = d
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [docmap[uid] for uid, _ in ranked]


def apply_rerank(question: str, docs: List[Any], use_rerank: bool) -> List[Any]:
    """Phase C: optional cross-encoder rerank (k=10 -> top_n=5)."""
    if not use_rerank:
        return docs[:K_CONTEXT]
    from src.reranker import rerank_documents

    ranked = rerank_documents(
        question, docs, top_n=RERANK_TOP_N, reranker_model=RERANKER_MODEL
    )
    return [entry["doc"] for entry in ranked]


def retrieve_docs(
    strategy: str,
    question: str,
    embedding_key: str,
    col,
    embedding,
    mq_llm,
    use_rerank: bool,
) -> List[Any]:
    retrieve_k = RERANK_K_RETRIEVAL if use_rerank else K_CONTEXT
    if strategy == "vector":
        docs = vector_search(col, embedding, question, retrieve_k)
    elif strategy.startswith("hybrid_"):
        alpha = float(strategy.split("_")[1])
        docs = hybrid_search(col, embedding, question, alpha, retrieve_k)
    elif strategy in ("mq_vector", "mq_hybrid_0.5"):
        variants = [question] + generate_query_variants(mq_llm, question)
        lists = []
        for v in variants:
            if strategy == "mq_vector":
                lists.append(vector_search(col, embedding, v, retrieve_k))
            else:
                lists.append(hybrid_search(col, embedding, v, 0.5, retrieve_k))
        docs = rrf_fuse(lists, top_k=retrieve_k)
    else:
        raise ValueError("Unknown strategy: " + strategy)
    # Parche B (2026-10-16): per-file quota on the ranking list, before the K cut.
    from src.geo_route import cap_per_file

    docs = cap_per_file(docs)
    return apply_rerank(question, docs, use_rerank)


# ---------------------------------------------------------------------------
# Graph context (Phase G, context-only)
# ---------------------------------------------------------------------------
_GRAPH_CACHE: Dict[str, Any] = {"graph": None, "loaded": False, "ms": None}


def get_biblio_graph():
    if _GRAPH_CACHE["loaded"]:
        return _GRAPH_CACHE["graph"]
    _GRAPH_CACHE["loaded"] = True
    try:
        from rdflib import Graph
    except ImportError:
        print("  graph: rdflib not installed — graph context disabled")
        return None
    if not GRAPH_PATH.is_file():
        print("  graph: file missing — graph context disabled: " + str(GRAPH_PATH))
        return None
    t0 = time.time()
    g = Graph()
    g.parse(str(GRAPH_PATH), format="nt")
    _GRAPH_CACHE["graph"] = g
    _GRAPH_CACHE["ms"] = (time.time() - t0) * 1000.0
    print(
        "  graph: loaded "
        + str(len(g))
        + " triples in "
        + str(round(_GRAPH_CACHE["ms"], 1))
        + " ms"
    )
    return g


def sparql_biblio(question: str, top_n: int = GRAPH_TOP_N) -> Tuple[List[Any], float]:
    """Keyword/title SPARQL match over the biblio RDF graph.

    Returns (graph_docs, ms_graph). SPARQL consumes no tokens; the latency is
    reported as ms_graph (folded into latency_retrieval_s by the caller).
    """
    from langchain_core.documents import Document

    g = get_biblio_graph()
    if g is None:
        return [], 0.0
    keywords = [
        w for w in re.findall(r"[\wáéíóúñüÁÉÍÓÚÑÜ]+", question.lower())
        if len(w) > 3
    ][:8]
    if not keywords:
        return [], 0.0
    safe = []
    for kw in keywords:
        kw2 = kw.replace('"', "").replace("\\", "")
        if kw2:
            safe.append(kw2)
    if not safe:
        return [], 0.0
    filters = " || ".join(
        'CONTAINS(LCASE(STR(?title)), "' + kw + '")' for kw in safe
    )
    sparql = (
        "PREFIX dcterms: <http://purl.org/dc/terms/> "
        "SELECT DISTINCT ?s ?title WHERE { ?s dcterms:title ?title . "
        "FILTER(" + filters + ") } LIMIT 100"
    )
    t0 = time.time()
    try:
        rows = list(g.query(sparql))
    except Exception as exc:
        print("  sparql_biblio error: " + str(exc))
        return [], (time.time() - t0) * 1000.0
    scored = []
    for row in rows:
        uri = str(row[0])
        title = str(row[1])
        tl = title.lower()
        hits = sum(1 for kw in safe if kw in tl)
        if hits:
            scored.append((hits, uri, title))
    scored.sort(key=lambda item: -item[0])
    docs = []
    for hits, uri, title in scored[:top_n]:
        docs.append(
            Document(
                page_content="[biblio_graph] " + title + " (" + uri + ")",
                metadata={
                    "source": "biblio_graph",
                    "uri": uri,
                    "title": title,
                    "hits": hits,
                },
            )
        )
    ms_graph = (time.time() - t0) * 1000.0
    return docs, ms_graph


def docs_to_context_append(docs: List[Any], graph_docs: List[Any]) -> str:
    parts = [d.page_content for d in docs[:K_CONTEXT]]
    parts.extend(d.page_content for d in graph_docs)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Token capture (Ollama prompt_eval_count/eval_count or LangChain usage)
# ---------------------------------------------------------------------------
def extract_usage(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort (prompt_tokens, completion_tokens); (None, None) otherwise."""
    if response is None:
        return None, None
    # Raw ollama-style dict response.
    if isinstance(response, dict):
        p = response.get("prompt_eval_count")
        c = response.get("eval_count")
        if p is not None or c is not None:
            return _as_int(p), _as_int(c)
    # LangChain AIMessage: .usage (newer), response_metadata, attributes.
    usage = getattr(response, "usage", None)
    if usage is None:
        md = getattr(response, "response_metadata", None)
        if isinstance(md, dict):
            usage = md.get("token_usage") or md.get("usage")
            if usage is None:
                # Some langchain_ollama builds leave prompt_eval_count here.
                p = md.get("prompt_eval_count")
                c = md.get("eval_count")
                if p is not None or c is not None:
                    return _as_int(p), _as_int(c)
    if usage is not None:
        if hasattr(usage, "prompt_tokens"):
            return _as_int(getattr(usage, "prompt_tokens", None)), _as_int(
                getattr(usage, "completion_tokens", None)
            )
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens", usage.get("prompt_eval_count"))
            c = usage.get("completion_tokens", usage.get("eval_count"))
            return _as_int(p), _as_int(c)
    # Attribute-style raw response.
    p = getattr(response, "prompt_eval_count", None)
    c = getattr(response, "eval_count", None)
    if p is not None or c is not None:
        return _as_int(p), _as_int(c)
    return None, None


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Phase grids
# ---------------------------------------------------------------------------
def combo_id(combo: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(combo.get("embedding")),
            str(combo.get("llm")),
            str(combo.get("prompt")),
            str(combo.get("temperature")),
            str(combo.get("strategy")),
            "rr=" + str(int(bool(combo.get("use_rerank")))),
            "g=" + str(int(bool(combo.get("graph_context")))),
        ]
    )


def _new_combo(**kwargs: Any) -> Dict[str, Any]:
    combo = {
        "embedding": DEFAULT_EMBEDDING,
        "llm": DEFAULT_LLM,
        "prompt": DEFAULT_PROMPT,
        "temperature": DEFAULT_TEMP,
        "strategy": "hybrid_0.5",
        "use_rerank": False,
        "graph_context": False,
    }
    combo.update(kwargs)
    combo["combo"] = combo_id(combo)
    return combo


def _latest_result_csv(fase: str, mode: str) -> Optional[Path]:
    files = sorted(RESULTS_DIR.glob("fase" + fase + "_" + mode + "_*.csv"))
    return files[-1] if files else None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def best_retrieval_from_phase(fase: str, mode: str) -> Optional[Tuple[str, str]]:
    """Best (embedding, strategy) by mean context_precision+context_recall."""
    path = _latest_result_csv(fase, mode)
    if path is None:
        return None
    rows = _read_csv_rows(path)
    if not rows:
        return None
    scores: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (row.get("embedding") or "", row.get("strategy") or "")
        vals = []
        for mk in ("context_precision", "context_recall"):
            raw = row.get(mk)
            if raw not in (None, "", "None"):
                try:
                    vals.append(float(raw))
                except ValueError:
                    pass
        if key[0] and key[1] and vals:
            scores.setdefault(key, []).extend(vals)
    if not scores:
        return None
    ranked = sorted(
        scores.items(), key=lambda kv: -(sum(kv[1]) / len(kv[1]))
    )
    return ranked[0][0]


def winner_strategies(fases: List[str], mode: str) -> List[str]:
    out: List[str] = []
    for fase in fases:
        best = best_retrieval_from_phase(fase, mode)
        if best and best[1] not in out:
            out.append(best[1])
    for strat in ("hybrid_0.5", "vector"):
        if strat not in out:
            out.append(strat)
    return out[:4]


def best_generation_from_phase(fase: str, mode: str) -> Optional[Tuple[str, str, float]]:
    """Best (llm, prompt, temperature) by mean faithfulness+answer_correctness."""
    path = _latest_result_csv(fase, mode)
    if path is None:
        return None
    rows = _read_csv_rows(path)
    if not rows:
        return None
    scores: Dict[Tuple[str, str, float], List[float]] = {}
    for row in rows:
        try:
            temp = float(row.get("temperature") or "")
        except ValueError:
            continue
        key = (row.get("llm") or "", row.get("prompt") or "", temp)
        vals = []
        for mk in ("faithfulness", "answer_correctness"):
            raw = row.get(mk)
            if raw not in (None, "", "None"):
                try:
                    vals.append(float(raw))
                except ValueError:
                    pass
        if key[0] and key[1] and vals:
            scores.setdefault(key, []).extend(vals)
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: -(sum(kv[1]) / len(kv[1])))
    return ranked[0][0]


def build_combos(fase: str, args: argparse.Namespace, mode: str) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    if fase == "A":
        # Retrieval only: 3 embeddings x 6 strategies; LLM/prompt/temp frozen.
        for emb in EMBEDDING_KEYS:
            for strat in STRATEGIES_A:
                combos.append(
                    _new_combo(
                        embedding=emb,
                        llm=DEFAULT_LLM,
                        prompt=DEFAULT_PROMPT,
                        temperature=DEFAULT_TEMP,
                        strategy=strat,
                        use_rerank=False,
                        graph_context=False,
                    )
                )
    elif fase == "B":
        # Freeze best of A (embedding + strategy); sweep 3x3x3 generation knobs.
        best = best_retrieval_from_phase("A", mode)
        emb = best[0] if best else DEFAULT_EMBEDDING
        strat = best[1] if best else (args.base_strategy or "hybrid_0.5")
        if args.base_strategy:
            strat = args.base_strategy
        from src.config import LLMS

        for llm in LLMS:
            for prompt in PROMPTS:
                for temp in TEMPERATURES:
                    combos.append(
                        _new_combo(
                            embedding=emb,
                            llm=llm,
                            prompt=prompt,
                            temperature=temp,
                            strategy=strat,
                            use_rerank=False,
                            graph_context=False,
                        )
                    )
    elif fase == "C":
        # Rerank on/off over A/B winner strategies; generation knobs frozen.
        strat_list = (
            [args.base_strategy] if args.base_strategy else winner_strategies(["A", "B"], mode)
        )
        gen = best_generation_from_phase("B", mode)
        llm = gen[0] if gen else DEFAULT_LLM
        prompt = gen[1] if gen else DEFAULT_PROMPT
        temp = gen[2] if gen else DEFAULT_TEMP
        for strat in strat_list:
            for rr in (False, True):
                combos.append(
                    _new_combo(
                        llm=llm,
                        prompt=prompt,
                        temperature=temp,
                        strategy=strat,
                        use_rerank=rr,
                        graph_context=False,
                    )
                )
    elif fase == "D":
        # Large models on B finalists; GPU-heavy (candidate for 2xA100).
        strat_list = (
            [args.base_strategy] if args.base_strategy else winner_strategies(["A", "B"], mode)
        )
        gen = best_generation_from_phase("B", mode)
        prompt = gen[1] if gen else DEFAULT_PROMPT
        temp = gen[2] if gen else DEFAULT_TEMP
        for strat in strat_list:
            for llm in LARGE_LLMS:
                combos.append(
                    _new_combo(
                        llm=llm,
                        prompt=prompt,
                        temperature=temp,
                        strategy=strat,
                        use_rerank=False,
                        graph_context=False,
                    )
                )
    elif fase == "G":
        # 4 combos: {vector | hybrid_0.5} x {graph off, graph on}; context-only.
        bases = [args.base_strategy] if args.base_strategy else list(BASE_STRATEGIES_G)
        if args.graph is True:
            graph_flags = [True]
        elif args.graph is False:
            graph_flags = [False]
        else:
            graph_flags = [False, True]
        gen = best_generation_from_phase("B", mode)
        llm = gen[0] if gen else DEFAULT_LLM
        prompt = gen[1] if gen else DEFAULT_PROMPT
        temp = gen[2] if gen else DEFAULT_TEMP
        for base in bases:
            for gflag in graph_flags:
                combos.append(
                    _new_combo(
                        llm=llm,
                        prompt=prompt,
                        temperature=temp,
                        strategy=base,
                        use_rerank=False,
                        graph_context=gflag,
                    )
                )
    else:
        raise ValueError("Unknown fase: " + fase)
    return combos


# ---------------------------------------------------------------------------
# RAGAS judge (local Ollama Qwen3-14B Q6_K, JSON)
# ---------------------------------------------------------------------------
def build_judge(ollama_base: str):
    """Return (metrics, run_config, judge_llm, judge_emb, raw_judge) or None."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
        from ragas import EvaluationDataset, evaluate as ragas_evaluate  # noqa: F401
        from ragas.dataset_schema import SingleTurnSample  # noqa: F401
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig
    except Exception as exc:
        print("  judge: ragas unavailable — metrics will be None (" + str(exc) + ")")
        return None

    class JSONCleaningLLM(LangchainLLMWrapper):
        def __call__(self, prompt, **kwargs):
            response = super().__call__(prompt, **kwargs)
            if isinstance(response, str):
                if response.startswith("```"):
                    response = response.split("\n", 1)[1]
                    if response.endswith("```"):
                        response = response.rsplit("\n", 1)[0]
                a, b = response.find("{"), response.rfind("}")
                if a >= 0 and b > a:
                    response = response[a:b + 1]
            return response

    raw_judge = ChatOllama(
        model=RAGAS_JUDGE,
        temperature=0.1,
        format="json",
        num_ctx=32768,
        think=False,
        base_url=ollama_base,
    )
    judge_llm = JSONCleaningLLM(raw_judge)
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu"},
        )
    )
    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerCorrectness(llm=judge_llm, embeddings=judge_emb),
    ]
    run_config = RunConfig(max_workers=1, max_retries=10, max_wait=120, timeout=900)
    return metrics, run_config, judge_llm, judge_emb, raw_judge


def run_judge_batch(samples, judge_bundle) -> Tuple[List[Dict[str, float]], float, Optional[int], Optional[int]]:
    """Evaluate one combo batch.

    Returns (per-sample metric dicts aligned with `samples`, total judge
    seconds, judge prompt tokens, judge completion tokens).
    """
    empty: List[Dict[str, float]] = [{} for _ in samples]
    if judge_bundle is None or not samples:
        return empty, 0.0, None, None
    metrics, run_config, judge_llm, judge_emb, raw_judge = judge_bundle
    try:
        from ragas import EvaluationDataset, evaluate as ragas_evaluate

        t0 = time.time()
        res = ragas_evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_emb,
            run_config=run_config,
        )
        elapsed = time.time() - t0
        # One metric dict PER sample (res.scores is aligned with the dataset).
        per_sample: List[Dict[str, float]] = []
        try:
            scores = res.scores
            for idx in range(len(samples)):
                metric_dict: Dict[str, float] = {}
                raw = scores[idx] if idx < len(scores) else {}
                for k, v in dict(raw).items():
                    mk = str(k).lower()
                    if mk in METRIC_KEYS:
                        try:
                            metric_dict[mk] = float(v)
                        except (TypeError, ValueError):
                            pass
                per_sample.append(metric_dict)
        except Exception:
            per_sample = [{} for _ in samples]
        # Token usage for the judge: RAGAS consumes LangChain messages and does
        # not surface usage; leave None unless the response carried it.
        p_tok, c_tok = extract_usage(getattr(res, "raw_response", None))
        return per_sample, elapsed, p_tok, c_tok
    except Exception as exc:
        print("  judge error: " + str(exc))
        return [{} for _ in samples], 0.0, None, None


# ---------------------------------------------------------------------------
# Row building / CSV IO
# ---------------------------------------------------------------------------
def build_row(
    fase: str,
    combo: Dict[str, Any],
    item: Dict[str, Any],
    answer: str,
    docs: List[Any],
    context: str,
    t_ret: float,
    t_gen: float,
    t_judge: float,
    gen_usage: Tuple[Optional[int], Optional[int]],
    judge_usage: Tuple[Optional[int], Optional[int]],
    metrics: Optional[Dict[str, float]],
    error: str,
) -> Dict[str, Any]:
    p_gen, c_gen = gen_usage
    p_judge, c_judge = judge_usage
    total_tokens = None
    known = [v for v in (p_gen, c_gen, p_judge, c_judge) if v is not None]
    if known:
        total_tokens = sum(known)
    row = {
        "fase": fase,
        "combo": combo["combo"],
        "embedding": combo["embedding"],
        "llm": combo["llm"],
        "prompt": combo["prompt"],
        "temperature": combo["temperature"],
        "strategy": combo["strategy"],
        "use_rerank": combo["use_rerank"],
        "graph_context": combo["graph_context"],
        "question": item.get("question"),
        "ground_truth": item.get("ground_truth"),
        "tipo": item.get("tipo"),
        "idioma": item.get("idioma"),
        "answer": answer,
        "n_docs": len(docs),
        "filenames": ";".join(
            str(d.metadata.get("filename") or d.metadata.get("title") or "N/A")
            for d in docs
        ),
        "context_chars": len(context),
        "latency_retrieval_s": round(t_ret, 4),
        "latency_gen_s": round(t_gen, 4),
        "latency_judge_s": round(t_judge, 4),
        "prompt_tokens_gen": p_gen,
        "completion_tokens_gen": c_gen,
        "prompt_tokens_judge": p_judge,
        "completion_tokens_judge": c_judge,
        "total_tokens": total_tokens,
        "faithfulness": (metrics or {}).get("faithfulness"),
        "context_precision": (metrics or {}).get("context_precision"),
        "context_recall": (metrics or {}).get("context_recall"),
        "answer_correctness": (metrics or {}).get("answer_correctness"),
        "error": error,
    }
    return row


def append_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Single (combo, item) execution
# ---------------------------------------------------------------------------
def run_one(
    fase: str,
    combo: Dict[str, Any],
    item: Dict[str, Any],
    w_client,
    ollama_base: str,
) -> Tuple[Dict[str, Any], Any]:
    """Run retrieval+generation for one combo/question.

    Returns (row_without_judge, sample_or_None_for_ragas).
    """
    from langchain_core.documents import Document

    question = item.get("question") or ""
    error = ""
    answer = ""
    docs: List[Any] = []
    context = ""
    t_ret = 0.0
    t_gen = 0.0
    gen_usage: Tuple[Optional[int], Optional[int]] = (None, None)
    ms_graph = 0.0

    try:
        from src.config import collection_name

        embedding = load_embeddings(combo["embedding"])
        collection = collection_name(combo["embedding"])
        col = w_client.collections.get(collection)
        mq_llm = get_mq_llm(ollama_base) if combo["strategy"].startswith("mq_") else None

        t0 = time.time()
        docs = retrieve_docs(
            combo["strategy"],
            question,
            combo["embedding"],
            col,
            embedding,
            mq_llm,
            bool(combo["use_rerank"]),
        )
        graph_docs: List[Any] = []
        if combo["graph_context"]:
            graph_docs, ms_graph = sparql_biblio(question, GRAPH_TOP_N)
        t_ret = time.time() - t0
        # ms_graph is a retrieval-phase latency; SPARQL adds no tokens.

        context = docs_to_context_append(docs, graph_docs) if combo["graph_context"] else "\n\n".join(
            d.page_content for d in docs[:K_CONTEXT]
        )

        from src.langfuse_monitor import get_prompt

        prompt_tmpl = get_prompt(combo["prompt"])
        messages = prompt_tmpl.format_messages(context=context, question=question)
        gen_llm = get_gen_llm(combo["llm"], combo["temperature"], ollama_base)
        t1 = time.time()
        out = gen_llm.invoke(messages)
        t_gen = time.time() - t1
        answer = out.content if hasattr(out, "content") else str(out)
        gen_usage = extract_usage(out)
        if gen_usage == (None, None):
            gen_usage = extract_usage(getattr(out, "response_metadata", None))
    except Exception as exc:
        error = str(exc)
        print(
            "  ERROR "
            + combo["combo"]
            + " / "
            + question[:50]
            + ": "
            + error
        )

    sample = None
    if not error:
        try:
            from ragas.dataset_schema import SingleTurnSample

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                reference=item.get("ground_truth"),
                retrieved_contexts=[d.page_content[:800] for d in docs[:K_CONTEXT]],
            )
        except Exception:
            sample = None

    row = build_row(
        fase=fase,
        combo=combo,
        item=item,
        answer=answer,
        docs=docs,
        context=context,
        t_ret=t_ret,
        t_gen=t_gen,
        t_judge=0.0,
        gen_usage=gen_usage,
        judge_usage=(None, None),
        metrics=None,
        error=error,
    )
    row["_ms_graph"] = ms_graph
    return row, sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one evaluation phase (A/B/C/D/G) against the persistent "
            "Weaviate+Ollama services. Never starts services."
        )
    )
    parser.add_argument(
        "--fase",
        required=True,
        choices=["A", "B", "C", "D", "G"],
        help="Evaluation phase to run (E is optional and manual).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use only the first SMOKE_N questions (default 3).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to the newest CSV for this fase+mode, skipping rows whose "
            "(combo, question) key already exists."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "ThreadPoolExecutor workers (default 1). Values >1 are "
            "EXPERIMENTAL: Ollama serializes requests anyway and the Weaviate "
            "client is not proven thread-safe."
        ),
    )
    parser.add_argument(
        "--kill-residual-weaviate",
        action="store_true",
        help=(
            "DANGEROUS opt-in: pkill local weaviate ONLY if WEAVIATE_PID in "
            "services.env is stale on THIS node. The runner never scancels "
            "jobs; prefer scancel of the residual non-persistent job manually."
        ),
    )
    parser.add_argument(
        "--base-strategy",
        choices=["vector", "hybrid_0.5"],
        default=None,
        help=(
            "Baseline strategy for phases B/C/D/G freeze (default: winner from "
            "previous phase CSVs, else hybrid_0.5). For G it selects one of "
            "the two baselines."
        ),
    )
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--graph",
        dest="graph",
        action="store_true",
        default=None,
        help="Phase G: only graph-on combos.",
    )
    graph_group.add_argument(
        "--no-graph",
        dest="graph",
        action="store_false",
        help="Phase G: only graph-off combos.",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip the Weaviate ready check (for debugging only).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 2

    mode = "smoke" if args.smoke else "full"
    services = load_services_env()
    weaviate_url = resolve_weaviate_url(services)
    ollama_base = resolve_ollama_base(services)
    print("fase        :", args.fase, "| mode:", mode)
    print("weaviate    :", weaviate_url)
    print("ollama      :", ollama_base)

    # Health guard: exactly one Weaviate must be ready (persistent services).
    if not args.skip_health:
        if not weaviate_ready(weaviate_url):
            if args.kill_residual_weaviate:
                kill_residual_weaviate(services)
                if not weaviate_ready(weaviate_url):
                    print_remediation(weaviate_url)
                    return 3
            else:
                print_remediation(weaviate_url)
                return 3
        else:
            print("weaviate    : READY")
    else:
        print("weaviate    : health check SKIPPED (--skip-health)")

    items = load_items(smoke=args.smoke)
    print("questions   :", len(items))
    combos = build_combos(args.fase, args, mode)
    print("combos      :", len(combos))

    # CSV path: --resume appends to the newest file for this fase+mode.
    out_path: Optional[Path] = None
    done_keys = set()
    if args.resume:
        out_path = _latest_result_csv(args.fase, mode)
        if out_path is not None:
            for row in _read_csv_rows(out_path):
                done_keys.add((row.get("combo") or "", row.get("question") or ""))
            print("resume      :", out_path, "| done keys:", len(done_keys))
        else:
            print("resume      : no existing CSV — starting a new one")
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / (
            "fase" + args.fase + "_" + mode + "_" + ts + ".csv"
        )
    print("csv         :", out_path)

    try:
        w_client = connect_weaviate_client(weaviate_url)
        print("connected   : weaviate is_ready =", w_client.is_ready())
    except Exception as exc:
        print("ERROR: Weaviate connect failed: " + str(exc), file=sys.stderr)
        return 4

    # Optional Langfuse import: never required for evaluation.
    try:
        import langfuse  # noqa: F401
    except Exception:
        pass

    judge_bundle = build_judge(ollama_base)
    print("judge       :", RAGAS_JUDGE if judge_bundle else "DISABLED (ragas missing)")

    jobs = []
    for combo in combos:
        for item in items:
            key = (combo["combo"], item.get("question") or "")
            if key in done_keys:
                continue
            jobs.append((combo, item))
    print("to run      :", len(jobs), "(skipped", len(done_keys), "resumed)")

    def _task(job):
        combo, item = job
        return run_one(args.fase, combo, item, w_client, ollama_base)

    # Group results per combo so the judge runs in batch (as in the notebook).
    results_by_combo: Dict[str, List[Tuple[Dict[str, Any], Any, Dict[str, Any]]]] = {}
    # Default workers=1 keeps everything serial (safe path).
    if args.workers == 1:
        for job in jobs:
            row, sample = _task(job)
            results_by_combo.setdefault(job[0]["combo"], []).append(
                (job[0], sample, row)
            )
    else:
        print(
            "WARN: --workers "
            + str(args.workers)
            + " is experimental (Ollama serializes; client thread-safety unproven)"
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # _task returns (row, sample); store as (combo, sample, row).
            for (job, out) in zip(jobs, pool.map(_task, jobs)):
                results_by_combo.setdefault(job[0]["combo"], []).append(
                    (job[0], out[1], out[0])
                )

    total_rows = 0
    for combo_key, entries in results_by_combo.items():
        # entries: (combo, sample, row)
        samples = []
        sample_rows = []
        for combo, sample, row in entries:
            if sample is not None:
                samples.append(sample)
                sample_rows.append(row)
        metrics_list: List[Dict[str, float]] = [{} for _ in entries]
        t_judge = 0.0
        judge_usage = (None, None)
        if samples:
            batch_metrics, t_judge, jp, jc = run_judge_batch(samples, judge_bundle)
            judge_usage = (jp, jc)
            # Map per-sample metrics back onto their rows (identity of row object).
            row_to_metrics = {}
            for row, metric_dict in zip(sample_rows, batch_metrics):
                row_to_metrics[id(row)] = metric_dict
            metrics_list = [
                row_to_metrics.get(id(row), {}) for combo, sample, row in entries
            ]
            # Judge latency is reported as the per-sample average of the batch.
            t_judge = t_judge / max(len(samples), 1)
        rows_out = []
        for (combo, sample, row), metric_dict in zip(entries, metrics_list):
            row.pop("_ms_graph", None)
            if sample is not None and metric_dict:
                row.update(metric_dict)
            row["latency_judge_s"] = round(t_judge, 4)
            row["prompt_tokens_judge"] = judge_usage[0]
            row["completion_tokens_judge"] = judge_usage[1]
            # Recompute total_tokens after judge fill-in.
            known = [
                row.get("prompt_tokens_gen"),
                row.get("completion_tokens_gen"),
                row.get("prompt_tokens_judge"),
                row.get("completion_tokens_judge"),
            ]
            known = [v for v in known if v is not None]
            row["total_tokens"] = sum(known) if known else None
            rows_out.append(row)
        append_rows(out_path, rows_out)
        total_rows += len(rows_out)
        print(
            "combo done  : "
            + combo_key
            + " | rows "
            + str(len(rows_out))
            + " | judge_ms/sample "
            + str(round(t_judge * 1000, 1))
        )

    try:
        w_client.close()
    except Exception:
        pass

    print("DONE: fase", args.fase, "| rows written:", total_rows, "|", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
