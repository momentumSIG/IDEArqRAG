"""Langfuse monitor for RAG-IDEArq (Langfuse 4.x API)."""
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Langfuse SDK imports
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

# Prompt source: "local" (default, frozen data/prompts.json) or "langfuse".
# Any other value behaves as "local".
PROMPT_SOURCE = os.getenv("PROMPT_SOURCE", "local").strip().lower()

_LANGFUSE_URL = os.getenv("LANGFUSE_BASE_URL", "http://localhost:4000")
_LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
_LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

_auth = HTTPBasicAuth(_LANGFUSE_PUBLIC_KEY, _LANGFUSE_SECRET_KEY)

# Module-level cache so the Langfuse health probe runs at most once per process.
_langfuse_reachable_cached = None


def _langfuse_reachable(timeout: float = 1.5) -> bool:
    """Return True if the Langfuse health endpoint answers 2xx.

    The first probe result is cached at module level. Any exception or
    non-2xx response counts as unreachable (returns False).
    """
    global _langfuse_reachable_cached
    if _langfuse_reachable_cached is not None:
        return _langfuse_reachable_cached
    try:
        from src import config

        base = str(config.LANGFUSE_BASE_URL).rstrip("/")
        resp = requests.get(f"{base}/api/public/health", timeout=timeout)
        reachable = 200 <= resp.status_code < 300
    except Exception:
        reachable = False
    _langfuse_reachable_cached = reachable
    return reachable


try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    class _NoopCallback(BaseCallbackHandler):
        """No-op LangChain callback handler used when Langfuse is unavailable."""

except Exception:  # langchain_core unavailable: never fail the import

    class _NoopCallback:
        """Fallback no-op handler when langchain_core is unavailable."""

        def on_llm_start(self, *args, **kwargs):
            return None

        def on_llm_new_token(self, *args, **kwargs):
            return None

        def on_llm_end(self, *args, **kwargs):
            return None

        def on_llm_error(self, *args, **kwargs):
            return None

        def on_chain_start(self, *args, **kwargs):
            return None

        def on_chain_end(self, *args, **kwargs):
            return None

        def on_chain_error(self, *args, **kwargs):
            return None

        def on_tool_start(self, *args, **kwargs):
            return None

        def on_tool_end(self, *args, **kwargs):
            return None

        def on_tool_error(self, *args, **kwargs):
            return None


# Langfuse client: None when Langfuse is unreachable or construction fails.
try:
    langfuse_client = Langfuse(
        secret_key=_LANGFUSE_SECRET_KEY,
        public_key=_LANGFUSE_PUBLIC_KEY,
        host=_LANGFUSE_URL,
    ) if _langfuse_reachable() else None
except Exception:
    langfuse_client = None

# Global CallbackHandler for automatic LangChain tracing.
# Falls back to a no-op handler so importing this module never fails.
try:
    langfuse_handler = CallbackHandler() if _langfuse_reachable() else _NoopCallback()
except Exception:
    langfuse_handler = _NoopCallback()


def get_callback(session_id=None, tags=None, trace_id=None, input_data=None, output_data=None):
    """Create a Langfuse trace via REST API and return trace info.
    
    Args:
        session_id: Session identifier (e.g., combo label)
        tags: List of tags for the trace
        trace_id: Optional pre-generated trace ID
        input_data: Optional input data to include in the trace
        output_data: Optional output data to include in the trace
    
    Returns:
        dict with trace_id and session info for later scoring
    """
    import uuid
    tid = trace_id or str(uuid.uuid4()).replace("-", "")
    
    payload = {
        "id": tid,
        "name": "rag_evaluation",
        "sessionId": session_id,
        "tags": tags or [],
    }
    
    if input_data:
        payload["input"] = input_data
    if output_data:
        payload["output"] = output_data
    
    try:
        resp = requests.post(
            f"{_LANGFUSE_URL}/api/public/traces",
            auth=_auth,
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            return {"trace_id": tid, "session_id": session_id, "tags": tags or []}
    except Exception:
        pass
    
    # Fallback: return trace_id even if creation failed
    return {"trace_id": tid, "session_id": session_id, "tags": tags or []}


def update_trace_output(trace_id, output_data):
    """Update a Langfuse trace with output data."""
    if not trace_id:
        return
    
    payload = {
        "output": output_data,
    }
    
    try:
        requests.patch(
            f"{_LANGFUSE_URL}/api/public/traces/{trace_id}",
            auth=_auth,
            json=payload,
            timeout=10,
        )
    except Exception as e:
        print(f"[Langfuse] Error updating trace output: {e}")


def log_retrieval_breakdown(trace_id, question, weaviate_docs, geojson_docs=0, metadata=None):
    """Log retrieval breakdown event to Langfuse."""
    if not trace_id:
        return
    
    payload = {
        "traceContext": {"traceId": trace_id},
        "name": "retrieval_breakdown",
        "input": {"question": question},
        "output": {
            "weaviate_docs": weaviate_docs,
            "geojson_docs": geojson_docs,
            "total_docs": weaviate_docs + geojson_docs,
        },
        "metadata": metadata or {},
    }
    
    try:
        requests.post(
            f"{_LANGFUSE_URL}/api/public/observations",
            auth=_auth,
            json=payload,
            timeout=10,
        )
    except Exception as e:
        print(f"[Langfuse] Error logging retrieval breakdown: {e}")


def score_trace(trace_id, name, value):
    """Score a trace in Langfuse via REST API."""
    if not trace_id:
        return
    
    payload = {
        "traceId": trace_id,
        "name": name,
        "value": value,
        "dataType": "NUMERIC",
    }
    
    try:
        resp = requests.post(
            f"{_LANGFUSE_URL}/api/public/scores",
            auth=_auth,
            json=payload,
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[Langfuse] Score error {name}: {resp.text[:200]}")
    except Exception as e:
        print(f"[Langfuse] Error scoring {name}: {e}")


def get_prompt(prompt_key: str):
    """Return the prompt template for ``prompt_key``.

    With PROMPT_SOURCE=local (default, or any other unrecognized value) the
    frozen template from data/prompts.json is returned with no network call.
    With PROMPT_SOURCE=langfuse the prompt is fetched from Langfuse; if the
    fetch fails, one warning is logged and the local template is returned.
    """
    from src.config import PROMPT_NAMES
    from src.prompts_loader import get_local_prompt

    # Accept both short keys ("zero_shot") and full artifact names.
    name = PROMPT_NAMES.get(prompt_key, prompt_key)

    if PROMPT_SOURCE != "langfuse":
        return get_local_prompt(name)

    try:
        prompt = langfuse_client.get_prompt(name=name, type="chat")
        return prompt
    except Exception:
        try:
            prompt = langfuse_client.get_prompt(name=name, type="text")
            return prompt
        except Exception as exc:
            print(
                f"[Langfuse] Warning: prompt fetch failed for {name!r}; "
                f"falling back to local prompt: {exc}"
            )
            return get_local_prompt(name)


def get_or_create_dataset_v3(questions, ground_truths, sources=None):
    """Create RAG-IDEArq-eval-v3 with 30 canonical questions via REST API."""
    from src.config import LANGFUSE_DATASET_NAME
    
    # Check if dataset exists
    try:
        resp = requests.get(
            f"{_LANGFUSE_URL}/api/public/datasets/{LANGFUSE_DATASET_NAME}",
            auth=_auth,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    
    # Create dataset
    try:
        resp = requests.post(
            f"{_LANGFUSE_URL}/api/public/datasets",
            auth=_auth,
            json={
                "name": LANGFUSE_DATASET_NAME,
                "description": "IDEArq RAG eval v3 — 32 preguntas (17 simples + 15 complejas)",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[Langfuse] Dataset create error: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"[Langfuse] Dataset create error: {e}")
        return None
    
    # Add items
    if sources is None:
        sources = ["manual_review"] * len(questions)
    
    for q, gt, src in zip(questions, ground_truths, sources):
        try:
            requests.post(
                f"{_LANGFUSE_URL}/api/public/dataset-items",
                auth=_auth,
                json={
                    "datasetName": LANGFUSE_DATASET_NAME,
                    "input": {"question": q},
                    "expectedOutput": {"ground_truth": gt, "source": src},
                },
                timeout=10,
            )
        except Exception as e:
            print(f"[Langfuse] Dataset item error: {e}")
    
    return {"name": LANGFUSE_DATASET_NAME}
