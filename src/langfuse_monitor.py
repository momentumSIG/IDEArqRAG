"""Langfuse monitor for RAG-IDEArq (Langfuse 4.x API)."""
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Langfuse SDK imports
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

_LANGFUSE_URL = os.getenv("LANGFUSE_BASE_URL", "http://localhost:4000")
_LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
_LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

_auth = HTTPBasicAuth(_LANGFUSE_PUBLIC_KEY, _LANGFUSE_SECRET_KEY)

# Langfuse client
langfuse_client = Langfuse(
    secret_key=_LANGFUSE_SECRET_KEY,
    public_key=_LANGFUSE_PUBLIC_KEY,
    host=_LANGFUSE_URL,
)

# Global CallbackHandler for automatic LangChain tracing
langfuse_handler = CallbackHandler()


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
    """Read prompt from Langfuse SDK (still works for prompts)."""
    from src.config import PROMPT_NAMES
    
    name = PROMPT_NAMES[prompt_key]
    try:
        prompt = langfuse_client.get_prompt(name=name, type="chat")
        return prompt
    except Exception:
        try:
            prompt = langfuse_client.get_prompt(name=name, type="text")
            return prompt
        except Exception:
            raise


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
                "description": "IDEArq RAG eval v3 — 30 preguntas (15 simples + 15 complejas)",
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
