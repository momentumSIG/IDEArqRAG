"""Resolve a reachable Weaviate endpoint regardless of which node the client runs on."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import requests
import weaviate

SERVICES_ENV = Path(
    "/mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/services.env"
)


def _parse_host(value: str | None) -> str | None:
    """Extract a hostname from a URL or bare host string; return None if unusable."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    candidate = value if "://" in value else f"http://{value}"
    try:
        parsed = urlparse(candidate)
    except Exception:
        return None
    return parsed.hostname or None


def _services_env_url() -> str | None:
    """Read WEAVIATE_URL or WEAVIATE_HOST from services.env if it exists.

    Never raises; returns None when the file is missing or has no usable value.
    """
    try:
        if not SERVICES_ENV.is_file():
            return None
        found: dict[str, str] = {}
        for line in SERVICES_ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key not in ("WEAVIATE_URL", "WEAVIATE_HOST"):
                continue
            value = value.strip().strip('"').strip("'")
            if value:
                found[key] = value
        return found.get("WEAVIATE_URL") or found.get("WEAVIATE_HOST")
    except Exception:
        return None


def _env_url() -> str | None:
    """Read WEAVIATE_URL or WEAVIATE_HOST from the process environment.

    Never raises; returns None when neither variable is set.
    """
    try:
        return (
            os.environ.get("WEAVIATE_URL")
            or os.environ.get("WEAVIATE_HOST")
            or None
        )
    except Exception:
        return None


def _candidate_hosts() -> list[str]:
    """Return ordered, de-duplicated candidate hosts to try.

    Order: ``localhost``, ``127.0.0.1``, the host from services.env, then the
    host from ``WEAVIATE_URL``/``WEAVIATE_HOST`` in the environment. Values are
    parsed with ``urllib.parse.urlparse`` (accepting ``http://host:8080`` and
    bare ``host``); empties and duplicates are dropped.
    """
    ordered: list[str] = []
    for raw in ("localhost", "127.0.0.1", _services_env_url(), _env_url()):
        host = _parse_host(raw)
        if host and host not in ordered:
            ordered.append(host)
    return ordered


def _is_ready(host: str, port: int = 8080, timeout: float = 1.5) -> bool:
    """Return True when the Weaviate ready endpoint answers with a 2xx status."""
    try:
        resp = requests.get(
            f"http://{host}:{port}/v1/.well-known/ready",
            timeout=(timeout, timeout),
        )
        return 200 <= resp.status_code < 300
    except Exception:
        return False


def connect_weaviate(headers: dict | None = None) -> weaviate.WeaviateClient:
    """Connect to the first reachable Weaviate candidate host.

    Iterates :func:`_candidate_hosts` in order. Each host that is ready is
    passed to ``weaviate.connect_to_local``; a failed connect is recorded and
    the next candidate is tried.

    Args:
        headers: Optional headers passed through verbatim to
            ``connect_to_local`` (e.g. an API-key header built by the caller).
            This module does not invent auth handling.

    Returns:
        A connected ``WeaviateClient``.

    Raises:
        RuntimeError: If no candidate host worked. The message lists every
            host attempted and the reason per host.
    """
    reasons: list[str] = []
    candidates = _candidate_hosts()
    for host in candidates:
        if not _is_ready(host):
            reasons.append(f"{host}: not ready (ready endpoint did not return 2xx)")
            continue
        try:
            return weaviate.connect_to_local(
                host=host,
                port=8080,
                grpc_port=50051,
                headers=headers,
            )
        except Exception as exc:
            reasons.append(f"{host}: connect_to_local failed: {exc}")
    if not reasons:
        reasons.append("no candidate hosts to try")
    raise RuntimeError(
        "Weaviate connection failed for every candidate host: " + "; ".join(reasons)
    )
