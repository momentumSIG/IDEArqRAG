"""Q1 probe: furthest Chalcolithic site from the city of Jaen, with the province
capital coordinates table (Nomenclator Basico IGN) enabled.

Runs Route B (``src.geo_route.retrieve_filtered``) against the live collection and
asks the local Qwen3-4B to answer from the retrieved context. Read-only except for
one JSON dump under ``data/results/``.

Usage (from the repo root):

    set -a; . /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/services.env; set +a
    GEO_USE_CITY_COORDS=1 GEO_CITY_COORDS_FILE="$PWD/data/geo/capitales_provincias.csv" \
        /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/env_rag/bin/python \
        scripts/q1_jaen_probe_qwen3.py

The query PDF vector leg needs an embedder registered with ``set_embedding``; this
probe does not set one, so that leg degrades (its error is recorded in the output)
and the BM25 leg still runs. The structured GeoJSON rows and the precomputed
distances - the part Q1 answers from - are unaffected.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.geo_route as geo  # noqa: E402

QUESTION = ("Which Chalcolithic site is the furthest from the city of Jaén, "
            "in the province of Jaén?")
GOLD = "The Eras del Alcázar site, situated about 50 km from the city of Jaén."
COLLECTION = "IdearqGTE_800_50_v2"
MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0"
OUT_JSON = REPO / "data" / "results" / "q1_jaen_probe_20261015.json"
OLLAMA = (os.environ.get("GEO_OLLAMA_URL") or os.environ.get("OLLAMA_URL")
          or "http://c206-10:11434")


def ask_ollama(prompt: str, timeout_s: int = 900) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 1200,
                    "num_thread": 16},
    }
    req = urllib.request.Request(
        OLLAMA.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message") or {}).get("content", "")


def main() -> int:
    t0 = time.time()
    result = geo.retrieve_filtered(QUESTION, collection=COLLECTION)
    if result is None:
        print("retrieve_filtered -> None (question did not route to Route B)")
        return 1
    ref = geo._city_in_question(QUESTION, (result.plan or {}).get("region"))
    sites = []
    for doc in result.structured_docs:
        meta = dict(getattr(doc, "metadata", {}) or {})
        sites.append({
            "yacimiento_id": meta.get("yacimiento_id"),
            "nombre": meta.get("yacimiento_nombre"),
            "distancia_km": meta.get("_distance_km"),
            "referencia": meta.get("_distance_from"),
            "bp_max": meta.get("bp_max"),
            "periodo": meta.get("periodo"),
            "region": meta.get("region"),
        })
    print("plan:", json.dumps(result.plan, ensure_ascii=False))
    print("referencia resuelta:", repr(ref))
    print("n_candidates:", result.n_candidates, "| n_sites:", result.n_sites,
          "| n_pdf_docs:", result.n_pdf_docs, "| degraded:", result.degraded)
    for w in result.warnings:
        print("WARN:", w)
    for e in result.errors:
        print("ERR:", e)
    for s in sites:
        print("SITE:", json.dumps(s, ensure_ascii=False))

    prompt = (
        "Answer the question using ONLY the context below. Be concise.\n\n"
        "Context:\n" + result.context + "\n\nQuestion: " + QUESTION + "\nAnswer:"
    )
    t1 = time.time()
    answer = ask_ollama(prompt)
    t2 = time.time()

    payload = {
        "question": QUESTION,
        "gold": GOLD,
        "collection": COLLECTION,
        "model": MODEL,
        "env": {
            "GEO_USE_CITY_COORDS": os.environ.get("GEO_USE_CITY_COORDS"),
            "GEO_CITY_COORDS_FILE": os.environ.get("GEO_CITY_COORDS_FILE"),
            "GEO_WEAVIATE_URL": os.environ.get("GEO_WEAVIATE_URL"),
        },
        "reference": list(ref) if ref and ref[0] else None,
        "plan": result.plan,
        "sites": sites,
        "warnings": result.warnings,
        "errors": result.errors,
        "n_candidates": result.n_candidates,
        "n_sites": result.n_sites,
        "n_pdf_docs": result.n_pdf_docs,
        "context": result.context,
        "answer": answer,
        "timings_s": {"retrieval": round(t1 - t0, 1), "generation": round(t2 - t1, 1)},
        "context_chars": len(result.context),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("ANSWER:\n" + answer)
    print("saved:", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
