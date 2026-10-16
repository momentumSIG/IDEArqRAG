#!/usr/bin/env python3
"""A/B smoke: the same five graph-only questions answered twice.

Side A — the RDF graph (SPARQL, deterministic). This is the reference answer
and it is the very same check that `scripts/smoke_graph.py` runs.

Side B — the Weaviate retrieval route used by the RAG pipeline:
  1. a BM25 probe asking the index directly for each answer literal, and
  2. the normal hybrid retrieval for the question, with an optional LLM answer.

The five questions were picked because their answers live ONLY in the graph —
not in the PDF corpus and not in the C14 geojson. Running them through
Weaviate measures how much of the graph's knowledge the vector index can
reproduce, which is the evidence needed to decide whether the graph layer
earns its keep.

The verdict is printed as a table and written to
`data/results/smoke_ab_graph_weaviate_<ts>.{json,md}`.

Exit code: 0 when side A passes all five checks and side B ran without
transport errors, 1 otherwise. Side B *failing to find* an answer is a
finding, not an error: it never flips the exit code.

Usage:
    python3 scripts/smoke_weaviate_ab.py
    python3 scripts/smoke_weaviate_ab.py --no-llm
    python3 scripts/smoke_weaviate_ab.py --collection IdearqGTE_800_50_v2 \
        --embedding gte-multilingual-base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import smoke_graph  # noqa: E402  (same directory: scripts/smoke_graph.py)
import run_fase  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COLLECTION = "IdearqMiniLM_800_50_v2"
DEFAULT_EMBEDDING = "all-MiniLM-L6-v2"
DEFAULT_ALPHA = 0.5
DEFAULT_K = 10
PROBE_LIMIT = 50

ANSWER_PROMPT = (
    "Responde a la siguiente pregunta usando SOLO el contexto. "
    "Si el contexto no contiene la respuesta, di exactamente: "
    "'No consta en el contexto'.\n\n"
    "Contexto:\n{context}\n\nPregunta: {question}\n\nRespuesta:"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def normalise_literal(value: str) -> str:
    """URIs collapse to their last segment so '.../bibo/Book' probes 'Book'."""
    value = str(value).strip()
    if value.startswith("http://") or value.startswith("https://"):
        value = value.rstrip("/").rsplit("/", 1)[-1]
    return value


def probe_literals(check: dict) -> List[str]:
    seen, out = set(), []
    for raw in list(check.get("expected", [])) + list(check.get("graph_only_literals", [])):
        lit = normalise_literal(raw)
        if len(lit) >= 3 and lit not in seen:
            seen.add(lit)
            out.append(lit)
    return out


def contains(text: str, needle: str) -> bool:
    return needle.lower() in (text or "").lower()


def docs_text(docs) -> str:
    return "\n".join(d.page_content for d in docs)


def bm25_probe(col, literal: str) -> Dict[str, Any]:
    """Ask the index for `literal` and confirm a retrieved chunk holds it.

    BM25 is tokenised, so a hit only counts when the verbatim literal is in
    the returned text. AND first (all tokens must match), OR as a fallback;
    both capped at PROBE_LIMIT documents. weaviate 4.23 wants a
    BM25Operator object here — a bare "AND" string is rejected.
    """
    from weaviate.classes.query import BM25Operator

    hits: List[Dict[str, Any]] = []
    attempts = []
    for label, operator in (
        ("AND", BM25Operator.and_()),
        ("OR", BM25Operator.or_(minimum_match=1)),
    ):
        try:
            res = col.query.bm25(
                query=literal,
                operator=operator,
                limit=PROBE_LIMIT,
                query_properties=run_fase.BM25_PROPS,
                return_metadata=["score"],
                return_properties=["filename", "content"],
            )
        except Exception as exc:  # operator unsupported -> report, keep going
            attempts.append(f"{label}: error {exc}")
            continue
        objs = res.objects or []
        # A literal counts as found when it is verbatim in a chunk OR when every
        # token of it is in that same chunk. The exact-substring test alone was a
        # bug: a multi-word window taken from a document is almost never a
        # contiguous substring of it (words with short/other words in between),
        # so the positive control reported SOSPECHOSO while BM25 was fine.
        tokens = [t for t in literal.split() if len(t) >= 3]

        def _holds(chunk: str) -> bool:
            if contains(chunk, literal):
                return True
            low = (chunk or "").lower()
            return bool(tokens) and all(t.lower() in low for t in tokens)

        matched = [o for o in objs if _holds(o.properties.get("content", ""))]
        attempts.append(f"{label}: {len(objs)} returned, {len(matched)} hold it")
        for o in matched[:3]:
            hits.append(
                {
                    "filename": o.properties.get("filename"),
                    "score": getattr(o.metadata, "score", None),
                }
            )
        if matched:
            break
    return {"found": bool(hits), "hits": hits, "attempts": attempts}


def self_check_probe(col) -> Dict[str, Any]:
    """Positive control: probe a literal taken from a document we just
    retrieved. If the probe cannot find text the index returned, the probe
    itself is broken and every 'not found' below must be read with care."""
    try:
        res = col.query.bm25(
            query="arqueología",
            limit=5,
            query_properties=run_fase.BM25_PROPS,
            return_metadata=["score"],
            return_properties=["filename", "content"],
        )
        objs = res.objects or []
        if not objs:
            return {"ok": False, "detail": "control query returned 0 documents"}
        content = objs[0].properties.get("content", "")
        words = [w for w in content.split() if len(w) > 5][:6]
        if not words:
            return {"ok": False, "detail": "control document had no usable words"}
        window = " ".join(words)
        check = bm25_probe(col, window)
        return {"ok": check["found"], "detail": f"window={window!r} -> {check['attempts']}"}
    except Exception as exc:
        return {"ok": False, "detail": f"control raised {exc}"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--embedding", default=DEFAULT_EMBEDDING)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="hybrid alpha")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="docs in the context")
    p.add_argument("--llm", default=run_fase.DEFAULT_LLM, help="LLM key from src.config.LLMS")
    p.add_argument("--temperature", type=float, default=run_fase.DEFAULT_TEMP)
    p.add_argument("--no-llm", action="store_true", help="skip the generation step")
    p.add_argument(
        "--graph",
        type=Path,
        default=smoke_graph.DEFAULT_GRAPH,
        help="RDF graph (.nt) for side A",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="extracted PDF text dir for the 'is it even in the corpus' scan",
    )
    p.add_argument("--out-prefix", default="smoke_ab_graph_weaviate")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()
    errors: List[str] = []

    # --- side A: the graph ------------------------------------------------
    graph = smoke_graph.load_graph(args.graph)
    triples = len(graph)
    a_rows = []
    for check in smoke_graph.CHECKS:
        ok, got = smoke_graph.run_check(graph, check)
        a_rows.append({"id": check["id"], "ok": ok, "got": got})
        if not ok:
            errors.append(f"side A {check['id']} failed: got {got}")

    # --- transport: services + index --------------------------------------
    services = run_fase.load_services_env()
    for key in ("WEAVIATE_URL", "OLLAMA_URL"):
        if key in services:
            import os

            os.environ.setdefault(key, services[key])
    weaviate_url = run_fase.resolve_weaviate_url(services)
    ollama_base = run_fase.resolve_ollama_base(services)
    print(f"weaviate : {weaviate_url}")
    print(f"ollama   : {ollama_base}")
    if not run_fase.weaviate_ready(weaviate_url):
        print("ERROR: Weaviate not ready — start rag-services.sbatch first")
        return 2

    client = run_fase.connect_weaviate_client(weaviate_url)
    try:
        # This client build returns plain strings; others return config objects.
        listed = client.collections.list_all()
        names = [c if isinstance(c, str) else getattr(c, "name", str(c)) for c in listed]
    except Exception as exc:
        print(f"ERROR: cannot list collections: {exc}")
        return 2
    if args.collection not in names:
        print(f"ERROR: collection {args.collection!r} not found. Available: {names}")
        return 2
    col = client.collections.get(args.collection)

    embedding = run_fase.load_embeddings(args.embedding)
    llm = None
    if not args.no_llm:
        llm = run_fase.get_gen_llm(args.llm, args.temperature, ollama_base)

    control = self_check_probe(col)
    print(f"probe self-check: {'OK' if control['ok'] else 'SUSPECT'} ({control['detail']})")
    print()

    # --- side B: Weaviate -------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for check, a_row in zip(smoke_graph.CHECKS, a_rows):
        question = check["question"]
        literals = probe_literals(check)
        row: Dict[str, Any] = {
            "id": check["id"],
            "topic": check["topic"],
            "question": question,
            "expected": [normalise_literal(e) for e in check["expected"]],
            "graph_ok": a_row["ok"],
            "graph_got": a_row["got"],
            "literals": [],
            "n_docs": 0,
            "filenames": [],
            "answer": None,
            "error": None,
        }

        # 1. does the index hold each answer literal at all?
        for lit in literals:
            probe = bm25_probe(col, lit)
            row["literals"].append({"literal": lit, **probe})

        # 2. normal hybrid retrieval for the question
        try:
            docs = run_fase.hybrid_search(col, embedding, question, args.alpha, args.k)
        except Exception as exc:
            errors.append(f"side B {check['id']} retrieval failed: {exc}")
            row["error"] = f"retrieval: {exc}"
            docs = []
        row["n_docs"] = len(docs)
        row["filenames"] = [d.metadata.get("filename") for d in docs]
        context = docs_text(docs)

        # 3. optional generation
        if llm is not None and docs:
            try:
                out = llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
                row["answer"] = (out.content if hasattr(out, "content") else str(out)).strip()
            except Exception as exc:
                errors.append(f"side B {check['id']} generation failed: {exc}")
                row["answer"] = None
                row["error"] = f"generation: {exc}"

        # 4. does the literal reach the context / the answer?
        for lit in row["literals"]:
            lit["in_context"] = contains(context, lit["literal"])
            lit["in_answer"] = contains(row["answer"] or "", lit["literal"])

        row["any_in_context"] = any(l["in_context"] for l in row["literals"])
        row["any_in_answer"] = any(l["in_answer"] for l in row["literals"])
        row["any_in_index"] = any(l["found"] for l in row["literals"])
        rows.append(row)

    # --- corpus scan (evidence that the literal is not in the PDFs) -------
    corpus_note = None
    if args.corpus and args.corpus.is_dir():
        wanted = sorted({lit for r in rows for lit, _ in [(l["literal"], None) for l in r["literals"]]})
        found = dict(smoke_graph.scan_corpus(wanted, args.corpus))
        corpus_note = {lit: found.get(lit, -1) for lit in wanted}
        for r in rows:
            for l in r["literals"]:
                l["corpus_hits"] = corpus_note.get(l["literal"], -1)
    else:
        corpus_note = f"skipped (corpus not available at {args.corpus})"

    # --- report ------------------------------------------------------------
    elapsed = time.time() - started
    print("=" * 78)
    print("A/B smoke — same questions: RDF graph vs Weaviate")
    print("=" * 78)
    print(f"graph     : {args.graph} ({triples} triples)")
    print(f"collection: {args.collection}   embedding: {args.embedding}")
    print(f"strategy  : hybrid alpha={args.alpha}, k={args.k}"
          + (f"   llm: {args.llm}" if llm else "   llm: disabled"))
    print(f"corpus    : {corpus_note if isinstance(corpus_note, str) else 'scanned'}")
    print("-" * 78)
    hdr = f"{'id':<4}{'graph':<8}{'in index':<10}{'in context':<12}{'in answer':<11}docs"
    print(hdr)
    for r in rows:
        print(
            f"{r['id']:<4}"
            f"{'PASS' if r['graph_ok'] else 'FAIL':<8}"
            f"{'yes' if r['any_in_index'] else 'NO':<10}"
            f"{'yes' if r['any_in_context'] else 'NO':<12}"
            f"{'yes' if r['any_in_answer'] else 'NO':<11}"
            f"{r['n_docs']}"
        )
    print("-" * 78)
    for r in rows:
        print(f"[{r['id']}] {r['topic']} — {r['question']}")
        print(f"    graph answer : {r['graph_got']}")
        for l in r["literals"]:
            print(f"    literal {l['literal']!r}: index={l['found']}"
                  f" context={l['in_context']} answer={l['in_answer']}")
            if isinstance(corpus_note, dict):
                print(f"      corpus hits: {corpus_note.get(l['literal'], -1)}")
        if r["answer"]:
            print(f"    rag answer   : {r['answer'][:300]}")
        elif r["error"]:
            print(f"    error        : {r['error']}")
        print(f"    retrieved    : {r['filenames']}")
        print()

    a_pass = sum(1 for r in a_rows if r["ok"])
    b_index = sum(1 for r in rows if r["any_in_index"])
    b_ctx = sum(1 for r in rows if r["any_in_context"])
    b_ans = sum(1 for r in rows if r["any_in_answer"])
    print("-" * 78)
    print(f"graph        : {a_pass}/{len(a_rows)} answers reproduced")
    print(f"weaviate     : literal in index {b_index}/{len(rows)} | "
          f"in context {b_ctx}/{len(rows)} | in answer {b_ans}/{len(rows)}")
    if not control["ok"]:
        print("WARNING: probe self-check failed — read the 'in index' column with care")
    print(f"elapsed      : {elapsed:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "graph": str(args.graph),
        "triples": triples,
        "collection": args.collection,
        "embedding": args.embedding,
        "alpha": args.alpha,
        "k": args.k,
        "llm": None if args.no_llm else args.llm,
        "temperature": args.temperature,
        "weaviate_url": weaviate_url,
        "ollama_url": ollama_base,
        "probe_self_check": control,
        "corpus": str(args.corpus) if args.corpus else None,
        "corpus_hits": corpus_note if isinstance(corpus_note, dict) else corpus_note,
        "elapsed_s": round(elapsed, 2),
        "summary": {
            "graph_passed": a_pass,
            "graph_total": len(a_rows),
            "literal_in_index": b_index,
            "literal_in_context": b_ctx,
            "literal_in_answer": b_ans,
            "questions": len(rows),
        },
        "rows": rows,
        "errors": errors,
    }
    json_path = RESULTS_DIR / f"{args.out_prefix}_{ts}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = RESULTS_DIR / f"{args.out_prefix}_{ts}.md"
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"report     : {json_path}")
    print(f"             {md_path}")

    try:
        client.close()
    except Exception:
        pass
    return 0 if (a_pass == len(a_rows) and not errors) else 1


def render_markdown(p: Dict[str, Any]) -> str:
    s = p["summary"]
    out = [
        "# A/B smoke: grafo RDF vs Weaviate",
        "",
        f"Fecha: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "Las **mismas 5 preguntas** del smoke del grafo (`scripts/smoke_graph.py`),",
        "resueltas por los dos caminos. Las respuestas viven **solo en el grafo**:",
        "ni en los PDF ni en el geojson, así que este experimento mide qué parte",
        "del conocimiento del grafo puede reproducir el índice vectorial.",
        "",
        "## Configuración",
        "",
        "| | |",
        "|---|---|",
        f"| grafo | `{p['graph']}` ({p['triples']} triples) |",
        f"| colección | `{p['collection']}` |",
        f"| embedding | `{p['embedding']}` |",
        f"| estrategia | hybrid alpha={p['alpha']}, k={p['k']} |",
        f"| LLM | `{p['llm'] or 'desactivado'}` (T={p['temperature']}) |",
        f"| Weaviate | `{p['weaviate_url']}` |",
        f"| Ollama | `{p['ollama_url']}` |",
        f"| autochequeo de la sonda | {'OK' if p['probe_self_check']['ok'] else 'SOSPECHOSO'}"
        f" — {p['probe_self_check']['detail']} |",
        f"| tiempo | {p['elapsed_s']} s |",
        "",
        "## Resultado",
        "",
        "| | respuestas |",
        "|---|---|",
        f"| grafo (SPARQL) | **{s['graph_passed']}/{s['graph_total']}** |",
        f"| literal presente en el índice (BM25) | {s['literal_in_index']}/{s['questions']} |",
        f"| literal llega al contexto (híbrido) | {s['literal_in_context']}/{s['questions']} |",
        f"| literal aparece en la respuesta del LLM | {s['literal_in_answer']}/{s['questions']} |",
        "",
        "| id | grafo | en índice | en contexto | en respuesta | docs |",
        "|---|---|---|---|---|---|",
    ]
    for r in p["rows"]:
        out.append(
            f"| {r['id']} | {'✅ PASS' if r['graph_ok'] else '❌ FAIL'} "
            f"| {'✅' if r['any_in_index'] else '❌'} "
            f"| {'✅' if r['any_in_context'] else '❌'} "
            f"| {'✅' if r['any_in_answer'] else '❌'} | {r['n_docs']} |"
        )
    out += ["", "## Detalle por pregunta", ""]
    for r in p["rows"]:
        out.append(f"### {r['id']} — {r['topic']}")
        out += ["", f"**Pregunta:** {r['question']}", "", f"**Respuesta del grafo:** `{r['graph_got']}`", ""]
        if r["answer"]:
            out.append(f"**Respuesta RAG:** {r['answer']}")
            out.append("")
        out += ["| literal | en índice | en contexto | en respuesta | en el corpus |", "|---|---|---|---|---|"]
        for l in r["literals"]:
            corpus = l.get("corpus_hits", "—")
            out.append(
                f"| `{l['literal']}` | {'✅' if l['found'] else '❌'} "
                f"| {'✅' if l['in_context'] else '❌'} "
                f"| {'✅' if l['in_answer'] else '❌'} | {corpus} |"
            )
        out += ["", f"**Documentos recuperados:** `{r['filenames']}`", ""]
    if p["errors"]:
        out += ["## Errores", ""] + [f"- {e}" for e in p["errors"]] + [""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
