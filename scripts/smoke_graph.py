#!/usr/bin/env python3
"""Deterministic smoke test for the RDF graph layer of the IDEArq RAG system.

The script loads the bibliographic RDF graph, runs five SPARQL questions whose
answers live only in the graph (never in the PDF corpus nor in the C14 geojson),
compares each result against the documented expected answer and prints a
PASS/FAIL report.

Exit code: 0 when every check passes, 1 otherwise.

No LLM, no Weaviate, no Slurm, no GPU. Runtime is well under 60 s.

Usage:
    python3 scripts/smoke_graph.py
    python3 scripts/smoke_graph.py --corpus /tmp/idearq_smoke_csusieag/pdftext
    python3 scripts/smoke_graph.py --graph data/biblio-graph/other.nt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH = PROJECT_ROOT / "data" / "biblio-graph" / "idearq-graph-instances-dedup-v2.nt"

PREDICATE_DCTERMS = "<http://purl.org/dc/terms/>"
PREDICATE_FOAF = "<http://xmlns.com/foaf/0.1/>"
PREDICATE_BIBO = "<http://purl.org/ontology/bibo/>"

# The five graph-only questions. `expected` is compared against the normalised
# SPARQL result (list of strings, sorted) via `normalise`.
CHECKS = [
    {
        "id": "Q1",
        "topic": "authorship",
        "question": (
            "Who are the authors recorded by the graph for "
            "'University of Pennsylvania radiocarbon dates XXI'?"
        ),
        "query": f"""
PREFIX dcterms: {PREDICATE_DCTERMS}
PREFIX foaf: {PREDICATE_FOAF}
SELECT ?author WHERE {{
  ?work dcterms:title "University of Pennsylvania radiocarbon dates XXI" ;
        dcterms:creator ?authorNode .
  ?authorNode foaf:name ?author .
}} ORDER BY ?author
""",
        "expected": ["Anne Meulengracht", "Barbara Lawn", "Patrick McGovern"],
        # Answer literals that must be absent from the document layer.
        "graph_only_literals": [
            "Anne Meulengracht",
            "Barbara Lawn",
            "Patrick McGovern",
        ],
    },
    {
        "id": "Q2",
        "topic": "publication-year",
        "question": (
            "In which year was the book 'Secuencias de cambio social en una "
            "región mediterránea...' published?"
        ),
        "query": f"""
PREFIX dcterms: {PREDICATE_DCTERMS}
SELECT ?issued WHERE {{
  ?work dcterms:title "Secuencias de cambio social en una región mediterránea. Análisis arqueológico de la depresión de Vera (Almería) entre los siglos V y XI" ;
        dcterms:issued ?issued .
}}
""",
        # rdflib normalises xsd:gYear literals to "YYYY-01-01"; the year is the
        # first four characters of the lexical form.
        "expected": ["2003"],
        "postprocess": lambda values: [str(v)[:4] for v in values],
        "graph_only_literals": [
            # The work itself is absent from the corpus, so no PDF can link the
            # work to its year. The bare token "2003" is not unique (16.5k hits).
            "Secuencias de cambio social en una región mediterránea",
        ],
    },
    {
        "id": "Q3",
        "topic": "journal-ISSN",
        "question": (
            "What is the ISSN of the journal that published "
            "'El Sistema Defensivo del Castro de La Ulaña'?"
        ),
        "query": f"""
PREFIX dcterms: {PREDICATE_DCTERMS}
PREFIX bibo: {PREDICATE_BIBO}
SELECT ?issn WHERE {{
  ?work dcterms:title "El Sistema Defensivo del Castro de La Ulaña" ;
        dcterms:isPartOf ?journal .
  ?journal bibo:issn ?issn .
}}
""",
        "expected": ["0066-6742"],
        "graph_only_literals": ["0066-6742", "00666742"],
    },
    {
        "id": "Q4",
        "topic": "DOI",
        "question": (
            "What is the DOI of 'Elfenbeinwerkstätten in Huelva und La "
            "Rebanadilla (Málaga)...'?"
        ),
        "query": f"""
PREFIX dcterms: {PREDICATE_DCTERMS}
PREFIX bibo: {PREDICATE_BIBO}
SELECT ?doi WHERE {{
  ?work dcterms:title "Elfenbeinwerkstätten in Huelva und La Rebanadilla (Málaga), den Ältesten Phönizischen Niederlassungen auf der Iberischen Halbinsel" ;
        bibo:doi ?doi .
}}
""",
        "expected": ["10.34780/mm.v57i0.1003"],
        "graph_only_literals": ["10.34780/mm.v57i0.1003"],
    },
    {
        "id": "Q5",
        "topic": "instance-type",
        "question": (
            "Which BIBO type does the graph assign to "
            "'A Neolithic Case of Cranial Trepanation (Eira Pedrinha, Portugal)'?"
        ),
        "query": f"""
PREFIX dcterms: {PREDICATE_DCTERMS}
SELECT ?type WHERE {{
  ?work dcterms:title "A Neolithic Case of Cranial Trepanation (Eira Pedrinha, Portugal)" ;
        a ?type .
}}
""",
        "expected": ["http://purl.org/ontology/bibo/Book"],
        "graph_only_literals": ["10.1201/b16983-12"],
    },
]


def load_graph(path: Path):
    """Load the .nt graph with rdflib, or exit with a clear message."""
    try:
        from rdflib import Graph
    except ImportError:
        sys.stderr.write(
            "ERROR: rdflib is not installed. Install it or run this script "
            "in an environment that provides rdflib.\n"
        )
        raise SystemExit(1)
    if not path.is_file():
        sys.stderr.write(f"ERROR: graph file not found: {path}\n")
        raise SystemExit(1)
    graph = Graph()
    graph.parse(str(path), format="nt")
    return graph


def run_check(graph, check: dict) -> tuple[bool, list[str]]:
    """Run one SPARQL check and compare it with the expected answer."""
    rows = graph.query(check["query"])
    values = [[str(cell) for cell in row] for row in rows]
    if len(values) == 1 and len(values[0]) == 1:
        flat = [values[0][0]]
    else:
        flat = sorted(v for row in values for v in row)
    if "postprocess" in check:
        flat = sorted(check["postprocess"](flat))
    expected = sorted(str(e) for e in check["expected"])
    return flat == expected, flat


def scan_corpus(literals: list[str], corpus: Path) -> list[tuple[str, int]]:
    """Count occurrences of each literal in an extracted text corpus."""
    texts: list[str] = []
    if corpus.is_dir():
        files = sorted(corpus.glob("*.txt"))
        texts = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    elif corpus.is_file():
        texts = [corpus.read_text(encoding="utf-8", errors="replace")]
    else:
        return [(lit, -1) for lit in literals]
    blob = "\n".join(texts)
    lower = blob.lower()
    return [(lit, blob.count(lit) + lower.count(lit.lower())) for lit in literals]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional directory of pdftotext .txt files (or a single text file) "
        "used to report graph-only evidence. Informational only: it never "
        "changes the exit code.",
    )
    args = parser.parse_args()

    start = time.time()
    graph = load_graph(args.graph)
    load_seconds = time.time() - start

    print("=" * 78)
    print("IDEArq RAG - graph layer smoke test")
    print("=" * 78)
    print(f"graph   : {args.graph}")
    print(f"triples : {len(graph)} (loaded in {load_seconds:.2f}s)")
    print(f"checks  : {len(CHECKS)} SPARQL questions")
    print("-" * 78)

    failures = 0
    for check in CHECKS:
        try:
            passed, got = run_check(graph, check)
        except Exception as exc:  # a query error is a failed check, not a crash
            passed, got = False, [f"<query error: {exc}>"]
        status = "PASS" if passed else "FAIL"
        failures += 0 if passed else 1
        print(f"[{status}] {check['id']} ({check['topic']})")
        print(f"       question : {check['question']}")
        print(f"       expected : {sorted(str(e) for e in check['expected'])}")
        print(f"       got      : {got}")
        if args.corpus is not None:
            for lit, hits in scan_corpus(check["graph_only_literals"], args.corpus):
                if hits < 0:
                    print(f"       evidence : corpus not found at {args.corpus}")
                else:
                    where = "NOT in corpus" if hits == 0 else f"FOUND {hits}x in corpus"
                    print(f"       evidence : {lit!r} -> {where}")
        print()

    total = len(CHECKS)
    print("-" * 78)
    if failures:
        print(f"RESULT: FAIL ({total - failures}/{total} checks passed)")
    else:
        print(f"RESULT: PASS ({total}/{total} checks passed)")
    print(f"elapsed: {time.time() - start:.2f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
