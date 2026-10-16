#!/usr/bin/env python3
"""Replay the round-1 retrieval CSV with the per-file cap (Parche B) applied.

Evidence for the round-2 tracker (odd/tasks/eval-round2-retrieval-and-labels.md):
it reproduces the hit@k/MRR of
data/results/eval_ablation_retrieval_full_20261009_112109.csv cell by cell
(same logic as notebook cell 26), then re-ranks every row with
cap_per_file(docs, cap=RANK_CAP_PER_FILE) and reports the flips.

Expected (2026-10-16): Q8 -> rank 4 and Q14 -> rank 7 in both strategies,
44 -> 48 hits over the 64 rows, no losses. Exit code 1 when the replay stops
reproducing the stored CSV (the evidence is stale and must not be quoted).

Usage:
    python scripts/simulate_cap_per_file.py

The per-row docs list is rebuilt as (filename, yacimiento_id) pairs: the id
column stores the .geojson rows in order, so ids are re-attached to the rows
whose filename ends in .geojson (route legs put the structured block first).
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "results" / "eval_ablation_retrieval_full_20261009_112109.csv"
LABELS_PATH = REPO / "data" / "eval_gold_labels.py"
K = 10


def _norm_filename(name: Any) -> str:
    """Accent/case-folded filename without the .pdf extension (cell 26 match key)."""
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("\xad", "").strip().lower()
    return s[:-4] if s.endswith(".pdf") else s


def _load_labels():
    spec = importlib.util.spec_from_file_location("eval_gold_labels", LABELS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GOLD_LABELS, module.gold_docs


def _gold_keys(labels, gold_docs, question: str) -> Optional[Set[Tuple[str, str]]]:
    idx = next((i for i, lb in labels.items()
                if lb.get("question", "").strip() == (question or "").strip()), None)
    if idx is None:
        return None
    keys: Set[Tuple[str, str]] = set()
    for doc in gold_docs(idx):
        if doc.get("yacimiento_id") is not None:
            keys.add(("id", str(doc["yacimiento_id"])))
        if doc.get("yacimiento_nombre"):
            keys.add(("name", doc["yacimiento_nombre"].strip().lower()))
        if doc.get("filename") and doc.get("yacimiento_id") is None:
            keys.add(("file", _norm_filename(doc["filename"])))
    return keys or None


def _rank(found_ids: List[str], found_files: List[str],
          keys: Optional[Set[Tuple[str, str]]]) -> Optional[int]:
    """Rank of the gold doc, replicating notebook cell 26 (id path, then filename)."""
    if not keys:
        return None
    rank = None
    for pos in range(K):
        if pos >= len(found_ids):
            break
        if ("id", found_ids[pos]) in keys:
            rank = pos + 1
            break
    if rank is None and any(kind == "file" for kind, _ in keys):
        gold_files = {val for kind, val in keys if kind == "file"}
        for pos in range(K):
            if pos >= len(found_files):
                break
            if found_files[pos] in gold_files:
                rank = pos + 1
                break
    return rank


class _Doc:
    """Minimal stand-in for a langchain Document: cap_per_file only reads metadata."""

    __slots__ = ("metadata",)

    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata


def _docs_with_ids(filenames: List[str], ids: List[str]) -> List[_Doc]:
    """Rebuild the row's docs list; the real cap_per_file keys on metadata."""
    docs = []
    pending = list(ids)
    for name in filenames:
        yid = pending.pop(0) if (name.lower().endswith(".geojson") and pending) else None
        docs.append(_Doc({"filename": name, "id": yid}))
    return docs


def main() -> int:
    sys.path.insert(0, str(REPO))
    from src.geo_route import RANK_CAP_PER_FILE, cap_per_file

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    labels, gold_docs = _load_labels()
    qidx = {v["question"].strip(): k for k, v in labels.items()}

    sanity_bad = 0
    hits_before = hits_after = 0.0
    flips: List[str] = []
    for row in rows:
        q = row["question"]
        keys = _gold_keys(labels, gold_docs, q)
        filenames = [x for x in row["filenames"].split(";") if x]
        ids = [x for x in row["yacimiento_ids"].split(";") if x]
        rank0 = _rank(ids, [_norm_filename(x) for x in filenames], keys)
        hit0 = 1.0 if rank0 else 0.0
        if hit0 != float(row["hit_at_k"]):
            sanity_bad += 1
            print(f"SANITY MISMATCH Q{qidx.get(q.strip(), '?')} {row['strategy']}")
        capped = cap_per_file(_docs_with_ids(filenames, ids), cap=RANK_CAP_PER_FILE)
        rank1 = _rank([d.metadata["id"] for d in capped if d.metadata["id"]],
                      [_norm_filename(d.metadata["filename"]) for d in capped], keys)
        hit1 = 1.0 if rank1 else 0.0
        hits_before += hit0
        hits_after += hit1
        if rank0 != rank1:
            tag = "FLIP+" if hit1 > hit0 else ("FLIP-" if hit1 < hit0 else "")
            qn = qidx.get(q.strip(), "?")
            flips.append(f"Q{qn} {row['strategy']} {rank0}->{rank1} {tag}")
            print(f"{qn:>3} | {row['strategy'][:12]:12s} | {row['route'][:28]:28s} | "
                  f"hit {hit0:.0f}->{hit1:.0f} | rank {rank0} -> {rank1} {tag}")

    print()
    print(f"sanity: {len(rows) - sanity_bad}/{len(rows)} rows reproduce the stored CSV hit@k")
    print(f"hits: {hits_before:.0f}/{len(rows)} -> {hits_after:.0f}/{len(rows)} "
          f"with cap_per_file(cap={RANK_CAP_PER_FILE})")
    print("flips:", flips if flips else "none")
    return 1 if sanity_bad else 0


if __name__ == "__main__":
    sys.exit(main())
