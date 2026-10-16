#!/usr/bin/env python3
"""Summarise an eval_ablation_retrieval CSV by layer.

    python scripts/eval_summary.py                  # newest full run
    python scripts/eval_summary.py path/to/file.csv

Prints three blocks, in the order the pipeline can break:

    1. retrieval  -> hit@k / MRR, split by strategy and by route
    2. answer     -> answer_pattern_ok / answer_gt_number_ratio on the questions
                     whose gold document WAS retrieved (so a bad answer there is
                     the LLM's fault, not the retriever's)
    3. by layer   -> per-question table with the route, so a miss is attributable
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"
GOLD_DOCS = ("483_2001_error.pdf", "1135_Poblado_2021.pdf")


QIDS: dict[str, str] = {}

def question_numbers() -> dict[str, str]:
    """question text -> 'Q12', numbering exactly as data/eval_questions.py does."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from data.eval_questions import EVAL_ITEMS  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - a missing index must not break the summary
        return {}
    return {item[1]: f"Q{i}" for i, item in enumerate(EVAL_ITEMS, start=1)}


def newest(pattern: str) -> Path:
    files = sorted(RESULTS.glob(pattern))
    if not files:
        sys.exit(f"no CSV matching {pattern} in {RESULTS}")
    return files[-1]


def num(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def block_retrieval(rows: list[dict]) -> None:
    print("\n== 1. RETRIEVAL (hit@k / MRR) ==")
    print(f"{'strategy':<14}{'rows':>6}{'labelled':>9}{'hit@1':>8}{'MRR':>8}")
    for strategy in sorted({r["strategy"] for r in rows}):
        group = [r for r in rows if r["strategy"] == strategy]
        labelled = [r for r in group if r.get("labelled", "").lower() == "true"]
        scored = labelled or group
        hits = [num(r.get("hit_at_k")) for r in scored]
        rrs = [num(r.get("reciprocal_rank")) for r in scored]
        print(f"{strategy:<14}{len(group):>6}{len(labelled):>9}"
              f"{mean(hits):>8.3f}{mean(rrs):>8.3f}")

    print(f"\n{'route':<28}{'rows':>6}{'hit@1':>8}{'MRR':>8}   examples")
    by_route: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("labelled", "").lower() == "true":
            by_route[row.get("route") or "(sin ruta)"].append(row)
    for route, group in sorted(by_route.items(), key=lambda kv: -len(kv[1])):
        hits = [num(r.get("hit_at_k")) for r in group]
        rrs = [num(r.get("reciprocal_rank")) for r in group]
        example = next(
            (str(r["qid"]) for r in group if num(r.get("hit_at_k")) < 1.0),
            "todas aciertan")
        print(f"{route:<28}{len(group):>6}{mean(hits):>8.3f}{mean(rrs):>8.3f}   {example}")


def block_answer(rows: list[dict]) -> None:
    print("\n== 2. ANSWER (¿el LLM agrega bien?) ==")
    answered = [r for r in rows if (r.get("answer") or "").strip()]
    if not answered:
        print("sin respuestas: EVAL_RUN_GENERATION estaba apagado")
        return
    print(f"filas con respuesta: {len(answered)}/{len(rows)}")

    with_gold = [r for r in answered if (r.get("hit_at_k") or "") not in ("", "0.0", "0")]
    print(f"de esas, con el gold doc recuperado (hit=1): {len(with_gold)}")
    for label, group in (("hit=1", with_gold),
                         ("hit=0", [r for r in answered if r not in with_gold])):
        if not group:
            continue
        patterns = [r.get("answer_pattern_ok", "") for r in group]
        ok = sum(1 for p in patterns if p == "True")
        ratios = [num(r.get("answer_gt_number_ratio"), default=-1)
                  for r in group if r.get("answer_gt_number_ratio", "").strip()]
        ratios = [x for x in ratios if x >= 0]
        print(f"  {label}: {len(group)} filas | answer_pattern_ok "
              f"{ok}/{len(group)}"
              + (f" | ratio de cifras del gold: {mean(ratios):.2f}" if ratios else ""))

    numeric = [r for r in answered if r.get("tipo") in ("agregacion", "geo")]
    if numeric:
        print("\n  preguntas de agregación/geo (ratio de cifras del gold):")
        print(f"  {'Q':>4}{'tipo':<12}{'ratio':>7}{'pat':>6}  respuesta")
        for row in numeric:
            ratio = num(row.get("answer_gt_number_ratio"), default=-1)
            print(f"  {row['qid']:>4}{row.get('tipo',''):<12}"
                  f"{ratio:>7.2f}{str(row.get('answer_pattern_ok','')):>6}  "
                  f"{(row.get('answer') or '')[:70]}")


def block_by_layer(rows: list[dict]) -> None:
    print("\n== 3. POR PREGUNTA (atribución de la capa) ==")
    print(f"{'Q':>4}{'tipo':<11}{'route':<26}{'hit':>5}{'rr':>6}"
          f"{'pat':>5}{'ratio':>7}  pregunta")
    seen: dict[str, dict] = {}
    for row in rows:
        key = row["qid"]
        if key not in seen or row.get("strategy") == "vector":
            seen[key] = row
    for row in seen.values():
        ratio = row.get("answer_gt_number_ratio", "")
        pat = row.get("answer_pattern_ok", "")
        print(f"{row['qid']:>4}{row.get('tipo',''):<11}"
              f"{(row.get('route') or '')[:25]:<26}"
              f"{num(row.get('hit_at_k')):>5.0f}{num(row.get('reciprocal_rank')):>6.2f}"
              f"{str(pat):>5}{str(ratio):>7}  "
              f"{(row.get('question') or '')[:56]}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest("eval_ablation_retrieval_full_*.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        sys.exit(f"{path} está vacío")
    QIDS.update(question_numbers())
    for row in rows:
        row["qid"] = QIDS.get(row.get("question") or "", "?")
    print(f"CSV: {path.name}  ({len(rows)} filas, "
          f"{len({r.get('question') for r in rows})} preguntas, "
          f"{sorted({r.get('strategy') for r in rows})})")
    block_retrieval(rows)
    block_answer(rows)
    block_by_layer(rows)

    routes = Counter(r.get("route") or "topk" for r in rows)
    print(f"\nrutas observadas: {dict(routes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
