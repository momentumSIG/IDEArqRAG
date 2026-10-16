"""Deterministic answer checks for the evaluation CSV.

hit@k and MRR say whether the RIGHT DOCUMENT came back. They say nothing about
the answer, and the prompt rounds (CTX_MODE, GEO_USE_CITY_COORDS) only change the
answer, never the retrieval. Without a cheap check, "did the LLM compute/aggregate
by itself?" can only be judged by eye, question by question, which does not scale
to 32 questions x N strategies.

Two deterministic signals, no judge and no LLM:

* ``answer_pattern_ok`` — per question, a small auditable list of regexes the
  answer must contain. It is a proxy for "the answer carries the expected fact",
  not a quality grade: a wrong answer that happens to cite the right site passes.
  That is the same trade the project already accepts for hit@k.
* ``answer_gt_number_ratio`` — what fraction of the numbers in the gold answer
  appear in the generated answer. For the aggregation questions ("the oldest
  date per autonomous community") this IS the measurement: the gold answer is a
  list of per-community values, so the ratio is how much of the aggregation the
  model reproduced on its own.

Both are None when there is no answer (generation disabled) or no ground truth,
so the columns never invent a score.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

# Per-question regexes, keyed by a distinctive substring of the question. Kept
# deliberately small: every entry is a claim the answer must carry, and each one
# was checked against the real gold labels of this project.
ANSWER_CHECKS: Dict[str, List[str]] = {
    "In what year did the Siret brothers": [r"\b1886\b"],
    "Which Chalcolithic site is the furthest": [r"eras del alc[aá]zar"],
    "Emergency excavations carried out by the Regional Government": [
        r"villaricos", r"g[aá]dor|castillejo"],
    "¿Es correcta la datación -1377": [r"\b175\b"],
    "¿Cuál es la fecha más antigua para un yacimiento funerario": [r"trikuaizti"],
}

# Numbers written as "51.914", "51,914" or "51914" are the same quantity. Four or
# more digits are treated as one number; smaller ones are ignored because years of
# publication and lab codes would drown the signal.
_NUMBER_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d{4,6}")


def _fold(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).lower()
    return "".join(c for c in text if not unicodedata.combining(c))


def _numbers(text: Any) -> List[int]:
    """Every 4+ digit quantity in the text, thousands separators removed."""
    out: List[int] = []
    for raw in _NUMBER_RE.findall(str(text or "")):
        digits = raw.replace(".", "").replace(",", "")
        if len(digits) >= 4:
            try:
                out.append(int(digits))
            except ValueError:
                continue
    return out


def check_patterns(question: str, answer: Optional[str]) -> Optional[bool]:
    """True/False when the question has checks, None when it has none (or no answer)."""
    if answer is None:
        return None
    folded = _fold(question)
    for key, patterns in ANSWER_CHECKS.items():
        if _fold(key) in folded:
            text = _fold(answer)
            return all(re.search(p, text) for p in patterns)
    return None


def gt_number_ratio(ground_truth: Optional[str], answer: Optional[str]) -> Optional[float]:
    """Fraction of the gold-answer numbers present in the generated answer.

    None when there is no answer or the gold answer has no numbers to look for:
    an empty ratio would read as "0% correct", which is a different claim.
    """
    if answer is None:
        return None
    wanted = _numbers(ground_truth)
    if not wanted:
        return None
    found = set(_numbers(answer))
    return sum(1 for value in set(wanted) if value in found) / len(set(wanted))


def annotate_answer_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add the two answer columns to every result row, in place."""
    for row in rows:
        answer = row.get("answer")
        if isinstance(answer, str) and answer.startswith("Error:"):
            answer = None  # a failed generation is not an answer to grade
        row["answer_pattern_ok"] = check_patterns(row.get("question", ""), answer)
        row["answer_gt_number_ratio"] = gt_number_ratio(row.get("ground_truth"), answer)
    return rows
