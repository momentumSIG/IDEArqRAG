"""Unit tests for src.answer_metrics. Plain runner, no pytest needed:

    python tests/test_answer_metrics.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import answer_metrics as am  # noqa: E402


JAEN_Q = ("Which Chalcolithic site is the furthest from the city of Jaén, "
          "in the province of Jaén?")
AGG_Q = ("Dataciones más antiguas (i.e, más altas) de yacimientos paleolíticos "
         "para cada comunidad autónoma.")


def test_pattern_ok_when_the_answer_carries_the_fact():
    assert am.check_patterns(JAEN_Q, "El yacimiento más lejano es Eras del Alcázar.") is True


def test_pattern_false_when_the_answer_names_another_site():
    # The real regression: the model answered Albalate with an invented origin.
    assert am.check_patterns(JAEN_Q, "El más alejado es Albalate, en Porcuna.") is False


def test_pattern_is_accent_insensitive():
    assert am.check_patterns(JAEN_Q, "es ERAS DEL ALCAZAR, en Úbeda") is True


def test_pattern_none_for_questions_without_checks():
    assert am.check_patterns("¿Qué procesos tafonómicos afectan...?", "lo que sea") is None


def test_pattern_none_when_generation_was_skipped():
    assert am.check_patterns(JAEN_Q, None) is None


def test_ratio_counts_thousands_separators_as_the_same_number():
    gt = "Andalucía, 51.914 ± 45; Aragón, 25.330 ± 80; Cantabria, 48.200 ± 80"
    answer = "Andalucía: 51914 BP. Aragón: 25.330 BP."
    # 51914 and 25330 out of 51914/25330/48200
    assert abs(am.gt_number_ratio(gt, answer) - (2 / 3)) < 1e-9


def test_ratio_is_one_for_a_perfect_aggregation():
    gt = "Andalucía, 51.914 ± 45; Aragón, 25.330 ± 80"
    assert am.gt_number_ratio(gt, "51.914 y 25.330 BP") == 1.0


def test_ratio_is_none_without_answer_or_without_numbers():
    assert am.gt_number_ratio("51.914 ± 45", None) is None
    assert am.gt_number_ratio("No hay cifras aquí.", "da igual") is None


def test_annotate_treats_a_failed_generation_as_no_answer():
    rows = [{"question": JAEN_Q, "ground_truth": "Eras del Alcázar, 50 km",
             "answer": "Error: Connection refused"}]
    am.annotate_answer_metrics(rows)
    assert rows[0]["answer_pattern_ok"] is None
    assert rows[0]["answer_gt_number_ratio"] is None


def test_annotate_adds_both_columns_without_touching_the_rest():
    rows = [{"question": AGG_Q,
             "ground_truth": "Andalucía, 51.914 ± 45; Aragón, 25.330 ± 80",
             "answer": "Andalucía 51.914; Aragón 25.330",
             "hit_at_k": 1.0}]
    am.annotate_answer_metrics(rows)
    assert rows[0]["answer_gt_number_ratio"] == 1.0
    assert rows[0]["hit_at_k"] == 1.0, "must not touch the retrieval metrics"


def _run_all() -> int:
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - plain runner
            print(f"FAIL {test.__name__} -> {exc!r}")
            failures.append(test.__name__)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        print("failures:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
