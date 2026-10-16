"""Unit tests for src.geo_route (Route B retrieval).

Pure unit tests: no network, no model load. Runnable with pytest when available
or, because pytest is not installed in env_rag, directly with plain Python:

    python tests/test_geo_route.py

The test file inserts the repo root on sys.path so it works from the repo root.
"""
from __future__ import annotations

import contextlib
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.geo_route as geo_route

# langchain may be absent when this runs under a bare interpreter; the route
# module only needs page_content + metadata, so a tiny stand-in is enough.
try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - only for interpreters without langchain
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str = "", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


# The real stored vocabulary of the GTE eval collection (subset of the values
# read from Weaviate), so plan_geo_query() never needs the network.
_REAL_PERIODS = [
    "calcolítico",
    "calcolítico, edad del bronce",
    "edad del bronce",
    "edad del hierro",
    "epipaleolítico / mesolítico",
    "neolítico",
    "neolítico, calcolítico",
    "paleolítico",
    "paleolítico, neolítico",
    "romano",
]
_REAL_REGIONS = [
    "A Coruña", "Almería", "Barcelona", "Córdoba", "Granada", "Jaén", "Madrid", "Sevilla",
]


def _stub_collection_values(collection, prop, limit=geo_route.VOCAB_LIMIT, warnings=None):
    """Injected vocabulary: replaces the network read in plan_geo_query()."""
    return {"periodo": list(_REAL_PERIODS), "region": list(_REAL_REGIONS)}.get(prop, [])


@contextlib.contextmanager
def _injected_vocabulary():
    original = geo_route._collection_values
    geo_route._collection_values = _stub_collection_values
    try:
        yield
    finally:
        geo_route._collection_values = original


# The eight evaluation questions of this project, with their real texts, and the
# plan kind the deterministic planner must return (None = not a filtered query).
_EVAL_QUESTIONS = [
    ("siret",
     "In what year did the Siret brothers excavate the La Bastida de Totana site?",
     None),
    ("jaen",
     "Which Chalcolithic site is the furthest from the city of Jaén, in the province of Jaén?",
     "superlative_space"),
    ("almeria",
     "Emergency excavations carried out by the Regional Government of Andalusia in the "
     "province of Almería, published in 2001 — which sites?",
     None),
    ("valencina",
     "¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza Ladera Sur "
     "como la más reciente del área?",
     "lookup_by_date"),
    ("neolitico",
     "¿Estas dataciones son compatibles con el periodo del Neolítico? "
     "(Comunidad Valenciana 33.900±60, Galicia 31.690±50, Murcia 12.030±0)",
     "lookup_by_date"),
    ("megalito",
     "¿Cuál es la fecha más antigua para un yacimiento funerario megalítico en la "
     "Península Ibérica?",
     "superlative_date"),
    ("catalan_mesolithic",
     "Quines són les cronologies de les manifestacions funeràries del Mesolític a les "
     "diferents regions peninsulars?",
     "group_by_region"),
    ("paleo_per_comunidad",
     "Dataciones más antiguas (i.e, más altas) de yacimientos paleolíticos para cada "
     "comunidad autónoma.",
     "group_by_region"),
]


def test_plan_geo_query_classifies_the_eight_questions():
    with _injected_vocabulary():
        for name, question, expected in _EVAL_QUESTIONS:
            plan = geo_route.plan_geo_query(question, "IdearqGTE_800_50_v2")
            kind = plan["kind"] if plan else None
            assert kind == expected, f"{name}: expected {expected!r}, got {kind!r}"


def test_plan_geo_query_superlative_space_uses_jaen_region():
    with _injected_vocabulary():
        plan = geo_route.plan_geo_query(_EVAL_QUESTIONS[1][1], "IdearqGTE_800_50_v2")
    assert plan is not None
    assert plan["order"] == "distance:desc"
    assert plan["region"] == "Jaén"


def test_plan_geo_query_lookup_by_date_normalizes_sign():
    with _injected_vocabulary():
        plan = geo_route.plan_geo_query(_EVAL_QUESTIONS[3][1], "IdearqGTE_800_50_v2")
    assert plan is not None
    assert plan["dates"] == [1377]


def test_ctx_text_renders_coordinates_and_distance():
    doc = Document(
        page_content="Eras del Alcázar",
        metadata={
            "yacimiento_nombre": "Eras del Alcázar",
            "lat": 38.01012,
            "lon": -3.36453,
            "_distance_km": 45.9,
        },
    )
    out = geo_route.ctx_text(doc)
    assert "Coordenadas de Eras del Alcázar: lat=38.01012, lon=-3.36453" in out
    assert "Distancia a la referencia: 45.9 km" in out


def test_ctx_text_uses_distance_origin_from_metadata():
    doc = Document(
        page_content="chunk",
        metadata={"yacimiento_nombre": "X", "lat": 1.0, "lon": 2.0,
                  "_distance_km": 45.9, "_distance_from": "jaén"},
    )
    out = geo_route.ctx_text(doc)
    assert "Distancia a jaén: 45.9 km" in out


def test_haversine_jaen_to_eras_del_alcazar():
    km = geo_route._haversine_km(37.77, -3.79, 38.01012, -3.36453)
    assert 45.0 <= km <= 47.0, f"expected 45-47 km, got {km}"


def test_fuse_rows_enforces_max_chunks_per_file():
    # One file dominates both legs: without the quota it would fill all top_n.
    vector_rows = [
        {"_additional": {"id": f"v{i}"}, "content": "x" * 60,
         "filename": "dom.pdf", "chunk_index": i}
        for i in range(7)
    ]
    bm25_rows = [
        {"_additional": {"id": f"b{i}"}, "content": "y" * 60,
         "filename": "dom.pdf", "chunk_index": i}
        for i in range(7)
    ] + [
        {"_additional": {"id": f"o{i}"}, "content": "z" * 60,
         "filename": "other.pdf", "chunk_index": i}
        for i in range(3)
    ]
    selected = geo_route._fuse_rows(vector_rows, bm25_rows, top_n=8)
    counts = Counter(r["filename"] for r in selected)
    assert counts["dom.pdf"] <= geo_route.MAX_CHUNKS_PER_FILE
    assert all(n <= geo_route.MAX_CHUNKS_PER_FILE for n in counts.values()), counts
    assert len(selected) >= geo_route.MAX_CHUNKS_PER_FILE


def test_budget_pdf_docs_caps_block():
    warnings = []
    docs = [Document("A" * 1000), Document("B" * 20000)]
    block, kept = geo_route._budget_pdf_docs(docs, warnings)
    assert len(block) <= geo_route.PDF_CHAR_BUDGET
    assert len(kept) == 1
    assert warnings, "a truncated block must emit a warning"
    assert "truncated" in warnings[-1]


def test_fuse_rows_dedups_on_object_id():
    row = {"_additional": {"id": "same"}, "content": "dup content " * 5,
           "filename": "dom.pdf", "chunk_index": 1}
    # Same id, different content[:80]: id keeps them fused into one entry.
    dup = {"_additional": {"id": "same"}, "content": "different " * 5,
           "filename": "dom.pdf", "chunk_index": 2}
    selected = geo_route._fuse_rows([row], [dup], top_n=8)
    assert len(selected) == 1


def test_retrieve_filtered_requires_explicit_collection():
    try:
        geo_route.retrieve_filtered("anything", collection=None)
    except ValueError:
        return
    raise AssertionError("retrieve_filtered() must raise ValueError on falsy collection")


def _geo_doc():
    return Document(
        page_content="Eras del Alcázar",
        metadata={
            "yacimiento_nombre": "Eras del Alcázar",
            "lat": 38.01012,
            "lon": -3.36453,
            "_distance_km": 45.9,
            "_distance_from": "jaén",
        },
    )


@contextlib.contextmanager
def _ctx_mode(mode):
    """Pin a context mode and restore the process default afterwards."""
    geo_route.set_ctx_mode(mode)
    try:
        yield
    finally:
        geo_route.set_ctx_mode(None)


def test_ctx_text_mode_no_distance_keeps_coordinates():
    with _ctx_mode("no_distance"):
        out = geo_route.ctx_text(_geo_doc())
    assert "Coordenadas de Eras del Alcázar" in out, out
    assert "Distancia a" not in out, out


def test_ctx_text_mode_no_coords_keeps_distance():
    with _ctx_mode("no_coords"):
        out = geo_route.ctx_text(_geo_doc())
    assert "Coordenadas de" not in out, out
    assert "Distancia a jaén: 45.9 km" in out, out


def test_ctx_text_mode_text_only_is_bare():
    with _ctx_mode("text_only"):
        out = geo_route.ctx_text(_geo_doc())
    assert out.strip() == "Eras del Alcázar", out


def test_set_ctx_mode_rejects_unknown_mode():
    try:
        geo_route.set_ctx_mode("no_cords")  # typo
    except ValueError:
        return
    raise AssertionError("set_ctx_mode() must raise ValueError on an unknown mode")


def test_ctx_mode_reads_the_environment():
    os.environ["CTX_MODE"] = "no_distance"
    geo_route.set_ctx_mode(None)
    try:
        out = geo_route.ctx_text(_geo_doc())
    finally:
        os.environ.pop("CTX_MODE", None)
    assert "Coordenadas de" in out and "Distancia a" not in out, out


def test_superlative_space_without_reference_point_warns():
    """An unknown city must degrade LOUDLY: no silently unordered results."""
    question = "Which Chalcolithic site is the furthest from the city of Almería?"
    original_graphql = geo_route._graphql
    geo_route._graphql = lambda query, timeout=None: []
    try:
        with _injected_vocabulary():
            result = geo_route.retrieve_filtered(question, collection="Stub_Collection")
    finally:
        geo_route._graphql = original_graphql
    assert result is not None, "the question should classify as superlative_space"
    assert result.plan["kind"] == "superlative_space", result.plan
    joined = " ".join(result.warnings)
    assert "without a reference point" in joined, result.warnings


def test_weaviate_url_prefers_services_env_over_a_stale_ambient_value():
    """A leftover WEAVIATE_URL=http://localhost:8080 in .env must NOT win.

    That regression made Route B talk to a dead host, so every plan silently
    degraded to None and the evaluation recorded plain top-k rows.
    """
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".env")
    with os.fdopen(fd, "w") as fh:
        fh.write('export WEAVIATE_URL="http://live-host:8080"\n')
    saved = {k: os.environ.get(k) for k in ("WEAVIATE_URL", "GEO_WEAVIATE_URL")}
    try:
        os.environ["WEAVIATE_URL"] = "http://localhost:8080"  # the stale .env value
        os.environ.pop("GEO_WEAVIATE_URL", None)
        assert geo_route._load_weaviate_url(path) == "http://live-host:8080"
        os.environ["GEO_WEAVIATE_URL"] = "http://explicit-override:9999"
        assert geo_route._load_weaviate_url(path) == "http://explicit-override:9999"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.remove(path)


def test_city_coords_table_can_be_disabled():
    """GEO_USE_CITY_COORDS=0 removes the reference point on purpose (round 1)."""
    question = ("Which Chalcolithic site is the furthest from the city of Jaén, "
                "in the province of Jaén?")
    os.environ["GEO_USE_CITY_COORDS"] = "0"
    try:
        assert geo_route._city_in_question(question, "Jaén") == (None, None)
        assert geo_route._use_city_coords() is False
    finally:
        os.environ.pop("GEO_USE_CITY_COORDS", None)
    name, coords = geo_route._city_in_question(question, "Jaén")
    assert coords is not None, "the table must work again once the flag is gone"


def test_superlative_space_without_the_table_warns_and_orders_deterministically():
    question = "Which Chalcolithic site is the furthest from the city of Jaén, in the province of Jaén?"
    rows = [
        {"_additional": {"id": "a"}, "content": "A", "filename": "c14_v2.geojson",
         "yacimiento_id": 7, "yacimiento_nombre": "Sitio B", "lat": 37.9, "lon": -3.9, "bp_max": 100.0},
        {"_additional": {"id": "b"}, "content": "B", "filename": "c14_v2.geojson",
         "yacimiento_id": 3, "yacimiento_nombre": "Sitio A", "lat": 38.1, "lon": -3.3, "bp_max": 500.0},
    ]
    original_graphql = geo_route._graphql
    geo_route._graphql = lambda query, timeout=None: rows
    os.environ["GEO_USE_CITY_COORDS"] = "0"
    try:
        with _injected_vocabulary():
            result = geo_route.retrieve_filtered(question, collection="Stub_Collection")
    finally:
        geo_route._graphql = original_graphql
        os.environ.pop("GEO_USE_CITY_COORDS", None)
    assert result is not None
    assert "without a reference point" in " ".join(result.warnings), result.warnings
    assert not any("_distance_km" in d.metadata for d in result.structured_docs)
    names = [d.metadata.get("yacimiento_nombre") for d in result.structured_docs]
    assert names == ["Sitio A", "Sitio B"], names  # bp_max desc, deterministic


def test_routing_miss_q7_now_routes_to_superlative_date():
    """'the oldest radiocarbon date of the islands' has no period/funerary cue."""
    question = ("Yacimiento con la datación radiocarbónica más antigua de las Islas "
                "Baleares y Canarias")
    with _injected_vocabulary():
        plan = geo_route.plan_geo_query(question, "Stub_Collection")
    assert plan is not None, "Q7 must not fall back to similarity search"
    assert plan["kind"] == "superlative_date", plan


def test_routing_miss_q20_now_routes_to_superlative_space_with_radius():
    question = ("Yacimientos neolíticos situados a menos de 150 km de Casa Montero.")
    with _injected_vocabulary():
        plan = geo_route.plan_geo_query(question, "Stub_Collection")
    assert plan is not None, "a stated distance must route spatially"
    assert plan["kind"] == "superlative_space", plan
    assert plan["radius_km"] == 150, plan


def test_candidate_phrases_finds_the_site_name_not_the_sentence_start():
    phrases = geo_route._candidate_phrases(
        "Yacimientos neolíticos situados a menos de 150 km de Casa Montero.")
    assert phrases and phrases[0] == "Casa Montero", phrases


def test_reference_point_comes_from_the_corpus_not_a_table():
    calls = []

    def fake_graphql(query, timeout=None):
        calls.append(query)
        return [{"yacimiento_nombre": "Casa Montero", "lat": 40.4, "lon": -3.5}]

    original = geo_route._graphql
    geo_route._graphql = fake_graphql
    try:
        with _injected_vocabulary():  # the phrase filter reads the corpus vocabulary
            name, lat, lon = geo_route._reference_from_corpus(
                "…150 km de Casa Montero.", "Stub_Collection")
    finally:
        geo_route._graphql = original
    assert (name, lat, lon) == ("Casa Montero", 40.4, -3.5)
    assert any('path:["yacimiento_nombre"]' in q for q in calls), calls


def test_superlative_space_uses_the_corpus_reference_with_the_table_off():
    """Round 1 disables the city table; the corpus reference must still work."""
    question = "Yacimientos neolíticos situados a menos de 150 km de Casa Montero."
    main_rows = [
        {"_additional": {"id": "a"}, "content": "cerca", "filename": "c14_v2.geojson",
         "yacimiento_id": 1, "yacimiento_nombre": "Sitio Cerca", "lat": 40.5, "lon": -3.5},
        {"_additional": {"id": "b"}, "content": "lejos", "filename": "c14_v2.geojson",
         "yacimiento_id": 2, "yacimiento_nombre": "Sitio Lejos", "lat": 43.0, "lon": -8.0},
    ]

    def fake_graphql(query, timeout=None):
        # The reference lookup is the query that FILTERS by yacimiento_nombre; the main
        # geo query only lists it as a requested field, so the path is the marker.
        if 'path:["yacimiento_nombre"]' in query:
            return [{"yacimiento_nombre": "Casa Montero", "lat": 40.4, "lon": -3.5}]
        return main_rows

    original = geo_route._graphql
    geo_route._graphql = fake_graphql
    os.environ["GEO_USE_CITY_COORDS"] = "0"
    try:
        with _injected_vocabulary():
            result = geo_route.retrieve_filtered(question, collection="Stub_Collection")
    finally:
        geo_route._graphql = original
        os.environ.pop("GEO_USE_CITY_COORDS", None)
    assert result is not None and result.plan["kind"] == "superlative_space"
    names = [d.metadata.get("yacimiento_nombre") for d in result.structured_docs]
    assert names == ["Sitio Cerca"], names  # the far one is filtered out by the 150 km radius
    assert any(d.metadata.get("_distance_from") == "Casa Montero"
               for d in result.structured_docs), [d.metadata for d in result.structured_docs]


def test_city_coords_file_overrides_and_extends():
    """GEO_CITY_COORDS_FILE rows win over the built-in table (accent/case-insensitive)."""
    import tempfile

    csv_text = (
        "id_ign,entidad,lat_etrs89,lon_etrs89,x_utm_etrs89,y_utm_etrs89\n"
        "2721829,Jaén,37.7651913,-3.7903594,430387.3292,4180057.084\n"
        "9999999,Ciudad Inventada,40.0,-4.0,0,0\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(csv_text)
        csv_path = fh.name
    saved = {key: os.environ.get(key)
             for key in ("GEO_CITY_COORDS_FILE", "GEO_USE_CITY_COORDS")}
    os.environ["GEO_CITY_COORDS_FILE"] = csv_path
    os.environ.pop("GEO_USE_CITY_COORDS", None)
    try:
        table = geo_route._city_coords()
        assert "jaén" not in table  # the CSV row (different case) replaced it
        assert len(table) == len(geo_route.CITY_COORDS) + 1
        name, coords = geo_route._city_in_question(
            "Which Chalcolithic site is the furthest from the city of Jaén?", None)
        assert name == "Jaén" and coords is not None
        assert abs(coords[0] - 37.7651913) < 1e-9
        assert abs(coords[1] - (-3.7903594)) < 1e-9
        _, new_coords = geo_route._city_in_question(
            "a menos de 5 km de Ciudad Inventada", None)
        assert new_coords == (40.0, -4.0)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.unlink(csv_path)


def test_cap_per_file_keeps_first_cap_per_file_and_order():
    docs = (
        [Document("a", metadata={"filename": "a.pdf"}) for _ in range(5)]
        + [Document("b", metadata={"filename": "b.pdf"}) for _ in range(2)]
    )
    out = geo_route.cap_per_file(docs, cap=3)
    assert [d.metadata["filename"] for d in out] == ["a.pdf", "a.pdf", "a.pdf", "b.pdf", "b.pdf"]


def test_cap_per_file_promotes_gold_pdf_into_k():
    # Replay of Q8 (round-1 CSV): the superlative route floods the list with
    # GeoJSON site docs; the gold PDF must land at rank 4, inside K=10.
    docs = (
        [Document("g", metadata={"filename": "c14_v2.geojson"}) for _ in range(10)]
        + [Document("gold", metadata={"filename": "1003_gold.pdf"})]
    )
    capped = geo_route.cap_per_file(docs)
    assert [d.metadata["filename"] for d in capped].index("1003_gold.pdf") + 1 == 4


def test_cap_per_file_default_cap_is_the_named_constant():
    assert geo_route.RANK_CAP_PER_FILE == 3
    docs = [Document("g", metadata={"filename": "g.geojson"}) for _ in range(4)]
    assert len(geo_route.cap_per_file(docs)) == 3


def test_cap_per_file_never_drops_a_files_first_occurrence():
    # Only repeats beyond the quota are dropped: a rare file behind a flood must
    # survive (its first occurrence is always kept) and move left, not vanish.
    docs = (
        [Document("a", metadata={"filename": "a.pdf"}) for _ in range(4)]
        + [Document("rare", metadata={"filename": "rare.pdf"})]
    )
    out = geo_route.cap_per_file(docs, cap=2)
    assert [d.metadata["filename"] for d in out] == ["a.pdf", "a.pdf", "rare.pdf"]


def test_cap_per_file_docs_without_filename_are_not_collapsed():
    docs = [Document("x", metadata={}), Document("y", metadata={})]
    assert len(geo_route.cap_per_file(docs)) == 2


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
