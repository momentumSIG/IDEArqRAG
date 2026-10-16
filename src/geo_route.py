"""Route B: deterministic query plan + filtered retrieval for the IDEARQ RAG.

Extracted from ``notebooks/evaluation/viz-geo-check.ipynb`` cell 15 (456 lines
of retrieval logic) so the route is importable and testable outside the notebook.

Known pending work: this module talks to Weaviate with a raw urllib GraphQL
helper (:func:`_graphql`). It does NOT use the official ``weaviate`` client and
does not share connection logic with ``src/weaviate_conn.py``. Unifying both is
pending; do NOT migrate to the official client in this pass.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Named constants (no bare magic numbers in the functions) ────────────────
RRF_K = 60                      # reciprocal-rank-fusion constant
VECTOR_LEG_WEIGHT = 3.0         # cross-lingual recall carries the most weight
BM25_LEG_WEIGHT = 1.0           # lexical booster for names and exact dates
LEG_MULTIPLIER = 2              # each PDF leg over-fetches top_n * this
GEO_ROWS_LIMIT = 600            # max GeoJSON rows fetched per filtered query
VOCAB_LIMIT = 3000              # max rows scanned to build a property vocabulary
HTTP_TIMEOUT_S = 90             # GraphQL request timeout
SCHEMA_TIMEOUT_S = 30           # schema probe timeout
PDF_CHAR_BUDGET = 16000         # hard cap of the assembled supporting-PDF block
MAX_CHUNKS_PER_FILE = 2         # per-filename chunk quota after PDF fusion
RANK_CAP_PER_FILE = 3           # per-filename quota on the ranking list (Parche B)
EARTH_RADIUS_KM = 6371.0        # mean Earth radius for the haversine distance

# ── Endpoint resolution (services job moves between nodes) ──────────────────
# The services job moves between nodes, so the endpoint is read from the same
# services.env the rest of the stack uses (a hardcoded c201-20 broke Route B).
SERVICES_ENV = "/mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/services.env"


def _services_env_value(key: str, path: str = SERVICES_ENV) -> Optional[str]:
    """Read one key from services.env (the project's live host file)."""
    if not os.path.exists(path):
        return None
    try:
        for raw in open(path):
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line:
                name, _, value = line.partition("=")
                if name.strip() == key:
                    return value.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _load_weaviate_url(path: str = SERVICES_ENV) -> str:
    """Resolve the Weaviate host, live-first.

    The order matters and it is not the obvious one. In a notebook run, cell 1
    calls ``load_dotenv(.env, override=True)`` and this repo's ``.env`` still
    carries the leftover ``WEAVIATE_URL=http://localhost:8080``. When the ambient
    value won, Route B talked to a dead host: the vocabulary read failed, every
    plan degraded to ``None`` and the evaluation recorded plain top-k rows. That
    failure LOOKS like a model problem and is really a wiring problem.

    So: explicit ``GEO_WEAVIATE_URL`` (deliberate override) > services.env
    (live source of truth, as the slurm launchers assume) > ambient
    ``WEAVIATE_URL`` (legacy) > default.
    """
    explicit = os.environ.get("GEO_WEAVIATE_URL")
    if explicit:
        return explicit.rstrip("/")
    live = _services_env_value("WEAVIATE_URL", path)
    if live:
        return live.rstrip("/")
    return (os.environ.get("WEAVIATE_URL") or "http://c202-22:8080").rstrip("/")


GEO_WEAVIATE_URL = _load_weaviate_url()

# ── Embedding hook ───────────────────────────────────────────────────────────
# The caller owns the query embedding model (the notebook already builds it) and
# sets it here before calling retrieve_filtered(). When it is None the vector PDF
# leg degrades visibly (errors + degraded), never silently.
embedding: Any = None


def set_embedding(embedder: Any) -> None:
    """Register the query embedder used by the supporting-PDF vector leg."""
    global embedding
    embedding = embedder


# ── Superlative cues per language present in the dataset (en, es, ca, fr, pt) ─
CUE_FAR_DISTANCE = ("furthest", "farthest", "most distant", "lejos", "alejado", "alejada",
                    "allunyat", "lointain", "mais distante")
CUE_NEAR_DISTANCE = ("closest", "nearest", "nearest to", "más cercano", "mas cercano",
                     "proche", "més proper", "mais proximo", "próximo")
CUE_OLDEST = ("oldest", "earliest", "most ancient", "más antiguo", "mas antiguo",
              "más antigua", "mas antigua", "más altas", "mas altas", "antiguas",
              "antiguo", "més antic", "mais antigo", "mais antiga")
CUE_NEWEST = ("newest", "most recent", "latest", "más reciente", "mas reciente",
              "més recent", "mais recente")
# The question writes the sign ("-1377±23"), the GeoJSON stores the BP value as a
# positive number (1377): capture the sign so plan_geo_query() can normalize it
# with abs() before filtering. Both signs mean the same calibration before present.
DATE_IN_TEXT = re.compile(r"(?<![\d.])([+-]?\d{3,5})\s*(?:±|\+/-)\s*\d+")   # "33.900±60", "-1377±23"
MEASURE_RE = re.compile(r"\b(\d{2,4})\s*(km|kilometers|kilómetros|kilometres)\b", re.I)
FUNERARY_CUES = ("funerar", "funerari", "megalit", "dolmen", "sepulc", "tomb", "burial",
                 "nekropolis", "necropolis")
MEGALITH_CUES = ("megalit", "dolmen", "sepulc", "tomb", "burial", "cairn", "cromlech")

# Closed vocabulary per collection, cached: this Weaviate build exposes no GraphQL
# groupBy, and a "NotNull" filter is rejected, so values are read and deduped here.
_VALUES_CACHE: Dict[Tuple[str, str], List[str]] = {}

# Read-only schema probe, cached per (collection, property): does the collection
# declare <prop>? bp_values was backfilled on the GeoJSON objects only, so a
# collection without it must keep the legacy bp_max path instead of failing.
_PROPERTY_CACHE: Dict[Tuple[str, str], bool] = {}

# Transient errors that justify the single retry in _graphql().
_TRANSIENT_HTTP = (urllib.error.URLError, urllib.error.HTTPError,
                   TimeoutError, ConnectionError)

# Errors the two supporting-PDF legs treat as a degraded (not fatal) result.
_PDF_LEG_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                   ConnectionError, RuntimeError, ValueError, KeyError, TypeError,
                   AttributeError)


def _fold(text: str) -> str:
    """Lowercase and strip accents so 'Megalitico' matches 'Megalítico'."""
    return "".join(c for c in unicodedata.normalize("NFD", (text or "").lower())
                   if unicodedata.category(c) != "Mn")


def _escape_graphql_string(value: Any) -> str:
    """Escape a value for interpolation inside a GraphQL string literal."""
    text = "" if value is None else str(value)
    return (text.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t"))


def _graphql(query: str, timeout: int = HTTP_TIMEOUT_S) -> List[Dict[str, Any]]:
    """Minimal GraphQL read helper (no client dependency) for the filter route.

    Retries once on a transient HTTP/socket error. A GraphQL ``errors`` payload is
    a deterministic rejection and is raised immediately (no retry).
    """
    payload = json.dumps({"query": query}).encode()
    last_exc: Optional[BaseException] = None
    data: Optional[Dict[str, Any]] = None
    for attempt in range(2):  # one retry
        try:
            req = urllib.request.Request(
                "%s/v1/graphql" % GEO_WEAVIATE_URL, data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            break
        except _TRANSIENT_HTTP as exc:
            last_exc = exc
            if attempt == 0:
                continue
            raise
    if data is None:  # defensive: loop cannot exit without data or an exception
        raise last_exc if last_exc is not None else RuntimeError("graphql: no response")
    if data.get("errors"):
        raise RuntimeError(data["errors"][0].get("message", "graphql error")[:200])
    # data looks like {"Get": {"<Collection>": [ {...}, ... ]}}: two levels down,
    # or the flat rows come back as strings if you extend with the dict keys.
    out: List[Dict[str, Any]] = []
    for block in (data.get("data") or {}).values():
        if isinstance(block, dict):
            for rows in block.values():
                out.extend(rows or [])
        else:
            out.extend(block or [])
    return out


# Backwards-compatible alias for the helper name the notebook used.
_gql_query = _graphql


def _collection_values(collection: str, prop: str, limit: int = VOCAB_LIMIT,
                       warnings: Optional[List[str]] = None) -> List[str]:
    """Distinct non-empty values of a text property: the vocabulary the planner matches."""
    key = (collection, prop)
    if key not in _VALUES_CACHE:
        try:
            rows = _graphql('{Get{%s(where:{path:["filename"],operator:Like,valueText:"*.geojson"},'
                            'limit:%d){%s}}}' % (collection, limit, prop))
            _VALUES_CACHE[key] = sorted({r[prop] for r in rows if r.get(prop)})
        except Exception as exc:
            if warnings is not None:
                warnings.append("  warn: no vocabulary for %s %s -> %s"
                                % (collection, prop, str(exc)[:80]))
            _VALUES_CACHE[key] = []
    return _VALUES_CACHE[key]


# Static vocabulary bridge. An English question names the Spanish categories with a
# different word ("Chalcolithic" vs "calcolítico"), and the Spanish keeps a final -o
# that English drops ("Mesolític" vs "mesolítico"). Both are fixed in code here: the
# question side is open/linguistic, but this mapping is static and auditable.
PERIOD_SYNONYMS = {
    "chalcolithic": "calcolítico", "chalcolith": "calcolítico",
    "neolithic": "neolítico",
    "paleolithic": "paleolítico", "palaeolithic": "paleolítico",
    "mesolithic": "mesolítico",
    "bronze age": "edad del bronce", "bronze": "bronce",
    "iron age": "edad del hierro", "iron": "hierro",
}


def _period_key(value: str) -> str:
    """Fold accents and drop the Spanish final -o: mesolitico -> mesolitic."""
    text = _fold(value)
    return text[:-3] + "ic" if text.endswith("ico") and len(text) > 6 else text


def _match_period(question: str, periods: List[str]) -> Optional[str]:
    """The collection's own label for the period the question asks about.

    Returned verbatim, never folded: Weaviate's Like is accent sensitive, so the
    stored spelling ("paleolítico") has to be the one used in the filter.
    """
    q = _fold(question)
    keys: Dict[str, str] = {}
    for value in periods:
        keys.setdefault(_period_key(value), value)
        # Multi-valued labels join with "," but some are "epipaleolítico / mesolítico",
        # so split on both or the plain "mesolítico" never reaches the vocabulary.
        for token in re.split(r"[,/]", value):
            token = token.strip()
            if token:
                keys.setdefault(_period_key(token), value)
    for key in sorted(keys, key=len, reverse=True):
        if key and key in q:
            return keys[key]
    for en, es in PERIOD_SYNONYMS.items():
        if _fold(en) in q:
            value = keys.get(_period_key(es))
            if value:
                return value
    return None


def plan_geo_query(question: str, collection: str,
                   warnings: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """-> query plan, or None when the question is not a filtered/aggregate one.

    Plan kinds:
      superlative_date  : "oldest funerary megalith" -> order by bp_max
      superlative_space : "furthest from Jaén"       -> order by distance to a city
      lookup_by_date    : dates written in the question -> filter to those dates
      group_by_region   : "Mesolithic funeraries by region" -> aggregate per region
    """
    q = _fold(question)
    periods = _collection_values(collection, "periodo", warnings=warnings)
    regions = _collection_values(collection, "region", warnings=warnings)
    matched_period = _match_period(question, periods)
    matched_region = next((r for r in regions if r and _fold(r) in q), None)
    # abs(): BP dates are stored positive (1377) while the question may write -1377.
    dates_in_q = [abs(int(v)) for v in DATE_IN_TEXT.findall(question.replace(".", ""))]
    measure = MEASURE_RE.search(question)
    # A stated distance ("a menos de 150 km de Casa Montero") IS a spatial query even
    # without the words "furthest"/"closest". Requiring those cues sent Q20 — which
    # has the radius filter implemented — down the similarity path.
    by_space = any(c in q for c in CUE_FAR_DISTANCE + CUE_NEAR_DISTANCE) or measure is not None
    by_date = any(c in q for c in CUE_OLDEST + CUE_NEWEST)
    is_funerary = any(c in q for c in FUNERARY_CUES)
    megalith = any(cue in q for cue in MEGALITH_CUES)

    per_region = any(cue in q for cue in
                     ("para cada", "cada comunidad", "cada region", "cada provincia",
                      "for each", "each region", "each community", "per region",
                      "per community", "a cada"))
    if matched_period and per_region:
        # "oldest date for EACH community" needs one winner per region, not the
        # global top-N: those are different answers.
        return {"kind": "group_by_region", "group_by": "region",
                "periodo": matched_period, "funerary": is_funerary, "megalith": megalith,
                "why": "aggregate the superlative per region"}
    if dates_in_q:
        return {"kind": "lookup_by_date", "dates": sorted(set(dates_in_q)),
                "periodo": matched_period, "region": matched_region,
                "megalith": megalith,
                "why": "the question states the dates explicitly"}
    if by_date:
        # The old "and (matched_period or is_funerary)" requirement was wrong: "the
        # oldest radiocarbon date of the islands" states a superlative over dates and
        # nothing else, so Q7 fell back to similarity search. A date superlative is
        # self-contained; the period, when present, only narrows it.
        direction = "desc" if any(c in q for c in CUE_OLDEST) else "asc"
        return {"kind": "superlative_date", "order": f"bp_max:{direction}",
                "periodo": matched_period, "region": matched_region,
                "typo_cue": is_funerary, "megalith": megalith,
                "why": "superlative over dates"}
    if by_space:
        # No region requirement either: the reference point can be a yacimiento named
        # in the question ("150 km from Casa Montero"), which retrieve_filtered
        # resolves from the corpus. A missing reference is a loud warning, never a
        # silent fallback.
        direction = "desc" if any(c in q for c in CUE_FAR_DISTANCE) else "asc"
        return {"kind": "superlative_space", "order": f"distance:{direction}",
                "region": matched_region, "periodo": matched_period,
                "radius_km": int(measure.group(1)) if measure else None,
                "why": "superlative over distance"}
    if matched_period and (is_funerary or "region" in q or "province" in q):
        return {"kind": "group_by_region", "group_by": "region",
                "periodo": matched_period, "funerary": is_funerary, "megalith": megalith,
                "why": "aggregate the same category per region"}
    return None


def _collection_has_property(collection: str, prop: str,
                             warnings: Optional[List[str]] = None) -> bool:
    """True when <collection> declares <prop> in its schema (one HTTP GET, cached)."""
    key = (collection, prop)
    if key not in _PROPERTY_CACHE:
        try:
            req = urllib.request.Request("%s/v1/schema/%s" % (GEO_WEAVIATE_URL, collection))
            with urllib.request.urlopen(req, timeout=SCHEMA_TIMEOUT_S) as resp:
                schema = json.load(resp)
            _PROPERTY_CACHE[key] = any(p.get("name") == prop
                                       for p in schema.get("properties") or [])
        except Exception as exc:
            if warnings is not None:
                warnings.append("  warn: schema probe failed for %s -> %s"
                                % (collection, str(exc)[:80]))
            _PROPERTY_CACHE[key] = False
    return _PROPERTY_CACHE[key]


def _where_clauses(plan: Dict[str, Any], collection: Optional[str] = None,
                   warnings: Optional[List[str]] = None) -> Tuple[List[str], Optional[str]]:
    """Build the GraphQL where clauses; return (clauses, date_property).

    Never mutates ``plan`` (the old version wrote ``plan["date_property"]``).
    """
    clauses = ['{path:["filename"],operator:Like,valueText:"*.geojson"}']
    # Values come from the collection vocabulary VERBATIM: Weaviate's Like is accent
    # sensitive, so folding "paleolítico" into "paleolitico" matches nothing.
    lookup = plan.get("kind") == "lookup_by_date"
    date_property: Optional[str] = None
    periodo = plan.get("periodo")
    if periodo and not lookup:
        clauses.append('{path:["periodo"],operator:Like,valueText:"*%s*"}'
                       % _escape_graphql_string(periodo))
    region = plan.get("region")
    if region and not lookup:
        clauses.append('{path:["region"],operator:Like,valueText:"*%s*"}'
                       % _escape_graphql_string(region))
    if plan.get("typo_cue") or plan.get("funerary"):
        clauses.append('{path:["tipologia_crono"],operator:Like,valueText:"*Funerario*"}')
    if plan.get("megalith"):
        # The corpus writes "Megalito", not "Megalitico" - match the corpus spelling.
        clauses.append('{path:["tipologia_crono"],operator:Like,valueText:"*Megalito*"}')
    if plan.get("kind") == "lookup_by_date" and plan.get("dates"):
        # bp_max only stores the HIGHEST date of the yacimiento, so asking for 1377
        # against bp_max=4676 (Valencina) matched 0 objects. bp_values holds EVERY
        # C14 date of the site -> filter there. bp_values is number[] and
        # indexFilterable (not indexSearchable): ContainsAny + valueNumber, never
        # valueNumberArray (Weaviate 1.33 rejects it) and never Like/Text.
        # Fallback: collections without the backfill (PDFs) keep the legacy bp_max.
        date_property = ("bp_values"
                         if _collection_has_property(collection or GEO_FILTER_COLLECTION,
                                                     "bp_values", warnings)
                         else "bp_max")
        dates = ",".join(str(int(d)) for d in plan["dates"])
        clauses.append('{path:["%s"],operator:ContainsAny,valueNumber:[%s]}'
                       % (_escape_graphql_string(date_property), dates))
    return clauses, date_property


# Reference points for distance superlatives. Keys are matched accent-folded by
# _city_in_question(), so accent variants (jaén/jaen) collapse to one entry.
CITY_COORDS = {
    "jaén": (37.77, -3.79), "úbeda": (38.01, -3.37),
    "granada": (37.18, -3.60), "madrid": (40.42, -3.70), "murcia": (37.99, -1.13),
    "valencia": (39.47, -0.38), "barcelona": (41.39, 2.17), "sevilla": (37.39, -5.98),
    "córdoba": (37.89, -4.78), "caceres": (39.48, -6.37), "badajoz": (38.88, -6.97),
    "zaragoza": (41.65, -0.89), "huesca": (42.13, -0.41), "teruel": (40.34, -1.11),
    "bilbao": (43.26, -2.93), "donostia": (43.32, -1.98), "vitoria": (42.85, -2.67),
    "santiago": (42.88, -8.54), "a coruña": (43.36, -8.41), "vigo": (42.24, -8.72),
    "palma": (39.57, 2.65), "las palmas": (28.12, -15.43),
    "santa cruz de tenerife": (28.46, -16.25), "melilla": (35.29, -2.94),
    "ceuta": (35.89, -5.32),
}


def _haversine_km(lat1, lon1, lat2, lon2):
    r = EARTH_RADIUS_KM
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _use_city_coords() -> bool:
    """Whether the hardcoded reference-city table may be used.

    ``GEO_USE_CITY_COORDS=0`` turns it off, which is what round 1 of the
    evaluation wants: without a reference point the code cannot compute the
    distance, so the question becomes "can the LLM do it alone?" instead of
    "can it read a number the code already computed".
    """
    raw = os.environ.get("GEO_USE_CITY_COORDS")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _city_coords() -> Dict[str, Tuple[float, float]]:
    """Reference-city table: hardcoded CITY_COORDS, overridden by an optional CSV.

    ``GEO_CITY_COORDS_FILE`` points to a CSV with a header row and the columns
    ``entidad,lat_etrs89,lon_etrs89`` (dot decimals; extra columns are ignored).
    File rows override matching entries (accent/case-insensitive) and add new ones;
    the built-in table stays as fallback, so an incomplete file never removes a
    reference point.
    """
    path = os.environ.get("GEO_CITY_COORDS_FILE")
    if not path:
        return CITY_COORDS
    table = dict(CITY_COORDS)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("entidad") or "").strip()
            if not name:
                continue
            folded = _fold(name)
            for existing in [key for key in table if _fold(key) == folded]:
                del table[existing]
            table[name] = (float(row["lat_etrs89"]), float(row["lon_etrs89"]))
    return table


def _city_in_question(question: str, region: Optional[str]):
    if not _use_city_coords():
        return None, None
    q = _fold(question)
    for name, coords in _city_coords().items():
        if _fold(name) in q:
            return name, coords
    return (region, None) if region else (None, None)


def _pdf_dedup_key(row: Dict[str, Any]) -> Any:
    """Dedup key for fused PDF rows: prefer the object id, fall back to the tuple."""
    obj_id = (row.get("_additional") or {}).get("id")
    if obj_id:
        return ("id", obj_id)
    return ("tuple", row.get("filename"), row.get("chunk_index"),
            (row.get("content") or "")[:80])


def _fuse_rows(vector_rows: Optional[List[Dict[str, Any]]],
               bm25_rows: Optional[List[Dict[str, Any]]],
               top_n: int) -> List[Dict[str, Any]]:
    """Weighted RRF of the two PDF legs, then the per-file quota + top_n cap."""
    fused: Dict[Any, Dict[str, Any]] = {}

    def _add(rows: Optional[List[Dict[str, Any]]], weight: float) -> None:
        for i, r in enumerate(rows or []):
            if not r.get("content"):
                continue
            entry = fused.setdefault(_pdf_dedup_key(r), {"row": r, "score": 0.0})
            entry["score"] += weight / (RRF_K + i)

    _add(vector_rows, VECTOR_LEG_WEIGHT)
    _add(bm25_rows, BM25_LEG_WEIGHT)

    ordered = sorted(fused.values(), key=lambda e: -e["score"])
    selected: List[Dict[str, Any]] = []
    per_file: Dict[Any, int] = {}
    for entry in ordered:
        filename = entry["row"].get("filename")
        if per_file.get(filename, 0) >= MAX_CHUNKS_PER_FILE:
            continue
        per_file[filename] = per_file.get(filename, 0) + 1
        selected.append(entry["row"])
        if len(selected) >= top_n:
            break
    return selected


def _candidate_phrases(question: str, max_words: int = 3) -> List[str]:
    """Capitalised word runs of the question that may name a site.

    "…a menos de 150 km de Casa Montero." -> ["Casa Montero"]. The first token is
    ignored on purpose: it is capitalised because it starts the sentence, not
    because it is a proper noun.
    """
    tokens = re.findall(r"[^\W\d_][\w'\u2019\-]*", question, flags=re.UNICODE)
    runs: List[List[str]] = []
    current: List[str] = []
    for index, token in enumerate(tokens):
        if token[:1].isupper() and index > 0:
            current.append(token)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    phrases = {" ".join(run[:max_words]) for run in runs
               if len(run) >= 2 or (run and len(run[0]) > 4)}
    return sorted(phrases, key=lambda p: (-len(p.split()), -len(p)))


def _plausible_site_phrases(question: str, collection: str,
                            warnings: Optional[List[str]] = None) -> List[str]:
    """Candidate site names, with the corpus vocabulary filtered out.

    ``_candidate_phrases`` alone would try "Chalcolithic" (a capitalised word of the
    question) as a yacimiento name. Anything that the collection already uses as a
    period, a region, or a funerary/megalith cue is not a site name.
    """
    known = set()
    for prop in ("periodo", "region"):
        for value in _collection_values(collection, prop, warnings=warnings):
            known.add(_fold(value))
    known.update(_fold(key) for key in PERIOD_SYNONYMS)
    known.update(_fold(value) for value in PERIOD_SYNONYMS.values())
    known.update(FUNERARY_CUES)
    known.update(MEGALITH_CUES)
    kept = []
    for phrase in _candidate_phrases(question):
        if len(phrase.split()) >= 2:
            kept.append(phrase)
        elif len(phrase) >= 5 and _fold(phrase) not in known:
            kept.append(phrase)
    return kept


def _reference_from_corpus(question: str, collection: str,
                           warnings: Optional[List[str]] = None,
                           max_candidates: int = 3
                           ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Resolve a yacimiento named in the question to its coordinates, from the corpus.

    This is data, not a gazetteer: the coordinates come from the same GeoJSON
    objects the route already reads, so a new reference site needs no code change
    and no hand-typed table. Returns ``(name, lat, lon)`` or ``(None, None, None)``.
    """
    for phrase in _plausible_site_phrases(question, collection, warnings)[:max_candidates]:
        query = ('{Get{%s(where:{operator:And,operands:['
                 '{path:["filename"],operator:Like,valueText:"*.geojson"},'
                 '{path:["yacimiento_nombre"],operator:Like,valueText:"*%s*"}]},limit:5)'
                 '{yacimiento_nombre lat lon}}}'
                 % (collection, _escape_graphql_string(phrase)))
        try:
            rows = _graphql(query)
        except Exception as exc:  # noqa: BLE001 - a failed lookup must not kill the route
            if warnings is not None:
                warnings.append("reference lookup failed for %r -> %s"
                                % (phrase, str(exc)[:120]))
            continue
        for row in rows or []:
            if row.get("lat") is not None and row.get("lon") is not None:
                return (row.get("yacimiento_nombre") or phrase,
                        float(row["lat"]), float(row["lon"]))
    return None, None, None


def _supporting_pdfs(question: str, collection: str, plan: Dict[str, Any],
                     top_n: int = 8,
                     errors: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Arreglo 5 (2026-10-07): Route B only returns GeoJSON rows, so an answer that
    lives in an article never reached the prompt (Q4 Ladera Sur 175+-20, Q7 El
    Collado). Two searches over the PDF subset, fused with reciprocal rank fusion:
      vector : weighted 3x because it carries the cross-lingual recall (Q7 is
               Catalan, the paper is English);
      BM25   : the lexical booster, proper names and exact dates (Q4).
    No metadata filters: tipologia_crono is null on every PDF and the PDF periodo
    vocabulary does not match the GeoJSON one, so any soft filter returned 0 rows.
    Rows, not Documents: the caller builds them. A lost leg is appended to
    ``errors`` so the evaluation can see the degradation."""
    if errors is None:
        errors = []
    pdf_where = '{path:["file_type"],operator:Equal,valueText:"pdf"}'
    fields = ("_additional{id} content filename chunk_index file_type year language "
              "periodo region title")

    # 1) vector search: needs the query embedding the notebook already owns.
    vector_rows: Optional[List[Dict[str, Any]]] = None
    try:
        emb = embedding.embed_query(question)
        vec = ",".join("%.6f" % float(x) for x in emb)
        vector_rows = _graphql('{Get{%s(nearVector:{vector:[%s]},where:{operator:And,operands:[%s]},'
                               'limit:%d){%s}}}'
                               % (collection, vec, pdf_where,
                                  top_n * LEG_MULTIPLIER, fields))
    except _PDF_LEG_ERRORS as exc:
        errors.append("supporting-PDF vector search failed -> %s" % str(exc)[:200])

    # 2) BM25: lexical fallback for names and dates.
    bm25_rows: Optional[List[Dict[str, Any]]] = None
    try:
        bm25_rows = _graphql('{Get{%s(where:{operator:And,operands:[%s]},bm25:{query:"%s"},'
                             'limit:%d){%s}}}'
                             % (collection, pdf_where,
                                _escape_graphql_string(question),
                                top_n * LEG_MULTIPLIER, fields))
    except _PDF_LEG_ERRORS as exc:
        errors.append("supporting-PDF BM25 failed -> %s" % str(exc)[:200])

    return _fuse_rows(vector_rows, bm25_rows, top_n)


# Context rendering modes. They exist to make the "is it the model or the design?"
# question measurable instead of debatable: the same retrieval can be served with
# the coordinates, with the precomputed distance, with both, or with neither, and
# the evaluation compares hit@k/MRR across rounds.
#   full        -> text + coordinates + distance      (production default)
#   no_distance -> text + coordinates                 (can the LLM compute the distance?)
#   no_coords   -> text + distance                    (is the distance line enough?)
#   text_only   -> text only                          (baseline: no structured help)
CTX_MODES = ("full", "no_distance", "no_coords", "text_only")
DEFAULT_CTX_MODE = "full"
_CTX_MODE: Optional[str] = None  # explicit override; None means "read the environment"


def set_ctx_mode(mode: Optional[str]) -> None:
    """Pin the context mode for this process. ``None`` restores env/default lookup.

    Raises ValueError on an unknown mode: a typo in an experiment must fail loudly,
    not silently fall back to the production rendering and poison the comparison.
    """
    global _CTX_MODE
    if mode is not None and mode not in CTX_MODES:
        raise ValueError("unknown CTX_MODE %r; expected one of %s" % (mode, CTX_MODES))
    _CTX_MODE = mode


def _ctx_mode() -> str:
    """Explicit override, else $CTX_MODE, else production default."""
    mode = _CTX_MODE or os.environ.get("CTX_MODE") or DEFAULT_CTX_MODE
    return mode if mode in CTX_MODES else DEFAULT_CTX_MODE


def ctx_text(doc: Any, reference: Optional[str] = None,
             mode: Optional[str] = None) -> str:
    """Single source of truth for serialising a Document into the prompt context.

    Serialises the chunk text, then the coordinates (byte-identical to the old
    ``_ctx_text``) and, when the planner attached ``_distance_km``, the
    precomputed distance line. The distance origin is ``reference``, else the
    ``_distance_from`` metadata field, else "la referencia".

    ``mode`` (or $CTX_MODE) selects how much structured help is rendered: see
    ``CTX_MODES``. The default keeps the production rendering.
    """
    mode = mode or _ctx_mode()
    txt = doc.page_content or ""
    meta = getattr(doc, "metadata", None) or {}
    if mode != "text_only":
        lat = meta.get("lat")
        lon = meta.get("lon")
        nombre = meta.get("yacimiento_nombre")
        if (lat is not None and lon is not None and nombre and mode != "no_coords"
                and "Coordenadas de" not in txt):
            try:
                txt += f"\nCoordenadas de {nombre}: lat={float(lat):.5f}, lon={float(lon):.5f}"
            except (TypeError, ValueError):
                pass
    if mode in ("full", "no_coords"):
        distance_km = meta.get("_distance_km")
        if distance_km is not None:
            origin = reference or meta.get("_distance_from") or "la referencia"
            txt += f"\nDistancia a {origin}: {distance_km} km"
    return txt


PDF_BLOCK_HEADER = ("SUPPORTING ARTICLES (PDF chunks, keyword search over the article "
                    "text - these are NOT structured records):")


def _budget_pdf_docs(pdf_docs: List[Any], warnings: List[str]) -> Tuple[str, List[Any]]:
    """Assemble the supporting-PDF block within PDF_CHAR_BUDGET, whole chunks only.

    Returns ``(block, kept_docs)``: ``kept_docs`` are exactly the documents whose
    text is in the block, so ``docs``/``n_pdf_docs`` stay consistent with context.
    """
    if not pdf_docs:
        return "", []
    sep = "\n\n"
    used = len(PDF_BLOCK_HEADER) + len(sep)
    kept: List[Any] = []
    for doc in pdf_docs:
        chunk = ctx_text(doc)
        add = len(chunk) + (len(sep) if kept else 0)
        if used + add <= PDF_CHAR_BUDGET:
            kept.append(doc)
            used += add
            continue
        if not kept:
            # Even the first chunk overflows: truncate its text so the rendered
            # block (text + coordinate/distance lines) still fits the budget.
            overhead = len(ctx_text(doc)) - len(doc.page_content or "")
            room = PDF_CHAR_BUDGET - (len(PDF_BLOCK_HEADER) + len(sep)) - overhead
            if room > 0:
                doc.page_content = (doc.page_content or "")[:room]
                kept.append(doc)
            warnings.append("supporting PDF block truncated at %d characters"
                            % PDF_CHAR_BUDGET)
        else:
            warnings.append("supporting PDF block truncated at %d characters (%d/%d chunks kept)"
                            % (PDF_CHAR_BUDGET, len(kept), len(pdf_docs)))
        break
    block = PDF_BLOCK_HEADER + sep + sep.join(ctx_text(d) for d in kept)
    return block, kept


def cap_per_file(docs: List[Any], cap: int = RANK_CAP_PER_FILE) -> List[Any]:
    """Keep at most ``cap`` docs per source file, preserving order.

    Parche B (2026-10-16): the ranking list must not be monopolised by one file
    (a GeoJSON holding every site, a PDF with many chunks). Apply it after the
    merge and before the cut to K, on both retrieval legs. A document without a
    filename keeps its own key (uuid or identity), so it is never lumped with
    another one.
    """
    seen: Dict[Any, int] = {}
    out: List[Any] = []
    for doc in docs:
        md = getattr(doc, "metadata", None) or {}
        key = md.get("filename") or md.get("uuid") or id(doc)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= cap:
            out.append(doc)
    return out


@dataclass
class RouteResult:
    """Route B result. Diagnostics live on warnings/errors, never in print()."""

    plan: Dict[str, Any]
    structured_docs: List[Any]
    pdf_docs: List[Any]
    docs: List[Any]
    context: str
    n_candidates: int
    n_sites: int
    n_pdf_docs: int
    empty: bool
    note: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    degraded: bool = False
    collection: str = ""


def _doc_to_site_row(doc: Any) -> Dict[str, Any]:
    """Rebuild the pre-refactor site row (content + metadata) from a Document."""
    row: Dict[str, Any] = {"content": getattr(doc, "page_content", None)}
    row.update(getattr(doc, "metadata", None) or {})
    return row


def as_legacy_dict(result: RouteResult) -> dict:
    """Return the OLD dict shape so harnesses that index the result keep working."""
    return {
        "plan": result.plan,
        "sites": [_doc_to_site_row(d) for d in result.structured_docs],
        "docs": result.docs,
        "context": result.context,
        "n_candidates": result.n_candidates,
        "n_sites": result.n_sites,
        "n_pdf_docs": result.n_pdf_docs,
        "collection": result.collection,
        "empty": result.empty,
        "note": result.note,
    }


def _row_to_document(row: Dict[str, Any]) -> Any:
    """Build a langchain Document from a GraphQL row (imported lazily on purpose:
    the module must import even where langchain_core is unavailable)."""
    from langchain_core.documents import Document

    return Document(
        page_content=row.get("content") or "",
        metadata={k: v for k, v in row.items()
                  if k not in ("content", "_additional") and v is not None},
    )


def retrieve_filtered(question: str, collection: Optional[str] = None,
                      top_n: int = 10) -> Optional[RouteResult]:
    """Route B. Returns None when the question is not a filtered/aggregate one.

    ``collection`` is explicit; there is NO implicit global default. A falsy value
    raises ValueError instead of silently falling back to GEO_FILTER_COLLECTION.
    """
    if not collection:
        raise ValueError(
            "retrieve_filtered() requires an explicit collection; "
            "GEO_FILTER_COLLECTION is a legacy constant, not a silent default")
    warnings: List[str] = []
    errors: List[str] = []
    plan = plan_geo_query(question, collection, warnings=warnings)
    if plan is None:
        return None
    # Own our copy of the plan: _where_clauses must never mutate the caller's plan.
    plan = dict(plan)
    clauses, date_property = _where_clauses(plan, collection, warnings=warnings)
    if date_property:
        plan["date_property"] = date_property
    where = clauses[0] if len(clauses) == 1 else "{operator:And,operands:[%s]}" % ",".join(clauses)
    fields = ('_additional{id} content filename yacimiento_id '
              'yacimiento_nombre unidad_territorial tipologia_crono lat lon '
              'bp_min bp_max bp_mean c14_count periodo region')
    if plan.get("date_property") == "bp_values":
        # The array is asked for too: dedup may keep the sibling object of a
        # yacimiento, and only bp_values tells whether THAT object holds the date.
        fields += " bp_values"
    rows = _graphql('{Get{%s(where:%s,limit:%d){%s}}}'
                    % (collection, where, GEO_ROWS_LIMIT, fields))
    # Site-level aggregation: the corpus stores two objects per yacimiento_id.
    sites: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        sid = r.get("yacimiento_id")
        if sid is None:
            continue
        cur = sites.get(sid)
        if cur is None or (r.get("bp_max") or -1) > (cur.get("bp_max") or -1):
            sites[sid] = r
    sites = list(sites.values())

    answer_docs: List[Dict[str, Any]] = []
    if plan["kind"] == "superlative_date":
        dated = [s for s in sites if s.get("bp_max") is not None]
        dated.sort(key=lambda s: s["bp_max"], reverse=(plan["order"].endswith("desc")))
        answer_docs = dated[:top_n]
    elif plan["kind"] == "superlative_space":
        city, coords = _city_in_question(question, plan.get("region"))
        if coords is None:
            # The reference may be a yacimiento named in the question, not a capital:
            # resolve it from the corpus. Data-driven, and it works even with the
            # city table disabled.
            ref_name, ref_lat, ref_lon = _reference_from_corpus(question, collection, warnings)
            if ref_lat is not None and ref_lon is not None:
                city, coords = ref_name, (ref_lat, ref_lon)
        with_coords = [s for s in sites if s.get("lat") is not None]
        if coords:
            for s in with_coords:
                s["_distance_km"] = round(_haversine_km(coords[0], coords[1], s["lat"], s["lon"]), 1)
                s["_distance_from"] = city
            with_coords.sort(key=lambda s: s["_distance_km"],
                             reverse=(plan["order"].endswith("desc")))
        else:
            # No reference point for this question, so there is no distance and the
            # "furthest/closest" ordering cannot be computed. Say it out loud instead
            # of returning an arbitrary slice that still LOOKS ranked.
            warnings.append(
                "superlative_space without a reference point: no city from CITY_COORDS "
                "matched the question (or GEO_USE_CITY_COORDS=0) and the region fallback "
                "carries no coordinates; results are NOT ordered by distance")
            # Deterministic but deliberately NOT distance-based: the CSV has to be
            # reproducible, and the order must not leak an answer the code did not
            # compute — that is the whole point of the round without the table.
            with_coords.sort(key=lambda s: (-(s.get("bp_max") if s.get("bp_max") is not None else -1),
                                            s.get("yacimiento_id") or 0))
        if plan.get("radius_km") and coords:
            with_coords = [s for s in with_coords if s["_distance_km"] <= plan["radius_km"]]
        answer_docs = with_coords[:top_n]
        plan["city"] = city
    elif plan["kind"] == "lookup_by_date":
        wanted = {abs(int(d)) for d in plan["dates"]}

        def _site_has_date(s: Dict[str, Any]) -> bool:
            # bp_values path: every C14 date of the yacimiento (sign normalized).
            # bp_max path (collection without the backfill): the old exact match.
            values = s.get("bp_values")
            if values:
                return any(v is not None and abs(int(float(v))) in wanted for v in values)
            return s.get("bp_max") in wanted

        answer_docs = [s for s in sites if _site_has_date(s)][:top_n]
    elif plan["kind"] == "group_by_region":
        best: Dict[str, Dict[str, Any]] = {}
        for s in sites:
            reg = s.get("region")
            if not reg:
                continue
            if s.get("bp_max") is None:
                continue
            if reg not in best or s["bp_max"] > best[reg]["bp_max"]:
                best[reg] = s
        # One winner per region: truncating here would drop regions silently, and the
        # province -> community step downstream needs every region present.
        answer_docs = sorted(best.values(), key=lambda s: -s["bp_max"])

    # Empty-filter fix (2026-10-05): a filter that matched nothing used to return
    # docs=[] and run_rag() silently topped it up with similarity PDFs, so the CSV
    # said "filter:<kind>" while the model read unrelated files. Now the context IS
    # the warning and "empty" blocks the fallback: no evidence, said out loud.
    # Context ALWAYS goes through ctx_text(): coordinates must reach the prompt.
    structured_docs = [_row_to_document(r) for r in answer_docs]

    # Arreglo 5 (2026-10-07): attach the supporting articles, because Route B
    # returns GeoJSON only and the answer often lives in a PDF (Q4, Q7).
    pdf_rows = _supporting_pdfs(question, collection, plan, errors=errors)
    pdf_docs_all = [_row_to_document(r) for r in pdf_rows]
    pdf_block, pdf_docs = _budget_pdf_docs(pdf_docs_all, warnings)
    # Parche B (2026-10-16): per-file quota on the ordered list, applied after the
    # merge and before the cut to K. The context is untouched (it is assembled from
    # structured_docs + the PDF block); only the ranking list is trimmed so one file
    # (a GeoJSON holding every site, a PDF with many chunks) cannot monopolise the
    # top slots. Round-1 replay: scripts/simulate_cap_per_file.py.
    docs = cap_per_file(structured_docs + pdf_docs)

    empty_note = ""
    if not rows and not answer_docs:
        if pdf_docs:
            empty_note = ("NOTE: the geo filter matched 0 records for plan '%s' "
                          "(property %s, values %s). No structured documents are "
                          "provided; the article chunks below are a keyword search "
                          "over the PDFs, not filter results."
                          % (plan["kind"], plan.get("date_property", "-"),
                             ",".join(str(d) for d in plan.get("dates") or []) or "-"))
        else:
            empty_note = ("NOTE: the geo filter matched 0 records for plan '%s' "
                          "(property %s, values %s). No structured documents are provided "
                          "and similarity results are NOT substituted; answer only from "
                          "what the question itself states."
                          % (plan["kind"], plan.get("date_property", "-"),
                             ",".join(str(d) for d in plan.get("dates") or []) or "-"))
    elif not answer_docs:
        empty_note = ("NOTE: the geo filter matched %d records for plan '%s' but kept "
                      "0 usable sites. No structured documents are provided and "
                      "similarity results are NOT substituted; answer only from what "
                      "the question itself states." % (len(rows), plan["kind"]))

    parts = []
    if empty_note:
        parts.append(empty_note)
    if structured_docs:
        parts.append("\n\n".join(ctx_text(d) for d in structured_docs))
    if pdf_docs:
        parts.append(pdf_block)
    context = "\n\n".join(parts)
    if empty_note:
        warnings.append("  filter route EMPTY: " + empty_note[:140])
    if pdf_docs:
        warnings.append("  supporting PDFs: "
                        + str([d.metadata.get("filename") for d in pdf_docs]))
    # "empty" stays True only when BOTH the filter and the article search failed:
    # run_rag() drops docs[] and labels the row "VACIO" on that flag, and with
    # supporting articles attached that label would be a lie.
    return RouteResult(
        plan=plan,
        structured_docs=structured_docs,
        pdf_docs=pdf_docs,
        docs=docs,
        context=context,
        n_candidates=len(rows),
        n_sites=len(sites),
        n_pdf_docs=len(pdf_docs),
        empty=bool(empty_note) and not pdf_docs,
        note=empty_note,
        warnings=warnings,
        errors=errors,
        degraded=bool(errors),
        collection=collection,
    )


# Legacy collection constant. The eval collection (IdearqGTE_800_50_v2) backfill
# is DONE, so this no longer blocks the GTE route. It is kept for callers that
# still want the old default; retrieve_filtered() never applies it silently.
GEO_FILTER_COLLECTION = "IdearqMiniLM_800_50_v2"
