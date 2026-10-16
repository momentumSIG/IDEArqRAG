#!/usr/bin/env python3
"""Backfill the C14 filter columns (bp_values, dataciones_c_14) from c14_v2.geojson.

Why: `bp_max` only stores the OLDEST date of a site, so a question naming one
concrete date ("does Valencina hold 1377±23?") can never be answered by a range
filter. `bp_values` (number[]) and `dataciones_c_14` (text) make every single
date filterable.

Safety contract
---------------
* Targets ONE collection only (default: IdearqMiniLM_800_50_v2).
* DRY RUN IS THE DEFAULT. Nothing is written unless --apply is passed.
* Properties only: objects are PATCHed, never replaced, never re-embedded.
  Vectors are snapshotted before and after and compared byte-for-byte; the run
  FAILS if any vector changed.
* Cross-checks the parser against the bp_min/bp_max/c14_count that a previous
  enrichment already stored, and against the `dataciones C14:` section of
  `content`, so a silently broken regex cannot pass unnoticed.

Stdlib only (urllib + json): no Weaviate client, no GPU, no Slurm job.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEOJSON = PROJECT_ROOT / "data" / "ingesta" / "geojson" / "c14_v2.geojson"
DEFAULT_COLLECTION = "IdearqMiniLM_800_50_v2"
DEFAULT_URL = "http://c201-20:8080"
TARGET_FILE = "c14_v2.geojson"
DATACIONES_MARK = "Dataciones C14:"
NEW_PROPERTIES = ("bp_values", "dataciones_c_14")

# Same rules as parse_c14_values() in notebooks/indexing/rag-indexacion-mejorado.ipynb
_C14_BP_MARK = re.compile(r"\bB\s*\.?\s*P\s*\.?", re.I)
_C14_VALUE_RE = re.compile(r"(\d{3,5})\s*(?:±|\+/-)\s*(\d+)")


# ── parsing ─────────────────────────────────────────────────────────────────
def split_entries(raw: str):
    return [s.strip() for s in re.split(r"<br\s*/?>", raw or "", flags=re.I) if s.strip()]


def parse_c14_values(raw: str):
    """Every nominal B.P. age of the site, in source order ([] if none)."""
    values = []
    for entry in split_entries(raw):
        if not _C14_BP_MARK.search(entry):
            continue
        values.extend(int(v) for v, _ in _C14_VALUE_RE.findall(entry))
    return values


def dataciones_from_content(content: str) -> str:
    """The `Dataciones C14:` tail of a stored `content` string ('' if absent)."""
    if DATACIONES_MARK not in (content or ""):
        return ""
    return content.split(DATACIONES_MARK, 1)[1].strip()


# ── minimal HTTP helpers ────────────────────────────────────────────────────
class Weaviate:
    def __init__(self, url: str, timeout: int = 120):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def gql(self, query: str):
        payload = self._request("POST", "/v1/graphql", {"query": query})
        if payload.get("errors"):
            raise RuntimeError(json.dumps(payload["errors"])[:800])
        return payload["data"]

    def rest(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw)
            except Exception:
                return exc.code, {"message": raw.decode(errors="replace")}

    def _request(self, method, path, body):
        status, payload = self.rest(method, path, body)
        if status >= 400:
            raise RuntimeError("%s %s -> %s %s" % (method, path, status, payload))
        return payload

    def schema_props(self, collection: str):
        status, payload = self.rest("GET", "/v1/schema")
        if status >= 400:
            raise RuntimeError("cannot read schema: %s" % payload)
        for cls in payload.get("classes", []):
            if cls.get("class") == collection:
                return {p["name"]: p for p in cls.get("properties", [])}
        raise RuntimeError("collection %r not found in Weaviate" % collection)

    def ensure_schema(self, collection: str) -> list:
        """Add the two properties if missing. Returns the names actually added."""
        existing = self.schema_props(collection)
        added = []
        for name in NEW_PROPERTIES:
            if name in existing:
                continue
            if name == "bp_values":
                spec = {"name": "bp_values", "dataType": ["number[]"],
                        "indexFilterable": True,
                        "description": "Every C14 B.P. age of the site (filter column)."}
            else:
                # indexSearchable=False keeps this text OUT of the BM25 index so
                # hybrid ranking cannot change. Equal/Like still work.
                spec = {"name": "dataciones_c_14", "dataType": ["text"],
                        "indexFilterable": True, "indexSearchable": False,
                        "description": "Raw C14 date strings from c14_v2.geojson."}
            status, payload = self.rest(
                "POST", "/v1/schema/%s/properties" % collection, spec)
            if status >= 400:
                raise RuntimeError("adding property %s failed: %s" % (name, payload))
            added.append(name)
        return added


def fetch_geo_objects(client: Weaviate, collection: str, new_fields, page: int = 1000):
    """All GeoJSON objects of the collection, with their vectors, paginated.

    `new_fields` are requested only when they already exist in the schema, so a
    dry run still works before the properties have been created.
    """
    fields = ("yacimiento_id yacimiento_nombre bp_min bp_max c14_count content "
              + " ".join(new_fields)
              + " _additional{id vector}")
    out, offset = [], 0
    while True:
        query = (
            '{Get{%s(where:{path:["filename"],operator:Equal,valueText:"%s"},'
            'limit:%d,offset:%d){%s}}}'
            % (collection, TARGET_FILE, page, offset, fields)
        )
        rows = client.gql(query)["Get"][collection]
        out.extend(rows)
        if len(rows) < page:
            return out
        offset += page
        if offset > 20000:
            raise RuntimeError("pagination runaway (>20000 objects)")


def load_geojson(path: Path):
    """-> (site_props by yacimiento_id, stats). Duplicate features must match."""
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    by_id, differing = {}, []
    for feat in features:
        props = feat.get("properties") or {}
        sid = props.get("yacimiento_id", props.get("id"))
        if sid is None:
            continue
        raw = props.get("dataciones_c_14") or ""
        entry = {
            "dataciones_c_14": raw.strip() or None,
            "bp_values": parse_c14_values(raw),
            "bp_min": None, "bp_max": None, "c14_count": 0,
        }
        values = entry["bp_values"]
        if values:
            entry["bp_min"] = min(values)
            entry["bp_max"] = max(values)
            entry["c14_count"] = len(values)
        if sid in by_id and by_id[sid] != entry:
            differing.append(sid)
        by_id.setdefault(sid, entry)
    stats = {
        "features": len(features),
        "unique_yacimiento_id": len(by_id),
        "duplicate_features": len(features) - len(by_id),
        "sites_with_date": sum(1 for v in by_id.values() if v["bp_values"]),
        "sites_without_date": sum(1 for v in by_id.values() if not v["bp_values"]),
        "total_bp_values": sum(len(v["bp_values"]) for v in by_id.values()),
        "inconsistent_duplicate_ids": differing,
    }
    return by_id, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without it the script only reports (dry run).")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="explicit dry run (already the default; kept for callers "
                         "that pass it by name)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="target collection (default: %s)" % DEFAULT_COLLECTION)
    ap.add_argument("--url", default=DEFAULT_URL, help="Weaviate base URL")
    ap.add_argument("--geojson", default=str(DEFAULT_GEOJSON), help="source GeoJSON")
    ap.add_argument("--workers", type=int, default=8, help="parallel PATCH workers")
    ap.add_argument("--ensure-schema", action="store_true",
                    help="create bp_values/dataciones_c_14 if they are missing")
    args = ap.parse_args()
    if args.apply and args.dry_run:
        ap.error("--apply and --dry-run are mutually exclusive")
    args.apply = args.apply and not args.dry_run

    mode = "APPLY" if args.apply else "DRY RUN"
    print("=" * 74)
    print("C14 date backfill  |  mode=%s" % mode)
    print("  collection : %s" % args.collection)
    print("  url        : %s" % args.url)
    print("  geojson    : %s" % args.geojson)
    print("=" * 74)
    if args.collection != DEFAULT_COLLECTION:
        print("WARNING: you are targeting a collection other than %s"
              % DEFAULT_COLLECTION)

    client = Weaviate(args.url)

    # ── 1. schema ──────────────────────────────────────────────────────────
    props = client.schema_props(args.collection)
    vectorizer = None
    status, full = client.rest("GET", "/v1/schema")
    for cls in full.get("classes", []):
        if cls.get("class") == args.collection:
            vectorizer = cls.get("vectorizer")
    print("\n[1] Schema")
    print("    vectorizer            : %s" % vectorizer)
    print("    properties present    : %s"
          % {n: (n in props) for n in NEW_PROPERTIES})
    if vectorizer != "none":
        print("    FATAL: vectorizer is not 'none' -> re-vectorization risk. STOP.")
        return 2

    added = []
    present = [n for n in NEW_PROPERTIES if n in props]
    if len(present) < len(NEW_PROPERTIES):
        if args.ensure_schema:
            added = client.ensure_schema(args.collection)
            print("    added via --ensure-schema: %s" % added)
            props = client.schema_props(args.collection)
            present = [n for n in NEW_PROPERTIES if n in props]
        elif args.apply:
            print("    FATAL: missing properties %s. Re-run with --ensure-schema."
                  % [n for n in NEW_PROPERTIES if n not in props])
            return 2
        else:
            print("    note: properties not created yet %s — a dry run does not "
                  "need them. Use --apply --ensure-schema to create + fill."
                  % [n for n in NEW_PROPERTIES if n not in props])
    for name in present:
        spec = props[name]
        print("    %-18s dataType=%s indexFilterable=%s indexSearchable=%s"
              % (name, spec.get("dataType"), spec.get("indexFilterable"),
                 spec.get("indexSearchable")))

    # ── 2. GeoJSON ─────────────────────────────────────────────────────────
    site_props, gst = load_geojson(Path(args.geojson))
    print("\n[2] GeoJSON")
    for key in ("features", "unique_yacimiento_id", "duplicate_features",
                "sites_with_date", "sites_without_date", "total_bp_values"):
        print("    %-24s: %s" % (key, gst[key]))
    if gst["inconsistent_duplicate_ids"]:
        print("    FATAL: duplicate features disagree for ids %s"
              % gst["inconsistent_duplicate_ids"][:5])
        return 2
    print("    duplicate features    : consistent (every yacimiento_id is unique "
          "after dedup)")

    # ── 3. objects + cross-checks + vector snapshot ────────────────────────
    objects = fetch_geo_objects(client, args.collection, present)
    print("\n[3] Weaviate objects (filename=%s)" % TARGET_FILE)
    print("    objects read          : %d" % len(objects))

    unmapped = [o for o in objects
                if o.get("yacimiento_id") not in site_props]
    print("    unmapped objects      : %d" % len(unmapped))

    mismatch_bp = mismatch_content = missing_prop = null_count = 0
    plan, upto_date = [], 0
    before_vectors = {}
    for obj in objects:
        uid = obj["_additional"]["id"]
        before_vectors[uid] = obj["_additional"].get("vector")
        sid = obj.get("yacimiento_id")
        site = site_props.get(sid)
        if site is None:
            continue
        # cross-check the parser against a previously stored enrichment.
        # c14_count is 0 or absent (null) for sites without a parseable date:
        # the earlier backfill skipped those objects on purpose, so both spellings
        # mean the same thing ("no date") and must not count as a mismatch.
        stored_count = obj.get("c14_count")
        if stored_count is None:
            null_count += 1
            stored_count = 0
        if (obj.get("bp_min") != site["bp_min"] or obj.get("bp_max") != site["bp_max"]
                or stored_count != site["c14_count"]):
            mismatch_bp += 1
        if (dataciones_from_content(obj.get("content") or "")
                != (site["dataciones_c_14"] or "")):
            mismatch_content += 1
        new_vals = obj.get("bp_values")
        new_txt = obj.get("dataciones_c_14")
        if new_vals == site["bp_values"] and new_txt == site["dataciones_c_14"]:
            upto_date += 1
            continue
        if new_vals is None or new_txt is None:
            missing_prop += 1
        plan.append((args.collection, uid, sid,
                     {"bp_values": site["bp_values"],
                      "dataciones_c_14": site["dataciones_c_14"]}))

    print("    cross-check bp_min/max/count vs stored : %d mismatches" % mismatch_bp)
    print("    objects with c14_count stored as null  : %d  (no-date sites the earlier "
          "backfill skipped)" % null_count)
    print("    cross-check content vs geojson text    : %d mismatches" % mismatch_content)
    print("    objects with the new props still null  : %d" % missing_prop)
    print("    objects already up to date             : %d" % upto_date)
    print("    objects planned to update              : %d" % len(plan))
    if mismatch_bp or mismatch_content or unmapped:
        print("    FATAL: cross-checks failed. STOP without writing.")
        return 2

    if not args.apply:
        print("\n[4] DRY RUN")
        print("    writes performed : 0 (dry run is the default; pass --apply to write)")
        print("    would update     : %d objects" % len(plan))
        print("\nRESULT: dry-run OK, 0 changes applied.")
        return 0

    # ── 4. apply ───────────────────────────────────────────────────────────
    def patch_one(item):
        collection, uid, _sid, body = item
        status, payload = client.rest(
            "PATCH", "/v1/objects/%s/%s" % (collection, uid),
            {"properties": body})
        return status in (200, 204), payload

    t0 = time.time()
    results = list(ThreadPoolExecutor(args.workers).map(patch_one, plan))
    ok = [r for r, _ in results if r]
    failed = [(plan[i], p) for i, (r, p) in enumerate(results) if not r]
    print("\n[4] Apply")
    print("    updated (HTTP 204)    : %d" % len(ok))
    print("    failed                : %d" % len(failed))
    print("    elapsed               : %.1fs" % (time.time() - t0))
    for item, payload in failed[:5]:
        print("      FAILED uuid=%s : %s" % (item[1], str(payload)[:160]))
    if failed:
        print("RESULT: partial — %d/%d objects failed." % (len(failed), len(plan)))
        return 1

    # ── 5. verification ────────────────────────────────────────────────────
    after = fetch_geo_objects(client, args.collection, NEW_PROPERTIES)
    n_bp = sum(1 for o in after if o.get("bp_values"))
    n_txt = sum(1 for o in after if o.get("dataciones_c_14"))
    n_empty = sum(1 for o in after if o.get("bp_values") == [])
    after_vectors = {o["_additional"]["id"]: o["_additional"].get("vector")
                     for o in after}
    changed = [uid for uid, vec in before_vectors.items()
               if after_vectors.get(uid) != vec]
    print("\n[5] Verification")
    print("    objects re-read       : %d" % len(after))
    print("    bp_values non-empty   : %d" % n_bp)
    print("    bp_values empty []    : %d  (sites whose dates carry no B.P. marker)"
          % n_empty)
    print("    dataciones_c_14 set   : %d" % n_txt)
    print("    vectors compared      : %d" % len(before_vectors))
    print("    vectors CHANGED       : %d" % len(changed))
    if changed:
        print("    FATAL: vectors changed for %s" % changed[:5])
        return 1

    sample = next((o for o in after
                   if o.get("yacimiento_id") == 14371), None)
    if sample is None:
        print("    FATAL: could not re-read the Valencina sample object")
        return 1
    print("    sample (yacimiento_id=14371): bp_values[:6]=%s"
          % (sample.get("bp_values") or [])[:6])

    query = ('{Get{%s(where:{operator:And,operands:['
             '{path:["filename"],operator:Equal,valueText:"%s"},'
             '{path:["bp_values"],operator:ContainsAny,valueNumber:[1377]}]},'
             'limit:5){yacimiento_id yacimiento_nombre bp_max}}}'
             % (args.collection, TARGET_FILE))
    try:
        hits = client.gql(query)["Get"][args.collection]
        print("    proof query ContainsAny valueNumber:[1377] -> %d hit(s)" % len(hits))
        for hit in hits:
            print("      * id=%s | %s | bp_max=%s"
                  % (hit.get("yacimiento_id"), hit.get("yacimiento_nombre"),
                     hit.get("bp_max")))
        if not hits:
            print("    FATAL: proof query returned nothing")
            return 1
    except Exception as exc:
        print("    FATAL: proof query failed: %s" % exc)
        return 1

    print("\nRESULT: apply OK — %d objects updated, 0 failed, 0 vectors changed."
          % len(ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
