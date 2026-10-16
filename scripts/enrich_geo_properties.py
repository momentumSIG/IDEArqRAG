#!/usr/bin/env python3
"""Populate the structured GeoJSON columns on every GeoJSON object of one collection.

This is the supported entry point for the enrichment that
`backfill_geo_properties()` performs in `notebooks/indexing/rag-indexacion-mejorado.ipynb`
(cells 6 + 9). It existed only as an ad-hoc script outside the repo, so the
collections drifted apart: IdearqGTE_800_50_v2 got `region`/`bp_*`/`c14_count`
while Idearqe5largeinstruct_800_50_v2 was left with them all null, and Route B
filters on `region`, so the route silently returns nothing on the un-enriched
collection. Running the same function on every collection keeps the evaluation
comparable.

Writes properties only (PATCH /v1/objects/...). It never re-embeds: vectors are
untouched, so no re-indexing is needed.

Usage:
    python scripts/enrich_geo_properties.py --collection Idearqe5largeinstruct_800_50_v2
    python scripts/enrich_geo_properties.py --collection <name> --url http://host:8080
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEO_NOTEBOOK = PROJECT_ROOT / "notebooks" / "indexing" / "rag-indexacion-mejorado.ipynb"
SERVICES_ENV = Path(
    "/mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/services.env"
)
# Cells 6 (helpers) and 9 (the backfill function) of the indexing notebook.
GEO_HELPER_CELLS = (6, 9)
DEFAULT_URL = "http://c202-22:8080"


def load_services_env(path: Path = SERVICES_ENV) -> None:
    """services.env is the single source of truth for the live host.

    It OVERRIDES the ambient environment on purpose: an interactive session can
    carry a stale WEAVIATE_URL (a dead host) and `setdefault` would keep it,
    producing a confusing "Connection refused" instead of using the live host.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def load_backfill_function(url: str):
    """Import `backfill_geo_properties` from the indexing notebook.

    The notebook hardcodes a dead host in one cell, so the URL is injected after
    exec instead of being trusted from the cell source.
    """
    if not GEO_NOTEBOOK.exists():
        raise SystemExit(f"FATAL: no encuentro {GEO_NOTEBOOK}")
    notebook = json.loads(GEO_NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict = {}
    for index in GEO_HELPER_CELLS:
        try:
            source = "".join(notebook["cells"][index]["source"])
        except IndexError:
            raise SystemExit(f"FATAL: la celda {index} ya no existe en el notebook")
        exec(compile(source, f"{GEO_NOTEBOOK.name}#cell{index}", "exec"), namespace)
    if "backfill_geo_properties" not in namespace:
        raise SystemExit("FATAL: backfill_geo_properties no definida en las celdas 6+9")
    namespace["GEO_WEAVIATE_URL"] = url
    return namespace["backfill_geo_properties"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", required=True,
                        help="colección Weaviate a enriquecer, p.ej. Idearqe5largeinstruct_800_50_v2")
    parser.add_argument("--url", default=None,
                        help="URL base de Weaviate (por defecto: services.env, luego %s)" % DEFAULT_URL)
    parser.add_argument("--workers", type=int, default=8, help="PATCH en paralelo (default 8)")
    args = parser.parse_args()

    load_services_env()
    url = args.url or os.environ.get("WEAVIATE_URL") or DEFAULT_URL
    print("collection :", args.collection)
    print("weaviate   :", url)
    print("notebook   :", GEO_NOTEBOOK.name, "celdas", GEO_HELPER_CELLS)

    backfill = load_backfill_function(url)
    report = backfill(args.collection, workers=args.workers)

    print("\nREPORT")
    for key, value in report.items():
        print("  %-24s %s" % (key, value))

    patched = report.get("objects_patched") or 0
    ok = (patched > 0
          and report.get("patched_ok") == patched
          and (report.get("coverage_bp_max") or 0) > 0
          and (report.get("coverage_region") or report.get("coverage_bp_max")))
    print("\nRESULT:", "OK" if ok else "PROBLEM — revisa la cobertura de arriba")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
