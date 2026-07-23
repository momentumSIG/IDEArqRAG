#!/usr/bin/env python3
"""Generate the notebook rag-graph-ontologia-mapeo.ipynb"""
import json

def mk_md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def mk_code(source):
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "source": [l + "\n" for l in lines],
        "outputs": []
    }

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.11"}
    },
    "cells": [

mk_md("""# RAG-IDEArq — Mapeo de Ontología BIBO con Morph-KGC

Generación de un grafo RDF a partir de `referencias_clean.csv` usando la ontología
`idearq-biblio-v6.ttl` (BIBO 1.3 + extensiones dc, foaf, org, skos).

**Herramienta**: [Morph-KGC](https://github.com/morph-kgc/morph-kgc) (YARRRML → RDF)

**Granularidad del grafo**:
- Documentos (`bibo:AcademicArticle`, `bibo:Book`, `bibo:Series`, `bibo:Proceedings`, `bibo:Document`)
- Autores (`foaf:Person`) — extraídos del campo `autores` (formato `1|Name; 2|Name`)
- Revistas (`bibo:Journal`) — deduplicadas por ISSN
- Temas (`skos:Concept`) — desde `sjr_materia`

**Salida**:
- `data/biblio-graph/idearq-graph-instances.ttl` (Turtle)
- `data/biblio-graph/idearq-graph-instances.nt` (N-Triples)
- `data/biblio-graph/_derived/{docs,authors,journals,subjects}.csv` (intermedios)"""),

mk_code("""import os, sys, re, json, textwrap
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

import pandas as pd
import morph_kgc
import rdflib
from rdflib import Graph, Namespace, URIRef, RDF, RDFS
from rdflib.namespace import DC, FOAF, SKOS, XSD
from importlib.metadata import version as pkg_version

print(f"Project root: {PROJECT_ROOT}")
print(f"pandas  {pd.__version__}")
print(f"rdflib  {rdflib.__version__}")
print(f"morph-kgc {pkg_version('morph-kgc')}")

DATA_DIR = PROJECT_ROOT / "data" / "biblio-graph"
DERIVED_DIR = DATA_DIR / "_derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_FILE = Path.cwd() / "mapping-idearq.ini"
OUTPUT_TTL = DATA_DIR / "idearq-graph-instances.ttl"
OUTPUT_NT  = DATA_DIR / "idearq-graph-instances.nt"

print(f"Data dir:    {DATA_DIR}")
print(f"Derived dir: {DERIVED_DIR}")
print(f"Mapping:     {MAPPING_FILE}")
print(f"Output TTL:  {OUTPUT_TTL}")
print(f"Output NT:   {OUTPUT_NT}")"""),

mk_code("""df = pd.read_csv(DATA_DIR / "referencias_clean.csv")

print(f"Shape: {df.shape}")
print(f"\\nColumnas: {list(df.columns)}")
print(f"\\nNulos por columna:")
print(df.isna().sum().to_string())
print(f"\\ntipo_fuente value_counts:")
print(df["tipo_fuente"].value_counts(dropna=False).to_string())
print(f"\\nMuestra (3 primeras filas, columnas clave):")
print(df[["id", "title", "year", "tipo_fuente", "doi", "issn", "fuente", "autores"]].head(3).to_string())"""),

mk_code("""BIBO = Namespace("http://purl.org/ontology/bibo/")
DC_  = Namespace("http://purl.org/dc/terms/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
ORG  = Namespace("https://www.w3.org/ns/org#")
IDEARQ = Namespace("http://idearq.org/resource/")

onto = Graph()
onto.parse(DATA_DIR / "idearq-biblio-v6.ttl", format="turtle")

print(f"Ontología: {len(onto)} triples")

classes = list(onto.objects(None, RDF.type))
from collections import Counter
class_counts = Counter(str(c) for c in classes)
print(f"\\nClases definidas: {len(set(str(c) for c in classes))}")

bibo_doc = BIBO.Document
subclasses = list(onto.subjects(RDFS.subClassOf, bibo_doc))
print(f"\\nSubclases de bibo:Document ({len(subclasses)}):")
for sc in sorted(subclasses, key=str)[:15]:
    label = onto.value(sc, RDFS.label)
    print(f"  {sc.split('/')[-1]:30s} {label or ''}")

obj_props = [
    "dc:creator", "dc:isPartOf", "dc:hasPart", "dc:subject",
    "dc:publisher", "bibo:authorList", "bibo:editorList",
    "foaf:homepage", "org:hasMember", "org:memberOf",
]
print(f"\\nObject properties clave:")
for p in obj_props:
    prefix, local = p.split(":")
    ns_map = {"dc": DC_, "bibo": BIBO, "foaf": FOAF, "org": ORG}
    prop = ns_map.get(prefix, Namespace(""))[local]
    domain = onto.value(prop, RDFS.domain)
    range_ = onto.value(prop, RDFS.range)
    print(f"  {p:25s} domain={domain} range={range_}")"""),

mk_code("""BASE_URI = "http://idearq.org/resource/"

TIPO_FUENTE_MAP = {
    "JournalArticle":   ("article",          str(BIBO.AcademicArticle)),
    "Book":             ("book",             str(BIBO.Book)),
    "BookSeries":       ("bookseries",       str(BIBO.Series)),
    "ProceedingsPaper": ("proceedings",      str(BIBO.Proceedings)),
}
DEFAULT_TYPE = ("document", str(BIBO.Document))

def get_uri_parts(tipo_fuente):
    return TIPO_FUENTE_MAP.get(tipo_fuente, DEFAULT_TYPE)

print("Mapping tipo_fuente → (uri_segment, rdf_type):")
for k, v in {**TIPO_FUENTE_MAP, "Expression/NaN": DEFAULT_TYPE}.items():
    print(f"  {k:20s} → {v}")"""),

mk_code("""import unicodedata

def slugify(name):
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name[:60] if name else None

def normalize_issn(issn):
    if pd.isna(issn):
        return None
    issn = str(issn).replace("-", "").replace(" ", "")
    if len(issn) == 8:
        return f"{issn[:4]}-{issn[4:]}"
    return None

def normalize_doi(doi):
    if pd.isna(doi):
        return None
    doi = str(doi).strip()
    if doi.startswith("http"):
        doi = doi.split("doi.org/")[-1].rstrip("/")
    return doi if doi else None

def parse_authors(autores_str):
    if pd.isna(autores_str):
        return []
    authors = []
    for part in str(autores_str).split(";"):
        part = part.strip()
        if not part:
            continue
        if "|" in part:
            order_str, name = part.split("|", 1)
            try:
                order = int(order_str.strip())
            except ValueError:
                order = 0
        else:
            order = 0
            name = part
        name = name.strip()
        if name:
            authors.append({"order": order, "name": name})
    return authors

# --- Preprocesado ---
df = df.rename(columns={"id": "doc_id"})
df["issn_clean"] = df["issn"].apply(normalize_issn)
df["doi_clean"] = df["doi"].apply(normalize_doi)
df["year_str"] = df["year"].apply(lambda x: str(int(x)) if pd.notna(x) else None)

df[["uri_base", "rdf_type"]] = df["tipo_fuente"].apply(
    lambda x: pd.Series(get_uri_parts(x))
)
df["doc_uri"] = BASE_URI + df["uri_base"] + "/" + df["doc_id"].astype(str)

df["journal_iri"] = df["issn_clean"].apply(
    lambda x: f"{BASE_URI}journal/{x}" if pd.notna(x) else None
)
df["sjr_slug"] = df["sjr_materia"].apply(
    lambda x: slugify(str(x)) if pd.notna(x) else None
)
df["subject_iri"] = df["sjr_slug"].apply(
    lambda x: f"{BASE_URI}subject/{x}" if pd.notna(x) else None
)

# --- docs.csv ---
docs_cols = [
    "doc_id", "doc_uri", "uri_base", "rdf_type", "title", "year_str",
    "doi_clean", "issn_clean", "abstract", "url_publicacion",
    "journal_iri", "subject_iri", "sjr_slug", "sjr_materia",
]
docs_df = df[docs_cols].copy()
docs_df.to_csv(DERIVED_DIR / "docs.csv", index=False)
print(f"docs.csv: {len(docs_df)} filas")

# --- authors.csv ---
authors_rows = []
for _, row in df.iterrows():
    for author in parse_authors(row["autores"]):
        authors_rows.append({
            "doc_id": row["doc_id"],
            "article_uri": row["doc_uri"],
            "author_name": author["name"],
            "author_order": author["order"],
            "author_slug": slugify(author["name"]),
        })
authors_df = pd.DataFrame(authors_rows)
authors_df.to_csv(DERIVED_DIR / "authors.csv", index=False)
print(f"authors.csv: {len(authors_df)} filas ({authors_df['author_slug'].nunique()} autores únicos)")

# --- journals.csv ---
journals_df = df[df["issn_clean"].notna()][["issn_clean", "fuente"]].copy()
journals_df = journals_df.rename(columns={"issn_clean": "issn"})
journals_df = journals_df.drop_duplicates(subset="issn")
journals_df["fuente"] = journals_df["fuente"].fillna("Unknown Journal")
journals_df.to_csv(DERIVED_DIR / "journals.csv", index=False)
print(f"journals.csv: {len(journals_df)} revistas únicas")

# --- subjects.csv ---
subjects_df = df[df["sjr_slug"].notna()][["sjr_slug", "sjr_materia"]].copy()
subjects_df = subjects_df.drop_duplicates(subset="sjr_slug")
subjects_df.to_csv(DERIVED_DIR / "subjects.csv", index=False)
print(f"subjects.csv: {len(subjects_df)} temas únicos")

print("\\nCSVs derivados guardados en:", DERIVED_DIR)"""),

mk_code("""# --- Config file (INI) ---
config_ini = f\"\"\"
[DataSource]
mappings: {Path.cwd() / 'mapping-idearq.yarrrml'}
\"\"\".strip()

with open(MAPPING_FILE, "w") as f:
    f.write(config_ini)
print(f"Config file written to: {MAPPING_FILE}")

# --- YARRRML mapping file ---
yarrrml_file = Path.cwd() / "mapping-idearq.yarrrml"

yarrrml = textwrap.dedent(f\"\"\"
prefixes:
  bibo: http://purl.org/ontology/bibo/
  dc: http://purl.org/dc/terms/
  foaf: http://xmlns.com/foaf/0.1/
  skos: http://www.w3.org/2004/02/skos/core#
  xsd: http://www.w3.org/2001/XMLSchema#
  idearq: {BASE_URI}
sources:
  docs:
    access: {DERIVED_DIR / 'docs.csv'}
    referenceForm: csv
  authors:
    access: {DERIVED_DIR / 'authors.csv'}
    referenceForm: csv
  journals:
    access: {DERIVED_DIR / 'journals.csv'}
    referenceForm: csv
  subjects:
    access: {DERIVED_DIR / 'subjects.csv'}
    referenceForm: csv
mappings:
  documents:
    sources: docs
    s: http://idearq.org/resource/$(uri_base)/$(doc_id)
    po:
      - [a, $(rdf_type)~iri]
      - [dc:title, $(title)]
      - [dc:issued, $(year_str), xsd:gYear]
      - [bibo:doi, $(doi_clean)]
      - [bibo:issn, $(issn_clean)]
      - [bibo:abstract, $(abstract)]
      - [bibo:uri, $(url_publicacion)]
      - [dc:identifier, $(doc_id)]
      - [dc:isPartOf, $(journal_iri)~iri]
      - [dc:subject, $(subject_iri)~iri]
  article_authors:
    sources: authors
    s: $(article_uri)
    po:
      - [dc:creator, http://idearq.org/resource/author/$(author_slug)~iri]
  persons:
    sources: authors
    s: http://idearq.org/resource/author/$(author_slug)
    po:
      - [a, foaf:Person]
      - [foaf:name, $(author_name)]
  journals:
    sources: journals
    s: http://idearq.org/resource/journal/$(issn)
    po:
      - [a, bibo:Journal]
      - [dc:title, $(fuente)]
      - [bibo:issn, $(issn)]
  subjects:
    sources: subjects
    s: http://idearq.org/resource/subject/$(sjr_slug)
    po:
      - [a, skos:Concept]
      - [skos:prefLabel, $(sjr_materia)]
\"\"\").strip()

with open(yarrrml_file, "w") as f:
    f.write(yarrrml)

print(f"YARRRML file written to: {yarrrml_file}")
print(f"Size: {len(yarrrml)} bytes")
print("\\n--- YARRRML Preview ---")
print(yarrrml[:1200])
print("...")"""),

mk_code("""print("Materializing RDF with Morph-KGC...")
g = morph_kgc.materialize(str(MAPPING_FILE))
print(f"\\nGraph generated: {len(g)} triples")

# Quick sanity checks
from rdflib.namespace import RDF
BIBO = Namespace("http://purl.org/ontology/bibo/")
DC_  = Namespace("http://purl.org/dc/terms/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

type_counts = {}
for obj in set(g.objects(None, RDF.type)):
    type_counts[str(obj).split("/")[-1]] = sum(1 for _ in g.subjects(RDF.type, obj))

print("\\nTriples by type:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:30s} {c:>6,}")

print(f"\\nTotal unique subjects: {len(set(g.subjects()))}")
print(f"Total unique predicates: {len(set(g.predicates()))}")
print(f"Total unique objects: {len(set(g.objects()))}")"""),

mk_code("""print(f"Writing Turtle → {OUTPUT_TTL}")
g.serialize(destination=str(OUTPUT_TTL), format="turtle")
print(f"  Size: {OUTPUT_TTL.stat().st_size / 1024:.1f} KB")

print(f"\\nWriting N-Triples → {OUTPUT_NT}")
g.serialize(destination=str(OUTPUT_NT), format="nt")
print(f"  Size: {OUTPUT_NT.stat().st_size / 1024:.1f} KB")

# Verify round-trip
g2 = Graph()
g2.parse(str(OUTPUT_NT), format="nt")
assert len(g2) == len(g), f"Mismatch: {len(g2)} != {len(g)}"
print(f"\\nRound-trip verification OK: {len(g2)} triples match")"""),

mk_code("""from rdflib import Namespace
BIBO = Namespace("http://purl.org/ontology/bibo/")
DC_  = Namespace("http://purl.org/dc/terms/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

def run_sparql(query, desc=""):
    if desc:
        print(f"\\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    results = list(g.query(query))
    if not results:
        print("  (no results)")
        return results
    # Print as table
    vars_ = results[0].labels if hasattr(results[0], 'labels') else [str(v) for v in results[0]]
    header = " | ".join(str(v) for v in vars_)
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for row in results[:15]:
        vals = []
        for v in row:
            s = str(v)
            if len(s) > 50:
                s = s[:47] + "..."
            vals.append(s)
        print(f"  {' | '.join(vals)}")
    if len(results) > 15:
        print(f"  ... ({len(results)} total)")
    return results

# 1. Total documents by type
run_sparql(\"\"\"
PREFIX bibo: <http://purl.org/ontology/bibo/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?type (COUNT(?s) AS ?count) WHERE {
  ?s rdf:type ?type .
  FILTER(STRSTARTS(STR(?type), "http://purl.org/ontology/bibo/"))
}
GROUP BY ?type ORDER BY DESC(?count)
\"\"\", "1. Documentos por tipo BIBO")

# 2. Top 10 journals
run_sparql(\"\"\"
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX bibo: <http://purl.org/ontology/bibo/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?title (COUNT(?doc) AS ?n) WHERE {
  ?doc dc:isPartOf ?j .
  ?j rdf:type bibo:Journal ;
     dc:title ?title .
}
GROUP BY ?title ORDER BY DESC(?n) LIMIT 10
\"\"\", "2. Top 10 revistas por nº de artículos")

# 3. 5 artículos más antiguos con DOI
run_sparql(\"\"\"
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX bibo: <http://purl.org/ontology/bibo/>
SELECT ?title ?year ?doi WHERE {
  ?s dc:title ?title ;
     dc:issued ?year ;
     bibo:doi ?doi .
}
ORDER BY ?year LIMIT 5
\"\"\", "3. 5 artículos más antiguos con DOI")

# 4. Autores con más publicaciones
run_sparql(\"\"\"
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name (COUNT(?doc) AS ?n) WHERE {
  ?doc dc:creator ?author .
  ?author foaf:name ?name .
}
GROUP BY ?name ORDER BY DESC(?n) LIMIT 10
\"\"\", "4. Top 10 autores por nº de publicaciones")

# 5. Stats summary
run_sparql(\"\"\"
PREFIX bibo: <http://purl.org/ontology/bibo/>
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT
  (COUNT(DISTINCT ?doc) AS ?docs)
  (COUNT(DISTINCT ?author) AS ?authors)
  (COUNT(DISTINCT ?journal) AS ?journals)
  (COUNT(DISTINCT ?subject) AS ?subjects)
WHERE {
  { SELECT (COUNT(DISTINCT ?doc) AS ?docs) WHERE { ?doc rdf:type ?t . FILTER(STRSTARTS(STR(?t), STR(bibo:))) } }
  { SELECT (COUNT(DISTINCT ?author) AS ?authors) WHERE { ?author rdf:type foaf:Person } }
  { SELECT (COUNT(DISTINCT ?journal) AS ?journals) WHERE { ?journal rdf:type bibo:Journal } }
  { SELECT (COUNT(DISTINCT ?subject) AS ?subjects) WHERE { ?subject rdf:type skos:Concept } }
}
\"\"\", "5. Resumen del grafo")

print(f"\\n{'='*60}")
print("  GRAFO GENERADO CON ÉXITO")
print(f"{'='*60}")
print(f"  Triples totales: {len(g):,}")
print(f"  Archivo TTL: {OUTPUT_TTL}")
print(f"  Archivo NT:  {OUTPUT_NT}")
print(f"  Mapping YARRRML: {MAPPING_FILE}")
print(f"  CSVs derivados: {DERIVED_DIR}/")"""),

    ]
}

output_path = "/home/raglinux/RAG/notebooks/graph/rag-graph-ontologia-mapeo.ipynb"
with open(output_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Notebook written to {output_path}")
print(f"Total cells: {len(nb['cells'])}")
