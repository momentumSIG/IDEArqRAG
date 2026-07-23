"""
Explorador de co-autorías del grafo bibliográfico IDEArq.

Genera un HTML interactivo con pyvis mostrando:
  - Top N autores (foaf:Person)
  - Sus artículos (bibo:AcademicArticle / bibo:Book)
  - Las revistas donde publican (bibo:Journal)

Uso:
  /home/raglinux/env_rag/bin/python explorar-coautorias.py
"""
import os, sys
from pathlib import Path
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import FOAF, RDF
import networkx as nx
from pyvis.network import Network

# ── Configuración ──────────────────────────────────────────────────────────
TOP_N_AUTHORS = 30
MIN_ARTICLES_PER_AUTHOR = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "biblio-graph"
NT_FILE = DATA_DIR / "idearq-graph-instances.nt"
OUTPUT_HTML = Path(__file__).parent / "explorar-coautorias.html"

BIBO = Namespace("http://purl.org/ontology/bibo/")
DC_  = Namespace("http://purl.org/dc/terms/")

# ── Cargar grafo ───────────────────────────────────────────────────────────
print(f"Cargando {NT_FILE} ...")
g = Graph()
g.parse(str(NT_FILE), format="nt")
print(f"  {len(g):,} triples cargados")

# ── SPARQL: top N autores por nº de publicaciones ──────────────────────────
sparql_top = f"""
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?author ?authorName (COUNT(?article) AS ?n) WHERE {{
  ?article dc:creator ?author .
  ?author foaf:name ?authorName .
}}
GROUP BY ?author ?authorName
ORDER BY DESC(?n)
LIMIT {TOP_N_AUTHORS}
"""

top_authors = list(g.query(sparql_top))
print(f"\nTop {len(top_authors)} autores:")
for row in top_authors:
    print(f"  {row.authorName:45s}  {row.n:>3} pubs")

author_uris = {str(row.author) for row in top_authors}

# ── SPARQL: artículos de estos autores + revistas ──────────────────────────
author_uris_str = " ".join(f"<{u}>" for u in author_uris)

sparql_details = f"""
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?authorUri ?authorName ?articleUri ?articleTitle ?journalName WHERE {{
  VALUES ?authorUri {{ {author_uris_str} }}
  ?articleUri dc:creator ?authorUri ;
              dc:title ?articleTitle .
  ?authorUri foaf:name ?authorName .
  OPTIONAL {{
    ?articleUri dc:isPartOf ?journal .
    ?journal dc:title ?journalName .
  }}
}}
ORDER BY ?authorName ?articleTitle
"""

details = list(g.query(sparql_details))
print(f"\n  {len(details)} filas recuperadas (autor → artículo → revista)")

# Filtrar autores con menos de MIN_ARTICLES_PER_AUTHOR artículos en el subset
from collections import Counter
author_article_count = Counter(str(r.authorUri) for r in details)
valid_authors = {a for a, c in author_article_count.items() if c >= MIN_ARTICLES_PER_AUTHOR}
details = [r for r in details if str(r.authorUri) in valid_authors]
print(f"  Después de filtrar (min {MIN_ARTICLES_PER_AUTHOR} artículos): {len(details)} filas")

# ── Construir grafo networkx ──────────────────────────────────────────────
G = nx.Graph()

author_degree = Counter()
article_authors = {}

for row in details:
    author_uri = str(row.authorUri)
    author_name = str(row.authorName)
    article_uri = str(row.articleUri)
    article_title = str(row.articleTitle)
    journal_name = str(row.journalName) if row.journalName else None

    author_degree[author_name] += 1
    article_authors.setdefault(article_title, []).append(author_name)

    # Nodos
    if not G.has_node(author_name):
        G.add_node(author_name, type="author", label=author_name,
                   size=10, color="#4A90D9", title=f"Autor: {author_name}")

    if not G.has_node(article_title):
        short = article_title[:60] + "..." if len(article_title) > 60 else article_title
        G.add_node(article_title, type="article", label=short,
                   size=5, color="#7ED321", title=article_title)

    G.add_edge(author_name, article_title, color="#999", width=1)

    # Opción C: solo crear nodo journal si existe (sin hub "Sin journal")
    if journal_name and journal_name != "Sin journal":
        journal_key = f"📚 {journal_name}"
        if not G.has_node(journal_key):
            G.add_node(journal_key, type="journal", label=journal_name,
                       size=8, color="#F5A623", title=f"Revista: {journal_name}")
        G.add_edge(article_title, journal_key, color="#B0D4F1", width=1)

# Ajustar tamaño de nodos por grado
for node in G.nodes:
    deg = G.degree(node)
    ntype = G.nodes[node]["type"]
    if ntype == "author":
        G.nodes[node]["size"] = max(10, min(deg * 3, 40))
    elif ntype == "journal":
        G.nodes[node]["size"] = max(12, min(deg * 2, 35))
    elif ntype == "article":
        G.nodes[node]["size"] = max(4, min(deg * 2, 15))

print(f"\nGrafo networkx: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
type_counts = Counter(G.nodes[n]["type"] for n in G.nodes)
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:12s}: {c}")

# ── Generar HTML con pyvis ─────────────────────────────────────────────────
print(f"\nGenerando visualización pyvis...")
net = Network(
    height="900px",
    width="100%",
    directed=False,
    notebook=False,
    bgcolor="#1a1a2e",
    font_color="#ffffff",
    cdn_resources="remote"
)
net.from_nx(G)

# Ajustar opciones de physics
net.toggle_physics(True)

# Configurar leyenda
net.set_options("""
{
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "centralGravity": 0.01,
      "springLength": 120,
      "springConstant": 0.05,
      "damping": 0.4
    },
    "stabilization": {
      "enabled": true,
      "iterations": 200,
      "fit": true
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "hideEdgesOnDrag": true
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "dynamic"
    }
  }
}
""")

net.save_graph(str(OUTPUT_HTML))
html_size = OUTPUT_HTML.stat().st_size / 1024
print(f"\n✅ HTML guardado: {OUTPUT_HTML}")
print(f"   Tamaño: {html_size:.1f} KB")
print(f"   Ábrelo con doble clic en el navegador.")
