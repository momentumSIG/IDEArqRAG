# Roadmap: Integración LDA + Grafo Bibliográfico

## Recursos disponibles en RAG/

### LDA Topic Modeling
- **Ubicación:** `RAG/lda/`
- **Modelos:** `models/lda_K9_BEST/` (9 tópicos), `models/lda_K5_BEST/` (5 tópicos)
- **Dictionary:** `data/06_dictionary.gensim`
- **Outputs:**
  - `outputs/papers_topics.csv` - 513 papers con `topic_dominant`, `topic_dominant_weight`, `topic_distribution`
  - `outputs/topics_keywords.csv` - Keywords por tópico
  - `outputs/topics_for_graph.json` - Listo para Neo4j

**Tópicos LDA (K=9):**

| Topic | Papers | Keywords | Temática |
|---|---|---|---|
| T0 | 145 | date, bone, human, neolithic | Bioarqueología / Dataciones |
| T1 | 41 | del, que, los, las | Contaminado (stopwords) |
| T2 | 104 | cueva, cova, sílex, lítico | Industria lítica / Cuevas |
| T3 | 42 | lisboa, anta, portugal | Arqueología portuguesa |
| T4 | 181 | bronce, funerario, asentamiento | Bronce/Funerario |
| T5 | 30 | ... | ... |
| T6 | 62 | ... | ... |
| T7 | 212 | ... | ... |
| T8 | 20 | ... | ... |

### Grafo Bibliográfico RDF
- **Ubicación:** `RAG/data/biblio-graph/`
- **Archivos:**
  - `idearq-biblio-v6.ttl`
  - `idearq-graph-instances-dedup-v2.ttl`
  - `idearq-graph-instances.ttl`
- **Ontologías:** dcterms, pro, foaf, fabio, bibo
- **Contenido:** Documentos, autores, fechas, títulos, relaciones bibliográficas

---

## Fase 1: Añadir LDA Topics a los metadatos de Weaviate

### 1.1. Modificar schema de Weaviate

Añadir `topic` (INT) a `INDEX_PROPERTIES_FULL` en `src/config.py`:

```python
INDEX_PROPERTIES_FULL = INDEX_PROPERTIES + [
    ...
    ("topic", "INT"),  # NUEVO: tópico LDA dominante
]
```

### 1.2. Modificar indexación para asignar tópicos

Durante la indexación de PDFs:

```python
import pandas as pd
import gensim

# Cargar LDA model y dictionary
lda_model = gensim.models.LdaModel.load("data/lda/models/lda_K9_BEST/lda_K9_BEST.model")
dictionary = gensim.corpora.Dictionary.load("data/lda/06_dictionary.gensim")

# Cargar mapeo filename → topic
lda_topics = pd.read_csv("data/lda/outputs/papers_topics.csv")
topic_map = dict(zip(lda_topics['filename'], lda_topics['topic_dominant']))

# Durante la indexación
for doc in docs:
    filename = doc.metadata['filename']
    doc.metadata['topic'] = int(topic_map.get(filename, -1))  # -1 si no tiene tópico
```

### 1.3. Reindexar

Ejecutar el notebook de indexación con el nuevo campo `topic`.

---

## Fase 2: Añadir nodo `topic_match` al grafo

### 2.1. Nuevo nodo

```python
def node_topic_match(state: EvalState) -> dict:
    """Clasifica la pregunta en tópicos LDA."""
    import gensim
    lda_model = gensim.models.LdaModel.load("data/lda/models/lda_K9_BEST/lda_K9_BEST.model")
    dictionary = gensim.corpora.Dictionary.load("data/lda/06_dictionary.gensim")
    
    # Preprocesar pregunta (mismo pipeline que LDA)
    tokens = preprocess_question(state["question"])
    bow = dictionary.doc2bow(tokens)
    
    # Inferir distribución de tópicos
    topic_dist = lda_model[bow]
    
    # Filtrar tópicos con probabilidad > threshold
    threshold = 0.15
    relevant_topics = [int(t) for t, p in topic_dist if p > threshold]
    
    return {"lda_topics": relevant_topics}
```

### 2.2. Modificar `node_retrieve` para filtrar por tópicos

```python
def node_retrieve(state: EvalState) -> dict:
    """Retrieve from Weaviate with optional LDA topic filtering."""
    # ... existing code ...
    
    topics = state.get("lda_topics", [])
    if topics:
        # Filtrar por tópicos LDA
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {
                    "operator": "Or",
                    "operands": [
                        {"path": ["topic"], "operator": "Equal", "valueInt": t}
                        for t in topics
                    ]
                }
            }
        )
    else:
        # Sin filtro
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    docs = retriever.invoke(state["question"])
    return {"retrieved_docs": docs}
```

### 2.3. Actualizar grafo

```python
builder.add_node("topic_match", node_topic_match)
builder.add_edge("retrieve", "topic_match")
builder.add_edge("topic_match", "rerank")
```

---

## Fase 3: Añadir grafo bibliográfico RDF

### 3.1. Nuevo nodo

```python
def node_biblio_query(state: EvalState) -> dict:
    """Query bibliographic RDF graph."""
    from rdflib import Graph
    
    rdf_graph = Graph()
    rdf_graph.parse("data/biblio-graph/idearq-biblio-v6.ttl", format="turtle")
    
    # Normalizar pregunta
    qn = _normalize(state["question"])
    years = _extract_years(state["question"])
    tokens = _extract_keywords(qn)
    
    # SPARQL query
    sparql = """
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX pro: <http://purl.org/spar/pro/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    
    SELECT DISTINCT ?doc ?title ?date ?authorName
    WHERE {
      ?doc dcterms:title ?title .
      OPTIONAL { ?doc dcterms:date ?date }
      OPTIONAL {
        ?role pro:relatesToDocument ?doc ; pro:isHeldBy ?person .
        OPTIONAL { ?person foaf:name ?authorName }
      }
    } LIMIT 400
    """
    
    results = []
    for r in rdf_graph.query(sparql):
        title, author, date = str(r.title), str(r.authorName), str(r.date)
        # Scoring por keywords
        score = sum(2 for t in tokens if t in title.lower())
        if years and any(y in date for y in years):
            score += 2
        if score > 0:
            results.append({
                "doc": str(r.doc), "title": title,
                "author": author or "N/A", "date": date or "N/A",
                "score": score,
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"biblio_results": results[:8]}
```

### 3.2. Actualizar State

```python
class EvalState(TypedDict, total=False):
    # ... existing fields ...
    biblio_results: List[Dict]  # NUEVO
```

### 3.3. Actualizar grafo

```python
builder.add_node("biblio", node_biblio_query)
builder.add_edge("biblio", "merge")
```

---

## Fase 4: Añadir query geoespacial

### 4.1. Reutilizar código existente

El código de `geo_query_nl()` ya existe en `RAG_graph/notebooks/graph_app.py`.

### 4.2. Nuevo nodo

```python
def node_geo_query(state: EvalState) -> dict:
    """Query GeoJSON for geospatial questions."""
    # Reuse existing geo_query_nl function
    result = geo_query_nl(state["question"])
    return {"geo_results": result}
```

---

## Fase 5: Merge de resultados múltiples

### 5.1. Nuevo nodo

```python
def node_merge_results(state: EvalState) -> dict:
    """Merge results from multiple sources."""
    docs = state.get("retrieved_docs", [])
    biblio = state.get("biblio_results", [])
    geo = state.get("geo_results", {})
    
    # Convertir biblio results a Documents
    biblio_docs = [
        Document(
            page_content=f"Título: {r['title']}\nAutor: {r['author']}\nFecha: {r['date']}",
            metadata={"source": "biblio_graph", "doc": r["doc"]},
        )
        for r in biblio
    ]
    
    # Convertir geo results a Documents
    geo_docs = []
    if geo and "top" in geo:
        top = geo["top"]
        geo_docs = [
            Document(
                page_content=f"Yacimiento: {top['nombre']}\nDistancia: {top['dist_km']:.2f} km",
                metadata={"source": "geojson", "yacimiento": top["nombre"]},
            )
        ]
    
    # Combinar todos los docs
    all_docs = docs + biblio_docs + geo_docs
    
    return {"merged_docs": all_docs, "context": "\n\n".join([d.page_content for d in all_docs[:10]])}
```

---

## Grafo final completo

```
Input (question + params)
    ↓
classify_query ─────────────────────────────────┐
    ↓                                           ↓
topic_match (LDA)          biblio_query (RDF)   │
    ↓                           ↓               ↓
retrieve_filtered         results          geo_query
    ↓                           ↓               ↓
rerank (opcional)         merge_results ←───────┘
    ↓                           ↓
generate ←──────────────────────┘
    ↓
evaluate (RAGAS)
    ↓
END
```

---

## Estimación de tiempo

| Fase | Descripción | Tiempo |
|---|---|---|
| **Fase 0 (COMPLETADA)** | Grafo básico: retrieve → rerank → generate → evaluate | 2.5 horas |
| **Fase 1** | Añadir LDA topics a metadatos + reindexar | 2-3 horas |
| **Fase 2** | Añadir nodo `topic_match` al grafo | 2 horas |
| **Fase 3** | Añadir grafo bibliográfico RDF + nodo `biblio_query` | 3-4 horas |
| **Fase 4** | Añadir query geoespacial + nodo `geo_query` | 2 horas |
| **Fase 5** | Merge de resultados múltiples + routing | 2 horas |

---

## Dependencias adicionales necesarias

| Paquete | Uso | Instalación |
|---|---|---|
| `gensim` | Cargar modelo LDA | `pip install gensim` |
| `rdflib` | Cargar grafo RDF | `pip install rdflib` |

Ambos ya están disponibles en `env_rag`.
