# Consultas al grafo bibliográfico — Tópicos

Consultas SPARQL de **solo lectura** sobre el grafo RDF de IDEArQ, centradas en los **tópicos/temas**.

> **Nota importante**: los tópicos **LDA (K=11)** NO están cargados en el grafo RDF. Viven
> únicamente en `lda/outputs/topics_for_graph_final.json` (pensado para Neo4j, nunca cargado).
>
> El "tema" que sí está en el grafo es `dc:subject`. **Hasta el 2026-10-16 salía de `sjr_materia`**
> (el área SJR de la revista); **desde el 2026-10-16 sale de `topics_openalex`** (OpenAlex),
> y es multivaluado: hasta 3 temas por artículo. Ver "De dónde sale el tema".

## Entorno

- Grafo: `data/biblio-graph/idearq-graph-instances-dedup-v2.nt` (17.211 triples)
- Parser: Python de `env_rag` con `rdflib 7.6.0`
- Ejecutar: `source env_rag/bin/activate && python <script>`

```python
from rdflib import Graph
g = Graph()
g.parse("data/biblio-graph/idearq-graph-instances-dedup-v2.nt", format="nt")
```

---

## Qué es `dc:subject` y qué es `skos:Concept` (y qué es un "tópico" aquí)

Hay **dos niveles** distintos para el mismo conjunto de temas:

| Nivel | En el grafo | Qué representa | Analogía |
|---|---|---|---|
| **Catálogo** (el tema existe y tiene nombre) | `?tema rdf:type skos:Concept` + `?tema skos:prefLabel "Archaeological and Geological Studies"` | Define los temas disponibles y su etiqueta legible | La lista de categorías |
| **Uso** (este documento es de este tema) | `?doc dc:subject ?tema` | Une cada documento con sus temas (1 a 3) | Las etiquetas pegadas a cada artículo |

No son alternativas: **son el mismo dato desde dos ángulos**. Q1 muestra el catálogo (con
nombres); Q2 cuenta cuántos documentos apuntan a cada tema. Verificado: el conjunto de
`skos:Concept` y el de subjects usados por documentos son **exactamente los mismos 29 nodos**
(ningún concepto queda sin usar).

Se pueden unir en **una sola consulta** (equivale a Q1 + Q2 juntas):

```sparql
PREFIX dc:   <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?label (COUNT(DISTINCT ?doc) AS ?n) WHERE {
  ?doc dc:subject ?s .
  OPTIONAL { ?s skos:prefLabel ?label }
} GROUP BY ?label ORDER BY DESC(?n)
```
→ mismas cuentas que Q2 (275, 176, 130, 108, 54, …), total 306 documentos.

### De dónde sale el "tema" (y por qué ya no es "dónde se publicó")

Desde el 2026-10-16 el tema **sí se extrae del artículo**, vía OpenAlex:

```yaml
# docs:          (sin dc:subject)
# document_topics: doc_uri       -> dc:subject        <- $(subject_iri)
# doc_topics.csv:  doc_uri, subject_iri               (1 fila por documento × tema)
# subjects:        s: .../subject/$(topic_slug)
#                  skos:prefLabel <- $(topic_label)
```

Es decir:
- `topics_openalex` = hasta 3 temas OpenAlex por artículo (separados por `;`), de
  `referencias_clean.csv` → `_derived/doc_topics.csv` (918 pares) y `_derived/subjects.csv`
  (29 temas).
- `dc:subject` del artículo = **los temas OpenAlex de ese artículo**.

Antes el tema era el **área SJR de la revista** (`sjr_materia`), es decir *dónde se publicó*,
no *de qué trata*. La columna `sjr_materia` sigue en el CSV, pero **ya no se mapea al grafo**:
`dc:subject` no tiene ninguna relación con la revista.

Cobertura: solo **310 de 1.174 filas** del CSV tienen `topics_openalex` (26,4 %), y el
enriquecimiento OpenAlex se hizo únicamente sobre artículos **desde el año 2000**. De esos
310 documentos hay **306 nodos distintos** en el grafo: 4 filas comparten `id` con otra y el
RDF las fusiona en el mismo documento (ver "Duplicados conocidos").

---

## Q1 — Conceptos de tema (skos:Concept)

```sparql
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?c ?label WHERE {
  ?c rdf:type skos:Concept .
  OPTIONAL { ?c skos:prefLabel ?label }
} ORDER BY ?label
```

**Resultado: 29 temas distintos, todos con etiqueta (`skos:prefLabel`):**

| slug | etiqueta |
|---|---|
| ancient_and_medieval_archaeology_studies | Ancient and Medieval Archaeology Studies |
| ancient_mediterranean_archaeology_and_history | Ancient Mediterranean Archaeology and History |
| animal_ecology_and_behavior_studies | Animal Ecology and Behavior Studies |
| archaeological_and_geological_studies | Archaeological and Geological Studies |
| archaeological_and_historical_studies | Archaeological and Historical Studies |
| archaeological_research_and_protection | Archaeological Research and Protection |
| archaeology_and_ancient_environmental_studies | Archaeology and ancient environmental studies |
| archaeology_and_cultural_heritage | Archaeology and Cultural Heritage |
| archaeology_and_rock_art_studies | Archaeology and Rock Art Studies |
| cultural_heritage_management_and_preservation | Cultural Heritage Management and Preservation |
| cultural_heritage_materials_analysis | Cultural Heritage Materials Analysis |
| evolution_and_paleontology_studies | Evolution and Paleontology Studies |
| fire_effects_on_ecosystems | Fire effects on ecosystems |
| forensic_and_genetic_research | Forensic and Genetic Research |
| forensic_anthropology_and_bioarchaeology_studies | Forensic Anthropology and Bioarchaeology Studies |
| genetic_and_phenotypic_traits_in_livestock | Genetic and phenotypic traits in livestock |
| genetic_diversity_and_population_structure | Genetic diversity and population structure |
| geology_and_paleoclimatology_research | Geology and Paleoclimatology Research |
| historical_and_architectural_studies | Historical and Architectural Studies |
| human_animal_interaction_studies | Human-Animal Interaction Studies |
| isotope_analysis_in_ecology | Isotope Analysis in Ecology |
| marine_and_environmental_studies | Marine and environmental studies |
| maritime_and_coastal_archaeology | Maritime and Coastal Archaeology |
| medieval_architecture_and_archaeology | Medieval Architecture and Archaeology |
| metallurgy_and_cultural_artifacts | Metallurgy and Cultural Artifacts |
| mineralogy_and_gemology_studies | Mineralogy and Gemology Studies |
| paleopathology_and_ancient_diseases | Paleopathology and ancient diseases |
| pleistocene_era_hominins_and_archaeology | Pleistocene-Era Hominins and Archaeology |
| soil_and_environmental_studies | Soil and Environmental Studies |

---

## Q2 — Número de documentos por tema (dc:subject)

```sparql
PREFIX dc: <http://purl.org/dc/terms/>
SELECT ?s (COUNT(?doc) AS ?n) WHERE { ?doc dc:subject ?s }
GROUP BY ?s ORDER BY DESC(?n)
```

**Resultado** (cobertura: **306 de 1.063 artículos = 28,8 %**; 918 triples `dc:subject`):

| Artículos | Tema |
|---:|---|
| 275 | Archaeological and Geological Studies |
| 176 | Archaeology and ancient environmental studies |
| 130 | Archaeological and Historical Studies |
| 108 | Ancient and Medieval Archaeology Studies |
| 54 | Pleistocene-Era Hominins and Archaeology |
| 39 | Forensic Anthropology and Bioarchaeology Studies |
| 31 | Soil and Environmental Studies |
| 11 | Cultural Heritage Materials Analysis |
| 11 | Metallurgy and Cultural Artifacts |
| 10 | Ancient Mediterranean Archaeology and History |
| 10 | Archaeology and Cultural Heritage |
| 10 | Geology and Paleoclimatology Research |
| 9 | Forensic and Genetic Research |
| 7 | Maritime and Coastal Archaeology |
| 6 | Archaeology and Rock Art Studies |
| 5 | Genetic diversity and population structure |
| 5 | Isotope Analysis in Ecology |
| 3 | Historical and Architectural Studies |
| 3 | Marine and environmental studies |
| 2 | Animal Ecology and Behavior Studies |
| 2 | Human-Animal Interaction Studies |
| 2 | Medieval Architecture and Archaeology |
| 2 | Mineralogy and Gemology Studies |
| 2 | Paleopathology and ancient diseases |
| 1 | Archaeological Research and Protection |
| 1 | Cultural Heritage Management and Preservation |
| 1 | Evolution and Paleontology Studies |
| 1 | Fire effects on ecosystems |
| 1 | Genetic and phenotypic traits in livestock |

Los 29 temas suman **918** pares documento×tema sobre **306 documentos** (todos los documentos
con tema tienen exactamente 3 temas). Los 918 triples recaen en 306 `bibo:AcademicArticle`.

---

## Q3 — Co-ocurrencia de temas (el grafo ahora es multivaluado)

Con el modelo anterior cada artículo tenía **un** tema; ahora tiene **tres**. La pregunta
útil ya no es "qué artículos tienen un tema raro" (la taxonomía de OpenAlex es de
subcampos y casi ningún slug contiene literalmente `archeolog`), sino **qué temas aparecen
juntos**.

```sparql
PREFIX dc:   <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?l1 ?l2 (COUNT(DISTINCT ?doc) AS ?n) WHERE {
  ?doc dc:subject ?s1 , ?s2 .
  ?s1 skos:prefLabel ?l1 .
  ?s2 skos:prefLabel ?l2 .
  FILTER(STR(?s1) < STR(?s2))
} GROUP BY ?l1 ?l2 ORDER BY DESC(?n) LIMIT 10
```

| Documentos | Par de temas |
|---:|---|
| 456 | Archaeological and Geological Studies + Archaeology and ancient environmental studies |
| 375 | Archaeological and Geological Studies + Archaeological and Historical Studies |
| 312 | Ancient and Medieval Archaeology Studies + Archaeological and Geological Studies |
| 198 | Ancient and Medieval Archaeology Studies + Archaeological and Historical Studies |
| 150 | Archaeological and Geological Studies + Pleistocene-Era Hominins and Archaeology |
| 144 | Archaeology and ancient environmental studies + Pleistocene-Era Hominins and Archaeology |
| 105 | Archaeological and Geological Studies + Forensic Anthropology and Bioarchaeology Studies |
| 102 | Archaeological and Historical Studies + Archaeology and ancient environmental studies |
| 84 | Archaeology and ancient environmental studies + Soil and Environmental Studies |
| 81 | Ancient and Medieval Archaeology Studies + Archaeology and ancient environmental studies |

> **Cambio de premisa**: la antigua Q3 ("artículos con tema ajeno a la arqueología") buscaba
> etiquetas SJR tipo `2701|Medicine`. Con OpenAlex el filtro `!CONTAINS(LCASE(?subject),'archeolog')`
> devuelve **los 306 documentos** — casi todos los pares de temas incluyen un subcampo que no
> lleva la palabra "archeolog" en el slug (`soil_…`, `geology_…`, `isotope_…`). La consulta se
> sustituye por la de co-ocurrencia, que sí distingue.

---

## Q4 — Tópicos de cada artículo (listado completo)

```sparql
PREFIX dc:   <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?doc ?title ?label WHERE {
  ?doc dc:title ?title ;
       dc:subject ?topic .
  ?topic skos:prefLabel ?label .
} ORDER BY ?title ?label
```

**Resultado: 930 filas sobre 306 documentos.** Ojo con el número: son 930 y no los 918
triples `dc:subject`, porque aquí el `?title` **también** se une y **9 nodos del grafo llevan
dos `dc:title`** (pares del CSV que comparten `id` y son la misma obra, ver "Duplicados
conocidos"). Cuatro de esos nodos tienen temas, y su segundo título multiplica sus 3 temas:
918 + 12 = 930.

Para una fila por documento (los temas juntos en una columna):

```sparql
PREFIX dc:   <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?doc (GROUP_CONCAT(DISTINCT ?label; separator=" | ") AS ?topicos) WHERE {
  ?doc dc:subject ?t .
  ?t skos:prefLabel ?label .
} GROUP BY ?doc ORDER BY ?doc LIMIT 10
```

→ 306 filas, una por documento. Ejemplo de celda:
`Archaeological and Geological Studies | Archaeology and ancient environmental studies | Soil and Environmental Studies`

Muestra del listado:

| Artículo | Tema |
|---|---|
| 8.2 Ka BP Paleoclimatic Event and the Ebro Valley Mesolithic Groups… | Archaeological and Geological Studies |
| 8.2 Ka BP Paleoclimatic Event… (el mismo) | Archaeology and ancient environmental studies |
| 8.2 Ka BP Paleoclimatic Event… (el mismo) | Pleistocene-Era Hominins and Archaeology |
| A Common Genetic Origin for Early Farmers from Mediterranean Cardial… | Archaeology and ancient environmental studies |
| A Common Genetic Origin… (el mismo) | Forensic and Genetic Research |
| A Common Genetic Origin… (el mismo) | Genetic diversity and population structure |

---

## Q5 — Tópicos de un artículo en concreto

Dos formas equivalentes, según lo que tengas a mano: el nodo o el texto del título.

**Por URI** (si ya conoces el nodo):

```sparql
PREFIX dc:   <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?label WHERE {
  <http://idearq.org/resource/article/180838> dc:subject ?topic .
  ?topic skos:prefLabel ?label .
} ORDER BY ?label
```

**Por título** (si solo tienes el texto); devuelve además el nodo:

```sparql
PREFIX dc:   <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?doc ?title ?label WHERE {
  ?doc dc:title ?title ; dc:subject ?topic .
  ?topic skos:prefLabel ?label .
  FILTER(CONTAINS(LCASE(?title), "beaker phenomenon"))
} ORDER BY ?label
```

**Resultado por URI** (`article/180838` = *The Beaker Phenomenon and the Genomic
Transformation of Northwest Europe*) — 3 filas, una por tema:

| Tema |
|---|
| Ancient and Medieval Archaeology Studies |
| Archaeological and Geological Studies |
| Forensic and Genetic Research |

**Resultado por título** — 6 filas, porque `"beaker phenomenon"` aparece en **dos**
artículos distintos, con 3 temas cada uno:

| Artículo | Tema |
|---|---|
| Absolute Chronology of the Beaker Phenomenon North of the Tagus Estuary | Archaeological and Geological Studies |
| Absolute Chronology of the Beaker Phenomenon… (el mismo) | Archaeological and Historical Studies |
| Absolute Chronology of the Beaker Phenomenon… (el mismo) | Archaeology and ancient environmental studies |
| The Beaker Phenomenon and the Genomic Transformation of Northwest Europe | Ancient and Medieval Archaeology Studies |
| The Beaker Phenomenon and the Genomic Transformation… (el mismo) | Archaeological and Geological Studies |
| The Beaker Phenomenon and the Genomic Transformation… (el mismo) | Forensic and Genetic Research |

> El `LCASE` del `FILTER` es necesario porque los títulos mezclan mayúsculas. Y si el título
> que buscas es corto o genérico, prepárate para más de un artículo: mejor localizar primero el
> nodo con `?doc dc:title ?title` y luego consultar por URI.

---

## Resumen de predicados del grafo

| Triples | Predicado |
|---:|---|
| 4.207 | dc:creator |
| 3.686 | rdf:type |
| 2.409 | foaf:name |
| 1.256 | dc:title |
| 1.159 | dc:identifier |
| 1.158 | dc:issued |
| 918 | dc:subject |
| 553 | bibo:issn |
| 547 | bibo:abstract |
| 494 | bibo:doi |
| 464 | dc:isPartOf |
| 331 | bibo:uri |
| 29 | skos:prefLabel |

---

## Duplicados conocidos (afectan a los temas)

`referencias_clean.csv` tiene 1.174 filas con **16 `id` repetidos** (32 filas) y **8 títulos
repetidos** (17 filas, 1 vacío). El URI del documento se construye con `(tipo_fuente, id)`, así
que dos filas con el mismo `id` **son el mismo nodo RDF** salvo que el mapeo las desambigüe.

De esos 16 pares, **15 son la misma obra ingerida dos veces** (título idéntico, o uno truncado
contenido token a token en el otro, o abstract idéntico) y **1 son dos obras distintas**
(`418930`: *University of Granada Radiocarbon Dates VII* vs *VIII*). Solo el segundo caso se
desambigua: sus filas reciben URIs propios (`.../article/418930` y `.../article/418930-2`), sin
tocar la columna `id`.

Cuatro de los pares que siguen compartiendo nodo tienen temas, y eso explica que 310 filas con
`topics_openalex` den 306 documentos con `dc:subject`:

| Filas del CSV | `id` | Evidencia de que es la misma obra | Efecto |
|---|---|---|---|
| 518 + 519 | 958612 | mismo DOI, mismo año, abstract IGUAL; un título es el otro truncado | 1 nodo |
| 812 + 813 | 958664 | filas idénticas | 1 nodo |
| 841 + 872 | 958673 | mismo DOI, abstract IGUAL; solo cambian las comillas del título | 1 nodo |
| 1138 + 1147 | 958649 | abstract IGUAL; el título largo lleva el corto detrás | 1 nodo |

No es un fallo del mapeo: son la misma publicación presente dos veces en el CSV fuente (el
export del portal, `articulos_idearq.csv`, ya trae 17 `id` repetidos en 1.189 filas).

---

## ¿Se parecen estos temas a los tópicos LDA?

**No.** Siguen siendo cosas de naturaleza distinta, aunque ahora ambas se extraen del artículo:

| | Grafo — `topics_openalex` | LDA K=11 |
|---|---|---|
| Qué es | Subcampos **OpenAlex** asignados al artículo | Tema de **contenido** inferido del texto completo |
| Se extrae de | `topics_openalex` (metadata OpenAlex) | Texto completo (lematizado) |
| Granularidad | Media: 29 subcampos, 3 por artículo | Fina: isótopos, campaniforme, megalitismo, arte rupestre |
| Cobertura | **306/1.063 (28,8 %)** | **513/513 (100 % del corpus LDA)** |
| Nº de temas | 29 | 11 |
| Asignación | **3 temas por artículo**, sin orden ni pesos | **distribución probabilística** por documento |

La diferencia clave ya no es "dónde se publicó vs de qué trata" (ambos son del artículo), sino
que **OpenAlex etiqueta con subcampos de catálogo** y **LDA mide mezcla temática del texto**.
Siguen siendo complementarios: OpenAlex da cobertura de catálogo (y solo desde 2000), LDA da
granularidad de contenido (y cubre los 513 documentos del corpus con texto).

---

*Ejecutado el 2026-10-16 sobre el grafo regenerado (17.211 triples). Solo lectura; el grafo no
se modificó. Memoria: `idearq-rag/graph/consultas-topicos`.*
