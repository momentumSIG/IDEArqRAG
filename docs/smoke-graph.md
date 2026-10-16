# Smoke test de la capa de grafo RDF

**Propósito:** demostrar en una reunión que el sistema RAG tiene una segunda fuente de
respuesta —el grafo RDF bibliográfico— capaz de contestar preguntas que **no están en
los 525 PDF del corpus ni en el geojson de dataciones C14**.

**Artefactos:** `scripts/smoke_graph.py` (script determinista) · este documento.

---

## 1. Qué comprueba

El script carga `data/biblio-graph/idearq-graph-instances-dedup-v2.nt` (17.211 triples)
con `rdflib`, lanza **5 consultas SPARQL** cuya respuesta es una propiedad literal del
grafo y la compara con la respuesta esperada documentada. No usa LLM, Weaviate, Slurm
ni GPU: tarda ~3 s.

```bash
python3 scripts/smoke_graph.py                                        # → exit 0
python3 scripts/smoke_graph.py --corpus /ruta/a/textos_extraidos      # + evidencia
python3 -m py_compile scripts/smoke_graph.py                          # chequeo sintáctico
```

### Cómo leerlo

| Salida | Significado |
|---|---|
| `[PASS] Qn (tema)` | la consulta devolvió exactamente la respuesta esperada |
| `[FAIL] Qn (tema)` | la consulta devolvió otra cosa o nada (`expected` vs `got` se imprimen) |
| `RESULT: PASS (5/5)` + `exit 0` | toda la capa de grafo responde |
| `RESULT: FAIL (n/5)` + `exit 1` | al menos una consulta no cuadra: el grafo cambió o la expectativa está desactualizada |

Con `--corpus <dir>` se añade una línea `evidence:` por respuesta con el número de
ocurrencias del literal en el texto extraído de los PDF (`0 → NOT in corpus`). Es
**informativa**: nunca cambia el código de salida, porque el corpus no siempre está
disponible en la máquina donde se ejecuta el smoke.

**Criterio de solo-grafo:** un literal es *solo-grafo* si no aparece en el corpus
documental (texto de los 525 PDF) ni en los 4 geojson. La expectativa se marca
`SKIPPED` si esa comprobación no puede hacerse.

---

## 2. Mapa del grafo (17.211 triples)

### Predicados por frecuencia

| Triples | Predicado |
|---:|---|
| 4.207 | `dcterms:creator` (obra → autor) |
| 3.686 | `rdf:type` |
| 2.409 | `foaf:name` (nombre visible de autor) |
| 1.256 | `dcterms:title` |
| 1.159 | `dcterms:identifier` |
| 1.158 | `dcterms:issued` (`xsd:gYear`) |
| 918 | `dcterms:subject` (obra → tema, 3 por artículo) |
| 553 | `bibo:issn` |
| 547 | `bibo:abstract` |
| 494 | `bibo:doi` |
| 464 | `dcterms:isPartOf` (artículo → revista) |
| 331 | `bibo:uri` |
| 29 | `skos:prefLabel` |

### Tipos de instancia

| Instancias | Tipo |
|---:|---|
| 2.409 | `foaf:Person` (autores, ya deduplicados) |
| 1.063 | `bibo:AcademicArticle` |
| 89 | `bibo:Journal` |
| 57 | `bibo:Book` |
| 34 | `bibo:Document` |
| 29 | `skos:Concept` |
| 4 | `bibo:Series` |
| 1 | `bibo:Proceedings` |

### Estructura y límites (importante para la reunión)

- **Esquema:** `artículo/obra —dcterms:creator→ autor —foaf:name→ nombre` y
  `artículo —dcterms:isPartOf→ revista —bibo:issn→ ISSN`. 464 obras están enlazadas a
  su revista.
- **NO existe ningún predicado de citación** (`bibo:cites`, `dcterms:references`,
  etc.). Con esta versión **no se puede responder "quién cita a quién"**; sí se pueden
  responder autoría, año, revista/ISSN, DOI, tipo y agregaciones.
- **Procedencia:** el grafo se materializa con Morph-KGC desde
  `data/biblio-graph/referencias_clean.csv` (1.174 registros bibliográficos del portal
  IDEArq), no a partir de los PDF. Por eso su cobertura (1.158 obras) es **más amplia
  que el corpus documental**: 181 obras del grafo no tienen ningún PDF en el corpus
  (verificado con 3 sondas sobre el título, sin coincidencias).

---

## 3. Las 5 preguntas bibliográficas

Todas se ejecutan tal cual en el script. La columna *solo-grafo* se apoya en dos greps:
(1) texto extraído de los 525 PDF con `pdftotext` (83.397.149 bytes, ningún PDF
devolvió texto vacío), (2) los 4 archivos de `data/ingesta/geojson/`.

| # | Pregunta | Respuesta esperada | Resultado | Solo-grafo | Evidencia |
|---|---|---|---|---|---|
| Q1 | ¿Quiénes son los autores de *«University of Pennsylvania radiocarbon dates XXI»*? | `Anne Meulengracht`, `Barbara Lawn`, `Patrick McGovern` | **PASS** | **Sí** | Los 3 nombres: **0** apariciones en el corpus y **0** en el geojson; título: **0** (3 sondas) |
| Q2 | ¿En qué año se publicó el libro *«Secuencias de cambio social en una región mediterránea. Análisis arqueológico de la depresión de Vera (Almería) entre los siglos V y XI»*? | `2003` | **PASS** | **Sí, con salvedad** | Título: **0** (3 sondas) ⇒ la obra no está en ningún PDF, así que ningún texto puede vincularla a su año. El token `2003` suelto sí aparece **5.532** veces (no es unívoco); la forma tipada `"2003"^^xsd:gYear`: **0** |
| Q3 | ¿Cuál es el ISSN de la revista donde se publicó *«El Sistema Defensivo del Castro de La Ulaña»*? | `0066-6742` | **PASS** | **Sí** | `0066-6742`: **0** en corpus y **0** en geojson; sin guion (`00666742`): **0**; título de la obra: **0**. (El nombre «Archivo Español de Arqueología» sí aparece 139 veces ignorando mayúsculas/acentos, pero nunca su ISSN: el enlace obra→ISSN solo existe en el grafo) |
| Q4 | ¿Cuál es el DOI de *«Elfenbeinwerkstätten in Huelva und La Rebanadilla (Málaga), den Ältesten Phönizischen Niederlassungen auf der Iberischen Halbinsel»*? | `10.34780/mm.v57i0.1003` | **PASS** | **Sí** | DOI: **0** en corpus y **0** en geojson; título: **0** (3 sondas) |

### Consultas SPARQL

```sparql
# Q1 — autoría
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?author WHERE {
  ?work dcterms:title "University of Pennsylvania radiocarbon dates XXI" ;
        dcterms:creator ?authorNode .
  ?authorNode foaf:name ?author .
} ORDER BY ?author

# Q2 — año de publicación
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?issued WHERE {
  ?work dcterms:title "Secuencias de cambio social en una región mediterránea. Análisis arqueológico de la depresión de Vera (Almería) entre los siglos V y XI" ;
        dcterms:issued ?issued .
}
# rdflib normaliza xsd:gYear a "2003-01-01"; el script toma los 4 primeros caracteres.

# Q3 — ISSN de la revista
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX bibo: <http://purl.org/ontology/bibo/>
SELECT ?issn WHERE {
  ?work dcterms:title "El Sistema Defensivo del Castro de La Ulaña" ;
        dcterms:isPartOf ?journal .
  ?journal bibo:issn ?issn .
}

# Q4 — DOI
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX bibo: <http://purl.org/ontology/bibo/>
SELECT ?doi WHERE {
  ?work dcterms:title "Elfenbeinwerkstätten in Huelva und La Rebanadilla (Málaga), den Ältesten Phönizischen Niederlassungen auf der Iberischen Halbinsel" ;
        bibo:doi ?doi .
}

```

---

## 4. Reproducir la evidencia

```bash
# 1. Extraer el texto de los PDF (solo lectura; ~45 s, se escribe FUERA del proyecto)
mkdir -p /tmp/idearq_smoke/textos
cd data/ingesta
ls -d *.pdf | tr '\n' '\0' | xargs -0 -P 8 -I{} sh -c \
  'pdftotext -q "{}" /tmp/idearq_smoke/textos/{}.txt'

# 2. Smoke con evidencia
python3 ../../scripts/smoke_graph.py --corpus /tmp/idearq_smoke/textos

# 3. Comprobación directa de un literal en el corpus y en el geojson
grep -F -o '0066-6742' /tmp/idearq_smoke/textos/*.txt | wc -l        # 0
cat data/ingesta/geojson/*.geojson | grep -F -o '0066-6742' | wc -l  # 0
```

---

## 5. Salida real (última ejecución)

```
==============================================================================
IDEArq RAG - graph layer smoke test
==============================================================================
graph   : .../data/biblio-graph/idearq-graph-instances-dedup-v2.nt
triples : 16357 (loaded in 1.59s)
checks  : 5 SPARQL questions
------------------------------------------------------------------------------
[PASS] Q1 (authorship)      expected = got = ['Anne Meulengracht', 'Barbara Lawn', 'Patrick McGovern']
[PASS] Q2 (publication-year) expected = got = ['2003']
[PASS] Q3 (journal-ISSN)    expected = got = ['0066-6742']
[PASS] Q4 (DOI)             expected = got = ['10.34780/mm.v57i0.1003']
[PASS] Q5 (instance-type)   expected = got = ['http://purl.org/ontology/bibo/Book']

RESULT: PASS (5/5 checks passed)
elapsed: 3.15s
```

Código de salida: `0`. `python3 -m py_compile scripts/smoke_graph.py` sin errores.
`rdflib` 5.0.0 está disponible en el `python3` del sistema, por lo que no hizo falta
recurrir a un parser manual de `.nt`.

---

## 6. Honestidad y límites

1. **Q2 es la más débil de las cinco en cuanto a evidencia literal**: el año `2003` es
   un token frecuente en un corpus con miles de dataciones. La prueba sólida es que la
   obra **no está en el corpus** (3 sondas sobre el título, 0 coincidencias, y tampoco
   aparece en los nombres de archivo). Si se quiere una expectativa 100 % literal, Q2
   debería sustituirse por Q3/Q4 (ISSN/DOI), que sí tienen literal no ambiguo.
2. **No se modificó nada**: ni notebooks, ni PDF, ni `.nt`, ni CSV. Solo se crearon
   `scripts/smoke_graph.py` y `docs/smoke-graph.md`.
3. **El grafo no modela citaciones** (apartado 2): cualquier pregunta de tipo
   "quién cita a quién" queda fuera de alcance hasta que se amplíe el mapeo YARRRML.
4. **La capa de grafo no sustituye al corpus**: añade metadatos bibliográficos
   (autoría, año, revista, ISSN, DOI, tipo) que RRF fusiona con los documentos
   recuperados por vector/BM25 en el notebook de evaluación
   (`GRAPH_CONTEXT=True`).

---

## 7. Tópicos: del área SJR de la revista a los temas OpenAlex

Preguntado por "keywords". Respuesta corta: **el grafo no tiene una propiedad
`keyword`**. Lo más parecido es `dcterms:subject`, y desde el 2026-10-16 sale de
`topics_openalex` (OpenAlex) y no de `sjr_materia`: ver 7.5.

### 7.1 Qué hay de tópicos en el esquema

| Pregunta | Dato |
|---|---|
| ¿Existe `keyword` o `topic`? | **No.** El grafo tiene 13 predicados (§2); el de temas es `dcterms:subject` |
| ¿Y en Dublin Core? | **`dcterms:subject` es la propiedad estándar** de DCMI para esto: *"A list of topics, subjects, or keywords that describe the content of the resource"*. Es justo la que ya usa el grafo |
| ¿Se usa `dc:subject` (la vocabulario antiguo)? | **0 triples.** Solo `dcterms:subject` |
| ¿Y las "Palabras clave:" de los artículos? | No son un campo: están metidas dentro del texto de `bibo:abstract`. De los **756 artículos sin tema, 238 tienen resumen y ninguno declara "Palabras clave"/"Keywords"** (única coincidencia de `clave`: la palabra "enclaves") |

### 7.2 Cobertura real

```
artículos (rdf:type AcademicArticle)  1063
  con dcterms:subject                 306  (28,8 %)
  sin dcterms:subject                 757
```

Los 918 triples `dcterms:subject` son **los 306 artículos**, sin excepción: con el mapeo
anterior había 2 no-artículos, pero los temas OpenAlex existen solo para artículos. Los
306 artículos apuntan a **29 tópicos distintos**, todos con `skos:prefLabel` legible, y
cada artículo tiene **3 temas**. La capa cubre **uno de cada 3,5 artículos**, y no cubre
más porque el enriquecimiento OpenAlex se hizo solo sobre artículos **desde el año 2000**.

### 7.3 Q6 — ¿cuántos artículos por tema?

```sparql
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?label (COUNT(?a) AS ?n) WHERE {
  ?a dcterms:subject ?s .
  ?s skos:prefLabel  ?label .
} GROUP BY ?label ORDER BY DESC(?n) ?label
```

Resultado (29 temas, 918 pares documento×tema sobre 306 artículos):

| n | Tema |
|---:|---|
| 275 | Archaeological and Geological Studies |
| 176 | Archaeology and ancient environmental studies |
| 130 | Archaeological and Historical Studies |
| 108 | Ancient and Medieval Archaeology Studies |
| 54 | Pleistocene-Era Hominins and Archaeology |
| 39 | Forensic Anthropology and Bioarchaeology Studies |
| 31 | Soil and Environmental Studies |
| 11 | Cultural Heritage Materials Analysis · Metallurgy and Cultural Artifacts |
| 10 | Ancient Mediterranean Archaeology and History · Archaeology and Cultural Heritage · Geology and Paleoclimatology Research |
| 9 | Forensic and Genetic Research |
| 7 | Maritime and Coastal Archaeology |
| 6 | Archaeology and Rock Art Studies |
| 5 | Genetic diversity and population structure · Isotope Analysis in Ecology |
| 3 | Historical and Architectural Studies · Marine and environmental studies |
| 2 | Animal Ecology and Behavior Studies · Human-Animal Interaction Studies · Medieval Architecture and Archaeology · Mineralogy and Gemology Studies · Paleopathology and ancient diseases |
| 1 | Archaeological Research and Protection · Cultural Heritage Management and Preservation · Evolution and Paleontology Studies · Fire effects on ecosystems · Genetic and phenotypic traits in livestock |

**Lectura:** el catálogo deja de estar dominado por "Archaeology" y se reparte por
subcampos OpenAlex. Cada artículo lleva **3 temas**, así que estas cuentas son de
documentos por tema y **no suman el total** (una suma ingenua daría 918).

### 7.4 Q7 — co-ocurrencia de temas (el grafo es multivaluado)

Con el modelo anterior cada artículo tenía **un** tema; ahora tiene **tres**, así que la
pregunta útil ya no es "qué artículos tienen un tema raro" (la taxonomía OpenAlex es de
subcampos y casi ningún slug contiene literalmente `archeolog`) sino **qué temas aparecen
juntos**:

```sparql
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX skos:    <http://www.w3.org/2004/02/skos/core#>
SELECT ?l1 ?l2 (COUNT(DISTINCT ?doc) AS ?n) WHERE {
  ?doc dcterms:subject ?s1 , ?s2 .
  ?s1  skos:prefLabel ?l1 .
  ?s2  skos:prefLabel ?l2 .
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

**Lectura:** los tres temas de un artículo casi nunca son del mismo bloque: el par más
frecuente une arqueología con geología, y aparecen mezclas con ecología, genética o
antropología forense. El corpus sigue siendo interdisciplinar, pero ahora se ve **desde
el artículo**, no desde la revista donde se publicó.

### 7.5 De dónde sale el tópico: la cadena CSV → grafo

La columna del tema existe en el CSV fuente y se llama `topics_openalex`. La cadena
completa es esta:

```
data/biblio-graph/referencias_clean.csv      1174 artículos
    columna topics_openalex .............  310 con valor (26,4 %), 864 nulos
        │  (hasta 3 temas por artículo, separados por ";")
        │  notebooks/graph/rag-graph-ontologia-mapeo.ipynb
        │  (parse_topics + slugify)
        ▼
_derived/doc_topics.csv     918 pares (documento, tema)
_derived/subjects.csv        29 conceptos únicos
        │
        │  notebooks/graph/mapping-idearq.yarrrml
        │  mapping document_topics: dc:subject → skos:Concept · skos:prefLabel
        ▼
grafo  idearq-graph-instances-dedup-v2.nt
        918 triples dcterms:subject = 306 AcademicArticle (3 temas cada uno)
```

**Relleno de las columnas del CSV fuente** (`referencias_clean.csv`, 1174 filas):

| Columna | Con valor |
|---|---|
| `tipo_fuente` | 1174 (100 %) |
| `year` | 1172 (99,8 %) |
| `autores` | 1165 (99,2 %) |
| `issn` | 577 (49,1 %) |
| `abstract` | 555 (47,3 %) |
| `doi` | 503 (42,8 %) |
| `url_publicacion` | 330 (28,1 %) |
| **`topics_openalex`** | **310 (26,4 %)** |
| `keywords_openalex` | 310 (26,4 %) |
| `primary_topic_openalex` | 310 (26,4 %) |
| `openalex_id` | 310 (26,4 %) |
| `sjr_materia` (ya sin mapear) | 91 (7,8 %) |
| `conc_ciencia` | 33 (2,8 %) |

Las cuatro columnas OpenAlex son ahora **las más llenas después de `url_publicacion`**;
`conc_ciencia` sigue siendo la más vacía.

**Cuatro detalles comprobados:**

1. **No se pierde nada en el mapeo.** Los 310 artículos con tema producen 918 pares
   documento×tema, y los 29 conceptos tienen `skos:prefLabel`.
2. **El tema es del artículo, no de la revista.** Las 51 revistas que tienen algún
   artículo con tema tienen **más de un tema distinto** entre sus propios artículos (la
   que más, 14). Si el tema viniera de la revista, habría 1 por revista.
3. **310 filas → 306 documentos.** Cuatro pares de filas comparten `id` y `tipo_fuente`
   (518/519, 812/813, 864/865, 1138/1147), así que el RDF las fusiona en un mismo nodo
   y sus temas se unen.
4. **La cobertura la limita el año.** El enriquecimiento
   (`data/biblio-graph/keywords_openalex_idearq.csv`) cubre solo artículos **desde 2000**
   y de ese tramo faltan 82 filas; `sjr_materia` (91 filas) ya no se mapea, así que el
   grafo tiene temas para 306 de 1174 filas.

### 7.6 Límites de estas dos consultas

1. **Cobertura del 8,3 %.** Describen 88 de 1062 artículos. Los otros 974 **no tienen
   tópico** y no se les puede preguntar nada por `dcterms:subject`.
2. **No hay keywords extraíbles con SPARQL.** Para consultar "Palabras clave" habría
   que extraerlas del texto de `bibo:abstract` con un paso de Python previo (y solo 7
   resúmenes las traen).
3. **No se modificó nada**: solo se leyó el `.nt` con `rdflib` y se añadió este
   apartado al documento.
