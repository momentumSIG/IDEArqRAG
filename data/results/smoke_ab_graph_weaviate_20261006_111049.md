# A/B smoke: grafo RDF vs Weaviate

Fecha: `2026-10-06T11:10:49`

Las **mismas 5 preguntas** del smoke del grafo (`scripts/smoke_graph.py`),
resueltas por los dos caminos. Las respuestas viven **solo en el grafo**:
ni en los PDF ni en el geojson, así que este experimento mide qué parte
del conocimiento del grafo puede reproducir el índice vectorial.

## Configuración

| | |
|---|---|
| grafo | `/mnt/netapp1/Store_CSIC/home/csic/usi/eag/idearq-rag/data/biblio-graph/idearq-graph-instances-dedup-v2.nt` (16357 triples) |
| colección | `IdearqMiniLM_800_50_v2` |
| embedding | `all-MiniLM-L6-v2` |
| estrategia | hybrid alpha=0.5, k=10 |
| LLM | `Qwen3-4B-Instruct-2507` (T=0.5) |
| Weaviate | `http://c201-20:8080` |
| Ollama | `http://c201-20:11434` |
| autochequeo de la sonda | SOSPECHOSO — window='Reunión Científica Arqueología Albacete Científica Arqueología' -> ['AND: 50 returned, 0 contain it', 'OR: 50 returned, 0 contain it'] |
| tiempo | 902.38 s |

## Resultado

| | respuestas |
|---|---|
| grafo (SPARQL) | **5/5** |
| literal presente en el índice (BM25) | 2/5 |
| literal llega al contexto (híbrido) | 0/5 |
| literal aparece en la respuesta del LLM | 0/5 |

| id | grafo | en índice | en contexto | en respuesta | docs |
|---|---|---|---|---|---|
| Q1 | ✅ PASS | ❌ | ❌ | ❌ | 10 |
| Q2 | ✅ PASS | ✅ | ❌ | ❌ | 10 |
| Q3 | ✅ PASS | ❌ | ❌ | ❌ | 10 |
| Q4 | ✅ PASS | ❌ | ❌ | ❌ | 10 |
| Q5 | ✅ PASS | ✅ | ❌ | ❌ | 10 |

## Detalle por pregunta

### Q1 — authorship

**Pregunta:** Who are the authors recorded by the graph for 'University of Pennsylvania radiocarbon dates XXI'?

**Respuesta del grafo:** `['Anne Meulengracht', 'Barbara Lawn', 'Patrick McGovern']`

**Respuesta RAG:** No consta en el contexto.

| literal | en índice | en contexto | en respuesta | en el corpus |
|---|---|---|---|---|
| `Anne Meulengracht` | ❌ | ❌ | ❌ | 0 |
| `Barbara Lawn` | ❌ | ❌ | ❌ | 0 |
| `Patrick McGovern` | ❌ | ❌ | ❌ | 0 |

**Documentos recuperados:** `['1002_gibaja2016.pdf', '1117_nueva serie de dataciones_2020.pdf', '799_2012.pdf', '984_DemografiayPoblamiento_2016.pdf', '1035_Chronological reassessment of the middle to upper paleolithic_2018.pdf', '1059_Archaeometry - Higham - Radiocarbon Dates from the Oxford AMS System  Archaeometry Datelist 36_2018.pdf', '750_2011.pdf', '947_Baylissetal2016Montelirio.pdf', '1058_Nuevos datos cronologicos cueva sepulcral neolotica_2018.pdf', '703_2009.pdf']`

### Q2 — publication-year

**Pregunta:** In which year was the book 'Secuencias de cambio social en una región mediterránea...' published?

**Respuesta del grafo:** `['2003']`

**Respuesta RAG:** No consta en el contexto.

| literal | en índice | en contexto | en respuesta | en el corpus |
|---|---|---|---|---|
| `2003` | ✅ | ❌ | ❌ | 11064 |
| `Secuencias de cambio social en una región mediterránea` | ❌ | ❌ | ❌ | 0 |

**Documentos recuperados:** `['793_2012.pdf', '1120_Prospecciones arqueologicas_2020.pdf', '586_2005.pdf', '654_2008.pdf', '675_2008.pdf', '695_2009.pdf', '852_2014.pdf', '1144_Secuencias cronologicas_2021.pdf', '684_2009.pdf', '1174_Aproximaciones bayesianas_2023.pdf']`

### Q3 — journal-ISSN

**Pregunta:** What is the ISSN of the journal that published 'El Sistema Defensivo del Castro de La Ulaña'?

**Respuesta del grafo:** `['0066-6742']`

**Respuesta RAG:** No consta en el contexto.

| literal | en índice | en contexto | en respuesta | en el corpus |
|---|---|---|---|---|
| `0066-6742` | ❌ | ❌ | ❌ | 0 |
| `00666742` | ❌ | ❌ | ❌ | 0 |

**Documentos recuperados:** `['c14_v2.geojson', 'c14_v2.geojson', '651_2008.pdf', 'c14_v2.geojson', 'c14_v2.geojson', '935_Acequion_2016.pdf', '898_Yacimiento de Santa Comba _2015.pdf', '935_Acequion_2016.pdf', '817_2013.pdf', '935_Acequion_2016.pdf']`

### Q4 — DOI

**Pregunta:** What is the DOI of 'Elfenbeinwerkstätten in Huelva und La Rebanadilla (Málaga)...'?

**Respuesta del grafo:** `['10.34780/mm.v57i0.1003']`

**Respuesta RAG:** No consta en el contexto.

| literal | en índice | en contexto | en respuesta | en el corpus |
|---|---|---|---|---|
| `10.34780/mm.v57i0.1003` | ❌ | ❌ | ❌ | 0 |

**Documentos recuperados:** `['c14_v2.geojson', 'c14_v2.geojson', '1108_Cronologia fundacional Malaka fenicia_2020.pdf', '945_ColonizacionFenicia_LopezCastro_2016.pdf', '604_2005.pdf', '819_2013.pdf', 'c14_v2.geojson', 'c14_v2.geojson', 'c14_v2.geojson', 'c14_v2.geojson']`

### Q5 — instance-type

**Pregunta:** Which BIBO type does the graph assign to 'A Neolithic Case of Cranial Trepanation (Eira Pedrinha, Portugal)'?

**Respuesta del grafo:** `['http://purl.org/ontology/bibo/Book']`

**Respuesta RAG:** No consta en el contexto.

| literal | en índice | en contexto | en respuesta | en el corpus |
|---|---|---|---|---|
| `Book` | ✅ | ❌ | ❌ | 701 |
| `10.1201/b16983-12` | ❌ | ❌ | ❌ | 0 |

**Documentos recuperados:** `['1058_Nuevos datos cronologicos cueva sepulcral neolotica_2018.pdf', 'c14_v2.geojson', 'c14_v2.geojson', '1011_Neolothic ditched_2017.pdf', '1168_The mesolithic of atlantic coastal spain_2022.pdf', '1063_Tracing the chronology_2018.pdf', '680_2009.pdf', '678_2009_1.pdf', '744_2010_Llorente2010-TheharesfromCovaFosca-.pdf', '1168_The mesolithic of atlantic coastal spain_2022.pdf']`
