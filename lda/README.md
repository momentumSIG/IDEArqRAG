# LDA Topic Modeling - IDEArq Archaeological Corpus

## Resumen

Análisis de modelado de tópicos (Topic Modeling) sobre el corpus de **513 artículos arqueológicos** de IDEArq mediante **LDA (Latent Dirichlet Allocation)** con pipeline multilingüe (EN, ES, FR, PT, CA). Se comparó también con **BERTopic** como alternativa basada en embeddings.

**Resultado final:** LDA con **K=5** (coherence c_v = 0.762) como modelo óptimo, con K=10 disponible para mayor granularidad.

---

## 1. Datos de entrada

| Parámetro | Valor |
|---|---|
| **Fuente** | `/home/raglinux/RAG_graph/ingesta/` |
| **Formato original** | 525 PDFs académicos de arqueología |
| **PDFs procesados** | 513 (12 descartados por texto insuficiente) |
| **Idiomas detectados** | ES 282, EN 168, PT 30, CA 22, FR 4, unknown 7 |
| **Rango temporal** | ~1999–2024 |

---

## 2. Pipeline de procesamiento

### 2.1 Arquitectura del pipeline

```
525 PDFs
  │
  ├─► 01_parse.py          PyMuPDF4LLM → texto markdown
  │     └─► 513 docs exitosos, 12 fallidos
  │
  ├─► 02_extract_meta.py   fitz (PyMuPDF) → DOI, título, autor
  │     └─► 288 con autor, 505 con título, 270 con DOI
  │
  ├─► 03_detect_lang.py    langdetect (primeros 2000 chars)
  │     └─► ES 282, EN 168, PT 30, CA 22, FR 4, unknown 7
  │
  ├─► 04_clean.py          Regex → URLs, refs bibliográficas, tablas, números
  │     └─► ~34,895 chars promedio/doc después de limpieza
  │
  ├─► 05_lemmatize.py      spaCy (5 modelos: en/es/fr/pt/ca) + stopwords custom
  │     └─► 2,870 tokens promedio/doc, 1,472,134 tokens totales
  │
  ├─► 06_build_corpus.py   gensim Dictionary + Bag-of-Words
  │     └─► 9,985 términos (vocabulario), 687 tokens/doc promedio
  │
  ├─► 07_train_lda.py      Grid search K∈{5,10,15,20,25,30} + coherence c_v
  │     └─► Mejor K=5 (coherence=0.762)
  │
  ├─► 08_assign.py         Asignación de tópico dominante por documento
  │     └─► T0=145, T1=41, T2=104, T3=42, T4=181
  │
  ├─► 09_visualize.py      pyLDAvis + top-15 papers por tópico
  │     └─► HTML interactivo + CSVs
  │
  └─► 10_export.py         CSV final + JSON para grafo + alineación cross-language
        └─► papers_topics.csv, topics_for_graph.json, topics_alignment.csv
```

### 2.2 Decisiones técnicas

| Decisión | Elección | Justificación |
|---|---|---|
| **Parser** | PyMuPDF4LLM | Rápido, extrae texto de PDFs con OCR ya aplicado |
| **Lematizador** | spaCy (5 modelos) | Lematización por idioma: `en_core_web_sm`, `es_core_news_sm`, `fr_core_news_sm`, `pt_core_news_sm`, `ca_core_news_sm` |
| **Stopwords** | spaCy + custom por idioma | Se añadieron stopwords arqueológicas poco informativas (archaeology, estudio, research, etc.) |
| **Vectorizador** | Bag-of-Words (gensim) | Requerido por LDA |
| **Filtrado diccionario** | no_below=10, no_above=0.4, keep_n=50,000 | Elimina términos muy raros y muy comunes |
| **LDA** | gensim LdaModel | Estándar de facto, mejor CoherenceModel que scikit-learn |
| **Métrica** | Coherence c_v | Basada en co-ocurrencia de palabras en ventanas deslizantes |
| **RAM cap** | 16 GB | Monitor con psutil, abort si >14 GB |
| **Checkpoint** | Pickle cada 50 docs | Permite retomar parseo si se interrumpe |

---

## 3. Grid Search: selección de K

Se probaron 6 valores de K (número de tópicos) y se seleccionó el de mayor coherence:

| K | Coherence (c_v) | Perplexity | Decisión |
|---|---|---|---|
| **5** | **0.762** | 210.9 | ← **MEJOR** |
| 10 | 0.692 | 220.6 | Más granularidad |
| 15 | 0.674 | — | |
| 20 | 0.644 | — | |
| 25 | 0.642 | — | |
| 30 | 0.630 | — | |

**Interpretación:** El coherence score decrece al aumentar K. Esto es esperable: con menos tópicos, cada uno es más amplio y coherente internamente. K=5 maximiza la coherencia pero sacrifica granularidad. K=10 ofrece un balance aceptable (0.692) con más resolución temática.

---

## 4. Resultados LDA K=5

### 4.1 Distribución de documentos

| Tópico | Docs | % | Idioma dominante |
|---|---|---|---|
| **T0** | 145 | 28.3% | EN |
| **T1** | 41 | 8.0% | EN (contaminado) |
| **T2** | 104 | 20.3% | ES |
| **T3** | 42 | 8.2% | PT |
| **T4** | 181 | 35.3% | ES |

### 4.2 Interpretación temática

#### T0 — Bioarqueología y Dataciones (145 docs, EN)
**Keywords:** `date, sample, human, bone, neolithic, individual, area, remain, early, result, burial, level, cave, find, period, different, late, structure`

Artículos anglófonos centrados en dataciones radiocarbónicas, análisis de restos humanos y óseos, estudios de poblaciones neolíticas, genética antigua. Es el tópico más "científico/método".

#### T1 — Contaminado por stopwords (41 docs)
**Keywords:** `del, que, los, las, con, una, omit, por, como, este, para, más, esta, entre, sobre, campaniforme, son, han, do, sin`

⚠️ Tópico ruidoso: stopwords españolas no filtradas correctamente. La única señal temática real es "campaniforme". Requiere mejora en el diccionario de stopwords.

#### T2 — Industria lítica y ocupaciones en cueva (104 docs, ES)
**Keywords:** `cueva, cova, abrigo, sílex, lítico, mesolítico, cavidad, secuencia, can, especie, aparecer, cardial, industria, valle, hueso, arte, cerámico, ebro, milenio, individuo`

Artículos sobre tecnología lítica (sílex), ocupaciones mesolíticas, neolítico cardial, secuencias estratigráficas en abrigos y cuevas, especialmente del valle del Ebro.

#### T3 — Arqueología portuguesa (42 docs, PT)
**Keywords:** `picturar, ser, lisboa, anta, ocupação, região, estrutura, ter, portugal, data, com, milénio, sítio, bronze, soares, datação, cerâmica, antigo, apresentar, rui`

Artículos en portugués centrados en Portugal (Lisboa, región), monumentos megalíticos (antas), Bronce, cerámica y dataciones. Refleja la comunidad arqueológica portuguesa.

#### T4 — Bronce/Hierro y prácticas funerarias (181 docs, ES)
**Keywords:** `bronce, issn, funerario, aparecer, borde, asentamiento, enterramiento, poblado, hierro, superficie, piedra, siglo, construcción, hueso, ajuar, cerámico, torno, hoyo, tierra, pieza`

El tópico más grande. Abarca asentamientos, enterramientos, ajuares, cerámica, arquitectura, metalurgia (bronce/hierro) de la Prehistoria reciente peninsular. Es el "gran cajón" de la arqueología ibérica.

### 4.3 Resultados LDA K=10 (alternativa más granular)

| Topic | Docs | Temática | Keywords principales |
|---|---|---|---|
| **T2** | ~50 | Industria lítica/cuevas | cueva, sílex, lítico, mesolítico, abrigo |
| **T3** | ~30 | Neolítico catalán | cova, nivell, neolític, jaciment, cardial |
| **T4** | ~70 | Bronce/Funerario | bronce, funerario, enterramiento, ajuar |
| **T5** | ~60 | Asentamientos | asentamiento, poblado, milenio, actividad |
| **T6** | ~50 | Bioarqueología/EN | date, sample, bone, human, individual |
| **T9** | ~40 | Arqueología portuguesa | lisboa, anta, ocupação, estrutura, portugal |

---

## 5. BERTopic: intento fallido

Se intentó BERTopic como alternativa moderna basada en embeddings semánticos. Se probaron 4 versiones:

| Versión | Input | Tópicos | Problema |
|---|---|---|---|
| v1 | Tokens lematizados | 3 | 1 tópico con 83% de docs |
| v2 | Tokens + UMAP ajustado | 3 | 1 tópico con 84% de docs |
| v3 | Tokens limpios de ruido | 3 | 1 tópico con 86% de docs |
| v4 | Texto completo (8000 chars) | 2 | 1 tópico con 90% de docs |

### ¿Por qué falla BERTopic?

1. **Homogeneidad del corpus**: Todos los papers comparten vocabulario arqueológico similar (yacimiento, nivel, estrato, cerámica...)
2. **Embeddings multilingües**: No separan bien cuando el contenido temático se solapa entre idiomas
3. **HDBSCAN**: No encuentra clusters densos separados en el espacio de embeddings de 384 dimensiones
4. **Textos cortos tras lematización**: ~687 tokens no proporcionan suficiente contexto semántico
5. **Ruido del parser**: Tokens como "picture intentionally omitted" contaminan los embeddings

**Conclusión:** LDA clásico es más adecuado para este corpus específico. BERTopic funcionaría mejor con corpus más heterogéneos o con textos más largos y diferenciados.

---

## 6. Estructura de outputs

```
/home/raglinux/RAG_graph/lda/
├── config.yaml                          # Configuración completa
├── scripts/
│   ├── utils.py                         # Funciones compartidas (RAM monitor, logging)
│   ├── 01_parse.py                      # PyMuPDF4LLM → parquet
│   ├── 02_extract_meta.py               # DOI, título, autor
│   ├── 03_detect_lang.py                # langdetect por documento
│   ├── 04_clean.py                      # Limpieza de texto
│   ├── 05_lemmatize.py                  # spaCy multilingüe
│   ├── 06_build_corpus.py               # Dictionary + BoW
│   ├── 07_train_lda.py                  # Grid search + coherence
│   ├── 08_assign.py                     # Tópico dominante
│   ├── 09_visualize.py                  # pyLDAvis + top-15
│   ├── 10_export.py                     # CSV + JSON + alineación
│   ├── bertopic_run.py                  # BERTopic v1
│   ├── bertopic_run_v2.py               # BERTopic v2
│   ├── bertopic_run_v3.py               # BERTopic v3
│   └── bertopic_run_v4.py               # BERTopic v4
├── data/
│   ├── 01_parsed.parquet                # Texto extraído
│   ├── 02_meta.parquet                  # + metadatos
│   ├── 03_lang.parquet                  # + idioma
│   ├── 04_cleaned.parquet               # + texto limpio
│   ├── 05_lemmatized.parquet            # + tokens lematizados
│   ├── 06_dictionary.gensim             # Diccionario gensim
│   ├── 06_corpus.pkl                    # Corpus BoW
│   ├── coherence_scores.csv             # Métricas de todos los K
│   └── memory_usage.csv                 # Log de RAM por fase
├── models/
│   ├── lda_K5_BEST/                     # Modelo óptimo
│   ├── lda_K10/                         # Modelo alternativo
│   ├── lda_K{15,20,25,30}/              # Modelos del grid
│   └── bertopic_v4/                     # BERTopic (no recomendado)
├── outputs/
│   ├── papers_topics.csv                # CSV principal (513 docs × metadatos × tópico)
│   ├── topics_keywords.csv              # Top-20 palabras por tópico
│   ├── top15_papers_per_topic.csv       # 15 papers más representativos
│   ├── pyldavis_K5.html                 # Visualización interactiva
│   ├── topics_for_graph.json            # JSON listo para Neo4j
│   ├── topics_alignment.csv             # Alineación cross-language
│   └── bertopic_v4_*.csv                # Outputs BERTopic
└── logs/
    └── *.log                            # Logs de cada script
```

---

## 7. Cómo reproducir

### Requisitos

```bash
# Activar entorno virtual
source /home/raglinux/RAG_graph/.venv/bin/activate

# Instalar dependencias
pip install pymupdf4llm spacy gensim langdetect pyldavis pyarrow pandas nltk psutil pyyaml scikit-learn matplotlib seaborn deep-translator

# Instalar modelos spaCy
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download pt_core_news_sm
python -m spacy download ca_core_news_sm
```

### Ejecutar pipeline completo

```bash
cd /home/raglinux/RAG_graph/lda/scripts

python 01_parse.py        # ~2 horas (525 PDFs)
python 02_extract_meta.py # ~10 segundos
python 03_detect_lang.py  # ~10 segundos
python 04_clean.py        # ~10 segundos
python 05_lemmatize.py    # ~5 minutos
python 06_build_corpus.py # ~5 segundos
python 07_train_lda.py    # ~5 minutos (grid search)
python 08_assign.py       # ~5 segundos
python 09_visualize.py    # ~10 segundos
python 10_export.py       # ~15 segundos (incluye traducción)
```

### Ejecutar BERTopic (no recomendado para este corpus)

```bash
python bertopic_run_v4.py  # ~2 minutos
```

---

## 8. Limitaciones y mejoras futuras

### Limitaciones actuales

1. **T1 contaminado**: Stopwords españolas no filtradas completamente. Se necesitan más stopwords custom.
2. **Tópicos por idioma**: LDA no fusiona automáticamente tópicos equivalentes entre idiomas (ej: "cerámica" ES vs "ceramic" EN). El post-proceso de alineación sugiere fusiones pero no las aplica.
3. **Metadatos incompletos**: 225 docs sin autor, 8 sin título, 243 sin DOI.
4. **12 PDFs descartados**: Texto insuficiente (<500 chars), posiblemente escaneados sin OCR.
5. **BERTopic no viable**: No encuentra estructura de clusters en este corpus homogéneo.

### Mejoras propuestas

1. **Ampliar diccionario de stopwords**: Añadir más stopwords específicas por idioma, especialmente para español.
2. **LDA con K=10 como base**: Mayor granularidad, mejor separación de temas.
3. **OCR para PDFs fallidos**: Usar `ocrmypdf` para los 12 PDFs descartados.
4. **BERTopic con texto completo sin lematizar**: Podría funcionar si se usa el texto raw en lugar de tokens lematizados.
5. **Integración con el grafo**: Los outputs `topics_for_graph.json` están listos para cargar en Neo4j como nodos `Topic` con relaciones `(:Paper)-[:BELONGS_TO {weight}]->(:Topic)`.

---

## 9. Conexión con el grafo de conocimiento

El archivo `outputs/topics_for_graph.json` contiene la estructura lista para Neo4j:

```json
[
  {
    "id": "T00",
    "topic_id": 0,
    "name": "Topic_00",
    "keywords": ["date", "sample", "human", "bone", ...],
    "num_papers": 145,
    "papers": [
      {
        "filename": "1000_oms_2017.pdf",
        "title": "First Evidence of Collective Human Inhumation...",
        "author": "F. Xavier Oms",
        "doi": "10.1080/00934690.2016.1260407",
        "language": "en",
        "weight": 0.85
      }
    ]
  }
]
```

### Cypher para cargar en Neo4j

```cypher
// Crear nodos Topic
UNWIND $topics AS topic
CREATE (t:Topic {
  id: topic.id,
  name: topic.name,
  keywords: topic.keywords,
  num_papers: topic.num_papers
})

// Crear nodos Paper y relaciones
UNWIND topic.papers AS paper
MERGE (p:Paper {filename: paper.filename})
SET p.title = paper.title,
    p.author = paper.author,
    p.doi = paper.doi,
    p.language = paper.language
CREATE (p)-[:BELONGS_TO {weight: paper.weight}]->(t)
```

---

## 10. Referencias

- **Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).** Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022.
- **Gensim:** Řehůřek, R. & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. *LREC Workshop*.
- **Coherence c_v:** Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the Space of Topic Coherence Measures. *WSDM*.
- **BERTopic:** Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.

---

## 11. Estadísticas de ejecución

| Fase | Docs | Tiempo | RAM pico |
|---|---|---|---|
| Parseo (PyMuPDF4LLM) | 513/525 | ~2 horas | 10.6 GB |
| Metadatos | 513 | ~2 segundos | 8.9 GB |
| Detección idioma | 513 | ~3 segundos | 9.0 GB |
| Limpieza | 513 | ~3 segundos | 9.0 GB |
| Lematización | 513 | ~5 minutos | 9.9 GB |
| Corpus BoW | 513 | ~2 segundos | 9.0 GB |
| Grid LDA (6 modelos) | 513 | ~5 minutos | 10.2 GB |
| Asignación tópicos | 513 | ~2 segundos | 9.2 GB |
| Visualización | 513 | ~3 segundos | 10.1 GB |
| Export + alineación | 513 | ~15 segundos | 8.8 GB |
| **Total** | **513** | **~2h 15min** | **10.6 GB** |

---

*Generado el 9 de junio de 2026 | Pipeline LDA para IDEArq | 513 documentos arqueológicos multilingües*
