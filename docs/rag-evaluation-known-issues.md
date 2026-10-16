# Problemas conocidos de la evaluación RAG de IDEArq

Estado de los fallos medidos en la evaluación de recuperación (MRR + hit@k) del
sistema RAG de IDEArq, con su evidencia y su estado actual. El vocabulario de
estado es: **arreglado** | **pendiente** | **fuera de alcance**.

Fecha de revisión: 2026-10-08.

---

## 1. `_distance_km` se calculaba y se descartaba al serializar

**Estado: arreglado** (en `src/geo_route.py`).

- **Síntoma.** En la pregunta de Jaén («Which Chalcolithic site is the furthest
  from the city of Jaén, in the province of Jaén?», ítem 1 de
  `data/eval_questions.py`) el retrieval acertaba el yacimiento, pero el modelo
  no lo verbalizaba: el contexto no llevaba la distancia y el modelo se
  inventaba el origen y recalculaba distancias por su cuenta.
- **Evidencia medida.**
  - `_distance_km` aparecía en la metadata del documento pero **no** en el
    contexto serializado: `docs/geo-filter-route.md` §4, tabla fila 0
    (línea 602), documenta que `_ctx_text` solo escribe `lat`/`lon` de cada
    yacimiento y que la ciudad y la distancia que la ruta ya había calculado no
    entraban en el prompt (`context_chars=9570`).
  - La respuesta del modelo usó el origen **37.88, -3.62** en lugar de
    **37.77, -3.79** (`CITY_COORDS["jaén"]` en `src/geo_route.py`), es decir, se
    inventó las coordenadas de la ciudad de Jaén (documentado en
    `docs/geo-filter-route.md` §7.2, «El modelo inventó las coordenadas de la
    ciudad de Jaén»).
  - La ruta devolvía como top-1 **Eras del Alcázar a 45,9 km**, la respuesta
    correcta (`docs/geo-filter-route.md` líneas 409 y 576).
- **Causa.** `retrieve_filtered` calculaba `_distance_km` y lo dejaba en la
  metadata del `Document`, pero la serialización del contexto (`_ctx_text`, más
  tarde `ctx_text`) solo emitía las coordenadas del yacimiento; la distancia se
  perdía entre la ruta y el prompt.
- **Arreglo.** `ctx_text()` en `src/geo_route.py` es ahora la única fuente de
  serialización e incluye la línea `Distancia a <origen>: <km> km`, usando
  `_distance_from` (o el `reference` que se le pase). La celda 13 del notebook
  consume `ctx_text` importado del módulo (ya no tiene copia propia).
- **Verificación.** Ejecutando la celda 15 para la pregunta de Jaén, el contexto
  contiene `Distancia a jaén: 45.9 km` y el primer sitio estructurado es
  `Eras del Alcázar`.

---

## 2. Configuración cruzada de colección (`EMBEDDING_KEY` vs `GEO_FILTER_COLLECTION`)

**Estado: arreglado.**

- **Síntoma.** La celda 1 declaraba
  `EMBEDDING_KEY = "all-MiniLM-L6-v2"` con el comentario
  `# -> colección IdearqGTE_800_50_v2`, que es falso. La ruta se invocaba como
  `retrieve_filtered(question)` sin `collection` y caía en
  `GEO_FILTER_COLLECTION` (MiniLM), mientras el resto del notebook podía ir por
  GTE. Ruta y contexto podían medir colecciones distintas sin avisar.
- **Evidencia medida.**
  - `src/config.py` → `collection_name()` mapea `all-MiniLM-L6-v2` →
    `IdearqMiniLM_800_50_v2` y `gte-multilingual-base` → `IdearqGTE_800_50_v2`
    (`src/config.py` líneas 39-47). El comentario de la celda 1 decía lo
    contrario.
  - `src/geo_route.py` → `GEO_FILTER_COLLECTION = "IdearqMiniLM_800_50_v2"`
    (constante legada).
  - La celda 15 antigua hacía `collection = collection or GEO_FILTER_COLLECTION`,
    un default implícito.
- **Causa.** Default implícito de colección más un comentario incorrecto en la
  celda de configuración.
- **Arreglo.** La celda 1 deriva explícitamente
  `ROUTE_COLLECTION = collection_name(EMBEDDING_KEY)` y la celda 13 llama
  `retrieve_filtered(question, collection=ROUTE_COLLECTION)`. En el módulo,
  `retrieve_filtered()` ya no aplica ningún default: lanza `ValueError` si la
  colección es falsa, de modo que es imposible consultar una colección distinta
  en silencio.

---

## 3. BM25 y multilingüismo

**Estado: fuera de alcance** (corresponde a la fase de ablación).

- **Síntoma.** La pregunta catalana del Mesolítico funerario (ítem 17 de
  `data/eval_questions.py`) no se responde: el fichero correcto entra en la
  recuperación, pero el hecho no.
- **Evidencia medida.**
  - BM25 indexa términos literales. Con la pregunta en catalán y el paper en
    inglés no hay correspondencia de términos: `cementiri` (ca) no casa con
    `cemeteries` (en) del artículo.
  - Los aciertos de `cementiri` en el corpus son un topónimo irrelevante
    (`833_2013.pdf`, «Puig Roig o Cementiri del Moros»), no el cementerio
    mesolítico que pide la pregunta.
  - La fusión RRF **no descarta una pierna mala**: sus candidatos desplazan a
    los buenos. El paper `919_JuanGibaja_2015.pdf` estaba en el rank 8 de la
    pierna vectorial y salió fusionado en el 15.
  - La fuente declarada del ítem 17 es
    `1168_The mesolithic of atlantic coastal spain_2022.pdf; 919_JuanGibaja_2015.pdf`
    (`data/eval_questions.py` línea 218).
- **Causa.** Léxico BM25 monolingüe (sin normalización interlingüe) más una
  fusión RRF sin poda de la pierna degradada.
- **Estado.** Fuera de alcance: cambiar BM25 o la fusión pertenece a la fase de
  ablación, no a este endurecimiento de la ruta.

---

## 4. Recuperación a nivel de trozo

**Estado: pendiente.**

- **Síntoma.** El fichero correcto entra en el conjunto recuperado, pero el
  fragmento que contiene el hecho no: los trozos elegidos son tablas de fechas.
- **Evidencia medida.**
  - En la pregunta catalana del Mesolítico, `919_JuanGibaja_2015.pdf` entra en
    la posición 8/8, pero el trozo con el hecho no está entre los recuperados
    (`odd/tasks/geo-route-hardening.md`, tabla de causas).
  - `docs/geo-filter-route.md` fila 5 (línea 605) documenta el caso simétrico:
    de un PDF de 561.758 caracteres entró un solo trozo.
- **Causa.** El ranking opera sobre embeddings de trozo; un solo trozo
  representativo del fichero no garantiza que el trozo relevante entre en el
  top-k.
- **Estado.** Pendiente (mejora de selección/expansión a nivel de trozo).

---

## 5. Presupuesto de contexto

**Estado: mitigado** (cuota por fichero y presupuesto de caracteres);
**pendiente** subir `num_ctx`.

- **Síntoma.** El prompt de la pregunta «máxima datación por comunidad» (ítem 18)
  llegó a 29.461 tokens y Ollama lo truncó a 8.194. El patrón buscado estaba en
  el offset 2.495 y se tiró.
- **Evidencia medida.** `odd/tasks/geo-route-hardening.md` línea 14:
  «prompt 29.461 tok truncado a 8.194; patrón en offset 2.495».
- **Causa.** `num_ctx` de Ollama por debajo del tamaño real del prompt agregado.
- **Arreglo parcial.** `src/geo_route.py` limita el bloque de PDFs con
  `PDF_CHAR_BUDGET = 16000` y aplica una cuota por fichero
  (`MAX_CHUNKS_PER_FILE = 2`) tras la fusión, de modo que un solo documento no
  monopoliza el contexto.
- **Pendiente.** Subir `num_ctx` en la configuración de Ollama.

---

## 6. Metadatos del extractor (`year` / `language`)

**Estado: extractor arreglado; backfill masivo pendiente de decisión.**

- **Síntoma.** Las portadas de ResearchGate se adherían al frente de muchos PDFs
  y contaminaban los metadatos: `year` y `language` incorrectos.
- **Evidencia medida.**
  - Auditoría del extractor: **27,3 %** de los `year` no coincide con el año del
    fichero; **20 %** tienen `language='en'` incorrecto.
  - El arreglo del extractor está en `src/reindex_gte_e5.py` línea 66
    («Drop the ResearchGate cover sheet glued to the front of many PDFs»).
  - No afecta a la recuperación porque `year` y `language` no se usan como
    filtros en la ruta (`_supporting_pdfs` no filtra por metadatos;
    `where` se limita a `file_type`/`filename`).
- **Causa.** La portada de ResearchGate aportaba texto y metadatos que el
  extractor tomaba como del documento.
- **Estado.** Extractor arreglado; el backfill masivo de los 300 PDFs ya
  indexados queda pendiente de decisión (implica reindexación).

---

## 7. Atributos estructurados (solo en el GeoJSON)

**Estado: fuera de alcance** (limitación del corpus; afecta al paso final del
agregado, no a la ruta).

- **Síntoma.** Los atributos estructurados (`bp_values`, `lat`, `lon`, `region`,
  `yacimiento_nombre`) solo los tiene `c14_v2.geojson`. Ninguno de los ~300 PDFs
  los tiene. Además, `region` guarda la **provincia**, no la comunidad autónoma.
- **Evidencia medida.**
  - `src/config.py` → `INDEX_PROPERTIES_FULL` documenta que los PDFs dejan esas
    propiedades en `null` (comentario en las líneas 66-95).
  - La ruta filtra por `*.geojson` (`src/geo_route.py`, cláusula
    `filename Like "*.geojson"`), por lo que los PDFs solo entran por la pierna
    de artículos de apoyo, como texto.
  - `docs/geo-filter-route.md` fila 4 (línea 616) y tabla de la línea 237:
    `region` = provincia (Jaén, Almería, Gipuzkoa…), y el paso de provincia a
    comunidad lo hace el modelo.
- **Causa.** El GeoJSON es la única fuente con estructura; los PDFs son texto.
- **Estado.** Fuera de alcance: es una limitación de datos. Cualquier cambio
  (backfill de atributos en PDFs) pertenece a indexación, no a esta ruta.

---

## 8. Etiquetas de evaluación

**Estado: pendiente.**

- **Síntoma.** La etiqueta 3 (Villaricos) arrastraba un `answer` que su propia
  `note` declara inexistente. El patrón de chequeo de la pregunta catalana
  (`cementiri`) es inalcanzable con el paper en inglés. El pin de la celda 7 usa
  un texto que ya no coincide con `data/eval_questions.py`.
- **Evidencia medida.**
  - `data/eval_gold_labels.py`, etiqueta 3, `note`: «the original GT (Avenida
    Pablo Iglesias / Rafaela Jiménez / Calle La Central / Calle Castillejo,
    Gádor) occurs in NO document, and 483_2001.pdf is absent (only
    483_2001_error.pdf exists…)». La etiqueta se reescribió el 2026-10-05 con la
    respuesta literal del índice; el `answer` actual ya es el nuevo, pero el
    texto viejo sigue citado en el historial y en el pin del notebook.
  - El patrón del chequeo catalán busca `cementiri` en el corpus, pero el paper
    relevante (`919_JuanGibaja_2015.pdf`) está en inglés y usa `cemeteries`
    (véase el problema 3).
  - **Pin desactualizado:** `notebooks/evaluation/viz-geo-check.ipynb` celda 7,
    `VIZ_SMOKE_PATTERNS`, incluye
    `"Which emergency excavation carried out in Villaricos"`, mientras que el
    texto actual del ítem 2 en `data/eval_questions.py` (línea 27) es
    «Emergency excavations carried out by the Regional Government of Andalusia
    in the province of Almería, published in 2001 — which sites?». El patrón no
    es subcadena de la pregunta: la celda 7 marcaría la pregunta como
    `MISSING/DUP` y lanzaría `RuntimeError` con el dataset local.
- **Causa.** Etiquetas y pin evolucionaron por separado; el dataset de Langfuse
  que la celda 7 intenta cargar primero puede seguir sirviendo la versión previa.
- **Estado.** Pendiente: actualizar `VIZ_SMOKE_PATTERNS` con el texto actual y
  verificar/actualizar el dataset de Langfuse. (El `answer` de
  `data/eval_gold_labels.py` ya está corregido.)

---

## 9. Deuda técnica aceptada

**Estado: fuera de alcance / aceptada.**

- **Cliente Weaviate.** `_gql_query` (alias de `_graphql`) sigue usando `urllib`
  en lugar del cliente oficial de `src/weaviate_conn.py`; no comparte lógica de
  conexión. Documentado en el docstring de `src/geo_route.py` (líneas 6-9).
- **Sin tests de integración.** Solo existe `tests/test_geo_route.py` (tests
  unitarios de `plan_geo_query`, `ctx_text`, etc.); no hay pruebas de
  integración contra Weaviate.
- **Sin git.** El repositorio no es git (`git rev-parse` → «not a git
  repository»), así que no hay commits por unidad de trabajo ni historial
  navegable.
