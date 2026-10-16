# Ruta geo + gold labels — registro completo del 30-09-2026

> Documento de trabajo. Recoge **qué** se cambió, **qué** falló por el camino y
> **qué** queda pendiente, con los resultados de las micro-pruebas y de la evaluación
> en viz. Vive en `docs/` junto al resto de documentación del proyecto (antes estaba
> en `odd/tasks/`).

---

## 1. Resumen

Las cinco preguntas del dataset cuyo Ground Truth está en el GeoJSON (ids **2, 5, 6,
18, 19**) no se podían responder con el retrieval anterior. Hoy las responde una
ruta nueva: **filtro sobre atributos estructurados + agregación**, en vez de top-k por
similitud. Además existe un fichero de **gold labels** para medir retrieval sin pagar
llamadas al juez LLM.

| Métrica | Resultado |
|---|---|
| Preguntas geo enrutadas a la ruta de filtro | 5 / 5 |
| hit@10 | 1 en las 5 |
| recall de gold docs | 100% (#2, #5, #6) · 58% (#18) · 44% (#19) |
| top-1 = GT (retrieval, **no** generación) | #2 sí (Eras del Alcázar, 45,9 km) · #6 sí (Trikuaizti 2, 12.015 BP) · el modelo aún no lo verbaliza (§7.3) |
| **Eval viz end-to-end (job 10180170, 01-10)** | **hit@k 5/5 · MRR 0,800 · faithfulness 0,732 · las 5 geo por `filter:*`, los controles reales por `topk` |

Detalle en §7.1.

---

## 2. El problema de fondo

**El top-k por similitud no puede responder preguntas de superlativo.** Para *"¿qué
yacimiento calcolítico está más alejado de Jaén?"*, los documentos que más se parecen a
la pregunta son los que están **más cerca** de Jaén: el más lejano es, por definición,
el peor *match* de la consulta.

Medido: `Eras del Alcázar` (la respuesta correcta) **no aparece en ningún top-100**,
con ningún valor de `alpha`, ni en inglés ni en español.

Detrás de eso había un segundo problema: **las fechas no eran números**. El campo
`bp_max` estaba declarado en el esquema pero vacío en los 4.478 objetos GeoJSON, así
que no se podía filtrar "dame la fecha más antigua".

---

## 3. Qué se implementó

### 3.1 Enriquecimiento del índice (`notebooks/indexing/rag-indexacion-mejorado.ipynb`)

`INDEX_PROPERTIES_FULL` ya declaraba `bp_min`, `bp_max`, `bp_mean`, `c14_count`,
`periodo` y `region`, pero `load_geojson()` nunca las rellenaba. Se añadió:

- Celda con los extractores puros: `parse_c14_bp`, `parse_periodo`, `parse_region`,
  `enrich_geo_props`.
- `load_geojson()` ahora hace `**enrich_geo_props(props)` en el metadata → un
  reindex futuro rellena las columnas solo.
- Celda de *backfill* `backfill_geo_properties()`: rellena **en sitio** con PATCH de
  propiedades, **sin recomputar los 58.236 vectores**. Idempotente.

Fuentes de datos: `dataciones_c_14` (fechas, separadas por `<br/>`), `tipologia_crono`
(periodo entre paréntesis: `Funerario, Megalito funerario (Neolítico)`) y
`unidad_territorial` (`A Coruña (A Coruña, ES)` → provincia).

Cobertura sobre los ficheros: 8.833/9.125 features con fecha (97%), periodo 97%,
región 99,6%. Los 292 sin fecha son `"sin resultado"` / `"Datación desconocida"`.

### 3.2 Cambios en Weaviate

| Colección | Estado | Contenido |
|---|---|---|
| `IdearqMiniLM_800_50_v2` | **enriquecida** | 4.478 objetos GeoJSON → 2.207 yacimientos con fecha; **4.414 de 4.478 objetos (98,6%)** con `bp_max`; 4.310 con `periodo`; 4.430 con `region` |
| `IdearqGTE_800_50_v2` | **sin tocar** | sigue con las columnas vacías, a propósito |
| `Idearqe5largeinstruct_800_50_v2` | **sin tocar** | pendiente |

Qué **no** se tocó en ningún caso: vectores, `page_content`, `lat`/`lon`,
`filename`, `yacimiento_id`, `yacimiento_nombre`. Por eso **`RETURN_PROPS` no necesitó
cambios** (el esquema no cambió).

Cómo se hizo: `PATCH /v1/objects/{coleccion}/{uuid}` con `{"properties": {...}}`. En esta
versión de Weaviate la clase va **en la ruta** (`/v1/objects/{uuid}` → error
`empty class`) y la respuesta es **204 sin cuerpo**.

### 3.3 La ruta de filtro (`notebooks/evaluation/rag-evaluacion-hibrida-multiquery.ipynb`)

Celdas nuevas `geo-route-md` / `geo-route-code`, más `ret-metrics-md` /
`ret-metrics-code`.

- **El planificador es determinista, sin LLM.** Se comparan palabras clave de
  superlativo y nombres de periodo contra **el vocabulario real de la colección**.
  Esto lo hace auditable y medible con hit@k; un clasificador LLM no lo sería.
- Tipos de plan: `superlative_space`, `superlative_date`, `lookup_by_date`,
  `group_by_region`.
- `retrieve_filtered()` deduplica por `yacimiento_id` (el corpus guarda 2 objetos por
  yacimiento) y **construye el contexto con `_ctx_text()`**, para que las coordenadas
  lleguen al prompt.
- `run_rag()` llama a la ruta primero y cae a la estrategia top-k si no hay plan. La
  fila del CSV lleva `route` = `topk` o `filter:<kind>`.
- Las 3 preguntas bibliográficas de control **deben** quedarse en `topk`: si no, la
  puerta está secuestrando preguntas que no son suyas.

### 3.4 Gold labels (`data/eval_gold_labels.py`)

**Actualizado 01-10: las 32 preguntas tienen etiqueta** — 31 graduables + 1 ingradable.
Antes había 5 etiquetas y 27 preguntas que el métrico saltaba en silencio (la fila quedaba
`hit_at_k = NaN` sin avisar de nada).

Claves del dict = **índice 1-based de la pregunta**. Cada entrada lleva ahora
`verified` / `needs_review` / `note` (y `ungradable` donde aplica).

**Dos orígenes de etiqueta, con procedencias distintas:**

| Origen | Preguntas | Cómo se localiza el gold doc |
|---|---|---|
| **GeoJSON** (las 5 originales + `#4`, `#7`, `#20`) | 8 | **desde el GT**: por nombre de yacimiento (`Eras del Alcázar`, `Trikuaizti 2`) o **coincidiendo la fecha exacta** que dice el GT (Ardales 51.914, Eirós 31.690, Higuera 12.030…). Nunca desde la agregación que evaluamos: si usáramos el mismo filtro, hit@k sería circular. |
| **PDF** (el resto) | 23 | del campo `fuente` del dataset, resuelto contra los 526 PDF del corpus (exacto → por id numérico `483_`, `1106_`… → por texto libre) y **verificado con `pdftotext`**: el texto del GT tiene que aparecer en el PDF etiquetado |

**Verificación con `pdftotext`:** **18 etiquetas `verified: True`, 9 `verified: False`
y marcadas `needs_review`** — se registra qué no se encontró y se conserva el candidato del
`fuente`; **no se cambió ningún documento para que la comprobación pasara**.

**Actualizado 05-10 — la etiqueta `#3` se reescribió** (ver detalle abajo). Antes decía:
*"A site on Avenida Pablo Iglesias… on Calle La Central in Villaricos… on Calle Castillejo
(Gádor, Almería)"* con `fuente = 483_2001.pdf`. Se comprobó contra los **525 PDF del
corpus** que **ninguna de esas calles aparece en ningún documento** (`Avenida Pablo Iglesias`,
`Rafaela Jiménez`, `Calle La Central`, `Calle Castillejo` → 0 coincidencias; lo que sí aparece
—`Castillejo`, `Pablo Iglesias`— son apellidos de autores y una calle de otra ciudad), y que
`483_2001.pdf` **no existe**: solo `483_2001_error.pdf`, un índice del *Anuario Arqueológico
de Andalucía*.

Como la pregunta no se podía graduar, **se rehízo la pregunta y su GT**, tomando la respuesta
**literal del PDF** (líneas 151-152 del índice):

> **Pregunta nueva:** *Which emergency excavation carried out in Villaricos, Cuevas del
> Almanzora (Almería) is listed in the index of the Anuario Arqueológico de Andalucía?*
> **GT nuevo:** *"Excavación de urgencia en Villaricos, Cuevas del Almanzora, Almería"* —
> `source = 483_2001_error.pdf`, `verified: True`, `needs_review: False`.

Se descartó deliberadamente conservar el "¿qué sitios?" original: el índice está fragmentado
por columnas en el OCR y otras urgencias de Almería (Almizaraque / Cabecico de Parra,
líneas 83-87) **no se pueden atribuir con certeza**, así que una enumeración completa habría
sido un GT inventado. **No se tocó ningún documento.**

| `needs_review` (11) | Motivo |
|---|---|
| `#4` | la fecha `175 ± 20` y "Ladera Sur" no están en el GeoJSON; sí en `1135_Poblado_2021.pdf`, que **no** es la fuente declarada → decidir si se promueve |
| `#15`, `#30` | sustituciones por texto libre (`993_Bartelheim…` → `1093_adressing_2020.pdf`, cubre 12 de 21 sitios; `Variscita_jadeita` sin coincidencia) |
| `#9` | error tipográfico del GT (`Torreló` vs `Torrelló d'Onda`) |
| `#17`, `#26`, `#27`, `#28` | términos del GT (tipología, armas, `12.000 cal BC`) ausentes en la fuente declarada |
| `#20` | derivada por haversine ≤ 150 km desde Casa Montero (24 yacimientos vs los 23 del GT) → **se calcula lo mismo que filtra la ruta**, riesgo de circularidad |
| `#31` | **ingradable**: la fuente es una URL externa (nationalgeographic) que no está en el corpus. `gold_docs = []`; en su nota consta que `fuente` también resuelve dos PDFs (ámbar/marfil) por si se promueven |

**Cambio en el métrico:** `annotate_retrieval_metrics()` solo comparaba `yacimiento_id`,
así que **una fila de PDF nunca podía puntuar**. Ahora hay clave `("file", nombre)` y
fallback por nombre de archivo normalizado (acentos, apóstrofes, guiones, `.pdf`), que solo
se activa para etiquetas sin `yacimiento_id` — **las 5 etiquetas geo se comportan igual que
antes** (verificado: sus filas salen byte-idénticas).

`UNLABELLED_QUESTIONS` pasó de 0-based (¡incompatible con las claves 1-based de
`GOLD_LABELS`!) a **1-based** y ahora vale `[31]` (solo la ingradable).

Las claves son independientes de la colección (`filename` + `yacimiento_id` + nombre +
fecha), así que la misma etiqueta sirve en cualquier colección.

### 3.5 Métricas de retrieval — `hit@k` y MRR

**La pregunta que responden:** ¿el documento que buscamos apareció en la lista que
devolvió el sistema?

Ejemplo real con la pregunta de Jaén (`k = 8`). El sistema devolvió 6 documentos, en
este orden:

```
1. Eras del Alcázar   ← el que tiene la respuesta
2. Albalate
3. …
```

- **`hit_at_k`** — acierto sí/no: ¿está el documento bueno entre los 8 primeros?
  Aquí **1.0** (sí). Si no llegara: **0**.
- **`reciprocal_rank` (MRR)** — en qué posición salió: `1 / posición`. Salió
  **primero** → `1/1 = 1.0`. Si saliera tercero → `1/3 = 0,33`.

**Para qué sirve cada una:**

| Métrica | Responde |
|---|---|
| `hit@k` | ¿el sistema **encuentra** el documento? |
| MRR | ¿lo encuentra **arriba**? (importa: el modelo solo lee los primeros) |

Media de las 5 preguntas geo en el job 10180170: **`hit@k = 1,0`** y **MRR = 0,80** →
siempre lo encontró, y en dos preguntas salió en segunda posición.

Se usan en lugar de `context_precision`/`context_recall` porque esas necesitan llamadas
al LLM (1 y `k` llamadas), tardan 54 s-8 min cada una y fueron la causa de todos los
NaN. Estas dos coste cero, son deterministas y comparables entre corridas.

### 3.6 Los cuatro filtros y las propiedades que hay que rellenar en Weaviate

Hay **dos formas** de buscar en la base:

| | Por similitud (`topk`) | Por filtro (`filter`) |
|---|---|---|
| Qué hace | calcula el parecido entre vectores | condiciones tipo SQL: *"que el periodo sea calcolítico"* |
| Preguntas típicas | bibliográficas | geo: fechas, distancias, regiones |
| ¿Usa embeddings? | sí | **no** |

`plan_geo_query()` lee la pregunta y decide qué filtro usar según las palabras clave:

| Si la pregunta dice… | Plan | Filtro | Orden |
|---|---|---|---|
| "el **más lejos** de Jaén" | `superlative_space` | periodo + región | por distancia (Python) |
| "la fecha **más antigua** de un dolmen" | `superlative_date` | periodo + funerario/megalito | por `bp_max` |
| "**¿33.900±60** y 12.030±0…?" | `lookup_by_date` | esas fechas concretas | sin orden |
| "lo más antiguo de **cada** comunidad" | `group_by_region` | periodo | mayor de cada región (Python) |

Consulta real de la pregunta de Jaén (el filtro va **dentro** del GraphQL):

```graphql
{Get{IdearqMiniLM_800_50_v2(where:{operator:And,operands:[
  {path:["filename"], operator:Like, valueText:"*.geojson"},
  {path:["periodo"],  operator:Like, valueText:"*calcolítico*"},
  {path:["region"],   operator:Like, valueText:"*Jaén*"}
]}, limit:600){content yacimiento_id …}}
```

Eso trae hasta 600 filas **sin ninguna puntuación de relevancia**. Después Python ordena
(esta versión de Weaviate no sabe ordenar por una distancia calculada) y se queda con las
primeras.

**Propiedades que hubo que rellenar en Weaviate** — estaban vacías en el índice; ese es
trabajo del enriquecimiento GeoJSON + `backfill_geo_properties()`:

| Propiedad | Qué guarda | Quién la usa |
|---|---|---|
| `filename` | de qué fichero viene el documento | filtro (dejar solo la base C14) |
| `periodo` | calcolítico, neolítico, mesolítico… | filtro |
| `region` | Jaén, Almería, Gipuzkoa… | filtro |
| `tipologia_crono` | funerario, megalito, hábitat… | filtro |
| `bp_max` / `bp_min` / `bp_mean` | la datación radiocarbónica | filtro **y** orden |
| `lat`, `lon` | coordenadas del yacimiento | distancia, **calculada en Python** |
| `yacimiento_id`, `yacimiento_nombre` | identificador del sitio | métricas `hit@k` / MRR |

Detalle importante: `lat` y `lon` **no se filtran** en GraphQL; se leen después, dentro
de Python, para el haversine. Y `CITY_COORDS` (coordenadas de las ciudades) es una lista
escrita en el código, no una propiedad de la base.

Desde **2026-10-15** se puede cargar una tabla externa con `GEO_CITY_COORDS_FILE` (ruta a un CSV
`nombre,lat,lon`): sus filas **sobrescriben** las entradas internas del mismo nombre (sin distinguir
tildes ni mayúsculas) y **añaden** el resto; sin la variable, todo sigue igual. La tabla en uso es
`data/geo/capitales_provincias.csv` — 50 capitales de provincia, Nomenclátor Geográfico Básico del
IGN — y Q1 ya se re-probó con ella encendida (`GEO_USE_CITY_COORDS=1`).

---

## 4. Problemas encontrados y cómo se resolvieron

| # | Problema | Causa | Solución |
|---|---|---|---|
| 1 | `context_precision` = 0/6, `answer_correctness` = 2/6 | El juez va a K llamadas secuenciales por fila; con la contención del vecino (0,06 t/s) el presupuesto de 900 s se agotaba | `RunConfig(timeout=3600, log_tenacity=True)` |
| 2 | La respuesta de Jaén decía "no hay coordenadas" | `load_geojson` ponía `lat`/`lon` solo en metadata, nunca en el texto | `_ctx_text()` inyecta `Coordenadas de <nombre>: lat=…, lon=…` + `RETURN_PROPS` pide `lat`/`lon` |
| 3 | ¿Por qué el micro con ES no trajo a Eras del Alcázar? | **No es un bug**: sin filtro top-k, no aparece ni a k=100 | Ruta de filtro (sección 3.3) |
| 4 | `NotebookValidationError: 'outputs' is a required property` | Las celdas nuevas se insertaron sin `outputs`/`execution_count` | Se rellenaron y se validó con `nbformat.validate` |
| 5 | El job viz falló en 30 s | (mismo problema que #4) | Corregido y reenviado |
| 6 | La puerta no se disparaba para Jaén | `_collection_values` usaba un filtro `NotNull` que este Weaviate rechaza → vocabulario vacío → `except` silencioso → `plan: None` | Lectura client-side con caché, y aviso si falla |
| 7 | `lookup_by_date` reventaba | `ContainsAny` sobre `bp_max` (número) con `valueText` | `valueNumber` |
| 8 | Filtro de `periodo` devolvía 0 objetos | **El `Like` de Weaviate es sensible a tildes**: filtrar `paleolitico` (fold) no encuentra `paleolítico` | Los valores del filtro se usan **verbatim** del vocabulario de la colección |
| 9 | Solo el **49%** de los objetos tenía fecha | El primer backfill deduplicaba por `yacimiento_id` y parcheaba 1 de los 2 objetos por yacimiento; mi verificación con `limit 400` fue engañosa porque Weaviate devuelve primero los parcheados | Se parchean los **2** objetos de cada yacimiento → 98,6%; verificación sobre el **100%** |
| 10 | "Chalcolithic" y "Mesolític" no casaban con `calcolítico` / `mesolítico` | La palabra inglesa y la española difieren, y el español mantiene la -o final | `PERIOD_SYNONYMS` (mapa estático) + `_period_key()` que quita la -o; el valor devuelto es el de la colección, sin fold |
| 11 | `periodo` multi-valor no aparecía | Se une con coma **y** con barra: `"epipaleolítico / mesolítico"` → había que partir por ambos |
| 12 | La 19 devolvía los 10 yacimientos más antiguos del mundo | `"para cada comunidad"` es group-by, no superlativo global; además se truncaba a `top_n` y se perdían regiones en silencio | `per_region` tiene prioridad; group_by no trunca |
| 13 | El filtro con tilde no encuentra "Megalític" | El corpus escribe **"Megalito"** (1.744 objetos) y también "Megalítico" (3.421) | Filtro por `*Megalito*` + nota en el markdown de indexación |
| 14 | El micro-test tardaba 1 h 41 para 6 filas | El juez va en CPU (c201-20, sin GPU): 25 t/s de prompt, 5,5 t/s de generación, 54 s-8 min por llamada | Para esta verificación se usa hit@k, que no paga juez |
| 15 | `nbconvert` execute falla con el notebook fuera del proyecto | El cwd del kernel es el directorio del **fichero de entrada**, así que `PROJECT_ROOT` se resuelve mal | La copia de prueba va dentro de `notebooks/evaluation/` |
| 16 | `pkill -f micro_jaen` mató mi propio shell | `pkill -f` casa con la línea de comandos del propio bash | No usar `pkill -f` con patrones presentes en el comando |
| 17 | El CSV dice `filter:lookup_by_date` pero entraron 8 PDF sin relación | Si el filtro devuelve **0 filas**, el código cae al camino de parecido vectorial **sin comprobarlo** y la columna `route` ya estaba escrita | **Resuelto 05-10** — `route → VACÍO, sin fallback topk`, `n_docs=0` y el contexto es el aviso (§4.1) |
| 18 | `lookup_by_date` no encontraba la datación que se preguntaba | Filtraba por `bp_max` (la fecha **más alta** del yacimiento): la preguntada casi nunca es el máximo (Valencina: 1377 vs `bp_max=4676`) | **Resuelto 05-10** — nueva propiedad `bp_values` (`number[]`) + `ContainsAny valueNumber`, con `abs()` sobre el signo (§4.2) |
| 19 | `dataciones_c_14` (todas las dataciones) no estaba en Weaviate | El GeoJSON la trae pero no entró en el esquema, así que no se podía filtrar por ella | **Resuelto 05-10** — `bp_values` + `dataciones_c_14` añadidas, 4478 objetos, **0 vectores cambiados** (§4.3) |

### 4.1 Bug resuelto: si el filtro no encontraba nada, caía a top-k sin avisar

**Qué pasaba.** `retrieve_filtered()` arma la consulta con condiciones tipo SQL. Si
Weaviate devolvía **cero filas**, el código no lo comprobaba: seguía como si hubiera
funcionado y devolvía documentos del camino de parecido vectorial (8 PDF cualquiera).
Además, la columna `route` del CSV se escribía **antes** de comprobar el resultado.

**Ejemplo real** (fila 7 del job 10180170). La pregunta era *"¿Es correcta la datación
-1377±23 en Valencina…?"*:

```
plan     : lookup_by_date
condición: bp_max ≈ 1377
realidad : el único yacimiento con esa fecha (id 14371) tiene bp_max = 4676
resultado: 0 filas  →  fallback silencioso a vector  →  8 PDF sin nada que ver
respuesta: "No hay información suficiente"
CSV       : route = filter:lookup_by_date   ← como si hubiera funcionado
```

**Por qué importaba.** Un fallo invisible: sin esta revisión habríamos leído *"el
filtro falló en Valencina"* como *"el modelo no supo contestar"*. Un bug que no se ve
no se puede arreglar.

**Qué se hizo (05-10):**

1. `retrieve_filtered()` detecta las 0 filas y devuelve `empty=True` con una nota.
2. El `route` del CSV pasa a `filter:lookup_by_date → VACÍO, sin fallback topk`, con
   `n_docs=0` y `yacimiento_ids` vacío.
3. **El contexto que recibe el modelo es el aviso**, no un PDF al azar.

Salida real de la verificación en frío:

```
filter route EMPTY: NOTE: the geo filter matched 0 records for plan 'lookup_by_date'
    (property bp_max, values 1377). No structured documents…
run_rag route='filter:lookup_by_date → VACÍO, sin fallback topk' n_docs=0
sentinel(topk) leaked: False
```

**Decisión: no se mantiene el fallback vectorial.** Los 8 PDF que entraban antes no
ayudaban (el modelo respondió igualmente *"No hay información suficiente"*), pero sí
podían provocar una respuesta confabulada con apariencia de evidencia. Con la nota,
el fallo es **medible** (`hit@k = 0` honesto, `context_chars` = el aviso) en vez de
quedar oculto. El código deja el fallback a un cambio de una línea, etiquetado, por si
algún día se decide.

Verificación: `nbformat.validate` OK · `compile()` **15/15** (maestro) y **16/16**
(`viz-geo-check.ipynb`, réplica byte-idéntica) · **24/24 comprobaciones PASS** en
cada notebook, con centinela anti-fuga al top-k.

**Estado: RESUELTO.**

> Queda pendiente de comprobación en un run real de Slurm: si RAGAS acepta la fila con
> `retrieved_contexts` vacío, o si el juez revienta y arrastra las métricas de la fila.

### 4.2 Arreglo: `lookup_by_date` busca la datación de verdad

**Antes:** `lookup_by_date` filtraba con `ContainsAny` sobre **`bp_max`**, que es la fecha
**más alta** de cada yacimiento. La fecha que se pregunta casi nunca es el máximo, así que
la consulta devolvía 0 filas (Valencina: pedía `1377`, `bp_max` era `4676`).

**Ahora** filtra con la propiedad nueva `bp_values` (§4.3):

```graphql
{Get{IdearqMiniLM_800_50_v2(where:{operator:And,operands:[
  {path:["filename"],operator:Like,valueText:"*.geojson"},
  {path:["bp_values"],operator:ContainsAny,valueNumber:[1377]}]},limit:600){…}}
```

- **Signo:** la pregunta escribe `-1377` (estilo a.C.) y el GeoJSON guarda `1377` positivo
  (BP). La regex pasó a capturar el signo —`(?<![\d.])([+-]?\d{3,5})\s*(?:±|\+/-)\s*\d+`—
  y el planificador aplica `abs()`, así que los dos signos llegan a lo mismo.
- **Retroceso:** si la colección no tiene `bp_values` (hoy `IdearqGTE_800_50_v2`), se usa
  el `bp_max` antiguo y no se lanza ninguna excepción. Verificado contra GTE.

| Pregunta | Antes | Ahora (verificado en frío) |
|---|---|---|
| **#4** Valencina `-1377±23` | **0 filas** → ruta VACÍO | **6 objetos / 3 sitios**: Cova des Càrritx, **Valencina id 14371**, Castro Ventosa → el oro entra en rango 2 |
| **#5** 33.900±60, 31.690±50, 12.030±0 (Neolítico) | 6 objetos / 3 sitios | **8 objetos / 4 sitios** (superconjunto: se añade Nerja, que sí tiene `12030`) → **hit@k sigue en 1,0** |

**La #5 es la prueba de que la rama funciona**: es la única pregunta de este tipo que ya
respondía bien en el job 10180170, y el arreglo la deja igual o mejor — no la rompe.

### 4.3 Propiedad de dataciones añadida (punto D)

Detalle completo en **`docs/dataciones-property.md`**. Resumen:

| | |
|---|---|
| **Qué** | `bp_values` (`number[]`) + `dataciones_c_14` (`text`), ambas filtrables |
| **Backfill** | **4478 / 4478** objetos actualizados, **0 fallos**, 1,7 s; idempotente |
| **Vectores** | `vectorizer: none` (los envía el cliente) → digest sha256 de los 4478 **idéntico antes/después**: **0 vectores cambiados** |
| **BM25** | `indexSearchable: false` en las dos: si no, una columna de ~1,5 KB habría alterado las puntuaciones híbridas sobre `content` |
| **Cobertura** | 2207 de 2239 yacimientos tienen fechas; 32 no (`bp_max = null`) |

---

## 5. Extremadura y el criterio "paleolítico"

Son **dos motivos distintos** de discrepancia, y conviene no mezclarlos:

### 5.1 Extremadura — hueco real del corpus

El GT dice *"Extremadura, 61.219 ± 70"*. En nuestra colección **el máximo de Extremadura
es 9.640 BP** (Pinho Monteiro). El yacimiento con esa datación **no está en el GeoJSON**,
así que esa línea del GT **no es reproducible con este corpus**. No es un fallo del
sistema: es un límite de los datos.

### 5.2 Asturias (y Baleares, País Vasco, Castilla y León) — criterio distinto

| Comunidad | GT | Nuestro máximo | Site |
|---|---|---|---|
| Principado de Asturias | 16.700 | **46.600** | Jou´l Llobu |
| Illes Balears | 24.220 | **44.800** | Cova de Moleta |
| País Vasco | 34.350 | **35.010** | Praileaitz 1 |
| Castilla y León | 30.300 | **32.510** | Cueva de la Zarzamora |

Aquí **no falta nada**: tenemos un yacimiento *más* antiguo que el GT. La razón es que el
GT aplica un criterio **"solo paleolítico"** y excluye los yacimientos epipaleolíticos
(Jou´l Llobu lo es). Es decir: **el GT no está mal; los dos lados no comparten definición
de "paleolítico"**.

Y un efecto colateral medido: forzar el filtro de periodo **empeora** el resultado
(6/16 exactas con filtro frente a **11/16 sin filtro**), porque nuestra normalización de
`periodo` es demasiado agresiva. La ruta **no fuerza** ese filtro.

---

## 6. Mejora progresiva (de nada a correcto)

| Corrida | Fecha | Qué cambió | faithfulness | context_precision | context_recall | answer_correctness | Recuperación top-1 en Jaén |
|---|---|---|---|---|---|---|---|
| smoke | 23-09 | línea base | 0/18 | 0/18 | 0/18 | 0/18 | — |
| smoke con grafo (job 10105423, 5 h 31, 6 filas) | 29-09 | grafo ON (`16.357 triples`) | 3/6 · 0,889 | 3/6 · 0,081 | **6/6 · 0,389** | 2/6 · 0,190 | "no hay información suficiente" |
| micro fix coordenadas | 30-09 13:10 | `_ctx_text` + `RETURN_PROPS` | 1,0 · 0,857 | 0 | 0 | 0,103 | vector: "no hay coordenadas" · mq: calcula con las coords inyectadas y responde **Cerro de la Horca** |
| micro + variante ES | 30-09 15:55 | variante ES en multiquery | 1,0 / 0,136 | 0 / 0 | 0 / 0 | 0,103 / 0,094 | mismo resultado: **Eras no entra** sin filtro |
| validación determinista | 30-09 | ruta de filtro + hit@k | — | — | — | — | **Eras del Alcázar 45,9 km** · Trikuaizti 2 12.015 BP |
| **eval viz end-to-end** (job 10180170, 01-10, 3 h 41) | ruta de filtro sobre `IdearqMiniLM_800_50_v2`, 8 filas, solo `faithfulness` | **0,732** (mín 0,080) | — (retirada) | — (retirada) | — (no ejecutada) | las 5 geo por `filter:*`, **hit@k 5/5** — pero la respuesta de Jaén sigue siendo **"Albalate"**, no Eras (§7.3) |

Lectura de la progresión: **el graph context ya dio context_recall 6/6 el 29-09**; el fix
de coordenadas hizo que el modelo **usara** las coordenadas (visible en la respuesta del
micro: calcula distancias euclidianas); el salto de fondo llega con la **ruta de filtro**,
que es lo que trae al yacimiento correcto — y el job 10180170 lo confirma de extremo a
extremo en el nodo de cálculo: **hit@k = 1.0 en las 5**. Lo que queda flojo no es el
retrieval sino el anclaje de la respuesta (`faithfulness` 0,080 en la de Jaén).

---

## 7. La evaluación en viz

**Script de lanzamiento** (el que corre en CESGA):

```
/mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/viz-geo-check.sbatch
```

```bash
cd /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag
sbatch viz-geo-check.sbatch
```

Cabecera Slurm:

```bash
#SBATCH -J viz-geo-check
#SBATCH -p viz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
```

Qué hace y qué **no** hace:

- **No** arranca Weaviate ni Ollama: espera al job persistente
  `rag-services.sbatch` (10103455, nodo `ilk-20` → servicios en `c201-20`). Lanzar un
  segundo Weaviate sobre el mismo `vectordb` lo corrompe.
- Ejecuta `jupyter nbconvert --execute` sobre
  `notebooks/evaluation/viz-geo-check.ipynb` con
  `--ExecutePreprocessor.timeout=25200` (7 h, por debajo de las 8 h del job).
- Sale por `logs/viz-geo-check-<jobid>.out` y el notebook ejecutado en
  `logs/viz-geo-check-<jobid>.ipynb`.

Configuración del run:

| Ajuste | Valor | Por qué |
|---|---|---|
| Colección | `all-MiniLM-L6-v2` → `IdearqMiniLM_800_50_v2` | es la única con el enriquecimiento GeoJSON |
| Preguntas | las 5 geo + 3 bibliográficas de control, **seleccionadas por texto de pregunta** contra `GOLD_LABELS` | comprobar que la puerta no secuestra preguntas que no son suyas |
| Estrategia | `vector` | A/B directo contra las micro-pruebas |
| K | 8 | deja sitio a los 6 yacimientos del superlativo de Jaén |
| Métricas LLM | solo `faithfulness` | `answer_correctness` reventó con `TimeoutError` a 3600 s; con 2 métricas × 8 filas ≈ 8,7 h > límite de 8 h |
| Grafo | `GRAPH_CONTEXT = True` | el smoke lo llevaba; no queremos perder esa comparación |

**Selección de preguntas (importante):** no se puede elegir por número. `EVAL_ITEMS` es
**0-based** (`id == posición`) y `load_items()` puede servir el dataset de Langfuse en otro
orden. Por eso la selección va contra las preguntas de `GOLD_LABELS`: `geo = 5`,
`biblio = 3`, `total = 8`.

**Jobs (cronología):**

| Job | Estado | Duración | Qué pasó |
|---|---|---|---|
| 10173916 | FAILED | 00:00:19 | `'outputs' is a required property` — falló antes de ejecutar nada (problema #4) |
| 10173945 | FAILED | 00:11:19 | `too many values to unpack (expected 5)` en la celda `In[9]` (problema #5) |
| 10174350 | COMPLETED | 02:31:58 | Primer resultado utilizable, **pero solo 3 de 8 preguntas**: `ITEMS[:SMOKE_N]` estaba *después* de la selección y truncaba |
| 10179668 | FAILED | 00:07:08 | `IndentationError`: al quitar `ITEMS[:SMOKE_N]` quedó un `if SMOKE_ONLY:` sin bloque |
| **10180170** | **COMPLETED** | **03:41:25** | **El bueno** — exit `0:0`, 0 trazas, nodo `t4-4` |

> Lección: `--time=08:00:00` es el **techo** que concede el planificador, no tiempo que se
> tenga que consumir. Los tres primeros murieron por errores de código, no por tiempo — se
> detectaron leyendo el log en segundos. Verificar `nbformat.validate` **no basta**: valida
> el esquema, no la sintaxis; hace falta `compile()` celda a celda.

### 7.1 Resultados (job 10180170, 01-10-2026 16:28)

CSV: `data/results/eval_ablation_retrieval_viz_geocheck_20261001_162847.csv` ·
nota ejecutada: `logs/viz-geo-check-10180170.ipynb`

| # | Pregunta | Ruta | hit@k | MRR | faithfulness |
|---|---|---|---|---|---|
| 0 | Yacimientos chalcolíticos más lejos de Jaén | `filter:superlative_space` | 1.0 | 1.0 | 0.080 |
| 1 | Dataciones compatibles con el Neolítico (CV/Gal/Mur) | `filter:lookup_by_date` | 1.0 | 1.0 | 0.571 |
| 2 | Fecha más antigua de un yacimiento funerario megalítico | `filter:superlative_date` | 1.0 | 1.0 | 0.917 |
| 3 | Cronologías funerarias del Mesolítico por región | `filter:group_by_region` | 1.0 | 0.5 | 0.735 |
| 4 | Dataciones paleolíticas más antiguas por comunidad | `filter:group_by_region` | 1.0 | 0.5 | 0.804 |
| 5 | Año de excavación de los Siret en La Bastida *(control)* | `topk` | — | — | 1.000 |
| 6 | Excavaciones de urgencia en Almería 2001 *(control)* | `topk` | — | — | 0.833 |
| 7 | Datación -1377±23 Valencina como más reciente | `filter:lookup_by_date` | — | — | 0.917 |

**Medias:** `hit@k = 1.000` (5/5) · `MRR = 0.800` · `faithfulness = 0.732` (min 0.080).

**Veredicto contra el criterio de aceptación:**

- ✅ **Las 5 preguntas geo salen todas por `filter:*`** — ninguna se coló por top-k.
- ✅ **`hit@k = 1.0` en las 5** — la ruta recupera el documento con la respuesta la primera vez.
- ✅ **Los 2 controles bibliográficos reales (5 y 6) se quedan en `topk`** — la puerta no secuestra.
- ⚠️ **MRR 0.8**: en los dos `group_by_region` (#3, #4) el GT cae en **rank 2**, no en 1.
  Recuperado pero no primero.
- ⚠️ **`faithfulness` en #0 = 0.080**: la respuesta del chalcolítico de Jaén es sobre todo
  *cadena de razonamiento* (distancias, top-N) y el juez marca las afirmaciones como no
  sustentadas por el contexto. **Es el eslabón débil**: retrieval perfecto, respuesta poco
  anclada.
- ⚠️ **Fallo de diseño en el control**: el #7 (Valencina) se eligió como "bibliográfico"
  porque no está en `GOLD_LABELS`, pero su campo de fuente **es `geojson`** → la ruta lo
  mandó a `filter:lookup_by_date`. No es un secuestro, pero el control solo tiene **2** filas
  reales, no 3.

*`hit@k`/`MRR` están vacíos en 5, 6 y 7 a propósito: solo se calculan sobre preguntas con
gold label (`annotate_retrieval_metrics` los rellena por texto de pregunta, no por índice).

### 7.2 Lectura de `faithfulness` (y por qué no se optimiza sola)

**Qué mide exactamente:** la proporción de afirmaciones de la respuesta que el juez encuentra
respaldadas **literalmente** por el contexto. **No** mide si la respuesta es cierta — eso es
`answer_correctness`, que aquí no corrió (fue el que reventó a 3600 s).

**#0 → 0,080 (el más bajo, y con razón).** La respuesta ocupa ~2.000 caracteres de cadena de
cálculo. En su propia letra dice:

> *"como no se proporciona explícitamente la ubicación de Jaén, se utiliza el centro de la
> provincia… se estima aproximadamente **Latitud 37.8500 / Longitud -3.6500**"*

El modelo **inventó las coordenadas de la ciudad de Jaén** (no están en el contexto) y a
partir de ahí calcula 6 distancias con su propia conversión de grados→km. Casi nada de lo que
escribe —coordenadas ajenas, factor 111,32 km/grado, raíces de sumas de cuadrados— aparece en
ningún documento, así que el juez lo desmonta frase a frase.

Y el daño es peor que la nota: **la respuesta es incorrecta**. Dice *Albalate, 49,88 km*;
el ground truth dice *Eras del Alcázar, ~50 km*. La ruta trajo los 6 yacimientos correctos
(`hit@k = 1.0`, rank 1) — falló la **generación**, no el retrieval. El juez tenía razón.

**#1 → 0,571.** Los valores de C14 sí están copiados del contexto (de ahí el 0,57 y no el 0),
pero las conversiones de rangos ("Neolítico temprano entre 10.000 y 5.000 a.C.") y la
conclusión no lo están. Además el GT dice *"No, más compatibles con Paleolítico/Mesolítico"*
y la respuesta dice *"Sí"*.

**Efecto abstinencia (lo que hace ingobernable el número por sí solo):** las filas 5, 6 y 7
responden literalmente *"No hay información suficiente en el contexto"* y sacan
**1,000 / 0,833 / 0,917**. Cuanto menos dice la respuesta, más sube `faithfulness`. Optimizar
este número aisladamente significa empujar al modelo a no responder — y en el #0 ya vimos que
el problema real es al revés: respondió de más, con datos que no tenía.

**Diagnóstico y palancas** (en orden de rendimiento):

| Palanca | Qué resuelve | Coste |
|---|---|---|
| 1. Inyectar el punto de referencia (coords de la ciudad de Jaén) en el contexto de `superlative_space` | el modelo deja de inventar y calcula sobre datos reales | bajo — mismo patrón que `_ctx_text` |
| 2. Calcular la distancia en el código y meter el ranking ordenado en el contexto | la respuesta pasa de *razonar* a *copiar* → sube `faithfulness` **y** acierta al Eras | medio — `retrieve_filtered` ya filtra para enrutar; hay que reutilizar ese cálculo |
| 3. Prompt: "usa solo valores del contexto, indica de dónde sale cada número" | reduce la sangría desde fuera | bajo |
| 4. Rehabilitar `answer_correctness` en subconjunto | detecta justo este fallo (Albalate vs Eras), que `faithfulness` no ve | alto — fue el que reventó |

La 2 es la buena: ataca la causa (el modelo hace aritmética que el contexto no puede
respaldar) y además sube la tasa de respuestas correctas. La 4 es el contrapeso obligatorio,
porque `faithfulness` solo dice *"esto no está en el contexto"*, no *"esto es falso"*.

### 7.3 Aclaración: "Eras del Alcázar" nunca lo respondió el modelo

Para que no quede ambigüedad en la historia:

| Qué se validó | Cuándo | Resultado | ¿Qué es? |
|---|---|---|---|
| La ruta recupera el Eras **primero** | 30-09 (validación determinista) | top-1 = Eras, 45,9 km | **retrieval** |
| El micro con coords inyectadas | 30-09 13:10 | respondió **"Cerro de la Horca"** | **generación** |
| Viz end-to-end (job 10180170) | 01-10 | `hit@k = 1.0`, `MRR = 1.0` | **retrieval** |
| Viz end-to-end (job 10180170) | 01-10 | respondió **"Albalate, 49,88 km"** | **generación** |

Nunca hubo una respuesta generada de "Eras del Alcázar": lo que siempre funcionó —y sigue
funcionando— es que el documento llegue **primero** al contexto. Las tablas anteriores usaban
"Respuesta en Jaén" para la columna de recuperación, y esa ambigüedad es exactamente lo que
hace falta corregir. No es una regresión: es una promesa mal formulada.

### 7.4 Análisis: por qué cada respuesta salió mal

Las ocho filas de la tabla son las del job 10180170
(`data/results/eval_ablation_retrieval_viz_geocheck_20261001_162847.csv`).
Para cada una se ha comprobado lo mismo en este orden: si el documento bueno aparece en
`filenames`, si el dato cabía en `context_chars`, y si la respuesta lo usa.

Aviso de método: la nota ejecutada no guarda el texto del contexto, solo su longitud
(`context_chars`). En la fila 5 el diagnóstico se apoya, por tanto, en comparar runs.

Causas: **A** = el documento no llegó · **B** = llegó pero el dato no estaba en el texto que
ve el modelo · **C** = el dato estaba y el modelo razona mal · **D** = la pregunta o la
etiqueta no se pueden cumplir con lo que hay.

| # | Pregunta (corta) | Veredicto | Causa | Evidencia | Arreglo propuesto |
|---|---|---|---|---|---|
| 0 | Chalcolíticos más lejos de Jaén | ❌ | **B** | `hit_at_k=1.0` y el primer `yacimiento_id` es 10013 (Eras del Alcázar), pero la respuesta dice que «no se proporciona explícitamente la ubicación de Jaén»: `_ctx_text` solo escribe `lat` y `lon` de cada yacimiento, y la ciudad y la distancia que la ruta ya calculó no entran en el prompt (`context_chars=9570`). | Escribir en `_ctx_text` la ciudad de referencia y la distancia ya calculada, para que el modelo copie en vez de volver a calcular. |
| 1 | ¿33.900, 31.690 y 12.030 son del Neolítico? | ❌ | **C** | Los tres registros llegaron (`filenames` = 3× `c14_v2.geojson`, `yacimiento_ids` = 14405;10288;14004, `hit_at_k=1.0`, `context_chars=3181`) y la propia respuesta escribe «33.900±60 → límite superior del Epipaleolítico» y aun así concluye «Sí». | Pedir en el prompt que la conclusión juzgue solo la fecha citada en la pregunta; además, el enunciado dice ±60/±50/±0 y el GeoJSON tiene ±310/±240/±40. |
| 2 | Fecha más antigua de un megalito | ✅ | — | Los 10 documentos son `c14_v2.geojson`, `hit_at_k=1.0` y `reciprocal_rank=1.0`, y la respuesta coincide con el GT (Trikuaizti 2, 12.015±145, un valor que aparece 4 veces en el GeoJSON). | No tocar nada: es el caso en el que la ruta y el modelo funcionan. |
| 3 | Cronologías del Mesolítico (catalán) | ⚠️ | **A** | La fuente declarada en `eval_questions.py` (pregunta 17) son los PDF `1168_The mesolithic of atlantic coastal spain_2022.pdf` y `919_JuanGibaja_2015.pdf`, pero en `filenames` solo hay 15 `c14_v2.geojson`, porque el filtro solo busca en `*.geojson` (`context_chars=10382`). | Dejar que la ruta traiga también PDF cuando la fuente declarada no es el GeoJSON; la etiqueta 18 está en `needs_review: True` y hay que revisarla, y el prompt exige responder en el idioma de la pregunta. |
| 4 | Paleolítico por comunidad (GT: 16) | ❌ | **B** | El filtro trajo 40 sitios (`n_docs=40`) pero el texto se arma con `docs[:K]` y `K=8`, así que solo 8 entraron (`context_chars=16428`), solo 7 de los 16 sitios dorados aparecen en esa lista, y aun así la respuesta pone «Cataluña — No hay yacimientos paleolíticos mencionados en el contexto» teniendo Cova Gran de Santa Linya (38.640) dentro de esos 8. | Subir `K` o repartir el listado en varias llamadas; ojo, `region` guarda la provincia y no la comunidad (docstring de `parse_region`), así que el paso final lo hace el modelo. |
| 5 | Año en que los Siret excavan La Bastida | ❌ | **B** | El PDF bueno `917_LaBastida_2015.pdf` aparece una sola vez entre 8 ficheros distintos y el modelo se abstiene con `context_chars=12392`; en el smoke del 29-09, con la misma estrategia `vector`, los 10 documentos eran trozos de ese mismo PDF (17.936 caracteres) y la respuesta sí citaba 1886, y el notebook no guarda el fragmento para compararlo directamente. | Evitar que el top-k gaste todos los huecos en ficheros distintos: si un PDF puntúa alto, llevar más trozos suyos. |
| 6 | Excavaciones de urgencia en Almería, 2001 | ❌ | **D** (además A) | La etiqueta 3 estaba en `needs_review: True` con `verified=False` porque ninguna de las calles del GT («Avenida Pablo Iglesias», «Calle La Central», «Calle Castillejo») aparece en **ninguno de los 525 PDF** (0 coincidencias con `pdftotext`), y `483_2001.pdf` no existe: solo `483_2001_error.pdf`, un índice. Ese fichero tampoco entró en `filenames`. | ✅ **Hecho (05-10)**: pregunta y GT **reescritos** con la respuesta literal del índice (`483_2001_error.pdf`, líneas 151-152). `verified: True`, `needs_review: False`. Detalle en §3.4. |
| 7 | ¿-1377±23 es la datación más reciente? | ❌ | **D** (además A) | `route=filter:lookup_by_date` pero `filenames` son 8 PDFs y `yacimiento_ids` está vacío: el filtro buscaba `bp_max=1377` y el único yacimiento que contiene «1377±23» (id 14371) tiene `bp_max=4676`, así que devolvió 0 filas y `run_rag` siguió con el top-k sin cambiar la ruta (`context_chars=15091`). | ✅ **Hecho (05-10)**: §4.1 (0 filas → visible, sin fallback) y §4.2 (filtro por `bp_values` → ahora sí encuentra el id 14371). ⚠️ **Queda**: la respuesta completa («175±20») solo está en `1135_Poblado_2021.pdf`, y esa ruta **no mira PDFs** — ver §7.5. |

**Patrón general.** El fallo ya no está en conseguir el documento: en las cinco filas con
métrica, `hit_at_k = 1.0`, y los ficheros buenos aparecen en `filenames` casi siempre (en la
fila 4 solo 7 de los 16 sitios dorados, y aun así cuenta como acierto). Lo que falla pasa
después. Primero, el texto que ve el modelo llega recortado: entran 8 documentos por llamada
(`K = 8`) y en la fila 4 había 40; en la fila 5 entró un solo trozo de un PDF de 561.758
caracteres; y en la fila 0 la ruta ya calculó la distancia y la ciudad, pero `_ctx_text` no
las escribe. Segundo, el modelo razona mal sobre datos que sí tiene (fila 1) o se contradice
mientras responde (fila 4). Y tercero, tres de las etiquetas implicadas (`#3`, `#4`, `#18`)
estaban en `needs_review`: un humano tiene que decidir si valen antes de penalizar la
respuesta — **`#3` ya se reescribió el 05-10, quedan 11 de 32 en revisión.**
En corto, el problema se ha movido de «no llega el documento» a «llega incompleto, o llega
bien y no se usa».

### 7.5 Estado de los arreglos (05-10-2026)

| # | Arreglo | Estado | Efecto esperado en la próxima evaluación |
|---|---|---|---|
| 1 | Filtro con 0 filas → visible (`route → VACÍO`, `n_docs=0`, contexto = aviso) | ✅ | la fila 7 dejará de parecer un fallo del modelo |
| 2 | Ciudad + distancia escritas en `_ctx_text` | ❌ **NO hecho** (decisión del usuario: sin el arreglo 2) | la **fila 0 seguirá fallando**: el modelo se seguirá calculando las distancias |
| 3 | `lookup_by_date` filtra por `bp_values` | ✅ | Valencina encuentra el id 14371; la #5 no se rompe |
| 4 | Etiquetas | ⚠️ parcial | `#3` reescrita; quedan `#4` (¿fuente → `1135_Poblado_2021.pdf`?) y `#18` (confirmación humana) |
| D | Propiedad `bp_values` + `dataciones_c_14` | ✅ | 4478 objetos, 0 vectores cambiados (`docs/dataciones-property.md`) |

**Lo que el sistema todavía NO puede hacer (documentarlo en la evaluación completa):**

1. **Responder la de Valencina completa.** La pregunta pide juzgar si `-1377±23` es la más
   reciente, y la respuesta buena (**`175±20`**) **solo está en `1135_Poblado_2021.pdf`**.
   La ruta geo solo mira el GeoJSON, y **por parecido tampoco llega**: en la fila 7 los 8
   PDF que entraron por fallback **no incluían** `1135_Poblado_2021.pdf`. Con el arreglo 3
   el sistema ya encuentra el yacimiento (id 14371) y sus dataciones… pero **el 175±20 no
   está en el GeoJSON**, así que seguirá sin poder dar la respuesta completa.
   *Pendiente de decisión:* traer también PDF cuando la pregunta pide verificar una fecha.
2. **Contestar preguntas cuya fuente son PDF con la ruta geo**: pasa en las filas 3 y 7 —
   el filtro solo busca `*.geojson`.
3. **Los agrupados con más de `K=8` regiones** (fila 4: 40 sitios recuperados, 8 al modelo).

### 7.6 Re-lanzar el smoke antes de la evaluación completa (06-10-2026)

Antes de tirar la evaluación entera hay que volver a pasar el smoke con los arreglos
nuevos y mirar las respuestas y las métricas.

**Qué es distinto respecto al job 10180170:**

| Cambio | Qué se espera ver en el CSV |
|---|---|
| Arreglo 1 (§4.1) | la fila de Valencina ya no dice `filter:lookup_by_date` como si hubiera funcionado: pasa a `route = filter:lookup_by_date → VACÍO, sin fallback topk`, `n_docs=0` |
| Arreglo 3 (§4.2) | el filtro por fechas encuentra sitios en vez de devolver 0 filas |
| Etiqueta `#3` reescrita (§3.4) | otra pregunta en la lista (Villaricos) |
| Celda 7 fijada a 8 preguntas | mismas 8 que el job 10180170 → las filas se comparan una a una |

**Comprobaciones hechas antes de lanzar:**

- `nbformat.validate` + `compile()` **31/31** en `viz-geo-check.ipynb`
  (el job 10173916 murió justo por una celda mal formada: `'outputs' is a required property`).
- `bash -n` en los dos `.sbatch`.

**Problema detectado a tiempo y corregido.** La celda 7 seleccionaba «geo + 3
bibliográficas» mirando qué labels tienen `question`. **Ahora los 32 labels la tienen**, así
que el selector habría cogido **32 preguntas** en vez de 8. Con el juez de RAGAS a
~10 min/llamada (medido en el job 10180170) eso no cabe en las **8 h** que permite la
partición `viz`. La celda 7 ahora **fija explícitamente las mismas 8 preguntas** del job
10180170 por subcadena distintiva (nunca por índice numérico) y **falla en seco** si no
las encuentra todas:

```python
VIZ_SMOKE_PATTERNS = [ ... 8 subcadenas ... ]     # 8/8 comprobado en local
```

**Comprobación de GPU (06-10).** No hay ninguna A100 libre: todos los nodos `a100-*`
tienen `gres/gpu=2` ya asignado o están en `drain`. En `viz` sí hay 3 nodos T4 idle
(`t4-8/9/10`), así que **se lanza en `viz`**.

**Jobs lanzados:**

| Job | Qué es | Estado |
|---|---|---|
| `10367977` | smoke geo: las mismas 8 preguntas con los arreglos | **COMPLETED** (exit 0, **4:59**, 06-10 10:49 → 15:59) |
| `10368130` | smoke A/B: grafo vs Weaviate con las 5 preguntas del grafo (`docs/smoke-ab-graph-weaviate.md`) | COMPLETED (exit 0, 15 min, 06-10 11:10) |
| `10368762` | re-lanzado del A/B con el control de la sonda arreglado | **COMPLETED** (exit 0, 7 min, 06-10 11:41) — control **OK**, mismos números |

**Resultado del A/B (cerrado, `docs/smoke-ab-graph-weaviate.md` §5):** el grafo contesta
**5/5** y Weaviate **0/5**; en 4 de las 5 preguntas la respuesta no aparece en ningún PDF.
Es lo que justifica la capa de grafo.

**Métricas que va a producir el smoke geo:** `faithfulness` (juez `Qwen3-14B` vía Ollama)
+ `hit_at_k` y `reciprocal_rank` (deterministas, sin LLM). `ContextPrecision` y
`ContextRecall` siguen **desactivados**: cada uno cuesta una llamada del juez por contexto
recuperado y ya provocaron `NaN` en corridas anteriores.

**Contra qué se comparan:** `data/results/eval_ablation_retrieval_viz_geocheck_20261001_162847.csv`
(8 filas, job 10180170). El criterio es: mismas preguntas, mismo `strategy=vector`,
mismo `K=8`, y ahora sí un `route` que no miente.

### 7.7 Resultado del smoke geo (job 10367977, 06-10 15:59)

**COMPLETED, exit 0, 4 h 59 min** (10:49 → 15:59, límite 8 h). Salida:

- `logs/viz-geo-check-10367977.ipynb` — 31 celdas, **sin excepciones de código**.
- `data/results/eval_ablation_retrieval_viz_geocheck_20261006_155906.csv` — 8 filas.
- `VIZ CHECK: 8/8 preguntas fijadas` — el pin de preguntas funcionó.

| # | Ruta | `n_docs` | `hit@k` | MRR | `faithfulness` |
|---|---|---|---|---|---|
| 1 Siret (topk) | `topk` | 8 | 1.0 | 1.0 | 1.000 |
| 2 Chalcolithic más lejos | `filter:superlative_space` | 6 | 1.0 | 1.0 | 0.071 |
| 3 Villaricos | `topk` | 8 | **0.0** | 0.0 | 0.857 |
| 4 Valencina -1377±23 | `filter:lookup_by_date` | 3 | 1.0 | 0.5 | 0.115 |
| 5 Dataciones Neolítico | `filter:lookup_by_date` | 4 | 1.0 | 1.0 | 0.526 |
| 6 Fecha megalítico más antigua | `filter:superlative_date` | 10 | 1.0 | 1.0 | **NaN** |
| 7 Cronologías mesolítico (ca) | `filter:group_by_region` | 15 | 1.0 | 0.5 | **NaN** |
| 8 Dataciones paleolíticos | `filter:group_by_region` | 40 | 1.0 | 0.5 | 0.719 |

**Medias:** `hit@k` **0.875** (7/8) · MRR **0.688** · `faithfulness` **0.548** (sobre 6 filas).

**Las rutas geo funcionan**: las 6 variantes de `route` devolvieron documentos
(`n_docs` entre 3 y 40), ninguna se quedó en 0 — que era justo el fallo original
(`n_docs = 0` en la fila 7 del job 10180170).

**Comparación con el baseline — con tres advertencias:**

1. **7 de 8 preguntas son idénticas.** La 3 (Villaricos) se reescribió el 05-10 con otro
   gold, así que su `hit@k = 0` **no es comparable** con el baseline: allí esa fila era
   otra pregunta.
2. **El baseline solo guardó `hit_at_k` en 5 de 8 filas** (3 con `NULL`), así que su media
   de 0.625 no es comparable con la 0.875 de ahora.
3. **`faithfulness` bajó de 0.732 a 0.548**, pero con 2 filas en `NaN` y con un juez
   ruidoso. No es una regresión demostrable con 8 muestras.

### 7.8 El coste real: el juez RAGAS se come las 5 horas

La celda 10 imprimió `Exception raised in Job[5]: TimeoutError()` y `Job[6]` — las filas
6 y 7 se quedaron sin `faithfulness`. Ese `RunConfig(timeout=3600)` son **1 hora por
intento fallido: 2 horas muertas de las 4:59**.

El motivo está medido: el juez corre en **CPU** (Ollama con `vram 0 GB`, nodo de solo
4 cores) y una llamada tarda **~22 min**. El propio notebook ya lo advertía:

> *judge prompt eval runs at 25 tok/s on CPU … one judge call costs 8-13 min. Two
> metrics x 8 rows does not fit in viz's 8 h.*

Desglose aproximado: 6 llamadas OK (6 × 22 min ≈ 2,2 h) + 2 timeouts (2 h) + generación
≈ las 4:59 del job. **Casi el 100 % del coste es el juez.** Las métricas de retrieval
(`hit@k`, MRR) se calculan en segundos, sin LLM.

**Decisión pendiente antes de la evaluación completa**, con números en la mano:

| Opción | Efecto |
|---|---|
| Quitar `ragas_metrics` del smoke | 8 preguntas en minutos; te quedas solo con `hit@k`/MRR |
| Subir `RunConfig(timeout)` | Arriesga chocar con las 8 h de `viz` |
| Nodo con más cores (o GPU) | El juez baja de 22 min a ~2 min, cabe todo |
| Aceptar `NaN` en las filas lentas | Ya es lo que pasó: 6/8 con dato |

### 7.9 Por qué Villaricos da 0, qué pasó con Valencina y de dónde salen los NaN

**Villaricos — `hit@k = 0` porque el PDF oro ni se recupera.**
La etiqueta #3 declara un solo oro: `filename = 483_2001_error.pdf` (sin `yacimiento_id`,
porque es un índice OCR del Anuario). El métrico compara nombres de fichero normalizados,
y los 8 documentos recuperados fueron:

```
742_2010_Texto del artículo-189-1-10-20100719.pdf     819_2013.pdf
1184_Necrópolis prehistórica … Alcaide_2024.pdf       577_yacimiento_Calcolitico_Santa_Lucia_2005.pdf
819_2013.pdf                                          935_Acequion_2016.pdf
655_2008.pdf                                          1184_Necrópolis … Alcaide_2024.pdf
```

Ninguno es `483_2001_error.pdf` → posición `None` → `hit_at_k = 0.0`. Dos detalles más:

1. **Las 8 casillas contienen solo 6 ficheros únicos** — `819_2013` y `1184_…Alcaide`
   aparecen dos veces. Los duplicados consumen posiciones del top-8.
2. La respuesta del modelo lo confirma: dice textualmente *"no se menciona ninguna
   excavación en Villaricos"* — el documento bueno no llegó al contexto.
3. *Pendiente de comprobar* (los servicios estaban caídos al hacer la pregunta): si
   `483_2001_error.pdf` **está indexado** en `IdearqMiniLM_800_50_v2` o si está fuera por
   OCR ilegible. Si está indexado es un fallo de ranking; si no, de indexación.

**Valencina — el `hit@k` está bien, la respuesta no, y la propia etiqueta está en revisión.**

- `hit_at_k = 1.0`, `reciprocal_rank = 0.5` → el oro (`yacimiento_id` 14234 y 14371) sí
  salió, en segunda posición. La ruta `filter:lookup_by_date` funcionó (3 documentos).
- **El modelo acierta la dirección y falla el contenido**: dice que `-1377±23` *"no es la
  más reciente"* (correcto según el oro), pero **nunca da el valor 175 ± 20**.
- **No podía darlo**: en `c14_v2.geojson` las cadenas `175` y `Ladera Sur` **no existen**
  (`Ladera Sur`: 0 apariciones). En las dos filas del oro, la datación más reciente que hay
  es justo `MAMS42069 1377±23 B.P.` (la otra fila llega hasta 4250±31). Es decir, **dentro
  del contexto recuperado, -1377±23 sí es la más reciente**, y el modelo afirma lo
  contrario.
- La respuesta se contradice además al llamar *"más reciente"* a `MAMS42070 3831±30`:
  en escala B.P. cuanto más grande, más antiguo.
- De ahí `faithfulness = 0.115`: el juez penaliza una respuesta que el contexto no sostiene.
- **La etiqueta #4 ya lo sabe**: `needs_review = True`, `verified = False`, y su nota dice
  que la frase del oro está en `1135_Poblado_2021.pdf`, que no es la fuente declarada
  (`geojson`), y que *«human must decide whether to add it»*. Es una de las 11 pendientes.

**Los dos `NaN` — sí, son del juez, y es un timeout, no un error de código.**
La celda 10 imprimió exactamente:

```
Exception raised in Job[5]: TimeoutError()
Exception raised in Job[6]: TimeoutError()
```

RAGAS evaluó las 8 muestras en orden con `max_workers=1`, así que `Job[5]` y `Job[6]` son
las muestras 6 y 7 — **las mismísimas filas sin `faithfulness`** (megalítico y mesolítico).
El plazo es `RunConfig(timeout=3600)` = 1 h por intento. El `try/except` de la celda lo
capturó y siguió, por eso el CSV se escribió con 8 filas y dos celdas vacías.

---

## 8. Por qué el grafo está deduplicado

Fichero: `data/biblio-graph/idearq-graph-instances-dedup-v2.nt` — **16.357 triples**
N-Triples, los mismos que reportó el smoke.

El grafo se construye desde los registros bibliográficos del proyecto IDEArq, donde una
**misma publicación o instancia aparece repetida** (distintas filas de la fuente, ids
distintos, o el mismo yacimiento publicado en varios estudios). Sin deduplicar, esas
repeticiones se conectan entre sí y:

1. **Inflan los resultados de SPARQL**: los mismos 3-4 nodos aparecen varias veces en el
   top-N, desplazando relaciones distintas.
2. **Distorsionan el RRF** al fusionar con los documentos: los mismos RDF docs entran
   varias veces en el top-10 de contexto.
3. **Multiplican el peso** de una fuente por su número de duplicados, sesgando el
   ranking de citations hacia lo más repetido.

Por eso hay dos ficheros: `idearq-graph-instances.nt` (el crudo) y
`idearq-graph-instances-dedup-v2.nt` (el que usa el smoke), junto con sus `.ttl`.

---

## 9. Qué falta

- **Backfill de `IdearqGTE_800_50_v2`**: la colección que usa la evaluación sigue con
  las columnas vacías. Es el mismo script (`backfill_geo_properties`), 5 minutos. **La
  condición que fijó el usuario ya se cumple**: la ruta responde de extremo a extremo en
  MiniLM (job 10180170, hit@k 5/5). Queda su visto bueno para lanzarlo. *Al hacerlo hay
  que añadir también `bp_values` y `dataciones_c_14`* (`scripts/backfill_c14_dates.py`
  con `--apply --ensure-schema`), o la ruta `lookup_by_date` caerá al retroceso `bp_max`.
- **Pase humano sobre las 11 etiquetas `needs_review`** (`#4`, `#15`, `#30` primero:
  son las que probablemente apuntan al documento equivocado). `#3` ya se reescribió el
  05-10. Ver detalle en §3.4.
- **Decidir si la ruta geo debe traer también PDF** — es lo que falta para la fila 7
  (Valencina) y para la 3 (mesolítico, cuyas fuentes son dos PDF). Hoy el filtro solo
  busca `*.geojson`. Ver §7.5.
- **Regenerar los CSV re-ejecutando las evaluaciones**: las filas bibliográficas del job
  10180170 siguen con `hit_at_k = NaN` guardado; el métrico ya las calcula, pero el número
  se escribe al correr el notebook. *Habrá que repetirlo igualmente: los arreglos del
  05-10 cambian el `route` de la fila 7 y los documentos de la 4 y la 7.*
  **Estado 07-10: los dos smokes están cerrados.**
  - A/B grafo vs Weaviate (jobs 10368130 + relanzado 10368762): **grafo 5/5, Weaviate 0/5**
    → `docs/smoke-ab-graph-weaviate.md` §5.
  - Smoke geo (job 10367977): **8/8 preguntas, `hit@k` 0.875, rutas todas con documentos**,
    pero **2 filas sin `faithfulness` por timeout de 1 h** del juez en CPU → §7.7 y §7.8.
  - **Decisión que falta: qué hacer con RAGAS** en la evaluación completa (tabla de
    opciones en §7.8). Con RAGAS a ~22 min/llamada, 32 preguntas costarían ~12 h solo de
    juez y no caben en las 8 h de `viz`.
- **Etiquetar por comunidad autónoma**: falta el paso `group_by_region` →
  provincia → comunidad (LLM con vocabulario cerrado: 26/26 y 56/56 provincias, 0 valores
  fuera de catálogo).
- **Verbalización por `#5` y `#18`**: sus etiquetas están `needs_review`.
- **Los agrupados y `K = 8`**: la fila 4 trajo 40 sitios y solo 8 llegaron al modelo;
  hace falta subir `K` o repartir el listado. **No decidido aún.**
- **Arreglo 2 (ciudad + distancia en `_ctx_text`): descartado por el usuario** el 05-10.
  Mientras tanto, la fila 0 (Jaén) seguirá fallando: es un fallo conocido y aceptado.
- **Comprobar en un run real** si RAGAS tolera una fila con `retrieved_contexts` vacío
  (arreglo 1); si revienta, el `try/except` de la celda 23 perdería las métricas de la
  estrategia entera.
- **LDA**: sigue pendiente de decisión.
- **Fase A**: 18 combos × 4 métricas LLM ≈ 20 días de cómputo. Con hit@k para retrieval
  y solo 2 métricas de respuesta baja a un orden de magnitud manejable.

## 10. Ficheros tocados

### Hasta el 01-10

| Fichero | Cambio |
|---|---|
| `notebooks/evaluation/rag-evaluacion-hibrida-multiquery.ipynb` | `_ctx_text` + `RETURN_PROPS` (coords) · `RunConfig(timeout=3600, log_tenacity=True)` · variante ES en `MQ_PROMPT` · ruta de filtro (2 celdas) · métricas hit@k/MRR (2 celdas) · `run_rag` enruta |
| `notebooks/evaluation/viz-geo-check.ipynb` | copia de la evaluación en viz (MiniLM, 8 preguntas elegidas **por texto**, solo `faithfulness`) · celda 26 igual que en el maestro |
| `notebooks/indexing/rag-indexacion-mejorado.ipynb` | extractores + `**enrich_geo_props(props)` + `backfill_geo_properties()` (2 celdas de código + 2 de markdown) |
| `data/eval_gold_labels.py` | **32 etiquetas** (31 graduables + 1 ingradable), `verified`/`needs_review`/`ungradable`, `UNLABELLED_QUESTIONS` a 1-based |
| `docs/geo-filter-route.md` | este documento (antes en `odd/tasks/`) |
| `viz-geo-check.sbatch` | **nuevo**: script de CESGA para la partición viz |

### 05-10 y 06-10-2026

| Fichero | Cambio |
|---|---|
| `notebooks/evaluation/rag-evaluacion-hibrida-multiquery.ipynb` | celdas 13 y 15: **arreglo 1** (0 filas → visible) y **arreglo 3** (`bp_values`). Copia `.bak-20261006-093145` |
| `notebooks/evaluation/viz-geo-check.ipynb` | misma réplica de las celdas 13 y 15 (byte-idénticas) |
| `src/config.py` | `INDEX_PROPERTIES_FULL` 23 → **25** (`bp_values`, `dataciones_c_14`) |
| `notebooks/indexing/rag-indexacion-mejorado.ipynb` | mapeo de las 2 propiedades nuevas + normalización nbformat |
| `scripts/backfill_c14_dates.py` | **nuevo**: backfill de dataciones, `--dry-run` por defecto, se niega si `vectorizer != none` |
| `scripts/smoke_graph.py` | **nuevo**: smoke determinista del grafo, 5 consultas SPARQL (PASS 5/5) |
| `src/reindex_gte_e5.py` | `NUMBER_ARRAY` en `DATA_TYPE_MAP` (evita un `KeyError` latente) |
| `data/eval_questions.py` | pregunta `#3` reescrita (Villaricos), `fuente` → `483_2001_error.pdf` |
| `data/eval_gold_labels.py` | etiqueta `#3` reescrita, `verified: True`, `needs_review: False` → **11 de 32** en revisión |
| `docs/dataciones-property.md` | **nuevo**: la propiedad de dataciones (punto D) |
| `docs/smoke-graph.md` | **nuevo**: el smoke del grafo, para la reunión |
| `docs/geo-filter-route.md` | §3.4 · §4.1-§4.3 · §7.4 · §7.5 · §7.6 · §9 · §10 |

### 06-10-2026 (smoke re-lanzado)

| Fichero | Cambio |
|---|---|
| `notebooks/evaluation/viz-geo-check.ipynb` | **celda 7**: selector fijado a las mismas 8 preguntas del job 10180170 (los 32 labels ya tienen `question`, antes elegía 32 y reventaba las 8 h de `viz`). Backup `.bak-20261006-pin` |
| `scripts/viz-geo-check.sbatch` | **recreado**: se había perdido del repo. Replica el lanzamiento del job 10180170 (`-p viz`, 8 h, espera a Weaviate/Ollama, `nbconvert`) |
| `scripts/smoke_weaviate_ab.py` | **nuevo**: smoke A/B, las 5 preguntas del grafo contestadas por Weaviate (`docs/smoke-ab-graph-weaviate.md`) |
| `scripts/smoke_weaviate_ab.py` | **control de la sonda arreglado** (`bm25_probe`: cadena exacta **o** todos los tokens en el mismo trozo) — el `SOSPECHOSO` del primer informe era un falso positivo nuestro, no un fallo de Weaviate |
| `scripts/smoke-weaviate-ab.sbatch` | **nuevo**: lanza el A/B en `viz` (no había A100 libre) |
| `docs/smoke-ab-graph-weaviate.md` | **nuevo**: método, control de la sonda BM25 y cómo leerlo · **§5** con el resultado del job 10368130 · **§7** con el falso positivo del control |
| `$LUSTRE/idearq-rag/pdftext/` | **525 `.txt`** (48 MB) copiados de `/tmp` a Lustre para que el A/B pueda comprobar «este literal no está en ningún PDF» desde el job |





