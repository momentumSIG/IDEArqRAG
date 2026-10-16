# Smoke A/B: grafo RDF vs Weaviate con las MISMAS 5 preguntas

**Propósito:** medir cuánto del conocimiento que tiene el grafo RDF puede reproducir el
índice de Weaviate. Es el experimento que justifica (o no) mantener la capa de grafo.

**Artefactos:** `scripts/smoke_weaviate_ab.py` · `scripts/smoke-weaviate-ab.sbatch` ·
este documento · los informes en `data/results/smoke_ab_graph_weaviate_*.{json,md}`.

**Documento hermano:** `docs/smoke-graph.md` (solo grafo, PASS 5/5).

---

## 1. La idea en una frase

Las **mismas 5 preguntas** del smoke del grafo —elegidas porque su respuesta vive **solo
en el grafo**: ni en los 525 PDF ni en el geojson de dataciones— se contestan por dos
caminos y se comparan:

| | Camino | Cómo responde |
|---|---|---|
| **A** | grafo RDF | consulta SPARQL determinista (misma que `scripts/smoke_graph.py`) |
| **B** | Weaviate | búsqueda BM25 + recuperación híbrida + respuesta del LLM |

Si B no encuentra lo que A encuentra, **el grafo aporta algo que el índice no tiene**.
Si B lo encuentra todo, el grafo es redundante para esas preguntas.

---

## 2. Qué mide exactamente el camino B

Tres pruebas por pregunta, de más barata a más cara:

| Prueba | Qué pregunta | Cómo se responde |
|---|---|---|
| **1. Sonda BM25** | «¿el índice contiene este literal?» | `col.query.bm25(literal, operator=AND, limit=50)` y se comprueba que un trozo devuelto **contenga** el literal: o la cadena exacta, o **todas** sus palabras en ese mismo trozo (BM25 trabaja por tokens, y un trozo trocean el texto). Si AND no lo trae, se reintenta con OR. Detalle y por qué la primera versión estaba mal: **§7**. |
| **2. Recuperación híbrida** | «¿la pregunta trae el documento?» | `hybrid_search(alpha=0.5, k=10)` — la misma ruta que usa la evaluación. Se comprueba si el literal aparece en el contexto resultante. |
| **3. Respuesta del LLM** | «¿el modelo lo dice?» | `Qwen3-4B-Instruct-2507`, T=0.5, con el contexto de la prueba 2. Se comprueba si el literal aparece en la respuesta. |

Y una **prueba de control**: se toma un fragmento de un documento que Weaviate acaba de
devolver y se le pasa por la misma sonda BM25. Si la sonda no encuentra su propio
documento, **la sonda está rota** y la columna «en índice» hay que leerla con cuidado
(el script lo imprime como `probe self-check`).

Se añade una **búsqueda en el corpus** (`--corpus`, 525 `.txt` extraídos) para responder
la pregunta de fondo: *¿ese literal está siquiera en los PDF?* Si no está en ningún PDF,
Weaviate no puede encontrarlo por mucho que le preguntemos — y esa es la prueba
definitiva de que el conocimiento es del grafo.

---

## 3. Por qué `k` de la prueba 1 es 50 y no 8

BM25 devuelve documentos por tokens, no por cadena exacta. Si pedimos 8 y la cadena está
en el 12.º, diríamos «no está» por un error de nuestra sonda. Con 50 y la comprobación
posterior de cadena exacta, el falso negativo cae mucho. Es un límite de sonda, **no** el
`k` del contexto del modelo (que sigue siendo 8).

---

## 4. Cómo se lanza

### Comprobación de GPU (06-10-2026)

| Partición | Estado |
|---|---|
| `ondemand` / `long` / `medium` (A100) | **ninguna GPU libre**: todos los nodos `a100-*` con `gres/gpu=2` ya asignados (`AllocTRES=cpu=...,gres/gpu=2`); el resto `drain`/`drng` |
| `viz` (T4) | **3 nodos idle**: `t4-8`, `t4-9`, `t4-10` (`gpu:t4:1`) |

Como no había A100 libre, se lanzó en **`viz`**, como pidió el usuario.

```bash
sbatch scripts/smoke-weaviate-ab.sbatch      # → job en partición viz, límite 1 h
```

El script **no arranca Weaviate ni Ollama**: espera a los servicios persistentes
(`rag-services.sbatch`, job 10103455 en `ilk-20`). Dos Weaviates sobre el mismo
`vectordb` lo corrompen.

### Salida

| Archivo | Contenido |
|---|---|
| `logs/smoke-ab-<job>.out` | tabla por pregunta + resumen |
| `data/results/smoke_ab_graph_weaviate_<ts>.json` | todo crudo, para análisis |
| `data/results/smoke_ab_graph_weaviate_<ts>.md` | informe legible |

---

## 5. Resultado

Job **10368130**: **COMPLETED**, exit 0, 15 min, 06-10-2026 11:10.
Informe crudo: `data/results/smoke_ab_graph_weaviate_20261006_111049.md`.

En esa primera corrida el control de la sonda salió `SOSPECHOSO`. Era un falso positivo
nuestro, no un fallo de Weaviate (**§7**). Se arregló y se relanzó:

Job **10368762**: **COMPLETED**, exit 0, 7 min, 06-10-2026 11:41.
Informe limpio: `data/results/smoke_ab_graph_weaviate_20261006_114104.md` →
`autochequeo: OK (AND: 50 returned, 50 hold it)` y **los mismos números**: 5/5 · 2/5 · 0/5 · 0/5.

| | respuestas |
|---|---|
| grafo (SPARQL) | **5/5** |
| literal presente en el índice (BM25) | **2/5** |
| literal llega al contexto (híbrido) | **0/5** |
| literal aparece en la respuesta del LLM | **0/5** |

Detalle por pregunta:

| id | grafo | en índice | en contexto | en respuesta |
|---|---|---|---|---|
| Q1 authorship | PASS | ❌ | ❌ | ❌ |
| Q2 publication-year | PASS | ✅ | ❌ | ❌ |
| Q3 journal-ISSN | PASS | ❌ | ❌ | ❌ |
| Q4 DOI | PASS | ❌ | ❌ | ❌ |
| Q5 instance-type | PASS | ✅ | ❌ | ❌ |

El grafo contesta las 5. El índice vectorial **no reproduce ninguna** de las 5 respuestas
de principio a fin: 0/5 en la respuesta del LLM.

### Qué hay en el corpus

Se escanearon los 525 textos de PDF en `$LUSTRE/idearq-rag/pdftext`. Ocurrencias:

| Literal | Veces en el corpus | Ficheros |
|---|---|---|
| `Anne Meulengracht` | **0** | 0 |
| `0066-6742` | **0** | 0 |
| `10.34780/mm.v57i0.1003` | **0** | 0 |
| `http://purl.org/ontology/bibo/Book` | **0** | 0 |
| `Secuencias de cambio social en una región mediterránea` | **0** | 0 |
| `2003` | 5532 | 361 |
| `Book` | 257 (444 sin distinguir mayúsculas) | 110 |

Los ceros significan que ese texto no aparece **en ningún** PDF. Es la prueba
directa de que el conocimiento vive en el grafo.

*Los recuentos de arriba los hemos verificado a mano con `grep -F` sobre los 525
ficheros.* El script `scan_corpus` suma las coincidencias sensibles a mayúsculas con las
insensibles, así que sus números salen inflados (`2003` → 11064, `Book` → 701). Los
ceros sí son fiables: no hay doble cuenta de cero.

### Configuración de la corrida

| Parámetro | Valor |
|---|---|
| grafo | `data/biblio-graph/idearq-graph-instances-dedup-v2.nt` (16357 triples) |
| colección | `IdearqMiniLM_800_50_v2` |
| embedding | `all-MiniLM-L6-v2` |
| híbrido | alpha=0.5, k=10 |
| LLM | `Qwen3-4B-Instruct-2507`, T=0.5 |
| tiempo | 902 s |

### Interpretación

Este es el punto del experimento.

| Hallazgo | Lectura |
|---|---|
| Grafo 5/5, respuesta 0/5 | El índice vectorial no reproduce ninguna respuesta de principio a fin. El grafo aporta lo que el índice no tiene. |
| **4 de 5** preguntas (Q1, Q3, Q4, Q5): la respuesta del grafo no está en ningún PDF | El conocimiento es **solo del grafo**. Ninguna cantidad de búsqueda lo puede producir. Esa es la justificación de mantener la capa de grafo. |
| Q5 sale en las dos filas de arriba | La **respuesta** del grafo es la URI `http://purl.org/ontology/bibo/Book`: **0** en los PDF. Lo que sí aparece es la palabra suelta `Book` (444 veces), y esa no es la respuesta. |
| Q2: `2003` sí está en el corpus | 5532 veces en 361 ficheros, pero es un año cualquiera: el índice no lo distingue de los demás y no llega al contexto de los 10 documentos. |
| Q3 y Q4: ruido en la recuperación | La búsqueda híbrida devolvió trozos de `c14_v2.geojson` en vez del artículo que hacía falta. |

Q5 sale en las dos filas porque son literales distintos. La respuesta del grafo es
`http://purl.org/ontology/bibo/Book`, que no está en ningún PDF (recuento propio de
`purl.org/ontology/bibo`: **0**). Lo que sí hay es la palabra suelta `Book`, y esa no
identifica nada.

Hay que decirlo con claridad: **el conocimiento que sí está en el corpus no llega al
contexto** (0/5). El fallo no es solo «datos que viven en el grafo». También es el
ranking de la recuperación.

---

## 6. Cómo leerlo

| Situación | Lectura |
|---|---|
| grafo 5/5, índice 0/5 | el grafo cubre conocimiento que el índice no tiene: **la capa de grafo se justifica** |
| grafo 5/5, índice 5/5 | para estas 5 preguntas el índice basta: el grafo no aporta nada (aunque puede aportar en otras) |
| `probe self-check: SUSPECT` | la sonda BM25 falla: la columna «en índice» **no es fiable**, corregir antes de concluir |
| `corpus hits: 0` en todos los literales | los PDF no contienen ni una de las respuestas: es imposible que la respuesta venga de la recuperación |

---

## 7. Un falso positivo: el control de la sonda

La primera corrida imprimió `autochequeo de la sonda: SOSPECHOSO`. **Era un bug de
nuestro propio control, no un fallo de Weaviate.**

Evidencia de esa primera corrida:

```
window = 'Reunión Científica Arqueología Albacete Científica Arqueología'
AND: 50 devueltos | frase completa dentro: 0 | LAS 6 palabras: 50/50
```

### Qué hacía mal el control

El control construía una ventana con las primeras 6 palabras de más de 5 caracteres de
un documento que Weaviate acababa de devolver. Las unía con espacios. Después exigía que
esa ventana entera apareciera **como una sola subcadena continua** dentro de un trozo.

Esas 6 palabras no son contiguas en el texto original. Por eso el control veía 0, cuando
BM25 estaba bien: los 50 trozos contenían las 6 palabras.

### Arreglo

Está en `scripts/smoke_weaviate_ab.py`, función `bm25_probe`. Un literal cuenta como
encontrado cuando aparece **tal cual** en el trozo, **o** cuando **todos** sus tokens
aparecen en ese mismo trozo.

Verificado de nuevo:

| Comprobación | Resultado |
|---|---|
| control | `OK` (`AND: 50 returned, 50 hold it`) |
| `2003` | sigue encontrado |
| `Book` | sigue encontrado |
| `Anne Meulengracht` | sigue sin encontrarse |
| `0066-6742` | sigue sin encontrarse |
| `10.34780/mm.v57i0.1003` | sigue sin encontrarse |

El arreglo no cambia ningún veredicto por literal. Solo arregla el control.

### Corrección pendiente de resultados

Job **10368762** re-lanzado para regenerar un informe limpio: **en marcha**
(06-10-2026 11:33). Aquí se rellena su resultado cuando termine.

