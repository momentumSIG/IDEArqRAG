# Pipeline del Grafo Bibliográfico IDEArq

Generación y mantenimiento del grafo RDF en `data/biblio-graph/`.

---

## Qué hay en esta carpeta

| Archivo | Tipo | Propósito |
|---|---|---|
| `referencias_clean.csv` | Datos | 1.174 artículos bibliográficos (input) |
| `idearq-biblio-v6.ttl` | Ontología | BIBO 1.3 + extensiones dc, foaf, org, skos |
| `idearq-graph-instances.nt` | Intermedio | Grafo original (se regenera con el notebook de mapeo) |
| `idearq-graph-instances-dedup-v2.nt` | **Output final** | Grafo limpio (16.361 triples, 2.409 autores) |
| `_derived/` | Intermedios | CSVs normalizados para Morph-KGC |

Los notebooks de ejecución están en `notebooks/graph/`.

---

## Pasos seguidos (lo que ya hice)

### 1. Crear el grafo base
Ejecuté `notebooks/graph/rag-graph-ontologia-mapeo.ipynb`:
- Lee `referencias_clean.csv` (1.174 artículos)
- Lee la ontología `idearq-biblio-v6.ttl` (BIBO + extensiones)
- Genera 4 CSVs derivados en `_derived/`
- Materializa el grafo con Morph-KGC (YARRRML)
- Serializa a `idearq-graph-instances.nt` (17.306 triples, 2.794 autores)

### 2. Detectar y fusionar duplicados de autores
Ejecuté `notebooks/graph/normalizar-autores-v2.ipynb` (autocontenido, lee el grafo original):
- **Paso 1 (slug exacto)**: 317 URIs fusionadas en 288 grupos
- **Paso 2 (fuzzy matching)**: 91 URIs adicionales en 65 grupos
- Genera `idearq-graph-instances-dedup-v2.nt` (16.361 triples, 2.409 autores)

### 3. Visualizar co-autorías
Ejecuté `notebooks/graph/explorar-coautorias.py`:
- Lee el grafo limpio
- Genera HTML interactivo con pyvis (red de autores-artículos-revistas)

### 4. Crear interfaz de consultas
`notebooks/graph/consultas-grafo.ipynb`:
- Carga el grafo v2 por defecto
- 11 queries predefinidas (autores, journals, documentos, estadísticas)
- Editor SPARQL libre

---

## Glosario

### ¿Qué es un **slug** (en este contexto)?

Un **slug** es una versión "normalizada" de un nombre de autor que sirve como identificador para detectar duplicados. La normalización aplica:

1. **Quitar acentos**: `João` → `Joao`, `Ángel` → `Angel`
2. **Lowercase**: `PABLO` → `pablo`
3. **Quitar puntuación**: `,`, `.`, `-` → espacio
4. **Ordenar tokens alfabéticamente**: `Cardoso Joao Luis` (orden independiente de apellido/nombre)
5. **Colapsar espacios**

**Ejemplo**:
- `Ángel Esparza Arroyo` → `angel arroyo esparza`
- `ÁNGEL ESPARZA-ARROYO` → `angel arroyo esparza`
- `Ángel Esparza-Arroyo` → `angel arroyo esparza`

Los tres generan el mismo slug → se fusionan en una sola URI de autor.

### ¿Qué es **fuzzy matching**?

Cuando dos nombres son similares pero **no idénticos** después de normalizar, no se pueden fusionar con un slug exacto. Fuzzy matching mide la similitud entre dos strings.

**Métricas que uso (combinadas)**:

1. **Jaccard sobre tokens**: solapamiento de palabras
   - `António M Monge Soares` vs `António Monge Soares`
   - Tokens: `{antonio, m, monge, soares}` vs `{antonio, monge, soares}`
   - Intersección: 3, Unión: 4 → Jaccard = 0.75

2. **Levenshtein normalizado**: distancia de edición / longitud máxima
   - Mide cuántas inserciones/borrados/sustituciones necesitas para convertir un string en otro
   - Normalizado a [0,1] donde 1 = idénticos

3. **Similitud final**: `max(jaccard, levenshtein)`

**Umbrales**:
- `≥ 0.85` → fusión automática
- `0.70 - 0.85` → caso dudoso (mostrado para revisión)
- `< 0.70` → no se fusiona

**Ejemplo real**:
- `António M. Monge Soares` vs `António Monge Soares` → 0.909 → fusionado
- `Pablo Arias` vs `Pablo Arias Cabal` → 0.667 → no fusionado
- `G. J. VAN KLINKEN` vs `Gibaja, Juan F.` → 0.267 → personas distintas

---

## Pasos a seguir cuando cambie la ontología o el CSV

### Escenario A: cambia el CSV `referencias_clean.csv`
Se añaden/eliminan/modifican artículos.

1. Ejecutar `notebooks/graph/rag-graph-ontologia-mapeo.ipynb`
   - Regenera `idearq-graph-instances.nt`
2. Ejecutar la **Cell 11 (bonus)** del mapeo
   - Encadena automáticamente `normalizar-autores-v2.ipynb`
   - Genera `idearq-graph-instances-dedup-v2.nt`
3. Re-ejecutar `notebooks/graph/explorar-coautorias.py`
   - Genera el HTML interactivo con los datos nuevos
4. Abrir `notebooks/graph/consultas-grafo.ipynb`
   - Las queries verán los datos actualizados

### Escenario B: cambia la ontología `idearq-biblio-v6.ttl`
Se añaden/modifican clases o propiedades en BIBO.

1. **Verificar** que el YARRRML (`notebooks/graph/mapping-idearq.yarrrml`) sigue mapeando correctamente
   - Si se añade una propiedad nueva a la ontología, hay que añadirla al mapping
   - Si se renombra una clase, hay que actualizar los `rdf:type` en el mapping
2. Ejecutar `notebooks/graph/rag-graph-ontologia-mapeo.ipynb`
3. Ejecutar la **Cell 11 (bonus)** del mapeo (encadena la deduplicación)
4. Re-ejecutar `explorar-coautorias.py`

### Escenario C: se quiere afinar la normalización
Cambiar umbrales en Cell 4 de `normalizar-autores-v2.ipynb`:
- `THRESHOLD_MERGE = 0.85` (subirlo = más conservador, más falsos negativos)
- `THRESHOLD_REVIEW = 0.70` (bajarlo = más casos para revisar)
- Re-ejecutar el notebook

### Escenario D: consulta puntual
1. Abrir `notebooks/graph/consultas-grafo.ipynb`
2. Cambiar `GRAPH_VERSION = "v2"` a `"original"` si se quiere ver el grafo sin deduplicar
3. Editar/escribir la query en la celda del editor libre

---

## Resumen del pipeline

```
referencias_clean.csv + idearq-biblio-v6.ttl
     │
     ▼
[mapeo.ipynb]  ─────►  idearq-graph-instances.nt  (2,794 autores, 17,306 triples)
     │                            │
     │                            ▼
     │                    [normalizar-v2.ipynb]
     │                            │
     │                            ▼
     │                    dedup-v2.nt  (2,409 autores, 16,361 triples)
     │                            │
     │                            ├──► [consultas-grafo.ipynb]  (queries SPARQL)
     │                            │
     │                            └──► [explorar-coautorias.py]  (HTML interactivo)
     │
     └──► [Cell 11 bonus]  ─────► encadena normalizar-v2 automáticamente
```

---

## Dependencias Python

Todas instaladas en `env_rag` (Python 3.12):

| Paquete | Uso |
|---|---|
| `morph-kgc` | Materialización RDF desde CSV (YARRRML) |
| `rdflib` | Manipulación de grafos RDF, serialización |
| `pyvis` | Visualización interactiva HTML |
| `networkx` | Construcción de grafos para pyvis |
| `matplotlib` | Gráficos estáticos |
| `pandas` | Pre-procesado de CSVs |

---

## Estructura de archivos

```
RAG/
├── data/biblio-graph/
│   ├── README.md                          ← este archivo
│   ├── referencias_clean.csv              ← input: 1.174 artículos
│   ├── idearq-biblio-v6.ttl               ← ontología BIBO
│   ├── idearq-graph-instances.nt          ← grafo original
│   ├── idearq-graph-instances-dedup-v2.nt ← grafo limpio
│   └── _derived/
│       ├── docs.csv                       ← documentos normalizados
│       ├── authors.csv                    ← autores (1 por fila)
│       ├── journals.csv                   ← revistas únicas
│       └── subjects.csv                   ← temas únicos
│
└── notebooks/graph/
    ├── rag-graph-ontologia-mapeo.ipynb    ← mapeo CSV → RDF
    ├── normalizar-autores-v2.ipynb        ← deduplicación (slug + fuzzy)
    ├── consultas-grafo.ipynb              ← queries SPARQL
    ├── explorar-coautorias.py             ← genera HTML interactivo
    ├── explorar-coautorias-dedup-v2.html  ← visualización v2
    ├── mapping-idearq.ini                 ← config Morph-KGC
    └── mapping-idearq.yarrrml             ← reglas YARRRML
```
