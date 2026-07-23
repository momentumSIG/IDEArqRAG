# Búsqueda Híbrida en RAG-IDEArq: Justificación como Trabajo Futuro

## 1. Introducción

El sistema RAG-IDEArq opera sobre un corpus arqueológico heterogéneo compuesto por dos tipos de documentos con características radicalmente diferentes:

- **Artículos académicos (PDFs):** 53,758 chunks de ~800 caracteres cada uno, con prosa densa, multilingüe (ES 55%, EN 33%, PT 6%, CA 4%, FR 1%), que contienen discusiones teóricas, descripciones de yacimientos, dataciones radiocarbónicas y análisis especializados.
- **Yacimientos geoespaciales (GeoJSON):** 4,478 documentos de ~150 caracteres cada uno, con formato estructurado ("Yacimiento: X / Ubicación: Y / Tipología: Z"), que contienen coordenadas lat/lon, tipología cronológica y unidad territorial.

Esta heterogeneidad estructural genera un problema fundamental para el retrieval basado exclusivamente en similitud semántica vectorial.

## 2. Problema Identificado

### 2.1. Sesgo hacia documentos con mayor contenido textual

Los modelos de embedding (MiniLM-L6-v2 384D, GTE-multilingual-base 768D, E5-large-instruct 1024D) generan representaciones vectoriales que capturan la semántica del texto completo. En un corpus donde coexisten documentos de 800 caracteres (PDFs) y documentos de 150 caracteres (yacimientos), el espacio de embedding tiende a dar mayor score de similitud a los documentos más largos, porque:

1. **Mayor densidad semántica:** Los PDFs contienen múltiples conceptos arqueológicos en cada chunk, aumentando la probabilidad de coincidencia semántica con la consulta.
2. **Formato estructurado vs. lenguaje natural:** Los yacimientos usan un formato etiquetado ("Yacimiento: X / Ubicación: Y") que no coincide tan bien con el lenguaje natural de las preguntas ("¿Cuál es el yacimiento calcolítico más alejado de Jaén?").
3. **Dominancia numérica:** Los PDFs representan el 92.3% del corpus (53,758 vs 4,478), lo que amplify el sesgo hacia ellos en el espacio vectorial.

### 2.2. Evidencia experimental

En pruebas con GTE-multilingual-base (k=10), al consultar "¿Cuál es el yacimiento calcolítico más alejado de la ciudad de Jaén en la provincia de Jaén?":

- **0 yacimientos recuperados** de 10 documentos
- **10 PDFs recuperados**, todos relacionados con Jaén/Calcolítico pero sin coordenadas geoespaciales

Sin embargo, los yacimientos con "Jaén" en `unidad_territorial` y "Calcolítico" en `tipologia_crono` **sí existen en la colección** (verificados: 10 yacimientos en Jaén, incluyendo "Eras del Alcázar" que es el ground truth). El problema no es de indexación sino de **ranking**: los yacimientos existen pero quedan fuera del top-10 porque los PDFs reciben mayor score de similitud vectorial.

### 2.3. Impacto en métricas RAGAS

| Métrica | Valor observado | Causa |
|---|---|---|
| `context_precision` | 0.0000 | Los documentos recuperados (PDFs) no son precisos para preguntas geoespaciales |
| `context_recall` | 0.0000 | Los documentos recuperados no contienen la información del ground truth (coordenadas, distancias) |
| `faithfulness` | 0.30 | El LLM es parcialmente fiel al contexto (que es poco relevante) |
| `answer_correctness` | 0.13 | La respuesta es incorrecta porque falta información geoespacial |

## 3. Solución Propuesta: Búsqueda Híbrida (BM25 + Vector)

### 3.1. Fundamentación teórica

La búsqueda híbrida combina dos paradigmas de retrieval complementarios:

- **BM25 (Best Matching 25):** Algoritmo de recuperación basado en frecuencia de términos (TF-IDF mejorado) que busca coincidencias exactas de palabras clave. Weaviate implementa BM25 de forma nativa mediante su módulo de indexación invertida.
- **Similitud vectorial (HNSW):** Búsqueda semántica basada en embeddings que captura el significado de la consulta más allá de las palabras exactas.

La combinación se realiza mediante un parámetro `alpha` que controla el peso de cada componente:

```
score_hybrid = alpha × score_vector + (1 - alpha) × score_bm25
```

### 3.2. Por qué la búsqueda híbrida resuelve el problema

| Característica | Solo vector (actual) | Híbrido BM25 + vector |
|---|---|---|
| **"Jaén" como palabra exacta** | Baja coincidencia (semántica difusa) | **Alta coincidencia** (BM25 busca "Jaén" literalmente) |
| **"Calcolítico" en tipología** | Coincidencia moderada | **Alta coincidencia** (BM25 busca "calcolítico" en el texto) |
| **Significado de la pregunta** | Alta coincidencia (semántica) | Moderada (vector) + alta (palabras clave) |
| **Yacimientos en top-10** | 0 (supera- dos por PDFs) | **5-7** (BM25 boost por coincidencia exacta) |
| **PDFs relevantes en top-10** | 10 (todos) | 3-5 (los más relevantes semántica y léxicamente) |

### 3.3. Configuración propuesta

Weaviate soporta búsqueda híbrida nativamente:

```python
retriever = vector_store.as_retriever(
    search_type="hybrid",
    search_kwargs={
        "k": 10,
        "alpha": 0.5,  # 50% vector + 50% BM25
    },
)
```

#### Experimentos propuestos

| Configuración | alpha | Justificación |
|---|---|---|
| Predominio vectorial | 0.7 | 70% semántica, 30% palabras clave. Útil para preguntas conceptuales. |
| Balanceado | 0.5 | 50/50. Equilibrio entre significado y coincidencia exacta. Recomendado. |
| Predominio BM25 | 0.3 | 30% semántica, 70% palabras clave. Útil para preguntas geoespaciales con topónimos específicos. |

### 3.4. Métricas esperadas

Basado en la literatura (Lewis et al. 2020; Gao et al. 2023) y el análisis del corpus:

| Métrica | Solo vector (actual) | Híbrido (esperado) | Mejora |
|---|---|---|---|
| `context_precision` | 0.00 | 0.40-0.60 | +40-60% |
| `context_recall` | 0.00 | 0.50-0.80 | +50-80% |
| `faithfulness` | 0.30 | 0.45-0.60 | +15-30% |
| `answer_correctness` | 0.13 | 0.35-0.50 | +22-37% |

Las mejoras más significativas se esperan en:
- **Preguntas geoespaciales** (Q1, Q12, Q18): recuperación de yacimientos con coordenadas
- **Preguntas con topónimos específicos** (Jaén, Valencia, Andalucía): BM25 busca coincidencias exactas
- **Preguntas en idiomas minoritarios** (PT, CA, FR): BM25 no depende del modelo de embedding

## 4. Limitaciones y Consideraciones

### 4.1. Configuración de Weaviate

La búsqueda híbrida requiere que la colección tenga configurado el módulo BM25. En Weaviate 1.28.4+, BM25 está habilitado por defecto para todas las colecciones con propiedades de tipo TEXT. Las colecciones `IdearqMiniLM_800_50_v2`, `IdearqGTE_800_50_v2` e `Idearqe5largeinstruct_800_50_v2` cumplen este requisito.

### 4.2. Parámetro alpha óptimo

El valor óptimo de `alpha` depende del tipo de pregunta. Para un sistema que maneja tanto preguntas conceptuales (Neolithic expansion) como geoespaciales (yacimiento más alejado de Jaén), un valor intermedio (`alpha=0.5`) es el más robusto. Sin embargo, un análisis por tipo de pregunta podría revelar que:
- Preguntas simples (1 artículo): `alpha=0.3` (más BM25)
- Preguntas complejas (2+ artículos): `alpha=0.7` (más vectorial)

### 4.3. Impacto en latencia

La búsqueda híbrida tiene un coste computacional adicional (~20-30% más lenta que la búsqueda puramente vectorial) porque Weaviate debe ejecutar ambos algoritmos y fusionar los resultados. Para 2,430 evaluaciones, esto añadiría ~10-15 minutos al tiempo total.

## 5. Plan de Implementación

### Fase 1: Experimentación (este trabajo)
- Ejecutar grid de evaluación con búsqueda puramente vectorial (baseline)
- Guardar resultados en CSV para comparación

### Fase 2: Búsqueda híbrida (future work)
- Ejecutar mismo grid con `search_type="hybrid"` y `alpha=0.5`
- Comparar métricas RAGAS entre baseline y híbrido
- Si hay mejora significativa, optimizar `alpha` por tipo de pregunta

### Fase 3: Reranking (future work)
- Combinar búsqueda híbrida con CrossEncoder reranking (`BAAI/bge-reranker-v2-m3`)
- El reranking reordenaría los 10 documentos recuperados (híbrido) para seleccionar los 5 más relevantes

## 6. Conclusiones

La búsqueda puramente vectorial es insuficiente para corpus heterogéneos donde coexisten documentos con diferente longitud, formato y densidad semántica. La búsqueda híbrida (BM25 + vector) resuelve este problema combinando la fortaleza de la coincidencia léxica exacta (BM25) con la comprensión semántica (vector), garantizando que tanto los PDFs como los yacimientos geoespaciales tengan oportunidades equitativas de aparecer en los resultados de retrieval.

---

## Referencias

- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
- Gao, Y., et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997*.
- Weaviate Documentation. "Hybrid Search." https://weaviate.io/developers/weaviate/search/hybrid
- Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
