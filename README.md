# RAG-IDEArq — Sistema de Evaluación 

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-blue.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=FFD43B)
[![LLMs](https://img.shields.io/badge/LLMs-3%20models-purple)](ontologies_generated/README.md)
---
Retrieval-Augmented Generation sobre documentación arqueológica de IDEArq
(http://www.idearqueologia.org). Evalúa modelos pequeños (Phi-3.5-mini,
Qwen3-4B, Llama-3.2-3B) con embeddings pequeños (MiniLM-L6-v2,
GTE-multilingual-base) y RAGAS con juez Mistral.

---

## Dataset RAG-IDEArq-eval-v3

30 preguntas de evaluación sobre arqueología ibérica (Paleolítico → Edad del
Hierro), divididas en **15 simples** (1 artículo) y **15 complejas** (2+ artículos).

### Distribución por idioma (proporcional al corpus LDA, 513 docs)

| Idioma | % Corpus (513 docs) | # Preguntas | Simples | Complejas |
|---|---|---|---|---|
| Español (ES)  | 55% (282 docs) | 16 | 8 | 8 |
| Inglés (EN)   | 33% (168 docs) | 10 | 5 | 5 |
| Portugués (PT)|  6% ( 30 docs) |  2 | 1 | 1 |
| Catalán (CA)  |  4% ( 22 docs) |  1 | 1 | 0 |
| Francés (FR)  |  1% (  4 docs) |  1 | 0 | 1 |
| **Total**     | 100%           | 30 | 15 | 15 |

### Definición de complejidad
- **Simple**: la respuesta se encuentra en un único paper/artículo del corpus.
- **Compleja**: la respuesta requiere sintetizar información de 2 o más papers
  (listas de yacimientos, comparaciones regionales, periodizaciones).

---

## Preguntas

### ESPAÑOL — Simples (8)

| # | Pregunta | Respuesta (ground truth) | Fuente (PDF) |
|---|---|---|---|
| 0  | ¿En qué año excavaron en el yacimiento de La Bastida de Totana los hermanos Siret? | En el año 1886. | manual_review |
| 1  | ¿Cuál es el yacimiento calcolítico más alejado de la ciudad de Jaén en la provincia de Jaén? | El yacimiento de Eras del Alcázar, a aproximadamente 50 km de la ciudad de Jaén. | manual_review |
| 2  | Excavaciones de urgencia de la Junta de Andalucía en la provincia de Almería publicadas en 2001 — ¿qué solares? | Solar en la avenida Pablo Iglesias esquina A Rafaela Jiménez, solar en la calle La Central de Villaricos (cuevas de Almanzora) y en la calle Castillejo (Gádor, Almería). | manual_review |
| 3  | ¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza Ladera Sur como la más reciente del área? | No, es errónea. La datación correcta más reciente es: Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20. | manual_review |
| 4  | ¿Estas dataciones son del Neolítico? (CV 33.900±60, Galicia 31.690±50, Murcia 12.030±0) | No, estas dataciones corresponden al período Paleolítico. | manual_review |
| 5  | ¿Cuál es la fecha más antigua para un yacimiento funerario megalítico en la Península Ibérica? | Trikuaizti 2 (Gipuzkoa), 12015 ± 145. | manual_review |
| 6  | Datación más reciente de los yacimientos calcolíticos en el área de Valencina de la Concepción (Sevilla). | Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20. | manual_review |
| 7  | Yacimiento con la data de C14 más antigua de las Islas Baleares. | Cova de Moleta (Sóller), 80000 BP. | manual_review |

### ESPAÑOL — Complejas (8)

| # | Pregunta | Respuesta | # Papers | Fuentes |
|---|---|---|---|---|
| 8  | ¿Cuáles son las cronologías de las manifestaciones funerarias del Mesolítico en las distintas regiones peninsulares? | Mediterráneo: cementerios ~9475-9300 cal BP (El Collado). Portugal (Muge): ~8409-8030 cal BP. Cantábrico: ~7981-6636 cal BP. | 3+ | manual_review |
| 9  | Periodización del Bronce Final en el Levante de la Península Ibérica, cronología de las fases y principales ejemplos de yacimientos. | Bronce tardío (c.1550/1500-1300/1250): Oropesa la Vella, Cabezo Redondo. Bronce Final I (1300-1000): Costamar, Cova d'en Pardo. Bronce Final II (1000-850): Ereta del Castellar, Mola d'Agres. Bronce Final III (850-725): La Vital, Peña Negra I. Hierro antiguo (725-550): Vinarragell, El Molón. | 5+ | manual_review |
| 10 | Yacimientos Calcolíticos de la Península Ibérica en los que se han hallado objetos de marfil. | Pre-campaniforme: Zambujal, Vila Nova de São Pedro, Leceia, Alcalar, Perdigões, Valencina, Los Millares. Campaniforme: Palmela, Pedra do Ouro, Verdelha dos Ruivos, Los Algarbes, Cerro de la Virgen. | 4+ | manual_review |
| 11 | Cronología y distribución espacial del poblamiento neolítico en la Meseta Sur. | Valle del Tajo: cuevas La Ventana, La Higuera (Sierra madrileña). La Mancha: ocupaciones en cuevas y abrigos. | 5+ | manual_review |
| 12 | Yacimientos neolíticos situados a menos de 150 km de Casa Montero. | 24 sitios: La Atalaya (Ávila), Portillo de las Cortes (Guadalajara), El Cañaveral, Casa Montero, Cueva de la Higuera (Madrid), Cueva de la Vaquera (Segovia), La Mina, Peña de la Abuela (Soria), El Castillejo (Toledo), entre otros. | 5+ | manual_review |
| 13 | Datación más antigua y más reciente de los yacimientos calcolíticos en el área de Valencina de la Concepción (Sevilla). | Antigua: Valencina, Instituto de Educación Secundaria, 4800 ± 100. Reciente: Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20. | 2+ | manual_review |
| 14 | Dataciones más antiguas (i.e, más altas) de yacimientos paleolíticos para cada comunidad autónoma. | Andalucía 51.914±45, Aragón 25.330±80, Cantabria 48.200±80, Castilla y León 30.300±25, Castilla-La Mancha 28.660±40, Cataluña 38.640±50, C. Madrid 30.280±28, Navarra 21.600±30, C. Valenciana 33.900±60, Extremadura 61.219±70, Galicia 31.690±50, Illes Balears 24.220±115, La Rioja 6.220±100, País Vasco 34.350±130, Asturias 16.700±30, Murcia 12.030±0. | 10+ | manual_review |
| 15 | Principales yacimientos de la Segunda Edad del Hierro en la provincia de León. | Valencia de Don Juan, Lancia, Regueras de Arriba, Castros del Teleno/Valdería/Bierzo, Castro de Chano, Peña del Castro (La Ercina), El Castrelín de San Juan de Paluezas, entre otros (15+ sitios). | 3+ | manual_review |

### INGLÉS — Simples (5)

| # | Pregunta | Respuesta | Fuente |
|---|---|---|---|
| 16 | In what year did the Siret brothers excavate the La Bastida de Totana site? | In 1886. | manual_review |
| 17 | What is the oldest C14 date for the Balearic Islands? | Cova de Moleta (Sóller), 80000 BP. | manual_review |
| 18 | When was the Casa Montero flint mine active? | Main episode: 5327-5215 cal BC (1σ), lasting just over a century. | manual_review |
| 19 | Which site in Sevilla has both the oldest and most recent C14 dates for the Chalcolithic? | Valencina de la Concepción (oldest: IES 4800±100; most recent: Cerro de la Cabeza 175±20). | manual_review |
| 20 | What is the oldest Paleolithic date for Andalucía? | Andalucía, 51.914 ± 45. | manual_review |

### INGLÉS — Complejas (5)

| # | Pregunta | Respuesta | # Papers |
|---|---|---|---|
| 21 | What are the main theoretical models of Neolithic expansion in Europe? | Demic diffusion (movement of Neolithic societies) vs Cultural diffusion (transmission of technology, plants, animals). | 2+ |
| 22 | Main Iron Age II sites in León province. | 15+ sites: Valencia de Don Juan, Lancia, Castros del Teleno, Castro de Chano, Peña del Castro, etc. | 3+ |
| 23 | Chalcolithic sites with ivory objects in the Iberian Peninsula. | Pre-BB (12 sites: Zambujal, VNSP, Perdigões, Valencina...) + BB (10 sites: Palmela, Los Algarbes...). | 4+ |
| 24 | Oldest Paleolithic dates by autonomous community in the Iberian Peninsula. | 17 communities with dates (Extremadura 61.219, Andalucía 51.914, Cantabria 48.200...). | 10+ |
| 25 | Main funerary chronologies of the Mesolithic across peninsular regions. | Mediterranean 9475-9300 BP, Atlantic Portugal 8409-8030 BP, Cantabrian 7981-6636 BP. | 3+ |

### PORTUGUÉS — Simples (1)

| # | Pregunta | Respuesta | Fuente |
|---|---|---|---|
| 26 | Quando os irmãos Siret escavaram La Bastida de Totana? | Em 1886. | manual_review |

### PORTUGUÉS — Complejas (1)

| # | Pregunta | Respuesta | # Papers | Fonte |
|---|---|---|---|---|
| 27 | Cronologia do Bronze Final no Levante peninsular. | 5 fases: Tardio (c.1550-1300), I (1300-1000), II (1000-850), III (850-725), Oriental (725-550). | 5+ | manual_review |

### CATALÁN — Simples (1)

| # | Pregunta | Respuesta | Fuente |
|---|---|---|---|
| 28 | Quina és la datació més antiga del jaciment neolític de Cingle del Mas Nou? | 8007-7583 cal BP. | manual_review |

### FRANCÉS — Complejas (1)

| # | Pregunta | Respuesta | # Papers | Source |
|---|---|---|---|---|
| 29 | Datations paléolithiques les plus anciennes par communauté autonome en péninsule ibérique. | 17 communautés (Andalousie 51.914, Estrémadure 61.219, Cantabrie 48.200...). | 10+ | manual_review |

---

## Cómo usar el dataset

1. Ejecutar el notebook de evaluación para crear `RAG-IDEArq-eval-v3` en Langfuse.
2. Rellenar la columna "Fuente" con el nombre del PDF exacto (ej. `1000_oms_2017.pdf`) en Langfuse UI.
3. Las preguntas se evalúan con RAGAS (faithfulness, context_precision, context_recall, answer_correctness).
4. El reporte se segmenta por `tipo` (simple/compleja), `idioma` y `temperatura`.

## Modelos evaluados
- **LLMs**: Phi-3.5-mini, Qwen3-4B-Instruct-2507, Llama-3.2-3B-Instruct
- **Temperaturas**: 0.3, 0.5, 0.7
- **Embeddings**: all-MiniLM-L6-v2 (384D), gte-multilingual-base (768D), e5-large-instruct (1024D)
- **Judge RAGAS**: Mistral API (`mistral-small-latest`)

## Estructura del proyecto
```
RAG/
├── src/                          ← Código Python
│   ├── config.py                 ← Parámetros de chunking y embeddings
│   ├── langfuse_monitor.py       ← Conexión a Langfuse
│   ├── app_streamlit.py          ← UI Streamlit
│   └── backend_flask.py          ← API Flask
├── notebooks/
│   ├── indexing/                 ← Notebooks de indexación
│   └── evaluation/               ← Notebooks de evaluación
├── data/
│   ├── ingesta/                  ← PDFs fuente
│   ├── results/                  ← Resultados de evaluación
│   └── eval_questions.py         ← Dataset v3 en código
├── deploy/docker/                ← Docker Compose (Weaviate)
└── weaviate_data/                ← Datos de Weaviate
```
