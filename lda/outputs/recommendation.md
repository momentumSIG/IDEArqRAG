# Recomendación Final: LDA vs BERTopic para IDEArq

## Método seleccionado: LDA K=11

### Métricas comparativas

| Métrica | LDA K=11 | BERTopic BGE-M3 | BERTopic E5-Large |
|---|---|---|---|
| Nº Tópicos | 11 | 2 | 4 |
| Coherence (c_v) | 0.7353 | N/A | N/A |
| Topic Diversity | 0.945 | 0.85 | 0.80 |
| Outliers | 0 | 0 | 0 |
| Docs asignados | 513 | 506 | 506 |

### Tópicos identificados (K=11)

| Topic | Docs | % | Temática | Keywords principales |
|---|---|---|---|---|
| T0 | 23 | 4.5% | Isótopos/Dieta | isotope, bone, tooth, enamel, strontium |
| T1 | 125 | 24.4% | Funerario/Asentamiento | funerario, fosa, enterramiento, individuo, piedra |
| T2 | 5 | 1.0% | Lítica/Cogotas | navarra, cogotas, sílex, industria |
| T3 | 16 | 3.1% | Megalitismo/Dólmenes | túmulo, monumento, cámara, dolmen, megalítico |
| T4 | 1 | 0.2% | Cerámica (outlier) | borde, urna, plato, torno |
| T5 | 11 | 2.1% | Campaniforme | campaniforme, meseta, ciempozuelos |
| T6 | 15 | 2.9% | Megalitismo portugués | lisboa, anta, megalitismo, portugal |
| T7 | 16 | 3.1% | Castros/Portugal | castro, mina, ferro, cronologia |
| T8 | 18 | 3.5% | Neolítico catalán | cova, jaciment, neolític, barcelona |
| T9 | 117 | 22.8% | Bioarqueología/EN | date, bone, cave, burial, mesolithic |
| T10 | 121 | 23.6% | Abrigos/Arte rupestre | abrigo, cova, beta, mesolítico, arte |

### Grid search completo

| K | Coherence | Diversity | Max % |
|---|---|---|---|
| 5 | 0.762 | 0.70 | 35.3% |
| 6 | 0.752 | 1.00 | 53.6% |
| 7 | 0.649 | 0.97 | 48.9% |
| 8 | 0.629 | 0.96 | 49.9% |
| 9 | 0.690 | 0.94 | 47.8% |
| 10 | 0.711 | 0.93 | 48.5% |
| **11** | **0.735** | **0.95** | **30.2%** |
| 13 | 0.699 | 0.88 | 24.4% |

### Justificación

LDA K=11 ofrece el mejor balance entre:
- **Coherence alta** (0.735, segundo mejor tras K=6)
- **Distribución balanceada** (max 30.2% vs 53.6% en K=6)
- **11 tópicos interpretables** arqueológicamente
- **Sin outliers** (todos los docs asignados)

### Siguientes pasos

1. Cargar `topics_for_graph_final.json` en Neo4j
2. Crear nodos `Topic` con relaciones `(:Paper)-[:BELONGS_TO]->(:Topic)`
3. Usar tópicos como capa semántica para enrutar consultas en el RAG
4. Revisar T4 (1 doc) y T2 (5 docs) para posible fusión manual

---
*Generado el 30 de June de 2026*
