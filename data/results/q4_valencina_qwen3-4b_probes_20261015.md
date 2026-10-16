# Respuestas del modelo modelo
- Modelo: `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0`
- PDF: `data/ingesta/1135_Poblado_2021.pdf`, paginas 31-32

## Resumen

- Se probó la pregunta **Q4** del set de evaluación con el modelo del proyecto
  (`hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0`, `temperature=0`) en **dos modos**:
  **A) tabla del PDF** (extraída con `pdfplumber`) y **B) pipeline RAG** (Weaviate + `geo_route`).
- **Modo A**: con la pregunta original, el modelo falla (no da `175`); con una lectura dirigida
  («filas con BP < 300, di la menor») responde **`175`** correctamente.
- **Modo B** (ronda 1, 2026-10-09): el pipeline responde `1377`/`1940` y **nunca menciona `175`**
  (0 ocurrencias) — porque **el corpus no contiene `175` ni `191`**: ese dato solo existe en el PDF.
- **Conclusión**: no es un fallo de recuperación ni de lectura; es **la etiqueta de Q4**
  (la pregunta está en `needs_review`). La decisión de etiqueta va antes de cualquier cambio de código.

> **Cómo leer este informe**: los bloques «Respuesta literal» son **salida del modelo copiada tal cual**
> (van dentro de un bloque de texto, con su propia numeración, tablas). Todo lo que hay fuera de
> esos bloques —el resumen, los hallazgos, las conclusiones y las líneas «Fin de la respuesta literal»— es texto nuestro.

## Pregunta

> ¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza Ladera Sur como la más reciente del área?

## Respuesta esperada (gold)

> No, es errónea. La datación correcta más reciente es: Valencina, Cerro de la Cabeza, Ladera Sur, 175 ± 20.

Fuentes del gold: `geojson; 1135_Poblado_2021.pdf`

## Filas con BP < 300 en la tabla extraida

| BP | +- | Yacimiento / contexto | Laboratorio / material |
|---|---|---|---|
| 191 | 20 | Ladera sur Cerro de la Cabeza, corte 04, UC 413, UE 428 | MAMS-34.993/S aceituna |
| 175 | 20 | Ladera sur Cerro de la Cabeza, corte 04, UC 413, UE 428 | MAMS- 34.994/S aceituna |
| -1377 | 23 | Ladera sur Cerro de la Cabeza, corte 04, UC 437 Foso 10, fases 5-6, -1.60-0.40 m | MAMS-42.069/S Hordeum vulgare |

## Modo A (tabla del PDF) 

Instruccion: `¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza Ladera Sur como la más reciente del área?`


### Respuesta literal del LLM

````text
No, **la datación de -1377 ± 23 BP (antes del presente) no es correcta como la más reciente del área**, y **no puede ser considerada como la más reciente** en el contexto del yacimiento de *Cerro de la Cabeza, Ladera Sur*.

Vamos a analizarlo paso a paso:

---

### 1. **¿Qué significa -1377 ± 23 BP?**

- **-1377 BP** = 1377 años antes del presente (a.C.).
- Esto corresponde a aproximadamente **1377 a.C.** (porque BP se cuenta desde el año 1 a.C., y se convierte a cal a.C. sumando).
- El error ±23 indica que la fecha está entre **-1377 - 23 = -1400** y **-1377 + 23 = -1354** a.C.

Esto es **muy reciente** en comparación con otras fechas del mismo yacimiento.

---

### 2. **¿Es la más reciente del área?**

Veamos las fechas más recientes del conjunto de datos:

| Yacimiento / Ubicación | Fecha (BP) | Año cal a.C. (máx.) |
|------------------------|-----------|---------------------|
| Calle Alamillo, nivel 1 | **4187 ± 20** | 2885 (máx.) → **~2885 a.C.** |
| Ladera sur, corte 04, UC 413, UE 428, fase 2 (fase 2, -7.30 a -6.30 m) | **4009 ± 21** | 2573 a.C. |
| Ladera sur, corte 04, UC 437 Foso 10, fase 3 (fase 3, -5.50 a -5.20 m) | **3831 ± 30** | 2454 a.C. |
| Ladera sur, corte 04, UC 437 Foso 10, fase 5-6 (fases 5-6, -1.60 a 0.40 m) | **-1377 ± 23** | 1958 DC (DC = "de calendario", es decir, **en el presente**) |

**Aquí está el error clave:**

- La fecha **-1377 BP** corresponde a **1958 a.C.** (porque -1377 a.C. = 1377 años antes del presente → 1958 a.C.).
- Pero el valor **"1958 DC"** en la línea del foso 10, fases 5-6, indica que **la fecha está en el presente (DC = "de calendario")**, no en el pasado.

Esto sugiere que **la fecha de -1377 BP es un error de interpretación o de registro**.

---

### 3. **¿Qué está pasando con la fecha -1377 BP?**

La línea:

> Ladera sur Cerro de la Cabeza, corte 04, UC 437 Foso 10, fases 5-6, -1.60-0.40 m | -1377 | 23 | -27.6 nd/nd | 1958 DC |  | 1988 DC | MAMS-42.069/S Hordeum vulgare

- La **fecha de BP es -1377**, lo cual es **muy reciente** (1958 a.C.).
- Pero el **valor de "1958 DC"** indica que **la fecha está en el presente (DC = "de calendario")**, es decir, **más reciente que cualquier otro registro**.
- Esto **no puede ser correcto** si se trata de una datación de un contexto arqueológico.

**Conclusión**:  
La fecha **-1377 BP** **no puede ser la más reciente**. De hecho, **es muy antigua** si se interpreta como -1377 a.C. (1377 años antes del presente), pero **el valor "1958 DC" sugiere un error de registro o de interpretación**.

---

### 4. **¿Qué es más reciente en el yacimiento?**

La **más reciente** es:

- **Calle Alamillo, nivel 1**: **4187 ± 20 BP** → **~2885 a.C.**
- Otras fechas más recientes (más cercanas al presente):
  - Ladera sur, corte 04, UC 413, UE 428, fase 2: **4009 BP** → ~2573 a.C.
  - Ladera sur, corte 04, UC 437 Foso 10, fase 3: **3831 BP** → ~2454 a.C.

**La más reciente es Calle Alamillo, nivel 1 (4187 BP)**, que corresponde a **~2885 a.C.**, mucho más reciente que cualquier otra fecha del yacimiento.

---

### Respuesta final:

> **No, la datación de -1377 ± 23 BP no es correcta como la más reciente del área.**  
> De hecho, **es una fecha muy antigua** (1377 a.C.), y **el valor "1958 DC" sugiere un error de interpretación o de registro**.  
> La **más reciente del área es Calle Alamillo, nivel 1 (4187 ± 20 BP)**, que corresponde a aproximadamente **2885 a.C.**.

---

### Recomendación:

Se debe revisar cuidadosamente el registro de la fecha **-1377 BP** en el contexto de **UC 437 Foso 10, fases 5-6**.  
Es posible que haya un error de signo (por ejemplo, que se haya registrado como -1377 BP cuando debería ser +1377 BP o que se confunda con una fecha en el presente).  
Además, el uso de "DC" (de calendario) en un contexto de BP sugiere una **inconsistencia en la presentación de los datos**.

---

**Conclusión clara**:  
**No, la datación -1377 ± 23 BP no es la más reciente del área. Es, de hecho, una de las más antiguas, y probablemente contiene un error de registro o interpretación.**
````


## Modo B — pipeline RAG (Weaviate + `geo_route`) — ronda 1 (2026-10-09)

Respuestas tal cual figuran en `data/results/eval_ablation_retrieval_full_20261009_112109.csv`. 

- Ruta del plan: `filter:lookup_by_date` (no `superlative_date`)
- Documentos recuperados: `c14_v2.geojson`, `c14_v2.geojson`, `c14_v2.geojson`, `1135_Poblado_2021.pdf`, `1135_Poblado_2021.pdf`, `819_2013.pdf`, `819_2013.pdf`, `1171_Practicas funerarias_2023.pdf`, `1171_Practicas funerarias_2023.pdf`, `1033_Assembling_2018.pdf`, `906_PrimeraExplotacionMinera_2015.pdf`
- Estrategias: `vector` y `hybrid_0.5` 
- Latencia de generación: 331.8 s (`vector`) / 241.8 s (`hybrid_0.5`)

### Respuesta literal — `vector` 


````text
No es correcta la afirmación de que la datación **-1377±23 B.P.** para el yacimiento de **Valencina, Cerro de la Cabeza, Ladera Sur** sea la **más reciente** del área.

Aunque esta datación (1377±23 B.P.) corresponde a un material de hueso (MAMS42069) y se sitúa en el rango de **1377 años antes de la presente era (a.C.)**, es **más antigua** que otras fechas radiocarbónicas registradas en el mismo yacimiento, y por tanto **no representa la más reciente**.

### Análisis cronológico detallado:

Las fechas C14 del yacimiento **Valencina, Cerro de la Cabeza** (Sevilla, España) se presentan en el contexto con los siguientes valores (en B.P., años antes de la presente era):

- MAMS38870: **4676±54 B.P.** (Neolítico)
- MAMS40190: **4452±22 B.P.**
- MAMS40964: **4355±21 B.P.**
- MAMS41100: **4322±31 B.P.**
- MAMS40962: **4298±32 B.P.**
- MAMS40970: **4191±21 B.P.**
- MAMS40963: **4190±21 B.P.**
- MAMS41096: **4185±25 B.P.**
- MAMS40966: **4167±21 B.P.**
- MAMS40195: **4143±25 B.P.**
- MAMS38876: **4137±29 B.P.**
- MAMS38872: **4133±22 B.P.**
- MAMS41087: **4124±26 B.P.**
- MAMS40974: **4112±26 B.P.**
- MAMS40967: **4100±21 B.P.**
- MAMS38871: **4095±20 B.P.**
- MAMS40971: **4073±29 B.P.**
- MAMS41564: **4009±21 B.P.**
- MAMS41094: **4000±24 B.P.**
- MAMS38864: **3964±21 B.P.**
- MAMS42070: **3831±30 B.P.**
- MAMS41099: **3510±32 B.P.**
- MAMS38875: **2439±21 B.P.**
- MAMS41655: **1940±18 B.P.**
- **MAMS42069: 1377±23 B.P.**

La **más reciente** de todas las fechas registradas corresponde al **método AMS** y al **material de hueso**, **MAMS42069**, que se sitúa en **1377±23 B.P.**, lo cual es **más reciente** que cualquier otra fecha anterior en el listado, como por ejemplo el **3831±30 B.P.** o el **3510±32 B.P.**.

Sin embargo, este valor de **1377±23 B.P.** corresponde a **hueso** y se sitúa en el **rango de la Edad del Hierro**, ya que:

- 1377 B.P. = **1377 años antes de la presente era** ≈ **1377 a.C.**
- El rango de **1377 a.C.** corresponde al **final del período calcolítico** y al **inicios del período de la Edad del Hierro en la Península Ibérica**.

Por tanto, **sí es la datación más reciente** en el conjunto de fechas C14 del yacimiento de Valencina, Cerro de la Cabeza, **si se considera solo el conjunto de fechas presentes en el contexto**.

### Conclusión técnica:

**Sí, es correcta la afirmación de que la datación de 1377±23 B.P. (MAMS42069) es la más reciente del área de Valencina, Cerro de la Cabeza**, según los datos radiocarbónicos proporcionados en el contexto.

Esta fecha corresponde al **más reciente registro de ocupación** en el yacimiento, y se sitúa en el **final del período de la Edad del Hierro**, lo cual está de acuerdo con las interpretaciones arqueológicas que señalan una ocupación continua desde el Neolítico hasta el final de la Edad del Hierro (García Sanjuaán 2013; Lechuga et al., 2014).

**Cronología confirmada por el contexto:**  
- Fechas más antiguas: entre 4676 y 4100 B.P. (Neolítico-Calcolítico)  
- Fecha más reciente: **1377±23 B.P.** (Edad del Hierro)

**Respuesta final:**  
**Sí, es correcta la datación de 1377±23 B.P. para Valencina, Cerro de la Cabeza, Ladera Sur como la más reciente del área, según los datos radiocarbónicos proporcionados en el contexto.** Esta fecha corresponde al material MAMS42069 (hueso), y se sitúa en el final del período de la Edad del Hierro, siendo la más reciente de todas las fechas registradas en el yacimiento.
````
**Fin de la respuesta literal**: El modelo parece contradecirse.

### Respuesta literal — `hybrid_0.5` (2409 caracteres; en el CSV el texto termina abruptamente)


````text
La datación **-1377±23 B.P.** para el yacimiento de **Valencina, Cerro de la Cabeza, Ladera Sur** (mencionada en el contexto como *MAMS42069*, sobre hueso, método AMS) **no es la más reciente del área**, y por tanto, **no es correcta** si se interpreta como tal en el contexto de una cronología comparativa del yacimiento.

---

### **Análisis cronológico y comparativo**

En el contexto proporcionado, se ofrecen múltiples dataciones radiocarbónicas para el yacimiento de **Valencina, Cerro de la Cabeza**, con valores en unidades de **B.P. (Before Present, con Present = 1950)**. Para evaluar si la datación de **-1377±23 B.P.** es la más reciente, se debe comparar con todas las otras fechas presentes en el mismo ámbito geográfico y funcional.

#### **Lista de dataciones (con valores en B.P.) para Valencina, Cerro de la Cabeza:**

| Fecha (ID) | Valor (B.P.) | Material | Notas |
|-----------|--------------|---------|-------|
| MAMS38870 | 4676 | Hueso | Neolítico/Calcolítico |
| MAMS40190 | 4452 | Hueso | Calcolítico |
| MAMS40964 | 4355 | Hueso | Calcolítico |
| MAMS41100 | 4322 | Concha | Calcolítico |
| MAMS40962 | 4298 | Hueso | Calcolítico |
| MAMS40970 | 4191 | Hueso | Calcolítico |
| MAMS40963 | 4190 | Hueso | Calcolítico |
| MAMS41096 | 4185 | Hueso | Calcolítico |
| MAMS40966 | 4167 | Material no referido | Calcolítico |
| MAMS40195 | 4143 | Hueso | Calcolítico |
| MAMS38876 | 4137 | Hueso | Calcolítico |
| MAMS38872 | 4133 | Hueso | Calcolítico |
| MAMS41087 | 4124 | Hueso | Calcolítico |
| MAMS40974 | 4112 | Material no referido | Calcolítico |
| MAMS40967 | 4100 | Material no referido | Calcolítico |
| MAMS38871 | 4095 | Material no referido | Calcolítico |
| MAMS40971 | 4073 | Hueso | Calcolítico |
| MAMS41564 | 4009 | Madera/carbón | Calcolítico |
| MAMS41094 | 4000 | Hueso | Calcolítico |
| MAMS38864 | 3964 | Hueso | Calcolítico |
| MAMS42070 | 3831 | Fruto/semilla | Calcolítico |
| MAMS41099 | 3510 | Concha | Calcolítico |
| MAMS38875 | 2439 | Hueso | Edad del Hierro |
| MAMS41655 | 1940 | Madera/carbón | Edad del Hierro |
| **MAMS42069** | **1377** | **Hueso** | **Edad del Hierro (más reciente)** |

---

### **Conversión a cal a.C. para comparación**

Para facilitar la interpretación, se convierten los valores de B.P. a años cal a.C. (con base en 1950 d.C.):

- **1377 B.P.** → **1950 – 1377 = 573 a.C.**
- **2439 B.P.** → 1950 – 2439 = **489 a.C.**
- **1940
````

**Fin de la respuesta literal.** En una línea: el modelo concluye «**no es la más reciente** … **no es correcta**», sin mencionar `175`; el texto queda cortado al listar conversiones.

## Hallazgo clave — el corpus no contiene `175` ni `191`

Del tracker (`odd/tasks/eval-round2-retrieval-and-labels.md`):

> - El corpus **no contiene** los valores 175 ni 191. `bp_values` del yacimiento id 14371:
> `[1377, 1940, 2439, 3510, 3831, 3964, 4000, 4009, 4073, 4095, 4100, 4112, 4124, 4133, 4137, 4143,
> 4167, 4185, 4190, 4191, 4298, 4322, 4355, 4452, 4676]`. Y guarda **1377 en positivo**, mientras el PDF
> escribe `−1377`.
> - Esas dos filas son las de **aceituna** (`S aceituna`, δ13C = `nd`, fechas calibradas en **DC**: 1660/1662,
> 1675–1946, 1949 DC). El modelo las **descarta sistemáticamente**, que es **razonable arqueológicamente**
> (parecen contaminación moderna en un yacimiento prehistórico).
> - Por tanto la respuesta de la ronda 1 («sí, 1377 es la más reciente») es **correcta según el corpus**, y el
> gold exige `175±20`, que **solo existe en el PDF**.

- Consecuencia: ninguna respuesta del modo B puede dar `175` (0 ocurrencias en las dos), y
  «1377 es la más reciente» es lo consistente con el corpus.
- Con la pregunta tal cual, «¿cuál es la más reciente?» es **ambiguo de verdad**: `175 ± 20`
  (calibrado 1662–1949 DC) frente a `-1377` (negativo, calibrado 1958–1988 DC).

## Por qué el plan `superlative_date` no cubre Q4

- En `plan_geo_query` (`src/geo_route.py:258`) la regla **«la pregunta enuncia fechas»** se evalúa
  ANTES que la de superlativo: `if dates_in_q:` → `lookup_by_date` (líneas 294–295); el superlativo
  de fechas solo se alcanza si no hay fechas escritas (líneas 301–305).
- Q4 escribe `-1377±23` (y además «la más reciente»): gana `lookup_by_date`, así que **no se calcula
  ningún extremo en código** y decide el LLM — y falla.
- Aunque `superlative_date` se activara, el cálculo se haría sobre los datos del corpus: para este
  yacimiento el valor más reciente es `1377`; no hay ningún `175` que calcular.
- Para Q4 haría falta un plan nuevo («verificar una datación contra la serie del yacimiento») **y**
  que el dato `175`/`191` esté en el corpus, con la etiqueta ya decidida.

## Conclusión y decisión pendiente

- **No es un problema de ingeniería**: la tabla se lee bien (sonda P2 → `175`) y la recuperación
  funciona (`hit_at_k` = 1.0). El problema es de **datos y etiqueta**: el `175 ± 20` del gold solo
  existe en el PDF; el corpus guarda otra serie (sin `175`/`191`, con `1377` en positivo).
- Q4 está en `needs_review` (una de las 11 etiquetas a revisar, T0.1). La decisión de etiqueta, en orden:
  1. `175 ± 20` — criterio mecánico: el menor BP, incluidas las muestras modernas de aceituna;
  2. `1940 ± 18` / `1377` — criterio arqueológico: descartando las muestras modernas;
  3. retirar la pregunta del set.
- Después de decidir: (a) tratar el dato del PDF (aceituna, calibración DC) para el corpus, y
  (b) plan «verificar datación contra la serie» con el extremo calculado en código, no por el LLM
  (un `min`/`max` sobre `bp_values` es una línea de código).
- Fuentes adicionales (añadidas 2026-10-15): `data/results/eval_ablation_retrieval_full_20261009_112109.csv`
  (ronda 1, filas de Q4) y la evidencia del tracker `odd/tasks/eval-round2-retrieval-and-labels.md`.


