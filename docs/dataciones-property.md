# Dataciones C14 filtrables: `bp_values` y `dataciones_c_14`

> Colección afectada: **`IdearqMiniLM_800_50_v2`** (Weaviate 1.33.0).
> Ninguna otra colección se ha tocado.

## 1. Qué se añadió

Dos propiedades nuevas en la colección, ambas **solo de filtro**:

| Propiedad | Tipo | Filtrable | Buscable con BM25 | Origen |
|---|---|---|---|---|
| `bp_values` | `number[]` | sí (`ContainsAny`, `ContainsAll`) | no aplica (números) | edades B.P. de `dataciones_c_14`, una por datación |
| `dataciones_c_14` | `text` | sí (`Like`, `Equal`) | **no** (`indexSearchable: false`) | texto literal del GeoJSON |

También se han añadido a `INDEX_PROPERTIES_FULL` en `src/config.py` (de 23 a 25
propiedades), a las funciones de enriquecimiento y de relleno del notebook
`notebooks/indexing/rag-indexacion-mejorado.ipynb`, y al mapa de tipos de
`src/reindex_gte_e5.py`.

## 2. Por qué hacía falta

El GeoJSON `data/ingesta/geojson/c14_v2.geojson` guarda **todas** las dataciones
de cada yacimiento en un solo texto, separadas por `<br/>`:

```
MAMS38870 4676±54 B.P. sobre Hueso (AMS)<br/>…<br/>MAMS42069 1377±23 B.P. sobre Hueso (AMS)
```

Ese texto ya estaba dentro de `content`, así que el modelo **podía leerlo**. Lo
que faltaba era **filtrar** por él.

El único filtro de fecha que existía era `bp_max`, que guarda **la datación más
antigua** del yacimiento. Para la pregunta

> «¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza…?»

`bp_max` de Valencina es **4676**, con lo que 1377 nunca coincide con nada: la
pregunta era incontestable por filtro.

Con `bp_values` cada fecha del yacimiento es un elemento de una lista y se puede
preguntar «¿qué yacimiento tiene la fecha 1377?».

Dos matices prácticos:

- El GeoJSON guarda la fecha como **1377** (años antes del presente, positivos).
  En la pregunta aparece «-1377»; hay que buscar `1377`, sin signo.
- `bp_max = max(bp_values)` siempre: las dos columnas se calculan con el mismo
  código, así que no se pueden desincronizar.

## 3. Cómo se usa (consulta GraphQL de ejemplo)

### Filtrar por una fecha concreta

```graphql
{
  Get {
    IdearqMiniLM_800_50_v2 (
      where: {
        operator: And
        operands: [
          { path: ["filename"], operator: Equal, valueText: "c14_v2.geojson" },
          { path: ["bp_values"], operator: ContainsAny, valueNumber: [1377] }
        ]
      }
      limit: 10
    ) {
      yacimiento_id
      yacimiento_nombre
      bp_max
    }
  }
}
```

**Resultado real (6 objetos = 3 yacimientos × 2 objetos):**

| yacimiento_id | yacimiento_nombre | bp_max |
|---|---|---|
| 11659 | Cova des Càrritx | 3325 |
| **14371** | **Valencina, Cerro de la Cabeza, Parcela Municipal** | **4676** |
| 11762 | Castro Ventosa | 1717 |

> **Atención al nombre del campo.** En Weaviate 1.33 **no existe**
> `valueNumberArray`; si lo pones, GraphQL responde *«In field
> "valueNumberArray": Unknown field»*. La forma correcta es `valueNumber: [1377]`
> (el escalar acepta una lista cuando el operador es `ContainsAny` /
> `ContainsAll` / `ContainsNone`).

Controles realizados:

- `ContainsAny valueNumber: [999999]` → **0 resultados** (el filtro de verdad filtra).
- `ContainsAll valueNumber: [4676, 1377]` → **solo Valencina** (los dos objetos),
  porque es el único sitio con ambas fechas.

### Filtrar por la fecha exacta, con su error

`bp_values` solo guarda la fecha nominal. Si necesitas el «±23», usa la columna
de texto:

```graphql
{
  Get {
    IdearqMiniLM_800_50_v2 (
      where: {
        operator: And
        operands: [
          { path: ["filename"], operator: Equal, valueText: "c14_v2.geojson" },
          { path: ["dataciones_c_14"], operator: Like, valueText: "*1377±23*" }
        ]
      }
      limit: 10
    ) {
      yacimiento_id
      yacimiento_nombre
    }
  }
}
```

**Resultado real:** 4 objetos → **Cova des Càrritx** y
**Valencina, Cerro de la Cabeza, Parcela Municipal** (los dos sitios que tienen
exactamente `1377±23`). Más estrecho que la consulta numérica, que además
devolvía Castro Ventosa (`1377±28`).

## 4. Resultados del relleno (backfill)

Script: `scripts/backfill_c14_dates.py` — **modo `--dry-run` por defecto**;
hay que pasar `--apply` para escribir.

```bash
python3 scripts/backfill_c14_dates.py --dry-run            # solo informe
python3 scripts/backfill_c14_dates.py --apply --ensure-schema   # crea + rellena
```

### `--apply` sobre `IdearqMiniLM_800_50_v2` (ejecutado, exit 0)

| Métrica | Valor |
|---|---|
| Objetos leídos (`filename = c14_v2.geojson`) | **4478** |
| Objetos sin `yacimiento_id` enlazable al GeoJSON | **0** |
| Discrepancias `bp_min`/`bp_max`/`c14_count` calculados vs. almacenados | **0** |
| Discrepancias entre el texto de `content` y el GeoJSON | **0** |
| Propiedades nuevas creadas en el esquema | `bp_values`, `dataciones_c_14` |
| Objetos actualizados (HTTP 204) | **4478** |
| Objetos fallidos | **0** |
| Tiempo | 1,7 s |
| `bp_values` con al menos una fecha | **4414** |
| `bp_values` a `[]` (32 yacimientos sin fecha parseable, ×2 objetos) | **64** |
| `dataciones_c_14` relleno | **4478** |
| Vectores comparados / cambiados | **4478 / 0** |

Fuente: 4478 características del GeoJSON → **2239 yacimientos únicos**
(el GeoJSON repite cada yacimiento exactamente 2 veces y las dos copias son
idénticas, comprobado); **2207 con fechas parseables**, **32 sin**;
**11355** valores B.P. en total.

Segunda pasada en `--dry-run`: `already up to date: 4478`, `planned: 0`,
`writes performed: 0`. El script es **idempotente**.

## 5. Riesgo de re-vectORIZación: descartado

**El temor.** Si Weaviate vectorizara el texto de la colección, añadir una
propiedad de texto y actualizar los objetos podría re-calcular sus vectores. Eso
cambiaría todos los resultados de recuperación y dejaría sin valor la evaluación
recién hecha.

**Comprobaciones realizadas, en este orden:**

1. **Esquema real leído antes de tocar nada** (`GET /v1/schema`): la clase
   `IdearqMiniLM_800_50_v2` tiene `"vectorizer": "none"` y `moduleConfig: null`.
   Las 7 colecciones del servidor están con `vectorizer: none`. Nadie vectoriza
   del lado de Weaviate: los vectores los envía el cliente y quedan guardados
   tal cual. Ninguna propiedad de la clase lleva `skip` porque no hay módulo de
   vectorización que saltar.
2. **Prueba en una clase desechable** (`IdearqCanaryScratch`, creada y borrada
   antes del cambio real): se creó la misma pareja de propiedades, se insertó un
   objeto con vector, se hizo `PATCH` de propiedades y **el vector siguió
   idéntico**. La clase se eliminó; solo quedan las 7 originales.
3. **Prueba real, antes y después del `--apply`:**
   - antes: 4478 vectores, `sha256` del conjunto =
     `425bcc2241ad14c93d800bafe0b7c5d888f0fdf0dac397575271e3dfb8f5203d`
   - después: 4478 vectores, **mismo `sha256`**, **0 diferencias**, 0 uuid de
     más y 0 de menos.
   - muestra concreta, `yacimiento_id = 14371` (Valencina, 384 dimensiones):
     antes y después `sha256(vec) = c7e93ea28f992bce`.

**Conclusión:** los vectores no cambian. Si en algún momento la colección
tuviera un vectorizador distinto de `none`, el script se detiene solo
(`FATAL: vectorizer is not 'none'`).

## 6. Riesgo secundario: el índice léxico (BM25) — también descartado

Al añadir una columna de texto larga habría otro modo de romper la evaluación:
si entrara en el índice BM25, cambiarían las puntuaciones híbridas y todos los
`hit@k`/`MRR` medidos hasta ahora.

Por eso `dataciones_c_14` se creó con **`indexSearchable: false`**. Weaviate lo
confirma él mismo:

```
Searching by property 'dataciones_c_14' requires inverted index.
Is `indexSearchable` option of property 'dataciones_c_14' enabled?
```

Es decir, la columna **no se puede buscar** con BM25, solo filtrar. Comprobado
además que:

- un BM25 explícito sobre `content` sigue funcionando y sigue devolviendo
  Valencina el primero;
- `bp_values` es numérico y los números nunca entran en BM25;
- el notebook de evaluación limita además su búsqueda a
  `BM25_PROPS = ["content", "tipologia_crono", "unidad_territorial"]`.

## 7. Cómo volver a ejecutarlo

```bash
# 1) solo informe, no escribe nada (es el comportamiento por defecto)
python3 scripts/backfill_c14_dates.py --dry-run

# 2) escribir (pide --ensure-schema solo si faltan las propiedades)
python3 scripts/backfill_c14_dates.py --apply --ensure-schema
```

El script solo mira `IdearqMiniLM_800_50_v2`. Si se le pasa otra colección
imprime un aviso. **`IdearqGTE_800_50_v2` queda pendiente del visto bueno del
usuario**: no se le ha añadido ninguna propiedad ni se le ha rellenado nada.

## 8. Hallazgos que no se han tocado

- **32 objetos con `c14_count` a `null`** (además de `periodo` y `region` en la
  muestra comprobada): son un objeto de cada uno de los 32 yacimientos sin
  fecha parseable. El relleno anterior los saltaba a propósito
  (`if bp_max is None: continue`). Son `null` o `0` según el objeto, con el
  mismo significado. No afecta a las columnas nuevas y queda fuera de este
  cambio.
- El notebook `notebooks/indexing/rag-indexacion-mejorado.ipynb` llegaba a
  fallar `nbformat.validate` **antes** de este cambio: tenía `nbformat_minor=4`
  con 4 celdas que ya llevaban `id`, y 2 celdas de código sin
  `outputs`/`execution_count`. Se normalizó a `minor=5` y se completaron esos
  campos; `validate` y `compile()` pasan ahora (16/16 celdas de código).
