# data/geo — coordenadas de referencia

## `capitales_provincias.csv`

Capitales de provincia (y ciudades uniprovinciales) de España. Alimenta los puntos de
referencia de `superlative_space` (preguntas de distancia) y, más adelante, al subagente
GIS (`odd/tasks/gis-subagent.md`).

- **Origen**: Nomenclátor Geográfico Básico de España (IGN) — tabla subida por el usuario,
  2026-10-15.
- **Transformaciones aplicadas**: separador decimal coma → punto; nombres reparados a mano
  (el pegado original venía con mojibake CP850→CP1252: «Ja‚n»→«Jaén», «M laga»→«Málaga»,
  «A Coru¤a»→«A Coruña»…).
- **Columnas**: `id_ign, entidad, lat_etrs89, lon_etrs89, x_utm_etrs89, y_utm_etrs89`
  (grados decimales; UTM en metros; ETRS89; REGCAN95 en Canarias — irrelevante para
  distancias).
- **Uso**: `GEO_CITY_COORDS_FILE=data/geo/capitales_provincias.csv` (+ `GEO_USE_CITY_COORDS`
  sin poner a `0`) en `src/geo_route.py`. Las filas del fichero sobreescriben las entradas
  con el mismo nombre y añaden las nuevas; la tabla interna queda como respaldo.
