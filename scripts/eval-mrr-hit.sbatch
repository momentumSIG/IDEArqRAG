#!/bin/bash
#SBATCH -J eval-mrr-hit
#SBATCH -p medium
#SBATCH -N 1
#SBATCH -n 1
# 24 CPUs: con 8 el job 10450431 no terminó en 12 h (TIMEOUT sin escribir nada).
# Ollama escala con los hilos, así que esto acorta la pared, no solo el margen.
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -o /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/eval-mrr-hit-%j.out
#SBATCH -e /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/eval-mrr-hit-%j.err
#SBATCH --export=ALL

# ============================================================
# IDEArq RAG — evaluación completa (32 preguntas) con hit@k y MRR.
#
# RONDA 1 (esta): CTX_MODE=no_distance
#   El contexto lleva las COORDENADAS de los yacimientos pero NO los km ya
#   calculados, para ver si el LLM hace el cálculo de la distancia él mismo.
#   AVISO IMPORTANTE: hit@k y MRR NO cambian entre rondas de CTX_MODE — son
#   métricas de recuperación y no leen el prompt. La diferencia entre rondas se
#   mira en la columna `answer` del CSV (la respuesta generada), no en hit/MRR.
#
# Estrategias por defecto: vector,hybrid_0.5 (comparación limpia). Ninguna usa
# multiquery, así que NO se necesita el LLM generador de variantes. Se pueden
# cambiar en el envío:
#   sbatch --export=ALL,EVAL_STRATEGIES=vector,hybrid_0.5,hybrid_0.3 scripts/eval-mrr-hit.sbatch
# Si se incluye alguna mq_*, esa ronda SÍ pide variantes a Ollama.
#
# GENERACIÓN SÍ, JUEZ NO: hace falta el LLM para tener la columna `answer`, pero
# no el juez RAGAS. Por eso el job arranca su PROPIO Ollama en 127.0.0.1:$PORT
# (el compartido de services.env se midió a 0,06-0,4 tok/s con dos runners
# residentes; el privado dio 7,3 tok/s). El notebook lo usa vía EVAL_OLLAMA_URL.
#
# Weaviate NO se arranca aquí: se espera al servicio persistente (rag-services).
#
# SUBMITIR SOLO CUANDO EL USUARIO LO INDIQUE. Este archivo no se envía solo.
# ============================================================

set -uo pipefail

export HOME="${HOME:-/home/csic/usi/eag}"
export STORE_LUSTRE="${STORE_LUSTRE:-/mnt/lustre/scratch/nlsas/home/csic/usi/eag}"
export STORE="${STORE:-/mnt/netapp1/Store_CSIC/home/csic/usi/eag}"

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi

ROOT="$STORE_LUSTRE/idearq-rag"
REPO="$STORE/idearq-rag"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

log() { echo "[$(date -Is)] $*"; }

# Los pesos de los embeddings viven en netapp1 y los nodos de cálculo NO tienen
# internet: sin esta ruta explícita el job intentaría descargarlos y moriría.
export HF_HOME="${HF_HOME:-$REPO/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
if [ ! -d "$HF_HUB_CACHE/models--Alibaba-NLP--gte-multilingual-base" ]; then
    log "ERROR: no encuentro gte-multilingual-base en $HF_HUB_CACHE"
    exit 1
fi
log "HF_HUB_CACHE=$HF_HUB_CACHE"

NOTEBOOK="$REPO/notebooks/evaluation/rag-evaluacion-hibrida-multiquery.ipynb"
OUT_NB="$LOGS/eval-mrr-hit-${SLURM_JOB_ID:-manual}.ipynb"
SERVICES_ENV="$ROOT/services.env"
JUPYTER="$ROOT/env_rag/bin/jupyter"
PY="$ROOT/env_rag/bin/python"
OLLAMA_BIN="$ROOT/ollama/bin/ollama"
OLLAMA_MODELS_DIR="$REPO/ollama/models"   # los blobs viven en netapp1, NO en lustre
PORT=11435
OLLAMA_HOME_DIR="${TMPDIR:-/tmp}/ollama-home-${SLURM_JOB_ID:-manual}"

# ── Palancas de la ronda ────────────────────────────────────────────────────
export EVAL_SMOKE_ONLY=false        # las 32 preguntas
export EVAL_RUN_RAGAS=false         # sin juez Qwen3-14B
export EVAL_RUN_GENERATION=true     # SÍ queremos la respuesta en el CSV
export EVAL_NUM_PREDICT=1024       # tope de tokens de respuesta (antes -1 = sin tope; una divagación llegaba a ~2 h)
export CTX_MODE=no_distance         # coordenadas sí, km NO: que lo calcule el LLM
export GEO_USE_CITY_COORDS=0        # ronda 1 SIN la tabla de ciudades de referencia:
                                    # sin punto de referencia no hay distancia que el
                                    # código pueda calcular, así que el LLM lo hace todo
export EVAL_STRATEGIES="${EVAL_STRATEGIES:-vector,hybrid_0.5}"

log "=========================================================="
log "eval-mrr-hit: node=$(hostname -s) job=${SLURM_JOB_ID:-?}"
log "notebook=$NOTEBOOK"
log "output=$OUT_NB"
log "CTX_MODE=$CTX_MODE | GEO_USE_CITY_COORDS=${GEO_USE_CITY_COORDS:-1} | EVAL_STRATEGIES=$EVAL_STRATEGIES"
log "generacion=ON | ragas=OFF | smoke=OFF | num_predict=${EVAL_NUM_PREDICT}"
log "=========================================================="

if [ ! -x "$JUPYTER" ] || [ ! -x "$PY" ] || [ ! -x "$OLLAMA_BIN" ]; then
    log "ERROR: falta env_rag/jupyter, env_rag/python o el binario de ollama en $ROOT"
    exit 2
fi
if [ ! -f "$NOTEBOOK" ]; then
    log "ERROR: notebook not found: $NOTEBOOK"
    exit 2
fi
if [ ! -f "$SERVICES_ENV" ]; then
    log "ERROR: $SERVICES_ENV missing — start rag-services.sbatch first"
    exit 3
fi
if [ ! -d "$OLLAMA_MODELS_DIR/manifests" ]; then
    log "ERROR: $OLLAMA_MODELS_DIR no tiene manifests/ (los modelos están en netapp1)"
    exit 3
fi

# ── Ollama privado del job (no toca el compartido) ──────────────────────────
mkdir -p "$OLLAMA_HOME_DIR"
export OLLAMA_HOME="$OLLAMA_HOME_DIR"
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_VULKAN=false
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=15m

log "arrancando ollama privado en $OLLAMA_HOST (modelos: $OLLAMA_MODELS)"
"$OLLAMA_BIN" serve > "$LOGS/ollama-private-${SLURM_JOB_ID:-manual}.log" 2>&1 &
OPID=$!

ready=0
for _ in $(seq 1 90); do
    if curl -sf --max-time 3 "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1; then
        ready=1; break
    fi
    sleep 2
done
log "ollama privado listo=$ready pid=$OPID"
if [ "$ready" != "1" ]; then
    log "ERROR: el ollama privado no respondió"; kill "$OPID" 2>/dev/null || true; exit 4
fi
if ! curl -s --max-time 5 "http://127.0.0.1:$PORT/api/tags" | grep -q "Qwen3-4B-Instruct-2507"; then
    log "ERROR: Qwen3-4B-Instruct-2507 no está en $OLLAMA_MODELS"
    kill "$OPID" 2>/dev/null || true; exit 4
fi

# El notebook resuelve EVAL_OLLAMA_URL antes que services.env.
export EVAL_OLLAMA_URL="http://127.0.0.1:$PORT"

# ── Espera a Weaviate (se relee services.env: el nodo puede cambiar) ────────
wait_ready() {
    local name="$1" key="$2" path="$3" deadline="$4"
    local tries=0 url=""
    while [ "$tries" -lt "$deadline" ]; do
        url="$(source "$SERVICES_ENV" 2>/dev/null; eval echo "\${$key:-}")"
        if [ -n "$url" ] && curl -s --connect-timeout 2 --max-time 3 "$url$path" >/dev/null 2>&1; then
            log "$name READY ($url$path)"
            return 0
        fi
        tries=$((tries + 10))
        log "waiting for $name (${tries}s/${deadline}s, host=${url:-?}) ..."
        sleep 10
    done
    log "ERROR: $name not ready after ${deadline}s (last url=${url:-?}$path)"
    return 1
}
wait_ready "Weaviate" "WEAVIATE_URL" "/v1/.well-known/ready" 3600 || { kill "$OPID" 2>/dev/null || true; exit 5; }

# ── Ejecutar el notebook ────────────────────────────────────────────────────
log "exec: jupyter nbconvert --execute -> $OUT_NB"
"$JUPYTER" nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=39600 \
    --output "$OUT_NB" \
    "$NOTEBOOK"
rc=$?

log "nbconvert exit=$rc"
kill "$OPID" 2>/dev/null || true
if [ "$rc" -ne 0 ]; then
    log "ERROR: eval-mrr-hit failed (exit $rc)"
    log "Inspect: $OUT_NB and $LOGS/eval-mrr-hit-${SLURM_JOB_ID:-?}.*"
    exit "$rc"
fi
log "OK: eval-mrr-hit finished -> $OUT_NB"
log "CSV esperado en $REPO/data/results/ (tag full, con hit_at_k y reciprocal_rank)"
exit 0
