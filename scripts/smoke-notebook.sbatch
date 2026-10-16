#!/bin/bash
#SBATCH -J smoke-notebook
#SBATCH -p medium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# Usually auto-launched by start_rag.sh once services are READY (no
# GPU-queue wait billed here). Presupuesto grande por petición del usuario
# (2026-09-29): smoke rápido (2 estrategias, SMOKE_N=3, CON grafo) ≈ 2 h,
# pero con margen holgado. medium admite hasta 3 días.
#SBATCH --time=24:00:00
#SBATCH -o /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/smoke-notebook-%j.out
#SBATCH -e /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/smoke-notebook-%j.err

# ============================================================
# IDEArq RAG — execute the evaluation notebook with SMOKE_ONLY=True.
#
# Does NOT start Weaviate/Ollama: waits for the persistent services
# (rag-services.sbatch / start_rag.sh) to be ready, then runs
# jupyter nbconvert --execute on the notebook.
#
# The notebook already has SMOKE_ONLY=True in cell 1 (3 questions
# + graph smoke). Output goes to logs/, source notebook is untouched.
# ============================================================

set -uo pipefail

export HOME="${HOME:-/home/csic/usi/eag}"
export STORE="${STORE:-/mnt/netapp1/Store_CSIC/home/csic/usi/eag}"
export LUSTRE="${LUSTRE:-/mnt/lustre/scratch/nlsas/home/csic/usi/eag}"

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi

LOGS="$LUSTRE/idearq-rag/logs"
mkdir -p "$LOGS"

log() { echo "[$(date -Is)] $*"; }

# El notebook maestro vive en STORE junto a src/ (cell 1 deriva
# PROJECT_ROOT del cwd del kernel: sin src/ hermano la importación
# `from src...` revienta — el ejemplar de Lustre no tiene src/).
NOTEBOOK="$STORE/idearq-rag/notebooks/evaluation/rag-evaluacion-hibrida-multiquery.ipynb"
OUT_NB="$LOGS/smoke-notebook-${SLURM_JOB_ID:-manual}.ipynb"
SERVICES_ENV="$LUSTRE/idearq-rag/services.env"
JUPYTER="$LUSTRE/idearq-rag/env_rag/bin/jupyter"
PY="$LUSTRE/idearq-rag/env_rag/bin/python"

log "=========================================================="
log "smoke-notebook: node=$(hostname -s) job=${SLURM_JOB_ID:-?}"
log "notebook=$NOTEBOOK"
log "output=$OUT_NB"
log "=========================================================="

if [ ! -x "$JUPYTER" ] || [ ! -x "$PY" ]; then
    log "ERROR: env_rag jupyter/python not found under $LUSTRE/idearq-rag/env_rag"
    exit 2
fi

if [ ! -f "$NOTEBOOK" ]; then
    log "ERROR: notebook not found: $NOTEBOOK"
    exit 2
fi

# ------------------------------------------------------------
# 1. Wait for persistent services (do NOT start them here).
#    Two Weaviates on the same vectordb would corrupt it.
#
#    services.env is RE-READ on every attempt: rag-services may
#    start on a different node and rewrite it while we wait
#    (a stale host would make us poll a dead node forever).
# ------------------------------------------------------------
if [ ! -f "$SERVICES_ENV" ]; then
    log "ERROR: $SERVICES_ENV missing — start rag-services.sbatch first"
    exit 3
fi

# $1 = service name, $2 = key in services.env (e.g. WEAVIATE_URL),
# $3 = path suffix (e.g. /v1/.well-known/ready), $4 = deadline seconds
wait_ready() {
    local name="$1" key="$2" path="$3" deadline="$4"
    local tries=0 url=""
    while [ "$tries" -lt "$deadline" ]; do
        # shellcheck disable=SC1090
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

# Normally start_rag.sh has already written a fresh services.env when
# it auto-submits this job, so these waits return in seconds; they only
# guard the manual-submission path (services.env re-read covers host race).
wait_ready "Weaviate" "WEAVIATE_URL" "/v1/.well-known/ready" 3600 || exit 4
wait_ready "Ollama"   "OLLAMA_URL"   "/api/tags"              600  || exit 5

# ------------------------------------------------------------
# 2. Execute the notebook (SMOKE_ONLY=True already in cell 1).
#    Executed copy goes to logs/; source stays pristine.
# ------------------------------------------------------------
log "exec: jupyter nbconvert --execute -> $OUT_NB"
"$JUPYTER" nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=43200 \
    --output "$OUT_NB" \
    "$NOTEBOOK"
rc=$?

log "nbconvert exit=$rc"
if [ "$rc" -ne 0 ]; then
    log "ERROR: notebook smoke failed (exit $rc)"
    log "Inspect: $OUT_NB and $LOGS/smoke-notebook-${SLURM_JOB_ID:-?}.*"
    exit "$rc"
fi

log "OK: notebook smoke finished -> $OUT_NB"
exit 0
