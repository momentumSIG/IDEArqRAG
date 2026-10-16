#!/bin/bash
#SBATCH -J eval-fase
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/eval-fase-%j.out
#SBATCH -e /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/eval-fase-%j.err

# ============================================================
# IDEArq RAG — one evaluation phase (run_fase.py) on Slurm.
#
# --time is NOT fixed here: override at submit time, e.g.
#   sbatch --time=02:00:00 scripts/eval-fase.sbatch
# CESGA derives QOS/partition FROM --time (JobSubmitPlugins =
# require_timelimit,lua). Do NOT pass --partition/--qos by hand:
# explicit --qos fails with "Invalid qos specification".
# --mem is mandatory or the job is rejected by accounting policy.
#
# Phase knobs — export before submit (or edit defaults below):
#   export FASE=A SMOKE=1 RESUME=1
#   export BASE_STRATEGY=hybrid_0.5 GRAPH=0 WORKERS=1
# ============================================================

set -uo pipefail

export HOME="${HOME:-/home/csic/usi/eag}"
export STORE="${STORE:-/mnt/netapp1/Store_CSIC/home/csic/usi/eag}"
export LUSTRE="${LUSTRE:-/mnt/lustre/scratch/nlsas/home/csic/usi/eag}"

# Modules (same pattern as rag-services.sbatch; harmless if already loaded).
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi

mkdir -p "$LUSTRE/idearq-rag/logs"

log() { echo "[$(date -Is)] $*"; }

log "=========================================================="
log "eval-fase: node=$(hostname -s) job=${SLURM_JOB_ID:-?}"
log "STORE=$STORE"
log "LUSTRE=$LUSTRE"
log "FASE=${FASE:-A} SMOKE=${SMOKE:-0} RESUME=${RESUME:-0}"
log "BASE_STRATEGY=${BASE_STRATEGY:-} GRAPH=${GRAPH:-} WORKERS=${WORKERS:-1}"
log "=========================================================="

# ------------------------------------------------------------
# Persistent services: do NOT start_rag.sh here.
# Weaviate + Ollama are kept alive by rag-services.sbatch (7d long job).
# Starting a second Weaviate on the same vectordb would corrupt it.
# Residual Weaviate guard: run_fase.py health-checks WEAVIATE_URL and
# refuses to start while it is not ready; it prints remediation
# (scancel the residual non-persistent job) and does NOT scancel by
# itself unless you pass --kill-residual-weaviate with a stale PID.
# ------------------------------------------------------------
SERVICES_ENV="$LUSTRE/idearq-rag/services.env"
if [ -f "$SERVICES_ENV" ]; then
    # shellcheck disable=SC1090
    source "$SERVICES_ENV"
    log "services.env: WEAVIATE_URL=${WEAVIATE_URL:-?} OLLAMA_URL=${OLLAMA_URL:-?}"
else
    log "WARN: $SERVICES_ENV missing — run_fase.py will fall back to env/.env"
fi

# ------------------------------------------------------------
# Python: the project virtualenv on Lustre (falls back to STORE copy).
# ------------------------------------------------------------
PY="$LUSTRE/idearq-rag/env_rag/bin/python"
if [ ! -x "$PY" ]; then
    PY="$STORE/idearq-rag/env_rag/bin/python"
fi
if [ ! -x "$PY" ]; then
    log "ERROR: env_rag python not found under $LUSTRE or $STORE"
    exit 2
fi

ARGS=(--fase "${FASE:-A}")
if [ "${SMOKE:-0}" = "1" ]; then
    ARGS+=(--smoke)
fi
if [ "${RESUME:-0}" = "1" ]; then
    ARGS+=(--resume)
fi
if [ -n "${BASE_STRATEGY:-}" ]; then
    ARGS+=(--base-strategy "$BASE_STRATEGY")
fi
# GRAPH only applies to fase G: 1 = --graph, 0 = --no-graph, unset = both.
case "${GRAPH:-}" in
    1) ARGS+=(--graph) ;;
    0) ARGS+=(--no-graph) ;;
    *) : ;;
esac
if [ -n "${WORKERS:-}" ]; then
    ARGS+=(--workers "$WORKERS")
fi

log "exec: $PY $STORE/idearq-rag/scripts/run_fase.py ${ARGS[*]}"
exec "$PY" "$STORE/idearq-rag/scripts/run_fase.py" "${ARGS[@]}"
