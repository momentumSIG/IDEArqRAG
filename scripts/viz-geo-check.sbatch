#!/bin/bash
#SBATCH -J viz-geo-check
#SBATCH -p viz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# viz tops out at 8 h (sinfo: viz up 8:00:00). The 2026-10-01 run (job
# 10180170, same 8 questions, single strategy) finished in 3h41m, so the full
# budget is headroom for a slower judge, not a guess.
#SBATCH --time=08:00:00
#SBATCH -o /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/viz-geo-check-%j.out
#SBATCH -e /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/viz-geo-check-%j.err

# ============================================================
# IDEArq RAG — geo-route smoke on the viz partition.
#
# Re-run of the 8-question smoke that produced job 10180170's CSV, now with:
#   * fix 1 (empty filter is visible: route -> VACIO, n_docs=0, no top-k fallback)
#   * fix 3 (lookup_by_date filters on bp_values instead of bp_max)
#   * gold label #3 rewritten (Villaricos question)
#   * cell 7 pins the SAME eight questions as 10180170 (all 32 labels now carry
#     a `question`, so the old selector would have picked 32 and blown the limit)
#
# Does NOT start Weaviate/Ollama: those stay alive under rag-services.sbatch
# (job 10103455). Two Weaviates on one vectordb would corrupt it.
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

NOTEBOOK="$STORE/idearq-rag/notebooks/evaluation/viz-geo-check.ipynb"
OUT_NB="$LOGS/viz-geo-check-${SLURM_JOB_ID:-manual}.ipynb"
SERVICES_ENV="$LUSTRE/idearq-rag/services.env"
JUPYTER="$LUSTRE/idearq-rag/env_rag/bin/jupyter"

log "=========================================================="
log "viz-geo-check: node=$(hostname -s) job=${SLURM_JOB_ID:-?}"
log "notebook=$NOTEBOOK"
log "output=$OUT_NB"
log "=========================================================="

if [ ! -x "$JUPYTER" ]; then
    log "ERROR: env_rag jupyter not found at $JUPYTER"
    exit 2
fi
if [ ! -f "$NOTEBOOK" ]; then
    log "ERROR: notebook not found: $NOTEBOOK"
    exit 2
fi

# ------------------------------------------------------------
# 1. Wait for the persistent services (never start them here).
#    services.env is re-read every attempt: rag-services may have moved nodes.
# ------------------------------------------------------------
if [ ! -f "$SERVICES_ENV" ]; then
    log "ERROR: $SERVICES_ENV missing — start rag-services.sbatch first"
    exit 3
fi

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

wait_ready "Weaviate" "WEAVIATE_URL" "/v1/.well-known/ready" 3600 || exit 4
wait_ready "Ollama"   "OLLAMA_URL"   "/api/tags"              600  || exit 5

# shellcheck disable=SC1090
source "$SERVICES_ENV"
log "services: WEAVIATE_URL=${WEAVIATE_URL:-?} OLLAMA_URL=${OLLAMA_URL:-?}"

# ------------------------------------------------------------
# 2. Execute. The kernel cwd becomes the notebook's directory, so cell 1
#    resolves PROJECT_ROOT to $STORE/idearq-rag (where src/ and data/ live).
# ------------------------------------------------------------
log "exec: jupyter nbconvert --execute -> $OUT_NB"
"$JUPYTER" nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=25200 \
    --output "$OUT_NB" \
    "$NOTEBOOK"
rc=$?

log "nbconvert exit=$rc"
if [ "$rc" -ne 0 ]; then
    log "ERROR: notebook smoke failed (exit $rc)"
    log "Inspect: $OUT_NB"
    exit "$rc"
fi

log "OK: finished -> $OUT_NB"
log "CSV: $STORE/idearq-rag/data/results/eval_ablation_retrieval_viz_geocheck_*.csv"
exit 0
