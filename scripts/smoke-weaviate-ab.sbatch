#!/bin/bash
#SBATCH -J smoke-ab-graph-wv
#SBATCH -p viz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# 5 questions x (SPARQL + 2 BM25 probes x 2 literals + hybrid + LLM answer).
# Loading MiniLM and one Qwen3-4B answer per question dominates the time.
#SBATCH --time=01:00:00
#SBATCH -o /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/smoke-ab-%j.out
#SBATCH -e /mnt/lustre/scratch/nlsas/home/csic/usi/eag/idearq-rag/logs/smoke-ab-%j.err

# ============================================================
# IDEArq RAG — A/B smoke: the five graph-only questions answered twice.
#
#   side A = RDF graph (SPARQL, deterministic; scripts/smoke_graph.py)
#   side B = Weaviate (BM25 probe for each answer literal + hybrid retrieval
#            + LLM answer)
#
# The answers live ONLY in the graph, so this measures how much of the graph's
# knowledge the vector index can reproduce. Side B failing to find an answer is
# a finding, not a job failure: the script still exits 0 when side A passes.
#
# Submitted to viz because no A100 was free (every a100 node had
# gres/gpu=2 already allocated; only t4-8/9/10 were idle).
# Does NOT start Weaviate/Ollama — rag-services.sbatch owns those.
# ============================================================

set -uo pipefail

export HOME="${HOME:-/home/csic/usi/eag}"
export STORE="${STORE:-/mnt/netapp1/Store_CSIC/home/csic/usi/eag}"
export LUSTRE="${LUSTRE:-/mnt/lustre/scratch/nlsas/home/csic/usi/eag}"
# Without this Python block-buffers stdout when it is not a TTY and the log
# stays empty until the process exits — progress becomes invisible.
export PYTHONUNBUFFERED=1

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi

log() { echo "[$(date -Is)] $*"; }

SCRIPT="$STORE/idearq-rag/scripts/smoke_weaviate_ab.py"
SERVICES_ENV="$LUSTRE/idearq-rag/services.env"
PY="$LUSTRE/idearq-rag/env_rag/bin/python"
CORPUS="$LUSTRE/idearq-rag/pdftext"

log "=========================================================="
log "smoke-ab: node=$(hostname -s) job=${SLURM_JOB_ID:-?}"
log "script=$SCRIPT"
log "=========================================================="

if [ ! -x "$PY" ]; then
    PY="$STORE/idearq-rag/env_rag/bin/python"
fi
if [ ! -x "$PY" ]; then
    log "ERROR: env_rag python not found under LUSTRE or STORE"
    exit 2
fi
if [ ! -f "$SCRIPT" ]; then
    log "ERROR: script not found: $SCRIPT"
    exit 2
fi

if [ -f "$SERVICES_ENV" ]; then
    # shellcheck disable=SC1090
    source "$SERVICES_ENV"
    log "services: WEAVIATE_URL=${WEAVIATE_URL:-?} OLLAMA_URL=${OLLAMA_URL:-?}"
else
    log "WARN: $SERVICES_ENV missing — script will fall back to env/.env"
fi

# The corpus scan is evidence ("this literal is in no PDF"), not a requirement:
# run_fase/smoke_graph fall back gracefully when it is absent.
ARGS=()
if [ -d "$CORPUS" ]; then
    ARGS+=(--corpus "$CORPUS")
else
    log "WARN: corpus not found at $CORPUS — 'corpus hits' column will be skipped"
fi

log "exec: $PY $SCRIPT ${ARGS[*]:-}"
cd "$STORE/idearq-rag" || exit 2
exec "$PY" "$SCRIPT" ${ARGS[@]+"${ARGS[@]}"}
