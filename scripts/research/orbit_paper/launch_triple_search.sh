#!/bin/bash
# Launch a new triple-search run in the background (12 cores, niced).
# Detaches so the parent shell can exit without killing the search.
#
# Usage: launch_triple_search.sh <orbit> <m>
# Example: launch_triple_search.sh h3 3

set -euo pipefail

ORBIT="${1:?Usage: $0 <orbit> <m>}"
M="${2:?Usage: $0 <orbit> <m>}"

cd /home/vrusso/Projects/uf/stabrank
mkdir -p sa_logs/triple_search

LOG="sa_logs/triple_search/${ORBIT}-m${M}.log"
CK="sa_logs/triple_search/ck-${ORBIT}-m${M}.json"
OUT="sa_logs/triple_search/result-${ORBIT}-m${M}.json"

# Use setsid -f to fully detach so closing the parent terminal won't kill it.
NUMBA_NUM_THREADS=12 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid -f nice -n 19 \
    uv run --with numba python \
    scripts/research/orbit_paper/exhaustive_triple_search_numba.py \
    --orbit "${ORBIT}" --m "${M}" --batch-size 200000 \
    --checkpoint "${CK}" \
    --output "${OUT}" \
    --log-every 10000000 \
    >"${LOG}" 2>&1

echo "[launch_triple_search] launched ${ORBIT} m=${M} (log: ${LOG})"
