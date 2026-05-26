#!/bin/bash
# Detached watcher for the H_3 m=3 exhaustive triple search.
#
# Polls for sa_logs/triple_search/result-h3-m3.json. When it appears, runs
# post_triple_search.sh h3 3 (validate → promote → PR → auto-merge), then
# stops. Does NOT auto-launch the Norrell search; that requires a separate
# human go-ahead per the project's compute-risk policy.
#
# Usage:
#   setsid -f bash scripts/research/orbit_paper/watch_h3_then_promote.sh
#
# Logs all output to sa_logs/triple_search/watcher-h3-m3.log.
# On successful promotion, writes a sentinel file:
#   sa_logs/triple_search/watcher-h3-m3.DONE
# On error, writes:
#   sa_logs/triple_search/watcher-h3-m3.FAILED

set -uo pipefail

cd /home/vrusso/Projects/uf/stabrank

RESULT="sa_logs/triple_search/result-h3-m3.json"
LOG="sa_logs/triple_search/watcher-h3-m3.log"
DONE="sa_logs/triple_search/watcher-h3-m3.DONE"
FAILED="sa_logs/triple_search/watcher-h3-m3.FAILED"
POLL_SECONDS=300  # 5 minutes; search ETA is ~60h so this is plenty fast

# Clean any prior sentinels
rm -f "${DONE}" "${FAILED}"

echo "[$(date -Iseconds)] watcher started, polling for ${RESULT} every ${POLL_SECONDS}s" >> "${LOG}"
echo "[$(date -Iseconds)] PID=$$" >> "${LOG}"

while [ ! -s "${RESULT}" ]; do
    sleep "${POLL_SECONDS}"
done

echo "[$(date -Iseconds)] result file detected: $(stat -c '%s' "${RESULT}") bytes" >> "${LOG}"
echo "[$(date -Iseconds)] running post_triple_search.sh h3 3" >> "${LOG}"

if bash scripts/research/orbit_paper/post_triple_search.sh h3 3 >> "${LOG}" 2>&1; then
    echo "[$(date -Iseconds)] promotion succeeded" >> "${LOG}"
    touch "${DONE}"
    exit 0
else
    rc=$?
    echo "[$(date -Iseconds)] promotion FAILED with exit code ${rc}" >> "${LOG}"
    touch "${FAILED}"
    exit "${rc}"
fi
