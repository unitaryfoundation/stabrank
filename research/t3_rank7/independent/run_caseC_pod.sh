#!/bin/bash
# Launch (or resume) the case-C scan on RunPod pod 3 under nice and the compute ledger.
# Usage: bash run_caseC_pod.sh [WORKERS]   (default 6; the pod's cgroup quota is 7.65 cores
# and the qldpc-search agent's job shares it, so do not raise this without checking load).
# caseC.py resumes from logs/caseC_progress.jsonl on its own; this script only records the
# session start so status_c.py can measure the rate of the current session.
set -e
W=${1:-6}
cd /root/work/genesis/stabrank-caseC/rank8
source /root/work/genesis/venv/bin/activate
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if pgrep -f "python caseC.py" > /dev/null; then
  echo "caseC.py is already running:"; pgrep -af "python caseC.py"; exit 1
fi
python - <<'EOF'
import json, os, time
steps = 0
if os.path.exists("logs/caseC_progress.jsonl"):
    for line in open("logs/caseC_progress.jsonl"):
        try: steps += json.loads(line)["steps"]
        except ValueError: pass
json.dump(dict(start_time=time.time(), steps_at_start=steps), open("logs/caseC_session.json", "w"))
print(f"session start recorded, {steps:.4e} steps already done")
EOF
nohup nice -n 10 python /root/work/genesis/ledger/compute_ledger.py \
  --ledger /root/work/genesis/ledger/compute_pod.jsonl wrap \
  --tool stabrank --provenance llm-agent --cwd /root/work/genesis/stabrank-caseC/rank8 \
  --notes "RunPod pod 3 (host edb0f1689805, cgroup quota 7.65 cores shared with the qldpc-search agent's CPU job); case C of the T3 m=3 rank-7 exclusion, three-pivot scan under G' of order 2916, $W workers at nice 10, resumable from logs/caseC_progress.jsonl" \
  --work workers=$W --work nice=10 --work tasks=326323 --work steps_predicted=53047312306749 \
  --artifact logs/caseC_progress.jsonl --artifact logs/caseC.json --artifact logs/caseC.out \
  -- python caseC.py "$W" >> logs/caseC.out 2>&1 &
echo "launched wrapper pid $!"
sleep 5
pgrep -af "python caseC.py" || echo "caseC.py did not start; see logs/caseC.out"
