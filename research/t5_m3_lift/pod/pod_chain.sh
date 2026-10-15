#!/bin/bash
# Chain for the slice-and-lift lower bound chi(|T5>^3) >= 6 on the RunPod pod
# (32 vCPU, Ubuntu 20.04, gcc 9.4, uv-managed Python 3.12). Nothing here is
# run by hand on the laptop; the user launches it when the pod is free.
#
# Setup (fresh pod, disk erased on stop; the branch t5-m3-lift is local and
# never pushed, so it travels as a bundle on top of the public main):
#   cd /Users/vincent.russo/Projects/genesis/stabrank
#   git bundle create /tmp/t5-m3-lift.bundle origin/main..t5-m3-lift
#   /tmp/podscp.sh /tmp/t5-m3-lift.bundle root@213.173.105.71:/root/t5-m3-lift.bundle
#   /tmp/podssh.sh 'mkdir -p /root/logs && (command -v g++-10 >/dev/null || (apt-get update -qq && apt-get install -y -qq gcc-10 g++-10 >/dev/null)) && g++-10 --version | head -1'
#   /tmp/podssh.sh 'cd /root && git clone -q https://github.com/unitaryfoundation/stabrank stabrank-t5m3 && cd stabrank-t5m3 && git fetch -q /root/t5-m3-lift.bundle t5-m3-lift:t5-m3-lift && git checkout -q t5-m3-lift && git log --oneline -1'
#   /tmp/podssh.sh 'export PATH=$HOME/.local/bin:$PATH; cd /root/stabrank-t5m3 && CC=gcc-10 CXX=g++-10 uv sync --extra challenge 2>&1 | tail -3'
# The kernel does not build with gcc 9: cpp/include/stabrank/{linalg,polynomial}.hpp
# include <span> and cpp/src/{clifford,polynomial,fidelity}.cpp include <numbers>,
# both C++20 headers that libstdc++ 9 lacks (first shipped with GCC 10).
#
# Launch (lift only, about 30 to 40 minutes: controls, the batch-73 rate probe,
# the aggregate's two seeded re-runs, the lift stage, the certificate under the
# verifier):
#   /tmp/podssh.sh 'setsid nohup bash /root/stabrank-t5m3/research/t5_m3_lift/pod/pod_chain.sh > /root/logs/t5m3_chain.log 2>&1 < /dev/null & disown'
# Launch with the census re-run (adds the 74 batches at WORKERS processes into a
# scratch directory and compares every deterministic hash with the committed
# record; about 13 to 15 CPU-hours, 35 to 45 minutes of wall time at 24 workers):
#   /tmp/podssh.sh 'WITH_CENSUS=1 setsid nohup bash /root/stabrank-t5m3/research/t5_m3_lift/pod/pod_chain.sh > /root/logs/t5m3_chain.log 2>&1 < /dev/null & disown'
# Watch:
#   /tmp/podssh.sh 'grep -E "^T5M3_" /root/logs/t5m3_chain.log; tail -3 /root/logs/t5m3_chain.log'
# Markers (anchored, grep ^MARKER): T5M3_START, T5M3_KERNEL_OK, T5M3_CONTROLS_DONE,
# T5M3_PROBE_DONE, T5M3_CENSUS_DONE (with WITH_CENSUS=1), T5M3_AGG_EXIT,
# T5M3_LIFT_EXIT, T5M3_CERT_EXIT, T5M3_SUMMARY, T5M3_CHAIN_DONE.
# Copy back afterwards:
#   rsync -avz -e "ssh -p 22904 -i ~/.ssh/id_ed25519" root@213.173.105.71:/root/stabrank-t5m3/research/t5_m3_lift/results/ research/t5_m3_lift/results/
#   rsync -avz -e "ssh -p 22904 -i ~/.ssh/id_ed25519" root@213.173.105.71:/root/stabrank-t5m3/research/t5q_m2_rank5/batch_manifest.json research/t5q_m2_rank5/
#   rsync -avz -e "ssh -p 22904 -i ~/.ssh/id_ed25519" root@213.173.105.71:/root/logs/t5m3_chain.log research/t5_m3_lift/results/pod/
set -u
REPO=${REPO:-/root/stabrank-t5m3}
WORKERS=${WORKERS:-24}
WITH_CENSUS=${WITH_CENSUS:-0}
SEED=${SEED:-20261001}
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"
OUT=research/t5_m3_lift/results
POD=$OUT/pod
mkdir -p "$POD" /root/logs
echo "T5M3_START $(date -u +%FT%TZ) host $(hostname) commit $(git rev-parse --short HEAD) workers $WORKERS with_census $WITH_CENSUS"

# the compiled 5-cover kernel must import (built with gcc-10 at setup)
if ! $PY -c "from stabrank.stabrank_core import cover5_pair" 2> "$POD/kernel_import.err"; then
  echo "T5M3_KERNEL_FAILED $(date -u +%FT%TZ): $(tail -1 "$POD/kernel_import.err")"
  echo "T5M3_CHAIN_DONE $(date -u +%FT%TZ) aborted"
  exit 1
fi
echo "T5M3_KERNEL_OK $(date -u +%FT%TZ)"

# controls of the census pipeline at this commit, then the lift controls
for C in control-pattern control-hash control-m1 control-planted control-rank4 control-reference; do
  nice -n 19 $PY -u research/t5q_m2_rank5/driver.py $C > "$POD/census_$C.log" 2>&1; echo "census $C rc=$?"
done
for C in control-stabtest control-basis control-rank3 control-product control-t3-basis control-t3-m2 control-n-m2; do
  nice -n 19 $PY -u research/t5_m3_lift/lift.py $C > "$POD/lift_$C.log" 2>&1; echo "lift $C rc=$?"
done
echo "T5M3_CONTROLS_DONE $(date -u +%FT%TZ)"

# rate probe: batch 73 (the smallest) from scratch into a scratch directory,
# its deterministic hash against the committed record
mkdir -p "$POD/probe"
nice -n 19 $PY -u research/t5q_m2_rank5/batch.py 73 --out-dir "$POD/probe" --force --max-seconds 3600 \
  > "$POD/probe/batch_73.log" 2>&1; echo "probe rc=$?"
$PY - <<'EOF'
import json
a = json.load(open("research/t5q_m2_rank5/results/batch_73.json"))
b = json.load(open("research/t5_m3_lift/results/pod/probe/batch_73.json"))
same = a["deterministic_sha256"] == b["deterministic_sha256"]
part = json.load(open("research/t5q_m2_rank5/partition.json"))
geo = part["batch_geometry"][73]
m2 = sum(u[2] ** 2 for u in part["units"][geo["start"]:geo["end"]])
rate = b["kernel_s"] / m2
print(f"probe batch 73: hash {'matches' if same else 'DIFFERS'}, kernel {b['kernel_s']:.0f}s, "
      f"{rate:.3g} s per M^2, projected census {rate * part['cost_model']['sum_m2'] / 3600:.1f} CPU-h")
EOF
echo "T5M3_PROBE_DONE $(date -u +%FT%TZ)"

if [ "$WITH_CENSUS" = "1" ]; then
  # the whole census again, into a scratch directory, every hash compared
  mkdir -p "$POD/census"
  seq 0 73 | xargs -P "$WORKERS" -n 1 sh -c \
    "nice -n 19 $PY research/t5q_m2_rank5/batch.py \"\$0\" --out-dir $POD/census --force --max-seconds 7200 \
     > $POD/census/batch_\"\$0\".log 2>&1"
  $PY - <<'EOF'
import json, os
part = json.load(open("research/t5q_m2_rank5/partition.json"))
diff, missing, hits = [], [], 0
for k in range(part["batches"]):
    p = f"research/t5_m3_lift/results/pod/census/batch_{k}.json"
    if not os.path.exists(p):
        missing.append(k)
        continue
    a = json.load(open(f"research/t5q_m2_rank5/results/batch_{k}.json"))
    b = json.load(open(p))
    hits += b["decompositions"]
    if a["deterministic_sha256"] != b["deterministic_sha256"]:
        diff.append(k)
print(f"census re-run: {part['batches'] - len(missing)}/{part['batches']} records, {hits} decomposition(s), "
      f"hash mismatches {diff}, missing {missing}")
EOF
  echo "T5M3_CENSUS_DONE $(date -u +%FT%TZ)"
fi

# the listing aggregate: plan, exact k = 1..4 censuses, every record, two
# seeded re-runs; writes research/t5q_m2_rank5/batch_manifest.json
nice -n 19 $PY -u research/t5q_m2_rank5/aggregate.py --list --recheck 2 --recheck-seed "$SEED" \
  --recheck-dir "$POD/recheck" > "$POD/aggregate_list.log" 2>&1; AGG=$?
echo "T5M3_AGG_EXIT $AGG $(date -u +%FT%TZ)"
tail -2 "$POD/aggregate_list.log"

# the lift stage with its own exact low-rank censuses
nice -n 19 $PY -u research/t5_m3_lift/lift.py cell --low-census > "$POD/lift_cell.log" 2>&1; LIFT=$?
echo "T5M3_LIFT_EXIT $LIFT $(date -u +%FT%TZ)"
tail -3 "$POD/lift_cell.log"

# the certificate as the verifier runs it, on the filled draft
nice -n 19 $PY research/t5_m3_lift/fill_draft.py --manifest research/t5q_m2_rank5/batch_manifest.json \
  --out "$OUT/T5-m3-lower-6.json" > "$POD/fill_draft.log" 2>&1; echo "fill rc=$?"
nice -n 19 $PY -u verify_challenge/stabrank_verify.py "$OUT/T5-m3-lower-6.json" > "$POD/verify.log" 2>&1; CERT=$?
echo "T5M3_CERT_EXIT $CERT $(date -u +%FT%TZ)"
tail -3 "$POD/verify.log"

M3=$(grep -q '^CERTIFIED chi(T5^3) >= 6' "$POD/lift_cell.log" && echo 6 || (grep -q '^DECOMPOSITION FOUND' "$POD/lift_cell.log" && echo 5 || echo undecided))
echo "T5M3_SUMMARY chi(T5^3)>=$M3 aggregate_exit=$AGG lift_exit=$LIFT certificate_exit=$CERT $(date -u +%FT%TZ)"
echo "T5M3_CHAIN_DONE $(date -u +%FT%TZ)"
