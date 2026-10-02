#!/bin/bash
# Chain for the rank-6 exclusion of |H3>^4 (7 <= chi(H3^4) <= 8) on the RunPod
# pod (32 vCPU, Ubuntu 20.04, gcc 9.4, uv-managed Python 3.12). Nothing here
# is run by hand on the laptop; the user launches it when the pod is free.
# Design: docs/notes/h3_m4_rank6_design.md; pipeline: research/h3_m4_rank6/.
#
# Setup (fresh pod, disk erased on stop; the branch h3-m4-rank6-matcher is
# local and never pushed, so it travels as a bundle on top of the public main):
#   cd /Users/vincent.russo/Projects/genesis/stabrank
#   git bundle create /tmp/h3-m4-rank6.bundle origin/main..h3-m4-rank6-matcher
#   /tmp/podscp.sh /tmp/h3-m4-rank6.bundle root@213.173.105.71:/root/h3-m4-rank6.bundle
#   /tmp/podssh.sh 'mkdir -p /root/logs && (command -v g++-10 >/dev/null || (apt-get update -qq && apt-get install -y -qq gcc-10 g++-10 >/dev/null)) && g++-10 --version | head -1'
#   /tmp/podssh.sh 'cd /root && git clone -q https://github.com/unitaryfoundation/stabrank stabrank-h3m4 && cd stabrank-h3m4 && git fetch -q /root/h3-m4-rank6.bundle h3-m4-rank6-matcher:h3-m4-rank6-matcher && git checkout -q h3-m4-rank6-matcher && git log --oneline -1'
#   /tmp/podssh.sh 'export PATH=$HOME/.local/bin:$PATH; cd /root/stabrank-h3m4 && CC=gcc-10 CXX=g++-10 uv sync --extra challenge --extra test 2>&1 | tail -3'
# The kernels (cover6_pair, SliceMatch3Kernel, dense_solve) do not build with
# gcc 9: cpp/include/stabrank/{linalg,polynomial}.hpp include <span> and
# cpp/src/{clifford,polynomial,fidelity}.cpp include <numbers>, C++20 headers
# that libstdc++ 9 lacks (first shipped with GCC 10). The uv venv of this
# checkout is /root/stabrank-h3m4/.venv (uv sync creates it; the pod's
# /root/g3b/.venv belongs to another job and is not touched).
#
# Launch (the whole exclusion: controls, lists, rates, partition, the first
# B6 batch timed, the batch loop at WORKERS processes, the aggregate with two
# seeded re-runs, the draft and the certificate under the verifier):
#   /tmp/podssh.sh 'WORKERS=28 setsid nohup bash /root/stabrank-h3m4/research/h3_m4_rank6/pod/pod_chain.sh > /root/logs/h3m4_chain.log 2>&1 < /dev/null & disown'
# With the --no-native replays of the lightest A6 batch and one B6 batch
# appended (hours; the Python references, so WITH_NONATIVE=1 only when the
# pod has time to spare after the aggregate):
#   /tmp/podssh.sh 'WORKERS=28 WITH_NONATIVE=1 setsid nohup bash /root/stabrank-h3m4/research/h3_m4_rank6/pod/pod_chain.sh > /root/logs/h3m4_chain.log 2>&1 < /dev/null & disown'
# Watch:
#   /tmp/podssh.sh 'grep -E "^H3M4_" /root/logs/h3m4_chain.log; tail -3 /root/logs/h3m4_chain.log'
#   /tmp/podssh.sh 'ls /root/stabrank-h3m4/research/h3_m4_rank6/results/batch_*.json | wc -l; grep -l "DECOMPOSITION FOUND\|undecided run" /root/stabrank-h3m4/research/h3_m4_rank6/results/batch_*.log'
# Markers (anchored, grep ^MARKER; no waiting echo contains a marker):
# H3M4_START, H3M4_KERNEL_OK | H3M4_KERNEL_FAILED, H3M4_TESTS_EXIT,
# H3M4_PLANTED_DONE, H3M4_LISTS_DONE, H3M4_CONTROLS_DONE, H3M4_SAMPLE_DONE,
# H3M4_PARTITION_DONE, H3M4_B6_PROBE_DONE, H3M4_RUN_DONE, H3M4_RERUN_DONE,
# H3M4_AGG_EXIT, H3M4_NONATIVE_DONE (with WITH_NONATIVE=1), H3M4_CERT_EXIT,
# H3M4_SUMMARY, H3M4_CHAIN_DONE.
# Copy back afterwards (the records, the lists, the partition, the manifest,
# the filled bound, the logs):
#   rsync -avz -e "ssh -p 22904 -i ~/.ssh/id_ed25519" root@213.173.105.71:/root/stabrank-h3m4/research/h3_m4_rank6/results/ research/h3_m4_rank6/results/
#   rsync -avz -e "ssh -p 22904 -i ~/.ssh/id_ed25519" root@213.173.105.71:/root/stabrank-h3m4/research/h3_m4_rank6/{reps_H3.json,partition.json,batch_manifest.json} research/h3_m4_rank6/
#   rsync -avz -e "ssh -p 22904 -i ~/.ssh/id_ed25519" root@213.173.105.71:/root/logs/h3m4_chain.log research/h3_m4_rank6/results/pod/
# Then move results/H3-m4-lower-7.json to bounds/ and run
# verify_challenge/cert_h3_m4_rank6_attested.py on the laptop.
set -u
REPO=${REPO:-/root/stabrank-h3m4}
WORKERS=${WORKERS:-28}
WITH_NONATIVE=${WITH_NONATIVE:-0}
SEED=${SEED:-20261002}
TARGET_S=${TARGET_S:-600}
GUARD_S=${GUARD_S:-3600}
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"
D=research/h3_m4_rank6
OUT=$D/results
POD=$OUT/pod
mkdir -p "$POD" /root/logs
echo "H3M4_START $(date -u +%FT%TZ) host $(hostname) commit $(git rev-parse --short HEAD) workers $WORKERS with_nonative $WITH_NONATIVE"

# the compiled kernels must import (built with gcc-10 at setup)
if ! $PY -c "from stabrank.stabrank_core import cover6_pair, SliceMatch3Kernel, dense_solve" 2> "$POD/kernel_import.err"; then
  echo "H3M4_KERNEL_FAILED $(date -u +%FT%TZ): $(tail -1 "$POD/kernel_import.err")"
  echo "H3M4_CHAIN_DONE $(date -u +%FT%TZ) aborted"
  exit 1
fi
echo "H3M4_KERNEL_OK $(date -u +%FT%TZ)"

# the test module (the matcher's controls at this commit), then the planted
# controls of the invisible stages as the first stage of the chain
nice -n 19 $PY -m pytest tests/test_h3_m4_rank6.py -q > "$POD/pytest.log" 2>&1; T=$?
echo "H3M4_TESTS_EXIT $T $(date -u +%FT%TZ) $(tail -1 "$POD/pytest.log")"
nice -n 19 $PY -u $D/driver.py control-planted --stage beta --kinds a,d,r --cap 120 > "$POD/control_planted_beta_adr.log" 2>&1; echo "planted beta a,d,r rc=$?"
nice -n 19 $PY -u $D/driver.py control-planted --stage beta --kinds h --cap 120 > "$POD/control_planted_beta_h.log" 2>&1; echo "planted beta h rc=$?"
nice -n 19 $PY -u $D/driver.py control-planted --stage gamma --cap 120 > "$POD/control_planted_gamma.log" 2>&1; echo "planted gamma rc=$?"
echo "H3M4_PLANTED_DONE $(date -u +%FT%TZ) $(grep -h '^control-planted' "$POD"/control_planted_*.log | sed 's/;.*//' | tr '\n' ' ')"

# the lists (B6 by the seven routes of degenerate6, C6, k5, k4), written
# with their hash and compared with the committed reps_H3.json
# (compared by content: the codes and kappa vectors; the record's seconds,
# git and generated fields differ on every build and sit inside the hash)
git show HEAD:$D/reps_H3.json > "$POD/reps_H3.committed.json" 2>/dev/null || true
nice -n 19 $PY -u $D/driver.py lists --write > "$POD/lists.log" 2>&1; echo "lists rc=$?"
LISTS=$($PY - <<'EOF'
import json, os
new = json.load(open("research/h3_m4_rank6/reps_H3.json"))
p = "research/h3_m4_rank6/results/pod/reps_H3.committed.json"
if not os.path.exists(p) or os.path.getsize(p) == 0:
    print("no committed file to compare")
else:
    old = json.load(open(p))
    same = all(new[k]["codes"] == old[k]["codes"] for k in ("B6", "C6", "k5", "k4")) and new["B6"]["kappa"] == old["B6"]["kappa"]
    print("same lists as the committed file (codes and kappa)" if same else "DIFFERENT lists from the committed file")
EOF
)
echo "H3M4_LISTS_DONE $(date -u +%FT%TZ) reps_H3.json: $LISTS; $(tail -1 "$POD/lists.log")"

# the remaining controls, each in its own log (b6 and c6 plants need the lists)
nice -n 19 $PY -u $D/driver.py control-lists > "$POD/control_lists.log" 2>&1; echo "control-lists rc=$?"
nice -n 19 $PY -u $D/driver.py control-planted --stage b6 --cap 120 > "$POD/control_planted_b6.log" 2>&1; echo "planted b6 rc=$?"
nice -n 19 $PY -u $D/driver.py control-planted --stage c6 --cap 120 > "$POD/control_planted_c6.log" 2>&1; echo "planted c6 rc=$?"
nice -n 19 $PY -u $D/driver.py control-orbit > "$POD/control_orbit.log" 2>&1; echo "control-orbit rc=$?"
nice -n 19 $PY -u $D/driver.py control-m3 > "$POD/control_m3.log" 2>&1; echo "control-m3 rc=$?"
nice -n 19 $PY -u $D/driver.py control-witness --cap 25 > "$POD/control_witness.log" 2>&1; echo "control-witness rc=$? (recorded, not required)"
echo "H3M4_CONTROLS_DONE $(date -u +%FT%TZ)"

# the rates of every stage on real items (results/rates.json), then the partition
nice -n 19 $PY -u $D/driver.py sample A6 --count 20000 --reference 100 > "$POD/sample_A6.log" 2>&1; echo "sample A6 rc=$?"
nice -n 19 $PY -u $D/driver.py sample B6 --count 40 --budget 600 > "$POD/sample_B6.log" 2>&1; echo "sample B6 rc=$?"
nice -n 19 $PY -u $D/driver.py sample C6 --count 10 --budget 480 --item-cap 40 > "$POD/sample_C6.log" 2>&1; echo "sample C6 rc=$?"
nice -n 19 $PY -u $D/driver.py sample beta --count 30 --budget 300 > "$POD/sample_beta.log" 2>&1; echo "sample beta rc=$?"
nice -n 19 $PY -u $D/driver.py sample gamma --count 30 --budget 200 > "$POD/sample_gamma.log" 2>&1; echo "sample gamma rc=$?"
echo "H3M4_SAMPLE_DONE $(date -u +%FT%TZ)"
nice -n 19 $PY -u $D/driver.py partition --target-s "$TARGET_S" > "$POD/partition.log" 2>&1; P=$?
echo "H3M4_PARTITION_DONE $P $(date -u +%FT%TZ) $(tail -1 "$POD/partition.log")"
if [ "$P" != "0" ]; then
  echo "H3M4_CHAIN_DONE $(date -u +%FT%TZ) aborted at the partition"
  exit 1
fi

# the first B6 kappa-1 batch timed before the rest is scheduled: its wall
# time per item against the partition's pod estimate, and the projection of
# the whole stage B6 at the measured rate
B6=$($PY - <<'EOF'
import json
part = json.load(open("research/h3_m4_rank6/partition.json"))
print(next(g["index"] for g in part["batch_geometry"] if g["stage"] == "B6" and g.get("class") == "kappa 1"))
EOF
)
nice -n 19 $PY -u $D/batch.py "$B6" --force --max-seconds "$GUARD_S" > "$OUT/batch_$B6.log" 2>&1; echo "B6 probe batch $B6 rc=$?"
$PY - "$B6" <<'EOF'
import json, sys
k = int(sys.argv[1])
part = json.load(open("research/h3_m4_rank6/partition.json"))
rec = json.load(open(f"research/h3_m4_rank6/results/batch_{k}.json"))
geo = part["batch_geometry"][k]
items = rec["items"]
rate = rec["wall_s"] / max(1, items)
n_b6 = part["reps"]["B6"]
est = part["estimated_s"]
print(f"B6 probe batch {k}: {items} items, {rec['matched']} matched, {rec['hit_count']} hits, "
      f"{len(rec['undecided'])} undecided, {rec['wall_s']:.0f}s wall ({rate:.2f} s per item against the "
      f"partition's {geo['estimated_s'] / max(1, items):.2f}); stage B6 projected {rate * n_b6 / 3600:.0f} CPU-h "
      f"at this rate (partition estimate {est['B6'] / 3600:.0f}); whole run estimate {est['total'] / 3600:.0f} CPU-h, "
      f"{est['total'] / 3600 / int(__import__('os').environ.get('WORKERS', '28')):.1f} h of wall time")
EOF
echo "H3M4_B6_PROBE_DONE $(date -u +%FT%TZ)"

# the whole partition at WORKERS processes; the loop skips complete records
# and redoes deadline-hit ones with --resume, so a second pass picks up
# batches that died without a record
B=$($PY -c "import json; print(json.load(open('research/h3_m4_rank6/partition.json'))['batches'])")
seq 0 $((B - 1)) | xargs -P "$WORKERS" -n 1 sh -c \
  "nice -n 19 $PY $D/batch.py \"\$0\" --resume --max-seconds $GUARD_S > $OUT/batch_\"\$0\".log 2>&1"
echo "H3M4_RUN_DONE $(date -u +%FT%TZ) records $(ls $OUT/batch_*.json | wc -l) of $B; decompositions $(grep -l '^DECOMPOSITION FOUND' $OUT/batch_*.log | wc -l); undecided $(grep -l 'undecided run' $OUT/batch_*.log | wc -l)"
seq 0 $((B - 1)) | xargs -P "$WORKERS" -n 1 sh -c \
  "test -f $OUT/batch_\"\$0\".json || nice -n 19 $PY $D/batch.py \"\$0\" --resume --max-seconds $((GUARD_S * 2)) > $OUT/batch_\"\$0\".log 2>&1"
seq 0 $((B - 1)) | xargs -P "$WORKERS" -n 1 sh -c \
  "grep -q deadline $OUT/batch_\"\$0\".log 2>/dev/null && nice -n 19 $PY $D/batch.py \"\$0\" --resume --max-seconds $((GUARD_S * 2)) > $OUT/batch_\"\$0\".log 2>&1 || true"
echo "H3M4_RERUN_DONE $(date -u +%FT%TZ) records $(ls $OUT/batch_*.json | wc -l) of $B"

# the aggregate: census re-run, list re-enumeration, every record, two seeded
# re-runs; writes batch_manifest.json when every check passes
nice -n 19 $PY -u $D/aggregate.py --recheck 2 --recheck-seed "$SEED" --recheck-dir "$POD/recheck" > "$POD/aggregate.log" 2>&1; AGG=$?
echo "H3M4_AGG_EXIT $AGG $(date -u +%FT%TZ)"
tail -3 "$POD/aggregate.log"

if [ "$WITH_NONATIVE" = "1" ]; then
  # the Python references (6-cover enumerator, stage A matcher, dense solve;
  # the B6 filter has no Python form, so B6 runs the reference matcher) on the
  # lightest A6 batch and the first B6 kappa-1 batch; hashes compared
  A6L=$($PY - <<'EOF'
import json
part = json.load(open("research/h3_m4_rank6/partition.json"))
a6 = [g for g in part["batch_geometry"] if g["stage"] == "A6"]
print(min(a6, key=lambda g: g["estimated_s"])["index"])
EOF
)
  mkdir -p "$OUT/nonative"
  for K in "$A6L" "$B6"; do
    STABRANK_NO_NATIVE=1 nice -n 19 $PY $D/batch.py "$K" --out-dir "$OUT/nonative" --force > "$OUT/nonative/batch_$K.log" 2>&1
    $PY - "$K" <<'EOF'
import json, sys
k = sys.argv[1]
a = json.load(open(f"research/h3_m4_rank6/results/batch_{k}.json"))
b = json.load(open(f"research/h3_m4_rank6/results/nonative/batch_{k}.json"))
print(f"no-native replay of batch {k}: deterministic hash {'matches' if a['deterministic_sha256'] == b['deterministic_sha256'] else 'DIFFERS'}, "
      f"{b['wall_s']:.0f}s against {a['wall_s']:.0f}s")
EOF
  done
  echo "H3M4_NONATIVE_DONE $(date -u +%FT%TZ)"
fi

# the filled draft and the certificate as the verifier runs it
CERT=1
if [ "$AGG" = "0" ]; then
  nice -n 19 $PY $D/fill_draft.py --out "$OUT/H3-m4-lower-7.json" > "$POD/fill_draft.log" 2>&1; echo "fill rc=$?"
  nice -n 19 $PY -u verify_challenge/stabrank_verify.py "$OUT/H3-m4-lower-7.json" > "$POD/verify.log" 2>&1; CERT=$?
  tail -3 "$POD/verify.log"
fi
echo "H3M4_CERT_EXIT $CERT $(date -u +%FT%TZ)"

CHI=$(grep -q '^CERTIFIED chi(H3^4) >= 7' "$POD/aggregate.log" && echo ">=7" || (grep -q '^DECOMPOSITION FOUND' "$POD/aggregate.log" && echo "<=6" || echo undecided))
echo "H3M4_SUMMARY chi(H3^4)$CHI aggregate_exit=$AGG certificate_exit=$CERT records=$(ls $OUT/batch_*.json | wc -l)/$B $(date -u +%FT%TZ)"
echo "H3M4_CHAIN_DONE $(date -u +%FT%TZ)"
