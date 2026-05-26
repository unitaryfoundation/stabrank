#!/bin/bash
# Post-triple-search workflow: validate result JSON, promote it into
# paper/certificates/, verify, commit, push, open PR, wait CI, merge.
#
# Usage: post_triple_search.sh <orbit> <m>
# Example: post_triple_search.sh strange 3

set -euo pipefail

ORBIT="${1:?Usage: $0 <orbit> <m>}"
M="${2:?Usage: $0 <orbit> <m>}"

cd /home/vrusso/Projects/uf/stabrank
RESULT="sa_logs/triple_search/result-${ORBIT}-m${M}.json"
CERT="paper/certificates/triple_${ORBIT}_m${M}.json"
BRANCH="paper/${ORBIT}-m${M}-certificate"

echo "[post_triple_search] orbit=${ORBIT} m=${M}"
echo "[post_triple_search] result=${RESULT}"

# 1. Validate result file
[ -s "${RESULT}" ] || { echo "ERROR: ${RESULT} missing or empty"; exit 1; }
python3 - "${RESULT}" "${ORBIT}" "${M}" <<'PY'
import json, sys
path, orbit, m = sys.argv[1:]
with open(path) as f:
    r = json.load(f)
assert r['orbit'] == orbit, f"orbit mismatch: {r['orbit']!r} vs {orbit!r}"
assert r['m'] == int(m), f"m mismatch: {r['m']} vs {m}"
assert r['n_triples_processed'] == r['n_triples_total'], (
    f"incomplete search: {r['n_triples_processed']:,} of {r['n_triples_total']:,}"
)
print(f"  certificate: {r['certificate']}")
print(f"  best_residual: {r['best_residual']:.10e}")
print(f"  best_triple: {r['best_triple']}")
print(f"  elapsed: {r['elapsed_seconds']:.0f}s")
PY

# 2. Sync main and branch
git checkout main
git pull --ff-only origin main
# If branch already exists from a prior aborted run, drop it.
git branch -D "${BRANCH}" 2>/dev/null || true
git checkout -b "${BRANCH}"

# 3. Promote certificate
cp "${RESULT}" "${CERT}"

# 4. Verify the new certificate (and re-verify all others, to confirm nothing
# else regressed).
uv run python scripts/research/orbit_paper/verify_certificates.py --strict

# 5. Commit
RESID=$(python3 -c "import json; print(f\"{json.load(open('${CERT}'))['best_residual']:.6e}\")")
HOURS=$(python3 -c "import json; print(f\"{json.load(open('${CERT}'))['elapsed_seconds']/3600:.1f}\")")
TRIPLE=$(python3 -c "import json; print(json.load(open('${CERT}'))['best_triple'])")
NPROC=$(python3 -c "import json; print(f\"{json.load(open('${CERT}'))['n_triples_processed']:,}\")")
VERDICT=$(python3 -c "import json; print(json.load(open('${CERT}'))['certificate'])")

ORBIT_TITLE=$(python3 -c "print('${ORBIT}'.replace('h3','H_3').replace('norrell','Norrell').replace('strange','Strange').replace('t3','T_3'))")

git add "${CERT}"
git commit -m "Add ${ORBIT_TITLE} m=${M} triple certificate (${VERDICT})

Search elapsed: ${HOURS}h. Best residual = ${RESID} at witness triple ${TRIPLE}.
Triples scanned: ${NPROC}.
Verified by scripts/research/orbit_paper/verify_certificates.py --strict.
"

# 6. Push and open PR
git push -u origin "${BRANCH}"

PR_BODY=$(cat <<EOF
Post-search certificate JSON for the deterministic ${VERDICT//(/\\(} search at m=${M}.

- Search elapsed: ${HOURS}h on a 12-core machine, niced
- Triples processed: ${NPROC}
- Best residual: ${RESID}
- Witness triple (best 3-stab approximator): \`${TRIPLE}\`
- Verdict: \`${VERDICT}\`

Verified locally via:
\`\`\`
uv run python scripts/research/orbit_paper/verify_certificates.py --strict
\`\`\`

This is the certificate file for an existing claim already reflected in the paper.
EOF
)
gh pr create --base main \
    --title "Add ${ORBIT_TITLE} m=${M} triple certificate" \
    --body "${PR_BODY}"

PR_NUM=$(gh pr view "${BRANCH}" --json number -q .number)
echo "[post_triple_search] Opened PR #${PR_NUM}"

# 7. Wait for CI to settle, then merge
echo "[post_triple_search] Waiting for CI to clear..."
until STATE=$(gh pr view "${PR_NUM}" --json mergeStateStatus -q .mergeStateStatus) \
      && [ "${STATE}" = "CLEAN" ]; do
    if [ "${STATE}" = "DIRTY" ] || [ "${STATE}" = "BLOCKED" ]; then
        echo "[post_triple_search] PR ${PR_NUM} blocked: state=${STATE}; aborting auto-merge"
        exit 1
    fi
    sleep 30
done

gh pr merge "${PR_NUM}" --merge --delete-branch
echo "[post_triple_search] Merged PR #${PR_NUM}"

# 8. Sync main
git checkout main
git pull --ff-only origin main

echo "[post_triple_search] DONE: ${ORBIT} m=${M} certificate promoted and merged"
