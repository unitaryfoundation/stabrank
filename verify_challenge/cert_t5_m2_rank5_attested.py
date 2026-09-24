"""Certificate: chi(|T5>^{ot 2}) >= 6, by the emptiness of the census of
5-sets of two-ququint stabilizer states whose span contains |T5>^2, over
stored batch outputs (attested tier), and by projection chi(|T5>^3) >= 6
and chi(|T5>^4) >= 6.

The argument (docs/notes/t5q_m2_rank5_exclusion.md): ranks 1 to 4 are
excluded for psi_2 = |T5>^2 (bounds/T5-m2-lower-5.json numerically, and
exactly by the k = 1..4 censuses the aggregate re-runs), so a rank-5
decomposition psi_2 = sum_{i=1}^5 c_i s_i has five distinct independent
states and five nonzero coefficients, and is nothing more than a 5-set of
dictionary states whose span contains psi_2. A unitary symmetry of psi_2
(order 50) carries such a 5-set to another, so one member of every orbit
contains its orbit's pivot (one of 98 representatives) and, after the
pivot's stabilizer acts, a partner from the pivot's partner list with the
other three members above it. For every one of the 155,422 (pivot,
partner) units the compiled kernel cpp/src/cover5.cpp lists a superset of
those 5-sets (a third pivot and a pair of members whose images modulo the
span of the target and the three pivots are parallel over F_65521) and
decides each modulo 2013265921 in the Q(zeta_5) field; the batches
re-decide every listed set exactly and numerically. No set is listed, so
chi(T5^2) >= 6. Since every amplitude of |T5> is nonzero, slicing carries a
rank-r decomposition of |T5>^3 to one of |T5>^2 (projection monotonicity,
Lean: stabRankP_powVecP_mono in lean_proofs/LeanProofs/Stabilizer/
SliceP.lean), so chi(T5^3) >= 6 and chi(T5^4) >= 6 as well.

What this script re-runs: research/t5q_m2_rank5/aggregate.py re-enumerates
the dictionary and the units and compares them with the partition (hashes
included), runs the k = 1..4 censuses (about 100 s), checks that every
batch output is present, hashes as stored and matches the partition's
geometry, re-decides every stored hit exactly, fails on any undecided
unit, and re-runs two batches from scratch, chosen from the printed seed,
with their deterministic hashes compared bit for bit. The completeness of
the enumeration rests on the stored outputs and on the committed runner
(research/t5q_m2_rank5/batch.py) that regenerates any batch. The controls
(research/t5q_m2_rank5/results/control_*.json) are part of the record and
are not re-run here.

Printed claims: CERTIFIED chi(T5^2) >= 6
                CERTIFIED chi(T5^3) >= 6 (projection monotonicity)
                CERTIFIED chi(T5^4) >= 6 (projection monotonicity)
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20260925
CLAIM = "CERTIFIED chi(T5^2) >= 6"


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="t5q_m2_rank5_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "t5q_m2_rank5", "aggregate.py"),
           "--recheck", str(RECHECK), "--recheck-seed", str(SEED), "--recheck-dir", out,
           "--no-manifest"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    lines = [l.strip() for l in proc.stdout.splitlines()]
    if proc.returncode != 0 or CLAIM not in lines:
        print("aggregation did not certify", file=sys.stderr)
        return 1
    # Projection monotonicity (Lean: stabRankP_powVecP_mono in
    # lean_proofs/LeanProofs/Stabilizer/SliceP.lean): |T5> is a nonzero
    # vector with every amplitude nonzero, so a rank-r decomposition of
    # |T5>^(m+1) slices to one of |T5>^m, and the m = 2 exclusion bounds the
    # m = 3 and m = 4 cells as well.
    print("CERTIFIED chi(T5^3) >= 6")
    print("CERTIFIED chi(T5^4) >= 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
