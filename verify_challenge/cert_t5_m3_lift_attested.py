"""Certificate: chi(|T5>^3) >= 6 by slice-and-lift at exact rank from the
census listing of the rank-5 decompositions of |T5>^2 (attested tier).

The argument (docs/notes/t5_m3_lift_design.md). chi(|T5>^2) = 5: the Lean
witness bounds/T5-m2-upper-5.json and the exact k = 1..4 censuses that the
aggregate re-runs here. Every amplitude of |T5> is nonzero, so a rank-5
decomposition of |T5>^3 sliced at the third ququint gives, at every one of
the five slice values, a decomposition of |T5>^2 with at most five terms,
hence exactly five nonzero pairwise non-parallel terms forming a 5-set
whose span contains |T5>^2 (no invisible or coincident term is possible
at exact rank). The census of research/t5q_m2_rank5 (74 batches, 155,422
pivot-partner units, every unit run, 2026-09-24) lists one such 5-set per
orbit of the unitary symmetry group; it found exactly one, the five Z(x)Z
eigensectors, fixed by the group. Each lifted term is then determined by
its slice pattern (which sector at which slice value), and the lift exists
if and only if five such pattern vectors are stabilizer states and form a
bijection at every slice. research/t5_m3_lift/lift.py decides this exactly
over Q(zeta_5): 3,125 patterns, 25 survive the exact prunes, none is a
stabilizer state. So chi(|T5>^3) >= 6, and by projection monotonicity
chi(|T5>^4) >= 6.

What this script re-runs: research/t5q_m2_rank5/aggregate.py --list checks
the partition against a fresh enumeration, re-runs the exact k = 1..4
censuses (ranks 1 to 4 excluded), checks that every batch output is
present, hashes as stored and matches the partition's geometry, re-decides
every stored hit, fails on any undecided unit, and re-runs two batches from
scratch, chosen from the printed seed, with their deterministic hashes
compared bit for bit. Then lift.py cell reads the records again, forms the
orbit closure of the listed sets, re-decides every set exactly over
Q(zeta_5), and runs the exact lift decision. The completeness of the
census rests on the stored outputs and on the committed runner
(research/t5q_m2_rank5/batch.py), under the modular assumption every
kernel census of the project makes (a member nonzero over Q(zeta_5) is
nonzero modulo 65521).

Printed claims: CERTIFIED chi(T5^3) >= 6
                CERTIFIED chi(T5^4) >= 6 (projection monotonicity)
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20261001
CLAIM_M3 = "CERTIFIED chi(T5^3) >= 6"
CLAIM_M4 = "CERTIFIED chi(T5^4) >= 6"


def run(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode, [line.strip() for line in proc.stdout.splitlines()]


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="t5q_m2_rank5_recheck_")
    rc, lines = run([sys.executable, os.path.join(ROOT, "research", "t5q_m2_rank5", "aggregate.py"), "--list",
                     "--recheck", str(RECHECK), "--recheck-seed", str(SEED), "--recheck-dir", out, "--no-manifest"])
    listing = [line for line in lines if line.startswith("LISTING COMPLETE:")]
    if rc != 0 or not listing:
        print("the census aggregate did not complete the listing", file=sys.stderr)
        return 1
    # The aggregate just re-ran the exact k = 1..4 censuses, so the lift
    # stage takes chi(T5^2) >= 5 from it rather than running them again.
    rc, lines = run([sys.executable, os.path.join(ROOT, "research", "t5_m3_lift", "lift.py"), "cell",
                     "--trust-low-census"])
    if rc != 0 or CLAIM_M3 not in lines:
        print("the lift stage did not certify", file=sys.stderr)
        return 1
    print(CLAIM_M3)
    # Projection monotonicity (Lean: stabRankP_powVecP_mono in
    # lean_proofs/LeanProofs/Stabilizer/SliceP.lean): every amplitude of
    # |T5> is nonzero, so a rank-r decomposition of T5^4 slices to one of
    # T5^3, and the m=3 bound holds at m=4.
    print(CLAIM_M4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
