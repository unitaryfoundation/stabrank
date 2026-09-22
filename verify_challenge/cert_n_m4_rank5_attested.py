"""Certificate: chi(|N>^{ot 4}) >= 6, by the exclusion of every rank-5
decomposition through a two-qutrit all-visible base slice, over stored batch
outputs (attested tier).

The argument (docs/notes/qutrit_m4_rank5_exclusion.md, section 2): every
one-qutrit slice of a rank-5 decomposition of psi_4 = |N>^4 has all five
terms nonzero (Fact A: chi(N^3) = 4 and the unique rank-4 decomposition of
|N>^3 does not lift, research/constructions/relaxed_lift.py case [A]), and
every two-qutrit slice point has at least four (Fact B, PR #86). Along
qutrits 1, 2 every term is therefore a nine-slice term or a line term along
a diagonal direction, at most one term is a line term, and a monomial
symmetry of |N> on qutrits 1, 2 carries the decomposition to one whose slice
at x_0 = (0, 0) has all five terms nonzero. That slice is a full 5-cover of
|N>^2 by two-qutrit stabilizer states, and the two-qutrit slice structure
lemma fixes the eight other slices of every term up to a finite set of
Pauli classes and cube-root phases. research/qutrit_m4_rank5 enumerates
every full 5-cover of |N>^2 up to the unitary symmetry of |N>^2 (stage A:
distinct independent states; stage B: dependent; stage C: a repeated
state), matches each at x_0 modulo 65521 (a superset of the true matches)
and re-decides every candidate mod 2013265921 and in floating point. No
match is a decomposition of psi_4, so chi(N^4) >= 6; with the Lean rank-7
witness bounds/N-m4-upper-7.json the cell stands at 6 <= chi(N^4) <= 7.

What this script re-runs: research/qutrit_m4_rank5/aggregate.py N checks
that the stage A units of the partition are exactly the enumerator's pivot
pairs, re-enumerates the degenerate covers of stages B and C and compares
them with the stored list, checks that every batch output named in the
partition is present, hashes as stored and matches the partition's geometry
and hashes, re-decides every stored hit exactly, and confirms that no batch
recorded an undecided run or a decomposition; it then re-runs two batches
from scratch, chosen from the printed seed, and compares their
deterministic hashes bit for bit with the stored records. The completeness
of the enumeration rests on the stored outputs and on the committed runner
(research/qutrit_m4_rank5/batch.py) that regenerates any batch; the
pipeline records that as the attested tier. The controls of the matcher
(research/qutrit_m4_rank5/results/N/control_*.json) are part of the record
but are not re-run here.

Printed claims: CERTIFIED chi(N^4) >= 6
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORBIT = "N"
RECHECK, SEED = 2, 20260922
CLAIM = f"CERTIFIED chi({ORBIT}^4) >= 6"


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix=f"{ORBIT}_m4_rank5_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "qutrit_m4_rank5", "aggregate.py"), ORBIT,
           "--recheck", str(RECHECK), "--recheck-seed", str(SEED), "--recheck-dir", out,
           "--no-manifest"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    lines = [l.strip() for l in proc.stdout.splitlines()]
    if proc.returncode != 0 or CLAIM not in lines:
        print("aggregation did not certify", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
