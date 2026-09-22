"""Certificate: chi(|H>^{ot 6}) >= 6, by the exclusion of every rank-5
decomposition through an all-visible base slice, over stored batch outputs
(attested tier).

The argument (docs/notes/h6_rank5_exclusion.md): a rank-5 decomposition of
psi_6 = |H>^6 has, along some three qubits S and at some base point x_0, all
five terms nonzero (the all-visible slice lemma, which rests on property P
of PR #87: every four-qubit slice of a rank-5 decomposition along a 2 + 4
bipartition has five nonzero terms, so every term's stabilizer group has no
pure-Z word of weight 1 or 2). The base slice is then a full 5-cover of
psi_3 by stabilizer states, and the slice structure lemma fixes the other
seven slices of every term up to a finite set of Pauli classes and phases.
research/h6_rank5 enumerates every full 5-cover of psi_3 up to the symmetry
of psi_3 (5,939,465 covers of distinct independent states, stage A; 12,390
dependent covers, stage B; 13,852 covers with a repeated state, stage C),
matches each at the four base points modulo 65521 (a superset of the true
matches) and re-decides every candidate mod 2013265921 and in floating
point. No match is a decomposition of psi_6, so chi(H^6) >= 6, and with the
rank-6 witness bounds/qubit_H-m6-upper-6.json, chi(H^6) = 6. The whole run
is about 36 CPU-hours, which no certificate budget can re-run.

What this script does re-run: research/h6_rank5/aggregate.py checks that
the stage A units of the partition are exactly the enumerator's pivot
pairs, re-enumerates the degenerate covers of stages B and C (about 190 s)
and compares them with the stored list, checks that every batch output
named in the partition is present, hashes as stored and matches the
partition's geometry and hashes, re-decides every stored hit exactly, and
confirms that no batch recorded an undecided run or a decomposition; it
then re-runs two batches from scratch, chosen from the printed seed, and
compares their deterministic hashes bit for bit with the stored records.
The completeness of the enumeration rests on the stored outputs and on the
committed runner (research/h6_rank5/batch.py) that regenerates any batch;
the pipeline records that as the attested tier. Both positive controls of
the matcher (the rank-4 decompositions of |H>^4 and the rank-6 witness from
its own base slices, research/h6_rank5/results/control_*.json) are part of
the record but are not re-run here.

Printed claims: CERTIFIED chi(qubit_H^6) >= 6
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20260921
CLAIM = "CERTIFIED chi(qubit_H^6) >= 6"


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="h6_rank5_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "h6_rank5", "aggregate.py"),
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
