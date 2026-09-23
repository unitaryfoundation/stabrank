"""Certificate: chi(|H>^{ot 5}) >= 6, by the exclusion of every rank-5
decomposition through a two-qubit base slice, over stored batch outputs
(attested tier).

The argument (docs/notes/h5_rank5_exclusion.md): in a rank-5 decomposition
of psi_5 = |H>^5 every term is full along every qubit (Fact 1, from the
residual tables of research/constructions/two_qubit_slice.py for the 30
rank-4 decompositions of |H>^4) and every two-qubit slice point sees at
least three terms (Fact 2, chi(H^3) = 3), so along qubits 1, 2 every term is
a plane or a diagonal-line term, and one of two cases holds at a base point
x_0 in {00, 01}: all five terms are visible there (stage alpha, the base a
full 5-cover of psi_3, the census of research/h6_rank5 reused by hash), or
four are visible and the fifth is a line term on the diagonal missing x_0
(stage beta, the base a full 4-cover of psi_3); the remaining configuration
(two line terms on each diagonal) is excluded by the same tables at the
ratio tan(pi/8)^{+-2}. research/h5_rank5 matches every base at both base
points modulo 65521 (a superset of the true matches) and re-decides every
candidate mod 2013265921 and in floating point. No match is a decomposition
of psi_5, so chi(H^5) >= 6, and with the rank-6 witness
bounds/qubit_H-m5-upper-6.json, chi(H^5) = 6.

What this script re-runs: research/h5_rank5/aggregate.py checks the
partition against the enumerator's pivot pairs and the census hash,
re-enumerates the degenerate 5-covers and the stage beta bases (about 200 s)
and compares them with the stored lists, checks that every batch output is
present, hashes as stored and matches the partition's geometry and hashes,
re-decides every stored hit exactly, fails on any refused or undecided run,
and re-runs two batches from scratch, chosen from the printed seed, with
their deterministic hashes compared bit for bit. The completeness of the
enumeration rests on the stored outputs and on the committed runner
(research/h5_rank5/batch.py) that regenerates any batch. The controls
(research/h5_rank5/results/control_*.json) are part of the record and are
not re-run here.

Printed claim: CERTIFIED chi(qubit_H^5) >= 6
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20260923
CLAIM = "CERTIFIED chi(qubit_H^5) >= 6"


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="h5_rank5_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "h5_rank5", "aggregate.py"),
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
