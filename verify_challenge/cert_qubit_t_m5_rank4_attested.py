"""Certificate: chi(|T>^{ot 5}) >= 5, by the exclusion of every rank-4
decomposition through a two-qubit base slice, over stored batch outputs
(attested tier).

The argument (docs/notes/t5_rank4_exclusion.md): in a rank-4 decomposition
of psi_5 = |T>^5 every term is full along every qubit (Fact 1, from the
residual tables of research/constructions/two_qubit_slice.py for the unique
rank-3 decomposition of |T>^4 at the slice ratio tau^{+-1}, tau = a_1 / a_0
= e^{i pi/4} tan(beta)) and every two-qubit slice point sees at least three
terms (Fact 2, chi(T^3) = 3), so along qubits 1, 2 every term is a plane or
a diagonal-line term with at most one line term per diagonal, and at a base
point x_0 in {00, 01} all four terms are visible; the one configuration
with a line term on each diagonal is excluded by the same tables for the
rank-3 decompositions of |T>^3 at the ratio tau^{+-2}. The base is a full
4-cover of |T>^3, listed up to the unitary symmetry of psi_3 in
research/t5_rank4/covers4.json (distinct independent states from the
modular pivot enumeration, the degenerate multisets over the full 3-covers);
research/t5_rank4 matches every base at both base points modulo 65521 (a
superset of the true matches, the amplitudes reduced from Q(zeta_24)) and
re-decides every candidate mod 2013265921 and in floating point. No match
is a decomposition of psi_5, so chi(T^5) >= 5, and by projection
chi(T^6) >= 5.

What this script re-runs: research/t5_rank4/aggregate.py checks the
partition, re-enumerates the census (about three minutes) and compares it
with the stored list, checks that every batch output is present, hashes as
stored and matches the partition's geometry and hashes, re-decides every
stored hit exactly, fails on any refused or undecided run, and re-runs two
batches from scratch, chosen from the printed seed, with their
deterministic hashes compared bit for bit. The completeness of the
enumeration rests on the stored outputs and on the committed runner
(research/t5_rank4/batch.py) that regenerates any batch. The controls
(research/t5_rank4/results/control_*.json, results/tables.json) are part of
the record and are not re-run here.

Printed claim: CERTIFIED chi(qubit_T^5) >= 5
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20260924
CLAIM = "CERTIFIED chi(qubit_T^5) >= 5"


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="t5_rank4_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "t5_rank4", "aggregate.py"),
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
