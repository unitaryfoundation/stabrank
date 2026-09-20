"""Certificate: chi(|T3>^{ot 3}) >= 8, by the exact three-pivot scan of rank-7
configurations, over stored batch outputs (attested tier).

The argument is that of cert_t3m3_rank7.py one rank up: after Galois descent
a rank-7 decomposition is seven independent stabilizer states whose span
contains V_3, so their images modulo V_3 span four dimensions, and for any
three of them chosen as pivots (first pivot the least orbit block, second
and third minimal under the stabilizers) the remaining four have images in
a one-dimensional quotient. research/t3_rank7/batch.py enumerates every such
class set and decides exactly, mod 2^31 - 1 under the Hadamard bound,
whether V_3 lies in its span; docs/notes/t3_rank7_exclusion.md has the
design and research/t3_rank7/README.md the validation. The scan is 1.31e13
inner steps in 459 batches, about 80 CPU-hours, which no certificate budget
can re-run.

What this script does re-run: research/t3_rank7/aggregate.py checks that
every batch output named in the partition is present, hashes as stored,
matches the partition's geometry and the shared setup hash, that the
inner-step counts sum to the partition total, that the ranges tile every
block, and that no batch recorded a class set containing V_3 or an
undecided one; it then re-runs two batches from scratch, chosen from the
printed seed, and compares them bit for bit with the stored records. The
completeness of the enumeration rests on the stored outputs; the pipeline
records that as the attested tier.

Printed claims: CERTIFIED chi(T3^3) >= 8
                CERTIFIED chi(T3^4) >= 8 (projection monotonicity)
                CERTIFIED chi(T3^5) >= 8 (projection monotonicity)
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20260919


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="t3_rank7_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "t3_rank7", "aggregate.py"),
           "--recheck", str(RECHECK), "--recheck-seed", str(SEED), "--recheck-dir", out]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    lines = [l.strip() for l in proc.stdout.splitlines()]
    if proc.returncode != 0 or "CERTIFIED chi(T3^3) >= 8" not in lines:
        print("aggregation did not certify", file=sys.stderr)
        return 1
    # Projection monotonicity: every amplitude of |T3> is nonzero, so
    # (I (x) <x|) carries a decomposition of T3^(m+1) to one of T3^m with no
    # more terms, and the m=3 value bounds the next two cells.
    print("CERTIFIED chi(T3^4) >= 8")
    print("CERTIFIED chi(T3^5) >= 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
