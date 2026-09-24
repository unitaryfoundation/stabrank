"""Certificate: chi(|N>^{ot 4}) >= 7, by the exclusion of every rank-6
decomposition through a two-qutrit base slice at the base point (2, 2),
over stored batch outputs (attested tier).

The argument (docs/notes/n4_rank6_exclusion.md): in a rank-6 decomposition
of psi_4 = |N>^4 sliced along qutrits 1, 2, every point of F_3^2 sees at
least four terms (Fact B, PR #86: no two-qutrit slice of a rank-5 or
rank-6 decomposition of |N>^4 has exactly three nonzero terms, and
chi(N^2) = 3 rules out fewer), so at the base point (2, 2) four, five or
six terms are visible and the base is a full k-multiset of |N>^2 with k in
{4, 5, 6}; the 6 - k invisible terms lie on the 16 affine flats of F_3^2
missing (2, 2) (eight points, eight lines). A unitary symmetry of |N>^2 on
the unsliced qutrits carries decompositions to decompositions with the
image base and the same flats, so one base per G_2 orbit suffices. The
stages: A6, the 37,201,212 full 6-covers of distinct independent states
(the modular pivot-pair enumeration cover6_pair, regenerated per pivot
pair); B6, the 259,655 orbit representatives of the dependent 6-sets (the
fresh-term filter on the first coordinate slice for kappa 1, the reference
matcher at a raised candidate cap beyond); C6, the 321,553 representatives
of the repeated 6-multisets; (beta'), the 54,488 representatives of the
full 5-multisets, each with the invisible term on every one of the 16
flats; (gamma), the 505 representatives of the full 4-multisets with every
one of the 136 flat multisets. research/n4_rank6 completes every base
slice by slice under the two-qutrit slice structure lemma, modulo 65521 (a
superset of the true matches, the amplitudes reduced from Q(omega_3)) with
every candidate re-decided mod 2013265921 and in floating point; a
configuration a matcher cannot decide raises and fails its batch. No match
is a decomposition of psi_4, so chi(N^4) >= 7, and with the Lean rank-7
witness bounds/N-m4-upper-7.json, chi(N^4) = 7.

What this script re-runs: research/n4_rank6/aggregate.py checks the
partition against the enumerator's pivot pairs and the census file's hash,
re-runs the 6-cover kernel over every pivot pair and compares the counts,
re-enumerates the orbit-representative lists from the rank-5 census files
and compares them with the stored ones, checks that every batch output is
present, hashes as stored and matches the partition's geometry and hashes,
re-decides every stored hit exactly, fails on any refused or undecided
run, and re-runs two batches from scratch, chosen from the printed seed,
with their deterministic hashes compared bit for bit. The completeness of
the enumeration rests on the stored outputs and on the committed runner
(research/n4_rank6/batch.py) that regenerates any batch. The controls
(research/n4_rank6/results/control_*.json) are part of the record and are
not re-run here.

Printed claims: CERTIFIED chi(N^4) >= 7
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20260925
CLAIM = "CERTIFIED chi(N^4) >= 7"


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="n4_rank6_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "n4_rank6", "aggregate.py"),
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
