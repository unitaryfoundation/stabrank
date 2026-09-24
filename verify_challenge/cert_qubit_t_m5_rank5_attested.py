"""Certificate: chi(|T>^{ot 5}) >= 6, by the exclusion of every rank-5
decomposition through a two-qubit base slice at the base point 00, over
stored batch outputs (attested tier).

The argument (docs/notes/t5_rank5_exclusion.md): in a rank-5 decomposition
of psi_5 = |T>^5 sliced along qubits 1, 2, every point of F_2^2 sees at
least three terms (Fact 2, chi(T^3) = 3), so at the base point 00 three,
four or five terms are visible and the base is a full k-cover of |T>^3
with k in {3, 4, 5}; the 5 - k invisible terms lie on the six affine flats
of F_2^2 missing 00 (three points, three lines). The three stages: (alpha),
k = 5, the full 5-covers of |T>^3 up to the unitary symmetry of psi_3
(6,115,136 covers of distinct independent states from the modular pivot
enumeration, regenerated per pivot pair, and 43,773 dependent or repeated
multisets in research/t5_rank5/degenerate5.json, the cancel-at-base
multisets included); (beta'), k = 4, the 4,709 full 4-covers of
research/t5_rank4/covers4.json, each run with the invisible term on every
one of the six flats; (gamma), k = 3, the 4 full 3-covers, each run with
every one of the 21 flat pairs. research/t5_rank5 completes every base
slice by slice under the slice structure lemma, modulo 65521 (a superset
of the true matches, the amplitudes reduced from Q(zeta_24)) with every
candidate re-decided mod 2013265921 and in floating point; a configuration
a matcher cannot decide raises and fails its batch. No match is a
decomposition of psi_5, so chi(T^5) >= 6, and with the Lean rank-6 witness
bounds/qubit_T-m5-upper-6.json, chi(T^5) = 6; by projection chi(T^6) >= 6,
which with bounds/qubit_T-m6-upper-6.json gives chi(T^6) = 6.

What this script re-runs: research/t5_rank5/aggregate.py checks the
partition against the enumerator's pivot pairs and the census file's hash,
re-enumerates the 3-covers, the 4-covers and the degenerate 5-multisets
(about two minutes) and compares them with the stored lists, checks that
every batch output is present, hashes as stored and matches the
partition's geometry and hashes, re-decides every stored hit exactly, fails
on any refused or undecided run, and re-runs two batches from scratch,
chosen from the printed seed, with their deterministic hashes compared bit
for bit. The completeness of the enumeration rests on the stored outputs
and on the committed runner (research/t5_rank5/batch.py) that regenerates
any batch. The controls (research/t5_rank5/results/control_*.json) are
part of the record and are not re-run here.

Printed claims: CERTIFIED chi(qubit_T^5) >= 6
                CERTIFIED chi(qubit_T^6) >= 6 (projection monotonicity)
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECHECK, SEED = 2, 20260924
CLAIM = "CERTIFIED chi(qubit_T^5) >= 6"


def main():
    print(f"seed: {SEED}")
    out = tempfile.mkdtemp(prefix="t5_rank5_recheck_")
    cmd = [sys.executable, os.path.join(ROOT, "research", "t5_rank5", "aggregate.py"),
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
    # lean_proofs/LeanProofs/Stabilizer/SliceP.lean): both amplitudes of |T>
    # are nonzero, so slicing carries a rank-r decomposition of T^6 to one of
    # T^5, and the m=5 exclusion bounds the m=6 cell as well.
    print("CERTIFIED chi(qubit_T^6) >= 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
