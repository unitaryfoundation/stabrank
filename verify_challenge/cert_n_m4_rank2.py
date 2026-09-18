"""Certificate: chi(|N>^{ot 4}) >= 3, by excluding every rank-2 decomposition
over all 7,439,040 four-qutrit stabilizer states.

The dictionary is enumerated as phase codes (qutrit_codes.py, count asserted
against 3^4 prod (3^j + 1)) and never held as complex amplitudes. psi lies in
the span of two stabilizer states exactly when their components off psi are
parallel, so one canonicalise-and-sort over the quotient directions decides
rank 2; each flagged pair is confirmed in the full space. The exclusion is
numerical and rests on the reported parallelism margin; the script refuses
to claim if it is under 0.01. Positive control: the same code path finds
the known rank-2 decomposition of the Strange state at m=2.

This runs in about two minutes. The rank-3 exclusion for the same cell
(cert_n_m4_rank3.py) is the stronger certificate and takes longer than
the default budget.

Printed claim: CERTIFIED chi(N^4) >= 3
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qutrit_codes import all_codes  # noqa: E402
from rank_exclusion_codes import _Geometry, rank2_search_codes, psi_for  # noqa: E402
from rank_exclusion import PARALLEL, MARGIN  # noqa: E402


def main():
    c2, k2 = all_codes(2)
    control, _ = rank2_search_codes(_Geometry(psi_for("S", 2), c2, k2))
    if control == "RANK1" or not control:
        print("positive control FAILED: the code path did not find the rank-2 decomposition "
              "of the Strange state at m=2", file=sys.stderr)
        return 1
    print(f"positive control: Strange m=2 found {len(control)} rank-2 pair(s)")
    codes, ks = all_codes(4)
    print(f"4 qutrits: {codes.shape[0]} states")
    pairs, worst = rank2_search_codes(_Geometry(psi_for("N", 4), codes, ks))
    if pairs == "RANK1" or pairs:
        print(f"N m=4: rank <= 2 decomposition FOUND {pairs if pairs != 'RANK1' else ''}")
        return 1
    slack = (1 - PARALLEL) - worst
    print(f"N m=4: no pair spans it; closest approach {worst:.6f}, clearing by {slack:.4f}")
    if slack < MARGIN:
        print("margin too thin to stand behind", file=sys.stderr)
        return 1
    print("CERTIFIED chi(N^4) >= 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
