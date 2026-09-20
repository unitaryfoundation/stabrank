"""Dictionary states in the span of a minimal decomposition.

An (r+1)-term decomposition of phi^m whose terms span only r dimensions is a
minimal rank-r decomposition plus one dictionary state inside its span. If
no dictionary state other than the r terms lies in the span, every
(r+1)-term decomposition of phi^m has r+1 independent terms (an irreducible
one), which is what the all-full configurations of relaxed_lift.py would
need and what the stored minimal lists cannot supply.

Usage: inspan.py ORBIT M RANK
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ORBIT_P, load_decompositions  # noqa: E402
from rank_exclusion import dictionary  # noqa: E402


def main(argv):
    orbit, m, rank = argv[1], int(argv[2]), int(argv[3])
    p = ORBIT_P[orbit]
    if (p, m) not in {(3, 1), (3, 2), (3, 3), (2, 1), (2, 2), (2, 3), (2, 4)}:
        print(f"{orbit} m={m}: the {m}-qudit dictionary is too large to hold; not run")
        return 0
    decs, _ = load_decompositions(orbit, m, rank)
    D = dictionary(p, m)
    counts = []
    for u, d in decs:
        Q, _ = np.linalg.qr(np.column_stack(u))
        res = D - Q @ (Q.conj().T @ D)
        inside = np.flatnonzero(np.linalg.norm(res, axis=0) < 1e-7)
        counts.append(len(inside) - len(u))
    print(f"{orbit} m={m} rank {rank}: {len(decs)} decompositions; dictionary states in the span "
          f"beyond the terms themselves: {sorted(set(counts))} (per decomposition: {counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
