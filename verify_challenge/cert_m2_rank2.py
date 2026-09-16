"""Certificates: chi(|N>^{ot 2}) >= 3 and chi(|H3>^{ot 2}) >= 3.

Both orbits have a lean-certified rank-3 decomposition at m=2, so excluding
rank 2 settles the cell exactly and removes it as a route to beating the
published exponent, which would need rank 2 there.

psi lies in the span of two stabilizer states exactly when those two states
have parallel components off psi. Quotienting by psi once therefore turns the
search over all 64620 pairs into finding parallel directions among 360
vectors: canonicalise each direction by a generic linear functional, sort, and
compare adjacent entries. Rank 1 is excluded in the same pass, since a state
parallel to psi leaves no component at all.

The exclusion is numerical, so it rests on a margin rather than on exact
arithmetic: the closest any two quotient directions come to parallel is
reported, and it is 0.18 or more below the threshold in every excluded case.
Float error on these inputs is around 1e-15, so no rounding can close that gap.
S at m=2 is the positive control. It has a known rank-2 decomposition, the same
code must find it, and the script fails if it does not: a certificate that can
only ever print success is worth nothing.

Printed claims: CERTIFIED chi(N^2) >= 3
                CERTIFIED chi(H3^2) >= 3
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank.examples.t3_galois_lower_bound import distinct_states
from stabrank_verify import target_vector

PARALLEL = 1e-6          # |<u,v>| above 1 - PARALLEL counts as parallel
MARGIN = 0.01            # an exclusion must clear the threshold by at least this


def rank2_search(psi, D):
    """(pairs spanning psi, closest approach to parallel among all pairs)."""
    psi = psi / np.linalg.norm(psi)
    q = D - np.outer(psi, psi.conj() @ D)
    nq = np.linalg.norm(q, axis=0)
    live = nq > 1e-9                     # a state parallel to psi would make it rank 1
    if not live.all():
        return "RANK1", 1.0
    u = q / nq
    rng = np.random.default_rng(7)
    can = rng.normal(size=u.shape[0]) + 1j * rng.normal(size=u.shape[0])
    key = rng.normal(size=u.shape[0]) + 1j * rng.normal(size=u.shape[0])
    c = can @ u
    ok = np.abs(c) > 1e-9
    u, idx, c = u[:, ok], np.where(ok)[0], c[ok]
    order = np.argsort((key @ u) / c)
    us, ids = u[:, order], idx[order]
    ov = np.abs(np.sum(us[:, :-1].conj() * us[:, 1:], axis=0))
    pairs = [(int(ids[a]), int(ids[a + 1])) for a in np.where(ov > 1 - PARALLEL)[0]]
    return pairs, float(ov.max())


def main():
    D = distinct_states(2)
    print(f"{D.shape[1]} distinct 2-qutrit stabilizer states, "
          f"{D.shape[1] * (D.shape[1] - 1) // 2} pairs\n")

    control, _ = rank2_search(
        np.array([complex(x) for x in target_vector("S", 2)]).ravel(), D)
    if control == "RANK1" or not control:
        print("positive control FAILED: no rank-2 decomposition found for S at "
              "m=2, where one is known to exist; the search is broken",
              file=sys.stderr)
        return 1
    print(f"positive control: S at m=2 found {len(control)} rank-2 pairs, as expected\n")

    ok = True
    for orbit in ("N", "H3"):
        psi = np.array([complex(x) for x in target_vector(orbit, 2)]).ravel()
        pairs, worst = rank2_search(psi, D)
        if pairs == "RANK1" or pairs:
            print(f"{orbit}: rank-2 decomposition FOUND, {pairs}")
            ok = False
            continue
        slack = (1 - PARALLEL) - worst
        print(f"{orbit}: no pair spans it. Closest approach to parallel "
              f"{worst:.12f}, clearing the threshold by {slack:.4f}")
        if slack < MARGIN:
            print(f"{orbit}: margin too thin to stand behind", file=sys.stderr)
            ok = False
            continue
        print(f"CERTIFIED chi({orbit}^2) >= 3")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
