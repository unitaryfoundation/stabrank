"""Certificate: chi(|T3>) >= 3, settling chi(|T3>) = 3 with the rank-3 upper bound.

|T3> = (|0> + w9|1> + w9^2|2>)/sqrt(3) lives on one qutrit, where there are
only 12 stabilizer states, so the rank-2 question is 66 pairs and is settled by
exhaustion rather than by the Galois argument. psi lies in the span of two
stabilizer states exactly when those two have parallel components off psi, so
quotienting by psi once turns the pair search into finding parallel directions.
Rank 1 falls out of the same pass, since a state parallel to psi would leave no
component at all.

The exclusion is numerical and rests on a margin: the closest any two candidate
directions come to parallel is reported below and clears the threshold by a
wide gap, against float error around 1e-15 on these inputs.

The Strange state at m=1 is the positive control. It has a known rank-2
decomposition, the same code must find it, and this script exits non-zero if it
ever stops finding it.

Printed claim: CERTIFIED chi(T3^1) >= 3
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank.examples.t3_galois_lower_bound import distinct_states
from stabrank_verify import target_vector

PARALLEL = 1e-6
MARGIN = 0.01


def rank2_search(psi, D):
    """(pairs spanning psi, closest approach to parallel among all pairs)."""
    psi = psi / np.linalg.norm(psi)
    q = D - np.outer(psi, psi.conj() @ D)
    nq = np.linalg.norm(q, axis=0)
    if (nq <= 1e-9).any():
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


def psi_for(orbit, m):
    return np.array([complex(x) for x in target_vector(orbit, m)]).ravel()


def main():
    D = distinct_states(1)
    print(f"{D.shape[1]} distinct 1-qutrit stabilizer states, "
          f"{D.shape[1] * (D.shape[1] - 1) // 2} pairs\n")

    control, _ = rank2_search(psi_for("S", 1), D)
    if control == "RANK1" or not control:
        print("positive control FAILED: no rank-2 decomposition found for the "
              "Strange state at m=1, where one is known; the search is broken",
              file=sys.stderr)
        return 1
    print(f"positive control: Strange at m=1 found {len(control)} rank-2 pair(s), "
          f"as expected\n")

    pairs, worst = rank2_search(psi_for("T3", 1), D)
    if pairs == "RANK1" or pairs:
        print(f"T3 m=1: rank-2 decomposition FOUND, {pairs}")
        return 1
    slack = (1 - PARALLEL) - worst
    print(f"T3 m=1: no pair spans it. Closest approach to parallel "
          f"{worst:.10f}, clearing the threshold by {slack:.4f}")
    if slack < MARGIN:
        print("margin too thin to stand behind", file=sys.stderr)
        return 1
    print("CERTIFIED chi(T3^1) >= 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
