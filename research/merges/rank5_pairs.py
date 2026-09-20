"""Minimal rank-5 decompositions of |N>^3 or |H3>^3 that contain a given pair
of dictionary states, one rank above slice_lift.decompositions_with_pivot.

Why: any rank-5 or rank-6 decomposition of |N>^4 or |H3>^4 has no minimal
slice (docs/notes/constructions_2026_09.md), so every slice is an
irreducible 5- or 6-term decomposition of the three-copy state, and the
stored minimal lists cannot supply it. Whether irreducible rank-5
decompositions of |N>^3 exist at all, and how many contain a term of the
unique rank-4 decomposition, is the question here.

Method: fix the pair (i, j). Project psi and the whole dictionary off s_j;
a set {i, j, a, b, c} spans psi exactly when {i, a, b, c} spans the
projected psi in the projected dictionary, so the compiled rank-4 pivot
search (pivot i, quotient by psi, s_i and the partner a, parallel pairs
b, c) does the rank-5 search with j fixed. Every hit is re-solved in the
full space and kept only if it is minimal (no four of the five span psi).

Partners j for a fixed pivot i run over one representative per orbit of the
stabilizer of s_i inside the unitary symmetry group of the target
(slice_lift.stabilizer_orbit_labels), so a "count" is a count of orbits of
pairs, each multiplied by its orbit size when the total is reported.

Usage: rank5_pairs.py ORBIT [--pivot-term 0] [--budget-s 780] [--partners N]
       ORBIT in {N, H3}; --pivot-term selects which of the four terms of the
       stored rank-4 decomposition is the pivot i (in data order).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mergelib import load_decompositions  # noqa: E402
from rank_exclusion import dictionary, psi_for, symmetry_orbit_reps  # noqa: E402
from slice_lift import decompositions_with_pivot, stabilizer_orbit_labels  # noqa: E402


def locate(D, v):
    ov = np.abs(v.conj() @ D) / np.linalg.norm(v)
    k = int(np.argmax(ov))
    assert ov[k] > 1 - 1e-8, "a stored term is not in the dictionary"
    return k


def minimal(D, psi, cols):
    A = D[:, list(cols)]
    if np.linalg.matrix_rank(A, tol=1e-8) < len(cols):
        return False
    x, *_ = np.linalg.lstsq(A, psi, rcond=None)
    if np.linalg.norm(A @ x - psi) > 1e-9:
        return False
    for drop in cols:
        sub = [c for c in cols if c != drop]
        B = D[:, sub]
        y, *_ = np.linalg.lstsq(B, psi, rcond=None)
        if np.linalg.norm(B @ y - psi) < 1e-8:
            return False
    return True


def rank5_with_pair(psi, D, i, j):
    """Sorted index tuples of minimal rank-5 decompositions of psi containing i and j."""
    e = D[:, j] / np.linalg.norm(D[:, j])
    psi_p = psi - e * (e.conj() @ psi)
    Dp = D - np.outer(e, e.conj() @ D)
    members = np.array([k for k in range(D.shape[1]) if k != j])
    quads = decompositions_with_pivot(psi_p / np.linalg.norm(psi_p), Dp, i, 4, members=members)
    out = set()
    for q in quads:
        cols = tuple(sorted(set(q) | {j}))
        if len(cols) == 5 and minimal(D, psi, cols):
            out.add(cols)
    return sorted(out)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit", choices=["N", "H3"])
    ap.add_argument("--pivot-term", type=int, default=0)
    ap.add_argument("--budget-s", type=float, default=780)
    ap.add_argument("--partners", type=int, default=0, help="stop after this many partner orbits (0 = budget only)")
    a = ap.parse_args(argv[1:])
    t0 = time.time()
    D = dictionary(3, 3)
    psi = psi_for(a.orbit, 3)
    psi = psi / np.linalg.norm(psi)
    decs, _ = load_decompositions(a.orbit, 3, 4)
    known = [locate(D, v) for v in decs[0][0]]
    reps, info = symmetry_orbit_reps(a.orbit, 3, D, antiunitary=False)
    i = known[a.pivot_term]
    labels, ngens = stabilizer_orbit_labels(info["perms"], i)
    orbit_of = {}
    for k, lab in enumerate(labels):
        orbit_of.setdefault(int(lab), []).append(k)
    partner_orbits = [members for lab, members in sorted(orbit_of.items()) if i not in members]
    print(f"{a.orbit}^3: dictionary {D.shape[1]}, symmetry group order {info['order']}, known rank-4 terms "
          f"{known}; pivot s_{i} (term {a.pivot_term}, flat dim of support {int(round(np.log(np.count_nonzero(np.abs(D[:, i]) > 1e-9)) / np.log(3)))}); "
          f"{len(partner_orbits)} partner orbits under Stab(s_i) ({ngens} Schreier generators) [{time.time() - t0:.0f}s]", flush=True)
    total = 0
    weighted = 0
    done = 0
    per = []
    known_set = set(known)
    for members in partner_orbits:
        if time.time() - t0 > a.budget_s or (a.partners and done >= a.partners):
            break
        j = members[0]
        t1 = time.time()
        found = rank5_with_pair(psi, D, i, j)
        dt = time.time() - t1
        per.append(dt)
        done += 1
        total += len(found)
        weighted += len(found) * len(members)
        tag = " (a known rank-4 term)" if j in known_set else ""
        print(f"  partner orbit of s_{j}{tag}, size {len(members)}: {len(found)} minimal rank-5 decompositions "
              f"containing the pair [{dt:.1f}s]", flush=True)
        for cols in found[:5]:
            print(f"    {cols}")
    remaining = len(partner_orbits) - done
    mean = float(np.mean(per)) if per else float("nan")
    print(f"done {done} of {len(partner_orbits)} partner orbits in {time.time() - t0:.0f}s "
          f"(mean {mean:.1f}s per orbit); minimal rank-5 decompositions found: {total} orbit representatives, "
          f"{weighted} pairs counted with orbit sizes; remaining orbits {remaining}, "
          f"estimated {remaining * mean / 60:.0f} min for this pivot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
