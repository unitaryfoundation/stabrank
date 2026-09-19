"""Exact inner-step counts for the three-pivot scan at m=3 under the symmetry
of V_3, with and without the Stab(i, j) reduction of the third pivot.

Reuses the symmetry construction of cert_t3m3_rank7 (about a minute of pure
Python) and writes research/t3_rank7/results/step_counts.json. The count
without Stab(i, j) is the sum over pivot pairs (i, j) of C(N - j - 1, 2),
which is what kernel3 performs up to the states skipped as Z members. The
count with Stab(i, j) restricts k > j to states minimal in their orbit under
the elements of Stab(i) that also fix j.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
import cert_t3m3_rank7 as C  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "step_counts.json")


def main():
    t0 = time.time()
    m = 3
    E = C.build_dictionary(m)
    N = E.shape[0]
    T = C.t3_targets(m)
    Vl, Tl = C.to_mod(E, C.Z6, C.ELL), C.to_mod(T, C.Z6, C.ELL)
    PD = C.quotient_projection(Vl, Tl, 6, 2024)
    isfree = np.all(PD == 0, axis=1).astype(np.int8)
    assert not isfree.any()
    print(f"[{time.time() - t0:.0f}s] dictionary {N}", flush=True)
    elems = C.monomial_symmetries(T, m)
    index = {k: i for i, k in enumerate(C.row_keys(E))}
    ident = np.arange(N)
    gmin = ident.copy()
    stab = {}
    for e in elems:
        P = np.fromiter((index[k] for k in C.row_keys(C.apply_element(E, e))), dtype=np.int64,
                        count=N)
        gmin = np.minimum(gmin, P)
        for i in np.flatnonzero(P == ident):
            stab.setdefault(int(i), []).append(P)
    reps = np.unique(gmin)
    print(f"[{time.time() - t0:.0f}s] group order {len(elems)}, {len(reps)} orbits", flush=True)
    pairs = 0
    steps_plain = 0          # k > j, l > k, no Stab(i, j) reduction
    steps_stab = 0           # k > j minimal under Stab(i, j), l > k
    pairs_nontrivial = 0
    per_rep = []
    for i in reps:
        S = stab[int(i)]          # includes the identity
        imin = ident.copy()
        for P in S:
            imin = np.minimum(imin, P)
        js = np.flatnonzero(imin == ident)
        pairs += len(js)
        rest = (N - js - 1).astype(np.int64)
        sp = int((rest * (rest - 1) // 2).sum())
        steps_plain += sp
        ss = 0
        for j in js:
            Sj = [P for P in S if P[j] == j]
            if len(Sj) <= 1:
                r = N - int(j) - 1
                ss += r * (r - 1) // 2
                continue
            pairs_nontrivial += 1
            kmin = ident.copy()
            for P in Sj:
                kmin = np.minimum(kmin, P)
            ks = np.flatnonzero(kmin[j + 1:] == ident[j + 1:]) + j + 1
            ss += int((N - ks - 1).sum())
        steps_stab += ss
        per_rep.append({"rep": int(i), "stab_order": len(S), "orbit_size": len(elems) // len(S),
                        "pairs": int(len(js)), "steps_plain": sp, "steps_stab": ss})
        print(f"[{time.time() - t0:.0f}s] rep {int(i)}: |Stab| {len(S)}, {len(js)} pairs, "
              f"{sp:.3e} plain, {ss:.3e} with Stab(i,j)", flush=True)
    # Orbit-block ordering: relabel the states so that the orbits form
    # contiguous blocks in increasing order of size. A configuration is moved
    # so that a member of its lowest block becomes the pivot i; every other
    # member then lies in a block at or above i's, so the scan for i runs on
    # the states with new label >= the block start, and j is minimal in its
    # Stab(i)-orbit with respect to the new labels.
    order = sorted(reps, key=lambda r: (len(elems) // len(stab[int(r)]), int(r)))
    newlabel = np.empty(N, dtype=np.int64)
    pos = 0
    block_start = {}
    for r in order:
        members = np.flatnonzero(gmin == r)
        block_start[int(r)] = pos
        newlabel[members] = np.arange(pos, pos + len(members))
        pos += len(members)
    assert pos == N
    pairs_blk = 0
    steps_blk = 0
    steps_blk_stab = 0
    per_rep_blk = []
    for r in order:
        S = stab[int(r)]
        start = block_start[int(r)]
        imin = newlabel.copy()
        for P in S:
            imin = np.minimum(imin, newlabel[P])
        js = np.flatnonzero((imin == newlabel) & (newlabel >= start))
        js = js[js != r]
        pairs_blk += len(js)
        rest = (N - newlabel[js] - 1).astype(np.int64)
        sp = int((rest * (rest - 1) // 2).sum())
        steps_blk += sp
        ss = 0
        for j in js:
            Sj = [P for P in S if P[j] == j]
            r_ = N - int(newlabel[j]) - 1
            if len(Sj) <= 1:
                ss += r_ * (r_ - 1) // 2
                continue
            kmin = newlabel.copy()
            for P in Sj:
                kmin = np.minimum(kmin, newlabel[P])
            ks = np.flatnonzero((kmin == newlabel) & (newlabel > newlabel[j]))
            ss += int((N - newlabel[ks] - 1).sum())
        steps_blk_stab += ss
        per_rep_blk.append({"rep": int(r), "orbit_size": len(elems) // len(S),
                            "block_start": int(start), "pairs": int(len(js)),
                            "steps_plain": sp, "steps_stab": ss})
        print(f"[{time.time() - t0:.0f}s] block rep {int(r)} (size {len(elems) // len(S)}, "
              f"start {start}): {len(js)} pairs, {sp:.3e} plain, {ss:.3e} with Stab(i,j)",
              flush=True)
    est = N ** 4 / (6 * len(elems))
    out = {"N": N, "group_order": len(elems), "orbits": int(len(reps)), "pairs": int(pairs),
           "pairs_with_nontrivial_stab_ij": int(pairs_nontrivial),
           "steps_plain": int(steps_plain), "steps_with_stab_ij": int(steps_stab),
           "estimate_N4_over_6G": est,
           "blocked_pairs": int(pairs_blk), "blocked_steps_plain": int(steps_blk),
           "blocked_steps_with_stab_ij": int(steps_blk_stab),
           "orbit_sizes_increasing": [len(elems) // len(stab[int(r)]) for r in order],
           "per_rep": per_rep, "per_rep_blocked": per_rep_blk,
           "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_rep"}, indent=1))


if __name__ == "__main__":
    main()
