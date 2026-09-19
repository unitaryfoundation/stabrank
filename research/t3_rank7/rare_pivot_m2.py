"""Finite checks behind docs/notes/t3_rank7_rare_pivot.md (the rare-type pivot
lemma for the rank-7 scan of |T3>^3 and why the slice route to it fails).

Everything is exact. Ranks are computed mod P = 2^31 - 1; at m = 2 every
matrix has at most 9 columns, so every rank is at most 9 and 9^9 < P makes the
Hadamard argument of cert_t3m3_rank7 apply: a rank computed mod P is the rank
over Q(w3). The m = 3 part only uses the dictionary, its restriction to
coordinate slices, and the symmetry group of V_3 from the certificate.

Checks, in order:

1. Restriction of V_3 to each slice x_p = c is V_2 (each psi_r restricts to a
   scalar multiple of psi'_{r - c}), and every three-qutrit stabilizer state
   restricts to zero or to a scalar multiple of a two-qutrit stabilizer state.
2. At m = 2, exactly three states lie in V_2, they are psi'_0, psi'_1,
   psi'_2, and they span V_2 (the unique size-3 configuration).
3. At m = 2, every size-4 configuration (four independent states whose span
   contains V_2) contains one of the psi'_r.
4. At m = 2, there are size-5 configurations made of full-support states
   only: 27 class sets of nine states each, in one orbit of the symmetry
   group, the first of them invariant under x1 <-> x2; an explicit
   independent 5-subset is verified. Seven states of the other full-support
   orbit (size 162), given by their quadratic forms, are independent and span
   V_2 as well.
5. The lift table: for each orbit of the m = 2 symmetry group, the set of
   m = 3 states whose restriction to x3 = 0 lies in that orbit, closed under
   the m = 3 symmetry group. The point orbit lifts to exactly the 1080 point
   and line states (the rare set of the design note); the 81-orbit of check 4
   lifts to 19926 of the 30240 states.

Writes research/t3_rank7/results/rare_pivot_m2.json. Runtime about two minutes
(the m = 3 symmetry group is rebuilt in pure Python).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
import cert_t3m3_rank7 as C  # noqa: E402

P = C.ELL2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "rare_pivot_m2.json")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.1f}s] {msg}", flush=True)


def quotient_coords(E2, T2):
    """Exact coordinates of the rows of E2 in F_P^dim / span(T2), by eliminating
    the pivot columns of T2 (one per row, disjoint supports)."""
    R_, dim = T2.shape
    piv = []
    for r in range(R_):
        c = next(c for c in range(dim) if T2[r, c] != 0
                 and all(T2[s, c] == 0 for s in range(R_) if s != r))
        piv.append(c)
    Q = E2.copy() % P
    for r in range(R_):
        c = (Q[:, piv[r]] * pow(int(T2[r, piv[r]]), P - 2, P)) % P
        Q = (Q - np.outer(c, T2[r])) % P
    assert np.all(Q[:, piv] == 0)
    keep = [c for c in range(dim) if c not in piv]
    return np.ascontiguousarray(Q[:, keep])


def residues(Q, pivots):
    """Rows of Q reduced modulo the span of the pivot rows (which must be
    independent); a zero residue means the row's image lies in that span."""
    R = Q.copy() % P
    used = []
    for pv in pivots:
        row = R[pv].copy()
        a = next((d for d in range(R.shape[1]) if row[d] != 0 and d not in used), None)
        assert a is not None, "dependent pivots"
        used.append(a)
        f = (R[:, a] * pow(int(row[a]), P - 2, P)) % P
        R = (R - np.outer(f, row)) % P
    return R


def contains_target(E2, T2, members):
    S = E2[list(members)]
    rS = C.rank_mod(S, P)
    return rS, C.rank_mod(np.vstack([S, T2]), P) == rS


def restrict(E, p, c):
    """Restriction of every row of a 27-column exponent table to the slice
    x_p = c, in the coordinate order of the two-qutrit dictionary."""
    xs = np.array([[pos // 9, (pos // 3) % 3, pos % 3] for pos in range(27)])
    sel = np.flatnonzero(xs[:, p] == c)
    others = [q for q in range(3) if q != p]
    order = np.argsort(xs[sel][:, others[0]] * 3 + xs[sel][:, others[1]])
    return E[:, sel[order]]


def quadratic_form(row):
    """Coefficients (a, b, c, d, e, f) of Q = a x1^2 + b x2^2 + c x1 x2 + d x1 + e x2 + f
    with w3^Q the (full-support) state, or None."""
    vals = np.array([row[3 * a + b] // 2 for a in range(3) for b in range(3)])
    A = np.array([[a * a, b * b, a * b, a, b, 1] for a in range(3) for b in range(3)]) % 3
    for coef in itertools.product(range(3), repeat=6):
        if np.all((A @ np.array(coef)) % 3 == vals % 3):
            return coef
    return None


def main():
    out = {}
    # ------------------------------------------------------------ data ----
    E3 = C.build_dictionary(3)
    T3 = C.t3_targets(3)
    E2x = C.build_dictionary(2)
    T2x = C.t3_targets(2)
    N2, N3 = E2x.shape[0], E3.shape[0]
    keys2 = C.row_keys(E2x)
    idx2 = {k: i for i, k in enumerate(keys2)}
    tkeys2 = C.row_keys(T2x)
    log(f"dictionaries: {N3} states at m=3, {N2} at m=2")

    # ------------------------------------------ 1. slices restrict V_3 to V_2 ----
    zero_counts = {}
    for p in range(3):
        for c in range(3):
            R = restrict(T3, p, c)
            for r in range(3):
                k = C.row_keys(R[r:r + 1])[0]
                assert k in tkeys2 and tkeys2.index(k) == (r - c) % 3, (p, c, r)
            RE = restrict(E3, p, c)
            nz = np.any(RE != C.NOSUPP, axis=1)
            for k in C.row_keys(RE[nz]):
                assert k in idx2, "a restricted state is not a two-qutrit stabilizer state"
            zero_counts[f"x{p + 1}={c}"] = int((~nz).sum())
    assert all(v == 720 for v in zero_counts.values())
    out["slices"] = {"restriction_of_V3_is_V2": True, "states_restricting_to_zero": zero_counts}
    log("1. every slice restricts V_3 to V_2 and stabilizer states to stabilizer states "
        "(720 of 30240 restrict to zero on each slice)")

    # --------------------------------------------- m = 2 exact machinery ----
    E2 = C.to_mod(E2x, C.Z6_2, P)
    T2 = C.to_mod(T2x, C.Z6_2, P)
    Q = quotient_coords(E2, T2)
    isfree = np.all(Q == 0, axis=1)
    supp = E2x != C.NOSUPP
    ksz = supp.sum(axis=1)
    kdim = np.array([{1: 0, 3: 1, 9: 2}[int(s)] for s in ksz])
    elems2 = C.monomial_symmetries(T2x, 2)
    perms2 = np.empty((len(elems2), N2), dtype=np.int64)
    for n, e in enumerate(elems2):
        perms2[n] = np.fromiter((idx2[k] for k in C.row_keys(C.apply_element(E2x, e))),
                                dtype=np.int64, count=N2)
    gmin2 = perms2.min(axis=0)
    reps2 = np.unique(gmin2)
    orb2 = np.searchsorted(reps2, gmin2)
    osz2 = np.bincount(orb2)
    log(f"m=2 symmetry group of order {len(elems2)}, {len(reps2)} orbits of sizes {osz2.tolist()}")

    # ------------------------------------------ 2. the size-3 configuration ----
    free = np.flatnonzero(isfree)
    assert len(free) == 3
    assert sorted(keys2[i] for i in free) == sorted(tkeys2)
    rS, ok = contains_target(E2, T2, free)
    assert rS == 3 and ok
    out["size3"] = {"states_in_V2": [int(i) for i in free], "are_psi_prime": True}
    log(f"2. exactly three states lie in V_2, they are psi'_0, psi'_1, psi'_2 (indices "
        f"{free.tolist()}), and they span it")

    # ------------------------------- 3. every size-4 configuration has a psi' ----
    # A size-4 configuration has image span of dimension 1 over Q(w3). If it
    # avoids the psi'_r, all four states have nonzero image (only the psi'_r
    # have zero image, mod P and hence over Q(w3)); the images mod P of the
    # four are then proportional to q_i for any member i, so the configuration
    # lies inside the class set {j : q_j parallel to q_i, j not psi'} and V_2
    # would lie in that class set's span. The rank pair mod P is exact.
    bad = []
    for i in np.flatnonzero(~isfree):
        Ri = residues(Q, [i])
        cls = np.flatnonzero(np.all(Ri == 0, axis=1) & ~isfree)
        assert i in cls
        if len(cls) >= 4:
            rS, ok = contains_target(E2, T2, cls)
            if ok:
                bad.append([int(x) for x in cls])
    assert not bad
    out["size4"] = {"configurations_avoiding_psi_prime": 0}
    log("3. no size-4 configuration avoids the psi'_r")

    # --------------------- 4. size-5 configurations of full-support states ----
    full = np.flatnonzero(kdim == 2)
    Qf = Q[full]
    classes = {}
    for a in range(len(full)):
        Ra = residues(Qf, [a])
        for b in range(a + 1, len(full)):
            if np.all(Ra[b] == 0):
                continue
            Rab = residues(Ra, [b])
            cls = tuple(int(full[t]) for t in np.flatnonzero(np.all(Rab == 0, axis=1)))
            if len(cls) < 5 or cls in classes:
                continue
            rS, ok = contains_target(E2, T2, cls)
            if ok:
                classes[cls] = rS
    assert classes, "expected size-5 full-support configurations"
    assert all(r == 5 and len(c) == 9 for c, r in classes.items())
    # closure under the group, orbit, and the x1 <-> x2 symmetry
    closed = set()
    for cls in classes:
        for g in perms2:
            closed.add(tuple(sorted(int(g[s]) for s in cls)))
    assert closed == set(classes), "class sets not closed under the symmetry group"
    members = sorted(set(s for cls in classes for s in cls))
    orbs = sorted(set(int(orb2[s]) for s in members))
    first = min(classes)
    swap = np.array([3 * (pos % 3) + pos // 3 for pos in range(9)])
    for s in first:
        assert keys2[s] == C.row_keys(E2x[s:s + 1][:, swap])[0], "not swap-symmetric"
    forms = {int(s): quadratic_form(E2x[s]) for s in first}
    five = None
    n_indep = 0
    for sub in itertools.combinations(first, 5):
        rS, ok = contains_target(E2, T2, sub)
        if rS == 5:
            assert ok
            n_indep += 1
            if five is None:
                five = [int(s) for s in sub]
    assert five is not None
    out["size5_full_support"] = {
        "class_sets": len(classes), "class_set_size": 9, "class_set_rank": 5,
        "member_orbit_sizes": [int(osz2[o]) for o in orbs], "members_total": len(members),
        "first_class_set_swap_symmetric": True, "first_class_set": [int(s) for s in first],
        "quadratic_forms_a_b_c_d_e_f": forms, "explicit_five": five,
        "independent_5_subsets_of_first_class_set": n_indep,
    }
    log(f"4. {len(classes)} class sets of image dimension 2 among full-support states span V_2; "
        f"each has 9 states of rank 5, all in the orbit of size {osz2[orbs[0]]} "
        f"({len(members)} states); the first class set is x1 <-> x2 symmetric; explicit "
        f"configuration {five} "
        f"({n_indep} of 126 5-subsets of the first class set are independent)")
    for s in first:
        a, b, c, d, e, f = forms[int(s)]
        log(f"     state {s}: Q = {a} x1^2 + {b} x2^2 + {c} x1x2 + {d} x1 + {e} x2 + {f}")

    # ------------- 4b. a size-7 configuration inside the orbit of 162 full states ----
    forms7 = [(0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 2, 0), (0, 1, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0),
              (0, 1, 2, 1, 2, 0), (1, 2, 0, 2, 1, 0), (1, 1, 1, 1, 0, 0)]
    mons = np.array([[a * a, b * b, a * b, a, b, 1] for a in range(3) for b in range(3)]) % 3
    seven = []
    for coef in forms7:
        row = ((2 * ((mons @ np.array(coef)) % 3)) % 6).astype(np.int8)
        seven.append(idx2[C.row_keys(row[None, :])[0]])
    rS, ok = contains_target(E2, T2, seven)
    assert rS == 7 and ok
    orb7 = set(int(orb2[s]) for s in seven)
    assert len(orb7) == 1 and osz2[orb7.pop()] == 162
    out["size7_in_162_orbit"] = {"states": [int(s) for s in seven],
                                 "quadratic_forms_a_b_c_d_e_f": forms7}
    log(f"4b. seven full-support states of the orbit of size 162, {seven}, are independent and "
        f"span V_2")

    # ---------------------------------------------------- 5. the lift table ----
    RE = restrict(E3, 2, 0)
    nz = np.any(RE != C.NOSUPP, axis=1)
    rest = np.full(N3, -1, dtype=np.int64)
    rest[nz] = [idx2[k] for k in C.row_keys(RE[nz])]
    elems3 = C.monomial_symmetries(T3, 3)
    assert all(C.target_image_ok(T3, e) for e in elems3)
    idx3 = {k: i for i, k in enumerate(C.row_keys(E3))}
    gmin3 = np.arange(N3)
    for e in elems3:
        Pm = np.fromiter((idx3[k] for k in C.row_keys(C.apply_element(E3, e))), dtype=np.int64,
                         count=N3)
        gmin3 = np.minimum(gmin3, Pm)
    reps3 = np.unique(gmin3)
    orb3 = np.searchsorted(reps3, gmin3)
    osz3 = np.bincount(orb3)
    kdim3 = np.rint(np.log((E3 != C.NOSUPP).sum(axis=1)) / np.log(3)).astype(int)
    assert len(elems3) == 2916 and len(reps3) == 45
    log(f"m=3 symmetry group of order {len(elems3)}, {len(reps3)} orbits")
    table = []
    for o in range(len(reps2)):
        lifted = (rest >= 0) & (orb2[np.maximum(rest, 0)] == o)
        hit = np.unique(orb3[lifted])
        closure = np.isin(orb3, hit)
        row = {"m2_orbit": int(o), "m2_orbit_size": int(osz2[o]),
               "m2_support_dim": int(kdim[np.flatnonzero(orb2 == o)[0]]),
               "lift_size": int(lifted.sum()), "closure_size": int(closure.sum()),
               "closure_support_dims": sorted(set(int(k) for k in kdim3[closure]))}
        table.append(row)
        log(f"5. m=2 orbit {o} (size {osz2[o]}, support dim {row['m2_support_dim']}): lift "
            f"{row['lift_size']} states, closure {row['closure_size']} states of support dims "
            f"{row['closure_support_dims']}")
    pt = next(r for r in table if r["m2_support_dim"] == 0)
    assert pt["lift_size"] == 82 * 9 and pt["closure_size"] == 1080
    assert pt["closure_size"] == int((kdim3 <= 1).sum())
    big = next(r for r in table if r["m2_orbit"] == orbs[0])
    assert big["closure_size"] == 19926
    # the closure of the lift of a single m=2 state is already the closure of
    # its whole orbit, so any hitting set tau gives a pivot set containing the
    # closures for one state of each explicit configuration of check 4 and 4b
    single = {}
    for t in range(N2):
        hit = np.unique(orb3[rest == t])
        single[t] = frozenset(int(o) for o in hit)
    for o in range(len(reps2)):
        mem = np.flatnonzero(orb2 == o)
        assert len(set(single[int(t)] for t in mem)) == 1
        assert int(np.isin(orb3, list(single[int(mem[0])])).sum()) == table[o]["closure_size"]
    union = single[five[0]] | single[seven[0]]
    lower = int(np.isin(orb3, list(union)).sum())
    assert lower == 21870 and lower == int((kdim3 == 3).sum()) + 729 + 1458
    out["lift_table"] = table
    out["pivot_set_lower_bound_from_checks_4_4b"] = lower
    log(f"5. the lift closure of one state equals that of its orbit; any hitting set gives a "
        f"pivot set of at least {lower} states (all {int((kdim3 == 3).sum())} full-support states "
        f"and two plane orbits), a factor of at most {N3 / lower:.2f}")
    out["seconds"] = time.time() - T0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {OUT}")
    print("the point orbit at m=2 lifts to exactly the 1080 point and line states at m=3; "
          "the rare-type lemma would follow from the slice argument iff every V_2-spanning "
          "configuration of size <= 7 contained a point state, which fails at size 3 (the "
          "psi'_r) and, with the psi'_r added, at size 5 (check 4).")


if __name__ == "__main__":
    main()
