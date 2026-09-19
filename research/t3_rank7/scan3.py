"""Three-pivot scan prototype for the exact exclusion of rank 7 for |T3>^3.

Companion to docs/notes/t3_rank7_exclusion.md. Everything here reuses the
conventions of verify_challenge/cert_t3m3_rank7.py: Galois descent to V_3
over Q(w3), reduction mod ell = 65521, a random linear projection of the
quotient Q(w3)^27 / V_3 to F_ell^6, exact hashing of canonical projective
coordinates, and one rank pair per candidate class mod 2^31 - 1.

A rank-7 configuration is seven linearly independent stabilizer states whose
images in the quotient span exactly four dimensions (the rank <= 6 cases are
excluded by the merged certificate). `kernel3` enumerates, for a first pivot i
and second pivots j, every class set: states with index above a third pivot k
whose images modulo span(q_i, q_j, q_k) vanish or are mutually parallel,
together with the states above j whose images vanish modulo span(q_i, q_j)
(the "Z members"). Every 7-set with image span <= 4 that contains i and j
as its two least members (in the pivot order of the note) lies inside one
class set, and V_3 lies in the span of some 7 members of a class set iff it
lies in the span of the whole class set.

Validation entry points (see README.md): `brute_force_check`,
`planted_check`, `m2_symmetry_check`, `m2_kernel2_consistency`, and
`time_m3_pairs`. None of them is a certificate.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
from numba import njit, int64, uint64

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
import cert_t3m3_rank7 as C  # noqa: E402

ELL, ELL2, INV = C.ELL, C.ELL2, C.INV
TSIZE = C.TSIZE


# ------------------------------------------------------------- kernel -----

@njit(cache=True)
def kernel3(PD, inv, i, jok, isfree, need, ell, max_out, max_memb, record):
    """Three-pivot scan for first pivot i and second pivots j with jok[j] != 0.

    Order: j is the least non-pivot member after minimisation under Stab(i),
    k is the least member outside span(V_3, s_i, s_j) (any index above j;
    Stab(i, j) fixes that subspace so the further minimisation of the note
    would apply here), the other members have index above k or lie in
    span(V_3, s_i, s_j) with index above j.

    Returns (classes, members, steps, ncand_total, npar) where classes rows
    are (j, k, nfree_total, ngroup), members rows are (class id, state), and
    steps counts every row reduction of the innermost loop. With record == 0
    nothing is stored and only the counts are returned.
    """
    N, D = PD.shape
    out_cand = np.empty((max_out, 4), dtype=np.int64)
    out_memb = np.empty((max_memb, 2), dtype=np.int64)
    nc = 0
    nm = 0
    ncand_total = 0
    steps = 0
    keys = np.zeros(N, dtype=np.uint64)
    tkey = np.zeros(TSIZE, dtype=np.uint64)
    tcnt = np.zeros(TSIZE, dtype=np.int64)
    tstamp = np.zeros(TSIZE, dtype=np.int64)
    slot_of = np.zeros(N, dtype=np.int64)
    stamp = 0
    Q1 = C._reduce_one(PD, inv, i, ell)
    D1 = D - 1
    skip1 = isfree.copy()
    nfree = 0
    for k in range(N):
        if isfree[k] != 0:
            nfree += 1
    npar = 0
    for k in range(N):
        if k != i and isfree[k] == 0:
            z = True
            for d in range(D1):
                if Q1[k, d] != 0:
                    z = False
                    break
            if z:
                skip1[k] = 1
                npar += 1
    skip1[i] = 1
    Q2 = np.empty((N, D1 - 1), dtype=np.int64)
    Q3 = np.empty((N, D1 - 2), dtype=np.int64)
    skip2 = np.empty(N, dtype=np.int8)
    zbuf = np.empty(N, dtype=np.int64)
    for j in range(N):
        if jok[j] == 0 or skip1[j] != 0:
            continue
        qj = Q1[j]
        b = -1
        for d in range(D1):
            if qj[d] != 0:
                b = d
                break
        ib = inv[qj[b]]
        for l in range(N):
            skip2[l] = skip1[l]
        nz = 0
        for l in range(j + 1, N):
            if skip1[l] != 0:
                continue
            c = (Q1[l, b] * ib) % ell
            col = 0
            zero = True
            for d in range(D1):
                if d == b:
                    continue
                v = (Q1[l, d] - c * qj[d]) % ell
                Q2[l, col] = v
                col += 1
                if v != 0:
                    zero = False
            if zero:
                zbuf[nz] = l
                nz += 1
                skip2[l] = 1
        steps += N - j - 1
        nfree_j = nfree + npar + nz
        for k in range(j + 1, N):
            if skip2[k] != 0:
                continue
            qk = Q2[k]
            b2 = -1
            for d in range(D1 - 1):
                if qk[d] != 0:
                    b2 = d
                    break
            ib2 = inv[qk[b2]]
            for l in range(k + 1, N):
                c = (Q2[l, b2] * ib2) % ell
                col = 0
                for d in range(D1 - 1):
                    if d == b2:
                        continue
                    Q3[l, col] = (Q2[l, d] - c * qk[d]) % ell
                    col += 1
            steps += N - k - 1
            stamp += 1
            nzero = C._group_keys(Q3, inv, ell, k + 1, -1, skip2, keys, tkey, tcnt, tstamp,
                                  stamp, slot_of)
            nzero_total = nzero + nfree_j
            if nzero_total >= need:
                ncand_total += 1
                if record != 0 and nc < max_out:
                    out_cand[nc, 0] = j
                    out_cand[nc, 1] = k
                    out_cand[nc, 2] = nzero_total
                    out_cand[nc, 3] = 0
                    for l in range(k + 1, N):
                        if slot_of[l] == -2 and nm < max_memb:
                            out_memb[nm, 0] = nc
                            out_memb[nm, 1] = l
                            nm += 1
                    for t in range(nz):
                        if nm < max_memb:
                            out_memb[nm, 0] = nc
                            out_memb[nm, 1] = zbuf[t]
                            nm += 1
                    nc += 1
            for l in range(k + 1, N):
                s = slot_of[l]
                if s < 0:
                    continue
                if tcnt[s] > 0 and tcnt[s] + nzero_total >= need:
                    ncand_total += 1
                    if record != 0 and nc < max_out:
                        out_cand[nc, 0] = j
                        out_cand[nc, 1] = k
                        out_cand[nc, 2] = nzero_total
                        out_cand[nc, 3] = tcnt[s]
                        h = keys[l]
                        for ll in range(k + 1, N):
                            if slot_of[ll] == -2 or (slot_of[ll] >= 0 and keys[ll] == h):
                                if nm < max_memb:
                                    out_memb[nm, 0] = nc
                                    out_memb[nm, 1] = ll
                                    nm += 1
                        for t in range(nz):
                            if nm < max_memb:
                                out_memb[nm, 0] = nc
                                out_memb[nm, 1] = zbuf[t]
                                nm += 1
                        nc += 1
                    tcnt[s] = -1
    return out_cand[:nc], out_memb[:nm], steps, ncand_total, npar


@njit(cache=True)
def rank_mod_small(A, ell, inv):
    """Rank of a small integer matrix mod a prime ell, in place on a copy."""
    M = A.copy() % ell
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pr = -1
        for t in range(r, rows):
            if M[t, c] != 0:
                pr = t
                break
        if pr < 0:
            continue
        if pr != r:
            for d in range(cols):
                tmp = M[r, d]
                M[r, d] = M[pr, d]
                M[pr, d] = tmp
        f = inv[M[r, c]]
        for d in range(cols):
            M[r, d] = (M[r, d] * f) % ell
        for t in range(rows):
            if t != r and M[t, c] != 0:
                g = M[t, c]
                for d in range(cols):
                    M[t, d] = (M[t, d] - g * M[r, d]) % ell
        r += 1
        if r == rows:
            break
    return r


# ------------------------------------------------------------ wrappers -----

def scan_pivot(PD, i, jok, isfree, need, max_out=1 << 20, max_memb=1 << 24, record=True):
    """Class sets for first pivot i: list of (i, j, k, nfree_total, ngroup, members)
    where members includes the Z members and every free state, plus the scan
    statistics (steps, candidates, parallel-to-pivot count). With record=False
    only the statistics are meaningful and the class list is empty."""
    isfree = np.ascontiguousarray(isfree, dtype=np.int8)
    jok = np.ascontiguousarray(jok, dtype=np.int8)
    cand, memb, steps, ntot, npar = kernel3(PD, INV, int(i), jok, isfree, int(need), ELL,
                                            max_out if record else 1, max_memb if record else 1,
                                            1 if record else 0)
    if record and ntot > cand.shape[0]:
        raise RuntimeError(f"candidate buffer overflow: {ntot} > {cand.shape[0]}")
    free = [int(k) for k in np.flatnonzero(isfree)]
    par = []
    if npar:
        Q1 = C._reduce_one(PD, INV, int(i), ELL)
        par = [int(k) for k in range(PD.shape[0])
               if k != i and not isfree[k] and not Q1[k].any()]
    by = {}
    for cid, l in memb:
        by.setdefault(int(cid), []).append(int(l))
    classes = [(int(i), int(cand[c, 0]), int(cand[c, 1]), int(cand[c, 2]), int(cand[c, 3]),
                by.get(c, []) + free + par) for c in range(cand.shape[0])]
    return classes, {"steps": int(steps), "candidates": int(ntot), "npar": int(npar)}


def decide(E2, T2, classes):
    """Exact decision per class set mod ELL2.

    Returns (found, spurious, hist). `found` lists class sets whose span
    contains span(T2) with their rank; `spurious` counts class sets of rank
    at least 8, which cannot lie in a 7-dimensional space and so came from a
    projection collision (a full certificate would re-split them exactly).
    """
    found, spurious, hist = [], 0, {}
    for i, j, k, nz, ng, members in classes:
        cls = sorted(set([i, j, k] + members))
        hist[(nz, ng)] = hist.get((nz, ng), 0) + 1
        S = E2[cls]
        rS = C.rank_mod(S, ELL2)
        if rS >= 8:
            spurious += 1
            continue
        rST = C.rank_mod(np.vstack([S, T2]), ELL2)
        if rST == rS:
            found.append((cls, rS))
    return found, spurious, hist


def rref_mod(T, ell):
    """Reduced row echelon form mod ell of the rows of T (rows assumed independent)."""
    A = T.copy() % ell
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        nz = np.flatnonzero(A[r:, c])
        if len(nz) == 0:
            continue
        pr = r + nz[0]
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        A[r] = (A[r] * pow(int(A[r, c]), ell - 2, ell)) % ell
        f = A[:, c].copy()
        f[r] = 0
        A = (A - np.outer(f, A[r])) % ell
        r += 1
        if r == rows:
            break
    assert r == rows, "target rows are dependent mod ell"
    return A


def setup(m, T_ell=None, seed=2024, D=6):
    """Dictionary and projected quotient images for target rows T_ell (mod ELL);
    default is V_m. Returns E, E2, T2, PD, isfree."""
    E = C.build_dictionary(m)
    Vl = C.to_mod(E, C.Z6, ELL)
    E2 = C.to_mod(E, C.Z6_2, ELL2)
    if T_ell is None:
        T = C.t3_targets(m)
        Tl = C.to_mod(T, C.Z6, ELL)
        T2 = C.to_mod(T, C.Z6_2, ELL2)
    else:
        Tl = rref_mod(np.asarray(T_ell, dtype=np.int64), ELL)
        T2 = None
    PD = C.quotient_projection(Vl, Tl, D, seed)
    isfree = np.all(PD == 0, axis=1).astype(np.int8)
    return E, E2, T2, PD, isfree, Tl


def scan_all(PD, isfree, need, reps=None, jok_of=None, verbose=False, record=True):
    """Scan every first pivot. With reps/jok_of absent the symmetry is trivial
    and the pivot order is i < j < k < others."""
    N = PD.shape[0]
    if reps is None:
        reps = [i for i in range(N) if not isfree[i]]
    classes, stats = [], {"steps": 0, "candidates": 0, "npar": 0}
    t0 = time.time()
    for n, i in enumerate(reps):
        if jok_of is None:
            jok = np.zeros(N, dtype=np.int8)
            jok[i + 1:] = 1
            jok[np.flatnonzero(isfree)] = 0
        else:
            jok = jok_of[int(i)]
        c, s = scan_pivot(PD, int(i), jok, isfree, need, record=record)
        classes.extend(c)
        for key in stats:
            stats[key] += s[key]
        if verbose and (n % 50 == 0 or n == len(reps) - 1):
            print(f"  pivot {n + 1}/{len(reps)}: {len(classes)} classes, {stats['steps']:.3e} "
                  f"steps, {time.time() - t0:.1f}s", flush=True)
    stats["seconds"] = time.time() - t0
    return classes, stats


def image_space_key(PD, members):
    """Canonical key of the projected image space spanned by the members."""
    A = np.array(PD[sorted(set(members))], dtype=np.int64)
    R = rref_mod_partial(A, ELL)
    return R.tobytes()


def rref_mod_partial(A, ell):
    """RREF mod ell keeping only the nonzero rows (rank rows), as a key."""
    M = A.copy() % ell
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0:
            continue
        pr = r + nz[0]
        if pr != r:
            M[[r, pr]] = M[[pr, r]]
        M[r] = (M[r] * pow(int(M[r, c]), ell - 2, ell)) % ell
        f = M[:, c].copy()
        f[r] = 0
        M = (M - np.outer(f, M[r])) % ell
        r += 1
        if r == rows:
            break
    return M[:r]


# ----------------------------------------------------------- validation -----

def brute_force_check(m=2, n_sub=24, seed=1, plant=True, verbose=True):
    """Exhaustive cross-check on a random sub-dictionary.

    Every 7-subset of the sub-dictionary whose projected images span at most
    four dimensions mod ell must be contained in some class set of the
    three-pivot scan with trivial symmetry. With `plant`, the target space
    is a random 3-dimensional subspace of the span of seven random states of
    the sub-dictionary, so at least one such 7-set exists.
    """
    rng = np.random.default_rng(seed)
    E = C.build_dictionary(m)
    N = E.shape[0]
    El = C.to_mod(E, C.Z6, ELL)
    sub = np.sort(rng.choice(N, size=n_sub, replace=False))
    if plant:
        star = rng.choice(n_sub, size=7, replace=False)
        coeffs = rng.integers(1, ELL, size=(3, 7))
        T_ell = (coeffs @ El[sub[star]]) % ELL
        Tl = rref_mod(T_ell, ELL)
    else:
        Tl = C.to_mod(C.t3_targets(m), C.Z6, ELL)
    PD = C.quotient_projection(El[sub], Tl, 6, int(rng.integers(1 << 30)))
    isfree = np.all(PD == 0, axis=1).astype(np.int8)
    classes, stats = scan_all(PD, isfree, 4)
    covered = [frozenset([i, j, k] + mem) for i, j, k, _, _, mem in classes]
    hits = []
    for sub7 in itertools.combinations(range(n_sub), 7):
        r = rank_mod_small(PD[list(sub7)], ELL, INV)
        if r <= 4:
            hits.append(frozenset(sub7))
    missing = [h for h in hits if not any(h <= c for c in covered)]
    if verbose:
        print(f"brute force m={m} n_sub={n_sub} plant={plant}: {len(hits)} 7-sets with image "
              f"span <= 4, {len(classes)} class sets, {len(missing)} uncovered; "
              f"{stats['steps']} steps")
    if plant:
        assert frozenset(star.tolist()) in hits, "planted 7-set not found by brute force"
    assert not missing, f"{len(missing)} 7-sets not covered by any class set"
    return len(hits), len(classes)


def planted_check(m=2, seed=3, verbose=True):
    """Full-dictionary planted test at m=2 (360 states, trivial symmetry):
    seven random states and a random 3-dimensional subspace of their span
    as the target; the scan must produce a class set containing the plant
    and the mod-ell decision must accept that class set."""
    rng = np.random.default_rng(seed)
    E = C.build_dictionary(m)
    N = E.shape[0]
    El = C.to_mod(E, C.Z6, ELL)
    while True:
        star = np.sort(rng.choice(N, size=7, replace=False))
        if C.rank_mod(El[star], ELL) == 7:
            break
    coeffs = rng.integers(1, ELL, size=(3, 7))
    T_ell = (coeffs @ El[star]) % ELL
    Tl = rref_mod(T_ell, ELL)
    PD = C.quotient_projection(El, Tl, 6, int(rng.integers(1 << 30)))
    isfree = np.all(PD == 0, axis=1).astype(np.int8)
    # count-only pass over every pivot for the timing (the six-dimensional
    # quotient at m=2 floods with class sets, so recording them all is slow)
    _, stats = scan_all(PD, isfree, 4, record=False)
    # recording pass for the plant's two least members only
    i0, j0 = int(star[0]), int(star[1])
    jok = np.zeros(N, dtype=np.int8)
    jok[j0] = 1
    classes, _ = scan_pivot(PD, i0, jok, isfree, 4)
    star_set = set(star.tolist())
    containing = [c for c in classes if star_set <= set([c[0], c[1], c[2]] + c[5])]
    assert containing, "planted configuration not contained in any class set"
    # decision mod ell (a control, not the Hadamard-exact decision)
    ok = False
    for i, j, k, nz, ng, mem in containing:
        cls = sorted(set([i, j, k] + mem))
        rS = C.rank_mod(El[cls], ELL)
        rST = C.rank_mod(np.vstack([El[cls], Tl]), ELL)
        if rS == rST:
            ok = True
    assert ok, "no class set containing the plant has the target in its span"
    if verbose:
        print(f"planted m={m}: plant {star.tolist()} inside {len(containing)} of {len(classes)} "
              f"class sets of pair ({i0},{j0}); count-only scan over all pivots: "
              f"{stats['candidates']} candidates, {stats['steps']:.3e} steps in "
              f"{stats['seconds']:.1f}s ({1e9 * stats['seconds'] / stats['steps']:.1f} ns/step "
              f"including per-pair overheads at N=360)")
    return stats


def _orbits_and_jok(perms, N, isfree):
    """Orbit representatives and per-representative second-pivot masks
    (minimal in its orbit under the stabilizer of the representative), as in
    cert_t3m3_rank7.symmetry but for an arbitrary permutation list."""
    ident = np.arange(N)
    gmin = ident.copy()
    stab = {}
    for P in perms:
        gmin = np.minimum(gmin, P)
        for i in np.flatnonzero(P == ident):
            stab.setdefault(int(i), []).append(P)
    reps = np.array([r for r in np.unique(gmin) if not isfree[r]])
    jok_of = {}
    for i in reps:
        imin = ident.copy()
        for P in stab.get(int(i), []):
            imin = np.minimum(imin, P)
        jok_of[int(i)] = ((imin == ident) & (isfree == 0)).astype(np.int8)
    return reps, jok_of, gmin


def m2_symmetry_check(n_states=40, seed=5, verbose=True):
    """At m=2 with the true V_2, on a G-invariant sub-dictionary (a union of
    orbits of the symmetry group of V_2 that includes the three states in
    V_2): the set of projected image spaces of the class sets found with the
    symmetry (one first pivot per orbit, second pivot minimal under its
    stabilizer), closed under the group, must equal the set found with
    trivial symmetry. The full dictionary at m=2 floods with tens of
    millions of class sets, the quotient being only six-dimensional, which
    is why the check runs on a sub-dictionary."""
    m = 2
    E, E2, T2, PD, isfree, Tl = setup(m)
    N = E.shape[0]
    T = C.t3_targets(m)
    assert int(isfree.sum()) == 3
    elems = C.monomial_symmetries(T, m)
    index = {k: i for i, k in enumerate(C.row_keys(E))}
    perms = [np.fromiter((index[k] for k in C.row_keys(C.apply_element(E, e))), dtype=np.int64,
                         count=N) for e in elems]
    _, _, gmin = _orbits_and_jok(perms, N, isfree)
    rng = np.random.default_rng(seed)
    chosen = set(np.flatnonzero(isfree).tolist())
    for o in rng.permutation(np.unique(gmin)):
        if len(chosen) >= n_states:
            break
        chosen |= set(np.flatnonzero(gmin == o).tolist())
    sub = np.array(sorted(chosen))
    pos = {int(s): t for t, s in enumerate(sub)}
    perms_sub = [np.array([pos[int(P[s])] for s in sub]) for P in perms]
    PDs, frees = PD[sub], isfree[sub]
    reps, jok_of, _ = _orbits_and_jok(perms_sub, len(sub), frees)
    t0 = time.time()
    c_sym, s_sym = scan_all(PDs, frees, 4, reps, jok_of)
    t1 = time.time()
    c_all, s_all = scan_all(PDs, frees, 4)
    t2 = time.time()
    keys_all = {image_space_key(PDs, [i, j, k] + mem) for i, j, k, _, _, mem in c_all}
    keys_sym = set()
    for i, j, k, _, _, mem in c_sym:
        members = np.array(sorted(set([i, j, k] + mem)))
        for P in perms_sub:
            keys_sym.add(image_space_key(PDs, P[members]))
    if verbose:
        print(f"m=2 symmetry on {len(sub)} states: group order {len(perms)}, {len(reps)} pivot "
              f"orbits; with symmetry {len(c_sym)} classes / {s_sym['steps']} steps / "
              f"{t1 - t0:.1f}s, without {len(c_all)} classes / {s_all['steps']} steps / "
              f"{t2 - t1:.1f}s; {len(keys_sym)} vs {len(keys_all)} image spaces")
    assert keys_sym == keys_all, "orbit closure of the symmetric scan differs from the full scan"
    found, spurious, hist = decide(E2[sub], T2, c_sym)
    best = min(r for _, r in found) if found else None
    if verbose:
        print(f"m=2 decision on the sub-dictionary: {len(found)} of {len(c_sym)} class sets "
              f"contain V_2, minimal rank {best}, {spurious} spurious")
    return s_sym, s_all


def m2_kernel2_consistency(verbose=True):
    """Every class set of the two-pivot kernel (six-state configurations,
    need 4) must be contained in a class set of the three-pivot kernel (need
    4) for the same first pivot and trivial symmetry, at m=2 with V_2."""
    m = 2
    E, E2, T2, PD, isfree, Tl = setup(m)
    N = E.shape[0]
    checked = 0
    for i in [300, 330, 350]:
        if isfree[i]:
            continue
        jok = np.zeros(N, dtype=np.int8)
        jok[i + 1:] = 1
        jok[np.flatnonzero(isfree)] = 0
        c2, mb, par = C.kernel2(PD, INV, i, jok, isfree, int(isfree.sum()), 4, ELL, 400000)
        by = {}
        for cid, k in mb:
            by.setdefault(int(cid), []).append(int(k))
        sets2 = [frozenset([i, int(c2[c, 0])] + by.get(c, [])) for c in range(c2.shape[0])]
        c3, _ = scan_pivot(PD, i, jok, isfree, 4)
        sets3 = [frozenset([a, b, k] + mem) for a, b, k, _, _, mem in c3]
        for s in sets2:
            assert any(s <= t for t in sets3), f"two-pivot class {sorted(s)} not inside a three-pivot class"
            checked += 1
    if verbose:
        print(f"m=2 kernel2/kernel3 consistency: {checked} two-pivot class sets all covered")
    return checked


def time_m3_pairs(js=(29000, 27000, 24000), i=None, verbose=True):
    """Per-step timing of kernel3 on the real m=3 data for a few second
    pivots j of one first pivot (cost of a pair is about (N - j)^2 / 2)."""
    E, E2, T2, PD, isfree, Tl = setup(3)
    N = E.shape[0]
    assert not isfree.any()
    i = 0 if i is None else int(i)
    out = []
    for j in js:
        jok = np.zeros(N, dtype=np.int8)
        jok[j] = 1
        t0 = time.time()
        classes, s = scan_pivot(PD, i, jok, isfree, 4)
        dt = time.time() - t0
        found, spurious, hist = decide(E2, T2, classes)
        rec = {"i": i, "j": int(j), "steps": s["steps"], "seconds": dt,
               "ns_per_step": 1e9 * dt / max(1, s["steps"]), "classes": len(classes),
               "found": len(found), "spurious": spurious,
               "hist": {f"{a},{b}": v for (a, b), v in sorted(hist.items())}}
        out.append(rec)
        if verbose:
            print(f"m=3 pair ({i},{j}): {s['steps']:.3e} steps, {dt:.2f}s, "
                  f"{rec['ns_per_step']:.1f} ns/step, {len(classes)} classes, "
                  f"{len(found)} contain V_3, {spurious} spurious, hist {rec['hist']}", flush=True)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "small"
    if what in ("small", "all"):
        brute_force_check(n_sub=22, seed=1, plant=True)
        brute_force_check(n_sub=22, seed=2, plant=True)
        brute_force_check(n_sub=26, seed=3, plant=False)
        planted_check(seed=3)
        m2_kernel2_consistency()
    if what in ("sym", "all"):
        m2_symmetry_check()
    if what in ("m3", "all"):
        time_m3_pairs()
    if what == "m3big":
        import json
        out = time_m3_pairs(js=(20000, 15000, 8000), i=0)
        out += time_m3_pairs(js=(22000, 12000), i=12345)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                            "m3_pair_timings.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
