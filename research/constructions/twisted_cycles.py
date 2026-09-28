"""Decompositions invariant under a twisted m-cycle of the copies (2026-09-28).

The symmetry group of |M>^m is G = U^m semidirect S_m with U the single-qudit
Clifford stabilizer of |M>; for the cat family only the diagonal u^(x m) fixes
the target, so there the local part is that diagonal subgroup rather than a
full m-fold product. `cyclic_symmetric.py` (2026-09-26) searched the term sets
invariant under an untwisted m-cycle P_sigma. This script covers the rest of
the m-cycles of G, namely g = (x_j u_j) P_sigma.

Classification up to gauge. Conjugating g by V = (x_j v_j) in U^m sends the
twist u to u'_j = v_j u_j v_{sigma^{-1}(j)}^{-1}, so the ordered product of the
u_j around the cycle changes only by conjugation in U, and every twist with a
given product is gauge-equivalent to (P, 1, ..., 1). The conjugacy class of P
in U is a complete invariant, and a gauge V fixes the target, so a g-invariant
decomposition is the V-image of a g_P-invariant one. Replacing g by another
generator of <g> replaces P by P^k with gcd(k, m ord(P)) = 1, which is the
coarser equivalence that matters for a term set. For the cat family the gauge
group is diagonal and commutes with P_sigma, so its elements are classes
outright.

Why a term is an eigenvector. g|M>^m = mu |M>^m for a phase mu, and in a
minimal decomposition the coefficients are unique, so g permutes the terms with
c_{g s} lambda_s = mu c_s. A g-orbit of size d inside the term set is carried by
a stabilizer state s with g^d s proportional to s, and contributes the single
vector sum_{j<d} mu^{-j} g^j s, the projection of s onto the mu-eigenspace of g
up to scale. Orbit sizes divide ord(g) and are at most the rank, so a cell fixes
a small list of admissible d.

Two ways to list the candidates, both exact:

  `--pool power`, the eigenspace handle. Let D be the least common multiple of
  the admissible orbit sizes; every term is then an eigenvector of h = g^D.
  When gcd(D, m) = 1 the permutation part of h is again an m-cycle and its cycle
  product is trivial, so h is gauge-equivalent to the plain shift P_{sigma^D}:
  there is a V in U^m with V h V^{-1} proportional to P_{sigma^D}. An eigenvector
  of a shift through an m-cycle has its eigenvalue both in the phase group of the
  amplitudes (order 4 at p = 2, order p at odd p) and in the m-th roots of unity,
  so when gcd(m, 4) = 1, or gcd(m, p) = 1 at odd p, the eigenvalue is 1 and the
  candidates are exactly V^{-1} T_rho F, with F the sigma-fixed set
  `cyclic_symmetric.py` enumerates and rho the position permutation j -> D j,
  which conjugates sigma to sigma^D. At qubit_H and cat m = 7 that replaces a
  scan of 8.1e10 seven-qubit stabilizer states by a filter on 66, and it is
  complete for every shape at rank at most 6.

  `--pool brute` streams the whole m-qudit stabilizer dictionary and records the
  exact g-orbit size of every state. It fits at p = 3, m <= 4 and p = 2, m <= 5,
  and is the control on the first method. Where the pool for an orbit size runs
  past `--cap` the count is still exact and the states are not stored, so the
  shape is reported as not covered rather than covered wrongly.

Usage:
    twisted_cycles.py --classes
    twisted_cycles.py --orbit cat --m 7 --rank 3 --pool power
    twisted_cycles.py --orbit H3 --m 4 --rank 6 --pool brute
    twisted_cycles.py --control
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "merges"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stabrank_verify import ORBIT_P, target_vector  # noqa: E402
from to_witness import term_from_vector, NotStabilizer  # noqa: E402
from mergelib import flats  # noqa: E402
import cyclic_symmetric as cyc  # noqa: E402
from cyclic_symmetric import min_rank  # noqa: E402
from witness_symmetry import local_factors, _proportional  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Record-capable open cells (the 2026-09-26 table, re-derived from
# docs/ledger.json) with the rank that would set a record.
RECORD_CELLS = [
    ("S", 5, 5), ("S", 6, 7), ("S", 7, 11), ("S", 8, 15),
    ("N", 5, 10), ("N", 6, 15),
    ("H3", 4, 6), ("H3", 5, 10), ("H3", 6, 15),
    ("T3", 4, 8), ("T3", 5, 15), ("T3", 6, 26),
    ("qubit_H", 7, 6), ("qubit_H", 8, 8), ("qubit_H", 9, 11), ("qubit_H", 10, 15),
    ("qubit_T", 8, 8), ("qubit_T", 10, 15),
    ("T5", 3, 11), ("T5", 4, 24),
    ("cat", 7, 3), ("cat", 8, 5),
]

# Cells the 2026-09-26 untwisted run decided (cyclic_symmetric.py).
COVERED = {("cat", 7), ("qubit_H", 7), ("qubit_T", 7), ("S", 5), ("N", 5),
           ("H3", 5), ("T3", 5), ("T5", 3)}


# ------------------------------------------------------- the local group ---

def group_table(facs, p):
    """Multiplication indices, identity, inverses and orders, mod phase."""
    n = len(facs)

    def find(M):
        for k, F in enumerate(facs):
            if _proportional(F, M):
                return k
        raise AssertionError("the local set is not closed under products")

    tab = [[find(facs[i] @ facs[j]) for j in range(n)] for i in range(n)]
    ident = [k for k, F in enumerate(facs)
             if _proportional(F, np.eye(p, dtype=complex))]
    assert len(ident) == 1
    e = ident[0]
    order = []
    for k in range(n):
        o, cur = 1, k
        while cur != e:
            cur = tab[cur][k]
            o += 1
        order.append(o)
    inv = [next(j for j in range(n) if tab[k][j] == e) for k in range(n)]
    return tab, e, inv, order


def power_index(tab, e, k, t):
    cur = e
    for _ in range(t):
        cur = tab[cur][k]
    return cur


def conjugacy_classes(tab, inv, n):
    seen, out = set(), []
    for k in range(n):
        if k in seen:
            continue
        cls = sorted({tab[tab[g][k]][inv[g]] for g in range(n)})
        out.append(cls)
        seen.update(cls)
    return out


def twist_classes(orbit, m):
    """One representative per gauge class of twisted m-cycle. Each entry has the
    index of the cycle product P in the local group, its order, the
    representative twist, and a `gen` label shared by the classes that generate
    the same cyclic group."""
    p = ORBIT_P[orbit]
    facs, diagonal_only = local_factors(orbit, m)
    tab, e, inv, order = group_table(facs, p)
    I = np.eye(p, dtype=complex)
    out = []
    if diagonal_only:
        for k in range(len(facs)):
            out.append({"P": k, "order": order[k], "twist": [facs[k]] * m,
                        "diagonal": True})
    else:
        for cls in conjugacy_classes(tab, inv, len(facs)):
            out.append({"P": cls[0], "order": order[cls[0]],
                        "twist": [facs[cls[0]]] + [I] * (m - 1), "diagonal": False})
    fams = []
    for c in out:
        N = m * c["order"]
        fams.append({power_index(tab, e, c["P"], t % c["order"])
                     for t in range(1, N + 1) if math.gcd(t, N) == 1})
    label, groups = {}, []
    for c, fam in zip(out, fams):
        hit = next((i for i, gs in enumerate(groups) if gs & fam), None)
        if hit is None:
            groups.append(set(fam))
            hit = len(groups) - 1
        else:
            groups[hit] |= fam
        label[c["P"]] = hit
    for c in out:
        c["gen"] = label[c["P"]]
    return out, facs, order


# ------------------------------------------------------ the cycle operator --

def position_permutation(perm, p, m):
    """dst[x] for T_pi with T_pi|d_0 ... d_{m-1}> = |e>, e_{pi(j)} = d_j."""
    dim = p ** m
    w = p ** (m - 1 - np.arange(m))
    dst = np.zeros(dim, dtype=np.int64)
    for x in range(dim):
        d = [(x // int(w[j])) % p for j in range(m)]
        e = [0] * m
        for j in range(m):
            e[perm[j]] = d[j]
        dst[x] = int(np.dot(e, w))
    return dst


def permutation_matrix(dst, dim):
    S = np.zeros((dim, dim), dtype=complex)
    S[dst, np.arange(dim)] = 1.0
    return S


def shift_matrix(p, m, step=1):
    """P_{sigma^step} with sigma the shift of `cyclic_symmetric.py`: the digit at
    position j moves to position j+1."""
    return permutation_matrix(
        position_permutation([(j + step) % m for j in range(m)], p, m), p ** m)


def cycle_operator(twist, p, m):
    """g = (x_j u_j) P_sigma as a dense p^m x p^m matrix."""
    L = twist[0]
    for j in range(1, m):
        L = np.kron(L, twist[j])
    return L @ shift_matrix(p, m)


def eigen_phase(g, psi, tol=1e-8):
    w = g @ psi
    i = int(np.argmax(np.abs(psi)))
    mu = w[i] / psi[i]
    assert np.linalg.norm(w - mu * psi) < tol, "the twisted cycle does not fix the target"
    return mu


def operator_order(g, cap):
    M, o = g.copy(), 1
    I = np.eye(g.shape[0], dtype=complex)
    while not _proportional(M, I):
        M = M @ g
        o += 1
        if o > cap:
            raise AssertionError("the cycle operator has no finite order mod phase")
    return o


def admissible_orbit_sizes(o, rank):
    return [d for d in range(1, rank + 1) if o % d == 0]


# -------------------------------------------------------- the dictionary ---

def stream_dictionary(p, n, chunk=1 << 13):
    """Every n-qudit stabilizer state, in blocks of columns.

    A state is x0 + the row space of W in reduced row echelon form carrying the
    phase w_p^{Q(y) + l.y} at odd p, with Q upper triangular including the
    diagonal, and i^{l.y}(-1)^{Q(y)} at p = 2 with Q strictly upper triangular
    and l in Z_4, which is the parametrization of `verify_challenge/to_witness.py`
    with its redundancy (a diagonal Q_ii against l_i + 2) removed. The total is
    asserted against p^n prod_{j<=n}(p^j + 1)."""
    gmod = 4 if p == 2 else p
    root = np.exp(2j * np.pi / gmod)
    total = 0
    for k, idx_all in flats(p, n):
        ys = (np.array(list(itertools.product(range(p), repeat=k)), dtype=np.int64)
              if k else np.zeros((1, 0), dtype=np.int64))
        if p == 2:
            pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
            rows = [2 * ys[:, i] * ys[:, j] for (i, j) in pairs]
            ranges = [2] * len(pairs) + [4] * k
        else:
            pairs = [(i, j) for i in range(k) for j in range(i, k)]
            rows = [ys[:, i] * ys[:, j] for (i, j) in pairs]
            ranges = [p] * len(pairs) + [p] * k
        rows += [ys[:, i] for i in range(k)]
        basis = (np.array(rows, dtype=np.int64) if rows
                 else np.zeros((0, p ** k), dtype=np.int64))
        nphase = 1
        for r in ranges:
            nphase *= r
        for start in range(0, nphase, chunk):
            stop = min(nphase, start + chunk)
            ids = np.arange(start, stop)
            digits = np.zeros((stop - start, len(ranges)), dtype=np.int64)
            rest = ids.copy()
            for t in range(len(ranges) - 1, -1, -1):
                digits[:, t] = rest % ranges[t]
                rest //= ranges[t]
            expo = ((digits @ basis) % gmod if len(ranges)
                    else np.zeros((1, p ** k), dtype=np.int64))
            amp = root ** expo / np.sqrt(p ** k)                 # (B, p^k)
            for f in range(idx_all.shape[0]):
                V = np.zeros((p ** n, amp.shape[0]), dtype=complex)
                V[idx_all[f][:, None], np.arange(amp.shape[0])[None, :]] = amp.T
                total += amp.shape[0]
                yield V
    expect = p ** n
    for j in range(1, n + 1):
        expect *= p ** j + 1
    assert total == expect, f"streamed {total} states, expected {expect}"


def brute_orbit_scan(p, n, g, sizes, cap=200_000, verbose=False):
    """Exact g-orbit size of every n-qudit stabilizer state, restricted to the
    admissible sizes. Returns {d: list of states} and {d: exact count}; a size
    whose count passes `cap` keeps its count and drops its list."""
    mats = {d: np.linalg.matrix_power(g, d) for d in sizes}
    kept = {d: [] for d in sizes}
    counts = {d: 0 for d in sizes}
    over = set()
    t0, seen = time.time(), 0
    for V in stream_dictionary(p, n):
        B = V.shape[1]
        seen += B
        ar = np.arange(B)
        piv = np.argmax(np.abs(V), axis=0)
        base = V[piv, ar]
        exact = np.zeros(B, dtype=np.int64)
        for d in sizes:
            W = mats[d] @ V
            ratio = W[piv, ar] / base
            ok = np.linalg.norm(W - ratio[None, :] * V, axis=0) < 1e-8
            fresh = ok & (exact == 0)
            exact[fresh] = d
        for d in sizes:
            sel = np.flatnonzero(exact == d)
            counts[d] += len(sel)
            if d in over or counts[d] > cap:
                over.add(d)
                kept[d] = None
                continue
            for c in sel:
                kept[d].append(V[:, c].copy())
    if verbose:
        print(f"    streamed {seen} states in {time.time() - t0:.1f}s", flush=True)
    return kept, counts, over


def brute_span(p, n, g, mu, sizes, psi, verbose=False):
    """The span of every g-orbit sum over the whole dictionary, accumulated as
    an orthonormal basis rather than stored state by state, together with the
    residual of the target on it. A nonzero residual closes every g-invariant
    decomposition of every rank whose orbits have a size in `sizes`."""
    mats = {d: np.linalg.matrix_power(g, d) for d in sizes}
    counts = {d: 0 for d in sizes}
    B = np.zeros((p ** n, 0), dtype=complex)
    buf = []
    t0, seen = time.time(), 0

    def absorb():
        """Re-orthonormalize through an SVD, which keeps the basis exactly
        orthonormal and its column count at the true rank; a Gram-Schmidt pass
        drifts and can report more columns than the dimension."""
        nonlocal B, buf
        if not buf:
            return
        M = np.column_stack([B] + buf) if B.shape[1] else np.column_stack(buf)
        buf = []
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        r = int((s > 1e-8 * max(float(s[0]), 1.0)).sum())
        B = U[:, :r]

    for V in stream_dictionary(p, n):
        Bn = V.shape[1]
        seen += Bn
        ar = np.arange(Bn)
        piv = np.argmax(np.abs(V), axis=0)
        base = V[piv, ar]
        exact = np.zeros(Bn, dtype=np.int64)
        for d in sizes:
            W = mats[d] @ V
            ratio = W[piv, ar] / base
            ok = np.linalg.norm(W - ratio[None, :] * V, axis=0) < 1e-8
            exact[ok & (exact == 0)] = d
        for d in sizes:
            sel = np.flatnonzero(exact == d)
            if not len(sel):
                continue
            counts[d] += len(sel)
            if B.shape[1] >= p ** n:
                continue
            S = np.zeros((p ** n, len(sel)), dtype=complex)
            cur = V[:, sel]
            for j in range(d):
                S = S + (mu ** -j) * cur
                cur = g @ cur
            nrm = np.linalg.norm(S, axis=0)
            good = nrm > 1e-8
            if not good.any():
                continue
            S = S[:, good] / nrm[good]
            # only the orbit sums that are mu-eigenvectors can appear
            S = S[:, np.linalg.norm(g @ S - mu * S, axis=0) < 1e-7]
            if S.shape[1]:
                res = S - B @ (B.conj().T @ S)
                S = S[:, np.linalg.norm(res, axis=0) > 1e-8]
            if S.shape[1]:
                buf.extend(S.T)
                if len(buf) >= 200:
                    absorb()
    absorb()
    r = float(np.linalg.norm(psi - B @ (B.conj().T @ psi))) if B.shape[1] else 1.0
    if verbose:
        print(f"    streamed {seen} states in {time.time() - t0:.1f}s", flush=True)
    return B, counts, r


def power_pool(g, p, m, D, facs, verbose=False, cap=1 << 24):
    """Every stabilizer state s with g^D s proportional to s, as V^{-1} T_rho F
    with F the sigma-fixed set of `cyclic_symmetric.py`."""
    assert math.gcd(D, m) == 1, "sigma^D is not an m-cycle"
    gmod = 4 if p == 2 else p
    assert math.gcd(m, gmod) == 1, (
        "a shift eigenvalue need not be 1 here, so the reduction does not apply")
    h = np.linalg.matrix_power(g, D)
    SD = shift_matrix(p, m, D)
    V, combo = None, None
    if _proportional(h, SD):
        V, combo = np.eye(p ** m, dtype=complex), "identity"
    else:
        if len(facs) ** m > 1 << 16:
            raise AssertionError("the local gauge search is over its cap")
        for cand in itertools.product(range(len(facs)), repeat=m):
            W = facs[cand[0]]
            for j in range(1, m):
                W = np.kron(W, facs[cand[j]])
            if _proportional(W @ h @ np.conj(W).T, SD):
                V, combo = W, cand
                break
    assert V is not None, "no local gauge takes g^D to the plain shift"
    F, skipped = cyc.fixed_stabilizer_states(m, p, cap=cap)
    assert not skipped, f"{len(skipped)} flats skipped in the sigma-fixed enumeration"
    Trho = permutation_matrix(position_permutation([(D * j) % m for j in range(m)],
                                                   p, m), p ** m)
    Vi = np.conj(V).T
    pool = []
    for v, _term in F:
        u = Vi @ (Trho @ v)
        u = u / np.linalg.norm(u)
        assert _proportional(h @ u, u), "a pooled state is not fixed by g^D"
        pool.append(u)
    if verbose:
        print(f"    sigma-fixed set {len(F)}, gauge {combo}, pool {len(pool)}",
              flush=True)
    return pool


# --------------------------------------------------------------- searching --

def orbit_sums(g, mu, states, rank, diag=None):
    """(vectors, sizes, representatives): one entry per distinct g-orbit sum."""
    vecs, sizes, reps = [], [], []
    seen = {}
    if diag is None:
        diag = {}
    diag.setdefault("by_size", {})
    diag.setdefault("dropped_eigenvalue", 0)
    diag.setdefault("dropped_oversize", 0)
    for v in states:
        cur, d = v.copy(), 1
        while d <= rank:
            cur = g @ cur
            if _proportional(cur, v):
                break
            d += 1
        if d > rank:
            diag["dropped_oversize"] += 1
            continue
        diag["by_size"][d] = diag["by_size"].get(d, 0) + 1
        s = np.zeros_like(v)
        cur = v.copy()
        for j in range(d):
            s = s + (mu ** -j) * cur
            cur = g @ cur
        if np.linalg.norm(s) < 1e-8:
            continue
        s = s / np.linalg.norm(s)
        # the coefficients on an orbit are forced, so the orbit sum has to be a
        # mu-eigenvector; when g^d s = lambda s with lambda != mu^d it is not,
        # and that orbit cannot sit inside a g-invariant decomposition.
        if np.linalg.norm(g @ s - mu * s) > 1e-7:
            diag["dropped_eigenvalue"] += 1
            continue
        # Distinct orbits can share an orbit sum, since a fixed stabilizer state
        # can be proportional to the sum over a longer orbit; the cheaper of the
        # two is the one the search may use.
        key = _key(s)
        if key in seen:
            i = seen[key]
            if d < sizes[i]:
                sizes[i], reps[i] = d, v
            continue
        seen[key] = len(vecs)
        vecs.append(s)
        sizes.append(d)
        reps.append(v)
    return vecs, sizes, reps


def _orthobasis(A, tol=1e-9):
    """Orthonormal basis of the column space, through an SVD so that a
    rank-deficient input does not get padded with arbitrary directions the way
    a reduced QR pads it."""
    if A.shape[1] == 0:
        return A
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, : int((s > tol * max(float(s[0]), 1.0)).sum())]


def _heavy_profiles(groups, rank):
    """Every multiset {size: count} of orbits of size above one whose total is
    at most `rank`, with the number of choices it carries. Counting by size
    class rather than over all heavy orbits at once is what keeps the shapes
    that contain one long orbit inside reach."""
    out = []

    def rec(sizes_left, budget, prof):
        if not sizes_left:
            n = 1
            for d, c in prof.items():
                n *= math.comb(len(groups[d]), c)
            out.append((dict(prof), n))          # the empty profile is all-fixed
            return
        d, rest = sizes_left[0], sizes_left[1:]
        for c in range(0, budget // d + 1):
            if c:
                prof[d] = c
            rec(rest, budget - c * d, prof)
            prof.pop(d, None)

    rec(sorted(groups), rank, {})
    return sorted(out, key=lambda t: sum(d * c for d, c in t[0].items()))


def _heavy_combos(groups, profile):
    parts = [itertools.combinations(groups[d], c) for d, c in sorted(profile.items())]
    for pick in itertools.product(*parts):
        yield tuple(i for part in pick for i in part)


def search(vecs, sizes, psi, rank, shape_cap=2_000_000, skipped=None):
    """Exhaustive: orbits of total size at most `rank` whose span holds psi.
    The orbits of size 1 go through `cyclic_symmetric.min_rank`; the heavier
    ones are forced a few at a time and projected out. A shape with more than
    `shape_cap` choices of the heavy orbits is recorded in `skipped` and left
    undecided rather than run."""
    ones = [i for i, d in enumerate(sizes) if d == 1]
    groups = {}
    for i, d in enumerate(sizes):
        if d > 1:
            groups.setdefault(d, []).append(i)
    if skipped is None:
        skipped = []
    ONES = (np.column_stack([vecs[i] for i in ones]) if ones
            else np.zeros((len(psi), 0), dtype=complex))
    QF = _orthobasis(ONES)
    # Every shape that uses fixed terms needs the target inside the span of the
    # chosen long orbits together with all of the fixed states, so the component
    # of the target off the fixed span must come from the long orbits alone.
    off = psi - QF @ (QF.conj().T @ psi) if QF.shape[1] else psi
    off_n = float(np.linalg.norm(off))
    for profile, count in _heavy_profiles(groups, rank):
        if count > shape_cap:
            skipped.append((profile, count))
            continue
        for combo in _heavy_combos(groups, profile):
            cost = sum(sizes[i] for i in combo)
            if cost > rank:
                continue
            if combo:
                B = _orthobasis(np.column_stack([vecs[i] for i in combo]))
                t = psi - B @ (B.conj().T @ psi)
            else:
                B, t = None, psi
            if np.linalg.norm(t) < 1e-9:
                if combo:
                    return list(combo), cost
                continue
            left = rank - cost
            if left == 0 or not ones:
                continue
            if off_n > 1e-9:
                if B is None:
                    continue
                Cp = B - QF @ (QF.conj().T @ B) if QF.shape[1] else B
                Cp = _orthobasis(Cp)
                if (Cp.shape[1] == 0
                        or np.linalg.norm(off - Cp @ (Cp.conj().T @ off)) > 1e-9):
                    continue
            cand = ONES - B @ (B.conj().T @ ONES) if B is not None else ONES
            nrm = np.linalg.norm(cand, axis=0)
            keep = np.flatnonzero(nrm > 1e-9)
            if not keep.size:
                continue
            C = cand[:, keep]
            cc, *_ = np.linalg.lstsq(C, t, rcond=None)
            if np.linalg.norm(C @ cc - t) > 1e-9:
                continue          # not even in the span of every size-1 orbit
            R, hit = min_rank([C[:, j] / nrm[keep[j]] for j in range(len(keep))],
                              t / np.linalg.norm(t), left)
            if R is not None:
                return list(combo) + [ones[int(keep[j])] for j in hit[0]], cost + R
    return None, None


# ------------------------------------------------------------------ driver --

def target_numeric(orbit, m):
    v = np.array([complex(x) for x in target_vector(orbit, m)]).ravel()
    return v / np.linalg.norm(v)


def span_report(vecs, psi, label):
    """Whether the target lies in the span of a sub-pool, which decides that
    shape at every rank at once."""
    if not vecs:
        print(f"    {label}: none")
        return None
    U = np.column_stack(vecs)
    coef, *_ = np.linalg.lstsq(U, psi, rcond=None)
    res = float(np.linalg.norm(U @ coef - psi))
    print(f"    {label}: {len(vecs)} of them, span dimension "
          f"{int(np.linalg.matrix_rank(U, tol=1e-8))}, residual of the target "
          f"{res:.3e}", flush=True)
    return res


def run_cell(orbit, m, rank, pool_mode="auto", only_class=None, cap=200_000,
             sizes_wanted=None, search_rank=None, verbose=True):
    p = ORBIT_P[orbit]
    psi = target_numeric(orbit, m)
    classes, facs, order = twist_classes(orbit, m)
    print(f"== {orbit} m={m} (p={p}), record rank {rank}: {len(classes)} gauge "
          f"classes of twisted {m}-cycle, {len({c['gen'] for c in classes})} up to "
          f"the choice of generator", flush=True)
    mode = pool_mode
    if mode == "auto":
        mode = "brute" if (p == 2 and m <= 5) or (p == 3 and m <= 4) else "power"
    out = []
    for c in classes:
        if only_class is not None and c["P"] != only_class:
            continue
        g = cycle_operator(c["twist"], p, m)
        mu = eigen_phase(g, psi)
        o = operator_order(g, 4 * m * max(order))
        ds = admissible_orbit_sizes(o, rank)
        if sizes_wanted is not None:
            ds = [d for d in ds if d in sizes_wanted]
        D = math.lcm(*ds) if len(ds) > 1 else ds[0]
        print(f"  class P={c['P']} (product order {c['order']}), generator class "
              f"{c['gen']}, ord(g)={o}, mu={np.round(mu, 6)}, admissible orbit "
              f"sizes {ds}, D={D}", flush=True)
        t0 = time.time()
        if mode == "power":
            pool = power_pool(g, p, m, D, facs, verbose=verbose)
            covered, counts = ds, None
            print(f"    power pool: {len(pool)} candidates, {time.time() - t0:.1f}s",
                  flush=True)
        else:
            kept, counts, over = brute_orbit_scan(p, m, g, ds, cap=cap,
                                                  verbose=verbose)
            covered = [d for d in ds if d not in over]
            pool = [v for d in covered for v in kept[d]]
            print(f"    dictionary scan: counts by orbit size "
                  f"{ {d: counts[d] for d in ds} }, covered sizes {covered}, "
                  f"{time.time() - t0:.1f}s", flush=True)
        od = {}
        vecs, sizes, reps = orbit_sums(g, mu, pool, rank, diag=od)
        print(f"    candidates by g-orbit size {od['by_size']}, "
              f"{od['dropped_oversize']} with an orbit longer than the rank, "
              f"{od['dropped_eigenvalue']} whose orbit sum is not a mu-eigenvector",
              flush=True)
        print(f"    {len(vecs)} distinct orbit sums, sizes "
              f"{ {d: sizes.count(d) for d in sorted(set(sizes))} }", flush=True)
        if not vecs:
            print("    VERDICT: no g-invariant term set of the covered shapes.")
            out.append((c["P"], 0, None, covered))
            continue
        for d in sorted(set(sizes)):
            span_report([v for v, e in zip(vecs, sizes) if e <= d], psi,
                        f"orbit sums of size at most {d}")
        U = np.column_stack(vecs)
        coef, *_ = np.linalg.lstsq(U, psi, rcond=None)
        res = float(np.linalg.norm(U @ coef - psi))
        if res > 1e-9:
            print(f"    VERDICT: the target is not in that span, so no g-invariant "
                  f"decomposition of any rank has all orbits of size in {covered}.")
            out.append((c["P"], len(vecs), None, covered))
            continue
        sr = rank if search_rank is None else search_rank
        if sr < rank:
            n1 = max(sizes.count(1), rank - 2)
            print(f"    the subset search is run to rank {sr} only: at rank {rank} "
                  f"the all-fixed shape alone leaves C({sizes.count(1)}, {rank - 2}) "
                  f"= {math.comb(n1, rank - 2)}", flush=True)
        t1 = time.time()
        sk = []
        idx, cost = search(vecs, sizes, psi, sr, skipped=sk)
        if sk:
            print(f"    shapes left undecided, (number of orbits of size above 1, "
                  f"choices): {sk}", flush=True)
        if idx is None:
            print(f"    VERDICT: no g-invariant decomposition of rank <= {sr} with "
                  f"all orbit sizes in {covered}, over the shapes that were run "
                  f"({time.time() - t1:.1f}s)")
            out.append((c["P"], len(vecs), None, covered))
        else:
            print(f"    HIT: rank {cost}, orbit sums {idx}, sizes "
                  f"{[sizes[i] for i in idx]}")
            _write_hit(orbit, m, c["P"], cost, idx, sizes, reps, p)
            out.append((c["P"], len(vecs), cost, covered))
    return out


def _write_hit(orbit, m, P, cost, idx, sizes, reps, p):
    os.makedirs(RESULTS, exist_ok=True)
    terms = []
    for i in idx:
        try:
            terms.append(term_from_vector(reps[i], p, m))
        except NotStabilizer:
            terms.append(None)
    path = os.path.join(RESULTS, f"twisted_{orbit}_m{m}_P{P}_rank{cost}.json")
    with open(path, "w") as f:
        json.dump({"orbit": orbit, "m": m, "rank": cost, "twist_class": P,
                   "orbit_sizes": [sizes[i] for i in idx],
                   "orbit_representatives": terms}, f, indent=1)
    print(f"    written {path}")


def classes_table():
    print(f"{'cell':<14}{'|U|':>5}{'classes':>9}{'generator classes':>20}"
          f"   untwisted class already run")
    for orbit, m, _rank in RECORD_CELLS:
        classes, facs, _order = twist_classes(orbit, m)
        print(f"{orbit + ' m=' + str(m):<14}{len(facs):>5}{len(classes):>9}"
              f"{len({c['gen'] for c in classes}):>20}"
              f"   {'yes' if (orbit, m) in COVERED else 'no'}")


def control():
    ok = True
    # 1. the dictionary stream reproduces the state counts
    for p, n in [(2, 3), (3, 2), (2, 4), (3, 3)]:
        total = sum(V.shape[1] for V in stream_dictionary(p, n))
        want = p ** n
        for j in range(1, n + 1):
            want *= p ** j + 1
        good = total == want
        ok &= good
        print(f"  dictionary stream p={p} n={n}: {total} states (want {want}) "
              f"{'OK' if good else 'MISMATCH'}", flush=True)
    # 2. the untwisted class through the power pool reproduces 2026-09-26
    for (orbit, m, want) in [("qubit_H", 7, 66), ("S", 5, 120)]:
        p = ORBIT_P[orbit]
        facs, _ = local_factors(orbit, m)
        pool = power_pool(shift_matrix(p, m), p, m, 1, facs)
        good = len(pool) == want
        ok &= good
        print(f"  untwisted p={p} m={m}: {len(pool)} sigma-fixed states "
              f"(cyclic_symmetric.py says {want}) {'OK' if good else 'MISMATCH'}",
              flush=True)
    # 3. the power pool agrees with the dictionary pool where both fit
    for orbit, m, rank in [("qubit_H", 5, 4), ("qubit_T", 5, 4), ("cat", 5, 4)]:
        p = ORBIT_P[orbit]
        classes, facs, _order = twist_classes(orbit, m)
        for c in classes:
            g = cycle_operator(c["twist"], p, m)
            o = operator_order(g, 200)
            ds = admissible_orbit_sizes(o, rank)
            D = math.lcm(*ds) if len(ds) > 1 else ds[0]
            gmod = 4 if p == 2 else p
            if math.gcd(D, m) != 1 or math.gcd(m, gmod) != 1:
                continue
            a = len(power_pool(g, p, m, D, facs))
            _kept, counts, _over = brute_orbit_scan(p, m, g, ds)
            b = sum(counts.values())
            good = a == b
            ok &= good
            print(f"  {orbit} m={m} P={c['P']} D={D}: power pool {a}, dictionary "
                  f"pool {b} {'OK' if good else 'MISMATCH'}", flush=True)
    # 4. the recovery control. The board's rank-3 witness for |F>^4 is fixed by
    #    a twisted 4-cycle and by no pure one, so the machinery must return it.
    pure, tw = mcycle_symmetry("qubit_T", 4, witness_vectors("qubit_T", 4, 3))
    good = bool(tw) and not pure
    ok &= good
    print(f"  qubit_T m=4, the board's rank-3 witness: {len(pure)} pure 4-cycles, "
          f"{len(tw)} twisted ones {'OK' if good else 'MISMATCH'}", flush=True)
    if tw:
        g = tw[0][2]
        want = {_key(v) for v in witness_vectors("qubit_T", 4, 3)}
        mu = eigen_phase(g, target_numeric("qubit_T", 4))
        ds = admissible_orbit_sizes(operator_order(g, 200), 3)
        kept, _counts, _over = brute_orbit_scan(2, 4, g, ds)
        pool = [v for d in ds for v in kept[d]]
        vecs, sizes, reps = orbit_sums(g, mu, pool, 3)
        got = any({_key(w) for w in _orbit_of(g, reps[i])} == want
                  for i in range(len(reps)))
        idx, cost = search(vecs, sizes, target_numeric("qubit_T", 4), 3)
        ok &= got and cost == 3
        print(f"  the witness term set is one of the {len(vecs)} orbits the "
              f"enumeration returns: {got}; the search finds rank {cost} "
              f"{'OK' if (got and cost == 3) else 'MISMATCH'}", flush=True)
    # 4b. the all-fixed shape has to be reached too: at cat m=6 the three terms
    #     of Qassim, Pashayan, and Gosset are each invariant under the 6-cycle,
    #     so the shape with no long orbit must return rank 3 through `search`.
    F, _sk = cyc.fixed_stabilizer_states(6, 2, cap=1 << 24)
    idx, cost = search([v for v, _t in F], [1] * len(F), target_numeric("cat", 6), 3)
    good = cost == 3
    ok &= good
    print(f"  cat m=6, the all-fixed shape through the shape enumeration: rank "
          f"{cost} {'OK' if good else 'MISMATCH'}", flush=True)
    # 5. the untwisted side at cat m=6: the three terms of Qassim, Pashayan, and
    #    Gosset are individually permutation invariant.
    F, skipped = cyc.fixed_stabilizer_states(6, 2, cap=1 << 24)
    R, _hit = min_rank([v for v, _t in F], target_numeric("cat", 6), 3)
    good = (not skipped) and len(F) == 105 and R == 3
    ok &= good
    print(f"  cat m=6 untwisted: {len(F)} sigma-fixed states (want 105), min rank "
          f"{R} (want 3) {'OK' if good else 'MISMATCH'}", flush=True)
    print("CONTROL", "PASSED" if ok else "FAILED")
    return ok


def _key(v):
    """Projective digest: the phase is fixed by the first entry of close to
    maximal modulus, so that ties among equal amplitudes resolve the same way
    for every representation of the state."""
    a = np.abs(v)
    i = int(np.argmax(a >= a.max() - 1e-9))
    return (np.round(v / v[i], 6) + 0.0).tobytes()


def _orbit_of(g, v):
    out, cur = [v], g @ v
    while not _proportional(cur, v):
        out.append(cur)
        cur = g @ cur
    return out


def witness_vectors(orbit, m, rank):
    """The term vectors of `bounds/<orbit>-m<m>-upper-<rank>.json`."""
    from common import term_vector
    path = os.path.join(ROOT, "bounds", f"{orbit}-m{m}-upper-{rank}.json")
    with open(path) as f:
        rec = json.load(f)
    terms = rec.get("witness", {}).get("terms") or rec.get("terms")
    p = ORBIT_P[orbit]
    return [term_vector(t, p, m) for t in terms]


def mcycle_symmetry(orbit, m, vectors):
    """Every m-cycle of the copies, twisted or not, that fixes the term set, as
    (position permutation, local indices, operator), split into the pure ones
    and the twisted ones."""
    p = ORBIT_P[orbit]
    facs, diagonal = local_factors(orbit, m)
    ident = [k for k, F in enumerate(facs)
             if _proportional(F, np.eye(p, dtype=complex))][0]
    want = {_key(v) for v in vectors}
    combos = ([(k,) * m for k in range(len(facs))] if diagonal
              else list(itertools.product(range(len(facs)), repeat=m)))
    assert len(combos) <= 1 << 16, "the local group is over the cap"
    pure, tw = [], []
    for pi in itertools.permutations(range(m)):
        seen, x = set(), 0
        while x not in seen:
            seen.add(x)
            x = pi[x]
        if len(seen) != m:
            continue
        T = permutation_matrix(position_permutation(list(pi), p, m), p ** m)
        for cb in combos:
            L = facs[cb[0]]
            for j in range(1, m):
                L = np.kron(L, facs[cb[j]])
            g = L @ T
            if {_key(g @ v) for v in vectors} == want:
                (pure if all(j == ident for j in cb) else tw).append((pi, cb, g))
    return pure, tw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbit")
    ap.add_argument("--m", type=int)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--pool", default="auto", choices=["auto", "brute", "power"])
    ap.add_argument("--class", dest="cls", type=int)
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--sizes", help="comma separated orbit sizes to keep")
    ap.add_argument("--search-rank", type=int)
    ap.add_argument("--classes", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--span-scan", action="store_true",
                    help="the span of every orbit sum, over the whole dictionary")
    ap.add_argument("--witness-cycles", action="store_true",
                    help="the m-cycles, twisted or not, fixing a board witness")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.classes:
        classes_table()
        return
    if a.control:
        control()
        return
    if a.span_scan:
        p = ORBIT_P[a.orbit]
        psi = target_numeric(a.orbit, a.m)
        classes, facs, order = twist_classes(a.orbit, a.m)
        for c in classes:
            if a.cls is not None and c["P"] != a.cls:
                continue
            g = cycle_operator(c["twist"], p, a.m)
            mu = eigen_phase(g, psi)
            o = operator_order(g, 4 * a.m * max(order))
            ds = admissible_orbit_sizes(o, a.rank)
            B, counts, res = brute_span(p, a.m, g, mu, ds, psi, verbose=True)
            print(f"  class P={c['P']} (product order {c['order']}), ord(g)={o}, "
                  f"orbit sizes {ds}, counts {counts}: the orbit sums span "
                  f"{B.shape[1]} dimensions, residual of the target {res:.3e}",
                  flush=True)
            if res > 1e-9:
                print(f"    VERDICT: no decomposition of |{a.orbit}>^{a.m} of any "
                      f"rank at most {a.rank} is invariant under this class.",
                      flush=True)
        return
    if a.witness_cycles:
        pure, tw = mcycle_symmetry(a.orbit, a.m, witness_vectors(a.orbit, a.m, a.rank))
        print(f"{a.orbit} m={a.m} rank-{a.rank} witness: {len(pure)} pure "
              f"{a.m}-cycles, {len(tw)} twisted {a.m}-cycles")
        for pi, cb, _g in tw[:8]:
            print(f"  permutation {pi}, local indices {cb}")
        return
    if not (a.orbit and a.m and a.rank):
        ap.error("give --orbit, --m and --rank, or --classes or --control")
    run_cell(a.orbit, a.m, a.rank, pool_mode=a.pool, only_class=a.cls, cap=a.cap,
             sizes_wanted=({int(x) for x in a.sizes.split(",")} if a.sizes else None),
             search_rank=a.search_rank, verbose=not a.quiet)


if __name__ == "__main__":
    main()
