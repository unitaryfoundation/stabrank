"""First-coordinate-slice filters for the degenerate bases of the rank-6
exclusion of |N>^4 (docs/notes/n4_rank6_design.md, section 3): sound
necessary conditions for a base multiset to admit any decomposition,
decided with the compiled dense_solve on small projections, so that the
reference matcher (research/qutrit_m4_rank5/matcher.py) runs only on the
bases that pass.

Setting. A base at x_0 is a 6-multiset of two-qutrit stabilizer states
whose distinct states carry an affine coefficient family d = d_0 + K lambda
(over F_P1, F_P2 and C; matcher.family_from). At the coordinate slice
x_0 + e each ordinary term i takes one of 28 options w_i (27 phased Pauli
translates of its base slice or absent), and the slice equation is
sum_i d_i(lambda) w_i + (block contributions) = rhs.

B6 (six distinct dependent states, kappa = 1). Pick a term j with K_j != 0
(the fresh term). In the basis of the nine Pauli translates Q_k s_j of its
base slice, write U = B^-1 (sum_{i != j} d_{0i} w_i - rhs) and
V = B^-1 sum_{i != j} K_i w_i. The fresh term's slice is
(d_{0j} + lambda K_j) w^l Q_k s_j or zero, so U + lambda V is supported on
at most one coordinate k. Partition the nine coordinates into parts
P_1, ..., P_m; the exceptional k lies in one part, so for the group
G = complement of that part, U_G + lambda V_G = 0, that is, U_G lies in
the span of V_G. Each group is a 1-parameter dense solve over the five
ordinary terms (sides 28^3 x 28^2, 1.7e7 feature pairs against 4.8e8 for
the six-term solve), decided by dense_solve mod P1 and P2 on the
projected vectors; the union over the m groups is a superset of the
exact first-slice solutions. The fresh term and the partition are chosen
to minimize the structural slack (options of an ordinary term whose
support in the translate basis lies inside one part, which that part's
group cannot see). Every surviving combination is completed with the
fresh term's code (class k and phase from the residual coordinate, or
absent) and decided exactly by Family.restrict, the reference's own
decision, so the solution list equals the reference's first-slice list.

C6 with one block (g copies of b) and ordinary terms whose family has
kappa <= 1. The block contributes a vector of the span of at most g
translates of b, so in the translate basis of b the residual
R = B^-1 (rhs - sum_i d_i(lambda) w_i) has at most g nonzero coordinates;
with g + 1 disjoint parts one part is zero. The reference's own
solve_slice3 (a meet in the middle per translate set) is already cheap
for kappa = 0, so the filter is for kappa = 1 only.

Both filters assume nothing beyond the base point (every option, absent
included, is allowed at the slice), so they are necessary conditions on
every decomposition with that base.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
from matcher import (E1, P1, P2, Block, _mm_int64, _split_sides, add, family_from, restrict,  # noqa: E402
                     slice_system)
from slice_cover import _native_symbol  # noqa: E402

ALL = tuple(range(9))


def inv_mod(M, p):
    """Inverse of a square integer matrix modulo p (Python ints)."""
    n = len(M)
    A = [[int(M[r][c]) % p for c in range(n)] + [1 if r == c else 0 for c in range(n)] for r in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c])
        A[c], A[piv] = A[piv], A[c]
        inv = pow(A[c][c], p - 2, p)
        A[c] = [(x * inv) % p for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [(x - f * y) % p for x, y in zip(A[r], A[c])]
    return np.array([row[n:] for row in A], dtype=np.int64)


def partitions(m):
    """Set partitions of the nine coordinates into m parts of sizes as equal
    as possible (m = 2: 4 + 5; m = 3: 3 + 3 + 3; m = 9: singletons)."""
    if m == 9:
        return [tuple((k,) for k in ALL)]
    if m == 2:
        return [(A, tuple(c for c in ALL if c not in A)) for A in itertools.combinations(ALL, 4)]
    if m == 3:
        out = []
        for a in itertools.combinations(ALL[1:], 2):
            A = (0,) + a
            r2 = [c for c in ALL if c not in A]
            for b in itertools.combinations(r2[1:], 2):
                B = (r2[0],) + b
                out.append((A, B, tuple(c for c in r2 if c not in B)))
        return out
    raise ValueError(m)


PARTS = {m: partitions(m) for m in (2, 3, 9)}
PART_MASKS = {m: np.array([[sum(1 << c for c in P) for P in parts] for parts in PARTS[m]], dtype=np.int64)
              for m in PARTS}


class TranslateBasis:
    """B^-1 over F_P1, F_P2 and C for the nine class translates of a base
    state (columns of B), cached per dictionary index by the caller, and
    the projected option tables of other states in this basis."""

    def __init__(self, o):
        cols = [3 * k for k in range(o.nclass)]
        self.B1 = np.ascontiguousarray(o.m1[cols].T)          # (9, 9): column k = translate k
        self.B2 = np.ascontiguousarray(o.m2[cols].T)
        self.inv1 = inv_mod(self.B1, P1)
        self.inv2 = inv_mod(self.B2, P2)
        self.inv1T = np.ascontiguousarray(self.inv1.T)
        self.inv2T = np.ascontiguousarray(self.inv2.T)
        self._proj = {}

    def coords(self, o, idx):
        """All 28 options of the term o in the nine translate coordinates,
        mod P1 and P2: (28, 9) each; cached by dictionary index (the cache
        is cleared when it holds 64 terms)."""
        if idx not in self._proj:
            if len(self._proj) >= 64:
                self._proj.clear()
            p1 = (o.m1 @ self.inv1T) % P1
            p2 = _mm_int64(o.m2, self.inv2T, P2)
            self._proj[idx] = (np.ascontiguousarray(p1, dtype=np.int64), np.ascontiguousarray(p2, dtype=np.int64))
        return self._proj[idx]

    def masks(self, o):
        """Support bitmasks of the 27 translate options of the term o in
        this basis (mod P1 only, uncached)."""
        p1 = (o.m1[:27] @ self.inv1T) % P1
        return ((p1 != 0) << np.arange(9)[None, :]).sum(axis=1)

    def vec(self, r1, r2):
        return (self.inv1 @ r1) % P1, _mm_int64(self.inv2, r2[:, None], P2)[:, 0]


class Filters:
    def __init__(self, matcher, seed=41, m_choices=(3,)):
        self.M = matcher
        self.dense = _native_symbol("dense_solve")
        if self.dense is None:
            raise RuntimeError("dense_solve is not available in stabrank_core")
        self.rng = np.random.default_rng(seed)
        self._tb = {}
        self.m_choices = m_choices
        self.cube1 = {int(v): l for l, v in enumerate(matcher.F1.wpow)}

    def basis(self, idx):
        if idx not in self._tb:
            self._tb[idx] = TranslateBasis(self.M.options(idx))
        return self._tb[idx]

    def family(self, distinct, target, x0):
        b1, b2, bC = target.rhs(x0)
        return family_from(self.M.U1[distinct], self.M.U2[distinct], self.M.C[:, distinct], b1, b2, bC)

    # ------------------------------------------------------------- B6 ----

    def plan_b6(self, distinct, fam):
        """The fresh term j and the partition with the least structural
        slack; returns (j, parts, slack)."""
        KC = fam.parts[2][1][:, 0]
        K1, K2 = fam.parts[0][1][:, 0], fam.parts[1][1][:, 0]
        best = None
        for j in range(6):
            if abs(KC[j]) < 1e-9 or K1[j] == 0 or K2[j] == 0:
                continue
            tb = self.basis(distinct[j])
            masks = np.concatenate([tb.masks(self.M.options(distinct[i])) for i in range(6) if i != j])
            for m in self.m_choices:
                pm = PART_MASKS[m]                                                # (n_partitions, m)
                inside = (masks[None, None, :] & ~pm[:, :, None]) == 0            # option inside a part
                slack = inside.any(axis=1).sum(axis=1)                            # (n_partitions,)
                t = int(np.argmin(slack))
                key = (int(slack[t]), m)
                if best is None or key < best[0]:
                    best = (key, j, PARTS[m][t])
                if key[0] == 0:
                    break
            if best is not None and best[0][0] == 0:
                break
        if best is None:
            return None
        return best[1], best[2], best[0][0]

    def b6_slice(self, cover, x0, target, e=E1, fam=None):
        """Exact solutions of the coordinate slice x0 + e for a base of six
        distinct states with kappa = 1, as (combo, (), family) triples in
        the reference's format over the distinct states in sorted order,
        with the filter's statistics."""
        M = self.M
        distinct = sorted(set(int(u) for u in cover))
        assert len(distinct) == 6 == len(cover)
        if fam is None:
            fam = self.family(distinct, target, x0)
        if fam is None:
            return None, {"refused": True}
        if fam.kappa != 1 or fam.kappa1 != 1:
            return None, {"refused": False, "outside": f"kappa {fam.kappa}, kappa1 {fam.kappa1}"}
        t0 = time.time()
        plan = self.plan_b6(distinct, fam)
        if plan is None:
            return None, {"refused": False, "outside": "no fresh term with a nonzero dependency mod both primes"}
        j, parts, slack = plan
        d01, K1 = fam.parts[0][0], fam.parts[0][1][:, 0]
        d02, K2 = fam.parts[1][0], fam.parts[1][1][:, 0]
        ords = [i for i in range(6) if i != j]
        tb = self.basis(distinct[j])
        co = [tb.coords(M.options(distinct[i]), distinct[i]) for i in ords]
        r1, r2, rC = target.rhs(add(x0, e))
        R1, R2 = tb.vec(r1, r2)
        t_plan = time.time() - t0
        cands = set()
        raw = 0
        sides = _split_sides([28] * 5)
        for P in parts:
            G = [c for c in ALL if c not in P]
            o1 = [np.ascontiguousarray(c[0][:, G]) for c in co]
            o2 = [np.ascontiguousarray(c[1][:, G]) for c in co]
            combos, nraw, _ = self.dense(
                o1, o2, np.ascontiguousarray(d01[ords] % P1), np.ascontiguousarray(K1[ords] % P1).reshape(5, 1),
                np.ascontiguousarray(d02[ords] % P2), np.ascontiguousarray(K2[ords] % P2).reshape(5, 1),
                np.ascontiguousarray(R1[G] % P1), np.ascontiguousarray(R2[G] % P2),
                [int(i) for i in sides[0]], [int(i) for i in sides[1]], 2_000_000, int(self.rng.integers(1, 1 << 62)))
            raw += int(nraw)
            for row in np.asarray(combos):
                cands.add(tuple(int(c) for c in row))
        t_dense = time.time() - t0 - t_plan
        # whole-vector test mod P1 on the survivors: U + lambda V supported on
        # one coordinate k (or none), which fixes lambda and the fresh code
        out = []
        n_p1 = 0
        if cands:
            C5 = np.array(sorted(cands), dtype=np.int64)                              # (n, 5)
            U = np.zeros((len(C5), 9), dtype=np.int64)
            V = np.zeros((len(C5), 9), dtype=np.int64)
            for t, i in enumerate(ords):
                U = (U + int(d01[i]) * co[t][0][C5[:, t]]) % P1
                V = (V + int(K1[i]) * co[t][0][C5[:, t]]) % P1
            U = (U - R1[None, :]) % P1
            arrays_all = [M.options(u).arrays() for u in distinct]
            oj = M.options(distinct[j])
            for n in range(len(C5)):
                u, v = U[n], V[n]
                fresh_codes = set()
                for k in list(ALL) + [None]:
                    rest = [c for c in ALL if c != k]
                    ur, vr = u[rest], v[rest]
                    nzv = np.flatnonzero(vr)
                    if len(nzv):
                        lam = (-int(ur[nzv[0]]) * pow(int(vr[nzv[0]]), P1 - 2, P1)) % P1
                        if np.any((ur + lam * vr) % P1):
                            continue
                        lams = [lam]
                    else:
                        if np.any(ur):
                            continue
                        lams = None                                 # lambda free on this slice
                    if k is None:
                        fresh_codes.add(oj.absent)
                        continue
                    if lams is None:
                        fresh_codes.update(3 * k + l for l in range(3))
                        continue
                    lam = lams[0]
                    mu = (-(int(u[k]) + lam * int(v[k]))) % P1
                    dj = (int(d01[j]) + lam * int(K1[j])) % P1
                    if mu == 0 or dj == 0:
                        continue                                    # a vanishing fresh coefficient: no full decomposition
                    ph = (mu * pow(dj, P1 - 2, P1)) % P1
                    if ph in self.cube1:
                        fresh_codes.add(3 * k + self.cube1[ph])
                n_p1 += bool(fresh_codes)
                for c6 in sorted(fresh_codes):
                    combo = [0] * 6
                    for i, c in zip(ords, C5[n]):
                        combo[i] = int(c)
                    combo[j] = int(c6)
                    f2 = restrict(fam, *slice_system(arrays_all, [], tuple(combo), (), (r1, r2, rC)))
                    if f2 is not None:
                        out.append((tuple(combo), (), f2))
        st = {"refused": False, "fresh": distinct[j], "parts": len(parts), "slack": slack, "dense_raw": raw,
              "survivors5": len(cands), "survivors_p1": n_p1, "solutions": len(out), "seconds_plan": t_plan,
              "seconds_dense": t_dense, "seconds": time.time() - t0}
        return out, st

    # ------------------------------------------------------------- C6 ----

    def c6_slice(self, cover, x0, target, e=E1, fam=None):
        """Filter for a base with exactly one repeated state (g copies) and
        ordinary terms whose family has kappa = 1: the exact solutions of
        the slice x0 + e as (combo over ordinary terms, translate set of the
        block, family) in the reference's format, decided by restrict on the
        projected equation, plus statistics. Returns (None, st) when the
        shape is outside the filter."""
        M = self.M
        cover = [int(u) for u in cover]
        distinct = sorted(set(cover))
        mult = {u: cover.count(u) for u in distinct}
        blocks = [u for u in distinct if mult[u] > 1]
        if len(blocks) != 1:
            return None, {"refused": False, "outside": "not one block"}
        b = blocks[0]
        g = mult[b]
        if fam is None:
            fam = self.family(distinct, target, x0)
        if fam is None:
            return None, {"refused": True}
        if fam.kappa1 != 1 or fam.kappa != 1:
            return None, {"refused": False, "outside": f"kappa {fam.kappa}, kappa1 {fam.kappa1}"}
        bpos = distinct.index(b)
        ords = [i for i in range(len(distinct)) if i != bpos]
        r = len(ords)
        tb = self.basis(b)
        co = [tb.coords(M.options(distinct[i]), distinct[i]) for i in ords]
        r1, r2, rC = target.rhs(add(x0, e))
        R1, R2 = tb.vec(r1, r2)
        d01, K1 = fam.parts[0][0], fam.parts[0][1]
        d02, K2 = fam.parts[1][0], fam.parts[1][1]
        groups = [list(ALL[t::g + 1]) for t in range(g + 1)]
        t0 = time.time()
        cands = set()
        raw = 0
        sides = _split_sides([28] * r)
        for G in groups:
            o1 = [np.ascontiguousarray(c[0][:, G]) for c in co]
            o2 = [np.ascontiguousarray(c[1][:, G]) for c in co]
            combos, nraw, _ = self.dense(
                o1, o2, np.ascontiguousarray(d01[ords] % P1), np.ascontiguousarray(K1[ords] % P1).reshape(r, -1),
                np.ascontiguousarray(d02[ords] % P2), np.ascontiguousarray(K2[ords] % P2).reshape(r, -1),
                np.ascontiguousarray(R1[G] % P1), np.ascontiguousarray(R2[G] % P2),
                [int(i) for i in sides[0]], [int(i) for i in sides[1]], 2_000_000, int(self.rng.integers(1, 1 << 62)))
            raw += int(nraw)
            for row in np.asarray(combos):
                cands.add(tuple(int(c) for c in row))
        t_dense = time.time() - t0
        # whole-vector test mod P1: R + lambda V' has at most g nonzero
        # coordinates for some lambda; then the reference's projected
        # decision over the translate sets inside that support
        out = []
        n_p1 = 0
        if cands:
            C = np.array(sorted(cands), dtype=np.int64)
            U = np.zeros((len(C), 9), dtype=np.int64)
            V = np.zeros((len(C), 9), dtype=np.int64)
            for t, i in enumerate(ords):
                U = (U + int(d01[i]) * co[t][0][C[:, t]]) % P1
                V = (V + int(K1[i, 0]) * co[t][0][C[:, t]]) % P1
            U = (U - R1[None, :]) % P1
            blk = Block(M.options(b), g, bpos)
            arrays = [M.options(distinct[i]).arrays() for i in ords]
            for n in range(len(C)):
                u, v = U[n], V[n]
                supports = set()
                for S in itertools.combinations(ALL, g):
                    rest = [c for c in ALL if c not in S]
                    ur, vr = u[rest], v[rest]
                    nzv = np.flatnonzero(vr)
                    if len(nzv):
                        lam = (-int(ur[nzv[0]]) * pow(int(vr[nzv[0]]), P1 - 2, P1)) % P1
                        if np.any((ur + lam * vr) % P1):
                            continue
                    elif np.any(ur):
                        continue
                    supports.add(S)
                n_p1 += bool(supports)
                if not supports:
                    continue
                # every translate set contained in a feasible support
                tried = set()
                for S in supports:
                    for j in range(g + 1):
                        for T in itertools.combinations(S, j):
                            if T in tried:
                                continue
                            tried.add(T)
                            f2 = restrict(fam, *slice_system(arrays, [blk], tuple(int(c) for c in C[n]), (T,),
                                                             (r1, r2, rC)))
                            if f2 is not None:
                                out.append((tuple(int(c) for c in C[n]), (T,), f2))
        st = {"refused": False, "block": b, "g": g, "kappa": int(fam.kappa), "raw": raw, "survivors": len(cands),
              "survivors_p1": n_p1, "solutions": len(out), "seconds_dense": t_dense, "seconds": time.time() - t0}
        return out, st
