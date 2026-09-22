"""Stage A of the H^6 matcher ported to p = 3 and a two-qutrit base (the 2 + 2
bipartition of |M>^4), in Python, for costing and controls.

Setting. psi^4 = sum_i c_i s_i sliced along qutrits 1, 2: u_i^(x) = (<x| (x) I) s_i,
x in F_3^2, sum_i c_i u_i^(x) = alpha_x psi^2. Base point x0 with all r terms
visible: the base slice is a full r-cover (d_i, u_i) of psi^2 with distinct,
independent states (stage A; dependent or repeated bases are refused here).
At x0 + e_1 and x0 + e_2 each term is w^l Q u_i (9 classes x 3 phases) or
absent; the flat of the term through x0 is read off the absence pattern
(plane, a coordinate line, or one of {point, diagonal line (1,1), diagonal
line (1,2)}), and the six composite points carry the quadratic phase pattern
of the two-qutrit structure lemma (27 rows for a plane term, 3 for a
coordinate line, 163 for the absent-at-both type).

Usage:
  q3match.py sample ORBIT CENSUS_JSON [--count K] [--x0 a,b]
  q3match.py planted ORBIT [--count K]
  q3match.py product ORBIT
  q3match.py lifts13 ORBIT CENSUS3_JSON [--count K]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
from cover_census import P1, P2, Field3, patterns, rank_mod  # noqa: E402
from rank_exclusion import dictionary, psi_for  # noqa: E402
from common import alpha as alpha_C  # noqa: E402

W3 = np.exp(2j * np.pi / 3)
PTS = [(a, b) for a in range(3) for b in range(3)]
E1, E2 = (1, 0), (0, 1)
COMP = [(1, 1), (2, 0), (0, 2), (1, 2), (2, 1), (2, 2)]     # composite offsets, solved in this order
NUM_TOL = 1e-8


def digits(n):
    x = np.arange(3 ** n)
    return (x[:, None] // 3 ** (n - 1 - np.arange(n))[None, :]) % 3


def pauli_apply(u, a, c, n):
    """(X^a Z^c u)[y + a] = w^{c.y} u[y] on n qutrits."""
    digs = digits(n)
    wts = 3 ** (n - 1 - np.arange(n))
    y = ((digs + np.array(a)) % 3) @ wts
    out = np.zeros(3 ** n, dtype=complex)
    out[y] = W3 ** ((digs @ np.array(c)) % 3) * u
    return out


def phase_codes(v):
    """Codes 0 (zero), 1..3 (1, w, w^2) of a vector with entries in {0} u
    {w^k}; raises otherwise."""
    codes = np.zeros(len(v), dtype=np.int64)
    for k in range(3):
        codes[np.abs(v - W3 ** k) < 1e-6] = k + 1
    if np.count_nonzero(codes) != np.count_nonzero(np.abs(v) > 1e-9):
        raise AssertionError("vector entries are not cube roots of unity or zero")
    return codes


class TermOpts:
    """Slice options of one term with base slice u on n qutrits (u with first
    nonzero entry 1): code 3 k + l is w^l T[k], T[k] = Q_k u for the class
    representative Q_k; code 3^(n+1) is absent."""

    def __init__(self, u, n, F1, F2):
        self.n = n
        dim = 3 ** n
        self.u = u
        seen, self.reps, self.T = {}, [], []
        for a in itertools.product(range(3), repeat=n):
            for c in itertools.product(range(3), repeat=n):
                v = pauli_apply(u, a, c, n)
                j = np.flatnonzero(np.abs(v) > 1e-9)[0]
                key = (np.round(v / v[j], 6) + 0.0).tobytes()
                if key not in seen:
                    seen[key] = len(self.T)
                    self.reps.append((a, c))
                    self.T.append(v)
        if len(self.T) != dim or self.reps[0] != ((0,) * n, (0,) * n):
            raise AssertionError("expected 3^n Pauli classes with the identity first")
        self.nclass = dim
        self.absent = 3 * dim
        vecs = [W3 ** l * t for t in self.T for l in range(3)]
        self.vecs = np.array(vecs + [np.zeros(dim, dtype=complex)])
        m1, m2 = [], []
        for v in vecs:
            cd = phase_codes(v)
            m1.append(F1.codes_to_field(cd))
            m2.append(F2.codes_to_field(cd))
        self.m1 = np.array(m1 + [np.zeros(dim, dtype=np.int64)])
        self.m2 = np.array(m2 + [np.zeros(dim, dtype=np.int64)])
        # composite table: class and phase of Q_{k2}^{x2} Q_{k1}^{x1} u
        self.cls = np.zeros((dim, dim, 9), dtype=np.int64)
        self.g = np.zeros((dim, dim, 9), dtype=np.int64)
        for k1 in range(dim):
            for k2 in range(dim):
                for x in PTS:
                    v = u.copy()
                    for _ in range(x[0]):
                        v = pauli_apply(v, *self.reps[k1], n)
                    for _ in range(x[1]):
                        v = pauli_apply(v, *self.reps[k2], n)
                    k, ph = self.code_of(v)
                    self.cls[k1, k2, pidx(x)] = k
                    self.g[k1, k2, pidx(x)] = ph

    def code_of(self, v):
        for k, t in enumerate(self.T):
            j = np.flatnonzero(np.abs(t) > 1e-9)[0]
            ratio = v[j] / t[j]
            for l in range(3):
                if abs(ratio - W3 ** l) < 1e-6 and np.allclose(v, W3 ** l * t, atol=1e-9):
                    return k, l
        raise AssertionError("vector is not a phased Pauli translate of the base slice")

    def composite_rows(self, c1, c2):
        """Codes at the six COMP offsets for every shape consistent with the
        coordinate codes c1 (at e_1) and c2 (at e_2)."""
        A = self.absent
        rows = []
        if c1 != A and c2 != A:
            k1, l1 = divmod(c1, 3)
            k2, l2 = divmod(c2, 3)
            for a, b, c in itertools.product(range(3), repeat=3):
                row = []
                for x in COMP:
                    q = (a * x[0] * x[0] + b * x[1] * x[1] + c * x[0] * x[1] + (l1 - a) * x[0] + (l2 - b) * x[1]) % 3
                    row.append(3 * self.cls[k1, k2, pidx(x)] + (q + self.g[k1, k2, pidx(x)]) % 3)
                rows.append(row)
        elif c1 != A:
            k1 = c1 // 3
            for l in range(3):
                row = [A] * 6
                row[COMP.index((2, 0))] = 3 * self.cls[k1, 0, pidx((2, 0))] + (self.g[k1, 0, pidx((2, 0))] + l) % 3
                rows.append(row)
        elif c2 != A:
            k2 = c2 // 3
            for l in range(3):
                row = [A] * 6
                row[COMP.index((0, 2))] = 3 * self.cls[0, k2, pidx((0, 2))] + (self.g[0, k2, pidx((0, 2))] + l) % 3
                rows.append(row)
        else:
            rows.append([A] * 6)                          # point term
            for first, second in (((1, 1), (2, 2)), ((1, 2), (2, 1))):
                for k in range(self.nclass):
                    for l in range(3):
                        for l2 in range(3):
                            row = [A] * 6
                            row[COMP.index(first)] = 3 * k + l
                            row[COMP.index(second)] = 3 * self.cls[k, 0, pidx((2, 0))] + (self.g[k, 0, pidx((2, 0))] + l2) % 3
                            rows.append(row)
        return np.unique(np.array(rows, dtype=np.int64), axis=0)


def pidx(x):
    return 3 * (x[0] % 3) + (x[1] % 3)


def add(x, y):
    return ((x[0] + y[0]) % 3, (x[1] + y[1]) % 3)


def solve_mod(A, b, p):
    """Unique solution of A x = b over F_p (A of full column rank), else None."""
    A = np.asarray(A, dtype=np.int64) % p
    b = np.asarray(b, dtype=np.int64) % p
    M = np.column_stack((A, b))
    rows, cols = A.shape
    rank = 0
    piv = []
    for c in range(cols):
        pr = None
        for r in range(rank, rows):
            if M[r, c]:
                pr = r
                break
        if pr is None:
            return None
        M[[rank, pr]] = M[[pr, rank]]
        M[rank] = (M[rank] * pow(int(M[rank, c]), p - 2, p)) % p
        for r in range(rows):
            if r != rank and M[r, c]:
                M[r] = (M[r] - M[r, c] * M[rank]) % p
        piv.append(c)
        rank += 1
    if np.any(M[rank:, cols]):
        return None
    return M[:cols, cols].copy()


class Matcher2:
    """Stage A for a base of r distinct independent states on n2 qutrits,
    sliced along two qutrits, against a target with slices rhs(x)."""

    def __init__(self, D, n2, F1, F2, seed=29, verbose=False):
        self.D, self.n2 = D, n2
        self.codes, self.C = patterns(D)
        self.F1, self.F2 = F1, F2
        self.U1 = F1.codes_to_field(self.codes)
        self.U2 = F2.codes_to_field(self.codes)
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.verbose = verbose

    def options(self, idx):
        if idx not in self.cache:
            self.cache[idx] = TermOpts(self.C[:, idx], self.n2, self.F1, self.F2)
        return self.cache[idx]

    def run(self, cover, x0, rhs1, rhs2, rhsC, target_C, target_2):
        """rhs*(x): the slice of the target at x (over F_p1, F_p2, C);
        target_C: the full target vector (3^(2 + n2)); target_2: its F_p2
        vector. Returns (hits, stats)."""
        r = len(cover)
        stats = {"coord_solutions": [], "joined": 0, "composite_candidates": 0, "raw_hits": 0, "hits": 0,
                 "refused": False}
        if len(set(cover)) != r:
            stats["refused"] = True
            return [], stats
        b1, b2, bC = rhs1(x0), rhs2(x0), rhsC(x0)
        d1 = solve_mod(self.U1[list(cover)].T, b1, P1)
        d2 = solve_mod(self.U2[list(cover)].T, b2, P2)
        A = self.C[:, list(cover)]
        if d1 is None or d2 is None or np.linalg.matrix_rank(A, tol=1e-8) < r:
            stats["refused"] = True
            return [], stats
        dC, *_ = np.linalg.lstsq(A, bC, rcond=None)
        if np.linalg.norm(A @ dC - bC) > NUM_TOL or np.any(np.abs(dC) < 1e-9) or np.any(d1 == 0):
            stats["refused"] = True
            return [], stats
        opts = [self.options(int(c)) for c in cover]
        # coordinate slices
        sols = []
        for e in (E1, E2):
            x = add(x0, e)
            s = self.solve_slice(opts, d1, d2, dC, rhs1(x), rhs2(x), rhsC(x))
            stats["coord_solutions"].append(len(s))
            sols.append(s)
            if not s:
                return [], stats
        hits = []
        for c1 in sols[0]:
            for c2 in sols[1]:
                stats["joined"] += 1
                rows = [opts[i].composite_rows(int(c1[i]), int(c2[i])) for i in range(r)]
                for full in self.composite(opts, d1, rows, x0, rhs1, stats):
                    stats["raw_hits"] += 1
                    codes = {E1: c1, E2: c2}
                    for t, x in enumerate(COMP):
                        codes[x] = full[t]
                    h = self.confirm(cover, opts, codes, x0, target_C, target_2)
                    if h is not None:
                        hits.append(h)
        stats["hits"] = len(hits)
        return hits, stats

    def solve_slice(self, opts, d1, d2, dC, r1, r2, rC):
        r = len(opts)
        h = r // 2
        f = self.rng.integers(1, P1, size=len(r1))

        def sums(ids, mod, d):
            S = (d[ids[0]] * mod[ids[0]]) % P1
            for i in ids[1:]:
                S = (S[:, None, :] + (d[i] * mod[i])[None, :, :]).reshape(-1, S.shape[1]) % P1
            return S

        left, right = list(range(h)), list(range(h, r))
        SL = sums(left, [o.m1 for o in opts], d1)
        SR = sums(right, [o.m1 for o in opts], d1)
        kL = (SL @ f) % P1
        kR = ((r1[None, :] - SR) @ f) % P1
        order = np.argsort(kL, kind="stable")
        sk = kL[order]
        lo = np.searchsorted(sk, kR, side="left")
        hi = np.searchsorted(sk, kR, side="right")
        rows = np.flatnonzero(hi > lo)
        out = []
        sizes = [o.absent + 1 for o in opts]
        for rr in rows:
            for l in order[lo[rr]:hi[rr]]:
                if np.any((SL[l] + SR[rr]) % P1 != r1):
                    continue
                combo = decode(int(l), sizes[:h]) + decode(int(rr), sizes[h:])
                v2 = sum((d2[i] * opts[i].m2[combo[i]]) % P2 for i in range(r)) % P2
                if np.any(v2 != r2):
                    continue
                vC = sum(dC[i] * opts[i].vecs[combo[i]] for i in range(r))
                if np.abs(vC - rC).max() > 1e-7:
                    continue
                out.append(tuple(combo))
        return out

    def composite(self, opts, d1, rows, x0, rhs1, stats):
        r = len(opts)
        alive = [np.ones(len(R), dtype=bool) for R in rows]

        def rec(t, alive):
            if t == len(COMP):
                sel = []
                for i in range(r):
                    a = np.flatnonzero(alive[i])
                    assert len(a) == 1
                    sel.append(rows[i][a[0]])
                yield np.array(sel).T          # (6, r) -> indexed [t][i]
                return
            x = add(x0, COMP[t])
            target = rhs1(x)
            cands = [np.unique(rows[i][alive[i], t]) for i in range(r)]
            S = (d1[0] * opts[0].m1[cands[0]]) % P1
            for i in range(1, r):
                S = (S[:, None, :] + (d1[i] * opts[i].m1[cands[i]])[None, :, :]).reshape(-1, S.shape[1]) % P1
            ok = np.flatnonzero(np.all(S == target[None, :], axis=1))
            stats["composite_candidates"] += len(S)
            sizes = [len(c) for c in cands]
            for o in ok:
                combo = decode(int(o), sizes)
                new = [alive[i] & (rows[i][:, t] == cands[i][combo[i]]) for i in range(r)]
                yield from rec(t + 1, new)

        for sel in rec(0, alive):
            yield [tuple(int(v) for v in sel[t]) for t in range(len(COMP))]

    def confirm(self, cover, opts, codes, x0, target_C, target_2):
        r = len(cover)
        dim2 = 3 ** self.n2
        terms = np.zeros((9 * dim2, r), dtype=complex)
        terms2 = np.zeros((r, 9 * dim2), dtype=np.int64)
        for i in range(r):
            terms[pidx(x0) * dim2:(pidx(x0) + 1) * dim2, i] = opts[i].u
            terms2[i, pidx(x0) * dim2:(pidx(x0) + 1) * dim2] = opts[i].m2[0]
            for off, cvec in codes.items():
                x = add(x0, off)
                terms[pidx(x) * dim2:(pidx(x) + 1) * dim2, i] = opts[i].vecs[cvec[i]]
                terms2[i, pidx(x) * dim2:(pidx(x) + 1) * dim2] = opts[i].m2[cvec[i]]
        c, *_ = np.linalg.lstsq(terms, target_C, rcond=None)
        res = float(np.linalg.norm(terms @ c - target_C))
        rank = int(np.linalg.matrix_rank(terms, tol=1e-8))
        r0 = rank_mod(terms2, P2)
        r1 = rank_mod(np.vstack((terms2, target_2[None, :])), P2)
        exact = r0 == r1
        if exact != (res < 1e-7):
            raise AssertionError("modular and numeric confirmations disagree")
        if not exact or rank < r or np.any(np.abs(c) < 1e-9):
            return None
        return {"terms": terms, "coeffs": c, "residual": res, "rank": rank}


def decode(idx, sizes):
    out = []
    for n in reversed(sizes):
        out.append(idx % n)
        idx //= n
    return list(reversed(out))


# ------------------------------------------------------------- targets ----

def psi_target(orbit, F1, F2):
    """rhs functions and full vectors for psi^4 sliced along qutrits 1, 2."""
    a1, a2, aC = F1.alpha(orbit), F2.alpha(orbit), alpha_C(orbit)
    p1 = np.kron(a1, a1) % P1
    p2 = np.kron(a2, a2) % P2
    pC = np.kron(aC, aC)
    rhs1 = lambda x: (a1[x[0]] * a1[x[1]] * p1) % P1
    rhs2 = lambda x: (a2[x[0]] * a2[x[1]] * p2) % P2
    rhsC = lambda x: aC[x[0]] * aC[x[1]] * pC
    tC = np.kron(pC, pC)
    t2 = np.kron(p2, p2) % P2              # p2 already reduced: no int64 overflow
    return rhs1, rhs2, rhsC, tC, t2


def vector_target(T_C, T_2, T_1, n2):
    dim2 = 3 ** n2
    rhs1 = lambda x: T_1[pidx(x) * dim2:(pidx(x) + 1) * dim2]
    rhs2 = lambda x: T_2[pidx(x) * dim2:(pidx(x) + 1) * dim2]
    rhsC = lambda x: T_C[pidx(x) * dim2:(pidx(x) + 1) * dim2]
    return rhs1, rhs2, rhsC, T_C, T_2


def base_of(T_C, x0, n2, D):
    """Dictionary indices of the slices at x0 of the term vectors T_C
    (columns), requiring each to be a dictionary state."""
    dim2 = 3 ** n2
    codes, C = patterns(D)
    key = {c.tobytes(): i for i, c in enumerate(codes)}
    out = []
    for col in T_C.T:
        s = col[pidx(x0) * dim2:(pidx(x0) + 1) * dim2]
        j = np.flatnonzero(np.abs(s) > 1e-9)
        if not len(j):
            return None
        cd = phase_codes(s / s[j[0]]).astype(np.int8)
        if cd.tobytes() not in key:
            raise AssertionError("a base slice is not a dictionary state")
        out.append(key[cd.tobytes()])
    return tuple(out)


def field_vector(v_C, F):
    """A complex vector with entries in Z[w] (small integers) to F_p."""
    out = np.zeros(len(v_C), dtype=np.int64)
    for k, z in enumerate(v_C):
        # z = a + b w with a, b integers
        b = int(round(z.imag / W3.imag))
        a = int(round(z.real - b * W3.real))
        assert abs(a + b * W3 - z) < 1e-6
        out[k] = (a + b * F.w) % F.p
    return out


# -------------------------------------------------------------- commands ----

def arg(args, key, default, conv=str):
    return conv(args[args.index(key) + 1]) if key in args else default


def cmd_sample(args):
    orbit, path = args[0], args[1]
    count = arg(args, "--count", 100, int)
    x0 = tuple(int(v) for v in arg(args, "--x0", "0,0").split(","))
    os.nice(19)
    F1, F2 = Field3(P1), Field3(P2)
    D = dictionary(3, 2)
    M = Matcher2(D, 2, F1, F2)
    with open(path) as f:
        cen = json.load(f)
    covers = [tuple(c) for c in cen["covers5"]]
    rng = np.random.default_rng(1)
    pick = rng.choice(len(covers), size=min(count, len(covers)), replace=False)
    rhs1, rhs2, rhsC, tC, t2 = psi_target(orbit, F1, F2)
    times, hist, hits, joined, cand = [], {}, 0, 0, 0
    for k in pick:
        cover = covers[k]
        t = time.time()
        h, st = M.run(cover, x0, rhs1, rhs2, rhsC, tC, t2)
        times.append(time.time() - t)
        key = tuple(st["coord_solutions"])
        hist[key] = hist.get(key, 0) + 1
        hits += len(h)
        joined += st["joined"]
        cand += st["composite_candidates"]
        if st["refused"]:
            print("refused", cover)
    times = np.array(times)
    print(f"{orbit} 2+2 stage A, x0={x0}: {len(pick)} covers, per cover mean {times.mean():.4f}s, "
          f"median {np.median(times):.4f}s, max {times.max():.4f}s; joined states {joined}, composite "
          f"candidates {cand}, hits {hits}")
    print(f"  coordinate-slice solution histogram: {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
    return 0


def random_term(rng, n, need_full=None):
    from common import term_vector
    from to_witness import _rref
    while True:
        k = int(rng.integers(2, n + 1))
        W = rng.integers(0, 3, size=(k, n))
        if len(_rref(W.tolist(), 3)[0]) != k:
            continue
        Q = np.triu(rng.integers(0, 3, size=(k, k)))
        term = {"k": k, "x0": rng.integers(0, 3, size=n).tolist(), "W": W.tolist(),
                "Q": Q.tolist(), "l": rng.integers(0, 3, size=k).tolist()}
        v = term_vector(term, 3, n)
        j = np.flatnonzero(np.abs(v) > 1e-9)[0]
        v = v / v[j]
        return v


def cmd_planted(args):
    """Planted five-term decompositions of a random target: the terms are
    random four-qutrit stabilizer states, the coefficients small integers,
    and the base x0 a point where all five slices are nonzero, distinct and
    independent; the matcher must return the planted decomposition."""
    count = arg(args, "--count", 10, int)
    os.nice(19)
    F1, F2 = Field3(P1), Field3(P2)
    D = dictionary(3, 2)
    M = Matcher2(D, 2, F1, F2)
    rng = np.random.default_rng(5)
    done, extra, tsum = 0, 0, 0.0
    while done < count:
        terms = np.column_stack([random_term(rng, 4) for _ in range(5)])
        coeffs = rng.integers(1, 3, size=5) * rng.choice([1, -1], size=5)
        T = terms @ coeffs.astype(complex)
        x0 = PTS[int(rng.integers(0, 9))]
        base = base_of(terms, x0, 2, D)
        if base is None or len(set(base)) < 5:
            continue
        if np.linalg.matrix_rank(D[:, list(base)], tol=1e-8) < 5:
            continue
        T_1, T_2 = field_vector(T, F1), field_vector(T, F2)
        t = time.time()
        h, st = M.run(base, x0, *vector_target(T, T_2, T_1, 2))
        tsum += time.time() - t
        if st["refused"]:
            continue
        # the planted decomposition must be among the hits (same term span)
        found = False
        for hit in h:
            A = np.column_stack((hit["terms"], terms))
            if np.linalg.matrix_rank(A, tol=1e-8) == 5:
                found = True
        if not found:
            raise AssertionError(f"planted decomposition not recovered: {st}")
        extra += len(h) - 1
        done += 1
    print(f"planted control: {done} random five-term decompositions recovered from their base slice "
          f"({extra} further decompositions with the same base), {tsum / done:.3f}s per run")
    return 0


def cmd_product(args):
    """phi (x) psi^2 with phi a full-support two-qutrit stabilizer state has
    the three-term decompositions phi (x) (rank-3 decomposition of psi^2);
    the matcher must find them from the base slice at any x0."""
    orbit = args[0]
    os.nice(19)
    F1, F2 = Field3(P1), Field3(P2)
    D = dictionary(3, 2)
    M = Matcher2(D, 2, F1, F2)
    from common import load_decompositions
    decs, _ = load_decompositions(orbit, 2, 3)
    codes, C = patterns(D)
    key = {c.tobytes(): i for i, c in enumerate(codes)}
    rng = np.random.default_rng(3)
    q = rng.integers(0, 3, size=5)
    phi = np.array([W3 ** ((q[0] * x[0] * x[0] + q[1] * x[1] * x[1] + q[2] * x[0] * x[1] + q[3] * x[0] + q[4] * x[1]) % 3)
                    for x in PTS])
    aC = alpha_C(orbit)
    psi2 = np.kron(aC, aC)
    T = np.kron(phi, psi2)
    phi1, phi2 = field_vector(phi, F1), field_vector(phi, F2)
    a1, a2 = F1.alpha(orbit), F2.alpha(orbit)
    T_1 = np.kron(phi1, np.kron(a1, a1) % P1) % P1
    T_2 = np.kron(phi2, np.kron(a2, a2) % P2) % P2
    nhit, nrun, t0 = 0, 0, time.time()
    for u, d in decs[:5]:
        base = []
        for ui in u:
            j = np.flatnonzero(np.abs(ui) > 1e-9)[0]
            base.append(key[phase_codes(ui / ui[j]).astype(np.int8).tobytes()])
        for x0 in [(0, 0), (1, 2)]:
            h, st = M.run(tuple(base), x0, *vector_target(T, T_2, T_1, 2))
            nrun += 1
            if st["refused"]:
                raise AssertionError("product base refused")
            if not h:
                raise AssertionError(f"product decomposition not found at {x0}: {st}")
            nhit += len(h)
    print(f"product control ({orbit}): {nrun} runs over 5 stored rank-3 decompositions x 2 base points, "
          f"{nhit} genuine three-term decompositions of phi (x) psi^2 found, none missing [{time.time() - t0:.1f}s]")
    return 0


def cmd_lifts13(args):
    """Cost of the 1 + 3 matcher (slice_lift.lifts, five terms) on 5-covers
    of psi^3 from a census file."""
    from slice_lift import lifts
    orbit, path = args[0], args[1]
    count = arg(args, "--count", 20, int)
    os.nice(19)
    with open(path) as f:
        cen = json.load(f)
    covers = [tuple(c) for c in cen["covers5"]][:count]
    D = dictionary(3, 3)
    psi = psi_for(orbit, 3)
    aC = alpha_C(orbit)
    times, nm = [], 0
    for cover in covers:
        states = [D[:, c] for c in cover]
        d, *_ = np.linalg.lstsq(np.column_stack(states), psi, rcond=None)
        t = time.time()
        L, (nmatch, floor) = lifts(states, d, aC, psi, 3)
        times.append(time.time() - t)
        nm += nmatch
        if L:
            print("LIFT FOUND", cover)
    times = np.array(times)
    print(f"{orbit} 1+3 matcher (slice_lift.lifts, 81^2 x 81^3 meet in the middle in C^27): {len(covers)} covers, "
          f"per cover mean {times.mean():.2f}s, median {np.median(times):.2f}s, max {times.max():.2f}s, "
          f"{nm} slice-2 matches, 0 lifts")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit({"sample": cmd_sample, "planted": cmd_planted, "product": cmd_product,
              "lifts13": cmd_lifts13}[cmd](sys.argv[2:]))
