"""The p = 3 matcher for a two-qutrit base slice: every decomposition of a
target on 2 + n2 qutrits whose slice along the first two qutrits at the base
point x0 is a given multiset of n2-qutrit stabilizer states.

Setting (docs/notes/qutrit_m4_rank5_exclusion.md, sections 1, 2 and 4). A
decomposition T = sum_i c_i s_i sliced along qutrits 1, 2 gives
u_i^(x) = (<x| (x) I) s_i for x in F_3^2 and sum_i c_i u_i^(x) = T^(x). The
nonzero slices of a term form an affine flat through x0 of dimension 2, 1
or 0, and by the two-qutrit structure lemma

    u^(x0 + x) = w^{q(x)} Q_2^{x_2} Q_1^{x_1} u^(x0)

for a plane term (Q_1, Q_2 Paulis, q a quadratic on F_3^2 with q(0) = 0),
u^(x0 + t v) = w^{a t^2 + b t} Q^t u^(x0) for a line term along v, and a
single nonzero slice for a point term. Per term the options at a slice are
the 27 vectors w^l Q u (9 Pauli classes modulo the stabilizer of u, 3 cube
roots; code 3 k + l) and absent (code 27).

Matching, in the shape of verify_challenge/slice_cover.SliceMatcher. The
multiset is split into its distinct states, whose coefficients form an
affine family d = d0 + K lambda over F_65521, F_2013265921 and C (a point
when the states are independent), and into blocks of repeated copies of one
state (at every slice the copies contribute a vector of the span of at most
g Pauli translates of the state). The two coordinate slices x0 + e_1,
x0 + e_2 are solved against the family (meet in the middle on a random
functional mod 65521 when the family is a point, the Laplace-feature dense
solve of slice_cover when parameters remain; every candidate is decided
exactly over the three fields) and each solution restricts the family. The
absence pattern at the coordinate slices fixes the flat type of every
ordinary term: present at both, a plane (27 quadratics); present at one,
that coordinate line (3 phases at its third point); absent at both, the
point or one of the two diagonal lines (1 + 81 + 81 shapes). The six
composite points are solved one at a time over the shapes still alive,
each solution restricting the family and the shapes; the blocks' copies are
reconstructed from the residual coordinates at the end; every hit is
confirmed against the whole target numerically and mod 2013265921.

The matcher allows every flat through x0. Facts A and B of the note only
guarantee that the enumerated bases (full 5-covers of |M>^2 at one base
point) are complete; nothing here assumes them. Generic in the number of
terms and in n2, so that the controls at m = 3 (n2 = 1) and at ranks 7 and
8 run through the same code.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
import slice_cover  # noqa: E402
from cover_census import P1, P2, Field3, patterns, rank_mod  # noqa: E402
from slice_cover import Family, _affine_solve_mod, _projectors, slice_system, solve_slice  # noqa: E402


def _affine_solve_C(A, b, tol=1e-7):
    """All solutions of A x = b over C as (x0, N), or None: the truncated-SVD
    solve, singular values below 1e-9 max(1, s_0) treated as zero. The
    slice_cover version solves with numpy's relative cutoff, so a matrix that
    is zero up to rounding (the sum-zero directions of a block's copies hit
    by two equal phases give A K of order 1e-16) is inverted into a garbage
    pin instead of being reported as a consistency condition; this version
    replaces it for every caller in this module and in slice_cover.Family."""
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if A.shape[1] == 0:
        return (np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=complex)) if np.linalg.norm(b) < tol else None
    if A.shape[0] == 0:
        return np.zeros(A.shape[1], dtype=complex), np.eye(A.shape[1], dtype=complex)
    U, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > 1e-9 * max(1.0, s[0] if len(s) else 1.0)))
    x0 = np.zeros(A.shape[1], dtype=complex)
    if rank:
        x0 = vh[:rank].conj().T @ ((U[:, :rank].conj().T @ b) / s[:rank])
    if np.linalg.norm(A @ x0 - b) > tol * max(1.0, np.linalg.norm(b)):
        return None
    return x0, vh[rank:].conj().T


slice_cover._affine_solve_C = _affine_solve_C

W3 = np.exp(2j * np.pi / 3)
W3P = np.array([1, W3, W3 ** 2])
PTS = [(a, b) for a in range(3) for b in range(3)]
E1, E2 = (1, 0), (0, 1)
COMP = [(1, 1), (2, 0), (0, 2), (1, 2), (2, 1), (2, 2)]     # composite offsets, solved in this order
OFFSETS = [E1, E2] + COMP                                  # every nonzero offset
NUM_TOL = 1e-8


def constructions_common():
    """research/constructions/common.py (alpha, target, term_vector,
    load_decompositions), loaded by path since this directory has a common.py
    of its own."""
    spec = importlib.util.spec_from_file_location(
        "constructions_common", os.path.join(ROOT, "research", "constructions", "common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pidx(x):
    return 3 * (x[0] % 3) + (x[1] % 3)


def add(x, y):
    return ((x[0] + y[0]) % 3, (x[1] + y[1]) % 3)


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


def exact_codes(v):
    """Phase codes of a complex vector with entries in {0} u {w^k} times a
    common scalar (first nonzero entry made 1), as int8, and the scalar."""
    nz = np.flatnonzero(np.abs(v) > 1e-9)
    return phase_codes(v / v[nz[0]]).astype(np.int8), v[nz[0]]


def term_from_codes(codes):
    c = np.asarray(codes, dtype=np.int64)
    return np.where(c > 0, W3P[(c - 1) % 3], 0).astype(complex)


class TermOpts:
    """Slice options of one term with base slice u on n qutrits (u with first
    nonzero entry 1): code 3 k + l is w^l T[k], T[k] = Q_k u for the class
    representative Q_k (k = 0 the identity); code 3^(n+1) is absent. Also
    the composite tables class and phase of Q_{k2}^{x2} Q_{k1}^{x1} u."""

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
        self._rows = {}

    def code_of(self, v):
        for k, t in enumerate(self.T):
            j = np.flatnonzero(np.abs(t) > 1e-9)[0]
            ratio = v[j] / t[j]
            for l in range(3):
                if abs(ratio - W3 ** l) < 1e-6 and np.allclose(v, W3 ** l * t, atol=1e-9):
                    return k, l
        raise AssertionError("vector is not a phased Pauli translate of the base slice")

    def arrays(self):
        return self.m1, self.m2, self.vecs

    def composite_rows(self, c1, c2):
        """Codes at the six COMP offsets for every shape consistent with the
        coordinate codes c1 (at e_1) and c2 (at e_2), as a (shapes, 6) array."""
        key = (int(c1), int(c2))
        if key in self._rows:
            return self._rows[key]
        A = self.absent
        rows = []
        if c1 != A and c2 != A:
            k1, l1 = divmod(int(c1), 3)
            k2, l2 = divmod(int(c2), 3)
            for a, b, c in itertools.product(range(3), repeat=3):
                row = []
                for x in COMP:
                    q = (a * x[0] * x[0] + b * x[1] * x[1] + c * x[0] * x[1] + (l1 - a) * x[0] + (l2 - b) * x[1]) % 3
                    row.append(3 * self.cls[k1, k2, pidx(x)] + (q + self.g[k1, k2, pidx(x)]) % 3)
                rows.append(row)
        elif c1 != A:
            k1 = int(c1) // 3
            for l in range(3):
                row = [A] * 6
                row[COMP.index((2, 0))] = 3 * self.cls[k1, 0, pidx((2, 0))] + (self.g[k1, 0, pidx((2, 0))] + l) % 3
                rows.append(row)
        elif c2 != A:
            k2 = int(c2) // 3
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
        out = np.unique(np.array(rows, dtype=np.int64), axis=0)
        self._rows[key] = out
        return out

    def valid_codes(self, codes):
        """True when the codes (offset -> code over the eight nonzero
        offsets) are those of a stabilizer state with base slice u: the
        composite codes are one of the structure lemma's shapes for the
        coordinate codes."""
        rows = self.composite_rows(codes[E1], codes[E2])
        want = np.array([codes[x] for x in COMP], dtype=np.int64)
        return bool((rows == want[None, :]).all(axis=1).any())


class Block:
    """g copies of one base state: at every slice they contribute a vector
    of the span of at most g Pauli translates Q_k u (the 3^n translates are
    a basis), so the block enters a slice equation through the translate set
    S only, and the equation is projected onto the annihilator of S. The
    attributes are those slice_cover.solve_slice and _projectors read."""

    def __init__(self, o, g, pos):
        self.o, self.g, self.pos = o, g, pos
        K = o.nclass
        self.subsets = [S for j in range(g + 1) for S in itertools.combinations(range(K), j)]
        cols = [3 * k for k in range(K)]
        self.V1 = np.ascontiguousarray(o.m1[cols].T)
        self.V2 = np.ascontiguousarray(o.m2[cols].T)
        self.VC = np.ascontiguousarray(o.vecs[cols].T)


def family_from(U1, U2, C, b1, b2, bC):
    """The affine family of coefficients with sum d_i u_i = b over the three
    fields (rows of U1, U2 and columns of C the states), or None."""
    s1 = _affine_solve_mod(np.asarray(U1).T, b1, P1)
    s2 = _affine_solve_mod(np.asarray(U2).T, b2, P2)
    sC = _affine_solve_C(C, bC)
    if s1 is None or s2 is None or sC is None:
        return None
    return Family([s1, s2, sC])


def restrict(fam, Ws, rhss):
    """fam.restrict with a fast path for a pinned family: the slice equation
    is then a plain equality over the three fields."""
    if fam.kappa1 or fam.kappa:
        return fam.restrict(Ws, rhss)
    d1, d2, dC = fam.parts[0][0], fam.parts[1][0], fam.parts[2][0]
    W1, W2, WC = Ws
    r1, r2, rC = rhss
    if len(rC) == 0:
        return fam                    # the blocks' translates span the slice: nothing to check
    if np.any(((W1 % P1) @ (d1 % P1)) % P1 != r1 % P1):
        return None
    if np.abs(WC @ dC - rC).max() > 1e-7 * max(1.0, float(np.abs(rC).max())):
        return None
    acc = np.zeros(W2.shape[0], dtype=np.int64)
    for i in range(W2.shape[1]):
        acc = (acc + (W2[:, i] % P2) * (int(d2[i]) % P2) % P2) % P2
    if np.any(acc != r2 % P2):
        return None
    return fam


def solve_slice3(opts, blocks, fam, rhs, rng, stats=None, log=None):
    """slice_cover.solve_slice, plus the case of no ordinary term (every
    distinct state repeated, which happens in the m = 3 control where
    chi(|M>) = 2): the equation is then that the slice lies in the span of
    the chosen translates, i.e. its projection onto their annihilator
    vanishes over the three fields."""
    if opts:
        return solve_slice(opts, blocks, fam, rhs, rng, stats=stats, log=log)
    out = []
    for Ssel in itertools.product(*[b.subsets for b in blocks]):
        Ps, _ = _projectors(blocks, Ssel)
        ok = True
        for fld, p in ((0, P1), (1, P2), (2, None)):
            P = Ps[fld]
            if P.shape[0] == 0:
                continue
            if p is None:
                ok &= bool(np.abs(P @ rhs[2]).max() < 1e-7 * max(1.0, float(np.abs(rhs[2]).max())))
            else:
                acc = np.zeros(P.shape[0], dtype=np.int64)
                for j in range(P.shape[1]):
                    acc = (acc + (P[:, j] % p) * (int(rhs[fld][j]) % p) % p) % p
                ok &= not np.any(acc)
            if not ok:
                break
        if ok:
            out.append(((), Ssel))
    if stats is not None:
        stats["candidates"] = stats.get("candidates", 0) + len(out)
    return out


def _refine_split(D, S, coords, splits, tol=1e-7):
    """Admissible (c_1, c_2), c_1 + c_2 = D, for a pair of copies whose slice
    contribution has coordinates `coords` on the translates S; None means
    unconstrained (slice_cover._refine_split with cube roots)."""
    def close(x, y):
        return abs(x - y) < tol * max(1.0, abs(D))
    if len(S) == 0:
        return splits
    cands, free = [], False
    if len(S) == 2:
        a1, a2 = coords
        for l1 in range(3):
            for l2 in range(3):
                c1, c2 = a1 / W3P[l1], a2 / W3P[l2]
                if close(c1 + c2, D):
                    cands += [(c1, c2), (c2, c1)]
    else:
        a = coords[0]
        for l1 in range(3):
            for l2 in range(3):
                if l1 == l2:
                    if close(a, D * W3P[l1]):
                        free = True
                else:
                    c1 = (a - D * W3P[l2]) / (W3P[l1] - W3P[l2])
                    cands.append((c1, D - c1))
        for l in range(3):
            cj = a / W3P[l]
            cands += [(cj, D - cj), (D - cj, cj)]
    cands = [(c1, c2) for c1, c2 in cands if abs(c1) > tol and abs(c2) > tol]
    if free:
        return splits
    if splits is None:
        return cands
    return [(c1, c2) for c1, c2 in splits if any(close(c1, e1) and close(c2, e2) for e1, e2 in cands)]


def reconstruct_block(o, g, D, data, rng):
    """Copies of a block from its per-slice residual coordinates. data: list
    of (offset, {class: coordinate}) over the eight non-base offsets, with
    the block's contribution at that offset sum_k a_k Q_k u. Returns every
    assignment (a list of (coefficient, codes) per copy, copies distinct and
    unordered) with coefficients summing to D, all nonzero, and each copy a
    valid stabilizer state; the flag says whether a coefficient family
    remained (degenerate)."""
    ones = np.ones((g, 1), dtype=complex)
    c0 = np.full(g, D / g, dtype=complex)
    Kc = _affine_solve_C(ones.T, np.zeros(1, dtype=complex))[1]     # sum-zero directions
    results = []

    def dfs(idx, c0, Kc, codes):
        if idx == len(data):
            degenerate = Kc.shape[1] > 0
            c = c0 + Kc @ (rng.normal(size=Kc.shape[1]) + 1j * rng.normal(size=Kc.shape[1])) if degenerate else c0
            if np.any(np.abs(c) < 1e-9):
                return
            full = [dict(cd) for cd in codes]
            for cd in full:
                if not o.valid_codes(cd):
                    return
            keyset = [tuple(sorted(cd.items())) for cd in full]
            if len(set(keyset)) < g:
                return
            results.append((list(zip(c, full)), degenerate))
            return
        x, a = data[idx]
        classes = sorted(a)
        for assign in itertools.product(classes + [None], repeat=g):
            if {k for k in assign if k is not None} != set(classes):
                continue
            present = [j for j in range(g) if assign[j] is not None]
            for phases in itertools.product(range(3), repeat=len(present)):
                A = np.zeros((len(classes), g), dtype=complex)
                b = np.array([a[k] for k in classes], dtype=complex)
                for j, l in zip(present, phases):
                    A[classes.index(assign[j]), j] = W3P[l]
                sol = _affine_solve_C(A @ Kc, b - A @ c0)
                if sol is None:
                    continue
                mu0, N = sol
                new = [dict(cd) for cd in codes]
                for j in range(g):
                    new[j][x] = o.absent
                for j, l in zip(present, phases):
                    new[j][x] = 3 * assign[j] + l
                dfs(idx + 1, c0 + Kc @ mu0, Kc @ N, new)

    dfs(0, c0, Kc, [dict() for _ in range(g)])
    seen, out = set(), []
    for copies, degenerate in results:
        key = tuple(sorted((tuple(sorted(cd.items())), complex(np.round(c, 8))) for c, cd in copies))
        if key not in seen:
            seen.add(key)
            out.append((copies, degenerate))
    return out


# --------------------------------------------------------------- targets ----

class Target:
    """A target on 2 + n2 qutrits over C, F_P1 and F_P2, with the first two
    qutrits the sliced ones: entry pidx(x) 3^n2 + y is the slice at x."""

    def __init__(self, T_C, T_1, T_2, n2):
        self.n2, self.dim2 = n2, 3 ** n2
        self.T_C = np.asarray(T_C, dtype=complex)
        self.T_1 = np.asarray(T_1, dtype=np.int64) % P1
        self.T_2 = np.asarray(T_2, dtype=np.int64) % P2
        assert len(self.T_C) == 9 * self.dim2

    def rhs(self, x):
        s = slice(pidx(x) * self.dim2, (pidx(x) + 1) * self.dim2)
        return self.T_1[s], self.T_2[s], self.T_C[s]

    @property
    def m(self):
        return 2 + self.n2


def psi_target(orbit, n2, F1, F2):
    """|M>^(2 + n2) as a Target."""
    a1, a2, aC = F1.alpha(orbit), F2.alpha(orbit), constructions_common().alpha(orbit)
    v1, v2, vC = a1, a2, aC
    for _ in range(1 + n2):
        v1 = np.kron(v1, a1) % P1
        v2 = np.kron(v2, a2) % P2
        vC = np.kron(vC, aC)
    return Target(vC, v1, v2, n2)


def field_vector(v_C, F):
    """A complex vector with entries in Z[w] to F_p."""
    out = np.zeros(len(v_C), dtype=np.int64)
    for k, z in enumerate(v_C):
        b = int(round(z.imag / W3.imag))
        a = int(round(z.real - b * W3.real))
        assert abs(a + b * W3 - z) < 1e-6, "entry is not in Z[w]"
        out[k] = (a + b * F.w) % F.p
    return out


def vector_target(T_C, n2, F1, F2):
    """A Target from a complex vector with entries in Z[w]."""
    return Target(T_C, field_vector(T_C, F1), field_vector(T_C, F2), n2)


# --------------------------------------------------------------- matcher ----

class Matcher:
    """Every decomposition of a target with a given base slice at a given
    base point along the first two qutrits (see the module note)."""

    def __init__(self, D, n2, F1=None, F2=None, seed=29, verbose=False):
        self.D, self.n2 = D, n2
        self.codes, self.C = patterns(D)
        self.F1 = F1 or Field3(P1)
        self.F2 = F2 or Field3(P2)
        self.U1 = self.F1.codes_to_field(self.codes)
        self.U2 = self.F2.codes_to_field(self.codes)
        self.N = D.shape[1]
        self.lookup = {self.codes[i].tobytes(): i for i in range(self.N)}
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def options(self, idx):
        if idx not in self.cache:
            self.cache[idx] = TermOpts(self.C[:, idx], self.n2, self.F1, self.F2)
        return self.cache[idx]

    def index_of(self, v):
        """Dictionary index of a slice vector (a dictionary state up to a
        scalar), or None when it is zero; raises when it is not a state."""
        if np.linalg.norm(v) < 1e-9:
            return None
        cd, _ = exact_codes(v)
        key = cd.tobytes()
        if key not in self.lookup:
            raise AssertionError("a base slice is not a dictionary state")
        return self.lookup[key]

    def run(self, cover, x0, target):
        """(hits, stats) for the base slice `cover` (a tuple of dictionary
        indices, repeats allowed) at x0 against `target`; each hit is a dict
        with the term vectors, coefficients, residual, rank and the exact
        span decision."""
        x0 = tuple(int(v) for v in x0)
        stats = {"kappa": None, "kappa1": None, "distinct": 0, "blocks": [], "coord_solutions": [],
                 "joined": 0, "composite_solutions": 0, "candidates": 0, "zero_coefficient": 0,
                 "split_pruned": 0, "reconstructions": 0, "hits": 0, "refused": False}
        distinct = sorted(set(int(u) for u in cover))
        mult = {u: list(cover).count(u) for u in distinct}
        b1, b2, bC = target.rhs(x0)
        fam = family_from(self.U1[distinct], self.U2[distinct], self.C[:, distinct], b1, b2, bC)
        if fam is None:
            stats["refused"] = True
            return [], stats
        blocks = [Block(self.options(u), mult[u], i) for i, u in enumerate(distinct) if mult[u] > 1]
        bpos = {b.pos for b in blocks}
        exempt = tuple(sorted(bpos))
        if fam.has_zero_coefficient(exempt):
            stats["refused"] = True
            return [], stats
        ords = [i for i in range(len(distinct)) if i not in bpos]
        opts = [self.options(distinct[i]) for i in ords]
        arrays = [o.arrays() for o in opts]
        stats.update(kappa=fam.kappa, kappa1=fam.kappa1, distinct=len(distinct), blocks=[b.g for b in blocks])
        states = [([], [], fam, [None] * len(blocks))]
        for e in (E1, E2):
            rhs = target.rhs(add(x0, e))
            new, count = [], 0
            for cl, bl, f, sp in states:
                sols = solve_slice3(arrays, blocks, f, rhs, self.rng, stats=stats, log=self.log)
                count += len(sols)
                for combo, Ssel in sols:
                    f2 = restrict(f, *slice_system(arrays, blocks, combo, Ssel, rhs))
                    if f2 is None or f2.has_zero_coefficient(exempt):
                        stats["zero_coefficient"] += f2 is not None
                        continue
                    sp2 = self._join_blocks(f2, arrays, blocks, combo, Ssel, rhs, sp)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    new.append((cl + [tuple(int(c) for c in combo)], bl + [Ssel], f2, sp2))
            stats["coord_solutions"].append(count)
            states = new
            self.log(f"  slice {e}: {count} solutions, {len(states)} states, "
                     f"parameters {sorted(set(f.kappa for _, _, f, _ in states))}")
            if not states:
                return [], stats
        stats["joined"] = len(states)
        hits = []
        for cl, bl, f, sp in states:
            hits.extend(self._complete(distinct, x0, opts, ords, blocks, cl, bl, f, sp, target, stats))
        hits = self._dedupe(hits)
        stats["hits"] = len(hits)
        return hits, stats

    def _join_blocks(self, fam, arrays, blocks, combo, Ssel, rhs, splits):
        """With a pinned family the residual must use every chosen translate
        (else the solution duplicates a smaller set), and for a pair the
        admissible coefficient splits are refined. None drops the state."""
        if not blocks or fam.kappa:
            return splits
        a = self._block_coordinates(fam, arrays, blocks, combo, Ssel, rhs)
        if a is None:
            return splits
        if not np.all(np.abs(a) > 1e-9):
            return None
        d = fam.coefficients(self.rng)
        new, pos = [], 0
        for b, S, sp in zip(blocks, Ssel, splits):
            coords = a[pos:pos + len(S)]
            pos += len(S)
            if b.g != 2:
                new.append(sp)
                continue
            sp2 = _refine_split(d[b.pos], S, coords, sp)
            if sp2 is not None and len(sp2) == 0:
                return None
            new.append(sp2)
        return new

    def _block_coordinates(self, fam, arrays, blocks, combo, Ssel, rhs):
        """Coordinates of the residual rhs - W d on the chosen translates
        (complex), or None when it does not lie in their span or the split
        between blocks is ambiguous."""
        d = fam.coefficients(self.rng)
        r_all = len(arrays) + len(blocks)
        bpos = {b.pos for b in blocks}
        ords = [i for i in range(r_all) if i not in bpos]
        W = np.zeros((rhs[2].shape[0], r_all), dtype=complex)
        for i, c in zip(ords, combo):
            W[:, i] = arrays[ords.index(i)][2][c]
        res = rhs[2] - W @ d
        _, Vs = _projectors(blocks, Ssel)
        V = Vs[2]
        if V.shape[1] == 0:
            return np.zeros(0) if np.linalg.norm(res) < 1e-7 else None
        sol = _affine_solve_C(V, res)
        if sol is None or sol[1].shape[1]:
            return None
        return sol[0]

    def _complete(self, distinct, x0, opts, ords, blocks, cl, bl, fam, sp, target, stats):
        r = len(opts)
        dim2 = 3 ** self.n2
        rows = [opts[i].composite_rows(cl[0][i], cl[1][i]) for i in range(r)]
        exempt = tuple(sorted(b.pos for b in blocks))
        states = [([], [], fam, sp, [np.ones(len(R), dtype=bool) for R in rows])]
        for c, x in enumerate(COMP):
            rhs = target.rhs(add(x0, x))
            new = []
            for cl2, bl2, f, sp1, alive in states:
                codes = [np.unique(rows[i][alive[i], c]) for i in range(r)]
                arrays = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
                sols = solve_slice3(arrays, blocks, f, rhs, self.rng, stats=stats, log=self.log)
                stats["composite_solutions"] += len(sols)
                for combo, Ssel in sols:
                    f2 = restrict(f, *slice_system(arrays, blocks, combo, Ssel, rhs))
                    if f2 is None or f2.has_zero_coefficient(exempt):
                        stats["zero_coefficient"] += f2 is not None
                        continue
                    sp2 = self._join_blocks(f2, arrays, blocks, combo, Ssel, rhs, sp1)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    chosen = tuple(int(codes[i][combo[i]]) for i in range(r))
                    alive2 = [alive[i] & (rows[i][:, c] == chosen[i]) for i in range(r)]
                    new.append((cl2 + [chosen], bl2 + [Ssel], f2, sp2, alive2))
            states = new
            if not states:
                return []
        hits = []
        full_arrays = [o.arrays() for o in opts]
        for cl2, bl2, f, _, _ in states:
            ord_terms = []
            for i in range(r):
                o = opts[i]
                t = np.zeros((9, dim2), dtype=complex)
                t[pidx(x0)] = o.u
                t[pidx(add(x0, E1))] = o.vecs[cl[0][i]]
                t[pidx(add(x0, E2))] = o.vecs[cl[1][i]]
                for c, x in enumerate(COMP):
                    t[pidx(add(x0, x))] = o.vecs[cl2[c][i]]
                ord_terms.append(t.ravel())
            d = f.coefficients(self.rng)
            coeffs = [d[i] for i in ords]
            if not blocks:
                hits.append(self.confirm(ord_terms, f.kappa, target))
                continue
            per_block = [[] for _ in blocks]
            ok = True
            for s, x in enumerate(OFFSETS):
                if s < 2:
                    combo, Ssel = cl[s], bl[s]
                else:
                    combo, Ssel = cl2[s - 2], bl2[s - 2]
                rhs = target.rhs(add(x0, x))
                a = self._block_coordinates(f, full_arrays, blocks, combo, Ssel, rhs)
                if a is None:
                    ok = False
                    break
                pos = 0
                for bi, (b, S) in enumerate(zip(blocks, Ssel)):
                    per_block[bi].append((x, {k: a[pos + j] for j, k in enumerate(S)}))
                    pos += len(S)
            if not ok:
                continue
            recon = [reconstruct_block(b.o, b.g, d[b.pos], per_block[bi], self.rng)
                     for bi, b in enumerate(blocks)]
            stats["reconstructions"] += 1
            for choice in itertools.product(*recon):
                terms = list(ord_terms)
                degenerate = f.kappa > 0
                for b, (copies, deg) in zip(blocks, choice):
                    degenerate |= deg
                    for _, cd in copies:
                        t = np.zeros((9, dim2), dtype=complex)
                        t[pidx(x0)] = b.o.u
                        for x, code in cd.items():
                            t[pidx(add(x0, x))] = b.o.vecs[code]
                        terms.append(t.ravel())
                hits.append(self.confirm(terms, int(degenerate), target))
        return hits

    def confirm(self, terms, free, target):
        """Residual, coefficients and rank of the terms against the target
        in floating point, and the span condition mod P2."""
        A = np.column_stack(terms)
        c, *_ = np.linalg.lstsq(A, target.T_C, rcond=None)
        res = float(np.linalg.norm(A @ c - target.T_C))
        rank = int(np.linalg.matrix_rank(A, tol=1e-8))
        codes = np.array([exact_codes(t)[0] for t in terms])
        U2 = self.F2.codes_to_field(codes.astype(np.int64))
        r0 = rank_mod(U2, P2)
        r1 = rank_mod(np.vstack([U2, target.T_2[None, :]]), P2)
        return {"terms": terms, "coeffs": c, "residual": res, "rank": rank, "exact": r0 == r1,
                "independent": rank == len(terms), "nonzero": bool(np.all(np.abs(c) > 1e-9)),
                "free_parameters": free}

    @staticmethod
    def _dedupe(hits):
        seen, out = set(), []
        for h in hits:
            key = tuple(sorted(exact_codes(t)[0].tobytes() for t in h["terms"]))
            if key not in seen:
                seen.add(key)
                out.append(h)
        return out


def slice_base(matcher, terms, x0):
    """Dictionary indices of the slices at x0 of the term vectors (columns of
    `terms`, on 2 + n2 qutrits with the sliced qutrits first), or None when
    a term vanishes there."""
    dim2 = 3 ** matcher.n2
    out = []
    for col in np.asarray(terms).T:
        s = col[pidx(x0) * dim2:(pidx(x0) + 1) * dim2]
        k = matcher.index_of(s)
        if k is None:
            return None
        out.append(k)
    return tuple(out)


def move_front(v, S, m):
    """The vector v on m qutrits with the qutrits S (a pair) moved to the
    front, in the order given."""
    perm = list(S) + [q for q in range(m) if q not in S]
    return v.reshape([3] * m).transpose(perm).reshape(-1)
