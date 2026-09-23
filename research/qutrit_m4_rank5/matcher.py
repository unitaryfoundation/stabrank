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

A base with no ordinary term (every distinct state repeated) takes a
different route, BlockOnlyMatcher: every slice is solved on its own for the
translate selections whose span contains it (dependent selections
included, with their coordinate family), the two points of each line
through x0 are paired by the class map k -> 2k, the four lines are joined
by the per-block class relation of the structure lemma, and each survivor
is reconstructed with the coordinate families' parameters as unknowns
shared across the blocks and confirmed.

The matcher allows every flat through x0. Facts A and B of the note only
guarantee that the enumerated bases (full 5-covers of |M>^2 at one base
point) are complete; nothing here assumes them. Generic in the number of
terms and in n2, so that the controls at m = 3 (n2 = 1) and at ranks 7 and
8 run through the same code.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
import slice_cover  # noqa: E402
from cover_census import P1, P2, Field3, patterns, rank_mod  # noqa: E402
from slice_cover import (Family, _affine_solve_mod, _annihilator_mod, _dense, _det_mod,  # noqa: E402
                         _independent_columns_mod, _mitm, _projectors, _split_sides, slice_system)


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


# ---------------------------------------------------------------- budget ----

def peak_rss_gb():
    """Peak resident set size of this process in GB (ru_maxrss is bytes on
    macOS and KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


def current_rss_gb():
    """Current resident set size in GB: /proc/self/statm where it exists
    (Linux), else the peak."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * resource.getpagesize() / 2 ** 30
    except (OSError, IndexError, ValueError):
        return peak_rss_gb()


class BudgetExceeded(Exception):
    """A run refused for cost: the message names the checkpoint and the
    quantity that exceeded its cap."""


class UnpinnedFamily(Exception):
    """A state reached the final loop of the family path in a shape its
    block reconstruction does not handle: several blocks with parameters
    left in the coefficient family (the parameter would have to be shared
    across the blocks' reconstructions), or a strict coordinate solve that
    cannot place a slice residual on the chosen translates (the translates
    of two blocks dependent, so the split of the residual between them is
    a family; a nonzero residual with no translate chosen; a residual
    outside the span at the solve tolerance). A single block with a
    coefficient family is reconstructed with the family parameter as an
    unknown (reconstruct_block), so it does not raise. The run raises
    instead of deciding the state at an arbitrary member, and the batch
    records the cover as undecided."""


class Budget:
    """Caps a Matcher.run honours instead of growing without bound: the
    process's resident set (GB), the number of joined states carried from
    one slice to the next, the number of solutions any one slice equation
    may return, the estimated size (GB) of a dense coefficient-family
    solve, and a wall-clock deadline. Every cap is checked at the point
    where the next step would exceed it and raises BudgetExceeded, so the
    caller records the reason instead of being killed."""

    def __init__(self, max_rss_gb=None, max_states=None, max_solutions=None, max_dense_gb=None,
                 seconds=None):
        self.max_rss_gb = max_rss_gb
        self.max_states = max_states
        self.max_solutions = max_solutions
        self.max_dense_gb = max_dense_gb if max_dense_gb is not None else max_rss_gb
        self.deadline = None if seconds is None else time.time() + seconds
        self.checks = 0

    def check(self, where, states=None, solutions=None):
        self.checks += 1
        if self.max_rss_gb is not None:
            rss = current_rss_gb()
            if rss > self.max_rss_gb:
                raise BudgetExceeded(f"{where}: resident set {rss:.2f} GB exceeds the cap {self.max_rss_gb} GB")
        if self.deadline is not None and time.time() > self.deadline:
            raise BudgetExceeded(f"{where}: past the wall-clock deadline")
        if states is not None and self.max_states is not None and states > self.max_states:
            raise BudgetExceeded(f"{where}: {states} states exceed the cap {self.max_states}")
        if solutions is not None and self.max_solutions is not None and solutions > self.max_solutions:
            raise BudgetExceeded(f"{where}: {solutions} slice solutions exceed the cap {self.max_solutions}")

    def check_dense(self, where, sizes, kv, nsel):
        """Refuse a dense solve whose feature matrices would not fit: with k
        parameters and n = k + 1 the Laplace features number C(2n, n) per
        side, held as float64 rows over the side product, and `nsel`
        translate-set selections repeat the solve."""
        if not kv or self.max_dense_gb is None:
            return 0.0
        n = kv + 1
        feats = math.comb(2 * n, n)
        sides = _split_sides(sizes)
        prods = [math.prod(sizes[i] for i in s) for s in sides]
        gb = sum(p * (feats + n * n) * 8 for p in prods) / 2 ** 30
        if gb > self.max_dense_gb:
            raise BudgetExceeded(f"{where}: dense solve with {kv} parameters over sides {prods} needs about "
                                 f"{gb:.1f} GB of features ({feats} per side, {nsel} translate-set selections), "
                                 f"above the cap {self.max_dense_gb} GB")
        return gb


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


def _mm_int64(A, B, p):
    """A @ B mod p in int64 for p below 2^31: B is split into 16-bit halves
    so that every dot product stays below 2^63 (inner dimension below 2^15).
    Replaces slice_cover._mm, whose path for p above 2^26 (the exact
    re-decision prime P2) multiplies object arrays in a Python loop and was
    the largest single cost of Family.restrict."""
    A = np.asarray(A, dtype=np.int64) % p
    B = np.asarray(B, dtype=np.int64) % p
    if A.shape[-1] == 0:
        return np.zeros(A.shape[:-1] + B.shape[1:], dtype=np.int64)
    if p < (1 << 26):
        return (A @ B) % p
    if p >= (1 << 31) or A.shape[-1] >= (1 << 15):
        return slice_cover_mm(A, B, p)
    hi = (A @ (B >> 16)) % p
    lo = (A @ (B & 0xFFFF)) % p
    return ((hi << 16) + lo) % p


slice_cover_mm = slice_cover._mm
slice_cover._mm = _mm_int64


class _ProjectorCache:
    """slice_cover._projectors per translate-set selection, computed once
    per run (the blocks are fixed within Matcher.run)."""

    def __init__(self, blocks):
        self.blocks, self.cache = blocks, {}

    def __call__(self, Ssel):
        if Ssel not in self.cache:
            self.cache[Ssel] = _projectors(self.blocks, Ssel)
        return self.cache[Ssel]


# ------------------------------------------------------- block-only bases ----
#
# A base with no ordinary term (every distinct state repeated: the m = 3
# control and the rank-8 H3 witness bases) is matched without the
# coefficient family. Every one of the eight slices is solved on its own for
# the translate selections whose span contains it, the selections at the two
# points of every line through x0 are paired by the class map k -> 2k, the
# four lines are joined by the per-block class relation of the structure
# lemma, and each survivor is reconstructed per block with the dependent
# slices' coordinate parameters as unknowns shared across the blocks, the
# blocks joined on those parameters and every combination confirmed against
# the whole target. The earlier path (the family pinned by blocks whose
# copies separate, the two coordinate slices joined by family compatibility,
# the composite points solved over the joined states) missed every
# decomposition whose selection at some slice is dependent (the four H3
# witness base states have rank 3, so its own selection at x0 + e_2 and
# x0 + 2 e_2 is), never pinned a family through copies that stay in one
# class, and materialised the product of the two coordinate slices'
# solution lists.

LINES = [(E1, (2, 0)), (E2, (0, 2)), ((1, 1), (2, 2)), ((1, 2), (2, 1))]     # x and 2x through x0
SECOND = dict(LINES)


def _side_combos(blocks, side):
    """Per column count, the (part, V1 mod P1, VC) of every product of the
    side's blocks' subsets."""
    dim = blocks[0].VC.shape[0]
    out = {}
    for part in itertools.product(*[blocks[i].subsets for i in side]):
        cols1 = [blocks[i].V1[:, list(S)] for i, S in zip(side, part) if len(S)]
        colsC = [blocks[i].VC[:, list(S)] for i, S in zip(side, part) if len(S)]
        V1 = np.column_stack(cols1) % P1 if cols1 else np.zeros((dim, 0), dtype=np.int64)
        VC = np.column_stack(colsC) if colsC else np.zeros((dim, 0), dtype=complex)
        out.setdefault(V1.shape[1], []).append((part, V1, VC))
    return out


def block_only_slice(blocks, rhs, rng, budget=None, where="", stats=None, chunk=20_000):
    """Every translate selection (at most g classes per block) whose span
    contains the slice rhs = (mod P1, mod P2, over C): the independent
    selections with every coordinate nonzero (else the selection duplicates
    a smaller one), with their unique coordinates, and the dependent
    selections with their affine coordinate family a = a0 + N mu, dropping
    those with a coordinate that vanishes on the whole family. Returns
    (Ssel, a0, N) triples in block order, N with zero columns for an
    independent selection.

    Candidates come from the meet in the middle of slice_cover._dense over
    two halves of the blocks: per column count (a, b) of the halves the
    dependence of [left | right | slice] under a random projection to
    a + b + 1 coordinates mod P1 is a Laplace expansion into minors of the
    two sides. A dependent selection makes every minor vanish whatever the
    slice, so the candidates include every dependent selection; they are
    decided in batches by the singular values of the stacked column
    matrices and the projection residual of the slice (exact enough for
    roots of unity; every hit is confirmed exactly at the end)."""
    dim = blocks[0].VC.shape[0]
    r1 = np.asarray(rhs[0], dtype=np.int64) % P1
    T = np.asarray(rhs[2], dtype=complex)
    scale = max(1.0, float(np.linalg.norm(T)))
    sides = _split_sides([len(b.subsets) for b in blocks])
    L, R = _side_combos(blocks, sides[0]), _side_combos(blocks, sides[1])
    out, ncand = [], 0

    def assemble(lp, rp):
        """The selection in block order and the permutation taking the
        column order (left side, then right side) to block order."""
        Ssel = [None] * len(blocks)
        for i, S in zip(sides[0], lp):
            Ssel[i] = tuple(S)
        for i, S in zip(sides[1], rp):
            Ssel[i] = tuple(S)
        col, start = {}, 0
        for i, S in list(zip(sides[0], lp)) + list(zip(sides[1], rp)):
            col[i] = list(range(start, start + len(S)))
            start += len(S)
        perm = [c for i in range(len(blocks)) for c in col[i]]
        return tuple(Ssel), perm

    for a, Ls in L.items():
        VL1 = np.stack([V1 for _, V1, _ in Ls])
        VLC = np.stack([VC for _, _, VC in Ls])
        for b, Rs in R.items():
            if budget is not None:
                budget.check(where, solutions=len(out))
            n = a + b
            VR1 = np.stack([V1 for _, V1, _ in Rs])
            VRC = np.stack([VC for _, _, VC in Rs])
            if n == 0:
                if np.linalg.norm(T) < 1e-7 * scale:
                    Ssel, _ = assemble(Ls[0][0], Rs[0][0])
                    out.append((Ssel, np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=complex)))
                continue
            if n < dim:
                s = n + 1
                F = rng.integers(1, P1, size=(s, dim))
                ML = (F[None, :, :] @ VL1) % P1                                         # (nL, s, a)
                Rcol = np.broadcast_to(r1[None, :, None], (len(Rs), dim, 1))
                MR = (F[None, :, :] @ np.concatenate([VR1, Rcol], axis=2)) % P1         # (nR, s, b + 1)
                rows = list(range(s))
                subsets = list(itertools.combinations(rows, a))
                FL = np.empty((len(Ls), len(subsets)))
                FR = np.empty((len(Rs), len(subsets)))
                for t, I in enumerate(subsets):
                    Ic = [x for x in rows if x not in I]
                    sg = -1 if (sum(I) + sum(range(a))) % 2 else 1
                    FL[:, t] = _det_mod(ML[:, list(I), :], P1) if a else 1.0
                    FR[:, t] = (sg * _det_mod(MR[:, Ic, :], P1)) % P1
                Z = FL @ FR.T
                ia, ib = np.nonzero(np.fmod(Z, P1) == 0)
            else:
                ia, ib = np.indices((len(Ls), len(Rs))).reshape(2, -1)
            ncand += len(ia)
            for start in range(0, len(ia), chunk):
                if budget is not None:
                    budget.check(where, solutions=len(out))
                ca, cb = ia[start:start + chunk], ib[start:start + chunk]
                V = np.concatenate([VLC[ca], VRC[cb]], axis=2)                          # (N, dim, n)
                U, sv, _ = np.linalg.svd(V, full_matrices=False)
                k = sv.shape[1]
                rank = (sv > 1e-9 * np.maximum(1.0, sv[:, :1])).sum(axis=1)
                coef = np.einsum("nij,i->nj", U.conj(), T)
                coef = np.where(np.arange(k)[None, :] < rank[:, None], coef, 0)
                res = np.linalg.norm(T[None, :] - np.einsum("nij,nj->ni", U, coef), axis=1)
                for idx in np.flatnonzero(res < 1e-7 * scale):
                    sol = _affine_solve_C(V[idx], T)
                    if sol is None:
                        continue
                    a0, N = sol
                    if N.shape[1] == 0:
                        if not np.all(np.abs(a0) > 1e-9):
                            continue
                    elif np.any((np.abs(a0) < 1e-9) & (np.abs(N).sum(axis=1) < 1e-9)):
                        continue
                    Ssel, perm = assemble(Ls[ca[idx]][0], Rs[cb[idx]][0])
                    out.append((Ssel, a0[perm], N[perm]))
    if stats is not None:
        stats["candidates"] = stats.get("candidates", 0) + ncand
    return out


def _meet_C(m0a, Ma, m0b, Mb):
    """The intersection of two affine subspaces {m0 + M nu} of C^n as
    (m0, M) with M orthonormal, or None when empty."""
    if len(m0a) == 0:
        return m0a, Ma
    if Ma.shape[1] == 0 and Mb.shape[1] == 0:
        return (m0a, Ma) if np.abs(m0a - m0b).max() < 1e-6 * max(1.0, float(np.abs(m0a).max())) else None
    if Mb.shape[1] == 0:
        m0a, Ma, m0b, Mb = m0b, Mb, m0a, Ma
    if Ma.shape[1] == 0:                                  # is the point m0a in the subspace b?
        r = m0a - m0b
        r = r - Mb @ (Mb.conj().T @ r)
        return (m0a, Ma) if np.abs(r).max() < 1e-6 * max(1.0, float(np.abs(m0a).max())) else None
    sol = _affine_solve_C(np.concatenate([Ma, -Mb], axis=1), m0b - m0a)
    if sol is None:
        return None
    z0, Z = sol
    ka = Ma.shape[1]
    m0, Mn = m0a + Ma @ z0[:ka], Ma @ Z[:ka]
    if Mn.shape[1]:
        U, s, _ = np.linalg.svd(Mn, full_matrices=False)
        Mn = U[:, s > 1e-9]
    return m0, Mn


def _orthonormal(T):
    if T.shape[1] == 0:
        return T
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    return U[:, s > 1e-9]


def copy_patterns(o):
    """The class patterns of one copy of a block over the four lines through
    x0: for every pair of coordinate codes and every shape of the structure
    lemma, the class (None when absent) at e_1, e_2, (1, 1) and (1, 2). The
    class at the second point of each line is 2k for the class k at the
    first (checked). Returns (patterns, the doubling map k -> 2k)."""
    A = o.absent
    dbl = {k: int(o.cls[k, 0, pidx((2, 0))]) for k in range(o.nclass)}

    def cl(code):
        return None if int(code) == A else int(code) // 3

    pats = set()
    for c1 in range(A + 1):
        for c2 in range(A + 1):
            for row in o.composite_rows(c1, c2):
                r = {x: int(v) for x, v in zip(COMP, row)}
                r[E1], r[E2] = c1, c2
                pat = tuple(cl(r[x]) for x, _ in LINES)
                for (x, y), k in zip(LINES, pat):
                    ky = cl(r[y])
                    if (k is None) != (ky is None) or (k is not None and dbl[k] != ky):
                        raise AssertionError("the class at the second point of a line is not the doubled class")
                pats.add(pat)
    return sorted(pats, key=lambda p: [-1 if v is None else v for v in p]), dbl


def block_relation(pats, g, drive):
    """For g copies with the given one-copy patterns: the map from the
    block's class sets on the two driving lines `drive` (indices into
    LINES) to the set of its class sets on the other two lines, in LINES
    order."""
    other = [i for i in range(4) if i not in drive]
    rel = {}
    for combo in itertools.combinations_with_replacement(pats, g):
        sets = [tuple(sorted({p[i] for p in combo if p[i] is not None})) for i in range(4)]
        rel.setdefault((sets[drive[0]], sets[drive[1]]), set()).add((sets[other[0]], sets[other[1]]))
    return rel


def _is_root_shift(z):
    """l with z = w^l, or None."""
    for l in range(3):
        if abs(z - W3P[l]) < 1e-7:
            return l
    return None


def _solve_configs(A, rhs, tol=1e-7):
    """Batched affine solve of A[i] z = rhs[i] for a stack A (n, r, c),
    rhs (n, r): per configuration (z0, Z) with Z the orthonormal null space
    of A[i], or None when inconsistent."""
    n, r, c = A.shape
    if n == 0:
        return []
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    k = s.shape[1]
    rank = (s > 1e-9 * np.maximum(1.0, s[:, :1])).sum(axis=1)
    coef = np.einsum("nij,ni->nj", U.conj(), rhs)[:, :k]
    inv = np.where(np.arange(k)[None, :] < rank[:, None], 1.0 / np.where(s > 0, s, 1.0), 0.0)
    z0 = np.einsum("nji,nj->ni", Vh[:, :k, :].conj(), coef * inv)
    res = np.linalg.norm(np.einsum("nij,nj->ni", A, z0) - rhs, axis=1)
    ok = res < tol * np.maximum(1.0, np.linalg.norm(rhs, axis=1))
    return [(z0[i], Vh[i, rank[i]:, :].conj().T) if ok[i] else None for i in range(n)]


class BlockConfigs:
    """The consistent configurations of one block's two copies on a line
    (x, 2x): per configuration the codes of the copies at both points
    (codes[i] = [[code_1(x), code_1(2x)], [code_2(x), code_2(2x)]]) and the
    affine subspace {z0 + Z nu} of u = (c_1, c_2, theta) it allows, theta
    the line's shared parameters (the family's lambda, then the two
    slices' coordinate parameters); t0, T its projection to theta and
    `pinned`, `vals` the pinned theta coordinates, for the hash join."""
    __slots__ = ("codes", "z0", "Z", "t0", "T", "pinned", "vals", "n")

    def __init__(self, codes, z0, Z, nth):
        self.n = len(codes)
        self.codes = np.array(codes, dtype=np.int64).reshape(self.n, 2, 2) if self.n else np.zeros((0, 2, 2), np.int64)
        self.z0 = np.array(z0, dtype=complex).reshape(self.n, -1) if self.n else np.zeros((0, 2 + nth), dtype=complex)
        self.Z = list(Z)
        self.t0 = self.z0[:, 2:]
        self.T = [_orthonormal(Zi[2:]) for Zi in self.Z]
        self.pinned = np.array([np.abs(Ti).sum(axis=1) < 1e-9 if Ti.shape[1] else np.ones(nth, dtype=bool)
                                for Ti in self.T], dtype=bool).reshape(self.n, nth)
        self.vals = np.where(self.pinned, self.t0, 0)

    def subset(self, idx):
        return BlockConfigs([self.codes[i] for i in idx], [self.z0[i] for i in idx], [self.Z[i] for i in idx],
                            self.t0.shape[1])


def _pin_family(c1, c2, d0b, Kb):
    """(z0, Z) over (c_1, c_2, lambda) for pinned coefficients whose sum
    must equal d0b + Kb lambda, or None."""
    kappa = len(Kb)
    if kappa == 0:
        return (np.array([c1, c2]), np.zeros((2, 0), dtype=complex)) if abs(c1 + c2 - d0b) < 1e-7 else None
    sol = _affine_solve_C(np.asarray(Kb, dtype=complex)[None, :], np.array([c1 + c2 - d0b]))
    if sol is None:
        return None
    l0, L = sol
    z0 = np.concatenate([[c1, c2], l0])
    Z = np.zeros((2 + kappa, L.shape[1]), dtype=complex)
    Z[2:] = L
    return z0, Z


_PH_BOTH = [(l1, m1, l2, m2) for l1 in range(3) for m1 in range(3) for l2 in range(3) for m2 in range(3)
            if (l1, m1) <= (l2, m2)]
_INV_BOTH = {ph: np.linalg.inv(np.array([[W3P[ph[0]], W3P[ph[2]]], [W3P[ph[1]], W3P[ph[3]]]]))
             for ph in _PH_BOTH if (ph[0] - ph[2] - ph[1] + ph[3]) % 3}


def block_line_configs(b, Sb, S2b, ax, Nx, a2, N2, d0b, Kb, nth, offx, off2, x, y, dbl):
    """The configurations of block b on the line (x, y = 2x): selection Sb
    at x (S2b = 2 Sb at y), coordinates ax + Nx mu_x and a2 + N2 mu_y
    (Nx, N2 with the columns of their slice's parameters, empty at an
    independent slice), merged coefficient d0b + Kb lambda. Copies are
    labelled by this line: in one class, copy 1 has the smaller (phase at
    x, phase at y) or is the present one; in two classes copy 1 sits in
    the first. With independent coordinates everything is in closed form:
    a copy alone in its class keeps its modulus along the line, so its
    phase at y is its phase at x plus the cube-root shift of the
    coordinate ratio; two copies in one class with distinct phase
    differences at the two points are solved by the 2 x 2 inverse. With a
    coordinate parameter the configurations go through the batched
    solve. Returns a BlockConfigs (possibly empty)."""
    A = b.o.absent
    kappa = len(Kb)
    du = 2 + nth
    codes, z0s, Zs = [], [], []

    def add_cfg(cd, z0, Z):
        full = np.zeros(du, dtype=complex)
        full[:len(z0)] = z0
        Zf = np.zeros((du, Z.shape[1] + du - len(z0)), dtype=complex)
        Zf[:len(z0), :Z.shape[1]] = Z
        Zf[len(z0):, Z.shape[1]:] = np.eye(du - len(z0))
        codes.append(cd)
        z0s.append(full)
        Zs.append(Zf)

    if len(Sb) == 0:
        base = np.zeros((1, 1, du), dtype=complex)
        base[0, 0, :2] = 1
        base[0, 0, 2:2 + kappa] = -np.asarray(Kb)
        sol = _solve_configs(base, np.array([[d0b]], dtype=complex))[0]
        if sol is not None:
            codes.append([[A, A], [A, A]])
            z0s.append(sol[0])
            Zs.append(sol[1])
        return BlockConfigs(codes, z0s, Zs, nth)
    mx, m2 = Nx.shape[1], N2.shape[1]
    dependent = mx > 0 or m2 > 0
    if len(Sb) == 1:
        k, k2 = Sb[0], S2b[0]
        if not dependent:
            rho = _is_root_shift(a2[0] / ax[0])
            rhs = np.array([ax[0], a2[0]])
            for ph in _PH_BOTH:
                l1, m1, l2, m2_ = ph
                if ph in _INV_BOTH:
                    c1, c2 = _INV_BOTH[ph] @ rhs
                    if abs(c1) < 1e-9 or abs(c2) < 1e-9:
                        continue
                    sol = _pin_family(c1, c2, d0b, Kb)
                elif rho is not None and (m1 - l1) % 3 == rho:
                    rows = np.zeros((2, 2 + kappa), dtype=complex)
                    rows[0, 0], rows[0, 1] = W3P[l1], W3P[l2]
                    rows[1, 0] = rows[1, 1] = 1
                    rows[1, 2:] = -np.asarray(Kb)
                    sol = _solve_configs(rows[None], np.array([[ax[0], d0b]]))[0]
                else:
                    continue
                if sol is not None:
                    add_cfg([[3 * k + l1, 3 * k2 + m1], [3 * k + l2, 3 * k2 + m2_]], *sol)
            if rho is not None:
                for l in range(3):
                    rows = np.zeros((2, 2 + kappa), dtype=complex)
                    rows[0, 0] = 1
                    rows[1, 0] = rows[1, 1] = 1
                    rows[1, 2:] = -np.asarray(Kb)
                    sol = _solve_configs(rows[None], np.array([[ax[0] / W3P[l], d0b]]))[0]
                    if sol is not None:
                        add_cfg([[3 * k + l, 3 * k2 + (l + rho) % 3], [A, A]], *sol)
            return BlockConfigs(codes, z0s, Zs, nth)
        TH = np.zeros((3, nth), dtype=complex)
        TH[0, offx:offx + mx] = -Nx[0]
        TH[1, off2:off2 + m2] = -N2[0]
        TH[2, :kappa] = -np.asarray(Kb)
        rhs = np.array([ax[0], a2[0], d0b], dtype=complex)
        PH = np.array([[[W3P[l1], W3P[l2]], [W3P[m1], W3P[m2_]], [1, 1]] for l1, m1, l2, m2_ in _PH_BOTH]
                      + [[[W3P[l], 0], [W3P[m], 0], [1, 1]] for l in range(3) for m in range(3)], dtype=complex)
        cds = [[[3 * k + l1, 3 * k2 + m1], [3 * k + l2, 3 * k2 + m2_]] for l1, m1, l2, m2_ in _PH_BOTH] \
            + [[[3 * k + l, 3 * k2 + m], [A, A]] for l in range(3) for m in range(3)]
        Amat = np.concatenate([PH, np.broadcast_to(TH, (len(PH),) + TH.shape)], axis=2)
        for cd, sol in zip(cds, _solve_configs(Amat, np.broadcast_to(rhs, (len(PH), 3)))):
            if sol is not None:
                codes.append(cd)
                z0s.append(sol[0])
                Zs.append(sol[1])
        return BlockConfigs(codes, z0s, Zs, nth)
    k, kp = Sb
    p, pp = S2b.index(dbl[k]), S2b.index(dbl[kp])
    if not dependent:
        r1, r2 = _is_root_shift(a2[p] / ax[0]), _is_root_shift(a2[pp] / ax[1])
        if r1 is None or r2 is None:
            return BlockConfigs(codes, z0s, Zs, nth)
        for l1 in range(3):
            for l2 in range(3):
                sol = _pin_family(ax[0] / W3P[l1], ax[1] / W3P[l2], d0b, Kb)
                if sol is not None:
                    add_cfg([[3 * k + l1, 3 * dbl[k] + (l1 + r1) % 3],
                             [3 * kp + l2, 3 * dbl[kp] + (l2 + r2) % 3]], *sol)
        return BlockConfigs(codes, z0s, Zs, nth)
    TH = np.zeros((5, nth), dtype=complex)
    TH[0, offx:offx + mx] = -Nx[0]
    TH[1, off2:off2 + m2] = -N2[p]
    TH[2, offx:offx + mx] = -Nx[1]
    TH[3, off2:off2 + m2] = -N2[pp]
    TH[4, :kappa] = -np.asarray(Kb)
    rhs = np.array([ax[0], a2[p], ax[1], a2[pp], d0b], dtype=complex)
    phs = [(l1, m1, l2, m2_) for l1 in range(3) for m1 in range(3) for l2 in range(3) for m2_ in range(3)]
    PH = np.array([[[W3P[l1], 0], [W3P[m1], 0], [0, W3P[l2]], [0, W3P[m2_]], [1, 1]] for l1, m1, l2, m2_ in phs],
                  dtype=complex)
    Amat = np.concatenate([PH, np.broadcast_to(TH, (len(PH),) + TH.shape)], axis=2)
    for (l1, m1, l2, m2_), sol in zip(phs, _solve_configs(Amat, np.broadcast_to(rhs, (len(PH), 5)))):
        if sol is not None:
            codes.append([[3 * k + l1, 3 * dbl[k] + m1], [3 * kp + l2, 3 * dbl[kp] + m2_]])
            z0s.append(sol[0])
            Zs.append(sol[1])
    return BlockConfigs(codes, z0s, Zs, nth)


def _compatible(t0, T, cfgs):
    """Indices of the configurations whose pinned theta coordinates do not
    conflict with those of the partial subspace (t0, T)."""
    if cfgs.n == 0:
        return np.zeros(0, dtype=np.int64)
    P = np.abs(T).sum(axis=1) < 1e-9 if T.shape[1] else np.ones(len(t0), dtype=bool)
    if not P.any():
        return np.arange(cfgs.n)
    conflict = (cfgs.pinned & P[None, :] & (np.abs(cfgs.vals - t0[None, :]) > 1e-6)).any(axis=1)
    return np.flatnonzero(~conflict)


def _theta_combos(per_block, nth, cap=None):
    """Every choice of one configuration per block whose theta subspaces
    meet, as (indices per block, t0, T); the blocks are visited in order
    of their configuration count and the pinned coordinates are hashed
    against before any meet. Raises BudgetExceeded past `cap` combos."""
    order = sorted(range(len(per_block)), key=lambda i: per_block[i].n)
    out = []

    def dfs(d, t0, T, choice):
        if d == len(order):
            out.append((choice, t0, T))
            if cap is not None and len(out) > cap:
                raise BudgetExceeded(f"{len(out)} configuration combinations on one line exceed the cap {cap}")
            return
        cfgs = per_block[order[d]]
        full = T.shape[1] == nth
        for j in _compatible(t0, T, cfgs):
            met = (cfgs.t0[j], cfgs.T[j]) if full else _meet_C(t0, T, cfgs.t0[j], cfgs.T[j])
            if met is not None:
                dfs(d + 1, met[0], met[1], choice + [(order[d], int(j))])

    dfs(0, np.zeros(nth, dtype=complex), np.eye(nth, dtype=complex), [])
    return [([dict(c)[i] for i in range(len(per_block))], t0, T) for c, t0, T in out]


def _small_selections(sub, r, tol=1e-6):
    """Independent selections (one subset of classes per block of `sub`)
    whose span contains r with every coordinate nonzero, with their
    coordinates in block order, by direct enumeration of the products of
    subsets (for the few blocks of a fourth-line completion)."""
    dim = sub[0].VC.shape[0]
    scale = max(1.0, float(np.linalg.norm(r)))
    out = []
    if np.linalg.norm(r) < tol * scale:
        return [(tuple(() for _ in sub), np.zeros(0, dtype=complex))]
    for parts in itertools.product(*[b.subsets for b in sub]):
        cols = [b.VC[:, list(S)] for b, S in zip(sub, parts) if len(S)]
        if not cols:
            continue
        V = np.column_stack(cols)
        if V.shape[1] > dim:
            continue
        sol = _affine_solve_C(V, r)
        if sol is None or sol[1].shape[1] or not np.all(np.abs(sol[0]) > 1e-9):
            continue
        out.append((tuple(tuple(S) for S in parts), sol[0]))
    return out


class LineSelection:
    """An independent paired selection on a line (S at x, S2 = 2S at 2x,
    the unique coordinates of both points) with, per block, the
    configurations that take part in a lambda-consistent combination over
    the blocks, and per block the lambda values those configurations pin
    (`keys`, rounded) and whether one of them leaves lambda free
    (`free`), for the join's prefilter."""
    __slots__ = ("S", "S2", "ax", "a2", "cfgs", "keys", "free")

    def __init__(self, S, S2, ax, a2, cfgs):
        self.S, self.S2, self.ax, self.a2, self.cfgs = S, S2, ax, a2, cfgs
        self.keys, self.free = [], []
        for c in cfgs:
            full = c.pinned.all(axis=1)
            r = np.round(c.vals, 6) + 0.0
            self.keys.append({tuple(zip(r[i].real.tolist(), r[i].imag.tolist())) for i in np.flatnonzero(full)})
            self.free.append(bool((~full).any()))


def _feasible_lambda(sels):
    """Whether some lambda is consistent with every block on every line of
    `sels` (a necessary condition the triple stage decides exactly): per
    block the lambda values pinned by one line must be matched by the other
    lines unless a configuration there leaves lambda free. Returns the
    per-block feasible sets (None for unconstrained) or None."""
    out = []
    for b in range(len(sels[0].keys)):
        feas = None                                        # unconstrained so far
        for s in sels:
            if s.free[b]:
                continue                                   # this line accepts any lambda on block b
            feas = set(s.keys[b]) if feas is None else feas & s.keys[b]
            if not feas:
                return None
        out.append(feas)
    common = None
    for feas in out:
        if feas is None:
            continue
        common = set(feas) if common is None else common & feas
        if not common:
            return None
    return out


def line_completion(o, missing):
    """For a copy with base slice u: from its codes at the six offsets of
    the three lines other than `missing` (LINES order, first then second
    point of each) to the set of its code pairs at the two points of the
    missing line, over every shape of the structure lemma."""
    A = o.absent
    known = [z for i, (x, y) in enumerate(LINES) if i != missing for z in (x, y)]
    mx, my = LINES[missing]
    table = {}
    for c1 in range(A + 1):
        for c2 in range(A + 1):
            for row in o.composite_rows(c1, c2):
                full = {x: int(v) for x, v in zip(COMP, row)}
                full[E1], full[E2] = c1, c2
                table.setdefault(tuple(full[z] for z in known), set()).add((full[mx], full[my]))
    return table


class BlockOnlyMatcher:
    """The block-only run for one base (the note at the head of this
    section): `run()` returns the raw hits; the counts land in `stats`.

    Every decomposition whose translate selection is independent (the
    chosen translates linearly independent, so their coordinates are
    unique) on at least three of the four lines through x0 is found: for
    each choice of three lines, the two with the fewest surviving
    selections drive a join, the third is looked up through the class
    relation of the structure lemma, and the fourth line is completed from
    the three (a copy present on two or more lines is a plane term whose
    codes there follow, a copy present on one line is a line term absent
    there, a copy absent on all three is a point or a line along the
    fourth direction and is read off the residual of the fourth line's two
    slices once every other copy's coefficient is pinned). A dependent
    selection (its coordinates form a family) is never searched: on its
    own line it constrains almost nothing, so a decomposition dependent on
    two or more lines is outside this matcher; stats["dependent_lines_limit"]
    records the limit and stats["slice_dependent"] how many selections
    each slice dropped."""

    def __init__(self, matcher, blocks, x0, target, stats, budget=None, log=None, hits=None):
        self.matcher, self.blocks, self.x0, self.target = matcher, blocks, x0, target
        self.stats, self.budget, self.log = stats, budget, (log or (lambda m: None))
        self.rng = matcher.rng
        self.dim = blocks[0].VC.shape[0]
        self.hits = [] if hits is None else hits      # filled as the passes run, so an abort keeps them
        self._completion = {}
        self._confirmed = {}                    # code key -> confirmed hit, once across the passes
        if any(b.g != 2 for b in blocks):
            raise AssertionError("the block-only matcher handles blocks of two copies")

    def _check(self, where, **kw):
        if self.budget is not None:
            self.budget.check(where, **kw)

    def completion(self, bi, missing):
        if (bi, missing) not in self._completion:
            self._completion[(bi, missing)] = line_completion(self.blocks[bi].o, missing)
        return self._completion[(bi, missing)]

    def run(self):
        st, blocks, x0 = self.stats, self.blocks, self.x0
        st.update(slice_solutions={}, slice_dependent={}, line_pairs={}, line_alive={}, triples={}, tuples=0,
                  block_leaves=0, combinations=0, completions=0, dependent_lines_limit=1)
        VC0 = np.column_stack([b.VC[:, 0] for b in blocks])
        sol0 = _affine_solve_C(VC0, self.target.rhs(x0)[2])
        if sol0 is None:
            st["refused"] = True
            return []
        self.d0, self.K = sol0
        self.kappa = self.K.shape[1]
        sols, seen = {}, {}
        for x in OFFSETS:
            t0 = time.time()
            rhs = self.target.rhs(add(x0, x))
            T = rhs[2]
            j = np.flatnonzero(np.abs(T) > 1e-9)
            key = None if len(j) == 0 else (np.round(T / T[j[0]], 8) + 0.0).tobytes()
            if key in seen:
                y = seen[key]
                s = T[j[0]] / self.target.rhs(add(x0, y))[2][j[0]]
                sols[x] = [(S, a0 * s, N) for S, a0, N in sols[y]]
                note = f"proportional to slice {y}"
            else:
                sols[x] = block_only_slice(blocks, rhs, self.rng, self.budget, f"slice {x}", st)
                seen[key] = x
                note = f"{sum(N.shape[1] > 0 for _, _, N in sols[x])} dependent"
            st["slice_solutions"][str(x)] = len(sols[x])
            st["slice_dependent"][str(x)] = sum(N.shape[1] > 0 for _, _, N in sols[x])
            st["seconds"][f"solve_{x[0]}{x[1]}"] = round(time.time() - t0, 2)
            self.log(f"  slice {x}: {len(sols[x])} selections ({note}), {time.time() - t0:.1f}s, "
                     f"rss {peak_rss_gb():.2f} GB")
            if not sols[x]:
                return []
        st["coord_raw"] = [len(sols[E1]), len(sols[E2])]
        pats = [copy_patterns(b.o) for b in blocks]
        self.dbls = [d for _, d in pats]
        self.pats = [p for p, _ in pats]
        lines = {}
        for li, (x, y) in enumerate(LINES):
            t0 = time.time()
            index = {S: a0 for S, a0, N in sols[y] if N.shape[1] == 0}
            paired, alive = 0, []
            for i, (S, a0, N) in enumerate(sols[x]):
                if i % 200 == 199:
                    self._check(f"line {x}")
                if N.shape[1]:
                    continue
                S2 = tuple(tuple(sorted(self.dbls[b][k] for k in Sb)) for b, Sb in enumerate(S))
                if S2 not in index:
                    continue
                paired += 1
                sel = self._line_selection(x, y, S, S2, a0, index[S2])
                if sel is not None:
                    alive.append(sel)
            lines[li] = alive
            st["line_pairs"][str(x)] = paired
            st["line_alive"][str(x)] = len(alive)
            st["seconds"][f"line_{x[0]}{x[1]}"] = round(time.time() - t0, 2)
            self.log(f"  line {x}, {y}: {paired} independent paired selections of {len(sols[x])}, {len(alive)} "
                     f"with a lambda-consistent configuration, {time.time() - t0:.1f}s, rss {peak_rss_gb():.2f} GB")
        del sols
        st["coord_solutions"] = [st["line_pairs"][str(E1)], st["line_pairs"][str(E2)]]
        st["coord_states"] = [len(lines[0]), len(lines[1])]
        if sum(bool(lines[li]) for li in range(4)) < 3:
            return []
        index = [{} for _ in LINES]
        for li in range(4):
            for sel in lines[li]:
                index[li].setdefault(sel.S, []).append(sel)
        hits, t0, done = self.hits, time.time(), set()
        for missing in range(4):
            known = [li for li in range(4) if li != missing]
            if not all(lines[li] for li in known):
                continue
            drive = tuple(sorted(sorted(known, key=lambda li: len(lines[li]))[:2]))
            third = [li for li in known if li not in drive][0]
            other = [i for i in range(4) if i not in drive]
            pos = other.index(third)
            rels = [block_relation(self.pats[b], blocks[b].g, drive) for b in range(len(blocks))]
            A, Bs = lines[drive[0]], lines[drive[1]]
            posting = [{} for _ in blocks]
            for n_, sel in enumerate(lines[third]):
                for i in range(len(blocks)):
                    posting[i].setdefault(sel.S[i], []).append(n_)
            pairs, new, feasible = 0, 0, 0
            for sa in A:
                for sb in Bs:
                    pairs += 1
                    if pairs % 5000 == 0:
                        self._check("line join", states=st["tuples"])
                    if _feasible_lambda((sa, sb)) is None:
                        continue
                    opts = [rels[i].get((sa.S[i], sb.S[i])) for i in range(len(blocks))]
                    if any(o is None for o in opts):
                        continue
                    feasible += 1
                    cands = [sorted({o[pos] for o in opt}) for opt in opts]
                    for sc in self._lookup(index[third], cands, lines[third], posting):
                        if _feasible_lambda((sa, sb, sc)) is None:
                            continue
                        sels = {drive[0]: sa, drive[1]: sb, third: sc}
                        key = (missing,) + tuple(sels[li].S for li in known)
                        if key in done:
                            continue
                        done.add(key)
                        st["tuples"] += 1
                        new += 1
                        hits.extend(self._triple_hits(sels, missing))
                        if st["tuples"] % 50 == 0:
                            self._check("triples", states=st["tuples"])
                            self.log(f"  lines {known}: {pairs} pairs, {st['tuples']} triples, {len(hits)} raw "
                                     f"hits, {time.time() - t0:.1f}s, rss {peak_rss_gb():.2f} GB")
            st["triples"][str(known)] = {"drive": list(drive), "pairs": pairs, "feasible_pairs": feasible,
                                         "triples": new}
            self.log(f"  lines {[LINES[li][0] for li in known]} (driving {[LINES[li][0] for li in drive]}): "
                     f"{pairs} pairs, {feasible} lambda-feasible, {new} new triples, {time.time() - t0:.1f}s")
        st["pairs"] = sum(v["pairs"] for v in st["triples"].values())
        st["joined"] = st["tuples"]
        st["seconds"]["join"] = round(time.time() - t0, 2)
        self.log(f"  join: {st['tuples']} triples of lines, {st['block_leaves']} block leaves, "
                 f"{st['combinations']} combinations, {st['completions']} completions confirmed, "
                 f"{len(hits)} raw hits, {time.time() - t0:.1f}s")
        return hits

    def _line_selection(self, x, y, S, S2, ax, a2):
        """The LineSelection of an independent paired selection, or None
        when no choice of one configuration per block is lambda-consistent."""
        kappa = self.kappa
        per_block, pos = [], 0
        for i, b in enumerate(self.blocks):
            n = len(S[i])
            none = np.zeros((n, 0), dtype=complex)
            cfgs = block_line_configs(b, S[i], S2[i], ax[pos:pos + n], none, a2[pos:pos + n], none, self.d0[i],
                                      self.K[i], kappa, kappa, kappa, x, y, self.dbls[i])
            pos += n
            if cfgs.n == 0:
                return None
            per_block.append(cfgs)
        combos = _theta_combos(per_block, kappa)
        if not combos:
            return None
        used = [sorted({c[i] for c, _, _ in combos}) for i in range(len(self.blocks))]
        return LineSelection(S, S2, ax, a2, [per_block[i].subset(used[i]) for i in range(len(used))])

    @staticmethod
    def _lookup(index, per_block, entries, posting, cap=64):
        """The line selections whose classes are, per block, one of the
        given sets: the product of the per-block sets looked up when small,
        else the intersection over the blocks of the posting lists of the
        allowed class sets."""
        if any(not c for c in per_block):
            return []
        if math.prod(len(c) for c in per_block) <= cap:
            out = []
            for key in itertools.product(*per_block):
                out.extend(index.get(key, ()))
            return out
        ids = None
        for i, c in enumerate(per_block):
            here = set()
            for S in c:
                here.update(posting[i].get(S, ()))
            ids = here if ids is None else ids & here
            if not ids:
                return []
        return [entries[n_] for n_ in sorted(ids)]

    def _triple_hits(self, sels, missing):
        """Hits of three line selections (dict line index -> LineSelection)
        with the fourth line `missing` completed: per block the choices of
        one configuration per line consistent in (c_1, c_2, lambda), the
        blocks joined on lambda, the missing line completed, every
        candidate confirmed."""
        blocks, st, kappa = self.blocks, self.stats, self.kappa
        known = [li for li in range(4) if li != missing]
        du = 2 + kappa
        per_block = []
        for bi, b in enumerate(blocks):
            leaves_codes, leaves_z0, leaves_Z = [], [], []

            def dfs(d, z0, Z, codes, fixed):
                if d == len(known):
                    if np.any((np.abs(z0[:2]) < 1e-9) & (np.abs(Z[:2]).sum(axis=1) < 1e-9)):
                        return                               # a copy no member of the subspace uses
                    leaves_codes.append([dict(codes[0]), dict(codes[1])])
                    leaves_z0.append(z0)
                    leaves_Z.append(Z)
                    return
                li = known[d]
                x, y = LINES[li]
                cfgs = sels[li].cfgs[bi]
                present = len(sels[li].S[bi]) > 0
                distinct = codes[0] != codes[1]          # a swap is a different assignment only then
                for j in range(cfgs.n):
                    for swap in ((False, True) if fixed and present and distinct else (False,)):
                        z0c, Zc, cd = cfgs.z0[j], cfgs.Z[j], cfgs.codes[j]
                        if swap:
                            perm = np.arange(du)
                            perm[0], perm[1] = 1, 0
                            z0c, Zc, cd = z0c[perm], Zc[perm], cd[::-1]
                        met = _meet_C(z0, Z, z0c, Zc)
                        if met is None:
                            continue
                        new = [dict(codes[0]), dict(codes[1])]
                        for cp in range(2):
                            new[cp][x], new[cp][y] = int(cd[cp][0]), int(cd[cp][1])
                        dfs(d + 1, met[0], met[1], new, fixed or present)

            dfs(0, np.zeros(du, dtype=complex), np.eye(du, dtype=complex), [{}, {}], False)
            st["block_leaves"] += len(leaves_codes)
            if not leaves_codes:
                return []
            bc = BlockConfigs(np.zeros((len(leaves_codes), 2, 2), dtype=np.int64), leaves_z0, leaves_Z, kappa)
            bc.codes = leaves_codes
            per_block.append(bc)
        hits = []
        for choice, _, _ in _theta_combos(per_block, kappa):
            st["combinations"] += 1
            for terms in self._complete_line(choice, per_block, missing):
                key = tuple(sorted(exact_codes(t)[0].tobytes() for t in terms))
                if key not in self._confirmed:           # the same copies in another labelling or pass
                    st["reconstructions"] += 1
                    self._confirmed[key] = self.matcher.confirm(terms, 0, self.target)
                    hits.append(self._confirmed[key])
        return hits

    @staticmethod
    def _joint_coefficients(choice, per_block):
        """The coefficients of every copy over the combination's joint
        subspace: c = c0 + cW w for the free parameters w that remain once
        the blocks' lambdas agree (none when everything is pinned), or None
        when the blocks' subspaces do not meet."""
        leaves = [(pb.z0[j], pb.Z[j]) for pb, j in zip(per_block, choice)]
        dims = [Z.shape[1] for _, Z in leaves]
        n = sum(dims)
        rows, rhs = [], []
        off = np.cumsum([0] + dims)
        for b in range(1, len(leaves)):
            z0a, Za = leaves[b - 1]
            z0b, Zb = leaves[b]
            R = np.zeros((Za.shape[0] - 2, n), dtype=complex)
            R[:, off[b - 1]:off[b]] = Za[2:]
            R[:, off[b]:off[b + 1]] = -Zb[2:]
            rows.append(R)
            rhs.append(z0b[2:] - z0a[2:])
        if rows:
            sol = _affine_solve_C(np.concatenate(rows), np.concatenate(rhs))
            if sol is None:
                return None
            w0, W = sol
        else:
            w0, W = np.zeros(n, dtype=complex), np.eye(n, dtype=complex)
        c0 = np.concatenate([z0[:2] + Z[:2] @ w0[off[b]:off[b + 1]] for b, (z0, Z) in enumerate(leaves)])
        cW = np.concatenate([Z[:2] @ W[off[b]:off[b + 1]] for b, (z0, Z) in enumerate(leaves)])
        return c0, cW

    def _complete_line(self, choice, per_block, missing):
        """The term vectors of the candidates of one combination with the
        missing line filled in:
        copies present on two or more known lines take their codes there
        from the completion table, copies present on one known line are
        absent there, copies absent on all three are read off the residual
        of the missing line's two slices by a pinned-coefficient solve over
        their 28 options at the first point (the class doubled and the
        phase free at the second), the remaining free parameters of the
        coefficients solved along."""
        blocks, st = self.blocks, self.stats
        known = [li for li in range(4) if li != missing]
        mx, my = LINES[missing]
        known_offsets = [z for li in known for z in LINES[li]]
        jc = self._joint_coefficients(choice, per_block)
        if jc is None:
            return []
        c0, cW = jc
        nw = cW.shape[1]
        fixed, floating = [], []
        for bi, (b, j) in enumerate(zip(blocks, choice)):
            for cp in range(2):
                cd = per_block[bi].codes[j][cp]
                npresent = sum(cd[LINES[li][0]] != b.o.absent for li in known)
                if npresent >= 2:
                    opts = self.completion(bi, missing).get(tuple(cd[z] for z in known_offsets))
                    if not opts:
                        return []
                    fixed.append((bi, cp, sorted(opts)))
                elif npresent == 1:
                    fixed.append((bi, cp, [(b.o.absent, b.o.absent)]))
                else:
                    floating.append((bi, cp))
        if 28 ** len(floating) > 200_000:
            raise BudgetExceeded(f"{len(floating)} copies absent on three lines: {28 ** len(floating)} "
                                 f"completions of the fourth line")
        Tx, Ty = self.target.rhs(add(self.x0, mx))[2], self.target.rhs(add(self.x0, my))[2]
        hits = []

        def consistent(pairs):
            """pairs: (bi, cp, code_x, code_y) for every copy. The residual
            of both slices must vanish for some w."""
            bx, by = Tx.copy(), Ty.copy()
            Mx = np.zeros((len(Tx), nw), dtype=complex)
            My = np.zeros((len(Ty), nw), dtype=complex)
            for bi, cp, cx, cy in pairs:
                i = 2 * bi + cp
                vx, vy = blocks[bi].o.vecs[cx], blocks[bi].o.vecs[cy]
                bx -= c0[i] * vx
                by -= c0[i] * vy
                Mx += np.outer(vx, cW[i])
                My += np.outer(vy, cW[i])
            if nw == 0:
                return np.abs(bx).max() < 1e-6 and np.abs(by).max() < 1e-6
            return _affine_solve_C(np.concatenate([Mx, My]), np.concatenate([bx, by])) is not None

        for combo in itertools.product(*[opts for _, _, opts in fixed]):
            self._check("completion", states=st["tuples"])
            base = [(bi, cp, cx, cy) for (bi, cp, _), (cx, cy) in zip(fixed, combo)]
            if not floating:
                if consistent(base):
                    st["completions"] += 1
                    hits.append(self._confirm_codes(choice, per_block, base, missing))
                continue
            if nw == 0:
                rx, ry = Tx.copy(), Ty.copy()
                for bi, cp, cx, cy in base:
                    rx -= c0[2 * bi + cp] * blocks[bi].o.vecs[cx]
                    ry -= c0[2 * bi + cp] * blocks[bi].o.vecs[cy]
                for extra in self._floating_from_residual(floating, c0, rx, ry):
                    st["completions"] += 1
                    hits.append(self._confirm_codes(choice, per_block, base + extra, missing))
                continue
            for codes_x in itertools.product(*[range(blocks[bi].o.absent + 1) for bi, _ in floating]):
                second = []
                for (bi, _), cx in zip(floating, codes_x):
                    if cx == blocks[bi].o.absent:
                        second.append([blocks[bi].o.absent])
                    else:
                        second.append([3 * self.dbls[bi][cx // 3] + m for m in range(3)])
                for codes_y in itertools.product(*second):
                    pairs = base + [(bi, cp, cx, cy) for (bi, cp), cx, cy in zip(floating, codes_x, codes_y)]
                    if consistent(pairs):
                        st["completions"] += 1
                        hits.append(self._confirm_codes(choice, per_block, pairs, missing))
        return [h for h in hits if h is not None]

    def _floating_from_residual(self, floating, c0, rx, ry):
        """The code pairs of the copies absent on the three known lines
        (`floating`: (block, copy) with pinned coefficients c0) at the two
        points of the missing line, from the residual rx, ry of those slices
        after every other copy. The residual must be a combination of at
        most one translate per floating copy: over the blocks involved the
        selections whose span contains rx are enumerated directly (their
        products of subsets, the sub-problem is small), paired with the
        doubled selection at the second point, and per block the copies
        are assigned to the classes with phases read off the coordinates
        (a copy alone in a class has coordinate c w^l; two copies in one
        class are solved over the nine phase pairs). A dependent
        sub-selection has no unique coordinates and is not searched."""
        blocks = self.blocks
        by_block = {}
        for bi, cp in floating:
            by_block.setdefault(bi, []).append(cp)
        order = sorted(by_block)
        sub = [Block(blocks[bi].o, len(by_block[bi]), i) for i, bi in enumerate(order)]
        if math.prod(len(b.subsets) for b in sub) > 200_000:
            raise BudgetExceeded(f"{len(floating)} copies absent on three lines over {len(sub)} blocks: "
                                 f"{math.prod(len(b.subsets) for b in sub)} selections of the fourth line")
        selx = _small_selections(sub, rx)
        sely = {S: a for S, a in _small_selections(sub, ry)}
        out = []
        for S, ax in selx:
            S2 = tuple(tuple(sorted(self.dbls[bi][k] for k in Sb)) for bi, Sb in zip(order, S))
            if S2 not in sely:
                continue
            ay = sely[S2]
            per_block, pos = [], 0
            for i, bi in enumerate(order):
                cps = by_block[bi]
                Sb, n = S[i], len(S[i])
                alpha, beta = ax[pos:pos + n], ay[pos:pos + n]
                pos += n
                A = blocks[bi].o.absent
                opts = []
                if n == 0:
                    opts.append([(bi, cp, A, A) for cp in cps])
                elif n == 1:
                    k, k2 = Sb[0], S2[i][0]
                    for cp in cps:                                   # one copy present
                        l, m = _is_root_shift(alpha[0] / c0[2 * bi + cp]), _is_root_shift(beta[0] / c0[2 * bi + cp])
                        if l is not None and m is not None:
                            opts.append([(bi, cp, 3 * k + l, 3 * k2 + m)] + [(bi, o, A, A) for o in cps if o != cp])
                    if len(cps) == 2:                                # both present in the class
                        c1, c2 = c0[2 * bi + cps[0]], c0[2 * bi + cps[1]]
                        for l1, l2, m1, m2 in itertools.product(range(3), repeat=4):
                            if (abs(c1 * W3P[l1] + c2 * W3P[l2] - alpha[0]) < 1e-6
                                    and abs(c1 * W3P[m1] + c2 * W3P[m2] - beta[0]) < 1e-6):
                                opts.append([(bi, cps[0], 3 * k + l1, 3 * k2 + m1),
                                             (bi, cps[1], 3 * k + l2, 3 * k2 + m2)])
                else:
                    for perm in ((0, 1), (1, 0)):
                        assign = []
                        for cp, q in zip(cps, perm):
                            k, k2 = Sb[q], self.dbls[bi][Sb[q]]
                            l = _is_root_shift(alpha[q] / c0[2 * bi + cp])
                            m = _is_root_shift(beta[S2[i].index(k2)] / c0[2 * bi + cp])
                            if l is None or m is None:
                                break
                            assign.append((bi, cp, 3 * k + l, 3 * k2 + m))
                        else:
                            opts.append(assign)
                if not opts:
                    break
                per_block.append(opts)
            else:
                for choice in itertools.product(*per_block):
                    out.append([t for part in choice for t in part])
        return out

    def _confirm_codes(self, choice, per_block, pairs, missing):
        """The term vectors of a completed candidate, or None when a copy's
        codes are not a stabilizer state's or the two copies of a block
        coincide."""
        blocks = self.blocks
        mx, my = LINES[missing]
        codes = [[dict(per_block[bi].codes[j][cp]) for cp in range(2)] for bi, j in enumerate(choice)]
        for bi, cp, cx, cy in pairs:
            codes[bi][cp][mx], codes[bi][cp][my] = int(cx), int(cy)
        terms = []
        for bi, b in enumerate(blocks):
            if codes[bi][0] == codes[bi][1]:
                return None
            for cp in range(2):
                cd = codes[bi][cp]
                if not b.o.valid_codes(cd):
                    return None
                t = np.zeros((9, self.dim), dtype=complex)
                t[pidx(self.x0)] = b.o.u
                for x, code in cd.items():
                    t[pidx(add(self.x0, x))] = b.o.vecs[code]
                terms.append(t.ravel())
        return terms


def solve_slice3(opts, blocks, fam, rhs, rng, stats=None, log=None, budget=None, where="", proj=None,
                 max_cand=2_000_000):
    """slice_cover.solve_slice with the same hashing (meet in the middle on a
    random functional mod P1 for a pinned family, the Laplace-feature dense
    solve when parameters remain), returning (combo, Ssel, family) triples
    with the family already restricted by the slice equation, so the caller
    does not decide each solution a second time. Differences from
    slice_cover: the blocks' projectors are cached per translate-set
    selection (`proj`), the budget is checked before every selection and
    after every candidate list, a dense solve whose feature matrices would
    exceed the budget is refused before it allocates, more than `max_cand`
    hash candidates raises BudgetExceeded instead of AssertionError. A slice
    with no ordinary term belongs to BlockOnlyMatcher."""
    if proj is None:
        proj = _ProjectorCache(blocks)
    if not opts:
        raise AssertionError("a slice equation with no ordinary term is solved by BlockOnlyMatcher")
    r = len(opts)
    d1, K1 = fam.parts[0]
    ords = [i for i in range(len(d1)) if i not in {b.pos for b in blocks}]
    assert len(ords) == r
    d0v = d1[ords]
    Kv = _independent_columns_mod(K1[ords].reshape(r, -1), P1)
    nsel = math.prod(len(b.subsets) for b in blocks)
    if budget is not None:
        budget.check_dense(where, [len(o[0]) for o in opts], Kv.shape[1], nsel)
    out = []
    for Ssel in itertools.product(*[b.subsets for b in blocks]):
        if budget is not None:
            budget.check(where, solutions=len(out))
        Ps = None
        if blocks:
            Ps, _ = proj(Ssel)
            popts = [(o[0] @ Ps[0].T) % P1 for o in opts]
            prhs = (Ps[0] @ rhs[0]) % P1
        else:
            popts = [o[0] for o in opts]
            prhs = rhs[0]
        sizes = [len(o) for o in popts]
        sides = _split_sides(sizes)
        t0 = time.time()
        if Kv.shape[1] == 0:
            cands = _mitm(popts, d0v, prhs, sides, rng)
        else:
            try:
                cands = _dense(popts, d0v, Kv, prhs, sides, rng, max_cand)
            except AssertionError as e:
                raise BudgetExceeded(f"{where}: {e}") from None
        if stats is not None:
            stats["candidates"] = stats.get("candidates", 0) + len(cands)
        if log is not None and (Kv.shape[1] or len(cands) > 100_000):
            log(f"    translate sets {Ssel}: {Kv.shape[1]} parameters, sizes {sizes}, "
                f"{len(cands)} candidates [{time.time() - t0:.1f}s]")
        if cands and Kv.shape[1] == 0:
            Cm = np.array(cands, dtype=np.int64)
            acc = np.zeros((len(Cm), prhs.shape[0]), dtype=np.int64)
            for i in range(r):
                acc = (acc + (int(d0v[i]) * popts[i][Cm[:, i]]) % P1) % P1
            keep = np.all(acc == (prhs % P1)[None, :], axis=1)
            cands = [c for c, k in zip(cands, keep) if k]
        for combo in cands:
            f2 = restrict(fam, *slice_system(opts, blocks, combo, Ssel, rhs, Ps))
            if f2 is not None:
                out.append((combo, Ssel, f2))
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
    of (offset, {class: coordinate}) over the eight non-base offsets in
    OFFSETS order (the two coordinate offsets first), with the block's
    contribution at that offset sum_k a_k Q_k u over the translate set the
    slice equation chose. Returns every assignment (a list of (coefficient,
    codes) per copy, copies distinct and unordered) with coefficients
    summing to D, all nonzero, and each copy a valid stabilizer state; the
    flag says whether a coefficient family remained (degenerate).

    When the state's coefficient family d = d0 + K lambda still has kappa
    parameters, D is the pair (D0, DK) with the merged coefficient
    D0 + DK lambda and every coordinate is the pair (a0, A) with
    a0 + A lambda; lambda then enters the solve as kappa further unknowns
    next to the g copy coefficients, so the copies are decided on the whole
    family at once (the reconstruction at a random member, which the review
    found, is gone), and a lambda left free at the end is a degenerate
    result like a free copy coefficient. A scalar D and scalar coordinates
    are the kappa = 0 case.

    The translate set records the classes with a nonzero net coordinate
    only. Two or more copies can sit in one further class with phases and
    coefficients that cancel there (c_1 w^{l_1} + c_2 w^{l_2} = 0), which the
    slice equation cannot see, so at every offset the copies may also use
    classes outside the set, each by at least two copies with net coordinate
    zero, as long as the classes used number at most g. Once a copy's two
    coordinate codes are fixed its composite codes are restricted to the
    shapes of the structure lemma for them (composite_rows), which keeps
    the extra freedom small: a plane copy has one class per composite point
    and at most three phases."""
    if isinstance(D, tuple):
        D0, DK = D[0], np.asarray(D[1], dtype=complex).reshape(-1)
    else:
        D0, DK = D, np.zeros(0, dtype=complex)
    kappa = len(DK)

    def affine(v):
        """(constant, row over lambda) of a coordinate given as a scalar or a pair."""
        if isinstance(v, tuple):
            return complex(v[0]), np.asarray(v[1], dtype=complex).reshape(-1)
        return complex(v), np.zeros(kappa, dtype=complex)

    # the unknowns z = (c_1, ..., c_g, lambda); sum_j c_j - DK lambda = D0
    row = np.concatenate([np.ones(g, dtype=complex), -DK])[None, :]
    c0, Kc = _affine_solve_C(row, np.array([D0], dtype=complex))
    if [x for x, _ in data[:2]] != [E1, E2]:
        raise AssertionError("block data must start with the two coordinate offsets")
    A_code = o.absent
    results = []

    def allowed(idx, codes):
        """Per copy the set of codes admissible at data[idx] given the codes
        chosen so far, or None when any code is (the coordinate offsets)."""
        if idx < 2:
            return [None] * g
        out = []
        for cd in codes:
            rows = o.composite_rows(cd[E1], cd[E2])
            mask = np.ones(len(rows), dtype=bool)
            for c in range(idx - 2):
                mask &= rows[:, c] == cd[COMP[c]]
            out.append({int(v) for v in rows[mask, idx - 2]})
        return out

    def dfs(idx, c0, Kc, codes):
        if idx == len(data):
            degenerate = Kc.shape[1] > 0
            z = c0 + Kc @ (rng.normal(size=Kc.shape[1]) + 1j * rng.normal(size=Kc.shape[1])) if degenerate else c0
            c = z[:g]
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
        S = sorted(a)
        allow = allowed(idx, codes)
        extra = [()]
        others = [k for k in range(o.nclass) if k not in a]
        for nz in range(1, (g - len(S)) // 2 + 1):
            extra += list(itertools.combinations(others, nz))
        for Z in extra:
            classes = S + list(Z)
            parts = [affine(a[k]) for k in S] + [(0.0, np.zeros(kappa, dtype=complex))] * len(Z)
            b = np.array([p[0] for p in parts], dtype=complex)
            Bl = np.array([p[1] for p in parts], dtype=complex).reshape(len(classes), kappa)
            for assign in itertools.product(classes + [None], repeat=g):
                used = [k for k in assign if k is not None]
                if set(used) != set(classes) or any(used.count(k) < 2 for k in Z):
                    continue
                present = [j for j in range(g) if assign[j] is not None]
                phase_opts = []
                for j in range(g):
                    al = allow[j]
                    if assign[j] is None:
                        if al is not None and A_code not in al:
                            break
                        continue
                    if al is None:
                        phase_opts.append(range(3))
                        continue
                    ls = [v % 3 for v in al if v != A_code and v // 3 == assign[j]]
                    if not ls:
                        break
                    phase_opts.append(ls)
                else:
                    for phases in itertools.product(*phase_opts):
                        A = np.zeros((len(classes), g + kappa), dtype=complex)
                        for j, l in zip(present, phases):
                            A[classes.index(assign[j]), j] = W3P[l]
                        A[:, g:] = -Bl                       # A c - Bl lambda = b
                        sol = _affine_solve_C(A @ Kc, b - A @ c0)
                        if sol is None:
                            continue
                        mu0, N = sol
                        new = [dict(cd) for cd in codes]
                        for j in range(g):
                            new[j][x] = A_code
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

    def __init__(self, D, n2, F1=None, F2=None, seed=29, verbose=False, budget=None):
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
        self.budget = budget                    # a Budget, or None for no caps
        self.last_stats = None                  # the stats of the current or last run (partial after an abort)
        self.last_hits = []                     # the raw hits found so far (partial after an abort)
        self._proj = None

    def _check(self, where, **kw):
        if self.budget is not None:
            self.budget.check(where, **kw)

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
                 "coord_states": [], "joined": 0, "composite_solutions": 0, "composite_states": [],
                 "candidates": 0, "zero_coefficient": 0, "split_pruned": 0, "reconstructions": 0,
                 "hits": 0, "refused": False, "seconds": {}, "peak_rss_gb": None}
        t_start = time.time()
        distinct = sorted(set(int(u) for u in cover))
        mult = {u: list(cover).count(u) for u in distinct}
        b1, b2, bC = target.rhs(x0)
        fam = family_from(self.U1[distinct], self.U2[distinct], self.C[:, distinct], b1, b2, bC)
        if fam is None:
            stats["refused"] = True
            return [], stats
        blocks = [Block(self.options(u), mult[u], i) for i, u in enumerate(distinct) if mult[u] > 1]
        self._proj = _ProjectorCache(blocks)
        self._compat_index = None
        self.last_stats = stats
        bpos = {b.pos for b in blocks}
        exempt = tuple(sorted(bpos))
        if fam.has_zero_coefficient(exempt):
            stats["refused"] = True
            return [], stats
        ords = [i for i in range(len(distinct)) if i not in bpos]
        opts = [self.options(distinct[i]) for i in ords]
        arrays = [o.arrays() for o in opts]
        stats.update(kappa=fam.kappa, kappa1=fam.kappa1, distinct=len(distinct), blocks=[b.g for b in blocks])
        if not opts:
            self.last_hits = []
            hits = BlockOnlyMatcher(self, blocks, x0, target, stats, self.budget, self.log, self.last_hits).run()
            hits = self._dedupe(hits)
            stats["hits"] = len(hits)
            stats["seconds"]["total"] = round(time.time() - t_start, 2)
            stats["peak_rss_gb"] = round(peak_rss_gb(), 3)
            return hits, stats
        # Both coordinate slices are solved once against the initial family;
        # the first slice's states are then joined with the second slice's
        # solutions by compatibility of their coefficient families, which
        # gives exactly the states of solving the second slice against each
        # first-slice state in turn (the family restricted by both slice
        # equations either way) without repeating the second slice's hashing
        # once per state, the cost that dominated at rank 7 with a block.
        states = [([], [], fam, [None] * len(blocks))]
        sols_by_slice = {}
        for e in (E1, E2):
            rhs = target.rhs(add(x0, e))
            t0 = time.time()
            sols = solve_slice3(arrays, blocks, fam, rhs, self.rng, stats=stats, log=self.log,
                                budget=self.budget, where=f"coordinate slice {e}", proj=self._proj)
            self._check(f"coordinate slice {e}", solutions=len(sols))
            sols_by_slice[e] = sols
            stats.setdefault("coord_raw", []).append(len(sols))
            stats["seconds"][f"solve_{e[0]}{e[1]}"] = round(time.time() - t0, 2)
            self.log(f"  slice {e}: {len(sols)} solutions against the initial family, {time.time() - t0:.1f}s, "
                     f"rss {peak_rss_gb():.2f} GB")
            if not sols:
                stats["coord_states"].append(0)
                stats["peak_rss_gb"] = round(peak_rss_gb(), 3)
                return [], stats
        for e in (E1, E2):
            rhs = target.rhs(add(x0, e))
            t0 = time.time()
            new, count = [], 0
            for cl, bl, f, sp in states:
                if e == E1:
                    pairs = sols_by_slice[e]
                else:
                    pairs = self._compatible(f, sols_by_slice[e], arrays, blocks, rhs)
                count += len(pairs)                 # the solutions of this slice over the states, as before
                for k, (combo, Ssel, f2) in enumerate(pairs):
                    if k % 10_000 == 9_999:
                        self._check(f"join at coordinate slice {e}", states=len(new))
                    if f2.has_zero_coefficient(exempt):
                        stats["zero_coefficient"] += 1
                        continue
                    sp2 = self._join_blocks(f2, arrays, blocks, combo, Ssel, rhs, sp)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    new.append((cl + [tuple(int(c) for c in combo)], bl + [Ssel], f2, sp2))
                self._check(f"join at coordinate slice {e}", states=len(new))
            stats["coord_solutions"].append(count)
            stats["coord_states"].append(len(new))
            stats["seconds"][f"join_{e[0]}{e[1]}"] = round(time.time() - t0, 2)
            states = new
            self.log(f"  slice {e}: {count} solutions over the states, {len(states)} states after the join, "
                     f"parameters {sorted(set(f.kappa for _, _, f, _ in states))}, "
                     f"{time.time() - t0:.1f}s, rss {peak_rss_gb():.2f} GB")
            if not states:
                stats["peak_rss_gb"] = round(peak_rss_gb(), 3)
                return [], stats
        stats["joined"] = len(states)
        hits = []
        t0 = time.time()
        for k, (cl, bl, f, sp) in enumerate(states):
            hits.extend(self._complete(distinct, x0, opts, ords, blocks, cl, bl, f, sp, target, stats))
            if self.verbose and (k + 1) % 200 == 0:
                self.log(f"  composite stage: {k + 1}/{len(states)} joined states, {len(hits)} raw hits, "
                         f"{time.time() - t0:.1f}s, rss {peak_rss_gb():.2f} GB")
        stats["seconds"]["composite"] = round(time.time() - t0, 2)
        hits = self._dedupe(hits)
        stats["hits"] = len(hits)
        stats["seconds"]["total"] = round(time.time() - t_start, 2)
        stats["peak_rss_gb"] = round(peak_rss_gb(), 3)
        return hits, stats

    def _compatible(self, f1, sols, arrays, blocks, rhs):
        """The solutions (combo, Ssel, f2raw) of a slice against the initial
        family whose family meets f1, each returned with f1 restricted by
        that slice equation (the meet). Candidates are found mod P1: a
        pinned f1 against the pinned solutions by the coefficient vector
        (a dict lookup) and against the unpinned ones by membership, an
        unpinned f1 against the pinned solutions by the annihilator of its
        direction space and against the unpinned ones pairwise; every
        candidate is then decided exactly by the restriction, so the list is
        complete (the P1 family of a solution contains its complex family)."""
        # the index is keyed on the solution list itself (a reference is
        # kept, so the list cannot be freed and its id reused): keyed on
        # id(sols) it was served stale to the next run whose second-slice
        # list landed at the same address, which dropped compatible pairs
        # unrecorded (8 of the 270 product-control runs lost their hit)
        idx = getattr(self, "_compat_index", None)
        if idx is None or idx[0] is not sols:
            pinned, free = {}, []
            for j, (_, _, f) in enumerate(sols):
                if f.kappa1 == 0:
                    pinned.setdefault(tuple(int(x) for x in f.parts[0][0] % P1), []).append(j)
                else:
                    free.append(j)
            flat = [j for js in pinned.values() for j in js]
            n = len(sols[0][2].parts[0][0])
            D = (np.array([sols[j][2].parts[0][0] for j in flat], dtype=np.int64) % P1 if flat
                 else np.zeros((0, n), dtype=np.int64))
            self._compat_index = (sols, pinned, free, D, flat)
        _, pinned, free, D, flat = self._compat_index
        d0, K = f1.parts[0]
        if K.shape[1] == 0:
            cand = list(pinned.get(tuple(int(x) for x in d0 % P1), [])) + free
        else:
            Ann = _annihilator_mod(K % P1, P1)                       # rows a with a K = 0 mod P1
            hit = np.flatnonzero(~np.any(_mm_int64(Ann, (D - d0[None, :]).T % P1, P1), axis=0)) if len(flat) else []
            cand = [flat[i] for i in hit] + free
        out = []
        for j in cand:
            combo, Ssel, _ = sols[j]
            Ps = self._proj(Ssel)[0] if blocks else None
            f2 = restrict(f1, *slice_system(arrays, blocks, combo, Ssel, rhs, Ps))
            if f2 is not None:
                out.append((combo, Ssel, f2))
        return out

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

    def _block_coordinates(self, fam, arrays, blocks, combo, Ssel, rhs, strict=False, affine=False):
        """Coordinates of the residual rhs - W d on the chosen translates
        (complex), or None when it does not lie in their span or the split
        between blocks is ambiguous (the chosen translates of different
        blocks dependent). In the join None means no pruning; at the final
        reconstruction (`strict`) it would mean a silently dropped state, so
        the run raises UnpinnedFamily there instead. With `affine` the
        coordinates are returned as (a0, A) with a(lambda) = a0 + A lambda
        over the family d = d0 + K lambda (A has kappa columns), which the
        reconstruction of a single block takes with lambda as an unknown;
        without it the residual is taken at a generic member of the family."""
        d0, K = fam.parts[2]
        if not affine:
            d0, K = fam.coefficients(self.rng), K[:, :0]
        r_all = len(arrays) + len(blocks)
        bpos = {b.pos for b in blocks}
        ords = [i for i in range(r_all) if i not in bpos]
        W = np.zeros((rhs[2].shape[0], r_all), dtype=complex)
        for i, c in zip(ords, combo):
            W[:, i] = arrays[ords.index(i)][2][c]
        res = np.column_stack([rhs[2] - W @ d0, -(W @ K)])          # columns: the constant part, then per lambda
        _, Vs = self._proj(Ssel)
        V = Vs[2]
        if V.shape[1] == 0:
            if np.abs(res).max() < 1e-7:
                out = np.zeros((0, K.shape[1] + 1), dtype=complex)
                return (out[:, 0], out[:, 1:]) if affine else out[:, 0]
            if strict:
                raise UnpinnedFamily("reconstruction: a slice residual is nonzero with no translate chosen")
            return None
        cols = []
        for j in range(res.shape[1]):
            sol = _affine_solve_C(V, res[:, j])
            if sol is None:
                if strict:
                    raise UnpinnedFamily("reconstruction: a slice residual lies outside the span of the chosen "
                                         "translates at the solve tolerance although the exact restriction "
                                         "accepted it")
                return None
            if sol[1].shape[1]:
                if strict:
                    raise UnpinnedFamily(f"reconstruction: the chosen translates of the blocks are dependent "
                                         f"({sol[1].shape[1]} free coordinates), so the split of the residual "
                                         "between the blocks is a family; carrying it as unknowns is not "
                                         "implemented")
                return None
            cols.append(sol[0])
        a = np.column_stack(cols)
        return (a[:, 0], a[:, 1:]) if affine else a[:, 0]

    def _complete(self, distinct, x0, opts, ords, blocks, cl, bl, fam, sp, target, stats):
        r = len(opts)
        dim2 = 3 ** self.n2
        rows = [opts[i].composite_rows(cl[0][i], cl[1][i]) for i in range(r)]
        exempt = tuple(sorted(b.pos for b in blocks))
        states = [([], [], fam, sp, [np.ones(len(R), dtype=bool) for R in rows])]
        if len(stats["composite_states"]) < len(COMP):
            stats["composite_states"] = [0] * len(COMP)
        for c, x in enumerate(COMP):
            rhs = target.rhs(add(x0, x))
            new = []
            for k, (cl2, bl2, f, sp1, alive) in enumerate(states):
                if k % 200 == 199:
                    self._check(f"composite slice {x}", states=len(new))
                codes = [np.unique(rows[i][alive[i], c]) for i in range(r)]
                arrays = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
                sols = solve_slice3(arrays, blocks, f, rhs, self.rng, stats=stats, log=self.log,
                                    budget=self.budget, where=f"composite slice {x}", proj=self._proj)
                stats["composite_solutions"] += len(sols)
                self._check(f"composite slice {x}", solutions=len(sols), states=len(new))
                for combo, Ssel, f2 in sols:
                    if f2.has_zero_coefficient(exempt):
                        stats["zero_coefficient"] += 1
                        continue
                    sp2 = self._join_blocks(f2, arrays, blocks, combo, Ssel, rhs, sp1)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    chosen = tuple(int(codes[i][combo[i]]) for i in range(r))
                    alive2 = [alive[i] & (rows[i][:, c] == chosen[i]) for i in range(r)]
                    new.append((cl2 + [chosen], bl2 + [Ssel], f2, sp2, alive2))
            states = new
            stats["composite_states"][c] += len(states)
            if not states:
                return []
        hits = []
        full_arrays = [o.arrays() for o in opts]
        for cl2, bl2, f, _, _ in states:
            if len(blocks) > 1 and f.kappa > 0:
                stats["unpinned"] = stats.get("unpinned", 0) + 1
                raise UnpinnedFamily(f"{f.kappa}-parameter coefficient family left after the nine slice "
                                     f"equations on a base with {len(blocks)} repeated states (blocks "
                                     f"{[b.g for b in blocks]}); the reconstruction with the parameter shared "
                                     "across several blocks is not implemented")
            if blocks and f.kappa > 0:
                stats["unpinned_reconstructed"] = stats.get("unpinned_reconstructed", 0) + 1
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
            if not blocks:
                hits.append(self.confirm(ord_terms, f.kappa, target))
                continue
            d0, K = f.parts[2]
            per_block = [[] for _ in blocks]
            for s, x in enumerate(OFFSETS):
                if s < 2:
                    combo, Ssel = cl[s], bl[s]
                else:
                    combo, Ssel = cl2[s - 2], bl2[s - 2]
                rhs = target.rhs(add(x0, x))
                a0, A = self._block_coordinates(f, full_arrays, blocks, combo, Ssel, rhs, strict=True, affine=True)
                pos = 0
                for bi, (b, S) in enumerate(zip(blocks, Ssel)):
                    per_block[bi].append((x, {k: (a0[pos + j], A[pos + j]) for j, k in enumerate(S)}))
                    pos += len(S)
            recon = [reconstruct_block(b.o, b.g, (d0[b.pos], K[b.pos]), per_block[bi], self.rng)
                     for bi, b in enumerate(blocks)]
            stats["reconstructions"] += 1
            for choice in itertools.product(*recon):
                terms = list(ord_terms)
                degenerate = False
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
