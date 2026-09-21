"""Multi-qubit slicing at the qubit H cells: a minimal slice along n_1 qubits.

Setting. Slice a decomposition psi^m = sum_i c_i s_i of |H>^m along the first
n_1 qubits: u_i^(x) = (<x| (x) I) s_i for x in F_2^{n_1}, a vector on the
remaining n_2 = m - n_1 qubits. Then sum_i c_i u_i^(x) = alpha_x psi^{n_2}
with alpha_x = prod_k alpha_{x_k}, and both amplitudes of |H> are nonzero,
so every slice is a decomposition of psi^{n_2}. The flat F_i of s_i projects
to an affine flat pi(F_i) of F_2^{n_1}, and the term is nonzero exactly on
the 2^j points of that flat.

Structure of one term (the qubit analogue of the slice-and-lift lemma). Let
s be an m-qubit stabilizer state with projected flat x_0 + V, dim V = j, and
slice u at x_0. Reading off the stabilizer group elements above a basis
v_1, ..., v_j of V gives Paulis Q_1, ..., Q_j on the n_2 qubits with

    u^(x_0 + t_1 v_1 + ... + t_j v_j) = i^{l.t} (-1)^{q(t)} Q_j^{t_j} ... Q_1^{t_1} u

for some l in Z_4^j and q a quadratic form over F_2 with no diagonal (the
diagonal is absorbed by l). Conversely every such expression is a stabilizer
state: it is C_j ... C_1 (phi (x) u) with C_k the Pauli Q_k controlled on
the coordinate t_k (a Clifford) and phi the full-support stabilizer state on
V with phases i^{l.t} (-1)^{q(t)}. Changing the representative of a Pauli
class (Q_k -> Q_k g with g in Stab(u), or a phase) only moves (l, q), so
with one fixed representative per class the data (V, class map, l, q) is in
bijection with the states having slice u at x_0. The 2^{n_2} Pauli classes
mod Stab(u) give an orthonormal basis of translates Q u, so each slice is a
basis vector times a fourth root of unity, and a term with base slice u at
x_0 has

    sum_j [subspaces V of dim j] (2^{n_2})^j 4^j 2^{j(j-1)/2}

shapes: 8385 for two sliced qubits and four remaining (8192 four-slice, 192
line, 1 point), which closes the six-qubit dictionary count exactly
(36720 x 8192 + 6 x 36720 x 64 + 4 x 36720 = 315,057,600).

Case M (minimal slice). Some slice x_0 has exactly r = chi(psi^{n_2}) nonzero
terms, so those slices form a minimal decomposition (d_i, u_i) of psi^{n_2},
one of the stored lists up to the unitary symmetry of psi^{n_2} (which acts
on the unsliced qubits and preserves the slicing), and c_i = alpha_{x_0} d_i.
The R - r invisible terms vanish at x_0, so each is supported on a flat of
F_2^{n_1} missing x_0, with a free coefficient. For every multiset of such
flats, k_x = number of invisible terms present at x, the visible terms must
leave at every other slice a residual alpha_x psi^{n_2} - sum_visible of
stabilizer rank at most k_x. Slices with k_x = 0 need an exact match over
Pauli classes and phases, k_x = 1 a stabilizer residual (looked up in the
n_2-qubit dictionary), k_x = 2 a rank-2 residual (exact in the tables on two
qubits, tested on candidates otherwise). The per-slice tables run over all
(4 2^{n_2} + 1)^r code combinations (65^4 = 17.8 million for four visible
terms), candidates are joined from the two most constraining slices by
bucketed shapes, checked on the rest by table lookup, and survivors go to an
exact completion of the invisible terms (a private slice pins a term's
slice, a shared slice is split over dictionary pairs, the other slices of
the flat run over the Pauli classes and phases with pruning).

One pattern has a single constrained slice: two invisible terms on the same
line of F_2^2 (rank r + 2, two sliced qubits), which is the shape of the
known rank-6 decomposition of |H>^6 and the case relaxed_lift.py calls B1.
It gets its own path: the visible terms are pinned at x_0 and at the exact
slice, the 129^4 remaining assignments are enumerated in chunks, and each
residual pair passes a necessary condition for stabilizer rank 2 derived
from the support structure of a s + b t (one modulus on each private part
of the two flats, at most four on their intersection; see rank2_filter)
before the exact rank-2 test and the completion. This costs about half an
hour per exact match and is used for the control, not for the rank-5
search, where every pattern has at least two constrained slices.

At m = 6, R = 5, n_1 = 2 (the record cell: rank 5 would give exponent
log_2(5)/6 = 0.387 below the published 0.3963) the one invisible term is a
line or a point of F_2^2, so at least one other slice is exact and the rest
have k <= 1. The slice ratio alpha_x / alpha_{x_0} is tan(pi/8)^(|x| - |x_0|),
and the tables say, for every one of the 30 stored decompositions: ratio 1
has one exact combination (the identity classes) and 260 to 360 stabilizer
residuals; ratios tan(pi/8)^(+-1) and tan(pi/8)^(+-2) have no exact and no
stabilizer-residual combination at all. The exact zero at tan(pi/8) is the
slice-and-lift exclusion behind qubit_H-m5-lower-5, recomputed rather than
assumed; the stabilizer zero is what closes the rank-5 case, since a slice
at Hamming distance one from x_0 then needs at least two invisible terms.

By the copy-permutation symmetry of psi^m the bipartition {1..n_1} | rest
is general, and permutations of the sliced qubits reduce x_0 to its Hamming
weight (|H> has no monomial symmetry).

Usage: two_qubit_slice.py [--m 6] [--n1 2] [--rank 5] [--x0 reps|all]
                          [--decs i,j,...] [--control] [--witness PATH]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (alpha, confirm_stabilizer, is_stabilizer_batch,  # noqa: E402
                    load_decompositions, target, term_vector)
from rank_exclusion import _parallel_groups, dictionary  # noqa: E402

TOL = 1e-7
FOURTH = np.array([1, 1j, -1, -1j])
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORBIT = "qubit_H"
RANKS = {2: 2, 3: 3, 4: 4}          # chi(|H>^n) for the stored minimal lists


# ------------------------------------------------------------ geometry ----

def subspaces(n):
    """Every linear subspace of F_2^n as (dim, basis) with a reduced basis;
    points are integers with bit n-1-k for coordinate k."""
    seen = {}
    for j in range(n + 1):
        for vecs in itertools.combinations(range(1, 1 << n), j):
            basis = _reduce(list(vecs))
            if len(basis) != j:
                continue
            pts = frozenset(_span(basis))
            if pts not in seen:
                seen[pts] = (j, basis)
    return sorted(seen.values(), key=lambda t: (t[0], t[1]))


def _reduce(vecs):
    """Reduced echelon basis of the span of integer bit vectors."""
    basis = []
    for v in vecs:
        for b in sorted(basis, reverse=True):
            if v ^ b < v:
                v ^= b
        if v:
            basis.append(v)
    basis.sort(reverse=True)
    for i in range(len(basis)):
        lead = 1 << (basis[i].bit_length() - 1)
        for k in range(len(basis)):
            if k != i and basis[k] & lead:
                basis[k] ^= basis[i]
    return sorted(basis, reverse=True)


def _span(basis):
    pts = [0]
    for b in basis:
        pts = pts + [p ^ b for p in pts]
    return pts


def flats_missing(n1, x0):
    """Every affine flat of F_2^{n1} of dimension < n1 not containing x0, as
    (dim, frozenset of points)."""
    out = set()
    for j, basis in subspaces(n1):
        if j == n1:
            continue
        V = _span(basis)
        for y in range(1 << n1):
            pts = frozenset(y ^ v for v in V)
            if x0 not in pts:
                out.add((j, pts))
    return sorted(out, key=lambda t: (t[0], sorted(t[1])))


def flat_basis(pts, x):
    """Reduced basis of the direction space of the flat `pts` through x."""
    return _reduce([p ^ x for p in pts if p != x])


# -------------------------------------------------------------- Paulis ----

class PauliEnv:
    """Pauli action on n2 qubits: (X^a Z^c u)[y xor a] = (-1)^{c.y} u[y]."""

    def __init__(self, n2):
        self.n2 = n2
        self.N = 1 << n2
        y = np.arange(self.N)
        self.par = np.array([[bin(c & t).count("1") % 2 for t in y] for c in y])
        self.sign = (-1.0) ** self.par                       # (c, y)
        self.idx = y

    def apply(self, u, a, c):
        out = np.empty(self.N, dtype=complex)
        out[self.idx ^ a] = self.sign[c] * u
        return out

    def classes(self, u):
        """One representative (a, c) per Pauli class mod Stab(u), the
        translate vectors T, and the map (a, c) -> (class, phase) with
        X^a Z^c u = i^phase T[class]. Exactly 2^{n2} classes."""
        reps, T, cls_of = [], [], {}
        for a in range(self.N):
            for c in range(self.N):
                v = self.apply(u, a, c)
                found = None
                for k, t in enumerate(T):
                    ph = phase_code(v, t)
                    if ph is not None:
                        found = (k, ph)
                        break
                if found is None:
                    T.append(v)
                    reps.append((a, c))
                    found = (len(T) - 1, 0)
                cls_of[(a, c)] = found
        if len(T) != self.N or cls_of[(0, 0)] != (0, 0):
            raise AssertionError("expected 2^n2 Pauli classes with the identity first")
        return reps, T, cls_of


def phase_code(v, ref, tol=1e-7):
    """ph in Z_4 with v = i^ph ref, or None."""
    j = int(np.argmax(np.abs(ref) > 1e-9))
    if abs(ref[j]) < 1e-9 or abs(v[j]) < 1e-9:
        return None
    r = v[j] / ref[j]
    ph = int(round(np.angle(r) / (np.pi / 2))) % 4
    if abs(r - FOURTH[ph]) < 1e-6 and np.abs(v - FOURTH[ph] * ref).max() < tol:
        return ph
    return None


def phase_tables(j):
    """Phase increments over the nonzero t in F_2^j for every (l, q): rows
    indexed by (l in Z_4^j, q over the j(j-1)/2 cross terms), columns by t =
    1..2^j-1, values (l.t + 2 q(t)) mod 4."""
    ts = np.arange(1, 1 << j)
    bits = (ts[:, None] >> np.arange(j)[None, :]) & 1                # (T, j)
    L = np.array(list(itertools.product(range(4), repeat=j)))         # (4^j, j)
    lin = (L @ bits.T) % 4                                             # (4^j, T)
    pairs = [(a, b) for a in range(j) for b in range(a + 1, j)]
    Q = np.zeros((1, 0), dtype=np.int64) if not pairs else np.array(
        list(itertools.product(range(2), repeat=len(pairs))), dtype=np.int64)
    quad = np.zeros((Q.shape[0], len(ts)), dtype=np.int64)
    for p, (a, b) in enumerate(pairs):
        quad += Q[:, p:p + 1] * (bits[:, a] * bits[:, b])[None, :]
    return ((lin[:, None, :] + 2 * quad[None, :, :]) % 4).reshape(-1, len(ts))


def shape_count(n1, n2):
    """The number of m-qubit stabilizer states with a given slice at a given
    point, from the lemma."""
    total = 0
    for j, _ in subspaces(n1):
        total += (1 << n2) ** j * 4 ** j * 2 ** (j * (j - 1) // 2)
    return total


class TermShapes:
    """Every m-qubit stabilizer state whose slice at x0 is u, as per-slice
    codes over the other points of F_2^{n1}: code = 4 * class + phase for a
    present slice (slice vector = i^phase T[class]), ABSENT otherwise."""

    def __init__(self, env, u, x0, n1):
        self.env = env
        self.u = u
        self.x0 = x0
        self.n1 = n1
        self.N1 = 1 << n1
        self.ABSENT = 4 * env.N
        self.reps, self.T, self.cls_of = env.classes(u)
        self.others = [x for x in range(self.N1) if x != x0]
        self.slot = {x: i for i, x in enumerate(self.others)}
        self.codes = self._all_codes()
        if self.codes.shape[0] != shape_count(n1, env.n2):
            raise AssertionError("shape enumeration does not match the lemma's count")

    def _all_codes(self):
        env = self.env
        blocks = [np.full((1, self.N1 - 1), self.ABSENT, dtype=np.int8)]      # the point
        for j, basis in subspaces(self.n1):
            if j == 0:
                continue
            ts = list(range(1, 1 << j))
            pts = [self.x0 ^ _combine(basis, t) for t in ts]
            slots = [self.slot[p] for p in pts]
            PH = phase_tables(j)                                    # (rows, 2^j - 1)
            for ks in itertools.product(range(env.N), repeat=j):
                # P(t) = Q_top ... Q_1 u with top the highest set bit of t
                P = {0: self.u}
                sym = {0: (0, 0)}
                cls = np.zeros(len(ts), dtype=np.int64)
                f = np.zeros(len(ts), dtype=np.int64)
                for t in ts:
                    i = t.bit_length() - 1
                    a, c = self.reps[ks[i]]
                    prev = t ^ (1 << i)
                    P[t] = env.apply(P[prev], a, c)
                    sym[t] = (sym[prev][0] ^ a, sym[prev][1] ^ c)
                    k, _ = self.cls_of[sym[t]]
                    ph = phase_code(P[t], self.T[k])
                    if ph is None:
                        raise AssertionError("a Pauli product is not a phased class representative")
                    cls[t - 1] = k
                    f[t - 1] = ph
                block = np.full((PH.shape[0], self.N1 - 1), self.ABSENT, dtype=np.int8)
                block[:, slots] = (4 * cls[None, :] + (f[None, :] + PH) % 4).astype(np.int8)
                blocks.append(block)
        return np.concatenate(blocks, axis=0)

    def slice_vectors(self):
        """(ABSENT + 1, 2^n2): the vector of each code (zero for ABSENT)."""
        V = np.zeros((self.ABSENT + 1, self.env.N), dtype=complex)
        for k in range(self.env.N):
            for ph in range(4):
                V[4 * k + ph] = FOURTH[ph] * self.T[k]
        return V


def _combine(basis, t):
    y = 0
    for i, b in enumerate(basis):
        if (t >> i) & 1:
            y ^= b
    return y


# ------------------------------------------------------------ dictionary ---

class Dict:
    """The n2-qubit dictionary with an exact membership key (entries divided
    by the first nonzero one lie in {0, +-1, +-i})."""

    def __init__(self, n2):
        self.n2 = n2
        self.D = dictionary(2, n2)
        self.keys = set(self._keys(self.D.T))
        rng = np.random.default_rng(5)
        self.f1 = rng.normal(size=self.D.shape[0]) + 1j * rng.normal(size=self.D.shape[0])
        self.f2 = rng.normal(size=self.D.shape[0]) + 1j * rng.normal(size=self.D.shape[0])
        self.f1D = self.f1 @ self.D
        self.f2D = self.f2 @ self.D

    @staticmethod
    def _keys(rows, tol=1e-6):
        """Canonical byte keys for the rows that are phased 0/+-1/+-i
        vectors (None for the rest)."""
        nz = np.abs(rows) > 1e-9
        first = np.argmax(nz, axis=1)
        piv = rows[np.arange(rows.shape[0]), first]
        W = rows / piv[:, None]
        Wr, Wi = np.rint(W.real), np.rint(W.imag)
        ok = (np.abs(W.real - Wr).max(axis=1) < tol) & (np.abs(W.imag - Wi).max(axis=1) < tol)
        K = np.concatenate((Wr, Wi), axis=1).astype(np.int8)
        out = []
        for i in range(rows.shape[0]):
            out.append(K[i].tobytes() if ok[i] else None)
        return out

    def member_mask(self, rows):
        if rows.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        return np.array([k is not None and k in self.keys for k in self._keys(rows)])

    def is_state(self, v):
        return bool(self.member_mask(v[None, :])[0])

    def split_two(self, r):
        """Every way to write r as a s_i + b s_j with two non-parallel
        dictionary states, both coefficients nonzero: the candidate summands
        a s_i (each appears once per role)."""
        D = self.D
        rn = r / np.linalg.norm(r)
        q = D - np.outer(rn, rn.conj() @ D)
        nq = np.linalg.norm(q, axis=0)
        keep = np.flatnonzero(nq > 1e-7)
        groups, _ = _parallel_groups(q[:, keep] / nq[keep][None, :], np.random.default_rng(11))
        out = []
        for g in groups:
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    i, j = int(keep[g[a]]), int(keep[g[b]])
                    A = np.column_stack((D[:, i], D[:, j]))
                    c, *_ = np.linalg.lstsq(A, r, rcond=None)
                    if np.linalg.norm(A @ c - r) < 1e-7 and min(abs(c)) > 1e-7:
                        out.append(c[0] * D[:, i])
                        out.append(c[1] * D[:, j])
        return out

    def rank_le2(self, r, tol=1e-6):
        """Is r a combination of at most two dictionary states. The quotient
        trick of rank_exclusion with the two functionals precomputed: the
        images of the states in C^d / span(r) are parallel for a spanning
        pair, so their projective keys agree; runs of near-equal keys are
        checked in full."""
        nr = np.linalg.norm(r)
        if nr < TOL:
            return True
        rn = r / nr
        coef = rn.conj() @ self.D
        if (1.0 - np.abs(coef) ** 2 < 1e-12).any():
            return True                                   # r is a dictionary state
        k1 = self.f1D - (self.f1 @ rn) * coef
        k2 = self.f2D - (self.f2 @ rn) * coef
        key = k1 / k2
        order = np.argsort(key.real, kind="stable")
        ks = key[order]
        gap = np.abs(np.diff(ks.real)) > tol * (1 + np.abs(ks.real[:-1]))
        starts = np.concatenate(([0], np.flatnonzero(gap) + 1, [len(ks)]))
        for a, b in zip(starts[:-1], starts[1:]):
            if b - a < 2:
                continue
            members = order[a:b]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if abs(key[members[i]] - key[members[j]]) > tol * (1 + abs(key[members[i]])):
                        continue
                    A = np.column_stack((self.D[:, members[i]], self.D[:, members[j]]))
                    c, *_ = np.linalg.lstsq(A, r, rcond=None)
                    if np.linalg.norm(A @ c - r) < 1e-7:
                        return True
        return False


def sum_table(arrays):
    S = arrays[0]
    for A in arrays[1:]:
        S = (S[:, None, :] + A[None, :, :]).reshape(-1, S.shape[1])
    return S


_FLAT_TABLES = {}


def proper_flat_table(n):
    """bool[mask]: the points of F_2^n in `mask` lie in a proper affine flat."""
    if n not in _FLAT_TABLES:
        N = 1 << n
        out = np.zeros(1 << N, dtype=bool)
        for mask in range(1 << N):
            pts = [y for y in range(N) if (mask >> y) & 1]
            out[mask] = len(pts) <= 1 or len(_reduce([q ^ pts[0] for q in pts[1:]])) < n
        _FLAT_TABLES[n] = out
    return _FLAT_TABLES[n]


def rank2_filter(R, n, tol=1e-6):
    """Necessary condition for stabilizer rank <= 2 of each row of R (2^n
    entries), from the support structure of r = a s + b t: |r| is one value A
    on F_s minus F_t, one value B on F_t minus F_s, at most four values on the
    intersection H, and zero elsewhere. Three cases cover everything:
    F_s = F_t (at most four distinct nonzero moduli); F_s the whole space
    (the points with |r| != A, zeros included, lie in the proper flat F_t, so
    A has multiplicity >= 2^(n-1) and sits at the middle of the sorted
    moduli); both proper and distinct (|H| <= 2^(n-2), the support has at
    most 3 2^(n-2) points, and the nonzero points outside the two most
    frequent moduli number at most |H|)."""
    A = np.abs(R)
    N = A.shape[1]
    table = proper_flat_table(n)
    S = np.sort(A, axis=1)
    nz = S > 1e-9
    supp = nz.sum(axis=1)
    distinct = ((np.diff(S, axis=1) > tol) & nz[:, 1:]).sum(axis=1) + nz[:, 0]
    ok = distinct <= 4
    weights = (1 << np.arange(N)).astype(np.int64)
    for pos in (N // 2 - 1, N // 2):
        Aval = S[:, pos]
        Z = (np.abs(A - Aval[:, None]) > tol) | (A <= 1e-9)
        Zm = (Z.astype(np.int64) * weights[None, :]).sum(axis=1)
        ok |= (Aval > 1e-9) & table[Zm]
    rest = np.flatnonzero(~ok & (supp <= 3 * (N // 4)) & (supp > 0))
    if len(rest):
        Ar = A[rest]
        nzr = Ar > 1e-9
        same = (np.abs(Ar[:, :, None] - Ar[:, None, :]) < tol) & nzr[:, None, :] & nzr[:, :, None]
        mult = same.sum(axis=2)
        top1 = mult.max(axis=1)
        v1 = Ar[np.arange(len(rest)), mult.argmax(axis=1)]
        other = nzr & (np.abs(Ar - v1[:, None]) > tol)
        top2 = (mult * other).max(axis=1)
        ok[rest] = nzr.sum(axis=1) - top1 - top2 <= N // 4
    return ok


def rank2_filter_joint(J, tol=1e-6):
    """Necessary condition for two stabilizer terms one qubit up: at most six
    distinct nonzero moduli, five if the support is full."""
    S = np.sort(np.abs(J), axis=1)
    nz = S > 1e-9
    distinct = ((np.diff(S, axis=1) > tol) & nz[:, 1:]).sum(axis=1) + nz[:, 0]
    return np.where(nz.all(axis=1), distinct <= 5, distinct <= 6)


def residual_table(Vs, d, beta_psi, dic, rank2="none"):
    """int8 table over all code combinations of the r visible terms: 0 if
    beta psi - sum_i d_i V_i[c_i] vanishes, 1 if it is a stabilizer state, 2
    if it has stabilizer rank 2 (rank2 = "exact", small dictionaries only),
    3 otherwise."""
    r = len(Vs)
    C = Vs[0].shape[0]
    N = Vs[0].shape[1]
    h = max(1, r // 2)
    A = sum_table([d[i] * Vs[i] for i in range(h)])
    B = sum_table([d[i] * Vs[i] for i in range(h, r)]) if r > h else np.zeros((1, N), dtype=complex)
    nB = B.shape[0]
    table = np.full(A.shape[0] * nB, 3, dtype=np.int8)
    blk = max(1, 3_000_000 // (nB * N))
    for s in range(0, A.shape[0], blk):
        Rb = (beta_psi[None, None, :] - A[s:s + blk, None, :] - B[None, :, :]).reshape(-1, N)
        nrm = np.linalg.norm(Rb, axis=1)
        cls = np.full(Rb.shape[0], 3, dtype=np.int8)
        cand = np.flatnonzero((nrm >= TOL) & is_stabilizer_batch(Rb, 2))
        if len(cand):
            cls[cand[dic.member_mask(Rb[cand])]] = 1
        cls[nrm < TOL] = 0
        if rank2 == "exact":
            for j in np.flatnonzero(cls == 3):
                if dic.rank_le2(Rb[j]):
                    cls[j] = 2
        table[s * nB:(s + blk) * nB] = cls
    return table.reshape((C,) * r)


# ------------------------------------------------------------ patterns ----

def coverage_patterns(n1, x0, n_inv):
    """Multisets of n_inv invisible flats (missing x0) grouped by their
    coverage vector over the other points."""
    others = [x for x in range(1 << n1) if x != x0]
    shapes = flats_missing(n1, x0)
    pats = {}
    for combo in itertools.combinations_with_replacement(range(len(shapes)), n_inv):
        k = tuple(sum(1 for s in combo if x in shapes[s][1]) for x in others)
        pats.setdefault(k, []).append(tuple(shapes[s] for s in combo))
    return shapes, pats


# ---------------------------------------------------------- completion ----

def _cover_count(x, shapes):
    return sum(1 for _, p in shapes if x in p)


class Completer:
    """Exact completion of the invisible terms."""

    def __init__(self, env, dic, n1):
        self.env = env
        self.dic = dic
        self.n1 = n1
        self.N1 = 1 << n1

    def slice_ok(self, r, k):
        if k == 0:
            return np.abs(r).max() < TOL
        if k == 1:
            return np.linalg.norm(r) > TOL and self.dic.is_state(r)
        return True

    def flat_terms(self, s, x, pts, blocks, rest):
        """Stabilizer terms (full m-qubit vectors) supported on the flat `pts`
        with slice s at x, pruned by the residual at every other point of the
        flat against the invisible shapes in `rest` still covering it."""
        basis = flat_basis(pts, x)
        j = len(basis)
        reps, _, _ = self.env.classes(s / np.linalg.norm(s))
        need = {p: _cover_count(p, rest) for p in pts}

        def rec(i, P):
            # P: dict t -> slice vector for t in span of the first i basis vectors
            if i == j:
                t_vec = np.zeros(self.N1 * self.env.N, dtype=complex)
                for t, v in P.items():
                    p = x ^ _combine(basis, t)
                    t_vec[p * self.env.N:(p + 1) * self.env.N] = v
                yield t_vec
                return
            prev = sorted(P)
            bit = 1 << i
            for a, c in reps:
                QP = {t: self.env.apply(P[t], a, c) for t in prev}
                for l in range(4):
                    for qbits in itertools.product(range(2), repeat=i):
                        new = {}
                        good = True
                        for t in prev:
                            sgn = sum(qbits[k] for k in range(i) if (t >> k) & 1) % 2
                            v = FOURTH[l] * (-1) ** sgn * QP[t]
                            p = x ^ _combine(basis, t | bit)
                            if not self.slice_ok(blocks[p] - v, need[p]):
                                good = False
                                break
                            new[t | bit] = v
                        if good:
                            P2 = dict(P)
                            P2.update(new)
                            yield from rec(i + 1, P2)

        yield from rec(0, {0: s})

    def complete(self, res, shapes):
        """Is res a sum of one stabilizer term per shape (flats missing x0),
        each nonzero exactly on its flat? Returns the term vectors or None."""
        if not shapes:
            return [] if np.abs(res).max() < TOL else None
        B = res.reshape(self.N1, self.env.N)
        order = sorted(range(len(shapes)),
                       key=lambda si: min(_cover_count(p, shapes) for p in shapes[si][1]))
        si = order[0]
        _, pts = shapes[si]
        rest = shapes[:si] + shapes[si + 1:]
        x = min(pts, key=lambda p: _cover_count(p, shapes))
        k = _cover_count(x, shapes)
        r = B[x]
        if np.linalg.norm(r) < TOL:
            return None
        if k == 1:
            if not self.dic.is_state(r):
                return None
            firsts = [r]
        elif k == 2:
            firsts = self.dic.split_two(r)
        else:
            raise NotImplementedError(f"every point of a shape is covered {k} times")
        for s in firsts:
            for t in self.flat_terms(s, x, pts, B, rest):
                out = self.complete(res - t, rest)
                if out is not None:
                    return [t] + out
        return None


# ---------------------------------------------------------------- search ----

def visible_vectors(TS, Vs, opts):
    out = []
    for t, V, o in zip(TS, Vs, opts):
        s = np.zeros(t.N1 * t.env.N, dtype=complex)
        s[t.x0 * t.env.N:(t.x0 + 1) * t.env.N] = t.u
        for x in t.others:
            code = t.codes[o, t.slot[x]]
            if code != t.ABSENT:
                s[x * t.env.N:(x + 1) * t.env.N] = V[code]
        out.append(s)
    return out


def bucket(codes_y, codes_z, C):
    key = codes_y.astype(np.int64) * C + codes_z.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cut = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(ks)]))
    return {(int(ks[a]) // C, int(ks[a]) % C): order[a:b] for a, b in zip(starts, ends)}


class Cell:
    """One (base decomposition, x0) case: tables per slice, then the join."""

    def __init__(self, m, n1, u, d, x0, dic, env, log, psi_m=None, gamma=None, tables=None,
                 rank2="none"):
        self.m, self.n1, self.n2 = m, n1, m - n1
        self.N1 = 1 << n1
        self.x0 = x0
        self.u, self.d = u, d
        self.dic, self.env, self.log = dic, env, log
        a = alpha(ORBIT)
        self.psi = target(ORBIT, self.n2)
        self.psi_m = target(ORBIT, m) if psi_m is None else psi_m
        if gamma is None:
            gamma = {x: np.prod([a[(x >> (n1 - 1 - k)) & 1] for k in range(n1)]) for x in range(self.N1)}
        self.gamma = gamma
        self.TS = [TermShapes(env, ui, x0, n1) for ui in u]
        self.Vs = [t.slice_vectors() for t in self.TS]
        self.others = self.TS[0].others
        self.C = self.TS[0].ABSENT + 1
        self.rank2 = rank2
        self.tables = {}
        for x in self.others:
            beta = gamma[x] / gamma[x0]
            key = complex(np.round(beta, 9))
            if tables is not None and key in tables:
                self.tables[x] = tables[key]
            else:
                t0 = time.time()
                self.tables[x] = residual_table(self.Vs, d, beta * self.psi, dic, rank2)
                if tables is not None:
                    tables[key] = self.tables[x]
                log(f"    table ratio {beta:.4f}: exact {(self.tables[x] == 0).sum()}, "
                    f"stabilizer {(self.tables[x] == 1).sum()}"
                    + (f", rank-2 {rank2} {(self.tables[x] == 2).sum()}" if rank2 != "none" else "")
                    + f" of {self.tables[x].size} [{time.time() - t0:.1f}s]")
        self.completer = Completer(env, dic, n1)
        self._buckets = {}
        self._rank2 = {}
        self._masks = {}

    def mask(self, x, k):
        if (x, k) not in self._masks:
            self._masks[(x, k)] = self.tables[x] <= k
        return self._masks[(x, k)]

    def allowed(self, x, k):
        return np.argwhere(self.mask(x, k))

    def residual(self, x, codes):
        return (self.gamma[x] / self.gamma[self.x0]) * self.psi - sum(
            di * V[c] for di, V, c in zip(self.d, self.Vs, codes))

    def rank2_exact(self, x, codes):
        key = (x, codes)
        if key not in self._rank2:
            self._rank2[key] = self.dic.rank_le2(self.residual(x, codes))
        return self._rank2[key]

    def candidate(self, opts, K, multisets, soft, n_inv, stats, hits):
        """One assignment of shapes to the visible terms: the exact rank-2
        test on the k = 2 slices, then the exact completion."""
        stats["candidates"] += 1
        r = len(self.u)
        for x in soft:
            codes = tuple(int(self.TS[i].codes[opts[i], self.TS[i].slot[x]]) for i in range(r))
            if not self.rank2_exact(x, codes):
                return
        stats["rank2_survivors"] += 1
        vis = visible_vectors(self.TS, self.Vs, opts)
        res = self.psi_m - self.gamma[self.x0] * sum(di * s for di, s in zip(self.d, vis))
        for ms in multisets:
            try:
                inv = self.completer.complete(res, list(ms))
            except NotImplementedError as exc:
                self.log(f"  skipped completion: {exc}")
                continue
            if inv is not None:
                stats["completed"] += 1
                hits.append((opts, ms, vis, inv))
                self.log(f"  HIT: x0 {self.x0:0{self.n1}b} options {opts} shapes "
                         f"{[(j, sorted(p)) for j, p in ms]}")

    def run(self, n_inv, verbose=False):
        """Every coverage pattern of n_inv invisible terms."""
        kmax = 2 if self.rank2 == "exact" else 1
        stats = {"patterns": 0, "dead": 0, "pairs": 0, "candidates": 0,
                 "rank2_survivors": 0, "completed": 0, "unhandled": 0}
        hits = []
        _, pats = coverage_patterns(self.n1, self.x0, n_inv)
        r = len(self.u)
        for K, multisets in pats.items():
            stats["patterns"] += 1
            allowed, dead = {}, False
            for x, k in zip(self.others, K):
                if k <= kmax:
                    A = self.allowed(x, k)
                    if len(A) == 0:
                        dead = True
                        break
                    allowed[x] = A
            if dead:
                stats["dead"] += 1
                continue
            constrained = sorted(allowed, key=lambda x: len(allowed[x]))
            soft = [x for x, k in zip(self.others, K) if k == 2 and self.rank2 != "exact"]
            t_pat = time.time()
            n_cand0 = stats["candidates"]
            if len(constrained) < 2:
                if n_inv == 2 and len(constrained) == 1 and len(soft) == 2 and len(self.others) == 3:
                    self.run_shared_line(K, multisets, allowed, constrained[0], soft, stats, hits)
                else:
                    stats["unhandled"] += 1
                    self.log(f"      pattern k={K}: fewer than two constrained slices, not searched")
                    continue
            else:
                y, z = constrained[0], constrained[1]
                rest = constrained[2:]
                for i in range(r):
                    if (i, y, z) not in self._buckets:
                        self._buckets[(i, y, z)] = bucket(self.TS[i].codes[:, self.TS[i].slot[y]],
                                                          self.TS[i].codes[:, self.TS[i].slot[z]], self.C)
                Bk = [self._buckets[(i, y, z)] for i in range(r)]
                for ty in allowed[y]:
                    for tz in allowed[z]:
                        stats["pairs"] += 1
                        lists = [Bk[i].get((int(ty[i]), int(tz[i]))) for i in range(r)]
                        if any(l is None for l in lists):
                            continue
                        ok = np.ones([len(l) for l in lists], dtype=bool)
                        for x in rest:
                            k = K[self.others.index(x)]
                            cs = [self.TS[i].codes[lists[i], self.TS[i].slot[x]] for i in range(r)]
                            ok &= self.mask(x, k)[np.ix_(*cs)]
                        for idx in zip(*np.nonzero(ok)):
                            opts = tuple(int(lists[i][idx[i]]) for i in range(r))
                            self.candidate(opts, K, multisets, soft, n_inv, stats, hits)
            if verbose:
                self.log(f"      pattern k={K}: allowed {[len(allowed[x]) for x in constrained]} at "
                         f"{[f'{x:0{self.n1}b}' for x in constrained]}, "
                         f"{stats['candidates'] - n_cand0} candidates [{time.time() - t_pat:.1f}s]")
        return hits, stats

    def run_shared_line(self, K, multisets, allowed, y, soft, stats, hits):
        """Two invisible terms on the same line {xa, xb} of F_2^2 (one
        constrained slice y). The visible terms are pinned at x0 and y; the
        remaining options (129 each for four remaining qubits) are enumerated
        in full, 129^4 = 277 million, in chunks: the residuals at xa and xb
        must each pass the rank-2 support filter and the joint residual (a
        five-qubit vector, two stabilizer terms) its moduli filter; survivors
        get the exact rank-2 test and the completion."""
        r = len(self.u)
        if r != 4:
            stats["unhandled"] += 1
            self.log(f"      pattern k={K}: shared-line path needs four visible terms")
            return
        xa, xb = soft
        Ka = (self.gamma[xa] / self.gamma[self.x0]) * self.psi
        Kb = (self.gamma[xb] / self.gamma[self.x0]) * self.psi
        for ty in allowed[y]:
            stats["pairs"] += 1
            lists = [np.flatnonzero(self.TS[i].codes[:, self.TS[i].slot[y]] == ty[i]) for i in range(r)]
            Va = [self.d[i] * self.Vs[i][self.TS[i].codes[lists[i], self.TS[i].slot[xa]]] for i in range(r)]
            Vb = [self.d[i] * self.Vs[i][self.TS[i].codes[lists[i], self.TS[i].slot[xb]]] for i in range(r)]
            La = sum_table(Va[:2])
            Lb = sum_table(Vb[:2])
            Ra = sum_table(Va[2:])
            Rb = sum_table(Vb[2:])
            nR = Ra.shape[0]
            n1_, n3_ = len(lists[1]), len(lists[3])
            blk = max(1, 4_000_000 // (nR * self.env.N))
            total = La.shape[0] * nR
            surv = 0
            t0 = time.time()
            for s0 in range(0, La.shape[0], blk):
                ra = (Ka[None, None, :] - La[s0:s0 + blk, None, :] - Ra[None, :, :]).reshape(-1, self.env.N)
                rb = (Kb[None, None, :] - Lb[s0:s0 + blk, None, :] - Rb[None, :, :]).reshape(-1, self.env.N)
                ok = rank2_filter(ra, self.n2)
                ok &= rank2_filter(rb, self.n2)
                idx = np.flatnonzero(ok)
                if len(idx):
                    ok2 = rank2_filter_joint(np.concatenate((ra[idx], rb[idx]), axis=1))
                    idx = idx[ok2]
                for flat in idx:
                    l = s0 + flat // nR
                    rr = flat % nR
                    opts = (int(lists[0][l // n1_]), int(lists[1][l % n1_]),
                            int(lists[2][rr // n3_]), int(lists[3][rr % n3_]))
                    if surv == 0:
                        # the index decoding must reproduce the chunk residual
                        codes = tuple(int(self.TS[i].codes[opts[i], self.TS[i].slot[xa]]) for i in range(r))
                        if np.abs(self.residual(xa, codes) - ra[flat]).max() > 1e-9:
                            raise AssertionError("shared-line index decoding is inconsistent")
                    surv += 1
                    self.candidate(opts, K, multisets, soft, 2, stats, hits)
            self.log(f"      shared line {xa:02b},{xb:02b} pinned at {y:02b} code {tuple(int(c) for c in ty)}: "
                     f"{total} assignments, {surv} passed the filters [{time.time() - t0:.1f}s]")


def rank2_mode(n2, max_inv):
    """How the tables treat rank-2 residuals: exactly on two qubits (60
    states), otherwise not in the tables (candidates get the exact test)."""
    if max_inv < 2:
        return "none"
    return "exact" if n2 == 2 else "none"


def check_hit(vis, inv, psi_m):
    A = np.column_stack(vis + inv)
    c, *_ = np.linalg.lstsq(A, psi_m, rcond=None)
    return float(np.linalg.norm(A @ c - psi_m)), np.linalg.matrix_rank(A, tol=1e-8)


def x0_reps(n1):
    """One point per Hamming weight (permutations of the sliced qubits)."""
    reps = {}
    for x in range(1 << n1):
        reps.setdefault(bin(x).count("1"), x)
    return [reps[w] for w in sorted(reps)]


def base_from_witness(path, qubits, x0, m, n1, r):
    """Slice the witness along `qubits` (moved to the front) at x0; the base
    decomposition (unit slices, coefficients d_i) if exactly r terms are
    nonzero there, else None."""
    w = json.load(open(path))
    perm = list(qubits) + [q for q in range(m) if q not in qubits]
    U, d = [], []
    a = alpha(ORBIT)
    ax0 = np.prod([a[(x0 >> (n1 - 1 - k)) & 1] for k in range(n1)])
    terms = [term_vector(t, 2, m) for t in w["witness"]["terms"]]
    psi = target(ORBIT, m)
    c, *_ = np.linalg.lstsq(np.column_stack(terms), psi, rcond=None)
    for ci, v in zip(c, terms):
        vp = v.reshape([2] * m).transpose(perm).reshape(-1)
        s = vp.reshape(1 << n1, -1)[x0]
        if np.linalg.norm(s) > 1e-9:
            U.append(s / np.linalg.norm(s))
            d.append(ci * np.linalg.norm(s) / ax0)
    if len(U) != r:
        return None
    return U, np.array(d)


# -------------------------------------------------------------- controls ----

def _random_term(rng, m):
    from to_witness import _rref
    k = int(rng.integers(0, m + 1))
    while True:
        W = rng.integers(0, 2, size=(k, m))
        if k == 0 or len(_rref(W.tolist(), 2)[0]) == k:
            break
    Q = np.triu(rng.integers(0, 2, size=(k, k)), 1)
    term = {"k": k, "x0": rng.integers(0, 2, size=m).tolist(), "W": W.tolist(),
            "Q": Q.tolist(), "l": rng.integers(0, 4, size=k).tolist()}
    return term_vector(term, 2, m)


def control_shapes(log, rng, m, n1, samples):
    env = PauliEnv(m - n1)
    N1, N2 = 1 << n1, 1 << (m - n1)
    cache = {}
    dims = {}
    for _ in range(samples):
        v = _random_term(rng, m)
        B = v.reshape(N1, N2)
        nz = [x for x in range(N1) if np.linalg.norm(B[x]) > 1e-9]
        x0 = nz[int(rng.integers(0, len(nz)))]
        scale = np.linalg.norm(B[x0])
        u = B[x0] / scale
        key = (x0, tuple(np.round(u, 6)))
        if key not in cache:
            cache[key] = TermShapes(env, u, x0, n1)
        ts = cache[key]
        code = np.full(N1 - 1, ts.ABSENT, dtype=np.int8)
        for x in ts.others:
            r = B[x] / scale
            if np.linalg.norm(r) < 1e-9:
                continue
            found = None
            for k in range(N2):
                ph = phase_code(r, ts.T[k])
                if ph is not None:
                    found = 4 * k + ph
                    break
            if found is None:
                raise AssertionError("a slice is not a phased Pauli translate of the base slice")
            code[ts.slot[x]] = found
        if not (ts.codes == code[None, :]).all(axis=1).any():
            raise AssertionError(f"shape of a random stabilizer state missing ({len(nz)} slices)")
        dims[len(nz)] = dims.get(len(nz), 0) + 1
    log(f"control 1 (m={m}, n1={n1}): {samples} random stabilizer states, shapes found; "
        f"nonzero-slice counts {dict(sorted(dims.items()))}; {shape_count(n1, m - n1)} shapes per base slice")


def control_completion(log, rng, m, n1, dic, trials=30):
    env = PauliEnv(m - n1)
    N1, N2 = 1 << n1, 1 << (m - n1)
    comp = Completer(env, dic, n1)
    x0 = 0
    ok = tried = 0
    while ok < trials:
        n = int(rng.integers(1, 3))
        terms, shapes = [], []
        while len(terms) < n:
            v = _random_term(rng, m)
            B = v.reshape(N1, N2)
            nz = frozenset(x for x in range(N1) if np.linalg.norm(B[x]) > 1e-9)
            if x0 in nz or len(nz) == N1:
                continue
            terms.append(v * (rng.normal() + 1j * rng.normal()))
            shapes.append((len(nz).bit_length() - 1, nz))
        res = sum(terms)
        tried += 1
        try:
            out = comp.complete(res, shapes)
        except NotImplementedError:
            continue
        if out is None or np.abs(sum(out) - res).max() > 1e-6:
            raise AssertionError(f"completion failed on a sum of {n} invisible terms: {shapes}")
        ok += 1
    log(f"control 2 (m={m}, n1={n1}): {ok} random sums of 1 or 2 invisible terms recovered exactly "
        f"({tried} tried)")


def control_witness(log, path, m, n1, R, pairs=None, x0s=None):
    n2 = m - n1
    r = RANKS[n2]
    dic = Dict(n2)
    env = PauliEnv(n2)
    psi_m = target(ORBIT, m)
    found = 0
    tried = 0
    for qubits in (pairs or itertools.combinations(range(m), n1)):
        for x0 in (range(1 << n1) if x0s is None else x0s):
            base = base_from_witness(path, qubits, x0, m, n1, r)
            if base is None:
                continue
            tried += 1
            u, d = base
            cell = Cell(m, n1, u, d, x0, dic, env, log, rank2=rank2_mode(n2, R - r))
            hits, stats = cell.run(R - r, verbose=True)
            good = [h for h in hits if check_hit(h[2], h[3], psi_m)[0] < 1e-8]
            log(f"  witness slice qubits {qubits} x0 {x0:0{n1}b}: {len(good)} exact hits, stats {stats}")
            if not good:
                raise AssertionError(f"the known rank-{R} witness was not recovered at qubits {qubits}, x0 {x0}")
            found += 1
    log(f"control 3: witness {os.path.basename(path)} recovered from every minimal slice "
        f"({found} (qubit set, x0) cases)")


def ratio_tables(decs, n2, J, dic, env, log):
    """Exact and stabilizer residual counts over all code combinations of
    the visible terms at the slice ratio tan(pi/8)^j, per decomposition. A
    slice whose ratio has no exact combo cannot have k = 0, one with no
    stabilizer combo cannot have k <= 1, whatever the number of sliced
    qubits; the ratio of a slice x to x0 is tan(pi/8)^(|x| - |x0|)."""
    a = alpha(ORBIT)
    t = a[1] / a[0]
    psi = target(ORBIT, n2)
    totals = {}
    for di, (u, d) in enumerate(decs):
        TS = [TermShapes(env, ui, 0, 1) for ui in u]
        Vs = [ts.slice_vectors() for ts in TS]
        row = {}
        for j in range(-J, J + 1):
            tab = residual_table(Vs, d, (t ** j) * psi, dic)
            row[j] = (int((tab == 0).sum()), int((tab == 1).sum()))
            totals.setdefault(j, [0, 0])
            totals[j][0] += row[j][0]
            totals[j][1] += row[j][1]
        log(f"  dec {di}: " + " ".join(f"j={j}:{row[j]}" for j in range(-J, J + 1)))
    log("  totals (exact, stabilizer) over all decompositions: "
        + " ".join(f"j={j}:{tuple(totals[j])}" for j in range(-J, J + 1)))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--n1", type=int, default=2, help="number of sliced qubits")
    ap.add_argument("--rank", type=int, default=5, help="largest rank searched")
    ap.add_argument("--x0", default="reps",
                    help="'reps' (one per Hamming weight), 'all', or explicit bit strings such as 01,10")
    ap.add_argument("--decs", default=None, help="comma-separated stored decomposition indices")
    ap.add_argument("--verbose", action="store_true", help="log every coverage pattern")
    ap.add_argument("--ratio-tables", type=int, default=None, metavar="J",
                    help="only tabulate, for every stored decomposition, the exact and stabilizer "
                         "residual counts at the slice ratios tan(pi/8)^j for |j| <= J, then stop")
    ap.add_argument("--control", action="store_true", help="run the controls and stop")
    ap.add_argument("--witness", default=None, help="recover this witness from each of its minimal slices")
    ap.add_argument("--pairs", default=None, help="witness control: qubit sets, e.g. 0,1;2,3")
    args = ap.parse_args(argv[1:])
    os.nice(19)
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)

    if args.control:
        rng = np.random.default_rng(3)
        control_shapes(log, rng, 6, 2, 300)
        control_shapes(log, rng, 4, 2, 200)
        control_shapes(log, rng, 6, 3, 100)
        control_shapes(log, rng, 6, 4, 6)
        control_completion(log, rng, 6, 2, Dict(4))
        control_completion(log, rng, 4, 2, Dict(2))
        return 0
    if args.witness:
        pairs = None
        if args.pairs:
            pairs = [tuple(int(q) for q in grp.split(",")) for grp in args.pairs.split(";")]
        x0s = None if args.x0 in ("reps", "all") else [int(v, 2) for v in args.x0.split(",")]
        control_witness(log, args.witness, args.m, args.n1, args.rank, pairs, x0s)
        return 0

    m, n1 = args.m, args.n1
    n2 = m - n1
    r = RANKS[n2]
    dic = Dict(n2)
    env = PauliEnv(n2)
    decs, _ = load_decompositions(ORBIT, n2, r)
    if args.ratio_tables is not None:
        ratio_tables(decs, n2, args.ratio_tables, dic, env, log)
        return 0
    if args.x0 == "all":
        x0s = list(range(1 << n1))
    elif args.x0 == "reps":
        x0s = x0_reps(n1)
    else:
        x0s = [int(v, 2) for v in args.x0.split(",")]
    idx = range(len(decs)) if args.decs is None else [int(s) for s in args.decs.split(",")]
    log(f"|H>^{m} at rank <= {args.rank}, slicing {n1} qubits: {len(decs)} stored rank-{r} "
        f"decompositions of |H>^{n2}, x0 in {[f'{x:0{n1}b}' for x in x0s]}, "
        f"{shape_count(n1, n2)} shapes per visible term")
    psi_m = target(ORBIT, m)
    total, all_hits = {}, []
    for di in idx:
        u, d = decs[di]
        tables = {}
        for x0 in x0s:
            log(f"  dec {di} x0 {x0:0{n1}b}")
            cell = Cell(m, n1, u, d, x0, dic, env, log, tables=tables, rank2=rank2_mode(n2, args.rank - r))
            for n_inv in range(args.rank - r + 1):
                hits, stats = cell.run(n_inv, verbose=args.verbose)
                log(f"    rank {r + n_inv}: {stats}")
                for k, v in stats.items():
                    total[k] = total.get(k, 0) + v
                for h in hits:
                    res, rk = check_hit(h[2], h[3], psi_m)
                    all_hits.append((di, x0, r + n_inv, res, rk, h))
    log(f"total {total}; hits {len(all_hits)}")
    if all_hits:
        os.makedirs(RESULTS, exist_ok=True)
        for n, (di, x0, R, res, rk, (opts, ms, vis, inv)) in enumerate(all_hits):
            vecs = [[[float(z.real), float(z.imag)] for z in v] for v in vis + inv]
            path = os.path.join(RESULTS, f"two_qubit_slice_H_m{m}_rank{R}_hit{n}.json")
            with open(path, "w") as f:
                json.dump(vecs, f)
            log(f"hit {n}: dec {di} x0 {x0:0{n1}b} rank {R} residual {res:.1e} independent terms {rk}; wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
