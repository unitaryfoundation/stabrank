"""Two-qutrit slicing at the m=4 cells: a minimal two-qutrit slice.

Setting. Slice a decomposition psi^4 = sum_i c_i s_i of |M>^4 along qutrits
1 and 2: u_i^(x) = (<x| (x) I) s_i for x in F_3^2. Then sum_i c_i u_i^(x) =
alpha_x psi^2 with alpha_x = alpha_{x_1} alpha_{x_2}. The flat F_i of s_i
projects onto an affine flat pi(F_i) of F_3^2 of dimension 0, 1 or 2, so a
term has 1, 3 or 9 nonzero slices, all of the same flat dimension.

Structure of one term (the two-qutrit analogue of the slice-and-lift lemma).
Let s be a four-qutrit stabilizer state whose stabilizer group G projects to
the symplectic coordinates of qutrits 1, 2 with image A. The kernel of that
projection is the set of elements I (x) Q, an isotropic subgroup on two
qutrits, so |A| >= 9, and the X-part of A is the direction space of pi(F).

  nine slices (pi(F) = F_3^2): G contains g_1 = lambda_1 X^{e_1} Z^{b_1} (x) Q_1
  and g_2 = lambda_2 X^{e_2} Z^{b_2} (x) Q_2, and applying them to s gives
      u^(x_0 + x) = w^{q(x)} Q_2^{x_2} Q_1^{x_1} u^(x_0),
  with q a quadratic polynomial on F_3^2 vanishing at 0 (the lambda's and b's
  contribute linear terms, the Pauli reordering a cross term). Conversely,
  for any two-qutrit stabilizer state u, any Paulis Q_1, Q_2 and any such q,
      sum_x |x_0 + x> (x) w^{q(x)} Q_2^{x_2} Q_1^{x_1} u
  is a stabilizer state: it is C_2 C_1 (phi_q (x) u) with
  C_j = sum_t |t><t| (x) Q_j^t a controlled Pauli (a Clifford, since
  (X^a Z^c)^t = w^{h(t)} X^{at} Z^{ct} with h quadratic) and phi_q the
  full-support stabilizer state with phases w^{q}. The nine translates
  Q u, Q over the Pauli group mod Stab(u), are an orthonormal basis, so the
  slice at x is one of nine basis states times a cube root of unity; the
  class map x -> class(Q_2^{x_2} Q_1^{x_1}) is linear (81 maps L) and the
  phases are a quadratic (243 with q(0) = 0), so a term with base slice u
  has 81 x 243 = 19683 shapes, and 360 x 19683 = 7,085,880 of the 7,439,040
  four-qutrit stabilizer states have nine nonzero slices.

  three slices (pi(F) a line x_0 + <a>): u^(x_0 + t a) = w^{q(t)} Q^t u for
  one Pauli class and any q: F_3 -> F_3 with q(0) = 0 (81 shapes per
  direction); one slice: s = |x_0> (x) u.

Why the "all nine slices nonzero for every term" case is not finite. Each
slice is then a six-term (non-minimal) decomposition of psi^2, and the
coefficients c_i are continuous unknowns fixed by no slice, so the lift
equations are the full rank-6 problem over the 7.09 million nine-slice
states with free coefficients; nothing collapses to a matching over Pauli
classes. The finite reduction of the one-qutrit lemma needs a slice that is
a minimal decomposition, and that is the case this script covers.

Case M (minimal two-qutrit slice). Some slice x_0 with alpha_{x_0} != 0 has
exactly three nonzero terms. Since chi(psi^2) = 3, those three slices form a
minimal decomposition (d_i, u_i) of psi^2, one of the stored list up to the
unitary symmetry of psi^2 (which acts on qutrits 3, 4 and preserves the
slicing), and c_i = alpha_{x_0} d_i. The three visible terms have one of the
20008 shapes above (nine-slice, line through x_0, or point at x_0); the
remaining R - 3 terms vanish at x_0, so each is a line-type term on one of
the 8 lines missing x_0 or a point term at one of the 8 other points, with a
free coefficient. For each pattern of R - 3 such shapes, k_x = number of
invisible terms present at slice x, and the visible terms must leave at
every slice x != x_0 a residual alpha_x psi^2 - sum_visible of stabilizer
rank at most k_x. Slices with k_x = 0 need an exact match over Pauli classes
and phases (as `slice_lift.lifts`), slices with k_x = 1 a stabilizer
residual, k_x = 2 a rank-2 residual (tested against the 360-state
dictionary), and k_x = 3 nothing. With three invisible terms every pattern
leaves at least five slices with k_x <= 1 (checked by enumeration below),
which is what makes the search finite: the per-slice allowed combos are
tabulated over 28^3 codes (27 present shapes plus absent), candidates are
enumerated from the two most constraining slices and checked on the rest,
and survivors go to an exact completion of the invisible terms.

By the copy-permutation symmetry of psi^4 the bipartition {1,2} | {3,4} is
general, and the monomial Clifford symmetries of |M> (the affine
permutations of F_3 fixing |M> up to phase) together with the swap of
qutrits 1, 2 reduce x_0 to three classes for N and for H3.

Usage: two_qutrit_slice.py ORBIT [--rank 6] [--x0 all|reps] [--decs i,j,...] [--control]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (alpha, confirm_stabilizer, is_stabilizer_batch,  # noqa: E402
                    load_decompositions, target)
from rank_exclusion import dictionary  # noqa: E402

W3 = np.exp(2j * np.pi / 3)
TOL = 1e-7
PTS = [(a, b) for a in range(3) for b in range(3)]
DIRS = [(0, 1), (1, 0), (1, 1), (1, 2)]
ABSENT = 27


def pidx(x):
    return 3 * (x[0] % 3) + (x[1] % 3)


def pauli_apply(u, a, c):
    """(X^a Z^c u)[y + a] = w^{c.y} u[y] on two qutrits."""
    out = np.zeros(9, dtype=complex)
    for y in PTS:
        ph = (c[0] * y[0] + c[1] * y[1]) % 3
        out[pidx(((y[0] + a[0]) % 3, (y[1] + a[1]) % 3))] = W3 ** ph * u[pidx(y)]
    return out


def phase_code(v, ref):
    """ph in {0,1,2} with v = w^ph ref, or None."""
    j = np.flatnonzero(np.abs(ref) > 1e-9)[0]
    r = v[j] / ref[j]
    for ph in range(3):
        if abs(r - W3 ** ph) < 1e-6 and np.abs(v - W3 ** ph * ref).max() < 1e-7:
            return ph
    return None


class TermShapes:
    """Every four-qutrit stabilizer state whose slice at x0 is u, as per-slice
    codes: code = 3 * class + phase for a present slice (slice vector =
    w^phase T[class]), ABSENT otherwise."""

    def __init__(self, u, x0):
        self.u = u
        self.x0 = x0
        # class representatives: one Pauli (a, c) per distinct translate
        self.reps = []          # (a, c)
        self.T = []             # translate vectors
        self.cls_of = {}        # symplectic vector (a, c) -> (class, phase)
        for a in PTS:
            for c in PTS:
                v = pauli_apply(u, a, c)
                found = None
                for k, t in enumerate(self.T):
                    ph = phase_code(v, t)
                    if ph is not None:
                        found = (k, ph)
                        break
                if found is None:
                    self.T.append(v)
                    found = (len(self.T) - 1, 0)
                    self.reps.append((a, c))
                self.cls_of[(a, c)] = found
        if len(self.T) != 9 or self.cls_of[((0, 0), (0, 0))] != (0, 0):
            raise AssertionError("expected nine Pauli classes with the identity first")
        self.others = [x for x in PTS if x != x0]
        self.slot = {x: i for i, x in enumerate(self.others)}
        self.codes = self._all_codes()

    def _quadratics(self):
        out = []
        for a, b, c, d, e in itertools.product(range(3), repeat=5):
            out.append([(a * x[0] * x[0] + b * x[1] * x[1] + c * x[0] * x[1] + d * x[0] + e * x[1]) % 3
                        for x in PTS])
        return np.array(out, dtype=np.int64)          # (243, 9), indexed by point of the offset x

    def _all_codes(self):
        x0 = self.x0
        codes = []
        Q = self._quadratics()
        # nine-slice shapes
        for k1 in range(9):
            for k2 in range(9):
                a1, c1 = self.reps[k1]
                a2, c2 = self.reps[k2]
                cls = np.zeros(9, dtype=np.int64)
                f = np.zeros(9, dtype=np.int64)
                for x in PTS:
                    v = self.u.copy()
                    for _ in range(x[0]):
                        v = pauli_apply(v, a1, c1)
                    for _ in range(x[1]):
                        v = pauli_apply(v, a2, c2)
                    a = ((x[0] * a1[0] + x[1] * a2[0]) % 3, (x[0] * a1[1] + x[1] * a2[1]) % 3)
                    c = ((x[0] * c1[0] + x[1] * c2[0]) % 3, (x[0] * c1[1] + x[1] * c2[1]) % 3)
                    k, _ = self.cls_of[(a, c)]
                    ph = phase_code(v, self.T[k])
                    if ph is None:
                        raise AssertionError("a Pauli product is not a cube-root multiple of its class representative")
                    cls[pidx(x)] = k
                    f[pidx(x)] = ph
                # slice at x0 + x has code 3 cls[x] + (q(x) + f(x))
                block = np.full((Q.shape[0], 8), ABSENT, dtype=np.int64)
                for x in PTS:
                    if x == (0, 0):
                        continue
                    y = ((x0[0] + x[0]) % 3, (x0[1] + x[1]) % 3)
                    block[:, self.slot[y]] = 3 * cls[pidx(x)] + (Q[:, pidx(x)] + f[pidx(x)]) % 3
                codes.append(block)
        # line shapes through x0
        for a in DIRS:
            for k in range(9):
                ra, rc = self.reps[k]
                v1 = pauli_apply(self.u, ra, rc)
                v2 = pauli_apply(v1, ra, rc)
                k1, f1 = self.cls_of[(ra, rc)]
                k2, f2 = self.cls_of[((2 * ra[0]) % 3, (2 * ra[1]) % 3), ((2 * rc[0]) % 3, (2 * rc[1]) % 3)]
                assert phase_code(v1, self.T[k1]) == f1 and k1 == k
                f2 = phase_code(v2, self.T[k2])
                assert f2 is not None
                for q1 in range(3):
                    for q2 in range(3):
                        row = np.full(8, ABSENT, dtype=np.int64)
                        y1 = ((x0[0] + a[0]) % 3, (x0[1] + a[1]) % 3)
                        y2 = ((x0[0] + 2 * a[0]) % 3, (x0[1] + 2 * a[1]) % 3)
                        row[self.slot[y1]] = 3 * k1 + (q1 + f1) % 3
                        row[self.slot[y2]] = 3 * k2 + (q2 + f2) % 3
                        codes.append(row[None, :])
        # point shape
        codes.append(np.full((1, 8), ABSENT, dtype=np.int64))
        C = np.concatenate(codes, axis=0)
        C = np.unique(C, axis=0)
        return C

    def slice_vectors(self):
        """(28, 9): vector for each code (w^ph T[class]; zero for ABSENT)."""
        V = np.zeros((28, 9), dtype=complex)
        for k in range(9):
            for ph in range(3):
                V[3 * k + ph] = W3 ** ph * self.T[k]
        return V


def coverage_patterns(x0, n_inv):
    """Multisets of n_inv invisible shapes (lines missing x0, points != x0)
    with the k-vector over the 8 other slices."""
    others = [x for x in PTS if x != x0]
    shapes = []
    for a in DIRS:
        seen = set()
        for y in PTS:
            pts = frozenset(((y[0] + t * a[0]) % 3, (y[1] + t * a[1]) % 3) for t in range(3))
            if pts in seen or x0 in pts:
                continue
            seen.add(pts)
            shapes.append(("line", pts))
    for y in others:
        shapes.append(("point", frozenset([y])))
    assert len(shapes) == 16
    pats = {}
    for combo in itertools.combinations_with_replacement(range(len(shapes)), n_inv):
        k = tuple(sum(1 for s in combo if x in shapes[s][1]) for x in others)
        pats.setdefault(k, []).append(tuple(shapes[s] for s in combo))
    return shapes, pats


def residual_ranks(V, d, beta_psi, D2pairs, want_rank2):
    """For all 28^3 combos, rank class of beta psi - sum_i d_i V_i[c_i]:
    0 exact, 1 stabilizer, 2 in a two-state span (if want_rank2), 3 otherwise."""
    S = (d[0] * V[0])[:, None, None, :] + (d[1] * V[1])[None, :, None, :] + (d[2] * V[2])[None, None, :, :]
    R = (beta_psi[None, None, None, :] - S).reshape(-1, 9)
    rank = np.full(R.shape[0], 3, dtype=np.int8)
    nrm = np.linalg.norm(R, axis=1)
    rank[nrm < TOL] = 0
    cand = np.flatnonzero((nrm >= TOL) & is_stabilizer_batch(R, 3))
    for j in cand:
        if confirm_stabilizer(R[j], 3, 2) is not None:
            rank[j] = 1
    if want_rank2:
        rest = np.flatnonzero(rank == 3)
        Bs = D2pairs                     # (npairs, 9, 2) orthonormal bases
        for k in range(0, len(rest), 512):
            blk = rest[k:k + 512]
            Rb = R[blk]                                            # (n, 9)
            G = np.einsum("pjk,nj->pnk", Bs.conj(), Rb)            # (npairs, n, 2)
            proj = (np.abs(G) ** 2).sum(axis=2)                    # (npairs, n)
            full = (np.abs(Rb) ** 2).sum(axis=1)[None, :]
            hit = (full - proj).min(axis=0) < 1e-12 * full[0]
            rank[blk[hit]] = 2
    return rank.reshape(28, 28, 28), R.reshape(28, 28, 28, 9)


def pair_bases(D2):
    """Orthonormal bases of every two-state span of the two-qutrit dictionary."""
    N = D2.shape[1]
    ii, jj = np.triu_indices(N, 1)
    A = D2[:, ii]
    B = D2[:, jj]
    B2 = B - A * np.einsum("jn,jn->n", A.conj(), B)[None, :]
    B2 /= np.linalg.norm(B2, axis=0)[None, :]
    return np.stack([A.T, B2.T], axis=2)          # (npairs, 9, 2)


def monomial_x0_reps(orbit):
    """One x0 per orbit under the affine permutations of F_3 that fix |M>
    up to phase (on either qutrit) and the swap of the two qutrits."""
    a = alpha(orbit)
    perms = []
    for s in (1, 2):
        for t in range(3):
            perm = [(s * x + t) % 3 for x in range(3)]
            v = np.array([a[perm[x]] for x in range(3)])
            if abs(abs(np.vdot(a, v)) - 1) < 1e-9:
                perms.append(perm)
    seen, reps = set(), []
    for x in PTS:
        if x in seen:
            continue
        reps.append(x)
        orb = {x}
        frontier = [x]
        while frontier:
            y = frontier.pop()
            for p in perms:
                for z in ((p[y[0]], y[1]), (y[0], p[y[1]]), (y[1], y[0])):
                    if z not in orb:
                        orb.add(z)
                        frontier.append(z)
        seen |= orb
    return reps, perms


# ------------------------------------------------------------ completion ----

def _line_points(pts):
    """The three points of a line as (y, a) with pts = {y, y+a, y+2a}."""
    pts = sorted(pts)
    y = pts[0]
    for a in DIRS:
        if ((y[0] + a[0]) % 3, (y[1] + a[1]) % 3) in pts:
            return y, a
    raise AssertionError("not a line")


def _blocks(vec81):
    return vec81.reshape(9, 9)


def _cover_count(x, shapes):
    return sum(1 for _, p in shapes if x in p)


def _slice_ok(r, k):
    """Can the slice residual r be the sum of k nonzero stabilizer slices
    (k = 0: zero; k = 1: a stabilizer state; k >= 2: unconstrained here)."""
    if k == 0:
        return np.abs(r).max() < 1e-7
    if k == 1:
        return np.linalg.norm(r) > 1e-7 and confirm_stabilizer(r, 3, 2) is not None
    return True


def _line_terms(s, x, pts, res_blocks, rest):
    """Line-type terms with slice s at x on the line `pts`: the other two
    slices are w^{q} Q^t s. Pruned by the residual left at those slices
    against the shapes in `rest` that still cover them."""
    _, a = _line_points(pts)
    x1 = ((x[0] + a[0]) % 3, (x[1] + a[1]) % 3)
    x2 = ((x[0] + 2 * a[0]) % 3, (x[1] + 2 * a[1]) % 3)
    k1, k2 = _cover_count(x1, rest), _cover_count(x2, rest)
    for qa in PTS:
        for qc in PTS:
            v1 = pauli_apply(s, qa, qc)
            v2 = pauli_apply(v1, qa, qc)
            for q1 in range(3):
                if not _slice_ok(res_blocks[pidx(x1)] - W3 ** q1 * v1, k1):
                    continue
                for q2 in range(3):
                    if not _slice_ok(res_blocks[pidx(x2)] - W3 ** q2 * v2, k2):
                        continue
                    t = np.zeros(81, dtype=complex)
                    t[9 * pidx(x):9 * pidx(x) + 9] = s
                    t[9 * pidx(x1):9 * pidx(x1) + 9] = W3 ** q1 * v1
                    t[9 * pidx(x2):9 * pidx(x2) + 9] = W3 ** q2 * v2
                    yield t


def _point_term(s, x):
    t = np.zeros(81, dtype=complex)
    t[9 * pidx(x):9 * pidx(x) + 9] = s
    return t


def _split_two(r, D2):
    """Every way to write r as a s_i + b s_j with dictionary states: the
    candidate first summands a s_i."""
    N = D2.shape[1]
    rn = r / np.linalg.norm(r)
    Q = D2 - np.outer(rn, rn.conj() @ D2)
    nq = np.linalg.norm(Q, axis=0)
    out = []
    for i in range(N):
        if nq[i] < 1e-7:
            continue
        ov = np.abs(Q[:, i].conj() @ Q) / (nq[i] * np.maximum(nq, 1e-12))
        for j in np.flatnonzero((ov > 1 - 1e-8) & (np.arange(N) > i) & (nq > 1e-7)):
            A = np.column_stack((D2[:, i], D2[:, j]))
            c, *_ = np.linalg.lstsq(A, r, rcond=None)
            if np.linalg.norm(A @ c - r) < 1e-7:
                out.append(c[0] * D2[:, i])
                out.append(c[1] * D2[:, j])
    return out


def complete_invisible(res, shapes, D2, depth=0):
    """Exact completion: is res (81-dim) a combination of one stabilizer term
    per shape in `shapes` (lines missing x0 or points)? Returns the list of
    term vectors or None. Each term's slices are nonzero on its shape."""
    if not shapes:
        return [] if np.abs(res).max() < 1e-7 else None
    B = _blocks(res)
    # prefer a shape with a private slice: its slice residual is the term's slice
    order = sorted(range(len(shapes)),
                   key=lambda si: min(_cover_count(x, shapes) for x in shapes[si][1]))
    si = order[0]
    kind, pts = shapes[si]
    rest = shapes[:si] + shapes[si + 1:]
    x = min(pts, key=lambda p: _cover_count(p, shapes))
    k = _cover_count(x, shapes)
    r = B[pidx(x)]
    if np.linalg.norm(r) < 1e-7:
        return None
    if k == 1:
        if confirm_stabilizer(r, 3, 2) is None:
            return None
        firsts = [r]
    elif k == 2:
        firsts = _split_two(r, D2)
    else:
        raise NotImplementedError(f"every slice of every shape is covered {k} times")
    for s in firsts:
        terms = [_point_term(s, x)] if kind == "point" else _line_terms(s, x, pts, B, rest)
        for t in terms:
            out = complete_invisible(res - t, rest, D2, depth + 1)
            if out is not None:
                return [t] + out
    return None


def rank2_in_span(r, P2):
    """Is r in the span of two dictionary states (P2 = pair_bases)."""
    G = np.einsum("pjk,j->pk", P2.conj(), r)
    proj = (np.abs(G) ** 2).sum(axis=1)
    full = (np.abs(r) ** 2).sum()
    return bool((full - proj).min() < 1e-10 * full)


# ---------------------------------------------------------------- search ----

def visible_vectors(TS, V, opts):
    """The three visible four-qutrit terms (81-dim, unit base slice) for
    option indices `opts`."""
    out = []
    for i, (t, o) in enumerate(zip(TS, opts)):
        s = np.zeros(81, dtype=complex)
        s[9 * pidx(t.x0):9 * pidx(t.x0) + 9] = t.u
        for x in t.others:
            code = t.codes[o, t.slot[x]]
            if code != ABSENT:
                s[9 * pidx(x):9 * pidx(x) + 9] = V[i][code]
        out.append(s)
    return out


def run_cell(orbit, dec_index, x0, max_inv, D2, P2, log, psi4=None, gamma=None):
    """Case M at one stored decomposition and one base slice x0. `psi4` and
    `gamma` (slice x -> its psi^2 coefficient) default to |M>^4 and
    alpha_{x_1} alpha_{x_2}; the control passes another target of the form
    sum_x gamma_x |x> (x) psi^2."""
    a = alpha(orbit)
    psi2 = target(orbit, 2)
    psi4 = target(orbit, 4) if psi4 is None else psi4
    gamma = {x: a[x[0]] * a[x[1]] for x in PTS} if gamma is None else gamma
    decs, _ = load_decompositions(orbit, 2, 3)
    u, d = decs[dec_index]
    TS = [TermShapes(ui, x0) for ui in u]
    V = [t.slice_vectors() for t in TS]
    others = TS[0].others
    a_x0 = gamma[x0]
    ranks, resid = {}, {}
    for x in others:
        beta = gamma[x] / a_x0
        rk, R = residual_ranks(V, d, beta * psi2, None, False)
        ranks[x], resid[x] = rk, R
    counts = {x: (int((ranks[x] == 0).sum()), int((ranks[x] == 1).sum())) for x in others}
    log(f"  {orbit} dec {dec_index} x0 {x0}: per slice (exact, stabilizer) combos "
        + " ".join(f"{x}:{counts[x]}" for x in others))

    def bucket(i, y, z):
        b = {}
        cy, cz = TS[i].codes[:, TS[i].slot[y]], TS[i].codes[:, TS[i].slot[z]]
        for o in range(len(cy)):
            b.setdefault((int(cy[o]), int(cz[o])), []).append(o)
        return {k: np.array(v) for k, v in b.items()}

    bucket_cache = {}
    rank2_cache = {}

    def rank2(x, c):
        key = (x, c)
        if key not in rank2_cache:
            rank2_cache[key] = rank2_in_span(resid[x][c], P2)
        return rank2_cache[key]

    stats = {"patterns": 0, "dead_patterns": 0, "pairs": 0, "candidates": 0,
             "rank2_survivors": 0, "completed": 0}
    hits = []
    for ninv in range(max_inv + 1):
        _, pats = coverage_patterns(x0, ninv)
        if min(sum(1 for k in K if k <= 1) for K in pats) < 5:
            raise AssertionError("a coverage pattern leaves fewer than five slices with k <= 1")
        for K, multisets in pats.items():
            stats["patterns"] += 1
            allowed = {}
            dead = False
            for x, k in zip(others, K):
                if k <= 1:
                    A = np.argwhere(ranks[x] <= k)
                    if len(A) == 0:
                        dead = True
                        break
                    allowed[x] = A
            if dead:
                stats["dead_patterns"] += 1
                continue
            constrained = sorted(allowed, key=lambda x: len(allowed[x]))
            if len(constrained) < 2:
                raise AssertionError("fewer than two constrained slices")
            y, z = constrained[0], constrained[1]
            rest = constrained[2:]
            k2 = [x for x, k in zip(others, K) if k == 2]
            for i in range(3):
                if (i, y, z) not in bucket_cache:
                    bucket_cache[(i, y, z)] = bucket(i, y, z)
            B = [bucket_cache[(i, y, z)] for i in range(3)]
            for ty in allowed[y]:
                for tz in allowed[z]:
                    stats["pairs"] += 1
                    lists = [B[i].get((int(ty[i]), int(tz[i]))) for i in range(3)]
                    if any(l is None for l in lists):
                        continue
                    L1, L2, L3 = lists
                    ok = np.ones((len(L1), len(L2), len(L3)), dtype=bool)
                    for x in rest:
                        k = K[others.index(x)]
                        c1 = TS[0].codes[L1, TS[0].slot[x]]
                        c2 = TS[1].codes[L2, TS[1].slot[x]]
                        c3 = TS[2].codes[L3, TS[2].slot[x]]
                        ok &= (ranks[x] <= k)[c1[:, None, None], c2[None, :, None], c3[None, None, :]]
                    for i1, i2, i3 in zip(*np.nonzero(ok)):
                        opts = (int(L1[i1]), int(L2[i2]), int(L3[i3]))
                        stats["candidates"] += 1
                        good = True
                        for x in k2:
                            c = tuple(int(TS[i].codes[opts[i], TS[i].slot[x]]) for i in range(3))
                            if ranks[x][c] > 2 and not rank2(x, c):
                                good = False
                                break
                        if not good:
                            continue
                        stats["rank2_survivors"] += 1
                        vis = visible_vectors(TS, V, opts)
                        res = psi4 - a_x0 * sum(di * s for di, s in zip(d, vis))
                        for ms in multisets:
                            inv = complete_invisible(res, list(ms), D2)
                            if inv is not None:
                                stats["completed"] += 1
                                hits.append((opts, ms, vis, inv))
                                log(f"  HIT: dec {dec_index} x0 {x0} options {opts} shapes {ms}")
    log(f"  stats {stats}")
    return hits, stats


# -------------------------------------------------------------- controls ----

def _random_term(rng):
    """A random four-qutrit stabilizer state from a random (k, x0, W, Q, l),
    built by common.term_vector (independent of the shape formulas here)."""
    from common import term_vector
    from to_witness import _rref
    k = int(rng.integers(0, 5))
    while True:
        W = rng.integers(0, 3, size=(k, 4))
        if k == 0 or len(_rref(W.tolist(), 3)[0]) == k:
            break
    Q = np.triu(rng.integers(0, 3, size=(k, k)))
    term = {"k": k, "x0": rng.integers(0, 3, size=4).tolist(), "W": W.tolist(),
            "Q": Q.tolist(), "l": rng.integers(0, 3, size=k).tolist()}
    return term_vector(term, 3, 4)


def _slice_shape(v):
    """Nonzero slices of a four-qutrit vector along qutrits 1, 2."""
    B = _blocks(v)
    return [x for x in PTS if np.linalg.norm(B[pidx(x)]) > 1e-9]


def control(log, samples=400, seed=3):
    rng = np.random.default_rng(seed)
    D2 = dictionary(3, 2)
    # 1. every random stabilizer state is one of the enumerated shapes of its
    #    own slice at some base point, and the completion recovers random
    #    invisible-type sums exactly.
    shape_hits = {"nine": 0, "line": 0, "point": 0}
    cache = {}
    for _ in range(samples):
        v = _random_term(rng)
        nz = _slice_shape(v)
        x0 = nz[int(rng.integers(0, len(nz)))]
        B = _blocks(v)
        base = B[pidx(x0)]
        scale = np.linalg.norm(base)
        u = base / scale
        key = (x0, tuple(np.round(u, 6)))
        if key not in cache:
            cache[key] = TermShapes(u, x0)
        ts = cache[key]
        code = np.full(8, ABSENT, dtype=np.int64)
        for x in ts.others:
            r = B[pidx(x)] / scale
            if np.linalg.norm(r) < 1e-9:
                continue
            found = None
            for k in range(9):
                ph = phase_code(r, ts.T[k])
                if ph is not None:
                    found = 3 * k + ph
                    break
            if found is None:
                raise AssertionError("a slice is not a phased Pauli translate of the base slice")
            code[ts.slot[x]] = found
        if not (ts.codes == code[None, :]).all(axis=1).any():
            raise AssertionError(f"shape of a random stabilizer state missing from the enumeration ({len(nz)} slices)")
        shape_hits[{9: "nine", 3: "line", 1: "point"}[len(nz)]] += 1
    log(f"control 1: {samples} random four-qutrit stabilizer states, shapes found in the "
        f"enumeration by kind {shape_hits}")
    # 2. completion of random invisible-type sums (including shared slices)
    x0 = (0, 0)
    ok = 0
    tried = 0
    while ok < 40:
        n = int(rng.integers(1, 4))
        terms, shapes = [], []
        while len(terms) < n:
            v = _random_term(rng)
            nz = _slice_shape(v)
            if x0 in nz:
                continue
            kind = "line" if len(nz) == 3 else "point"
            terms.append(v * (rng.normal() + 1j * rng.normal()))
            shapes.append((kind, frozenset(nz)))
        # skip configurations the completion does not handle (a slice covered
        # three times by every shape's every slice cannot happen with <= 3 terms
        # unless all three shapes coincide)
        res = sum(terms)
        tried += 1
        try:
            out = complete_invisible(res, shapes, D2)
        except NotImplementedError:
            continue
        if out is None or np.abs(sum(out) - res).max() > 1e-6:
            raise AssertionError(f"completion failed on a genuine sum of {n} invisible terms: {shapes}")
        ok += 1
    log(f"control 2: {ok} random sums of 1 to 3 line/point terms recovered exactly ({tried} tried)")
    # 3. the driver finds the product decompositions of phi (x) psi^2 for a
    #    full-support two-qutrit stabilizer state phi (three nine-slice terms,
    #    no invisible term), for N.
    from stabrank_verify import ORBIT_P  # noqa: F401
    psi2 = target("N", 2)
    q = rng.integers(0, 3, size=5)
    phi = np.array([W3 ** ((q[0] * x[0] * x[0] + q[1] * x[1] * x[1] + q[2] * x[0] * x[1] + q[3] * x[0] + q[4] * x[1]) % 3)
                    for x in PTS]) / 3
    T = np.kron(phi, psi2)
    gamma = {x: phi[pidx(x)] for x in PTS}
    P2 = pair_bases(D2)
    hits, _ = run_cell("N", 0, (1, 2), 0, D2, P2, log, psi4=T, gamma=gamma)
    if not hits:
        raise AssertionError("control 3 found no product decomposition of phi (x) psi^2")
    for opts, ms, vis, inv in hits:
        A = np.column_stack(vis + inv)
        c, *_ = np.linalg.lstsq(A, T, rcond=None)
        assert np.linalg.norm(A @ c - T) < 1e-8
    log(f"control 3: phi (x) |N>^2 with phi = w^q full support: {len(hits)} exact three-term lifts found")
    return True


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit", choices=["N", "H3"])
    ap.add_argument("--rank", type=int, default=6, help="largest rank searched (3 + invisible terms)")
    ap.add_argument("--x0", default="reps", help="'reps' (one per monomial-symmetry orbit) or 'all'")
    ap.add_argument("--decs", default=None, help="comma-separated stored decomposition indices")
    ap.add_argument("--control", action="store_true", help="run the three controls and stop")
    args = ap.parse_args(argv[1:])
    os.nice(19)
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)

    if args.control:
        control(log)
        return 0

    D2 = dictionary(3, 2)
    P2 = pair_bases(D2)
    decs, _ = load_decompositions(args.orbit, 2, 3)
    reps, perms = monomial_x0_reps(args.orbit)
    x0s = PTS if args.x0 == "all" else reps
    idx = range(len(decs)) if args.decs is None else [int(s) for s in args.decs.split(",")]
    log(f"{args.orbit}: {len(decs)} stored rank-3 decompositions of |M>^2, x0 in {x0s} "
        f"(monomial symmetries {perms}), ranks 3..{args.rank}")
    total = {}
    all_hits = []
    for x0 in x0s:
        for di in idx:
            hits, stats = run_cell(args.orbit, di, x0, args.rank - 3, D2, P2, log)
            all_hits += hits
            for k, v in stats.items():
                total[k] = total.get(k, 0) + v
    log(f"total {total}; hits {len(all_hits)}")
    if all_hits:
        import json
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(out, exist_ok=True)
        for n, (opts, ms, vis, inv) in enumerate(all_hits):
            vecs = [[[float(z.real), float(z.imag)] for z in v] for v in vis + inv]
            path = os.path.join(out, f"two_qutrit_slice_{args.orbit}_m4_hit{n}.json")
            with open(path, "w") as f:
                json.dump(vecs, f)
            log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
