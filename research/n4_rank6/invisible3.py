"""Stages (beta') and (gamma) of the rank-6 exclusion of |N>^4: the base
point (2, 2) along qutrits 1, 2 with five or four visible terms and one or
two invisible terms whose flats are given (docs/notes/n4_rank6_design.md,
section 6; docs/notes/n4_rank6_exclusion.md, section 4).

Setting. Slice psi_4 = sum_{i=1}^6 c_i s_i along qutrits 1, 2. At the base
point X0 = (2, 2) the visible terms' slices form a full k-multiset of
psi_2 = |N>^2 (the base, k in {4, 5}) with coefficients d_i = c_i /
alpha_{X0}. Each invisible term vanishes at X0 and is nonzero exactly on
one of the 16 affine flats of F_3^2 missing X0: the 8 other points and the
8 lines not through X0. Its slice at the first processed point of its flat
is c v for a two-qutrit stabilizer state v (a dictionary state); along a
line the two other slices are w^{q(t)} Q^t v for a Pauli Q and a quadratic
q (the two-qutrit structure lemma for a line term): 27 phased translates of
v at the second point (code 3 k + l, class k of Q, phase l), and at the
third point the three phases of the class of Q^2.

What the matcher assumes and what it enumerates. The flats of the
invisible terms are parameters (every flat and every flat multiset is run
by the pipeline); nothing is assumed about the flats of the visible terms:
each visible term's codes over the eight nonzero offsets range over every
shape of the structure lemma with its base slice (a plane with 19,683
shapes, a coordinate line, a diagonal line or the point X0: 20,008 rows,
`shape_table`), the rows alive are restricted by every code chosen, and
the codes offered at a point are the values the alive rows take there. So
the presence pattern is a subspace through X0 with the lemma's composite
codes by construction. A repeated base state is a block in the sense of
the qutrit matcher: its copies contribute a vector of the span of at most
g Pauli translates at every slice, the equations are projected onto the
annihilator of the chosen translates, and the copies are rebuilt at the
end by matcher.reconstruct_block from the residual coordinates at the
eight offsets, with a coefficient family carried as unknowns for one
block. A dependent base has an affine coefficient family d = d0 + K lambda
over F_65521, F_2013265921 and C (matcher.family_from), restricted by
every slice equation.

The coefficients of the invisible terms join the family when the terms
are born: the family gains one free coordinate per born term and the
slice equation at the birth point restricts it (Family.restrict), so a
pinned base pins the coefficient at once and a base with parameters
carries them along; a family that still has parameters at a later point
goes through the Laplace-feature dense solve (dense_solve for one or two
parameters, the Python reference beyond) like every other slice of this
project.

The slice equations, point by point, in the order that keeps the number
of invisible terms whose slice is still unknown ("fresh") as small as
possible (the points where no invisible term is present first, then by
fewest fresh terms):

  exact   no fresh term: sum_i d_i w_i + block + sum_j c_j t_j = rhs over
          the visible options, the block's translate sets and, for every
          born invisible term present, its options (27 at the second point
          of its line, 3 at the third), by matcher.solve_slice3 (meet in
          the middle mod 65521 for a pinned family, the dense solve
          otherwise; every candidate decided over the three fields);
  scan    one fresh term: the residual after the visible terms, the block
          and the born terms must be c v for a dictionary state v. With a
          pinned family every option combination is formed at once mod
          65521, the residual projected onto the annihilator of the
          block's translates, normalized and looked up in the table of
          the normalized projected dictionary states; with parameters the
          residual is an affine family r_0 + lambda r_1 and the states in
          its span mod 65521 are the candidates. A zero projected residual
          with no translate chosen means the fresh term is absent at a
          point of its flat, another flat's configuration, and is not a
          solution here; with translates chosen the states inside their
          span are the candidates (the fresh term's slice can hide in the
          block's span; its coefficient then stays a family parameter and
          the block reconstruction decides it). Every candidate is decided
          by the exact restriction of the extended family;
  scan2   two fresh terms (stage (gamma), when both flats share their
          first processed point: two equal point flats or two equal line
          flats): the residual r must be c_4 v_4 + c_5 v_5. Distinct
          states are found by a projective hash (v and w span r exactly
          when their images in F^9 / span(r) are parallel), every
          collision decided exactly mod 2013265921 and over C with the two
          required to agree. For two line terms the states may coincide:
          r = D v (the same slice at the first point, D = c_4 + c_5 the
          merged coefficient) is carried as two born terms with the same
          state whose split is a family parameter, decided at the line's
          other points; r = 0 (a cancelling pair, c_5 = -c_4, the common
          state v unknown) is carried to the line's second point, where
          the residual is scanned for a stabilizer residual (both copies
          in one Pauli class of v, phases distinct) or a rank-2 residual
          whose two states lie in one Pauli orbit with coefficients
          cancelling up to a cube root, or vanishes again (the pair agrees
          at two points and must differ at the third, where the residual
          is a stabilizer residual); the common slice at the first point
          runs over the nine class translates. Two equal point flats with
          equal states or a zero residual would be the same term twice or
          two cancelling copies of one term and are rejected.

Every surviving state is assembled into six terms (the copies of a block
through reconstruct_block) and confirmed as in the rank-5 pipeline
(Matcher.confirm: residual against psi_4, rank, independence, nonzero
coefficients, the span condition mod 2013265921); batch.py re-decides
every hit again from its phase codes (common.decide_terms).

Refusals and undecided runs. A base whose family is empty or has a dead
ordinary coefficient is refused (the lists exclude such multisets, so a
refusal fails the aggregate). Raised as UnpinnedFamily, which batch.py
records as undecided and the aggregate fails on: two fresh terms at a
point of a base with a block or with an unpinned family (the rank-2 scan
is written for a pinned family without blocks), a family left with
parameters at the end on a base with two or more blocks, the strict
coordinate solve of the block reconstruction (dependent translates of two
blocks, a residual outside the span), a residual whose complex and
modular decisions disagree, an invisible term not determined on its whole
flat, and any configuration the enumeration above does not cover.
BudgetExceeded is raised at the dense solve's candidate cap and at the
batch deadline.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

import common  # noqa: E402  (research/n4_rank6/common.py)
from common import FLATS, OFFSETS, P1, P2, X0, add, offset_index, pidx  # noqa: E402
from matcher import (COMP, E1, E2, W3P, Block, BudgetExceeded, Family, TermOpts, UnpinnedFamily,  # noqa: E402
                     _affine_solve_C, _ProjectorCache, pauli_apply, rank_mod, reconstruct_block, restrict,
                     slice_system, solve_slice3, vector_target)
from cover_census import _canon_rows, _reduce  # noqa: E402
from slice_cover import _affine_solve_mod  # noqa: E402

TOL = 1e-7
N_SHAPES = 20008                                     # two qutrits: 19,683 planes + 4 x 81 lines + 1 point
SCAN_CHUNK = 200_000


def shape_table(o):
    """Every valid code assignment of a visible term with base slice o.u
    over the eight nonzero offsets, in OFFSETS order (e_1, e_2, then COMP),
    as an (N_SHAPES, 8) int8 array: the coordinate codes (c_1, c_2) run
    over the 28 x 28 pairs and the composite codes over the shapes of the
    structure lemma for that pair (TermOpts.composite_rows)."""
    A = o.absent
    rows = []
    for c1 in range(A + 1):
        for c2 in range(A + 1):
            comp = o.composite_rows(c1, c2)
            full = np.empty((len(comp), 8), dtype=np.int64)
            full[:, 0] = c1
            full[:, 1] = c2
            full[:, 2:] = comp
            rows.append(full)
    T = np.concatenate(rows)
    K = o.nclass
    want = (3 * K) ** 2 * 27 + 2 * (3 * K) * 3 + 1 + 2 * K * 9      # planes, coordinate lines, point, diagonal lines
    if len(T) != want or len(np.unique(T, axis=0)) != want:
        raise AssertionError(f"{len(T)} shapes, expected {want} distinct")
    return T.astype(np.int8)


def _extend_free(fam, n):
    """The family with n new free coordinates appended (the coefficients of
    n invisible terms about to be born)."""
    parts = []
    for fld in range(3):
        d0, K = fam.parts[fld]
        r, k = K.shape
        d0n = np.concatenate([d0, np.zeros(n, dtype=d0.dtype)])
        Kn = np.zeros((r + n, k + n), dtype=K.dtype)
        Kn[:r, :k] = K
        Kn[r:, k:] = np.eye(n, dtype=K.dtype)
        parts.append((d0n, Kn))
    return Family(parts)


def _extend_pinned(fam, triples):
    """The family with the pinned coefficients `triples` (mod P1, mod P2,
    over C) appended."""
    parts = []
    for fld, p in ((0, P1), (1, P2), (2, None)):
        d0, K = fam.parts[fld]
        vals = [t[fld] for t in triples]
        if p is None:
            d0n = np.concatenate([d0, np.asarray(vals, dtype=complex)])
        else:
            d0n = np.concatenate([d0, np.asarray([int(v) % p for v in vals], dtype=np.int64)])
        Kn = np.concatenate([K, np.zeros((len(vals), K.shape[1]), dtype=K.dtype)])
        parts.append((d0n, Kn))
    return Family(parts)


def _zero_modulo(F, rows, basis):
    """Mask of the rows (n, d) over F_p that lie in the span of the basis
    vectors (a list of length-d vectors), by reduction."""
    rows = np.asarray(rows, dtype=np.int64) % F.p
    B = np.asarray(basis, dtype=np.int64).reshape(-1, rows.shape[1]) % F.p
    while len(B):
        b, B = B[0], B[1:]
        if not b.any():
            continue
        rows, _ = _reduce(F, rows, b)
        if len(B):
            B, _ = _reduce(F, B, b)
    return ~rows.any(axis=1)


class _Table:
    __slots__ = ("f", "keys", "order", "PVn", "pos_of", "inside", "PV")


class InvisibleMatcher3:
    """Every decomposition of a target whose slice at X0 along qutrits 1, 2
    has exactly the terms of `cover` visible, with the invisible terms on
    the given flats. `M` is a qutrit matcher.Matcher over the two-qutrit
    dictionary (its option tables, fields and block routines are reused);
    the target is given per run (psi_4, or a planted sum)."""

    def __init__(self, M, seed=31, verbose=False, max_cand=2_000_000):
        self.M = M
        self.n2 = M.n2
        self.dim = 3 ** M.n2
        self.F1, self.F2 = M.F1, M.F2
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.max_cand = max_cand
        self.deadline = None                 # epoch seconds; checked between points
        self._shapes = {}                    # dictionary index -> shape table
        self._tables = {}                    # (block states, translate sets) -> projected dictionary table
        self._orbit = None                   # dictionary index -> Pauli orbit id
        self._hash_f = None

    # -- shared pieces --
    def options(self, idx):
        return self.M.options(int(idx))

    def shapes(self, idx):
        idx = int(idx)
        if idx not in self._shapes:
            self._shapes[idx] = shape_table(self.options(idx))
        return self._shapes[idx]

    def dbl(self, o, k):
        """The class of Q_k^2 for the class k of Q_k."""
        return int(o.cls[k, 0, pidx((2, 0))])

    def orbit_id(self, v):
        """The Pauli orbit of a dictionary state (the least index among its
        class images), computed once for the whole dictionary."""
        if self._orbit is None:
            M = self.M
            ids = np.full(M.N, -1, dtype=np.int64)
            for i in range(M.N):
                if ids[i] >= 0:
                    continue
                members = set()
                for a in itertools.product(range(3), repeat=self.n2):
                    for c in itertools.product(range(3), repeat=self.n2):
                        members.add(M.index_of(pauli_apply(M.C[:, i], a, c, self.n2)))
                root = min(members)
                for m in members:
                    ids[m] = root
            self._orbit = ids
        return int(self._orbit[int(v)])

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _check_deadline(self, where):
        if self.deadline is not None and time.time() > self.deadline:
            raise BudgetExceeded(f"{where}: past the batch deadline")

    # -- the projected dictionary tables and the one-fresh-term scan --
    def _table(self, blocks, Ssel, Ps):
        key = tuple((b.idx, tuple(S)) for b, S in zip(blocks, Ssel))
        if key in self._tables:
            return self._tables[key]
        M = self.M
        if Ps is None:
            PV = np.ascontiguousarray(M.U1.T % P1)
            PVC = M.C
        else:
            PV = (Ps[0] @ M.U1.T) % P1
            PVC = Ps[2] @ M.C
        inside = np.linalg.norm(PVC, axis=0) < TOL
        zero1 = ~np.any(PV != 0, axis=0)
        if not np.array_equal(inside, zero1):
            raise UnpinnedFamily("a dictionary state's projection vanishes mod 65521 but not over C, or the "
                                 "reverse; the scan table is unreliable for this translate set")
        valid = np.flatnonzero(~inside)
        PVv = PV[:, valid]
        nz = PVv != 0
        j = np.argmax(nz, axis=0)
        piv = PVv[j, np.arange(len(valid))]
        inv = self.F1.inv(piv)
        PVn = (PVv * inv[None, :]) % P1
        f = self.rng.integers(1, P1, size=PV.shape[0])
        keys = (f @ PVn) % P1
        order = np.argsort(keys, kind="stable")
        tab = _Table()
        tab.f, tab.keys, tab.order, tab.PVn, tab.pos_of = f, keys[order], valid[order], PVn, valid
        tab.inside = [int(v) for v in np.flatnonzero(inside)]
        tab.PV = PV
        if len(self._tables) >= 512:
            self._tables.clear()
        self._tables[key] = tab
        return tab

    @staticmethod
    def _grid(sizes, start, stop):
        """Combination index tuples start..stop of the product of sizes, as
        a (stop - start, r) array (row-major, last index fastest)."""
        idx = np.arange(start, stop, dtype=np.int64)
        out = np.empty((len(idx), len(sizes)), dtype=np.int64)
        for i in range(len(sizes) - 1, -1, -1):
            out[:, i] = idx % sizes[i]
            idx //= sizes[i]
        return out

    def _residual_rows(self, popts, dv, prhs, grid):
        """The residuals rhs - sum_i dv_i w_i mod P1 of the combinations
        `grid` (n, r), as an (n, dim') array."""
        R = np.tile(prhs % P1, (len(grid), 1))
        for i, o in enumerate(popts):
            R = (R - (int(dv[i]) % P1) * o[grid[:, i]]) % P1
        return R

    def _scan(self, popts, dv, prhs, table):
        """Combinations whose projected residual is zero, and the
        (combination, dictionary state) pairs whose normalized residual
        equals the state's normalized projection. A superset of the exact
        solutions."""
        sizes = [len(o) for o in popts]
        total = int(np.prod(sizes)) if sizes else 1
        zero, matches = [], []
        for s in range(0, total, SCAN_CHUNK):
            grid = self._grid(sizes, s, min(total, s + SCAN_CHUNK))
            R = self._residual_rows(popts, dv, prhs, grid)
            nz = R != 0
            has = nz.any(axis=1)
            zero.extend(tuple(int(c) for c in grid[q]) for q in np.flatnonzero(~has))
            rows = np.flatnonzero(has)
            if not len(rows):
                continue
            j = np.argmax(nz[rows], axis=1)
            piv = R[rows, j]
            inv = self.F1.inv(piv)
            Rn = (R[rows] * inv[:, None]) % P1
            keys = (Rn @ table.f) % P1
            lo = np.searchsorted(table.keys, keys, side="left")
            hi = np.searchsorted(table.keys, keys, side="right")
            for q in np.flatnonzero(hi > lo):
                for pos in range(lo[q], hi[q]):
                    v = int(table.order[pos])
                    col = int(np.searchsorted(table.pos_of, v))
                    if np.array_equal(Rn[q], table.PVn[:, col]):
                        matches.append((tuple(int(c) for c in grid[rows[q]]), v))
        return zero, matches

    def _scan_family(self, popts, d0v, Kv, prhs, table):
        """The one-fresh-term scan with parameters left in the family: per
        combination the residual is r_0 + lambda r_1 (+ ...), and the
        dictionary states whose projection lies in span(r_0, r_1, ...) mod
        P1 are the candidates (the states with zero projection included).
        A superset of the exact solutions."""
        sizes = [len(o) for o in popts]
        total = int(np.prod(sizes)) if sizes else 1
        out = []
        PVT = np.ascontiguousarray(table.PV.T)                         # (N, dim')
        for s in range(0, total, SCAN_CHUNK):
            grid = self._grid(sizes, s, min(total, s + SCAN_CHUNK))
            R0 = self._residual_rows(popts, d0v, prhs, grid)
            Rks = []
            for j in range(Kv.shape[1]):
                Rj = np.zeros_like(R0)
                for i, o in enumerate(popts):
                    Rj = (Rj - (int(Kv[i, j]) % P1) * o[grid[:, i]]) % P1
                Rks.append(Rj)
            for q in range(len(grid)):
                basis = [R0[q]] + [Rj[q] for Rj in Rks]
                mask = _zero_modulo(self.F1, PVT, basis)
                combo = tuple(int(c) for c in grid[q])
                out.extend((combo, int(v)) for v in np.flatnonzero(mask))
        return out

    def _residuals(self, arrays, dvs, combo, rhs):
        """rhs minus the ordinary terms' contribution, in the three fields."""
        out = []
        for fld, p in ((0, P1), (1, P2), (2, None)):
            acc = rhs[fld].copy()
            for i, c in enumerate(combo):
                if p is None:
                    acc = acc - dvs[fld][i] * arrays[i][fld][c]
                else:
                    acc = (acc - (int(dvs[fld][i]) % p) * arrays[i][fld][c]) % p
            out.append(acc)
        return out

    # -- the two-fresh-term scan (stage gamma, no blocks, pinned family) --
    def _solve_states(self, res, states):
        """The coefficients of the residual on the dictionary states
        `states` in the three fields: unique with every entry nonzero in
        all three, or None when the residual is not in their span; the
        complex and P2 decisions must agree."""
        M = self.M
        solC = _affine_solve_C(M.C[:, states], res[2])
        sol1 = _affine_solve_mod(M.U1[states].T % P1, res[0], P1)
        sol2 = _affine_solve_mod(M.U2[states].T % P2, res[1], P2)
        if (solC is not None) != (sol2 is not None):
            raise UnpinnedFamily("the complex and modular (2013265921) decisions of a residual disagree")
        if solC is None:
            return None
        if solC[1].shape[1] or sol2[1].shape[1] or sol1 is None or sol1[1].shape[1]:
            raise UnpinnedFamily("a residual on dependent dictionary states")
        if np.any(np.abs(solC[0]) < 1e-9) or np.any(sol2[0] == 0):
            return None
        return [(int(sol1[0][t]), int(sol2[0][t]), complex(solC[0][t])) for t in range(len(states))]

    def _rank2_scan(self, arrays, dvs, rhs, stats, tag, chunk=96):
        """Every option combination of the ordinary terms whose residual is
        zero ('zero'), a nonzero multiple of a dictionary state ('stab',
        (v, c)), or a combination c_v v + c_w w of two distinct dictionary
        states with both coefficients nonzero ('rank2', (v, w, c_v, c_w)); a
        combination may appear once as 'stab' and several times as 'rank2'.
        The rank-2 test: v and w span r exactly when their images in
        F^9 / span(r) are parallel, so their projective keys (ratios of
        three random functionals) agree; every collision is decided exactly
        mod 2013265921 and over C, and the two must agree."""
        M = self.M
        N, dim = M.N, self.dim
        sizes = [len(a[0]) for a in arrays]
        total = int(np.prod(sizes)) if sizes else 1
        if total > 2_000_000:
            raise BudgetExceeded(f"{total} option combinations at a two-fresh-term point")
        grid = self._grid(sizes, 0, total)
        R = self._residual_rows([a[0] for a in arrays], dvs[0], rhs[0] % P1, grid)
        if self._hash_f is None:
            self._hash_f = [self.rng.integers(1, P1, size=dim) for _ in range(3)]
        f1, f2, f3 = self._hash_f
        U1 = M.U1
        out = []
        n_coll = 0

        def decide_pair(combo, res, v, w):
            r_mod = rank_mod(np.array([res[1], M.U2[v], M.U2[w]], dtype=np.int64), P2)
            r_num = int(np.linalg.matrix_rank(np.column_stack([res[2], M.C[:, v], M.C[:, w]]), tol=1e-8))
            if (r_mod <= 2) != (r_num <= 2):
                raise UnpinnedFamily("modular and numeric rank of a rank-2 residual disagree")
            if r_mod > 2:
                return
            sol = self._solve_states(res, [v, w])
            if sol is not None:
                out.append((combo, "rank2", (v, w, sol[0], sol[1])))

        for s in range(0, len(R), chunk):
            Rb = R[s:s + chunk]
            idx0 = np.arange(s, s + len(Rb))
            zero_rows = ~(Rb != 0).any(axis=1)
            for q in np.flatnonzero(zero_rows):
                combo = tuple(int(c) for c in grid[idx0[q]])
                res = self._residuals(arrays, dvs, combo, rhs)
                if np.linalg.norm(res[2]) < TOL and not np.any(res[1]):
                    out.append((combo, "zero", None))
            keep = np.flatnonzero(~zero_rows)
            if not len(keep):
                continue
            Rb, idxb = Rb[keep], idx0[keep]
            c = np.argmax(Rb != 0, axis=1)
            Rc = Rb[np.arange(len(Rb)), c]
            inv_rc = self.F1.inv(Rc)
            Vc = U1[:, c].T                                                  # (b, N)
            coef = (Vc * inv_rc[:, None]) % P1
            img = (U1[None, :, :] - coef[:, :, None] * Rb[:, None, :]) % P1   # (b, N, dim)
            k1 = (img @ f1) % P1
            k2 = (img @ f2) % P1
            k3 = (img @ f3) % P1
            img_zero = ~img.any(axis=2)                                      # v in span(r): a stabilizer residual
            for q, v in zip(*np.nonzero(img_zero)):
                combo = tuple(int(cc) for cc in grid[idxb[q]])
                res = self._residuals(arrays, dvs, combo, rhs)
                sol = self._solve_states(res, [int(v)])
                if sol is not None:
                    out.append((combo, "stab", (int(v), sol[0])))
            kz = (k1 == 0) & ~img_zero
            kzz = kz & (k2 == 0)
            inv1 = self.F1.inv(np.where(k1 == 0, 1, k1))
            inv2 = self.F1.inv(np.where(k2 == 0, 1, k2))
            key = ((k2 * inv1) % P1) * P1 + (k3 * inv1) % P1
            key = np.where(kz, P1 * P1 + (k3 * inv2) % P1, key)
            key = np.where(img_zero | kzz, -1 - np.arange(N)[None, :], key)
            order = np.argsort(key, axis=1, kind="stable")
            sk = np.take_along_axis(key, order, axis=1)
            eq = (sk[:, 1:] == sk[:, :-1]) & (sk[:, 1:] >= 0)
            for q in np.flatnonzero(eq.any(axis=1)):
                combo = tuple(int(cc) for cc in grid[idxb[q]])
                res = self._residuals(arrays, dvs, combo, rhs)
                l = 0
                while l < N - 1:
                    if not eq[q, l]:
                        l += 1
                        continue
                    e = l + 1
                    while e < N - 1 and eq[q, e]:
                        e += 1
                    run = [int(x) for x in order[q, l:e + 1]]
                    for v, w in itertools.combinations(run, 2):
                        n_coll += 1
                        decide_pair(combo, res, v, w)
                    l = e + 1
            for q in np.flatnonzero(kzz.any(axis=1)):
                combo = tuple(int(cc) for cc in grid[idxb[q]])
                res = self._residuals(arrays, dvs, combo, rhs)
                for v in np.flatnonzero(kzz[q]):
                    for w in range(N):
                        if w != int(v):
                            decide_pair(combo, res, int(v), w)
        stats[tag + "_collisions"] += n_coll
        stats[tag + "_solutions"] += len(out)
        return out

    # -- the cancelling pair --
    @staticmethod
    def _cube_inverse(l4, l5, F):
        """(w^{l4} - w^{l5})^{-1} mod F.p."""
        d = (int(F.wpow[l4]) - int(F.wpow[l5])) % F.p
        return pow(d, F.p - 2, F.p)

    def _cancel_pair(self, a, l4, l5):
        """Coefficient triples (c_4, c_5 = -c_4) with c_4 (w^{l4} - w^{l5}) = a
        for the residual coefficient a (a triple)."""
        c4C = a[2] / (W3P[l4] - W3P[l5])
        c41 = (int(a[0]) * self._cube_inverse(l4, l5, self.F1)) % P1
        c42 = (int(a[1]) * self._cube_inverse(l4, l5, self.F2)) % P2
        return (c41, c42, c4C), ((-c41) % P1, (-c42) % P2, -c4C)

    def _class_states(self, w):
        """The dictionary indices of the nine class translates Q_k w."""
        ow = self.options(w)
        return [self.M.index_of(ow.T[k]) for k in range(ow.nclass)]

    # -- the run --
    def _setup(self, cover, target, stats):
        M = self.M
        distinct = sorted(set(int(u) for u in cover))
        mult = {u: list(cover).count(u) for u in distinct}
        b1, b2, bC = target.rhs(X0)
        fam = matcher_family(M, distinct, b1, b2, bC)
        if fam is None:
            stats["refused"] = True
            return None
        blocks = []
        for i, u in enumerate(distinct):
            if mult[u] > 1:
                b = Block(self.options(u), mult[u], i)
                b.idx = u
                blocks.append(b)
        bpos = {b.pos for b in blocks}
        exempt = tuple(sorted(bpos))
        if fam.has_zero_coefficient(exempt):
            stats["refused"] = True
            return None
        ords = [i for i in range(len(distinct)) if i not in bpos]
        stats.update(kappa=int(fam.kappa), kappa1=int(fam.kappa1), distinct=len(distinct), blocks=[b.g for b in blocks])
        return distinct, fam, blocks, ords, exempt

    def _order(self, flats):
        """The processing order of the eight points: those where no
        invisible term is present first, then by the fewest fresh terms;
        with the presence lists and the mode of every point."""
        present = {y: [j for j, F in enumerate(flats) if y in F] for y in common.PTS if y != X0}
        order = [y for y in present if not present[y]]
        mode_of = {y: "exact" for y in order}
        rem = [y for y in present if present[y]]
        known = set()
        while rem:
            y = min(rem, key=lambda q: (sum(1 for j in present[q] if j not in known), q))
            n_fresh = sum(1 for j in present[y] if j not in known)
            mode_of[y] = {0: "exact", 1: "scan", 2: "scan2"}[n_fresh]
            order.append(y)
            known |= set(present[y])
            rem.remove(y)
        return order, present, mode_of

    def _born_codes(self, b):
        """The option codes of a born invisible term at the next point of
        its line: 27 at its second point, the three phases of the doubled
        class at its third."""
        ov = self.options(b["v"])
        fixed = b["codes"]
        if len(fixed) == 1:
            return list(range(ov.nclass * 3))
        if len(fixed) == 2:
            k = next(int(c) // 3 for t, c in fixed.items() if t != b["birth"])
            return [3 * self.dbl(ov, k) + m for m in range(3)]
        raise UnpinnedFamily("an invisible term with every point of its flat fixed was offered again")

    def run(self, cover, flats, target):
        """(hits, stats) for the base multiset `cover` with the invisible
        terms on `flats` (a list of one or two point tuples)."""
        stats = {"flats": [list(map(list, F)) for F in flats], "kappa": None, "kappa1": None, "distinct": 0,
                 "blocks": [], "order": [], "modes": [], "exact_solutions": [], "scan_candidates": 0,
                 "scan_solutions": 0, "scan2_collisions": 0, "scan2_solutions": 0, "pair_solutions": 0,
                 "absent_at_flat": 0, "same_state": 0, "cancelling": 0, "zero_coefficient": 0, "split_pruned": 0,
                 "candidates": 0, "dense_raw": 0, "reconstructions": 0, "hits": 0, "refused": False,
                 "native": False, "seconds": {}}
        t_start = time.time()
        setup = self._setup(cover, target, stats)
        if setup is None:
            return [], stats
        distinct, fam, blocks, ords, exempt = setup
        M = self.M
        proj = _ProjectorCache(blocks)
        M._proj = proj
        opts = [self.options(distinct[i]) for i in ords]
        tabs = [self.shapes(distinct[i]) for i in ords]
        n_dist = len(distinct)
        order, present, mode_of = self._order(flats)
        stats["order"] = [list(y) for y in order]
        stats["modes"] = [[list(y), mode_of[y]] for y in order]
        both_lines = len(flats) == 2 and flats[0] == flats[1] and len(flats[0]) == 3
        # a state: (alive masks per ordinary term, Ssel per offset, family, born terms, pair marker, splits)
        states = [([np.ones(len(tab), dtype=bool) for tab in tabs], {}, fam, [], None, [None] * len(blocks))]
        for y in order:
            self._check_deadline(f"point {y}")
            t = offset_index(y)
            rhs = target.rhs(y)
            here = present[y]
            new = []
            n_exact = 0
            for alive, Ssels, f, born, pair, sp in states:
                # the options of the visible ordinary terms and of the born terms at this point
                codes = [np.unique(tab[al, t]).astype(np.int64) for tab, al in zip(tabs, alive)]
                arrays = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
                full = [o.arrays() for o in opts]
                born_codes = []
                for b in born:
                    ov = self.options(b["v"])
                    cd = np.asarray(self._born_codes(b) if y in flats[b["flat"]] else [ov.absent], dtype=np.int64)
                    born_codes.append(cd)
                    arrays.append((ov.m1[cd], ov.m2[cd], ov.vecs[cd]))
                    full.append(ov.arrays())
                allowed = codes + born_codes
                if pair is not None:
                    new.extend(self._pair_continue(y, t, pair, alive, Ssels, f, born, sp, arrays, full, allowed,
                                                   blocks, exempt, rhs, tabs, stats))
                    continue
                fresh = [j for j in here if all(b["flat"] != j for b in born)]

                def restricted(alive, codes_all, combo):
                    al2 = [al & (tab[:, t] == int(allowed_i[c])) for tab, al, allowed_i, c in
                           zip(tabs, alive, codes_all[:len(ords)], combo[:len(ords)])]
                    return al2

                def born_with(born, codes_all, combo, extra):
                    out = []
                    for b, cd, c in zip(born, codes_all[len(ords):], combo[len(ords):len(ords) + len(born)]):
                        b2 = dict(b)
                        b2["codes"] = dict(b["codes"])
                        if y in flats[b["flat"]]:
                            b2["codes"][t] = int(cd[c])
                        out.append(b2)
                    return out + extra

                if not fresh:
                    sols = solve_slice3(arrays, blocks, f, rhs, self.rng, stats=stats, proj=proj,
                                        where=f"exact {y}", max_cand=self.max_cand)
                    n_exact += len(sols)
                    for combo, Ssel, f2 in sols:
                        if f2.has_zero_coefficient(exempt):
                            stats["zero_coefficient"] += 1
                            continue
                        fcombo = tuple(int(allowed[i][c]) for i, c in enumerate(combo))
                        sp2 = M._join_blocks(f2, full, blocks, fcombo, Ssel, rhs, sp)
                        if sp2 is None:
                            stats["split_pruned"] += 1
                            continue
                        new.append((restricted(alive, allowed, combo), {**Ssels, t: Ssel}, f2,
                                    born_with(born, allowed, combo, []), None, sp2))
                elif len(fresh) == 1:
                    j = fresh[0]
                    for combo, Ssel, v in self._scan_candidates(arrays, blocks, f, rhs, proj, stats):
                        ov = self.options(v)
                        f_ext = _extend_free(f, 1)
                        arr_ext = arrays + [(ov.m1[:1], ov.m2[:1], ov.vecs[:1])]
                        combo_ext = tuple(combo) + (0,)
                        Ps = proj(Ssel)[0] if blocks else None
                        f2 = restrict(f_ext, *slice_system(arr_ext, blocks, combo_ext, Ssel, rhs, Ps))
                        if f2 is None:
                            continue
                        if f2.has_zero_coefficient(exempt):
                            stats["zero_coefficient"] += 1
                            continue
                        stats["scan_solutions"] += 1
                        fcombo = tuple(int(allowed[i][c]) for i, c in enumerate(combo)) + (0,)
                        sp2 = M._join_blocks(f2, full + [ov.arrays()], blocks, fcombo, Ssel, rhs, sp)
                        if sp2 is None:
                            stats["split_pruned"] += 1
                            continue
                        nb = {"v": int(v), "codes": {t: 0}, "birth": t, "flat": j}
                        new.append((restricted(alive, allowed, combo), {**Ssels, t: Ssel}, f2,
                                    born_with(born, allowed, combo, [nb]), None, sp2))
                else:
                    if blocks:
                        raise UnpinnedFamily("two fresh invisible terms at a point of a base with a repeated "
                                             "state: the rank-2 residual scan with a block is not implemented")
                    if f.kappa1 or f.kappa or born:
                        raise UnpinnedFamily("two fresh invisible terms with an unpinned family or born terms "
                                             "present: not implemented")
                    dvs = (f.parts[0][0][ords], f.parts[1][0][ords], f.parts[2][0][ords])
                    for combo, kind, data in self._rank2_scan(arrays, dvs, rhs, stats, "scan2"):
                        al2 = restricted(alive, allowed, combo)
                        if kind == "rank2":
                            v, w, cv, cw = data
                            ov, ow = self.options(v), self.options(w)
                            f_ext = _extend_free(f, 2)
                            arr_ext = arrays + [(ov.m1[:1], ov.m2[:1], ov.vecs[:1]), (ow.m1[:1], ow.m2[:1], ow.vecs[:1])]
                            f2 = restrict(f_ext, *slice_system(arr_ext, [], tuple(combo) + (0, 0), (), rhs))
                            if f2 is None or f2.has_zero_coefficient(exempt):
                                continue
                            nb = [{"v": int(v), "codes": {t: 0}, "birth": t, "flat": fresh[0]},
                                  {"v": int(w), "codes": {t: 0}, "birth": t, "flat": fresh[1]}]
                            new.append((al2, {**Ssels, t: ()}, f2, nb, None, sp))
                        elif kind == "stab":
                            if not both_lines:
                                continue                    # two equal point terms: one term
                            v, _ = data
                            ov = self.options(v)
                            f_ext = _extend_free(f, 2)
                            arr_ext = arrays + [(ov.m1[:1], ov.m2[:1], ov.vecs[:1])] * 2
                            f2 = restrict(f_ext, *slice_system(arr_ext, [], tuple(combo) + (0, 0), (), rhs))
                            if f2 is None:
                                continue
                            stats["same_state"] += 1
                            nb = [{"v": int(v), "codes": {t: 0}, "birth": t, "flat": fresh[0]},
                                  {"v": int(v), "codes": {t: 0}, "birth": t, "flat": fresh[1]}]
                            new.append((al2, {**Ssels, t: ()}, f2, nb, None, sp))
                        elif both_lines:                    # otherwise two cancelling copies of one term
                            stats["cancelling"] += 1
                            new.append((al2, {**Ssels, t: ()}, f, [], ("cancel", t), sp))
            if mode_of[y] == "exact":
                stats["exact_solutions"].append(n_exact)
            states = new
            if not states:
                break
        stats["seconds"]["points"] = round(time.time() - t_start, 3)
        hits = []
        for alive, Ssels, f, born, pair, sp in states:
            if pair is not None:
                raise UnpinnedFamily("a cancelling pair survived the last point of its line undetermined")
            hits.extend(self._assemble(alive, Ssels, f, born, blocks, ords, opts, tabs, flats, target, stats))
        hits = M._dedupe(hits)
        stats["hits"] = len(hits)
        stats["seconds"]["total"] = round(time.time() - t_start, 3)
        return hits, stats

    def _scan_candidates(self, arrays, blocks, fam, rhs, proj, stats):
        """(combination, translate sets, dictionary state) candidates of a
        one-fresh-term point: a superset of the exact solutions, each
        decided by the caller through the extended family's restriction."""
        d1, K1 = fam.parts[0]
        ords = [i for i in range(len(d1)) if i not in {b.pos for b in blocks}]
        assert len(ords) == len(arrays)
        d0v = d1[ords] % P1
        Kv = K1[ords].reshape(len(ords), -1) % P1
        Kv = Kv[:, np.any(Kv != 0, axis=0)]
        out = []
        for Ssel in itertools.product(*[b.subsets for b in blocks]):
            if blocks:
                Ps, _ = proj(Ssel)
                popts = [(a[0] @ Ps[0].T) % P1 for a in arrays]
                prhs = (Ps[0] @ rhs[0]) % P1
            else:
                Ps = None
                popts = [a[0] % P1 for a in arrays]
                prhs = rhs[0] % P1
            table = self._table(blocks, Ssel, Ps)
            if Kv.shape[1] == 0:
                zero, matches = self._scan(popts, d0v, prhs, table)
                stats["scan_candidates"] += len(matches) + len(zero) * len(table.inside)
                for combo in zero:
                    if not table.inside:
                        stats["absent_at_flat"] += 1
                    out.extend((combo, Ssel, v) for v in table.inside)
                out.extend((combo, Ssel, v) for combo, v in matches)
            else:
                cands = self._scan_family(popts, d0v, Kv, prhs, table)
                stats["scan_candidates"] += len(cands)
                out.extend((combo, Ssel, v) for combo, v in cands)
        return out

    def _pair_continue(self, y, t, pair, alive, Ssels, f, born, sp, arrays, full, allowed, blocks, exempt, rhs,
                       tabs, stats):
        """The next point of a line shared by a cancelling pair (c_5 = -c_4,
        the common state at the first point unknown): the residual after
        the visible terms is scanned for a stabilizer residual (both copies
        in one class, phases distinct), a rank-2 residual (two states of
        one Pauli orbit with coefficients cancelling up to a cube root) or
        zero (the pair agrees here too and differs at the third point). The
        first point's common slice runs over the nine class translates of
        the found state. Returns states with both terms born."""
        M = self.M
        ords_n = len(arrays)
        dvs = (f.parts[0][0][:ords_n], f.parts[1][0][:ords_n], f.parts[2][0][:ords_n])
        t0 = pair[1]
        out = []

        def restricted(combo):
            return [al & (tab[:, t] == int(allowed_i[c])) for tab, al, allowed_i, c in
                    zip(tabs, alive, allowed, combo)]

        for combo, kind, data in self._rank2_scan(arrays, dvs, rhs, stats, "scan2"):
            al2 = restricted(combo)
            if kind == "zero":
                if pair[0] == "cancel":
                    out.append((al2, {**Ssels, t: ()}, f, [], ("cancel0", t0, t), sp))
                continue                                  # agreeing at all three points: one term twice
            if kind == "stab":
                w, a = data
                ow = self.options(w)
                for l4 in range(3):
                    for l5 in range(3):
                        if l4 == l5:
                            continue
                        c4, c5 = self._cancel_pair(a, l4, l5)
                        for s in self._class_states(w):
                            os_ = self.options(s)
                            if pair[0] == "cancel":
                                k4, m4 = os_.code_of(W3P[l4] * M.C[:, w])
                                k5, m5 = os_.code_of(W3P[l5] * M.C[:, w])
                                nb = [{"v": s, "codes": {t0: 0, t: 3 * k4 + m4}, "birth": t0, "flat": 0},
                                      {"v": s, "codes": {t0: 0, t: 3 * k5 + m5}, "birth": t0, "flat": 1}]
                                stats["pair_solutions"] += 1
                                out.append((al2, {**Ssels, t: ()}, _extend_pinned(f, [c4, c5]), nb, None, sp))
                            else:
                                t1 = pair[2]
                                k2, _ = os_.code_of(M.C[:, w])
                                k1 = self.dbl(os_, k2)
                                k4, m4 = os_.code_of(W3P[l4] * M.C[:, w])
                                k5, m5 = os_.code_of(W3P[l5] * M.C[:, w])
                                for l in range(3):
                                    nb = [{"v": s, "codes": {t0: 0, t1: 3 * k1 + l, t: 3 * k4 + m4}, "birth": t0,
                                           "flat": 0},
                                          {"v": s, "codes": {t0: 0, t1: 3 * k1 + l, t: 3 * k5 + m5}, "birth": t0,
                                           "flat": 1}]
                                    stats["pair_solutions"] += 1
                                    out.append((al2, {**Ssels, t: ()}, _extend_pinned(f, [c4, c5]), nb, None, sp))
                continue
            # rank2: the terms are (s, v') and (s, w^{-l} w') for s a class translate of v', with c_v + w^l c_w = 0
            if pair[0] != "cancel":
                continue
            v, w, cv, cw = data
            if self.orbit_id(v) != self.orbit_id(w):
                continue
            for l in range(3):
                num = abs(cv[2] + W3P[l] * cw[2]) < TOL * max(1.0, abs(cv[2]))
                mod = (cv[1] + int(self.F2.wpow[l]) * cw[1]) % P2 == 0
                if num != mod:
                    raise UnpinnedFamily("numeric and modular decisions of a cancelling pair disagree")
                if not num:
                    continue
                c5 = ((cw[0] * int(self.F1.wpow[l])) % P1, (cw[1] * int(self.F2.wpow[l])) % P2, cw[2] * W3P[l])
                for s in self._class_states(v):
                    os_ = self.options(s)
                    k4, m4 = os_.code_of(M.C[:, v])
                    k5, m5 = os_.code_of(W3P[(-l) % 3] * M.C[:, w])
                    nb = [{"v": s, "codes": {t0: 0, t: 3 * k4 + m4}, "birth": t0, "flat": 0},
                          {"v": s, "codes": {t0: 0, t: 3 * k5 + m5}, "birth": t0, "flat": 1}]
                    stats["pair_solutions"] += 1
                    out.append((al2, {**Ssels, t: ()}, _extend_pinned(f, [cv, c5]), nb, None, sp))
        return out

    def _assemble(self, alive, Ssels, f, born, blocks, ords, opts, tabs, flats, target, stats):
        M = self.M
        dim = self.dim
        if blocks and f.kappa > 0 and len(blocks) > 1:
            raise UnpinnedFamily(f"{f.kappa}-parameter coefficient family left after every slice equation on a "
                                 f"base with {len(blocks)} repeated states; the reconstruction with the "
                                 "parameter shared across several blocks is not implemented")
        if len(born) != len(flats):
            raise UnpinnedFamily("a state survived the last point with an invisible term not born")
        rows = []
        for tab, al in zip(tabs, alive):
            idx = np.flatnonzero(al)
            if len(idx) != 1:
                raise UnpinnedFamily(f"{len(idx)} shapes alive for a visible term after every point")
            rows.append(tab[idx[0]].astype(np.int64))
        ord_terms = []
        for o, row in zip(opts, rows):
            tv = np.zeros((9, dim), dtype=complex)
            tv[pidx(X0)] = o.u
            for s, x in enumerate(OFFSETS):
                tv[pidx(add(X0, x))] = o.vecs[int(row[s])]
            ord_terms.append(tv.ravel())
        inv_terms = []
        for b in born:
            F = flats[b["flat"]]
            if {OFFSETS[t] for t in b["codes"]} != {((y[0] - X0[0]) % 3, (y[1] - X0[1]) % 3) for y in F}:
                raise UnpinnedFamily("an invisible term was not determined on its whole flat")
            ov = self.options(b["v"])
            tv = np.zeros((9, dim), dtype=complex)
            for t, cd in b["codes"].items():
                tv[pidx(add(X0, OFFSETS[t]))] = ov.vecs[int(cd)]
            inv_terms.append(tv.ravel())
        terms = ord_terms + inv_terms
        if not blocks:
            return [M.confirm(terms, int(f.kappa), target)]
        full = [o.arrays() for o in opts] + [self.options(b["v"]).arrays() for b in born]
        d0, K = f.parts[2]
        per_block = [[] for _ in blocks]
        for s, x in enumerate(OFFSETS):
            combo = [int(row[s]) for row in rows]
            for b in born:
                combo.append(int(b["codes"].get(s, self.options(b["v"]).absent)))
            if s not in Ssels:
                raise UnpinnedFamily("a slice offset without a translate selection at the reconstruction")
            rhs = target.rhs(add(X0, x))
            a0, A = M._block_coordinates(f, full, blocks, tuple(combo), Ssels[s], rhs, strict=True, affine=True)
            pos = 0
            for bi, (b, S) in enumerate(zip(blocks, Ssels[s])):
                per_block[bi].append((x, {k: (a0[pos + j], A[pos + j]) for j, k in enumerate(S)}))
                pos += len(S)
        recon = [reconstruct_block(b.o, b.g, (d0[b.pos], K[b.pos]), per_block[bi], self.rng)
                 for bi, b in enumerate(blocks)]
        stats["reconstructions"] += 1
        hits = []
        for choice in itertools.product(*recon):
            terms2 = list(terms)
            degenerate = False
            for b, (copies, deg) in zip(blocks, choice):
                degenerate |= deg
                for _, cd in copies:
                    tv = np.zeros((9, dim), dtype=complex)
                    tv[pidx(X0)] = b.o.u
                    for x, code in cd.items():
                        tv[pidx(add(X0, x))] = b.o.vecs[code]
                    terms2.append(tv.ravel())
            hits.append(M.confirm(terms2, int(degenerate), target))
        return hits

    # -- entry points --
    def run_one(self, cover, flat, target):
        """(hits, stats) for the base 5-multiset `cover` with one invisible
        term on the flat `flat` (a name of common.FLATS)."""
        if len(cover) != 5:
            raise UnpinnedFamily("stage (beta') needs a 5-multiset")
        hits, stats = self.run(cover, [FLATS[flat]], target)
        stats["flat"] = flat
        return hits, stats

    def run_two(self, cover, pair, target):
        """(hits, stats) for the base 4-multiset `cover` with two invisible
        terms on the flats `pair` (two names of common.FLATS)."""
        if len(cover) != 4:
            raise UnpinnedFamily("stage (gamma) needs a 4-multiset")
        hits, stats = self.run(cover, [FLATS[pair[0]], FLATS[pair[1]]], target)
        stats["pair"] = list(pair)
        return hits, stats


def matcher_family(M, distinct, b1, b2, bC):
    from matcher import family_from
    return family_from(M.U1[distinct], M.U2[distinct], M.C[:, distinct], b1, b2, bC)


def planted_target(terms, coeffs, F1, F2):
    """A Target from planted terms (columns with entries in {0} u {w^k})
    and integer coefficients."""
    vec = np.column_stack(terms) @ np.asarray(coeffs, dtype=complex)
    return vector_target(vec, 2, F1, F2)


def hist_key(stage, st):
    """The deterministic per-run key of the solution histogram."""
    if stage in ("beta", "gamma"):
        return (f"{'/'.join(str(v) for v in st['exact_solutions'])},{st['scan_solutions']},{st['scan2_solutions']},"
                f"{st['pair_solutions']}")
    return ",".join(str(b) for b in st["coord_solutions"])


__all__ = ["InvisibleMatcher3", "shape_table", "planted_target", "hist_key", "TermOpts", "COMP", "E1", "E2"]
