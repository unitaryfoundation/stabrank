"""Stages (beta') and (gamma) of the rank-5 exclusion of |T>^5: the base
point 00 along qubits 1, 2 with four or three visible terms and one or two
invisible terms whose flats are given (docs/notes/t5_rank5_exclusion.md,
section 4).

Setting. Slice psi_5 = sum_{i=1}^5 c_i s_i along qubits 1, 2. At 00 the
visible terms' slices form a full k-cover of psi_3 (the base, k in {3, 4})
with coefficients d_i = c_i / alpha_00. Each invisible term vanishes at 00
and is nonzero exactly on one of the six affine flats missing 00: the
points p01, p10, p11 and the lines B = {01, 10}, C1 = {10, 11}, D1 =
{01, 11}. Its slice at the first point of its flat is c v for a
three-qubit stabilizer state v (a dictionary state) and at the second
point of a line c i^l Q v for a Pauli Q (the structure lemma for a line
term with base slice v). The points of F_2^2 are numbered x = 2 x_1 + x_2.

What the matcher assumes and what it enumerates. The flats of the
invisible terms are parameters (every flat and every flat pair is run by
the pipeline); nothing is assumed about the flats of the visible terms: at
every one of the three other points each is absent or one of the 32
phased Pauli translates of its base slice, the codes at a point are
restricted to those compatible with the codes already fixed at the other
two points (a term present at two points is a plane and its third code is
the composition up to the quadratic sign, a term present at exactly one
is a line through 00 and absent at the third, a term absent at both is
free), and the presence pattern is checked to be a subspace through 00
with the structure lemma's composite code at assembly (valid_term_codes).
A repeated base state (the twelve (2, 1, 1) bases of stage (beta')) is a
block in the sense of slice_cover: its copies contribute an arbitrary
vector of the span of at most two Pauli translates at every slice, the
equations are projected onto the annihilator of the chosen translates,
and the copies are rebuilt by slice_cover.reconstruct_block (cancelling
copies allowed).

The slice equations, point by point. The points are processed in an
order that keeps the number of invisible terms whose slice is still
unknown ("fresh") at each point as small as possible: first the points
where no invisible term is present (exact equations), then the others.

  exact   no fresh term: sum_i d_i w_i + block + sum_j c_j t_j = rhs over
          the visible options, the block's translate sets and, for every
          invisible term already known, its 32 phased translates, by
          slice_cover.solve_slice (meet in the middle mod 65521, every
          candidate decided over C and mod 2013265921);
  scan    one fresh term: the residual after the visible terms, the block
          and the known invisible terms must be c v for a dictionary
          state v. All option combinations are formed at once mod 65521,
          the residual projected onto the annihilator of the block's
          translates, normalized and looked up in the table of the
          normalized projected dictionary states; every match is
          re-decided exactly (the system (translates | v)(a, c) = r solved
          over C, mod 65521 and mod 2013265921, unique with every entry
          nonzero). A zero residual means the fresh term is absent at a
          point of its flat, which is another flat's configuration, and is
          not a solution here;
  scan2   two fresh terms (stage (gamma) only, when both flats share a
          point that is the first processed point of both: two equal
          point flats, or two equal line flats): the residual r must be
          c_4 v_4 + c_5 v_5 for dictionary states. Distinct v_4, v_5 are
          found by a projective hash (v and w span r exactly when their
          images in F^8 / span(r) are parallel), every collision decided
          exactly mod 2013265921 and over C with the two required to
          agree. For two line terms the states may coincide: r = D v with
          D = c_4 + c_5 (a stabilizer residual, the split unknown) or
          r = 0 (a cancelling pair, c_5 = -c_4, v unknown); both are
          carried to the line's other point, where the pair contributes a
          vector of the span of at most two translates of v (with v known
          the coordinates in the translate basis of v, decided over C and
          mod both primes, must agree on their zero pattern, and the pair
          may also cancel there inside one class; with v unknown the
          residual is scanned for a stabilizer or rank-2 residual whose
          states lie in one Pauli orbit) and the split is solved from the
          phase relations, numerically and mod 2013265921. Two equal point
          flats with equal states or a zero residual would be the same
          term twice and are rejected.

Every surviving state is assembled into five terms (the copies of a block
through reconstruct_block) and confirmed as at H^6: residual against
psi_5, rank, independence, nonzero coefficients, and the span condition
mod 2013265921 (SliceMatcher.confirm); batch.py re-decides every hit again
from its phase codes (common.decide_terms).

The ambiguous case (stage (beta') with a block). At the scan point the
residual can lie in the span of the block's chosen translates with no
fifth term needed there; the fifth term's slice could then be a dictionary
state inside that span and the slice equation cannot separate it from the
copies. `_pair_brute` decides it by enumeration: for every dictionary state
v inside the span, every assignment of the two copies' codes at the three
points, every code of the fifth term at the second point of its flat and
every admissible option of the visible terms there, the coefficients
(c_1, c_2, c_5) are the solution of the linear system of the slice
equations in the translate basis of the repeated state together with
c_1 + c_2 = D, solved in batches point by point; a survivor whose system
leaves a coefficient free raises UnpinnedFamily.

Refusals and undecided runs. A base whose coefficient family is empty or
has a dead ordinary coefficient is refused (the lists exclude such
multisets, so a refusal fails the aggregate). A base with a coefficient
family, a block other than one pair, a residual whose complex and modular
decisions disagree, or any configuration the enumeration above does not
cover raises slice_cover.UnpinnedFamily, which batch.py records as
undecided; the aggregate fails on any such record.
"""
from __future__ import annotations

import copy
import itertools
import types

import numpy as np

from slice_cover import (P1, P2, Block, Family, SliceMatcher, UnpinnedFamily, _affine_solve_C,  # noqa: E402
                         _affine_solve_mod, _projectors, _rank_mod, _refine_split, exact_codes, pauli_reps,
                         reconstruct_block, solve_slice, valid_term_codes)

N1 = 2
X0 = 0
POINTS = (1, 2, 3)
FLATS = {"p01": (1,), "p10": (2,), "p11": (3,), "B": (1, 2), "C1": (2, 3), "D1": (1, 3)}
FOURTH_ARR = np.array([1, 1j, -1, -1j])
TOL = 1e-7


def to_field(F, v):
    """A vector with Gaussian-integer entries reduced to F_p."""
    re, im = np.round(v.real).astype(np.int64), np.round(v.imag).astype(np.int64)
    if not np.allclose(v, re + 1j * im, atol=1e-9):
        raise AssertionError("vector entries are not Gaussian integers")
    return (re + F.i * im) % F.p


def _restrict(arr, codes):
    codes = list(codes)
    return arr[0][codes], arr[1][codes], arr[2][codes]


def _extend(fam, cs):
    """The point family with the known coefficients `cs` (triples over
    F_P1, F_P2, C) appended."""
    d1, d2, dC = fam.parts[0][0], fam.parts[1][0], fam.parts[2][0]
    n = len(d1) + len(cs)
    return Family([(np.append(d1, [c[0] for c in cs]).astype(np.int64) % P1, np.zeros((n, 0), dtype=np.int64)),
                   (np.append(d2, [c[1] for c in cs]).astype(np.int64) % P2, np.zeros((n, 0), dtype=np.int64)),
                   (np.append(dC, [c[2] for c in cs]).astype(complex), np.zeros((n, 0), dtype=complex))])


def codes_key(terms):
    return sorted(exact_codes(t)[0].tobytes() for t in terms)


class LightOptions:
    """The 32 phased Pauli translates i^l Q u of a base slice u (option code
    4 k + l) plus 'absent' (code 4 2^n), over F_P1, F_P2 and C, as
    slice_cover.TermOptions without its class-product table: an invisible
    term's slices are never composed, only enumerated, and the table is the
    expensive part of TermOptions when thousands of states are touched."""

    def __init__(self, u, n, F1, F2):
        self.n = n
        self.u = u
        self.reps, imgs = pauli_reps(u, n)
        vecs = [ph * im for im in imgs for ph in FOURTH_ARR]
        self.vecs = np.array(vecs + [np.zeros(1 << n, dtype=complex)])
        self.absent = len(vecs)
        codes = [exact_codes(v) for v in vecs]
        m1 = np.array([F1.codes_to_field(c) for c, _ in codes] + [np.zeros(1 << n, dtype=np.int64)])
        m2 = np.array([F2.codes_to_field(c) for c, _ in codes] + [np.zeros(1 << n, dtype=np.int64)])
        scal = np.array([sc for _, sc in codes])
        mod = np.abs(scal)
        ph = np.round(np.angle(scal / mod) / (np.pi / 2)).astype(int) % 4
        if not np.allclose(scal / mod, FOURTH_ARR[ph]):
            raise AssertionError("a translate's leading entry is not a fourth root of unity")
        self.m1 = (m1 * F1.ipow[np.append(ph, 0)][:, None]) % F1.p
        self.m2 = (m2 * F2.ipow[np.append(ph, 0)][:, None]) % F2.p

    def arrays(self):
        return self.m1, self.m2, self.vecs

    def code_of(self, v):
        """(class, phase) with v = i^phase imgs[class]; AssertionError when v
        is not a phased translate of u."""
        for k in range(1 << self.n):
            w = self.vecs[4 * k]
            nz = np.flatnonzero(np.abs(w) > 1e-9)
            ratio = v[nz[0]] / w[nz[0]]
            for l in range(4):
                if abs(ratio - FOURTH_ARR[l]) < 1e-6 and np.allclose(v, FOURTH_ARR[l] * w, atol=1e-9):
                    return k, l
        raise AssertionError("vector is not a phased Pauli translate of the base slice")


class InvisibleMatcher:
    """Every decomposition of psi_5 whose slice at 00 along qubits 1, 2 has
    exactly the terms of `cover` visible, with the invisible terms on the
    given flats."""

    def __init__(self, E, seed=31, verbose=False, max_cand=2_000_000):
        self.E = E
        self.n2 = E.n
        self.dim = 1 << E.n
        self.F1, self.F2 = E.F1, E.F2
        self.SM = SliceMatcher(E, N1, native=False, seed=seed, max_cand=max_cand)
        self.max_cand = max_cand
        self.rng = np.random.default_rng(seed + 1)
        self.verbose = verbose
        self._tables = {}          # (block state, translate set) -> projected dictionary table
        self._inv = {}             # dictionary index -> LightOptions of an invisible term's base slice
        self._orbit = None         # dictionary index -> Pauli orbit id
        self._tbasis_cache = {}    # dictionary index -> its translate basis and inverses
        self._hash_f = None

    # -- shared pieces --
    def options(self, idx):
        return self.SM.options(idx)

    def inv_options(self, v):
        if v not in self._inv:
            self._inv[v] = LightOptions(self.E.C[:, v], self.n2, self.F1, self.F2)
        return self._inv[v]

    def orbit_id(self, v):
        """The Pauli orbit of a dictionary state (the least index among its
        class images), computed once for the whole dictionary."""
        if self._orbit is None:
            E = self.E
            lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
            ids = np.full(E.N, -1, dtype=np.int64)
            for i in range(E.N):
                if ids[i] >= 0:
                    continue
                _, imgs = pauli_reps(E.C[:, i], self.n2)
                members = [lookup[exact_codes(im)[0].tobytes()] for im in imgs]
                root = min(members)
                for m in members:
                    ids[m] = root
            self._orbit = ids
        return int(self._orbit[v])

    def rhs(self, x):
        return self.SM.rhs(X0, x)

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def compatible_codes(o, fixed, y):
        """The option codes of a visible term at the point y compatible with
        its codes at the points already fixed: with both other points fixed,
        a plane term's code is the composition partner (two quadratic
        signs), a term present at exactly one of them is a line term and
        absent at y, a term absent at both is free; with fewer than two
        fixed, every code."""
        A = o.absent
        others = [z for z in POINTS if z != y]
        if not all(z in fixed for z in others):
            return list(range(A + 1))
        ca, cb = fixed[others[0]], fixed[others[1]]
        if ca != A and cb != A:
            return [c for c in range(A) if any(o.compose(ca, c, s) == cb for s in (0, 1))]
        if ca != A or cb != A:
            return [A]
        return list(range(A + 1))

    # -- the projected dictionary tables and the one-fresh-term scan (as the H^5 stage beta) --
    def _table(self, blocks, Ssel, Ps):
        key = tuple((b.idx, tuple(S)) for b, S in zip(blocks, Ssel))
        if key in self._tables:
            return self._tables[key]
        E = self.E
        if Ps is None:
            PV = np.ascontiguousarray(E.U1.T)
            PVC = E.C
        else:
            PV = (Ps[0] @ E.U1.T) % P1
            PVC = Ps[2] @ E.C
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
        inv = self.F1.inv_table[piv]
        PVn = (PVv * inv[None, :]) % P1
        f = self.rng.integers(1, P1, size=PV.shape[0])
        keys = (f @ PVn) % P1
        order = np.argsort(keys, kind="stable")
        tab = types.SimpleNamespace(f=f, keys=keys[order], order=valid[order], PVn=PVn, pos_of=valid,
                                    inside=[int(v) for v in np.flatnonzero(inside)])
        self._tables[key] = tab
        return tab

    @staticmethod
    def _all_residuals(popts, dv, prhs):
        """The residuals of every option combination at once mod P1, as a
        (prod sizes, dim') array, and the sizes."""
        dimp = prhs.shape[0]
        R = (prhs % P1)[None, :]
        for i, o in enumerate(popts):
            R = (R[:, None, :] - ((int(dv[i]) * o) % P1)[None, :, :]) % P1
            R = R.reshape(-1, dimp)
        return R, tuple(len(o) for o in popts)

    def _scan(self, popts, dv, prhs, table):
        """Combinations whose projected residual is zero, and the
        (combination, dictionary state) pairs whose normalized residual
        equals the state's normalized projection. A superset of the exact
        solutions."""
        R, sizes = self._all_residuals(popts, dv, prhs)
        nz = R != 0
        has = nz.any(axis=1)
        zero = [tuple(int(c) for c in np.unravel_index(int(z), sizes)) for z in np.flatnonzero(~has)]
        rows = np.flatnonzero(has)
        if not len(rows):
            return zero, []
        j = np.argmax(nz[rows], axis=1)
        piv = R[rows, j]
        inv = self.F1.inv_table[piv]
        Rn = (R[rows] * inv[:, None]) % P1
        keys = (Rn @ table.f) % P1
        lo = np.searchsorted(table.keys, keys, side="left")
        hi = np.searchsorted(table.keys, keys, side="right")
        matches = []
        for q in np.flatnonzero(hi > lo):
            for pos in range(lo[q], hi[q]):
                v = int(table.order[pos])
                col = int(np.searchsorted(table.pos_of, v))
                if np.array_equal(Rn[q], table.PVn[:, col]):
                    matches.append((tuple(int(c) for c in np.unravel_index(int(rows[q]), sizes)), v))
        return zero, matches

    def _residuals(self, arrays, dvs, combo, rhs):
        """rhs minus the ordinary terms' contribution, in the three fields."""
        out = []
        for fld, p in ((0, P1), (1, P2), (2, None)):
            acc = rhs[fld].copy()
            for i, c in enumerate(combo):
                if p is None:
                    acc = acc - dvs[fld][i] * arrays[i][fld][c]
                else:
                    acc = (acc - int(dvs[fld][i]) * arrays[i][fld][c]) % p
            out.append(acc)
        return out

    def _solve_fifth(self, blocks, Ssel, res, v, stats):
        """The coordinates of the residual on the chosen translates of the
        blocks and on the fresh term's slice v (None when no fresh term is
        present): (a, c, ambiguous), with a and c unique over C, mod P1 and
        mod P2 and every entry nonzero, or None when the residual is not in
        the span (the modular superset let it through). `ambiguous` is set
        when v is None and the residual lies in the span of a nonempty
        translate set: the fresh term's slice could then be a dictionary
        state inside that span."""
        E = self.E
        _, Vs = _projectors(blocks, Ssel) if blocks else (None, None)
        cols = [[], [], []]
        k = 0
        if Vs is not None and Vs[2].shape[1]:
            cols = [[Vs[0]], [Vs[1]], [Vs[2]]]
            k = Vs[2].shape[1]
        if v is not None:
            cols[0].append(E.U1[v][:, None])
            cols[1].append(E.U2[v][:, None])
            cols[2].append(E.C[:, v][:, None])
        if not cols[2]:
            if np.linalg.norm(res[2]) > TOL:
                return None
            return np.zeros(0, dtype=complex), None, False
        AC = np.column_stack(cols[2])
        solC = _affine_solve_C(AC, res[2])
        if solC is None:
            return None
        if solC[1].shape[1]:
            raise UnpinnedFamily("the chosen translates and the fresh term's slice are dependent at a slice "
                                 "point; the split is a family")
        A1 = np.column_stack(cols[0]) % P1
        A2 = np.column_stack(cols[1]) % P2
        sol1 = _affine_solve_mod(A1, res[0], P1)
        sol2 = _affine_solve_mod(A2, res[1], P2)
        if sol1 is None or sol2 is None or sol1[1].shape[1] or sol2[1].shape[1]:
            raise UnpinnedFamily("the complex and modular solves of a slice residual disagree")
        x = solC[0]
        if np.any(np.abs(x) < 1e-9):
            return None                # a zero block coordinate duplicates a smaller translate set
        if v is None:
            stats["ambiguous"] += 1
            return x, None, True
        return x[:k], (int(sol1[0][-1]), int(sol2[0][-1]), complex(x[-1])), False

    def _coordinate_scan(self, arrays, blocks, dvs, rhs, stats, tag):
        """Solutions of a slice equation with one fresh term: a list of
        (combo, Ssel, fresh, a, ambiguous) with fresh = None (the projected
        residual is zero) or (v, c) in the three fields, and a the block
        coordinates."""
        out = []
        subsets = [b.subsets for b in blocks]
        for Ssel in itertools.product(*subsets):
            Ps, _ = _projectors(blocks, Ssel) if blocks else (None, None)
            if Ps is None:
                popts = [a[0] for a in arrays]
                prhs = rhs[0] % P1
            else:
                popts = [(a[0] @ Ps[0].T) % P1 for a in arrays]
                prhs = (Ps[0] @ rhs[0]) % P1
            table = self._table(blocks, Ssel, Ps)
            zero, matches = self._scan(popts, dvs[0], prhs, table)
            stats[tag + "_candidates"] += len(matches) + len(zero)
            for combo in zero:
                res = self._residuals(arrays, dvs, combo, rhs)
                sol = self._solve_fifth(blocks, Ssel, res, None, stats)
                if sol is None:
                    continue
                out.append((combo, Ssel, None, sol[0], sol[2]))
            for combo, v in matches:
                res = self._residuals(arrays, dvs, combo, rhs)
                sol = self._solve_fifth(blocks, Ssel, res, v, stats)
                if sol is None:
                    continue
                out.append((combo, Ssel, (v, sol[1]), sol[0], False))
        stats[tag + "_solutions"] += len(out)
        return out

    # -- the two-fresh-term scan (stage gamma, no blocks) --
    def _solve_states(self, res, states):
        """The coefficients of the residual on the dictionary states
        `states` in the three fields: unique with every entry nonzero in
        all three, or None when the residual is not in their span; the
        complex and P2 decisions must agree."""
        E = self.E
        solC = _affine_solve_C(E.C[:, states], res[2])
        sol1 = _affine_solve_mod(E.U1[states].T % P1, res[0], P1)
        sol2 = _affine_solve_mod(E.U2[states].T % P2, res[1], P2)
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
        F^8 / span(r) are parallel, so their projective keys (ratios of
        three random functionals) agree; every collision is decided exactly
        mod 2013265921 and over C, and the two must agree."""
        E = self.E
        N, dim = E.N, self.dim
        R, sizes = self._all_residuals([a[0] for a in arrays], dvs[0], rhs[0] % P1)
        if self._hash_f is None:
            self._hash_f = [self.rng.integers(1, P1, size=dim) for _ in range(3)]
        f1, f2, f3 = self._hash_f
        U1 = E.U1
        out = []
        n_coll = 0

        def combo_of(q):
            return tuple(int(cc) for cc in np.unravel_index(int(q), sizes))

        def decide_pair(combo, res, v, w):
            r_mod = _rank_mod(np.array([res[1], E.U2[v], E.U2[w]], dtype=np.int64), P2)
            r_num = int(np.linalg.matrix_rank(np.column_stack([res[2], E.C[:, v], E.C[:, w]]), tol=1e-8))
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
                combo = combo_of(idx0[q])
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
                combo = combo_of(idxb[q])
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
                combo = combo_of(idxb[q])
                res = self._residuals(arrays, dvs, combo, rhs)
                # every pair inside a run of equal keys (three or more states with parallel images
                # give every pair among them)
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
            # the first two functionals both zero: those states against every other, by brute force
            for q in np.flatnonzero(kzz.any(axis=1)):
                combo = combo_of(idxb[q])
                res = self._residuals(arrays, dvs, combo, rhs)
                for v in np.flatnonzero(kzz[q]):
                    for w in range(N):
                        if w != int(v):
                            decide_pair(combo, res, int(v), w)
        stats[tag + "_collisions"] += n_coll
        stats[tag + "_solutions"] += len(out)
        return out

    def _exact_known(self, arr, dvs, known, rhs, stats):
        """Solutions of an exact slice equation without blocks: the visible
        terms over their (restricted) options `arr` with the coefficients
        dvs, and one or two known invisible terms `known` = [(options,
        coefficient triple)] each over its 32 phased translates. Every
        visible combination's residual is formed at once mod P1 and met with
        the known terms' options through a random functional (a superset);
        every candidate is decided exactly in the three fields. Returns the
        combos (visible codes, then the known terms' codes)."""
        R, sizes = self._all_residuals([a[0] for a in arr], dvs[0], rhs[0] % P1)
        f = self.rng.integers(1, P1, size=self.dim)
        kR = (R @ f) % P1
        A = [((int(c[0]) * ov.m1[:ov.absent]) % P1) for ov, c in known]
        kA = [(a @ f) % P1 for a in A]
        cands = []
        if len(known) == 1:
            order = np.argsort(kA[0], kind="stable")
            sk = kA[0][order]
            lo, hi = np.searchsorted(sk, kR, "left"), np.searchsorted(sk, kR, "right")
            for q in np.flatnonzero(hi > lo):
                for pos in range(lo[q], hi[q]):
                    cands.append((int(q), (int(order[pos]),)))
        elif len(known) == 2:
            need = (kR[:, None] - kA[0][None, :]) % P1                 # (n, 32): what the second term must give
            order = np.argsort(kA[1], kind="stable")
            sk = kA[1][order]
            lo, hi = np.searchsorted(sk, need, "left"), np.searchsorted(sk, need, "right")
            for q, i in zip(*np.nonzero(hi > lo)):
                for pos in range(lo[q, i], hi[q, i]):
                    cands.append((int(q), (int(i), int(order[pos]))))
        else:
            raise UnpinnedFamily("the exact step is written for at most two known invisible terms")
        stats["candidates"] += len(cands)
        out = []
        for q, js in cands:
            combo = tuple(int(c) for c in np.unravel_index(q, sizes))
            res = self._residuals(arr, dvs, combo, rhs)
            zero = []
            for fld, p in ((0, P1), (1, P2), (2, None)):
                acc = res[fld].copy()
                for (ov, c), j in zip(known, js):
                    if p is None:
                        acc = acc - c[2] * ov.vecs[j]
                    else:
                        acc = (acc - int(c[fld]) * ov.arrays()[fld][j]) % p
                zero.append(np.linalg.norm(acc) < TOL if p is None else not np.any(acc))
            if zero[1] != zero[2]:
                raise UnpinnedFamily("the complex and modular (2013265921) decisions of an exact slice equation "
                                     "disagree")
            if all(zero):
                out.append(combo + js)
        return out

    # -- blocks --
    def _coords(self, fam, arrays, blocks, combo, Ssel, rhs):
        """The block coordinates of a solved slice (strict: a residual the
        solve cannot place raises), or None when a coordinate vanishes
        (the solution duplicates a smaller translate set)."""
        if not blocks:
            return np.zeros(0, dtype=complex)
        a = self.SM._block_coordinates(fam, arrays, blocks, combo, Ssel, rhs, strict=True)
        if not np.all(np.abs(a) > 1e-9):
            return None
        return a

    def _refine(self, blocks, Ssel, a, sp, dC):
        if not blocks:
            return sp
        new, pos = [], 0
        for b, S, s in zip(blocks, Ssel, sp):
            coords = a[pos:pos + len(S)]
            pos += len(S)
            if b.g != 2:
                new.append(s)
                continue
            s2 = _refine_split(dC[b.pos], S, coords, s)
            if s2 is not None and len(s2) == 0:
                return None
            new.append(s2)
        return new

    @staticmethod
    def _block_data(blocks, Ssels, coords):
        per_block = [[] for _ in blocks]
        for off in POINTS:
            Ssel, a = Ssels[off], coords[off]
            pos = 0
            for bi, (b, S) in enumerate(zip(blocks, Ssel)):
                per_block[bi].append((off, {k: a[pos + j] for j, k in enumerate(S)}))
                pos += len(S)
        return per_block

    # -- vectors and assembly --
    def _term(self, o, codes):
        t = np.zeros((1 << N1, self.dim), dtype=complex)
        t[X0] = o.u
        for off, cd in codes.items():
            t[off] = o.vecs[cd]
        return t.ravel()

    def _vector(self, slices):
        """An invisible term from its slices {point: vector}."""
        t = np.zeros((1 << N1, self.dim), dtype=complex)
        for y, vec in slices.items():
            t[y] = vec
        return t.ravel()

    def _assemble(self, opts, ords, blocks, dC, codes, invisible, Ss, cs, stats):
        """codes: per visible ordinary term its codes at the three points;
        invisible: (vector, coefficient) pairs; Ss, cs: the blocks' translate
        sets and coordinates per point."""
        for i, o in enumerate(opts):
            if not valid_term_codes(o, N1, codes[i]):
                return []
        terms = [self._term(o, codes[i]) for i, o in enumerate(opts)]
        coeffs = [dC[ords[i]] for i in range(len(opts))]
        for vec, c in invisible:
            terms.append(vec)
            coeffs.append(c)
        if not blocks:
            return [self.SM.confirm(terms, coeffs, 0)]
        data = self._block_data(blocks, Ss, cs)
        recon = [reconstruct_block(b.o, b.g, dC[b.pos], data[bi], N1, self.rng) for bi, b in enumerate(blocks)]
        stats["reconstructions"] += 1
        hits = []
        for choice in itertools.product(*recon):
            terms2, cs2, degenerate = list(terms), list(coeffs), False
            for b, (copies, deg) in zip(blocks, choice):
                degenerate |= deg
                for cval, cd in copies:
                    terms2.append(self._term(b.o, cd))
                    cs2.append(cval)
            hits.append(self.SM.confirm(terms2, cs2, int(degenerate)))
        return hits

    def _setup(self, cover, stats, allow_blocks):
        E = self.E
        distinct = sorted(set(cover))
        mult = {u: cover.count(u) for u in distinct}
        fam = Family.from_cover(E, distinct)
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
        stats.update(kappa=fam.kappa, distinct=len(distinct), blocks=[b.g for b in blocks])
        if fam.kappa:
            raise UnpinnedFamily(f"base with a {fam.kappa}-parameter coefficient family; the lists have none and "
                                 "the matcher takes the base coefficients as a point")
        if blocks and (not allow_blocks or len(blocks) != 1 or blocks[0].g != 2):
            raise UnpinnedFamily(f"blocks {[b.g for b in blocks]}: the matcher is written for at most one pair block")
        ords = [i for i in range(len(distinct)) if i not in bpos]
        opts = [self.options(distinct[i]) for i in ords]
        arrays = [o.arrays() for o in opts]
        d1, d2, dC = fam.parts[0][0], fam.parts[1][0], fam.parts[2][0]
        dvs = (d1[ords], d2[ords], dC[ords])
        return distinct, fam, blocks, ords, opts, arrays, dvs, dC

    # ------------------------------------------------------ stage (beta') --
    def run_one(self, cover, flat):
        """(hits, stats) for the base 4-multiset `cover` with one invisible
        term on the flat `flat` (a name of FLATS)."""
        F = FLATS[flat]
        stats = {"flat": flat, "kappa": None, "distinct": 0, "blocks": [], "exact_solutions": [],
                 "scan_candidates": 0, "scan_solutions": 0, "absent_at_flat": 0, "ambiguous": 0, "brute": 0,
                 "brute_systems": 0, "second_solutions": 0, "split_pruned": 0, "candidates": 0,
                 "reconstructions": 0, "hits": 0, "refused": False, "native": False}
        if len(cover) != 4:
            raise UnpinnedFamily("stage (beta') needs a 4-multiset")
        setup = self._setup(cover, stats, allow_blocks=True)
        if setup is None:
            return [], stats
        distinct, fam, blocks, ords, opts, arrays, dvs, dC = setup
        E = self.E
        rhs = {y: self.rhs(y) for y in POINTS}
        exact_pts = [y for y in POINTS if y not in F]
        scan_pt = F[0]
        second = F[1] if len(F) == 2 else None
        r = len(opts)
        states = [([dict() for _ in range(r)], {}, {}, [None] * len(blocks))]
        for z in exact_pts:
            new = []
            for codes, Ss, cs, sp in states:
                allowed = [self.compatible_codes(opts[i], codes[i], z) for i in range(r)]
                arr = [_restrict(arrays[i], allowed[i]) for i in range(r)]
                sols = solve_slice(arr, blocks, fam, rhs[z], self.rng, max_cand=self.max_cand, stats=stats,
                                   log=self.log)
                for combo, Ssel in sols:
                    a = self._coords(fam, arr, blocks, combo, Ssel, rhs[z])
                    if a is None:
                        continue
                    sp2 = self._refine(blocks, Ssel, a, sp, dC)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    codes2 = [{**codes[i], z: int(allowed[i][combo[i]])} for i in range(r)]
                    new.append((codes2, {**Ss, z: Ssel}, {**cs, z: a}, sp2))
            stats["exact_solutions"].append(len(new))
            states = new
            if not states:
                return [], stats
        hits = []
        for codes, Ss, cs, sp in states:
            allowed = [self.compatible_codes(opts[i], codes[i], scan_pt) for i in range(r)]
            arr = [_restrict(arrays[i], allowed[i]) for i in range(r)]
            sols = self._coordinate_scan(arr, blocks, dvs, rhs[scan_pt], stats, "scan")
            for combo, Ssel, fresh, a, amb in sols:
                codes_s = [{**codes[i], scan_pt: int(allowed[i][combo[i]])} for i in range(r)]
                if fresh is None:
                    if not amb:
                        stats["absent_at_flat"] += 1        # the fifth term would vanish on its flat
                        continue
                    hits.extend(self._pair_brute(opts, ords, blocks, dC, codes_s, Ssel, F, rhs, stats))
                    continue
                v, c5 = fresh
                sp1 = self._refine(blocks, Ssel, a, sp, dC)
                if sp1 is None:
                    stats["split_pruned"] += 1
                    continue
                Ss1, cs1 = {**Ss, scan_pt: Ssel}, {**cs, scan_pt: a}
                if second is None:
                    inv = [(self._vector({scan_pt: E.C[:, v]}), c5[2])]
                    hits.extend(self._assemble(opts, ords, blocks, dC, codes_s, inv, Ss1, cs1, stats))
                    continue
                ov = self.inv_options(v)
                allowed2 = [self.compatible_codes(opts[i], codes_s[i], second) for i in range(r)]
                arr2 = [_restrict(arrays[i], allowed2[i]) for i in range(r)] + \
                    [_restrict(ov.arrays(), range(ov.absent))]
                fam5 = _extend(fam, [c5])
                sols5 = solve_slice(arr2, blocks, fam5, rhs[second], self.rng, max_cand=self.max_cand,
                                    stats=stats, log=self.log)
                stats["second_solutions"] += len(sols5)
                for combo5, S2 in sols5:
                    a2 = self._coords(fam5, arr2, blocks, combo5, S2, rhs[second])
                    if a2 is None:
                        continue
                    sp2 = self._refine(blocks, S2, a2, sp1, dC)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    codes2 = [{**codes_s[i], second: int(allowed2[i][combo5[i]])} for i in range(r)]
                    inv = [(self._vector({scan_pt: E.C[:, v], second: ov.vecs[int(combo5[r])]}), c5[2])]
                    hits.extend(self._assemble(opts, ords, blocks, dC, codes2, inv, {**Ss1, second: S2},
                                               {**cs1, second: a2}, stats))
        hits = SliceMatcher._dedupe(hits)
        stats["hits"] = len(hits)
        return hits, stats

    # -- the joint reconstruction of a pair block and a fifth term inside its span --
    @staticmethod
    def _consistent(A, b, tol=TOL):
        """Batched least squares: for the stack of systems A[n] z = b[n],
        the mask of consistent ones, their solutions and their ranks."""
        Z = (np.linalg.pinv(A) @ b[..., None])[..., 0]
        res = np.linalg.norm((A @ Z[..., None])[..., 0] - b, axis=1)
        scale = np.maximum(1.0, np.linalg.norm(b, axis=1))
        ranks = np.linalg.matrix_rank(A, tol=1e-8)
        return res < tol * scale, Z, ranks

    def _pair_brute(self, opts, ords, blocks, dC, codes_fixed, S_scan, F, rhs, stats):
        """The ambiguous configuration of a base with one pair block: at the
        scan point the residual after the visible ordinary terms lies in
        the span of the block's chosen translates S_scan, so the fifth
        term's slice v there may be any dictionary state inside that span.
        Decided by enumeration over v, the copies' codes at every point,
        the fifth term's code at the second point of its flat (if any) and
        the visible terms' admissible options there; the coefficients
        (c_1, c_2, c_5) solve the slice equations in the translate basis of
        the repeated state together with c_1 + c_2 = D, point by point in
        batches. codes_fixed holds the visible terms' codes at every point
        but the second point of the flat."""
        if len(blocks) != 1 or blocks[0].g != 2:
            raise UnpinnedFamily("the joint reconstruction of a block and a fifth term inside its span is "
                                 "written for one pair block only")
        stats["brute"] += 1
        E = self.E
        block = blocks[0]
        o = block.o
        D = dC[block.pos]
        r = len(opts)
        K = 1 << self.n2
        T = np.column_stack([o.vecs[4 * k] for k in range(K)])
        Tinv = np.linalg.inv(T)
        A_code = o.absent
        OPT = np.zeros((A_code + 1, K), dtype=complex)
        for k in range(K):
            for l in range(4):
                OPT[4 * k + l, k] = FOURTH_ARR[l]
        scan_pt = F[0]
        second = F[1] if len(F) == 2 else None
        exact_pts = [y for y in POINTS if y not in F]

        def rho(y, codes_y):
            res = rhs[y][2].copy()
            for i in range(r):
                res = res - dC[ords[i]] * opts[i].vecs[codes_y[i]]
            return Tinv @ res

        def extend(surv, rho_y, third, y):
            """Every survivor times every code pair of the copies at y, with
            the fifth term's column `third` (None: absent there)."""
            new = []
            for Ab, bb, jc in surv:
                A = np.zeros((len(pairs), K, 3), dtype=complex)
                A[:, :, 0] = OPT[pairs[:, 0]]
                A[:, :, 1] = OPT[pairs[:, 1]]
                if third is not None:
                    A[:, :, 2] = third[None, :]
                A = np.concatenate([np.broadcast_to(Ab, (len(pairs),) + Ab.shape), A], axis=1)
                b = np.concatenate([np.broadcast_to(bb, (len(pairs),) + bb.shape),
                                    np.broadcast_to(rho_y, (len(pairs), K))], axis=1)
                keep, Z, ranks = self._consistent(A, b)
                stats["brute_systems"] += len(pairs)
                for q in np.flatnonzero(keep):
                    new.append((A[q], b[q], {**jc, y: (int(pairs[q, 0]), int(pairs[q, 1]))}, Z[q], int(ranks[q])))
            return new

        Ps, _ = _projectors(blocks, S_scan)
        vs = self._table(blocks, S_scan, Ps).inside
        pairs = np.indices((A_code + 1, A_code + 1)).reshape(2, -1).T
        hits = []
        for v in vs:
            nu = Tinv @ E.C[:, v]
            ov = self.inv_options(v)
            omega = np.array([Tinv @ ov.vecs[j] for j in range(A_code)])            # (32, K)
            surv = [(np.array([[1.0, 1.0, 0.0]], dtype=complex), np.array([D], dtype=complex), {})]
            for z in exact_pts:
                surv = [(A, b, jc) for A, b, jc, _, _ in
                        extend(surv, rho(z, [codes_fixed[i][z] for i in range(r)]), None, z)]
                if not surv:
                    break
            if not surv:
                continue
            surv = extend(surv, rho(scan_pt, [codes_fixed[i][scan_pt] for i in range(r)]), nu, scan_pt)
            if not surv:
                continue
            if second is None:
                for A1, b1, jc, z, rank in surv:
                    if rank < 3:
                        raise UnpinnedFamily("joint reconstruction of the pair block and the fifth term: the "
                                             "coefficients are not determined by the slice equations")
                    if np.any(np.abs(z) < 1e-9):
                        continue
                    hits.extend(self._brute_assemble(opts, ords, dC, codes_fixed, o, jc, z,
                                                     self._vector({scan_pt: E.C[:, v]})))
                continue
            allowed2 = [self.compatible_codes(opts[i], codes_fixed[i], second) for i in range(r)]
            triples = np.indices((A_code + 1, A_code + 1, A_code)).reshape(3, -1).T
            for combo2 in itertools.product(*[range(len(cd)) for cd in allowed2]):
                c2 = [int(allowed2[i][combo2[i]]) for i in range(r)]
                codes = [{**codes_fixed[i], second: c2[i]} for i in range(r)]
                rho2 = rho(second, c2)
                for A1, b1, jc, z1, rank1 in surv:
                    found = []
                    if rank1 == 3:
                        if np.any(np.abs(z1) < 1e-9):
                            continue
                        lhs = z1[0] * OPT[pairs[:, 0]] + z1[1] * OPT[pairs[:, 1]]             # (1089, K)
                        rhs2 = rho2[None, :] - z1[2] * omega                                   # (32, K)
                        diff = np.linalg.norm(lhs[:, None, :] - rhs2[None, :, :], axis=2)
                        stats["brute_systems"] += len(pairs)
                        for qp, q5 in zip(*np.nonzero(diff < TOL * max(1.0, np.linalg.norm(rho2)))):
                            found.append((z1, (int(pairs[qp, 0]), int(pairs[qp, 1])), int(q5)))
                    else:
                        A = np.zeros((len(triples), K, 3), dtype=complex)
                        A[:, :, 0] = OPT[triples[:, 0]]
                        A[:, :, 1] = OPT[triples[:, 1]]
                        A[:, :, 2] = omega[triples[:, 2]]
                        A = np.concatenate([np.broadcast_to(A1, (len(triples),) + A1.shape), A], axis=1)
                        b = np.concatenate([np.broadcast_to(b1, (len(triples),) + b1.shape),
                                            np.broadcast_to(rho2, (len(triples), K))], axis=1)
                        keep, Z, ranks = self._consistent(A, b)
                        stats["brute_systems"] += len(triples)
                        for q in np.flatnonzero(keep):
                            if ranks[q] < 3:
                                raise UnpinnedFamily("joint reconstruction of the pair block and the fifth "
                                                     "term: the coefficients are not determined by the slice "
                                                     "equations")
                            if np.any(np.abs(Z[q]) < 1e-9):
                                continue
                            found.append((Z[q], (int(triples[q, 0]), int(triples[q, 1])), int(triples[q, 2])))
                    for z, j2, j5 in found:
                        fifth = self._vector({scan_pt: E.C[:, v], second: ov.vecs[j5]})
                        hits.extend(self._brute_assemble(opts, ords, dC, codes, o, {**jc, second: j2}, z, fifth))
        return hits

    def _brute_assemble(self, opts, ords, dC, codes, o, jc, z, fifth):
        """One survivor of the joint reconstruction: the copies' codes jc
        ({point: (code_1, code_2)}), the coefficients z = (c_1, c_2, c_5)."""
        copy_codes = [{y: jc[y][0] for y in POINTS}, {y: jc[y][1] for y in POINTS}]
        if not all(valid_term_codes(o, N1, cd) for cd in copy_codes):
            return []
        if copy_codes[0] == copy_codes[1]:
            return []
        if not all(valid_term_codes(opts[i], N1, codes[i]) for i in range(len(opts))):
            return []
        terms = [self._term(opts[i], codes[i]) for i in range(len(opts))]
        coeffs = [dC[ords[i]] for i in range(len(opts))]
        for cval, cd in zip(z[:2], copy_codes):
            terms.append(self._term(o, cd))
            coeffs.append(cval)
        terms.append(fifth)
        coeffs.append(z[2])
        return [self.SM.confirm(terms, coeffs, 0)]

    # ------------------------------------------------------ stage (gamma) --
    def _tbasis(self, v):
        """The translate basis of the dictionary state v (its 2^n Pauli
        class images, an orthogonal basis) over C and mod both primes, with
        the inverses over C and mod P1."""
        if v not in self._tbasis_cache:
            ov = self.inv_options(v)
            K = 1 << self.n2
            cols = [4 * k for k in range(K)]
            TC = np.column_stack([ov.vecs[c] for c in cols])
            T1 = np.ascontiguousarray(ov.m1[cols].T)
            T2 = np.ascontiguousarray(ov.m2[cols].T)
            T1inv = np.zeros((K, K), dtype=np.int64)
            for k in range(K):
                e = np.zeros(K, dtype=np.int64)
                e[k] = 1
                sol = _affine_solve_mod(T1, e, P1)
                if sol is None or sol[1].shape[1]:
                    raise UnpinnedFamily("the translate basis of a dictionary state is singular mod 65521")
                T1inv[:, k] = sol[0]
            self._tbasis_cache[v] = (np.linalg.inv(TC), T1, T2, T1inv)
        return self._tbasis_cache[v]

    def _translate_coords(self, v, res):
        """The coordinates of a residual in the translate basis of v over C
        and mod P2, whose zero patterns (and the one mod P1) must agree,
        and the support (classes with a nonzero coordinate)."""
        TCinv, T1, T2, _ = self._tbasis(v)
        xC = TCinv @ res[2]
        s1 = _affine_solve_mod(T1, res[0], P1)
        s2 = _affine_solve_mod(T2, res[1], P2)
        if s1 is None or s2 is None or s1[1].shape[1] or s2[1].shape[1]:
            raise UnpinnedFamily("the translate basis of a dictionary state is singular modulo a prime")
        zC = np.abs(xC) < 1e-9
        if not (np.array_equal(zC, s2[0] == 0) and np.array_equal(zC, s1[0] == 0)):
            raise UnpinnedFamily("the complex and modular coordinates of a residual in a translate basis "
                                 "disagree on their zero pattern")
        return xC, s2[0], [int(k) for k in np.flatnonzero(~zC)]

    def _pair_split(self, D, xC, x2, support):
        """The (c_4, c_5, code_4, code_5) assignments of two line terms with
        a common known slice at one point of their line and the residual
        coordinates (xC over C, x2 mod P2) in its translate basis at the
        other point, with c_4 + c_5 = D (a triple). Two classes in the
        support: c_4 i^l4 = a_1, c_5 i^l5 = a_2; one class: c_4 i^l4 + c_5
        i^l5 = a with l4 != l5 (equal phases would make the two terms
        one); an empty support (the pair cancels there inside one class):
        c_4 i^l4 + c_5 i^l5 = 0 with l4 != l5, over every class. Decided
        numerically and mod P2, which must agree."""
        i2 = self.F2.ipow
        DC, D2 = D[2], int(D[1])
        out = []

        def agree(num, mod):
            if num != mod:
                raise UnpinnedFamily("numeric and modular decisions of a pair split disagree")
            return num

        if len(support) == 2:
            k1, k2 = support
            a1, a2, b1, b2 = xC[k1], xC[k2], int(x2[k1]), int(x2[k2])
            for l4 in range(4):
                for l5 in range(4):
                    c4, c5 = a1 / FOURTH_ARR[l4], a2 / FOURTH_ARR[l5]
                    num = abs(c4 + c5 - DC) < TOL * max(1.0, abs(DC), abs(c4), abs(c5))
                    mod = (b1 * pow(int(i2[l4]), P2 - 2, P2) + b2 * pow(int(i2[l5]), P2 - 2, P2) - D2) % P2 == 0
                    if agree(num, mod):
                        out.append((c4, c5, 4 * k1 + l4, 4 * k2 + l5))
        elif len(support) == 1:
            k = support[0]
            a, b = xC[k], int(x2[k])
            for l4 in range(4):
                for l5 in range(4):
                    if l4 == l5:
                        continue
                    c4 = (a - DC * FOURTH_ARR[l5]) / (FOURTH_ARR[l4] - FOURTH_ARR[l5])
                    c5 = DC - c4
                    den = (int(i2[l4]) - int(i2[l5])) % P2
                    c4m = (b - D2 * int(i2[l5])) * pow(den, P2 - 2, P2) % P2
                    c5m = (D2 - c4m) % P2
                    if agree(abs(c4) > 1e-9 and abs(c5) > 1e-9, c4m != 0 and c5m != 0):
                        out.append((c4, c5, 4 * k + l4, 4 * k + l5))
        else:
            for k in range(1 << self.n2):
                for l4 in range(4):
                    for l5 in range(4):
                        if l4 == l5:
                            continue
                        c4 = DC / (1 - FOURTH_ARR[l4 - l5])
                        c5 = DC - c4
                        out.append((c4, c5, 4 * k + l4, 4 * k + l5))
        return out

    def _same_orbit(self, v, w):
        """True when the dictionary states v and w are phased Pauli
        translates of each other."""
        return self.orbit_id(v) == self.orbit_id(w)

    def run_two(self, cover, pair):
        """(hits, stats) for the base 3-cover `cover` with two invisible
        terms on the flats `pair` (two names of FLATS)."""
        F = [FLATS[pair[0]], FLATS[pair[1]]]
        stats = {"flats": list(pair), "kappa": None, "distinct": 0, "blocks": [], "order": [], "modes": [],
                 "exact_solutions": [], "scan_candidates": 0, "scan_solutions": 0, "scan2_collisions": 0,
                 "scan2_solutions": 0, "absent_at_flat": 0, "same_state": 0, "cancelling": 0, "pair_solutions": 0,
                 "candidates": 0, "hits": 0, "refused": False, "native": False}
        if len(cover) != 3 or len(set(cover)) != 3:
            raise UnpinnedFamily("stage (gamma) needs three distinct base states")
        setup = self._setup(cover, stats, allow_blocks=False)
        if setup is None:
            return [], stats
        distinct, fam, blocks, ords, opts, arrays, dvs, dC = setup
        E = self.E
        r = len(opts)
        rhs = {y: self.rhs(y) for y in POINTS}
        present = {y: [j for j in (0, 1) if y in F[j]] for y in POINTS}
        # processing order: the exact points, then the point with the fewest fresh terms
        order, known = [y for y in POINTS if not present[y]], set()
        rem = [y for y in POINTS if present[y]]
        while rem:
            y = min(rem, key=lambda q: (sum(1 for j in present[q] if j not in known), q))
            order.append(y)
            known |= set(present[y])
            rem.remove(y)
        stats["order"] = order
        known_m = set()
        for y in order:
            n_fresh = sum(1 for j in present[y] if j not in known_m)
            stats["modes"].append([y, {0: "exact", 1: "scan", 2: "scan2"}[n_fresh]])
            known_m |= set(present[y])
        both_lines = len(F[0]) == 2 and len(F[1]) == 2
        # a state: the visible codes, the invisible terms {j: ("known", first point, v, c, {point: code})
        # or ("slices", {point: vector}, c)}, and the pair state after a scan2 with equal states
        states = [([dict() for _ in range(r)], {}, None)]
        exact_here = {}
        for y in order:
            new = []
            for codes, inv, ps in states:
                allowed = [self.compatible_codes(opts[i], codes[i], y) for i in range(r)]
                arr = [_restrict(arrays[i], allowed[i]) for i in range(r)]
                if ps is not None:
                    new.extend(self._pair_continue(y, ps, codes, allowed, arr, dvs, rhs, stats))
                    continue
                here = present[y]
                fresh = [j for j in here if j not in inv]
                known_here = [j for j in here if j in inv]
                arr_k, cs_k = list(arr), []
                for j in known_here:
                    ovj = self.inv_options(inv[j][2])
                    arr_k.append(_restrict(ovj.arrays(), range(ovj.absent)))
                    cs_k.append(inv[j][3])
                dv_k = tuple(np.append(dvs[f], [c[f] for c in cs_k]).astype(dvs[f].dtype) for f in range(3))

                def with_known(combo, inv):
                    inv2 = dict(inv)
                    for t, j in enumerate(known_here):
                        _, fp, v, c, cd = inv[j]
                        inv2[j] = ("known", fp, v, c, {**cd, y: int(combo[r + t])})
                    return inv2

                def with_codes(combo):
                    return [{**codes[i], y: int(allowed[i][combo[i]])} for i in range(r)]

                if not fresh:
                    if known_here:
                        known = [(self.inv_options(inv[j][2]), inv[j][3]) for j in known_here]
                        sols = self._exact_known(arr, dvs, known, rhs[y], stats)
                    else:
                        sols = [combo for combo, _ in solve_slice(arr, [], fam, rhs[y], self.rng,
                                                                  max_cand=self.max_cand, stats=stats, log=self.log)]
                    exact_here[y] = exact_here.get(y, 0) + len(sols)
                    for combo in sols:
                        new.append((with_codes(combo), with_known(combo, inv), None))
                elif len(fresh) == 1:
                    for combo, _, fr, _, _ in self._coordinate_scan(arr_k, [], dv_k, rhs[y], stats, "scan"):
                        if fr is None:
                            stats["absent_at_flat"] += 1
                            continue
                        v, c = fr
                        inv2 = with_known(combo, inv)
                        inv2[fresh[0]] = ("known", y, v, c, {})
                        new.append((with_codes(combo), inv2, None))
                else:
                    for combo, kind, data in self._rank2_scan(arr, dvs, rhs[y], stats, "scan2"):
                        if kind == "rank2":
                            v, w, cv, cw = data
                            new.append((with_codes(combo), {0: ("known", y, v, cv, {}), 1: ("known", y, w, cw, {})},
                                        None))
                        elif kind == "stab":
                            if both_lines:                  # otherwise two equal point terms: one term
                                stats["same_state"] += 1
                                new.append((with_codes(combo), {}, ("same", y, data[0], data[1])))
                        elif both_lines:                    # otherwise two cancelling copies of one term
                            stats["cancelling"] += 1
                            new.append((with_codes(combo), {}, ("cancel", y)))
            if y in exact_here:
                stats["exact_solutions"].append(exact_here[y])
            states = new
            if not states:
                break
        hits = []
        for codes, inv, ps in states:
            if ps is not None or len(inv) != 2:
                raise UnpinnedFamily("a stage (gamma) state survived the last point undetermined")
            invisible = []
            for j in (0, 1):
                ent = inv[j]
                if ent[0] == "known":
                    _, fp, v, c, cd = ent
                    slices = {fp: E.C[:, v]}
                    for y2, code in cd.items():
                        slices[y2] = self.inv_options(v).vecs[code]
                    cC = c[2]
                else:
                    _, slices, cC = ent
                if set(slices) != set(F[j]):
                    raise UnpinnedFamily("an invisible term was not determined on its whole flat")
                invisible.append((self._vector(slices), cC))
            if codes_key([invisible[0][0]]) == codes_key([invisible[1][0]]):
                continue                                    # the same term twice
            hits.extend(self._assemble(opts, ords, [], dC, codes, invisible, {}, {}, stats))
        hits = SliceMatcher._dedupe(hits)
        stats["hits"] = len(hits)
        return hits, stats

    def _pair_continue(self, y, ps, codes, allowed, arr, dvs, rhs, stats):
        """The second point of a line shared by two invisible terms after a
        scan2 with equal states ('same': the common state v and D = c_4 +
        c_5 known) or a cancelling pair ('cancel': c_5 = -c_4, the common
        state unknown): the pair contributes a vector of the span of at
        most two translates of the common state. Returns completed states
        with both invisible terms as explicit slices."""
        E = self.E
        r = len(arr)
        out = []

        def with_codes(combo):
            return [{**codes[i], y: int(allowed[i][combo[i]])} for i in range(r)]

        if ps[0] == "same":
            _, y0, v, D = ps
            ov = self.inv_options(v)
            R, sizes = self._all_residuals([a[0] for a in arr], dvs[0], rhs[y][0] % P1)
            X1 = (R @ self._tbasis(v)[3].T) % P1
            for q in np.flatnonzero((X1 != 0).sum(axis=1) <= 2):
                combo = tuple(int(c) for c in np.unravel_index(int(q), sizes))
                res = self._residuals(arr, dvs, combo, rhs[y])
                xC, x2, support = self._translate_coords(v, res)
                if len(support) > 2:
                    continue
                for c4, c5, j4, j5 in self._pair_split(D, xC, x2, support):
                    stats["pair_solutions"] += 1
                    out.append((with_codes(combo),
                                {0: ("slices", {y0: E.C[:, v], y: ov.vecs[j4]}, c4),
                                 1: ("slices", {y0: E.C[:, v], y: ov.vecs[j5]}, c5)}, None))
            return out
        _, y0 = ps
        for combo, kind, data in self._rank2_scan(arr, dvs, rhs[y], stats, "scan2"):
            if kind == "zero":
                continue                                    # both terms absent at a point of their line
            if kind == "stab":
                # both copies in one class at y: c_4 (i^l4 - i^l5) = a, c_5 = -c_4, the common slice at
                # y0 any class representative of the Pauli orbit of the state (its phase is absorbed)
                w, a = data
                ow = self.inv_options(w)
                for l4 in range(4):
                    for l5 in range(l4 + 1, 4):
                        c4 = a[2] / (FOURTH_ARR[l4] - FOURTH_ARR[l5])
                        for t in range(ow.absent):
                            stats["pair_solutions"] += 1
                            tv = ow.vecs[t]
                            out.append((with_codes(combo),
                                        {0: ("slices", {y0: tv, y: FOURTH_ARR[l4] * E.C[:, w]}, c4),
                                         1: ("slices", {y0: tv, y: FOURTH_ARR[l5] * E.C[:, w]}, -c4)}, None))
                continue
            # distinct classes at y: the terms are (t, v) and (mu t, w) with t a phased translate of
            # v (and of w) and mu a fourth root of unity, and they cancel at y0: c_v + mu c_w = 0
            v, w, cv, cw = data
            if not self._same_orbit(v, w):
                continue
            ov = self.inv_options(v)
            for l in range(4):
                num = abs(cv[2] + FOURTH_ARR[l] * cw[2]) < TOL * max(1.0, abs(cv[2]))
                mod = (cv[1] + int(self.F2.ipow[l]) * cw[1]) % P2 == 0
                if num != mod:
                    raise UnpinnedFamily("numeric and modular decisions of a cancelling pair disagree")
                if not num:
                    continue
                for t in range(ov.absent):
                    stats["pair_solutions"] += 1
                    tv = ov.vecs[t]
                    out.append((with_codes(combo), {0: ("slices", {y0: tv, y: E.C[:, v]}, cv[2]),
                                                    1: ("slices", {y0: FOURTH_ARR[l] * tv, y: E.C[:, w]}, cw[2])},
                                None))
        return out


class PlantedInvisibleMatcher(InvisibleMatcher):
    """InvisibleMatcher run against a planted five-qubit target Psi (a sum of
    stabilizer states with Gaussian-integer coefficients) instead of psi_5:
    the base coefficients are solved against Psi's slice at 00 and every
    slice equation against Psi's slice there. confirm() still measures the
    residual against psi_5, so hits are compared by their term codes."""

    def __init__(self, E, Psi, **kw):
        rows = Psi.reshape(1 << N1, -1)
        E2 = copy.copy(E)
        E2.psi = rows[X0]
        E2.psi1, E2.psi2 = to_field(E.F1, rows[X0]), to_field(E.F2, rows[X0])
        super().__init__(E2, **kw)
        self._rows = rows

    def rhs(self, x):
        v = self._rows[x]
        return to_field(self.F1, v), to_field(self.F2, v), v
