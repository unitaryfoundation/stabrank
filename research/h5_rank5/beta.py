"""Stage (beta) of the rank-5 exclusion of |H>^5: four visible terms at the
base point x_0 and one invisible line term on the diagonal missing x_0
(docs/notes/h5_rank5_exclusion.md, section 3).

Setting. Slice psi_5 = sum_{i=1}^5 c_i s_i along qubits 1, 2. At x_0 in
{00, 01} four terms are nonzero; their slices u_1, ..., u_4 form a full
4-cover of psi_3 (the base, a 4-multiset over the 1080 three-qubit
stabilizer states) with coefficients d_i = c_i / alpha_{x_0}. The fifth
term vanishes at x_0; by Fact 1 (every term is full along every qubit) its
flat is the diagonal line {x_0 + 01, x_0 + 10}, so it is absent at
x_b = x_0 + 11 and present at both coordinate points x_a1 = x_0 + 01 and
x_a2 = x_0 + 10, where its slices are c_5 v and c_5 i^l Q v for a
three-qubit stabilizer state v, a Pauli Q on the three unsliced qubits and
l in Z_4 (the structure lemma for a line term with base slice v).

What the matcher assumes and what it enumerates. Nothing about the flats
of the four visible terms: at every one of the three other points each is
absent or one of the 32 phased Pauli translates of its base slice, and the
presence pattern is filtered to the subspaces of F_2^2 through x_0 (the
composite point's code follows the structure lemma for a plane term). The
fifth term is taken present at the two coordinate points and absent at
x_b (Fact 1); a zero residual at one coordinate point is also accepted, so
a fifth term that is a point term at the other coordinate point would be
found as well. A repeated base state (the six (2, 1, 1) multisets over the
two full 3-covers) is a block in the sense of slice_cover: its copies
contribute an arbitrary vector of the span of at most two Pauli translates
at every slice, the equations are projected onto the annihilator of the
chosen translates, and the copies are reconstructed at the end by
slice_cover.reconstruct_block (cancelling copies allowed).

The three slice equations, in order:

  x_b   exact: sum_i d_i w_i + block = alpha_{x_b} / alpha_{x_0} psi_3 over
        33 options per visible term and the translate sets of the block,
        by slice_cover.solve_slice (meet in the middle mod 65521, every
        candidate decided over C and mod 2013265921);
  x_a1  residual: r = rhs - sum_i d_i w_i - block must be c_5 v for a
        dictionary state v, or zero. All 33^4 (or 33^2 x 37) option
        combinations are formed at once mod 65521, the residual projected
        onto the annihilator of the block's translates, normalised (first
        nonzero entry 1) and looked up in the table of the normalised
        projected dictionary states by a random functional; every match is
        re-decided exactly: the system (translates | v) (a, c_5) = r is
        solved over C, mod 65521 and mod 2013265921 and must have a unique
        solution with every entry nonzero;
  x_a2  exact, given v and c_5: the fifth term is one of the 32 phased
        translates of v or absent, each visible term's options are those
        compatible with its codes at x_a1 and x_b (a plane term's code is
        fixed up to the quadratic sign, a line term's is absent, a term
        absent at both is free), and solve_slice decides the equation with
        the fifth term as a fifth ordinary term of known coefficient; when
        the fifth term was absent at x_a1 the residual test of x_a1 is run
        at x_a2 instead (a point term there).

Every surviving state is assembled into five terms (the copies of a block
through reconstruct_block) and confirmed as at H^6: residual against
psi_5, rank, independence, nonzero coefficients, and the span condition mod
2013265921 (SliceMatcher.confirm); batch.py re-decides every hit again
from its phase codes (common.decide_terms).

Refusals and undecided runs. A base whose coefficient family is empty or
has a dead ordinary coefficient is refused (the list excludes such
multisets, so a refusal fails the aggregate). A coordinate-point residual
that lies in the span of the block's chosen translates with a nonempty
translate set (the fifth term's slice could then be a translate of the
repeated state and the split between the block and the fifth term is a
family), a residual whose complex and modular solves disagree, or a base
with a coefficient family raise slice_cover.UnpinnedFamily, which batch.py
records as undecided; the aggregate fails on any such record.
"""
from __future__ import annotations

import copy
import itertools
import types

import numpy as np

from slice_cover import (P1, P2, Block, Family, SliceMatcher, TermOptions, UnpinnedFamily,  # noqa: E402
                         _affine_solve_C, _affine_solve_mod, _projectors, _refine_split, exact_codes,
                         reconstruct_block, solve_slice, valid_term_codes)

N1 = 2
COORD = (1, 2)            # the coordinate offsets: x_a1 = x_0 ^ 1, x_a2 = x_0 ^ 2
DIAG = 3                  # the offset of x_b = x_0 ^ 11
FOURTH_ARR = np.array([1, 1j, -1, -1j])


def to_field(F, v):
    """A vector with Gaussian-integer entries reduced to F_p."""
    re, im = np.round(v.real).astype(np.int64), np.round(v.imag).astype(np.int64)
    if not np.allclose(v, re + 1j * im, atol=1e-9):
        raise AssertionError("vector entries are not Gaussian integers")
    return (re + F.i * im) % F.p


class BetaMatcher:
    """Every decomposition of psi_5 whose slice at x_0 along qubits 1, 2 has
    exactly the four terms of `cover` visible and a fifth term on the
    diagonal line missing x_0."""

    def __init__(self, E, seed=31, verbose=False, max_cand=2_000_000):
        self.E = E
        self.n2 = E.n
        self.dim = 1 << E.n
        self.F1, self.F2 = E.F1, E.F2
        self.SM = SliceMatcher(E, N1, native=False, seed=seed, max_cand=max_cand)
        self.max_cand = max_cand
        self.rng = np.random.default_rng(seed + 1)
        self.verbose = verbose
        self._tables = {}          # (block state, translate set) pairs -> projected dictionary table
        self._fifth = {}           # dictionary index -> TermOptions of the fifth term's base slice

    # -- pieces shared with the stage (alpha) matcher --
    def options(self, idx):
        return self.SM.options(idx)

    def fifth_options(self, v):
        if v not in self._fifth:
            self._fifth[v] = TermOptions(self.E.C[:, v], self.n2, self.F1, self.F2)
        return self._fifth[v]

    def rhs(self, x0, x):
        return self.SM.rhs(x0, x)

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # -- the projected dictionary tables --
    def _table(self, blocks, Ssel, Ps):
        """The dictionary states projected onto the annihilator of the chosen
        translates, normalised mod P1 and keyed by a random functional; the
        states inside the span (zero projection over C) are excluded, and
        the zero pattern mod P1 must agree with the one over C."""
        key = tuple((b.idx, tuple(S)) for b, S in zip(blocks, Ssel))
        if key in self._tables:
            return self._tables[key]
        E = self.E
        if Ps is None:
            PV = np.ascontiguousarray(E.U1.T)                         # (dim, N)
            PVC = E.C
        else:
            PV = (Ps[0] @ E.U1.T) % P1                                # (dim', N)
            PVC = Ps[2] @ E.C
        inside = np.linalg.norm(PVC, axis=0) < 1e-7
        zero1 = ~np.any(PV != 0, axis=0)
        if not np.array_equal(inside, zero1):
            raise UnpinnedFamily("a dictionary state's projection vanishes mod 65521 but not over C, or the "
                                 "reverse; the coordinate-point lookup table is unreliable for this translate set")
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

    def _scan(self, popts, dv, prhs, table):
        """All option combinations at once mod P1: the combinations whose
        projected residual is zero, and the (combination, dictionary state)
        pairs whose normalised residual equals the state's normalised
        projection. A superset of the exact solutions."""
        dimp = prhs.shape[0]
        R = (prhs % P1)[None, :]
        for i, o in enumerate(popts):
            R = (R[:, None, :] - ((int(dv[i]) * o) % P1)[None, :, :]) % P1
            R = R.reshape(-1, dimp)
        sizes = tuple(len(o) for o in popts)
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
        """rhs minus the visible terms' contribution, in the three fields."""
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
        blocks and on the fifth term's slice v (None when the fifth term is
        absent here): (a, c_5, ambiguous) with a unique over C, mod P1 and
        mod P2 and every entry nonzero, or None when the residual is not in
        the span (the modular superset let it through). Raises
        UnpinnedFamily when the span membership holds but the solution is
        not unique. `ambiguous` is set when the residual lies in the span of
        a nonempty translate set with the fifth term absent: the fifth
        term's slice could then be a dictionary state inside that span, a
        case the caller decides by the joint reconstruction `_pair_brute`."""
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
            if np.linalg.norm(res[2]) > 1e-7:
                return None
            return np.zeros(0, dtype=complex), None, False
        AC = np.column_stack(cols[2])
        solC = _affine_solve_C(AC, res[2])
        if solC is None:
            return None
        if solC[1].shape[1]:
            raise UnpinnedFamily("the chosen translates and the fifth term's slice are dependent at a "
                                 "coordinate point; the split is a family")
        A1 = np.column_stack(cols[0]) % P1
        A2 = np.column_stack(cols[1]) % P2
        sol1 = _affine_solve_mod(A1, res[0], P1)
        sol2 = _affine_solve_mod(A2, res[1], P2)
        if sol1 is None or sol2 is None or sol1[1].shape[1] or sol2[1].shape[1]:
            raise UnpinnedFamily("the complex and modular solves of a coordinate-point residual disagree")
        x = solC[0]
        if np.any(np.abs(x) < 1e-9):
            return None                # a zero block coordinate duplicates a smaller translate set
        if v is None:
            stats["ambiguous"] += 1
            return x, None, True
        return x[:k], (int(sol1[0][-1]), int(sol2[0][-1]), complex(x[-1])), False

    def _coordinate_scan(self, arrays, blocks, dvs, rhs, stats, tag):
        """Solutions of a coordinate-point equation with an unknown fifth
        slice: a list of (combo, Ssel, fifth, a) with fifth = None (the
        residual is zero: the fifth term is absent here) or (v, c_5) in the
        three fields, and a the block coordinates."""
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

    @staticmethod
    def xa2_codes(o, c1, cb):
        """The option codes of a visible term at x_a2 compatible with its
        codes at x_a1 and x_b: a plane term's code is the composition
        partner of its x_a1 code for its x_b code (two quadratic signs), a
        term present at exactly one of x_a1, x_b is a line term and absent
        at x_a2, a term absent at both is a point term or the line through
        x_a2 (free)."""
        A = o.absent
        if c1 != A and cb != A:
            return [c2 for c2 in range(A) if any(o.compose(c1, c2, s) == cb for s in (0, 1))]
        if c1 != A or cb != A:
            return [A]
        return list(range(A + 1))

    def _refine(self, blocks, Ssel, a, sp, dC):
        """The pair blocks' admissible coefficient splits after a slice with
        block coordinates a (as SliceMatcher._join_blocks)."""
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

    def _block_data(self, blocks, Ssels, coords):
        """reconstruct_block's per-block data: (offset, {class: coordinate})
        for the offsets 1, 2, 3 in order."""
        per_block = [[] for _ in blocks]
        for off in (1, 2, 3):
            Ssel, a = Ssels[off], coords[off]
            pos = 0
            for bi, (b, S) in enumerate(zip(blocks, Ssel)):
                per_block[bi].append((off, {k: a[pos + j] for j, k in enumerate(S)}))
                pos += len(S)
        return per_block

    def _assemble(self, x0, opts, ords, blocks, dC, codes, fifth_terms, block_data, stats):
        for i, o in enumerate(opts):
            if not valid_term_codes(o, N1, codes[i]):
                return []
        terms, coeffs = [], []
        for i, o in enumerate(opts):
            t = np.zeros((1 << N1, self.dim), dtype=complex)
            t[x0] = o.u
            for off, cd in codes[i].items():
                t[x0 ^ off] = o.vecs[cd]
            terms.append(t.ravel())
            coeffs.append(dC[ords[i]])
        for vec, c in fifth_terms:
            terms.append(vec)
            coeffs.append(c)
        if not blocks:
            return [self.SM.confirm(terms, coeffs, 0)]
        recon = [reconstruct_block(b.o, b.g, dC[b.pos], block_data[bi], N1, self.rng)
                 for bi, b in enumerate(blocks)]
        stats["reconstructions"] += 1
        hits = []
        for choice in itertools.product(*recon):
            terms2, cs2, degenerate = list(terms), list(coeffs), False
            for b, (copies, deg) in zip(blocks, choice):
                degenerate |= deg
                for cval, cd in copies:
                    t = np.zeros((1 << N1, self.dim), dtype=complex)
                    t[x0] = b.o.u
                    for x, code in cd.items():
                        t[x0 ^ x] = b.o.vecs[code]
                    terms2.append(t.ravel())
                    cs2.append(cval)
            hits.append(self.SM.confirm(terms2, cs2, int(degenerate)))
        return hits

    # -- the joint reconstruction of a pair block and a fifth term inside its span --

    @staticmethod
    def _consistent(A, b, tol=1e-7):
        """Batched least squares: for the stack of systems A[n] z = b[n],
        the mask of consistent ones, their solutions and their ranks."""
        Z = (np.linalg.pinv(A) @ b[..., None])[..., 0]
        res = np.linalg.norm((A @ Z[..., None])[..., 0] - b, axis=1)
        scale = np.maximum(1.0, np.linalg.norm(b, axis=1))
        ranks = np.linalg.matrix_rank(A, tol=1e-8)
        return res < tol * scale, Z, ranks

    def _pair_brute(self, x0, opts, ords, blocks, dC, arrays, cb, c1codes, codes2, S1, rhs, stats):
        """The ambiguous configuration of a base with one pair block: the
        residual at x_a1 after the visible ordinary terms lies in the span of
        the block's chosen translates S1, so the fifth term's slice v there
        may be any dictionary state inside that span and the x_a1 equation
        cannot separate it from the copies. Decided by enumeration: for
        every such v, every assignment of the two copies' codes at x_b, x_a1
        and x_a2 (absent or one of 32 phased translates each), every code of
        the fifth term at x_a2 (a phased translate of v or absent) and every
        admissible option of the visible terms at x_a2, the coefficients
        (c_1, c_2, c_5) are the solution of the linear system formed by the
        three slice equations in the translate basis of the repeated state
        and c_1 + c_2 = D; the system is solved in batches, offset by offset
        (x_b, then x_a1), keeping the consistent assignments; when the first
        two offsets pin the three coefficients (the usual case) the x_a2
        equation is matched directly with known coefficients over the
        copies' 33 x 33 codes and the fifth term's 33, otherwise the full
        system is solved per triple. A surviving assignment whose system
        leaves a coefficient free raises UnpinnedFamily. Every survivor is
        assembled and confirmed like any
        other hit. The visible terms' codes at x_b and x_a1 are fixed by the
        caller (cb, c1codes)."""
        if len(blocks) != 1 or blocks[0].g != 2:
            raise UnpinnedFamily("the joint reconstruction of a block and a fifth term inside its span is "
                                 "written for one pair block only")
        stats["brute"] += 1
        E = self.E
        block = blocks[0]
        o = block.o
        D = dC[block.pos]
        r = len(opts)
        T = np.column_stack([o.vecs[4 * k] for k in range(1 << self.n2)])
        Tinv = np.linalg.inv(T)
        A_code = o.absent
        OPT = np.zeros((A_code + 1, 1 << self.n2), dtype=complex)
        for k in range(1 << self.n2):
            for l in range(4):
                OPT[4 * k + l, k] = FOURTH_ARR[l]

        def rho(off, combo):
            res = rhs[off][2].copy()
            for i, c in enumerate(combo):
                res = res - dC[ords[i]] * arrays[i][2][c]
            return Tinv @ res

        Ps, _ = _projectors(blocks, S1)                # S1 is the translate-set tuple over the blocks
        vs = self._table(blocks, S1, Ps).inside
        hits = []
        pairs = np.indices((A_code + 1, A_code + 1)).reshape(2, -1).T
        rho3, rho1 = rho(3, cb), rho(1, c1codes)
        for v in vs:
            nu = Tinv @ E.C[:, v]
            ov = self.fifth_options(v)
            omega = np.array([Tinv @ ov.vecs[j] for j in range(A_code + 1)])       # (33, 8)
            # x_b: the fifth term is absent
            A = np.zeros((len(pairs), 9, 3), dtype=complex)
            A[:, :8, 0] = OPT[pairs[:, 0]]
            A[:, :8, 1] = OPT[pairs[:, 1]]
            A[:, 8, 0] = A[:, 8, 1] = 1.0
            b = np.zeros((len(pairs), 9), dtype=complex)
            b[:, :8] = rho3
            b[:, 8] = D
            keep, _, _ = self._consistent(A, b)
            stats["brute_systems"] += len(pairs)
            surv_b = [(A[q], b[q], (int(pairs[q, 0]), int(pairs[q, 1]))) for q in np.flatnonzero(keep)]
            # x_a1: the fifth term is c_5 v
            surv_1 = []
            for Ab, bb, jb in surv_b:
                A = np.zeros((len(pairs), 8, 3), dtype=complex)
                A[:, :, 0] = OPT[pairs[:, 0]]
                A[:, :, 1] = OPT[pairs[:, 1]]
                A[:, :, 2] = nu[None, :]
                A = np.concatenate([np.broadcast_to(Ab, (len(pairs),) + Ab.shape), A], axis=1)
                b = np.concatenate([np.broadcast_to(bb, (len(pairs),) + bb.shape),
                                    np.broadcast_to(rho1, (len(pairs), 8))], axis=1)
                keep, Z, ranks = self._consistent(A, b)
                stats["brute_systems"] += len(pairs)
                for q in np.flatnonzero(keep):
                    surv_1.append((A[q], b[q], jb, (int(pairs[q, 0]), int(pairs[q, 1])), Z[q], int(ranks[q])))
            if not surv_1:
                continue
            # x_a2: the copies, the fifth term's translate (or absence) and the visible terms' options.
            # When x_b and x_a1 pin (c_1, c_2, c_5) (the usual case) the x_a2 equation has known
            # coefficients and is matched directly; otherwise the full system is solved per triple.
            triples = np.indices((A_code + 1, A_code + 1, A_code + 1)).reshape(3, -1).T
            for combo2 in itertools.product(*[range(len(cd)) for cd in codes2]):
                c2 = tuple(int(codes2[i][combo2[i]]) for i in range(r))
                rho2 = rho(2, c2)
                for A1, b1, jb, j1, z1, rank1 in surv_1:
                    found = []
                    if rank1 == 3:
                        if np.any(np.abs(z1) < 1e-9):
                            continue
                        lhs = z1[0] * OPT[pairs[:, 0]] + z1[1] * OPT[pairs[:, 1]]          # (1089, 8)
                        rhs2 = rho2[None, :] - z1[2] * omega                                # (33, 8)
                        diff = np.linalg.norm(lhs[:, None, :] - rhs2[None, :, :], axis=2)
                        stats["brute_systems"] += len(pairs)
                        for qp, q5 in zip(*np.nonzero(diff < 1e-7 * max(1.0, np.linalg.norm(rho2)))):
                            found.append((z1, (int(pairs[qp, 0]), int(pairs[qp, 1])), int(q5)))
                    else:
                        A = np.zeros((len(triples), 8, 3), dtype=complex)
                        A[:, :, 0] = OPT[triples[:, 0]]
                        A[:, :, 1] = OPT[triples[:, 1]]
                        A[:, :, 2] = omega[triples[:, 2]]
                        A = np.concatenate([np.broadcast_to(A1, (len(triples),) + A1.shape), A], axis=1)
                        b = np.concatenate([np.broadcast_to(b1, (len(triples),) + b1.shape),
                                            np.broadcast_to(rho2, (len(triples), 8))], axis=1)
                        keep, Z, ranks = self._consistent(A, b)
                        stats["brute_systems"] += len(triples)
                        for q in np.flatnonzero(keep):
                            if ranks[q] < 3:
                                raise UnpinnedFamily("joint reconstruction of the pair block and the fifth term: "
                                                     "the coefficients are not determined by the three slice "
                                                     "equations")
                            if np.any(np.abs(Z[q]) < 1e-9):
                                continue
                            found.append((Z[q], (int(triples[q, 0]), int(triples[q, 1])), int(triples[q, 2])))
                    for z, j2, j5 in found:
                        copy_codes = [{1: j1[0], 2: j2[0], 3: jb[0]}, {1: j1[1], 2: j2[1], 3: jb[1]}]
                        if not all(valid_term_codes(o, N1, cd) for cd in copy_codes):
                            continue
                        codes = [{1: c1codes[i], 2: c2[i], 3: cb[i]} for i in range(r)]
                        if not all(valid_term_codes(opts[i], N1, codes[i]) for i in range(r)):
                            continue
                        terms, coeffs = [], []
                        for i, oo in enumerate(opts):
                            t = np.zeros((1 << N1, self.dim), dtype=complex)
                            t[x0] = oo.u
                            for off, cd in codes[i].items():
                                t[x0 ^ off] = oo.vecs[cd]
                            terms.append(t.ravel())
                            coeffs.append(dC[ords[i]])
                        for cval, cd in zip(z[:2], copy_codes):
                            t = np.zeros((1 << N1, self.dim), dtype=complex)
                            t[x0] = o.u
                            for off, code in cd.items():
                                t[x0 ^ off] = o.vecs[code]
                            terms.append(t.ravel())
                            coeffs.append(cval)
                        terms.append(self._fifth_vector(x0, v, j5))
                        coeffs.append(z[2])
                        hits.append(self.SM.confirm(terms, coeffs, 0))
        return hits

    def _fifth_vector(self, x0, v, code2):
        """The fifth term as a (2^n1 x 2^n2) vector: slice v at x_a1 and its
        phased translate `code2` at x_a2 (absent when code2 is the absent
        code: a point term at x_a1)."""
        ov = self.fifth_options(v)
        t = np.zeros((1 << N1, self.dim), dtype=complex)
        t[x0 ^ COORD[0]] = self.E.C[:, v]
        t[x0 ^ COORD[1]] = ov.vecs[code2]
        return t.ravel()

    def _point_vector(self, x0, off, v):
        t = np.zeros((1 << N1, self.dim), dtype=complex)
        t[x0 ^ off] = self.E.C[:, v]
        return t.ravel()

    def run(self, cover, x0):
        """(hits, stats) for the base 4-multiset `cover` at x0 in {0, 1}."""
        E = self.E
        stats = {"kappa": None, "distinct": 0, "blocks": [], "b_solutions": 0, "split_pruned": 0,
                 "a1_candidates": 0, "a1_solutions": 0, "a2_candidates": 0, "a2_solutions": 0,
                 "a2_point": 0, "candidates": 0, "reconstructions": 0, "ambiguous": 0, "point_ambiguous": 0,
                 "brute": 0, "brute_systems": 0, "hits": 0, "refused": False, "native": False}
        distinct = sorted(set(cover))
        mult = {u: cover.count(u) for u in distinct}
        fam = Family.from_cover(E, distinct)
        if fam is None:
            stats["refused"] = True
            return [], stats
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
            return [], stats
        stats.update(kappa=fam.kappa, distinct=len(distinct), blocks=[b.g for b in blocks])
        if fam.kappa:
            raise UnpinnedFamily(f"stage (beta) base with a {fam.kappa}-parameter coefficient family; the "
                                 "list has none and the matcher takes the base coefficients as a point")
        ords = [i for i in range(len(distinct)) if i not in bpos]
        opts = [self.options(distinct[i]) for i in ords]
        arrays = [o.arrays() for o in opts]
        d1, d2, dC = fam.parts[0][0], fam.parts[1][0], fam.parts[2][0]
        dvs = (d1[ords], d2[ords], dC[ords])
        rhs = {off: self.rhs(x0, x0 ^ off) for off in (1, 2, 3)}
        r = len(opts)
        # x_b: exact
        sols_b = solve_slice(arrays, blocks, fam, rhs[DIAG], self.rng, max_cand=self.max_cand, stats=stats,
                             log=self.log)
        states_b = []
        for combo, Ssel in sols_b:
            a_b = (self.SM._block_coordinates(fam, arrays, blocks, combo, Ssel, rhs[DIAG], strict=True)
                   if blocks else np.zeros(0, dtype=complex))
            if blocks and not np.all(np.abs(a_b) > 1e-9):
                continue                                        # duplicates a smaller translate set
            sp = self._refine(blocks, Ssel, a_b, [None] * len(blocks), dC)
            if sp is None:
                stats["split_pruned"] += 1
                continue
            states_b.append((combo, Ssel, sp, a_b))
        stats["b_solutions"] = len(states_b)
        if not states_b:
            return [], stats
        # x_a1: residual (independent of the x_b solution)
        sols_a1 = self._coordinate_scan(arrays, blocks, dvs, rhs[COORD[0]], stats, "a1")
        if not sols_a1:
            return [], stats
        hits = []
        for cb, Sb, sp, a_b in states_b:
            for c1, S1, fifth, a_1, amb1 in sols_a1:
                codes2 = [self.xa2_codes(opts[i], c1[i], cb[i]) for i in range(r)]
                if any(not cd for cd in codes2):
                    continue
                if amb1:
                    # the residual at x_a1 lies in the span of the block's translates: besides the
                    # fifth term being absent there (below), its slice could be a dictionary
                    # state inside that span; the joint reconstruction decides this case
                    hits.extend(self._pair_brute(x0, opts, ords, blocks, dC, arrays, cb, c1, codes2, S1, rhs,
                                                 stats))
                sp1 = self._refine(blocks, S1, a_1, sp, dC)
                if sp1 is None:
                    stats["split_pruned"] += 1
                    continue
                arrays2 = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes2)]
                if fifth is None:
                    # the fifth term is absent at x_a1: at x_a2 it is a point term (or absent, in
                    # which case the four visible terms alone decompose psi_5)
                    sols_a2 = self._coordinate_scan(arrays2, blocks, dvs, rhs[COORD[1]], stats, "a2")
                    for c2i, S2, fifth2, a_2, amb2 in sols_a2:
                        if amb2:
                            # a point term at x_a2 with its slice inside the block's span: a
                            # product term, which Fact 1 excludes; counted, not searched
                            stats["point_ambiguous"] += 1
                        sp2 = self._refine(blocks, S2, a_2, sp1, dC)
                        if sp2 is None:
                            stats["split_pruned"] += 1
                            continue
                        c2 = tuple(int(codes2[i][c2i[i]]) for i in range(r))
                        codes = [{1: c1[i], 2: c2[i], 3: cb[i]} for i in range(r)]
                        fifth_terms = []
                        if fifth2 is not None:
                            stats["a2_point"] += 1
                            fifth_terms = [(self._point_vector(x0, COORD[1], fifth2[0]), fifth2[1][2])]
                        data = self._block_data(blocks, {1: S1, 2: S2, 3: Sb}, {1: a_1, 2: a_2, 3: a_b})
                        hits.extend(self._assemble(x0, opts, ords, blocks, dC, codes, fifth_terms, data, stats))
                    continue
                v, c5 = fifth
                ov = self.fifth_options(v)
                arrays5 = arrays2 + [ov.arrays()]
                n = len(distinct)
                fam5 = Family([(np.append(d1, c5[0]) % P1, np.zeros((n + 1, 0), dtype=np.int64)),
                               (np.append(d2, c5[1]) % P2, np.zeros((n + 1, 0), dtype=np.int64)),
                               (np.append(dC, c5[2]), np.zeros((n + 1, 0), dtype=complex))])
                sols5 = solve_slice(arrays5, blocks, fam5, rhs[COORD[1]], self.rng, max_cand=self.max_cand,
                                    stats=stats, log=self.log)
                stats["a2_solutions"] += len(sols5)
                for combo5, S2 in sols5:
                    a_2 = (self.SM._block_coordinates(fam5, arrays5, blocks, combo5, S2, rhs[COORD[1]], strict=True)
                           if blocks else np.zeros(0, dtype=complex))
                    if blocks and not np.all(np.abs(a_2) > 1e-9):
                        continue
                    sp2 = self._refine(blocks, S2, a_2, sp1, dC)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    c2 = tuple(int(codes2[i][combo5[i]]) for i in range(r))
                    codes = [{1: c1[i], 2: c2[i], 3: cb[i]} for i in range(r)]
                    fifth_terms = [(self._fifth_vector(x0, v, int(combo5[r])), c5[2])]
                    data = self._block_data(blocks, {1: S1, 2: S2, 3: Sb}, {1: a_1, 2: a_2, 3: a_b})
                    hits.extend(self._assemble(x0, opts, ords, blocks, dC, codes, fifth_terms, data, stats))
        hits = SliceMatcher._dedupe(hits)
        stats["hits"] = len(hits)
        return hits, stats


class PlantedBetaMatcher(BetaMatcher):
    """BetaMatcher run against a planted five-qubit target Psi (a sum of
    stabilizer states with Gaussian-integer coefficients) instead of psi_5:
    the base coefficients are solved against Psi's slice at x0 and every
    slice equation against Psi's slice there. confirm() still measures the
    residual against psi_5, so hits are compared by their term codes."""

    def __init__(self, E, Psi, x0, **kw):
        rows = Psi.reshape(1 << N1, -1)
        E2 = copy.copy(E)
        E2.psi = rows[x0]
        E2.psi1, E2.psi2 = to_field(E.F1, rows[x0]), to_field(E.F2, rows[x0])
        super().__init__(E2, **kw)
        self._rows = rows

    def rhs(self, x0, x):
        v = self._rows[x]
        return to_field(self.F1, v), to_field(self.F2, v), v


def codes_key(terms):
    return sorted(exact_codes(t)[0].tobytes() for t in terms)
