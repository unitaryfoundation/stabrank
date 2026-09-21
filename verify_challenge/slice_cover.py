"""Rank exclusion for |H>^m by a base slice with every term visible.

Setting. Slice a decomposition psi^m = sum_i c_i s_i of |H>^m along n_1
qubits: u_i^(x) = (<x| (x) I) s_i for x in F_2^{n_1}, and
sum_i c_i u_i^(x) = alpha_x psi^{n_2} with alpha_x = cos(pi/8)^{n_1 - |x|}
sin(pi/8)^{|x|}, never zero. A term is nonzero exactly on an affine flat
x_0 + V of F_2^{n_1} (V the projection of its direction space), and with
v_1, ..., v_j a basis of V there are Paulis Q_k on the remaining qubits,
l in Z_4^j and a quadratic form q without diagonal such that

    u^(x_0 + t_1 v_1 + ... + t_j v_j) = i^{l.t} (-1)^{q(t)} Q_j^{t_j} ... Q_1^{t_1} u^(x_0)

(the qubit structure lemma of research/constructions/two_qubit_slice.py).
Consequences used here, with the base slice x_0 chosen so that all r terms
are nonzero there:

  (i)  the base slice is an r-cover of psi^{n_2}: r distinct stabilizer
       states whose span contains psi^{n_2}, with coefficients
       d_i = c_i / alpha_{x_0} all nonzero;
  (ii) at the slice x_0 + v_k the term is i^{l_k} Q_k u_i, one of 2^{n_2}
       Pauli classes times a fourth root of unity, or zero when v_k is not
       in V; at x_0 + v_a + v_b it is a sign times the product of the two
       class representatives, and at x_0 + v_1 + v_2 + v_3 the sign is the
       product of the three pair signs.

Enumeration. r-covers of psi^{n_2} with all coefficients nonzero ("full
covers") are listed by a pivot-pair kernel over the dictionary, modulo the
prime 65521 (which is 1 mod 16, so Q(zeta_16) reduces to F_p and every
amplitude of every stabilizer state and of psi is an element of F_p). A
linear dependency over Q(zeta_16) reduces to one mod p, so the modular
kernel lists a superset of the true covers, and every candidate is decided
again mod the prime 2013265921 and in floating point. Symmetry: the local
Clifford stabilizer of |H> ({I, H}) on each remaining qubit and the
permutations of the remaining qubits fix psi^m, preserve slices, and carry
covers to covers, so one cover per orbit is kept (pivot from one orbit
representative, partner minimal in its stabilizer orbit, as in
slice_lift.all_decompositions).

Matching. For each cover and each base point x_0 (one per Hamming weight,
by the permutation symmetry of the sliced qubits), the slices x_0 + e_k
are matched first: sum_i d_i w_i = tan(pi/8)^{|x_0 + e_k| - |x_0|} psi^{n_2}
over the 4 2^{n_2} + 1 options per term (absent included), by meet in the
middle on a random linear functional mod 65521, with every collision
checked in full. Survivors of the three basis slices are combined with
the flat types they imply and the remaining slices are checked with the
sign freedom of (ii). Every hit is confirmed as a decomposition of psi^m
in floating point.
"""

from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_exclusion import dictionary, psi_for, symmetry_orbit_reps  # noqa: E402
from slice_lift import stabilizer_orbit_labels  # noqa: E402

P1 = 65521            # 1 mod 16, hashing and enumeration
P2 = 2013265921       # 1 mod 16, exact re-check of candidates
NUM_TOL = 1e-9


# --------------------------------------------------------------- field -----

def _root16(p):
    for g in range(2, p):
        z = pow(g, (p - 1) // 16, p)
        if pow(z, 8, p) != 1:
            return z
    raise ValueError


class Field:
    """Q(zeta_16) reduced modulo a prime p = 1 mod 16."""

    def __init__(self, p):
        self.p = p
        self.zeta = _root16(p)
        z, zi = self.zeta, pow(self.zeta, p - 2, p)
        self.i = pow(z, 4, p)
        inv2 = pow(2, p - 2, p)
        self.cos = (z + zi) * inv2 % p
        self.sin = (z - zi) * inv2 % p * pow(self.i, p - 2, p) % p
        assert (self.cos ** 2 + self.sin ** 2) % p == 1
        assert (2 * self.cos * self.sin) ** 2 % p == inv2          # sin(pi/4)^2 = 1/2
        self.tan = self.sin * pow(self.cos, p - 2, p) % p
        self.ipow = np.array([pow(self.i, k, p) for k in range(4)], dtype=np.int64)
        self.inv_table = None
        if p < 1 << 20:
            t = np.zeros(p, dtype=np.int64)
            t[1:] = [pow(int(a), p - 2, p) for a in range(1, p)]
            self.inv_table = t

    def inv(self, a):
        a = np.asarray(a, dtype=np.int64) % self.p
        if self.inv_table is not None:
            return self.inv_table[a]
        return np.vectorize(lambda x: pow(int(x), self.p - 2, self.p), otypes=[np.int64])(a)

    def target(self, n):
        """psi^n as a vector over F_p, entry x is cos^(n - |x|) sin^|x|."""
        w = np.array([bin(x).count("1") for x in range(1 << n)])
        return np.array([pow(self.cos, n - int(k), self.p) * pow(self.sin, int(k), self.p) % self.p
                         for k in w], dtype=np.int64)

    def codes_to_field(self, codes):
        """Phase codes (0 zero, 1..4 = 1, i, -1, -i) to F_p."""
        out = self.ipow[(codes - 1) % 4]
        return np.where(codes > 0, out, 0).astype(np.int64)


def patterns(D):
    """Exact phase codes of the dictionary columns (first nonzero entry
    made 1), shape (N, dim), and the unnormalised complex matrix (dim, N)."""
    dim, N = D.shape
    nz = np.abs(D) > 1e-9
    first = np.argmax(nz, axis=0)
    ref = D[first, np.arange(N)]
    W = D / ref[None, :]
    codes = np.zeros((N, dim), dtype=np.int8)
    for c, val in enumerate([1, 1j, -1, -1j], start=1):
        codes[(np.abs(W.T - val) < 1e-6)] = c
    assert np.array_equal(codes > 0, nz.T), "a dictionary entry is not a fourth root of unity"
    C = np.zeros((dim, N), dtype=complex)
    for c, val in enumerate([1, 1j, -1, -1j], start=1):
        C[codes.T == c] = val
    return codes, C


# ----------------------------------------------------- modular kernels -----

def _reduce(F, rows, v):
    """rows - rows[:, c] v / v[c] with c the first nonzero coordinate of v,
    column c removed; returns (residues, c)."""
    c = int(np.flatnonzero(v)[0])
    f = (rows[:, c] * F.inv(v[c])) % F.p
    res = (rows - f[:, None] * v[None, :]) % F.p
    return np.delete(res, c, axis=1), c


def _canon_rows(F, R):
    """Rows scaled so the first nonzero entry is 1; zero rows unchanged."""
    nz = R != 0
    has = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    lead = R[np.arange(len(R)), first]
    scale = np.where(has, F.inv(lead), 0)
    return (R * scale[:, None]) % F.p, has


def _groups_by_key(keys):
    """Indices grouped by equal key, groups of size >= 2 only."""
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    brk = np.flatnonzero(sk[1:] != sk[:-1]) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [len(sk)]))
    return [order[s:e] for s, e in zip(starts, ends) if e - s >= 2]


class CoverEnumerator:
    """Full r-covers of psi^n (r = 3, 4, 5) over the n-qubit dictionary, one
    per orbit of the unitary symmetry group, modulo P1 with exact re-checks.
    """

    def __init__(self, n, D=None, verbose=False, seed=17):
        self.n = n
        self.D = dictionary(2, n) if D is None else D
        self.N = self.D.shape[1]
        self.codes, self.C = patterns(self.D)
        self.F1, self.F2 = Field(P1), Field(P2)
        self.U1 = self.F1.codes_to_field(self.codes)         # (N, dim)
        self.U2 = self.F2.codes_to_field(self.codes)
        self.psi1, self.psi2 = self.F1.target(n), self.F2.target(n)
        self.psi = psi_for("qubit_H", n)
        reps, info = symmetry_orbit_reps("qubit_H", n, self.D, antiunitary=False)
        self.reps, self.info = np.sort(reps), info
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        # quotient by psi (coordinate 0, where psi = cos^n is nonzero)
        self.Q1, _ = _reduce(self.F1, self.U1, self.psi1)      # (N, dim - 1)

    def log(self, s):
        if self.verbose:
            print(s, flush=True)

    # -- exact checks --
    def rank_mod2(self, idx, with_psi):
        M = [self.U2[k] for k in idx]
        if with_psi:
            M.append(self.psi2)
        return _rank_mod(np.array(M, dtype=np.int64), P2)

    def solve(self, idx):
        """Coefficients d with sum d_k u_k = psi (least squares) and the
        residual; also the nullspace dimension of the states."""
        A = self.C[:, list(idx)]
        d, *_ = np.linalg.lstsq(A, self.psi, rcond=None)
        return d, float(np.linalg.norm(A @ d - self.psi)), len(idx) - np.linalg.matrix_rank(A, tol=1e-8)

    def is_cover(self, idx):
        """psi in span of the states, decided mod P2 and numerically."""
        r0 = self.rank_mod2(idx, False)
        r1 = self.rank_mod2(idx, True)
        _, res, _ = self.solve(idx)
        exact = r0 == r1
        if exact != (res < NUM_TOL):
            raise AssertionError(f"modular and numeric cover tests disagree on {idx}")
        return exact

    def is_full(self, idx):
        """Some solution of sum d_k u_k = psi has every d_k nonzero: the
        solution set is d0 + ker, and coordinate k vanishes identically
        iff d0_k = 0 and the kernel has zero k-th row."""
        A = self.C[:, list(idx)]
        d0, *_ = np.linalg.lstsq(A, self.psi, rcond=None)
        if np.linalg.norm(A @ d0 - self.psi) > NUM_TOL:
            return False
        _, s, vh = np.linalg.svd(A)
        rank = int(np.sum(s > 1e-8))
        K = vh[rank:].conj().T                     # (r, r - rank) kernel basis
        dead = (np.abs(d0) < 1e-9) & (np.abs(K).sum(axis=1) < 1e-9 if K.shape[1] else True)
        return not np.any(dead)

    # -- kernels --
    def pivot_plan(self, i):
        """Members (indices in orbits at or above the pivot's) and partners
        (one per stabilizer orbit, least index) for pivot i."""
        roots = self.info["roots"]
        orbit_size = int(np.count_nonzero(roots == i))
        labels, _ = stabilizer_orbit_labels(self.info["perms"], int(i),
                                            stabilizer_order=self.info["order"] // orbit_size)
        cand = np.flatnonzero(roots >= i)
        _, first = np.unique(labels[cand], return_index=True)
        partners = cand[first]
        partners = partners[partners != i]
        return cand, partners

    def covers(self, r, max_run=64, max_found=None, pivots=None):
        """All full r-covers (r in 3, 4, 5) up to symmetry, as a sorted list
        of index tuples, plus the count of modular candidates."""
        F = self.F1
        found, ncand = set(), 0
        t0 = time.time()
        for i in (self.reps if pivots is None else pivots):
            i = int(i)
            if max_found is not None and len(found) >= max_found:
                break
            members, partners = self.pivot_plan(i)
            member_mask = np.zeros(self.N, dtype=bool)
            member_mask[members] = True
            Qi, _ = _reduce(F, self.Q1, self.Q1[i])
            if r == 3:
                # psi in span(u_i, u_l, u_m): images of l, m mod span(psi, u_i) parallel
                keep = member_mask.copy()
                keep[i] = False
                ids = np.flatnonzero(keep)
                Rc, has = _canon_rows(F, Qi[ids])
                ids, Rc = ids[has], Rc[has]
                key = (Rc @ self.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
                for g in _groups_by_key(key):
                    for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                        ncand += 1
                        idx = tuple(sorted((i, a, b)))
                        if self.is_cover(idx) and self.is_full(idx):
                            found.add(idx)
                continue
            for j in partners:
                j = int(j)
                if max_found is not None and len(found) >= max_found:
                    break
                got, nc = self.pair_covers(r, i, j, Qi, member_mask, max_run)
                found.update(got)
                ncand += nc
            self.log(f"  pivot {i}: {len(found)} covers so far, {ncand} candidates [{time.time() - t0:.1f}s]")
        return sorted(found), ncand

    def pair_covers(self, r, i, j, Qi, member_mask, max_run=64):
        """Full r-covers (r = 4, 5) containing the pivot i and the partner j,
        with the other members above j inside member_mask; Qi is the
        dictionary reduced modulo span(psi, u_i). Returns (set, candidates)."""
        F = self.F1
        found, ncand = set(), 0
        if not np.any(Qi[j]):
            return found, 0                                    # u_j in span(psi, u_i)
        R, _ = _reduce(F, Qi, Qi[j])                   # mod span(psi, u_i, u_j)
        keep = member_mask & (np.arange(self.N) > j)
        keep[i] = False
        ids = np.flatnonzero(keep)
        if len(ids) < r - 2:
            return found, ncand
        if r == 4:
            Rc, has = _canon_rows(F, R[ids])
            ids2, Rc = ids[has], Rc[has]
            key = (Rc @ self.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
            for g in _groups_by_key(key):
                for a, b in itertools.combinations(sorted(ids2[g].tolist()), 2):
                    ncand += 1
                    idx = tuple(sorted((i, j, a, b)))
                    if self.is_cover(idx) and self.is_full(idx):
                        found.add(idx)
            return found, ncand
        # r == 5: third pivot k over the members above j, residues of
        # l, m > k mod span(psi, u_i, u_j, u_k) parallel and nonzero
        Rm = R[ids]                                    # (M, 5)
        has = Rm.any(axis=1)
        ids, Rm = ids[has], Rm[has]
        M = len(ids)
        if M < 3:
            return found, ncand
        first = np.argmax(Rm != 0, axis=1)             # pivot column per k
        lead = Rm[np.arange(M), first]
        Rk = (Rm * F.inv(lead)[:, None]) % F.p         # (M, 5), leading entry 1
        coef = Rm[:, first].T                          # coef[k, l] = Rm[l, first[k]]
        res = (Rm[None, :, :] - coef[:, :, None] * Rk[:, None, :]) % F.p    # (M, M, 5)
        nz = res != 0
        hasres = nz.any(axis=2)
        f2 = np.argmax(nz, axis=2)
        lead2 = np.take_along_axis(res, f2[:, :, None], axis=2)[:, :, 0]
        sc = np.where(hasres, F.inv(lead2), 0)
        res = (res * sc[:, :, None]) % F.p
        key = (res @ self.rng.integers(1, F.p, size=5)) % F.p       # (M, M)
        # mask l <= k and zero residues with unique sentinels
        sent = F.p + np.arange(M)
        bad = ~hasres | (np.arange(M)[None, :] <= np.arange(M)[:, None])
        key = np.where(bad, sent[None, :], key)
        order = np.argsort(key, axis=1, kind="stable")
        sk = np.take_along_axis(key, order, axis=1)
        eq = sk[:, 1:] == sk[:, :-1]
        for k, l in zip(*np.nonzero(eq)):
            # run starting at l (only report at the start of a run)
            if l > 0 and eq[k, l - 1]:
                continue
            e = l + 1
            while e < M - 1 and eq[k, e]:
                e += 1
            run = order[k, l:e + 1]
            if len(run) > max_run:
                raise AssertionError(f"parallel class of size {len(run)} at pivot {i}, {j}, {ids[k]}")
            for a, b in itertools.combinations(sorted(ids[run].tolist()), 2):
                ncand += 1
                idx = tuple(sorted((i, j, int(ids[k]), a, b)))
                if self.is_cover(idx) and self.is_full(idx):
                    found.add(idx)
        return found, ncand

    def in_span(self, idx):
        """Dictionary states in the span of the states `idx` (mod P1, then
        exact), excluding `idx` themselves."""
        F = self.F1
        rows = self.U1.copy()
        basis = self.U1[list(idx)].copy()
        for b in range(len(idx)):
            v = basis[b]
            if not np.any(v):
                raise AssertionError("dependent basis in in_span")
            rows, c = _reduce(F, rows, v)
            basis = np.delete((basis - (basis[:, c] * F.inv(v[c]))[:, None] * v[None, :]) % F.p, c, axis=1)
        cand = np.flatnonzero(~rows.any(axis=1))
        out = []
        for x in cand:
            if int(x) in idx:
                continue
            if self.rank_mod2(tuple(idx) + (int(x),), False) == len(idx):
                out.append(int(x))
        return out

    def dependent_covers(self, r, covers3, covers4):
        """Full r-covers whose states are linearly dependent, from the full
        3-covers (rank-3 decompositions) and full 4-covers: a 3-cover with
        r - 3 states of its span or with a pair of states parallel modulo it,
        a 4-cover with r - 4 states of its span (see the module note)."""
        F = self.F1
        out = set()
        for T in covers3:
            span = self.in_span(T)
            for extra in itertools.combinations(span, r - 3):
                idx = tuple(sorted(T + extra))
                if self.is_full(idx):
                    out.add(idx)
            if r == 5:
                # pairs x, y outside span(T) with y in span(T, x)
                rows = self.U1.copy()
                for b in T:
                    rows, _ = _reduce(F, rows, rows[b])
                Rc, has = _canon_rows(F, rows)
                ids = np.flatnonzero(has)
                key = (Rc[ids] @ self.rng.integers(1, F.p, size=Rc.shape[1])) % F.p
                for g in _groups_by_key(key):
                    for a, b in itertools.combinations(sorted(ids[g].tolist()), 2):
                        idx = tuple(sorted(T + (a, b)))
                        if self.is_cover(idx) and self.is_full(idx):
                            out.add(idx)
        if r == 5:
            for Cv in covers4:
                for x in self.in_span(Cv):
                    idx = tuple(sorted(Cv + (x,)))
                    if self.is_full(idx):
                        out.add(idx)
        return sorted(out)


def _rank_mod(M, p):
    """Rank of an integer matrix over F_p (Python ints, exact)."""
    A = [[int(x) % p for x in row] for row in M]
    rows, cols = len(A), len(A[0]) if A else 0
    rank = 0
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if A[r][c]:
                piv = r
                break
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        inv = pow(A[rank][c], p - 2, p)
        A[rank] = [(x * inv) % p for x in A[rank]]
        for r in range(rows):
            if r != rank and A[r][c]:
                f = A[r][c]
                A[r] = [(x - f * y) % p for x, y in zip(A[r], A[rank])]
        rank += 1
    return rank


# ---------------------------------------------------------- Pauli codes ----

def pauli_reps(u, n):
    """One representative (a, c) per Pauli class X^a Z^c modulo the
    stabilizer of u (2^n classes), the image vectors Q u (complex,
    unnormalised) and the exact phase codes of each image."""
    dim = 1 << n
    x = np.arange(dim)
    par = np.array([[bin(c & t).count("1") % 2 for c in range(dim)] for t in x])
    PH = (-1.0) ** par                                       # PH[t, c] = (-1)^(c.t)
    seen, reps, imgs = {}, [], []
    for a in range(dim):
        for c in range(dim):
            qu = np.zeros(dim, dtype=complex)
            qu[x ^ a] = PH[:, c] * u
            nz = np.flatnonzero(np.abs(qu) > 1e-9)
            w = qu / qu[nz[0]]
            key = (np.round(w, 6) + 0.0).tobytes()
            if key not in seen:
                seen[key] = len(reps)
                reps.append((a, c))
                imgs.append(qu)
    if len(reps) != dim:
        raise AssertionError(f"{len(reps)} Pauli classes, expected {dim}")
    return reps, imgs


def apply_pauli(u, a, c, n):
    dim = 1 << n
    x = np.arange(dim)
    out = np.zeros(dim, dtype=complex)
    sign = (-1.0) ** np.array([bin(c & t).count("1") % 2 for t in x])
    out[x ^ a] = sign * u
    return out


def exact_codes(v):
    """Phase codes of a complex vector with entries in {0, +-1, +-i} times a
    common scalar (first nonzero entry made 1)."""
    nz = np.flatnonzero(np.abs(v) > 1e-9)
    w = v / v[nz[0]]
    codes = np.zeros(len(v), dtype=np.int8)
    for c, val in enumerate([1, 1j, -1, -1j], start=1):
        codes[np.abs(w - val) < 1e-6] = c
    if np.count_nonzero(codes) != len(nz):
        raise AssertionError("vector is not a phase pattern")
    return codes, v[nz[0]]


FOURTH = np.array([1, 1j, -1, -1j])


class TermOptions:
    """The slice options of one term with base slice u: 4 2^n vectors
    i^l Q u (class k, phase l -> option 4 k + l) plus 'absent' (option
    4 2^n), over F_P1, F_P2 and C."""

    def __init__(self, u, n, F1, F2):
        self.n = n
        self.reps, imgs = pauli_reps(u, n)
        self.u = u
        vecs = [ph * im for im in imgs for ph in FOURTH]
        self.vecs = np.array(vecs + [np.zeros(1 << n, dtype=complex)])    # (4 2^n + 1, dim)
        self.absent = len(vecs)
        codes = [exact_codes(v)[0] for v in vecs]
        self.m1 = np.array([F1.codes_to_field(c) for c in codes] + [np.zeros(1 << n, dtype=np.int64)])
        self.m2 = np.array([F2.codes_to_field(c) for c in codes] + [np.zeros(1 << n, dtype=np.int64)])
        # the unit pattern: exact_codes normalises the first nonzero entry to 1,
        # so fix the global scalar of every option relative to the complex vector
        scal = np.array([exact_codes(v)[1] for v in vecs])
        # every scalar is a fourth root of unity times a common modulus
        mod = np.abs(scal)
        assert np.allclose(mod, mod[0])
        ph = np.round(np.angle(scal / mod) / (np.pi / 2)).astype(int) % 4
        assert np.allclose(scal / mod, FOURTH[ph])
        self.m1 = (self.m1 * F1.ipow[np.append(ph, 0)][:, None]) % F1.p
        self.m2 = (self.m2 * F2.ipow[np.append(ph, 0)][:, None]) % F2.p
        self.mod = float(mod[0])          # |u| entries: the common modulus (1 for pattern-normalised u)

    def product(self, opt_a, opt_b):
        """The two candidate vectors at x_0 + v_a + v_b given the options at
        x_0 + v_a and x_0 + v_b: +- i^(l_a + l_b) Q_b Q_a u, as option-free
        complex vectors (pair of arrays)."""
        ka, la = divmod(opt_a, 4)
        kb, lb = divmod(opt_b, 4)
        (aa, ca), (ab, cb) = self.reps[ka], self.reps[kb]
        v = apply_pauli(apply_pauli(self.u, aa, ca, self.n), ab, cb, self.n) * FOURTH[(la + lb) % 4]
        return v, -v


def match_slice(opts, d1, rhs1, F1, opts2, d2, rhs2, F2, optsC, dC, rhsC, rng, ntop=2):
    """Solutions (option tuples) of sum_i d_i vec_i(opt_i) = rhs, by meet in
    the middle on a random functional mod P1 with full checks mod P1, mod
    P2 and in floating point. opts: list of (n_opt, dim) int64 arrays."""
    r = len(opts)
    dim = opts[0].shape[1]
    f = rng.integers(1, F1.p, size=dim)
    hs = [((d1[i] * o) % F1.p @ f) % F1.p for i, o in enumerate(opts)]      # per-term hashed options
    target = int(rhs1 @ f % F1.p)
    left, right = list(range(ntop)), list(range(ntop, r))
    L = np.zeros(1, dtype=np.int64)
    for i in left:
        L = (L[:, None] + hs[i][None, :]).ravel() % F1.p
    R = np.zeros(1, dtype=np.int64)
    for i in right:
        R = (R[:, None] + hs[i][None, :]).ravel() % F1.p
    need = (target - L) % F1.p
    order = np.argsort(R, kind="stable")
    Rs = R[order]
    lo = np.searchsorted(Rs, need, side="left")
    hi = np.searchsorted(Rs, need, side="right")
    sizes = [len(o) for o in opts]
    sols = []
    for a in np.flatnonzero(hi > lo):
        for pos in range(lo[a], hi[a]):
            b = order[pos]
            combo = _decode(int(a), [sizes[i] for i in left]) + _decode(int(b), [sizes[i] for i in right])
            v1 = sum(d1[i] * opts[i][combo[i]] for i in range(r)) % F1.p
            if not np.array_equal(v1, rhs1 % F1.p):
                continue
            v2 = sum(d2[i] * opts2[i][combo[i]] for i in range(r)) % F2.p
            if not np.array_equal(v2, rhs2 % F2.p):
                continue
            vC = sum(dC[i] * optsC[i][combo[i]] for i in range(r))
            if np.linalg.norm(vC - rhsC) > 1e-7 * max(1.0, np.linalg.norm(rhsC)):
                continue
            sols.append(tuple(combo))
    return sols


def _decode(idx, sizes):
    out = []
    for s in reversed(sizes):
        out.append(idx % s)
        idx //= s
    return out[::-1]


# --------------------------------------------------------------- flats -----

def flat_types(n1):
    """Subspaces of F_2^{n1} of dimension >= n1 - 1 (the flat directions a
    term can have along the sliced qubits when it has property P), each as
    the set of its nonzero points."""
    pts = range(1, 1 << n1)
    out = [frozenset(pts)]
    if n1 >= 2:
        for z in pts:                         # hyperplane z . x = 0
            out.append(frozenset(x for x in pts if bin(x & z).count("1") % 2 == 0))
    return out


def presence_from_basis(absent_basis, n1):
    """Flat types (as point sets) consistent with the given set of absent
    basis directions e_k (bit k)."""
    out = []
    for W in flat_types(n1):
        ab = {k for k in range(n1) if (1 << k) not in W}
        if ab == set(absent_basis):
            out.append(W)
    return out


# ------------------------------------------------------------- matching ----

class SliceMatcher:
    """Match the seven non-base slices for one full cover at one base point."""

    def __init__(self, enum, n1, verbose=False):
        self.E = enum
        self.n1, self.n2 = n1, enum.n
        self.F1, self.F2 = enum.F1, enum.F2
        self.verbose = verbose
        self.rng = np.random.default_rng(29)
        self.cache = {}

    def options(self, idx):
        if idx not in self.cache:
            self.cache[idx] = TermOptions(self.E.C[:, idx], self.n2, self.F1, self.F2)
        return self.cache[idx]

    def coeffs(self, cover):
        """d mod P1, mod P2 and complex, with sum d_k u_k = psi; requires the
        cover independent and the system nonsingular mod both primes."""
        A1 = self.E.U1[list(cover)].T % P1                      # (dim, r)
        A2 = self.E.U2[list(cover)].T % P2
        d1 = _solve_mod(A1, self.E.psi1, P1)
        d2 = _solve_mod(A2, self.E.psi2, P2)
        dC, res, nul = self.E.solve(cover)
        if d1 is None or d2 is None or nul or res > NUM_TOL:
            return None
        return d1, d2, dC

    def run(self, cover, x0, rank_target=None):
        """Every decomposition of psi^{n1 + n2} with base slice `cover` at
        x0 (all terms present), as lists of term vectors; also the stage
        statistics."""
        n1 = self.n1
        r = len(cover)
        co = self.coeffs(cover)
        if co is None:
            raise ValueError("dependent cover")
        d1, d2, dC = co
        opts = [self.options(k) for k in cover]
        w0 = bin(x0).count("1")
        stats = {"basis_solutions": [], "combos": 0, "hits": 0}

        def rhs(x):
            e = bin(x).count("1") - w0
            r1 = (self.E.psi1 * pow(self.F1.tan, e % (P1 - 1), P1)) % P1
            r2 = (self.E.psi2 * pow(self.F2.tan, e % (P2 - 1), P2)) % P2
            rC = self.E.psi * (np.tan(np.pi / 8) ** e)
            return r1, r2, rC

        basis_sols = []
        for k in range(n1):
            r1, r2, rC = rhs(x0 ^ (1 << k))
            sols = match_slice([o.m1 for o in opts], d1, r1, self.F1,
                               [o.m2 for o in opts], d2, r2, self.F2,
                               [o.vecs for o in opts], dC, rC, self.rng)
            basis_sols.append(sols)
            stats["basis_solutions"].append(len(sols))
            if not sols:
                return [], stats
        hits = []
        composite = [x for x in range(1, 1 << n1) if bin(x).count("1") >= 2]
        for combo in itertools.product(*basis_sols):
            # combo[k][i] is the option of term i at slice x0 + e_k
            types = []
            ok = True
            for i in range(r):
                absent = [k for k in range(n1) if combo[k][i] == opts[i].absent]
                Ws = presence_from_basis(absent, n1)
                if not Ws:
                    ok = False
                    break
                types.append(Ws)
            if not ok:
                continue
            for Wsel in itertools.product(*types):
                stats["combos"] += 1
                hits.extend(self._complete(cover, x0, opts, combo, Wsel, dC, composite, rhs))
        stats["hits"] = len(hits)
        return hits, stats

    def _complete(self, cover, x0, opts, combo, Wsel, dC, composite, rhs):
        """Given basis options and flat types, enumerate the composite-slice
        vectors of every term (sign choices, free codes on planes not spanned
        by basis directions) and keep the assignments solving every slice."""
        n1, r = self.n1, len(cover)
        # per term: list of candidate dicts {slice point: complex vector}
        per_term = []
        for i in range(r):
            W = Wsel[i]
            o = opts[i]
            basis_in = [k for k in range(n1) if (1 << k) in W]
            fixed = {1 << k: o.vecs[combo[k][i]] for k in basis_in}
            # points of W not spanned as a pair sum of basis directions in W
            cands = [dict(fixed)]
            pts = sorted(W)
            # choose a basis of W: basis directions in W first, then further points
            Wbasis = [1 << k for k in basis_in]
            span = {0}
            for b in Wbasis:
                span |= {s ^ b for s in span}
            free_pts = []
            for pnt in pts:
                if pnt not in span:
                    Wbasis.append(pnt)
                    free_pts.append(pnt)
                    span |= {s ^ pnt for s in span}
            # free points get any of the 4 2^n options (not absent)
            for pnt in free_pts:
                cands = [{**c, pnt: o.vecs[t]} for c in cands for t in range(o.absent)]
            # remaining points of W: sums of two or three basis points, with signs
            out = []
            for c in cands:
                out.extend(self._fill_products(c, Wbasis, o, W))
            per_term.append(out)
        # now match composite slices: for each composite point, sum_i d_i vec = rhs
        hits = []
        for choice in itertools.product(*per_term):
            good = True
            for x in composite:
                _, _, rC = rhs(x0 ^ x)
                s = sum(dC[i] * choice[i].get(x, 0) for i in range(r))
                if np.linalg.norm(s - rC) > 1e-7:
                    good = False
                    break
            if good:
                terms = []
                for i in range(r):
                    t = np.zeros((1 << n1, len(opts[i].u)), dtype=complex)
                    t[x0] = opts[i].u
                    for x, v in choice[i].items():
                        t[x0 ^ x] = v
                    terms.append(t.ravel())
                hits.append(terms)
        return hits

    def _fill_products(self, c, Wbasis, o, W):
        """Extend the assignment c (basis points of W -> vector) to all of W
        with the sign freedom of pair products; the triple sum, when W has
        three basis points, takes the product of the three pair signs."""
        # Represent vectors by (Pauli rep, phase) to compute products: recover
        # the option index for basis points by matching against o.vecs.
        def opt_of(v):
            for t in range(o.absent):
                if np.allclose(o.vecs[t], v):
                    return t
            raise AssertionError("basis vector is not an option")
        opt = {pnt: opt_of(c[pnt]) for pnt in Wbasis}
        pairs = list(itertools.combinations(range(len(Wbasis)), 2))
        outs = []
        for signs in itertools.product((0, 1), repeat=len(pairs)):
            assign = dict(c)
            for (a, b), s in zip(pairs, signs):
                pv = o.product(opt[Wbasis[a]], opt[Wbasis[b]])[s]
                assign[Wbasis[a] ^ Wbasis[b]] = pv
            if len(Wbasis) == 3:
                # triple: sign = product of the pair signs, vector from the class product
                tot = sum(signs) % 2
                ka, la = divmod(opt[Wbasis[0]], 4)
                kb, lb = divmod(opt[Wbasis[1]], 4)
                kc, lc = divmod(opt[Wbasis[2]], 4)
                v = o.u
                for k in (ka, kb, kc):
                    a_, c_ = o.reps[k]
                    v = apply_pauli(v, a_, c_, o.n)
                v = v * FOURTH[(la + lb + lc) % 4] * (-1) ** tot
                assign[Wbasis[0] ^ Wbasis[1] ^ Wbasis[2]] = v
            assert set(assign) == set(W), (set(assign), set(W))
            outs.append(assign)
        return outs


def _solve_mod(A, b, p):
    """Unique solution of A x = b over F_p (A of full column rank, consistent),
    or None when the columns are dependent mod p or the system inconsistent."""
    A = [[int(v) % p for v in row] for row in A]
    b = [int(v) % p for v in b]
    rows, cols = len(A), len(A[0])
    M = [A[r] + [b[r]] for r in range(rows)]
    rank = 0
    pivcols = []
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if M[r][c]:
                piv = r
                break
        if piv is None:
            return None
        M[rank], M[piv] = M[piv], M[rank]
        inv = pow(M[rank][c], p - 2, p)
        M[rank] = [(x * inv) % p for x in M[rank]]
        for r in range(rows):
            if r != rank and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % p for x, y in zip(M[r], M[rank])]
        pivcols.append(c)
        rank += 1
    for r in range(rank, rows):
        if M[r][cols]:
            return None
    x = np.zeros(cols, dtype=np.int64)
    for r, c in enumerate(pivcols):
        x[c] = M[r][cols]
    return x


def x0_reps(n1):
    """One base point per Hamming weight."""
    return [(1 << w) - 1 for w in range(n1 + 1)]


def confirm_decomposition(terms, m):
    """Residual of psi^m against the span of the terms and the coefficients."""
    A = np.column_stack(terms)
    psi = psi_for("qubit_H", m)
    c, *_ = np.linalg.lstsq(A, psi, rcond=None)
    return float(np.linalg.norm(A @ c - psi)), c
