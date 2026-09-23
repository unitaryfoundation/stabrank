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
by the permutation symmetry of the sliced qubits) the multiset cover is
split into its distinct states, whose coefficients form an affine family
d = d_0 + K lambda (a point when the states are independent), and into
blocks of repeated copies of one state. The slices x_0 + e_k are solved one
at a time against the current family: sum_i d_i w_i = tan(pi/8)^{|x_0 +
e_k| - |x_0|} psi^{n_2} over the 4 2^{n_2} + 1 options per term (absent
included), by meet in the middle on a random functional mod 65521 when the
family is a point, and, when parameters remain, by the condition that
(u, v_1, ..., v_kappa) = (sum d_0i w_i - rhs, sum K_i1 w_i, ...) be
dependent, tested as the vanishing mod 65521 of the determinant of kappa + 1
random functionals applied to them, Laplace-expanded into features of two
halves of the terms and joined by a dense product. A block of g copies
contributes any vector of the span of at most g Pauli translates of its
base state (the 2^{n_2} translates are a basis), so the equation is
projected onto the annihilator of the chosen translates; the copies'
coefficients, classes and phases are reconstructed at the end from the
residuals, and for a pair the admissible coefficient splits are tracked
from slice to slice. Every hash collision is decided mod 65521 on the whole
equation, then over C and mod 2013265921, and each solution restricts the
family. The absence pattern on the coordinate slices fixes the flat of every
ordinary term (all 2^{n_1}-point subspaces are allowed; property P is not
assumed by the matcher), the composite slices are solved point by point
over the option codes of the structure lemma (class products with one free
sign per basis pair, the product of the pair signs on a triple, free codes
on basis points that are not coordinate directions), and every hit is
confirmed as a decomposition of psi^m in floating point and mod
2013265921.
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


def _native_cover5():
    """The compiled 5-cover pair kernel from stabrank_core, or None."""
    if os.environ.get("STABRANK_NO_NATIVE"):
        return None
    try:
        from stabrank.stabrank_core import cover5_pair
    except ImportError:
        return None
    return cover5_pair


class CoverEnumerator:
    """Full r-covers of psi^n (r = 3, 4, 5) over the n-qubit dictionary, one
    per orbit of the unitary symmetry group, modulo P1 with exact re-checks.
    """

    def __init__(self, n, D=None, verbose=False, seed=17, native=True):
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
        self.rng_seed = seed
        self.verbose = verbose
        # quotient by psi (coordinate 0, where psi = cos^n is nonzero)
        self.Q1, _ = _reduce(self.F1, self.U1, self.psi1)      # (N, dim - 1)
        self.native_cover5 = _native_cover5() if native else None

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
        dictionary reduced modulo span(psi, u_i). Returns (set, candidates).
        For r = 5 the compiled kernel (cpp/src/cover5.cpp) runs the same
        search when stabrank_core provides it and STABRANK_NO_NATIVE is not
        set; this method's own code is the reference."""
        F = self.F1
        found, ncand = set(), 0
        if r == 5 and self.native_cover5 is not None:
            return self._pair_covers_native(i, j, member_mask, max_run)
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

    def _pair_covers_native(self, i, j, member_mask, max_run):
        """pair_covers(5, ...) through cover5_pair: the kernel decides the span
        condition mod P2 and the fullness mod both primes; a cover full modulo
        exactly one prime (a modular accident) is decided here numerically."""
        idx, flags, ncand, _ = self.native_cover5(
            self.Q1, self.U1, self.psi1, self.U2, self.psi2, int(i), int(j),
            np.ascontiguousarray(member_mask, dtype=np.uint8), int(max_run), int(self.rng_seed))
        found = set()
        for row, (f1, f2) in zip(np.asarray(idx), np.asarray(flags)):
            t = tuple(int(x) for x in row)
            if f1 and f2:
                found.add(t)
            elif f1 or f2:
                if self.is_full(t):
                    found.add(t)
        return found, int(ncand)

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


class UnpinnedFamily(Exception):
    """A state with a block reached the final reconstruction in a form the
    reconstruction cannot decide: parameters left in its coefficient family
    (the residual coordinates would be taken at an arbitrary member, one per
    slice), or the chosen translates of the blocks dependent (the split of
    the residual between the blocks is a family), or a residual that fails
    the complex solve although the exact restriction accepted it. The run
    raises instead of dropping the state, and a batch records the cover as
    undecided. Carrying the parameters into the reconstruction as unknowns
    shared by the blocks is the treatment this exception stands in for."""


class TermOptions:
    """The slice options of one term with base slice u: the 4 2^n vectors
    i^l Q u (Pauli class k, phase l, option code 4 k + l) plus 'absent'
    (code 4 2^n), over F_P1, F_P2 and C, and the class products Q_b Q_a u."""

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
        # exact_codes normalises the first nonzero entry to 1, so fix the
        # global scalar of every option relative to the complex vector
        scal = np.array([exact_codes(v)[1] for v in vecs])
        mod = np.abs(scal)
        assert np.allclose(mod, mod[0])
        ph = np.round(np.angle(scal / mod) / (np.pi / 2)).astype(int) % 4
        assert np.allclose(scal / mod, FOURTH[ph])
        self.m1 = (self.m1 * F1.ipow[np.append(ph, 0)][:, None]) % F1.p
        self.m2 = (self.m2 * F2.ipow[np.append(ph, 0)][:, None]) % F2.p
        self.mod = float(mod[0])
        # class products: Q_kb Q_ka u = i^l imgs[kc], stored as (kc, l)
        K = 1 << n
        self.prod = np.zeros((K, K, 2), dtype=np.int64)
        for ka in range(K):
            for kb in range(K):
                (aa, ca), (ab, cb) = self.reps[ka], self.reps[kb]
                v = apply_pauli(apply_pauli(u, aa, ca, n), ab, cb, n)
                self.prod[ka, kb] = self.code_of(v)

    def code_of(self, v):
        """(class, phase) with v = i^phase imgs[class]."""
        for k in range(1 << self.n):
            w = self.vecs[4 * k]
            nz = np.flatnonzero(np.abs(w) > 1e-9)
            ratio = v[nz[0]] / w[nz[0]]
            for l in range(4):
                if abs(ratio - FOURTH[l]) < 1e-6 and np.allclose(v, FOURTH[l] * w, atol=1e-9):
                    return k, l
        raise AssertionError("vector is not a phased Pauli translate of the base slice")

    def compose(self, code_a, code_b, sign=0):
        """Code of (-1)^sign i^(l_a + l_b) Q_b Q_a u from the codes of
        i^l_a Q_a u and i^l_b Q_b u."""
        ka, la = divmod(int(code_a), 4)
        kb, lb = divmod(int(code_b), 4)
        kc, l = self.prod[ka, kb]
        return 4 * int(kc) + (la + lb + int(l) + 2 * sign) % 4

    def arrays(self):
        return self.m1, self.m2, self.vecs


# ------------------------------------------------------ affine families ----

def _affine_solve_mod(A, b, p):
    """All solutions of A x = b over F_p as (x0, N) with N a basis of the
    kernel (columns), or None when the system is inconsistent."""
    A = np.asarray(A, dtype=np.int64) % p
    b = np.asarray(b, dtype=np.int64) % p
    rows, cols = A.shape
    M = [[int(v) for v in A[r]] + [int(b[r])] for r in range(rows)]
    rank, pivcols = 0, []
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if M[r][c]:
                piv = r
                break
        if piv is None:
            continue
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
    x0 = np.zeros(cols, dtype=np.int64)
    for r, c in enumerate(pivcols):
        x0[c] = M[r][cols]
    free = [c for c in range(cols) if c not in pivcols]
    N = np.zeros((cols, len(free)), dtype=np.int64)
    for j, fc in enumerate(free):
        N[fc, j] = 1
        for r, c in enumerate(pivcols):
            N[c, j] = (-M[r][fc]) % p
    return x0, N


def _affine_solve_C(A, b, tol=1e-7):
    """All solutions of A x = b over C as (x0, N), or None."""
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if A.shape[1] == 0:
        return (np.zeros(0, dtype=complex), np.zeros((0, 0), dtype=complex)) if np.linalg.norm(b) < tol else None
    x0, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.linalg.norm(A @ x0 - b) > tol * max(1.0, np.linalg.norm(b)):
        return None
    _, s, vh = np.linalg.svd(A)
    rank = int(np.sum(s > 1e-8 * max(1.0, s[0] if len(s) else 1.0)))
    return x0, vh[rank:].conj().T


def _mm(A, B, p):
    """A @ B mod p without overflow (object arithmetic for large p)."""
    A, B = np.asarray(A), np.asarray(B)
    if A.shape[-1] == 0:
        return np.zeros(A.shape[:-1] + B.shape[1:], dtype=np.int64)
    if p < (1 << 26):
        return (np.asarray(A, dtype=np.int64) @ np.asarray(B, dtype=np.int64)) % p
    R = np.asarray(A).astype(object) @ np.asarray(B).astype(object)
    return np.array([[int(x) % p for x in row] for row in np.atleast_2d(R)], dtype=np.int64).reshape(R.shape)


class Family:
    """The affine family d = d0 + K lambda of base coefficients, over F_P1,
    F_P2 and C. kappa is the exact dimension (from C, checked against P2);
    kappa1 is the dimension mod P1, which can exceed kappa by a modular rank
    accident, in which case the P1 family is a superset and only weakens
    the hashing filters."""

    def __init__(self, parts):
        self.parts = parts
        self.kappa = parts[2][1].shape[1]
        if parts[1][1].shape[1] != self.kappa:
            raise AssertionError("coefficient family dimension differs mod P2 and over C")
        self.kappa1 = parts[0][1].shape[1]
        if self.kappa1 < self.kappa:
            raise AssertionError("coefficient family smaller mod P1 than over C")

    @classmethod
    def from_cover(cls, E, cover):
        """The coefficients of sum d_i u_i = psi over the (multi)set cover, or
        None when psi is not in the span."""
        idx = list(cover)
        s1 = _affine_solve_mod(E.U1[idx].T, E.psi1, P1)
        s2 = _affine_solve_mod(E.U2[idx].T, E.psi2, P2)
        sC = _affine_solve_C(E.C[:, idx], E.psi)
        if s1 is None or s2 is None or sC is None:
            return None
        return cls([s1, s2, sC])

    def restrict(self, Ws, rhss):
        """The subfamily with W d = rhs (W: columns are the terms' slice
        vectors), or None when empty. Decided mod P1 first (cheap), then
        over C, then mod P2."""
        new = [None, None, None]
        for fld, p in ((0, P1), (2, None), (1, P2)):
            (d0, K), W, rhs = self.parts[fld], Ws[fld], rhss[fld]
            if p is None:
                sol = _affine_solve_C(W @ K, rhs - W @ d0)
            else:
                sol = _affine_solve_mod(_mm(W, K, p), (rhs - _mm(W, d0[:, None], p)[:, 0]) % p, p)
            if sol is None:
                return None
            l0, N = sol
            if p is None:
                new[fld] = (d0 + K @ l0, K @ N)
            else:
                new[fld] = ((d0 + _mm(K, l0[:, None], p)[:, 0]) % p, _mm(K, N, p))
        return Family(new)

    def coefficients(self, rng):
        """A member of the complex family (a generic one when parameters remain)."""
        d0, K = self.parts[2]
        if K.shape[1] == 0:
            return d0
        lam = rng.normal(size=K.shape[1]) + 1j * rng.normal(size=K.shape[1])
        return d0 + K @ lam

    def has_zero_coefficient(self, exempt=()):
        """True when some coefficient outside `exempt` vanishes on the whole
        family (a term no member of the family uses). Block positions are
        exempt: the merged coefficient of repeated copies may cancel."""
        d0, K = self.parts[2]
        dead = np.abs(d0) < 1e-9
        if K.shape[1]:
            dead &= np.abs(K).sum(axis=1) < 1e-9
        if len(exempt):
            dead[list(exempt)] = False
        return bool(np.any(dead))


# -------------------------------------------------------- slice solving ----

def _annihilator_mod(V, p):
    """Rows P with P V = 0 over F_p (a basis of the left kernel)."""
    dim = V.shape[0]
    if V.shape[1] == 0:
        return np.eye(dim, dtype=np.int64)
    _, N = _affine_solve_mod(V.T % p, np.zeros(V.shape[1], dtype=np.int64), p)
    return np.ascontiguousarray(N.T)


def _annihilator_C(V):
    dim = V.shape[0]
    if V.shape[1] == 0:
        return np.eye(dim, dtype=complex)
    _, N = _affine_solve_C(V.T, np.zeros(V.shape[1], dtype=complex))
    return np.ascontiguousarray(N.T)


def _independent_columns_mod(K, p):
    """A maximal set of independent columns of K over F_p."""
    keep = []
    for c in range(K.shape[1]):
        if _rank_mod(K[:, keep + [c]].T, p) > len(keep):
            keep.append(c)
    return K[:, keep]


class Block:
    """g copies of one base state u (a repeated state of the multiset
    cover). At every slice the copies together contribute an arbitrary
    vector of the span of at most g Pauli translates Q_k u, so the block
    enters a slice equation only through the choice of the translate set S
    (|S| <= g), and the equation is projected onto the annihilator of S.
    The copies' coefficients, classes and phases are reconstructed at the
    end from the residuals (`reconstruct`)."""

    def __init__(self, o, g, pos):
        self.o, self.g, self.pos = o, g, pos
        K = 1 << o.n
        self.subsets = [S for j in range(g + 1) for S in itertools.combinations(range(K), j)]
        cols = [4 * k for k in range(K)]
        self.V1 = np.ascontiguousarray(o.m1[cols].T)        # (dim, K): translate k in column k
        self.V2 = np.ascontiguousarray(o.m2[cols].T)
        self.VC = np.ascontiguousarray(o.vecs[cols].T)


def _projectors(blocks, Ssel):
    """Annihilator rows (P1, P2, PC) of the chosen translates of all blocks,
    and the translate matrices (V1, V2, VC) for the coordinate solve."""
    if not blocks:
        return None, None
    V1 = np.column_stack([b.V1[:, list(S)] for b, S in zip(blocks, Ssel)]) if any(Ssel) else np.zeros((blocks[0].V1.shape[0], 0), dtype=np.int64)
    V2 = np.column_stack([b.V2[:, list(S)] for b, S in zip(blocks, Ssel)]) if any(Ssel) else np.zeros((blocks[0].V2.shape[0], 0), dtype=np.int64)
    VC = np.column_stack([b.VC[:, list(S)] for b, S in zip(blocks, Ssel)]) if any(Ssel) else np.zeros((blocks[0].VC.shape[0], 0), dtype=complex)
    return (_annihilator_mod(V1, P1), _annihilator_mod(V2, P2), _annihilator_C(VC)), (V1, V2, VC)


def _split_sides(sizes):
    """Terms split into two sides of balanced product size."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    sides, prods = ([], []), [1, 1]
    for i in order:
        s = 0 if prods[0] <= prods[1] else 1
        sides[s].append(i)
        prods[s] *= sizes[i]
    return sides


def _combos(sizes):
    """All index tuples over the given sizes, as a (prod, len) array."""
    if not sizes:
        return np.zeros((1, 0), dtype=np.int64)
    grids = np.indices(sizes).reshape(len(sizes), -1).T
    return np.ascontiguousarray(grids, dtype=np.int64)


def _det_mod(M, p):
    """Determinants mod p of a stack (N, k, k) of small matrices."""
    k = M.shape[1]
    if k == 0:
        return np.ones(M.shape[0], dtype=np.int64)
    if k == 1:
        return M[:, 0, 0] % p
    out = np.zeros(M.shape[0], dtype=np.int64)
    for j in range(k):
        minor = np.delete(M[:, 1:, :], j, axis=2)
        term = (M[:, 0, j] * _det_mod(minor, p)) % p
        out = (out + (term if j % 2 == 0 else -term)) % p
    return out


def _assemble(sides, idxL, idxR, a, b, r):
    combo = [0] * r
    for i, o in zip(sides[0], idxL[a]):
        combo[i] = int(o)
    for i, o in zip(sides[1], idxR[b]):
        combo[i] = int(o)
    return tuple(combo)


def _mitm(popts, dv, prhs, sides, rng, p=P1):
    """Combinations with sum_i dv_i w_i = rhs (mod p) by meet in the middle
    on a random functional; a superset of the exact solutions. popts: per
    term the (projected) option vectors mod p."""
    dim = prhs.shape[0]
    f = rng.integers(1, p, size=dim)
    hs = [((dv[i] * o) % p @ f) % p for i, o in enumerate(popts)]
    target = int(prhs @ f % p)
    sizes = [len(h) for h in hs]

    def side(bl):
        S = np.zeros(1, dtype=np.int64)
        for i in bl:
            S = (S[:, None] + hs[i][None, :]).ravel() % p
        return S, _combos([sizes[i] for i in bl])

    L, idxL = side(sides[0])
    R, idxR = side(sides[1])
    need = (target - L) % p
    order = np.argsort(R, kind="stable")
    Rs = R[order]
    lo = np.searchsorted(Rs, need, side="left")
    hi = np.searchsorted(Rs, need, side="right")
    out = []
    for a in np.flatnonzero(hi > lo):
        for pos in range(lo[a], hi[a]):
            out.append(_assemble(sides, idxL, idxR, a, order[pos], len(popts)))
    return out


def _dense(popts, d0v, Kv, prhs, sides, rng, max_cand, p=P1):
    """Combinations for which u = sum d0v_i w_i - rhs lies in the span of
    v_j = sum Kv_ij w_i (necessary condition: the (k + 1) x (k + 1) matrix of
    k + 1 random functionals applied to (u, v_1, ..., v_k) is singular mod
    p), by a Laplace expansion of the determinant into features of the two
    sides and a dense product. A superset of the exact solutions."""
    k = Kv.shape[1]
    n = k + 1
    dim = prhs.shape[0]
    f = rng.integers(1, p, size=(n, dim))
    coef = np.column_stack([d0v, Kv]) % p                                    # (r, n)
    T = [(((o % p) @ f.T % p)[:, :, None] * coef[i][None, None, :]) % p for i, o in enumerate(popts)]
    frhs = (f @ (prhs % p)) % p

    def side(bl, left):
        M = np.zeros((1, n, n), dtype=np.int64)
        for i in bl:
            M = (M[:, None, :, :] + T[i][None, :, :, :]).reshape(-1, n, n) % p
        if left:
            M[:, :, 0] = (M[:, :, 0] - frhs[None, :]) % p
        return M, _combos([T[i].shape[0] for i in bl])

    ML, idxL = side(sides[0], True)
    MR, idxR = side(sides[1], False)
    rows = list(range(n))
    feats = [(S, C) for kk in range(n + 1) for S in itertools.combinations(rows, kk)
             for C in itertools.combinations(rows, kk)]

    def features(M, left):
        F = np.empty((len(M), len(feats)), dtype=np.float64)
        for t, (S, C) in enumerate(feats):
            if left:
                F[:, t] = _det_mod(M[:, list(S), :][:, :, list(C)], p)
            else:
                Sc = [a for a in rows if a not in S]
                Cc = [a for a in rows if a not in C]
                sg = -1 if (sum(S) + sum(C)) % 2 else 1
                F[:, t] = (sg * _det_mod(M[:, Sc, :][:, :, Cc], p)) % p
        return F

    FL, FR = features(ML, True), features(MR, False)
    out = []
    chunk = max(1, 20_000_000 // len(FR))
    for s in range(0, len(FL), chunk):
        Z = FL[s:s + chunk] @ FR.T
        ia, ib = np.nonzero(np.fmod(Z, p) == 0)
        for a, b in zip(ia, ib):
            out.append(_assemble(sides, idxL, idxR, s + a, b, len(popts)))
        if len(out) > max_cand:
            raise AssertionError(f"{len(out)} slice candidates: structural degeneracy of the family")
    return out


def solve_slice(opts, blocks, fam, rhs, rng, max_cand=2_000_000, stats=None, log=None):
    """Solutions of sum_i d_i w_i + (block contributions) = rhs for some d
    in the family: a list of (combo, Ssel) with combo the option index per
    ordinary term and Ssel the translate set per block. opts: per ordinary
    term (o1, o2, oC) option arrays; rhs: (r1, r2, rC). Hashing mod P1, every
    candidate decided exactly (F_P1, C, F_P2)."""
    r = len(opts)
    d1, K1 = fam.parts[0]
    ords = [i for i in range(len(d1)) if i not in {b.pos for b in blocks}]
    assert len(ords) == r
    d0v = d1[ords]
    Kv = _independent_columns_mod(K1[ords].reshape(r, -1), P1)
    out = []
    for Ssel in itertools.product(*[b.subsets for b in blocks]):
        Ps, Vs = _projectors(blocks, Ssel)
        if Ps is None:
            popts = [o[0] for o in opts]
            prhs = rhs[0]
        else:
            popts = [(o[0] @ Ps[0].T) % P1 for o in opts]
            prhs = (Ps[0] @ rhs[0]) % P1
        sizes = [len(o) for o in popts]
        sides = _split_sides(sizes)
        t0 = time.time()
        if Kv.shape[1] == 0:
            cands = _mitm(popts, d0v, prhs, sides, rng)
        else:
            cands = _dense(popts, d0v, Kv, prhs, sides, rng, max_cand)
        if stats is not None:
            stats["candidates"] = stats.get("candidates", 0) + len(cands)
        if log is not None and (Kv.shape[1] or len(cands) > 100_000):
            log(f"    translate sets {Ssel}: {Kv.shape[1]} parameters, sizes {sizes}, "
                f"{len(cands)} candidates [{time.time() - t0:.1f}s]")
        if cands and Kv.shape[1] == 0:
            # the whole projected equation mod P1 at once; survivors go to the exact check
            Cm = np.array(cands, dtype=np.int64)
            acc = np.zeros((len(Cm), prhs.shape[0]), dtype=np.int64)
            for i in range(r):
                acc = (acc + (int(d0v[i]) * popts[i][Cm[:, i]]) % P1) % P1
            keep = np.all(acc == (prhs % P1)[None, :], axis=1)
            cands = [c for c, k in zip(cands, keep) if k]
        for combo in cands:
            if fam.restrict(*slice_system(opts, blocks, combo, Ssel, rhs, Ps)) is not None:
                out.append((combo, Ssel))
    return out


def slice_system(opts, blocks, combo, Ssel, rhs, Ps=None):
    """(Ws, rhss) of the slice equation for one solution, projected onto
    the annihilator of the blocks' translates; W has a zero column at every
    block position."""
    if Ps is None and blocks:
        Ps, _ = _projectors(blocks, Ssel)
    r_all = len(opts) + len(blocks)
    bpos = {b.pos for b in blocks}
    ords = [i for i in range(r_all) if i not in bpos]
    Ws, rhss = [], []
    for fld, p in ((0, P1), (1, P2), (2, None)):
        W = np.zeros((rhs[fld].shape[0], r_all), dtype=complex if p is None else np.int64)
        for i, c in zip(ords, combo):
            W[:, i] = opts[ords.index(i)][fld][c]
        if Ps is not None:
            W = _mm(Ps[fld], W, p) if p is not None else Ps[fld] @ W
            rh = (_mm(Ps[fld], rhs[fld][:, None], p)[:, 0] if p is not None else Ps[fld] @ rhs[fld])
        else:
            rh = rhs[fld]
        Ws.append(W)
        rhss.append(rh)
    return tuple(Ws), tuple(rhss)


# --------------------------------------------------------------- flats -----

def subspaces(n1):
    """All subspaces of F_2^{n1}, each as the frozenset of its nonzero points."""
    pts = range(1, 1 << n1)
    out = set()
    for k in range(n1 + 1):
        for gens in itertools.combinations(pts, k):
            span = {0}
            for g in gens:
                span |= {s ^ g for s in span}
            out.add(frozenset(span - {0}))
    return sorted(out, key=lambda W: (len(W), sorted(W)))


def is_subspace(W):
    W = set(W) | {0}
    return all((a ^ b) in W for a in W for b in W)


def flats_with_presence(n1, present):
    """Subspaces containing exactly the coordinate directions e_k, k in present."""
    present = set(present)
    return [W for W in subspaces(n1) if {k for k in range(n1) if (1 << k) in W} == present]


def composite_codes(o, W, ccode, composite):
    """Every assignment of option codes at the composite points of a term
    with flat W and coordinate codes ccode (k -> code), as rows over the
    points `composite` (o.absent outside W). The basis of W is the present
    coordinate directions and then further points, which are free (any of
    the 4 2^n codes); the other points are class products with one free
    sign per basis pair and the product of the pair signs on a triple (the
    quadratic form of the structure lemma)."""
    basis = [1 << k for k in sorted(ccode)]
    assert all(b in W for b in basis)
    span = {0}
    for b in basis:
        span |= {s ^ b for s in span}
    free = []
    for pnt in sorted(W):
        if pnt not in span:
            basis.append(pnt)
            free.append(pnt)
            span |= {s ^ pnt for s in span}
    j = len(basis)
    pairs = list(itertools.combinations(range(j), 2))
    rows = []
    for fcodes in itertools.product(range(o.absent), repeat=len(free)):
        bcode = {1 << k: c for k, c in ccode.items()}
        bcode.update(zip(free, fcodes))
        for signs in itertools.product((0, 1), repeat=len(pairs)):
            q = dict(zip(pairs, signs))
            code = {}
            for t in range(1, 1 << j):
                bits = [a for a in range(j) if t >> a & 1]
                pnt = 0
                for a in bits:
                    pnt ^= basis[a]
                c = bcode[basis[bits[0]]]
                for a in bits[1:]:
                    c = o.compose(c, bcode[basis[a]])
                s = sum(q[(a, b)] for a, b in itertools.combinations(bits, 2)) % 2
                if s:
                    c = 4 * (c // 4) + (c % 4 + 2) % 4
                code[pnt] = c
            rows.append([code.get(x, o.absent) for x in composite])
    if not composite:
        return np.zeros((1, 0), dtype=np.int64)
    return np.unique(np.array(rows, dtype=np.int64).reshape(-1, len(composite)), axis=0)


def valid_term_codes(o, n1, codes):
    """True when the codes (offset -> code, all nonzero offsets) are those of
    a stabilizer state with base slice u: the present offsets form a
    subspace and the composite codes are among the structure lemma's."""
    present = [x for x, c in codes.items() if c != o.absent]
    if not is_subspace(present):
        return False
    W = frozenset(present)
    ccode = {k: codes[1 << k] for k in range(n1) if (1 << k) in W}
    composite = [x for x in range(1, 1 << n1) if bin(x).count("1") >= 2]
    rows = composite_codes(o, W, ccode, composite)
    want = np.array([codes[x] for x in composite], dtype=np.int64)
    return bool((rows == want[None, :]).all(axis=1).any()) if composite else True


def copy_composite_rows(o, n1, composite, ccodes):
    """Every admissible assignment of composite codes for a copy of the base
    state of `o` whose coordinate codes are `ccodes` (a tuple over the n1
    coordinate directions, o.absent where the copy vanishes), as rows over
    the offsets `composite`: the union of composite_codes over every flat
    with that coordinate presence. Cached on `o`."""
    cache = o.__dict__.setdefault("_composite_rows", {})
    key = (n1, tuple(composite), tuple(int(c) for c in ccodes))
    if key not in cache:
        present = [k for k in range(n1) if ccodes[k] != o.absent]
        ccode = {k: int(ccodes[k]) for k in present}
        rows = [composite_codes(o, W, ccode, list(composite)) for W in flats_with_presence(n1, present)]
        cache[key] = np.unique(np.vstack(rows), axis=0) if composite else np.zeros((1, 0), dtype=np.int64)
    return cache[key]


def reconstruct_block(o, g, D, data, n1, rng):
    """Copies of a block from its per-slice residual coordinates. data: list
    of (offset, {class: coordinate}) over the 2^n1 - 1 non-base offsets, the
    n1 coordinate offsets first in order and then the composite offsets,
    with the block's contribution at that offset sum_k a_k Q_k u over the
    translate set the slice equation chose. Returns every assignment (as a
    list of (coefficient, codes) per copy, copies distinct and unordered)
    with coefficients summing to D, all nonzero, and each copy a valid
    stabilizer state; the flag says whether a coefficient family remained
    (degenerate).

    The translate set records the classes with a nonzero net coordinate
    only. Two or more copies can sit in one further class with phases and
    coefficients that cancel there (c_1 i^{l_1} + c_2 i^{l_2} = 0), which
    the slice equation cannot see, so at every offset the copies may also
    use classes outside the set, each by at least two copies with net
    coordinate zero, as long as the classes used number at most g. Once a
    copy's coordinate codes are fixed its composite codes are restricted to
    the structure lemma's shapes for them (copy_composite_rows), which keeps
    the extra freedom small; valid_term_codes decides every copy at the end
    regardless."""
    K = 1 << o.n
    ones = np.ones((g, 1), dtype=complex)
    c0 = np.full(g, D / g, dtype=complex)
    Kc = _affine_solve_C(ones.T, np.zeros(1, dtype=complex))[1]     # sum-zero directions
    if [x for x, _ in data[:n1]] != [1 << k for k in range(n1)]:
        raise AssertionError("block data must start with the coordinate offsets in order")
    composite = [x for x, _ in data[n1:]]
    A_code = o.absent
    results = []

    def allowed(idx, codes):
        """Per copy the set of codes admissible at data[idx] given the codes
        chosen so far, or None when any code is (the coordinate offsets)."""
        if idx < n1:
            return [None] * g
        out = []
        for cd in codes:
            rows = copy_composite_rows(o, n1, composite, tuple(cd[1 << k] for k in range(n1)))
            mask = np.ones(len(rows), dtype=bool)
            for c in range(idx - n1):
                mask &= rows[:, c] == cd[composite[c]]
            out.append({int(v) for v in rows[mask, idx - n1]})
        return out

    def dfs(idx, c0, Kc, codes):
        if idx == len(data):
            degenerate = Kc.shape[1] > 0
            c = c0 + Kc @ (rng.normal(size=Kc.shape[1]) + 1j * rng.normal(size=Kc.shape[1])) if degenerate else c0
            if np.any(np.abs(c) < 1e-9):
                return
            full = [dict(cd) for cd in codes]
            for cd in full:
                if not valid_term_codes(o, n1, cd):
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
        others = [k for k in range(K) if k not in a]
        for nz in range(1, (g - len(S)) // 2 + 1):
            extra += list(itertools.combinations(others, nz))
        for Z in extra:
            classes = S + list(Z)
            b = np.array([a[k] for k in S] + [0.0] * len(Z), dtype=complex)
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
                        phase_opts.append(range(4))
                        continue
                    ls = [v % 4 for v in al if v != A_code and v // 4 == assign[j]]
                    if not ls:
                        break
                    phase_opts.append(ls)
                else:
                    for phases in itertools.product(*phase_opts):
                        A = np.zeros((len(classes), g), dtype=complex)
                        for j, l in zip(present, phases):
                            A[classes.index(assign[j]), j] = FOURTH[l]
                        sol = _affine_solve_C(A @ Kc, b - A @ c0)
                        if sol is None:
                            continue
                        mu0, N = sol
                        new = [dict(cd) for cd in codes]
                        for j in range(g):
                            new[j][x] = A_code
                        for j, l in zip(present, phases):
                            new[j][x] = 4 * assign[j] + l
                        dfs(idx + 1, c0 + Kc @ mu0, Kc @ N, new)

    dfs(0, c0, Kc, [dict() for _ in range(g)])
    # copies are unordered: keep one representative per set of copies
    seen, out = set(), []
    for copies, degenerate in results:
        key = tuple(sorted((tuple(sorted(cd.items())), complex(np.round(c, 8))) for c, cd in copies))
        if key not in seen:
            seen.add(key)
            out.append((copies, degenerate))
    return out


def _refine_split(D, S, coords, splits, tol=1e-7):
    """Admissible (c_1, c_2), c_1 + c_2 = D, for a pair of copies whose slice
    contribution has coordinates `coords` on the translates S; None means
    unconstrained. Two translates: c_1 i^l_1 = a_1, c_2 i^l_2 = a_2 (either
    order). One translate: both copies in the class (c_1 i^l_1 + c_2 i^l_2 =
    a, pinned when l_1 != l_2, free when l_1 = l_2 and a = D i^l) or one copy
    absent (c_j i^l = a). No translate: no constraint. The result is
    intersected with the previous candidates."""
    def close(x, y):
        return abs(x - y) < tol * max(1.0, abs(D))
    if len(S) == 0:
        return splits
    cands, free = [], False
    if len(S) == 2:
        a1, a2 = coords
        for l1 in range(4):
            for l2 in range(4):
                c1, c2 = a1 / FOURTH[l1], a2 / FOURTH[l2]
                if close(c1 + c2, D):
                    cands += [(c1, c2), (c2, c1)]
    else:
        a = coords[0]
        for l1 in range(4):
            for l2 in range(4):
                if l1 == l2:
                    if close(a, D * FOURTH[l1]):
                        free = True
                else:
                    c1 = (a - D * FOURTH[l2]) / (FOURTH[l1] - FOURTH[l2])
                    cands.append((c1, D - c1))
        for l in range(4):
            cj = a / FOURTH[l]
            cands += [(cj, D - cj), (D - cj, cj)]
    cands = [(c1, c2) for c1, c2 in cands if abs(c1) > tol and abs(c2) > tol]
    if free:
        return splits                                    # this slice adds no constraint
    if splits is None:
        return cands
    return [(c1, c2) for c1, c2 in splits if any(close(c1, e1) and close(c2, e2) for e1, e2 in cands)]


# ------------------------------------------------------------- matching ----

def _native_kernel_class():
    """The compiled stage A matcher from stabrank_core, or None."""
    if os.environ.get("STABRANK_NO_NATIVE"):
        return None
    try:
        from stabrank.stabrank_core import SliceMatchKernel
    except ImportError:
        return None
    return SliceMatchKernel


class SliceMatcher:
    """Every decomposition of psi^{n1 + n2} with a given base slice at a
    given base point. The multiset cover is split into its distinct states;
    their coefficients form an affine family (a point when the states are
    independent), repeated states become blocks (span of at most g
    translates at every slice). The n1 coordinate slices are solved and
    joined, each join restricting the family; the absence pattern fixes the
    flat of every ordinary term; the composite slices are solved point by
    point over the composite code assignments; the blocks' copies are
    reconstructed from the residuals; every hit is confirmed against
    psi^{n1 + n2}."""

    def __init__(self, enum, n1, verbose=False, seed=29, native=True):
        self.E = enum
        self.n1, self.n2 = n1, enum.n
        self.F1, self.F2 = enum.F1, enum.F2
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        # the compiled stage A kernel (cpp/src/slice_match.cpp) for distinct,
        # independent base states; STABRANK_NO_NATIVE=1 or native=False keeps
        # everything in this module, which stays the reference
        self.native = None
        cls = _native_kernel_class() if native else None
        if cls is not None:
            self.native = cls(np.ascontiguousarray(enum.codes, dtype=np.int8), n1,
                              self.target_slices(self.F1, enum.psi1), self.target_slices(self.F2, enum.psi2),
                              seed)

    def target_slices(self, F, psi_p):
        """The slices psi^{n1 + n2} restricted to x on the sliced qubits, as
        the (2^n1, 2^n2) array alpha_x psi^{n2} over F_p."""
        n1 = self.n1
        out = np.zeros((1 << n1, len(psi_p)), dtype=np.int64)
        for x in range(1 << n1):
            w = bin(x).count("1")
            alpha = pow(F.cos, n1 - w, F.p) * pow(F.sin, w, F.p) % F.p
            out[x] = [int(v) * alpha % F.p for v in psi_p]
        return out

    def _run_native(self, cover, x0, stats):
        """Stage A through the compiled kernel; None when the kernel declines
        (repeated or dependent base states, or a refused cover), in which case
        the caller runs the reference path."""
        res = self.native.run([int(c) for c in cover], int(x0))
        if res["status"] != 0:
            return None
        hits = []
        for h in np.asarray(res["hits"]):
            terms = [np.where(c > 0, FOURTH[(c.astype(np.int64) - 1) % 4], 0) for c in h]
            hits.append(self.confirm(terms, None, 0))
        hits = self._dedupe(hits)
        stats.update(kappa=0, kappa1=0, distinct=len(cover), blocks=[],
                     coord_solutions=[int(v) for v in res["coord_solutions"]], joined=int(res["joined"]),
                     types=int(res["types"]), composite_solutions=int(res["composite_solutions"]),
                     candidates=int(res["candidates"]), hits=len(hits), native=True)
        return hits, stats

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def options(self, idx):
        if idx not in self.cache:
            self.cache[idx] = TermOptions(self.E.C[:, idx], self.n2, self.F1, self.F2)
        return self.cache[idx]

    def rhs(self, x0, x):
        e = bin(x).count("1") - bin(x0).count("1")
        r1 = (self.E.psi1 * pow(self.F1.tan, e % (P1 - 1), P1)) % P1
        r2 = (self.E.psi2 * pow(self.F2.tan, e % (P2 - 1), P2)) % P2
        rC = self.E.psi * (np.tan(np.pi / 8) ** e)
        return r1, r2, rC

    def run(self, cover, x0):
        """(hits, stats) for the base slice `cover` (a tuple of dictionary
        indices, repeats allowed) at x0; each hit is a dict with the term
        vectors, coefficients, residual and rank."""
        n1 = self.n1
        stats = {"kappa": None, "kappa1": None, "distinct": 0, "blocks": [], "coord_solutions": [],
                 "joined": 0, "types": 0, "composite_solutions": 0, "candidates": 0,
                 "zero_coefficient": 0, "split_pruned": 0, "reconstructions": 0, "unpinned": 0, "hits": 0,
                 "refused": False, "native": False}
        if self.native is not None and len(set(cover)) == len(cover):
            out = self._run_native(cover, x0, stats)
            if out is not None:
                return out
        distinct = sorted(set(cover))
        mult = {u: cover.count(u) for u in distinct}
        fam = Family.from_cover(self.E, distinct)
        if fam is None:
            stats["refused"] = True
            return [], stats
        blocks = [Block(self.options(u), mult[u], i) for i, u in enumerate(distinct) if mult[u] > 1]
        bpos = {b.pos for b in blocks}
        exempt = tuple(sorted(bpos))
        if fam.has_zero_coefficient(exempt):
            stats["refused"] = True                   # a distinct state no member of the family uses
            return [], stats
        ords = [i for i in range(len(distinct)) if i not in bpos]
        opts = [self.options(distinct[i]) for i in ords]
        arrays = [o.arrays() for o in opts]
        stats.update(kappa=fam.kappa, kappa1=fam.kappa1, distinct=len(distinct), blocks=[b.g for b in blocks])
        coord = [x0 ^ (1 << k) for k in range(n1)]
        composite = [x for x in range(1, 1 << n1) if bin(x).count("1") >= 2]
        # a state: ordinary combos, block translate sets, family, split candidates per block
        states = [([], [], fam, [None] * len(blocks))]
        for k in range(n1):
            rhs = self.rhs(x0, coord[k])
            new, count = [], 0
            for cl, bl, f, sp in states:
                sols = solve_slice(arrays, blocks, f, rhs, self.rng, stats=stats, log=self.log)
                count += len(sols)
                for combo, Ssel in sols:
                    f2 = f.restrict(*slice_system(arrays, blocks, combo, Ssel, rhs))
                    if f2 is None or f2.has_zero_coefficient(exempt):
                        stats["zero_coefficient"] += f2 is not None
                        continue
                    sp2 = self._join_blocks(f2, arrays, blocks, combo, Ssel, rhs, sp)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    new.append((cl + [combo], bl + [Ssel], f2, sp2))
            stats["coord_solutions"].append(count)
            states = new
            self.log(f"  slice {k}: {count} solutions, {len(states)} states, "
                     f"parameters {sorted(set(f.kappa for _, _, f, _ in states))}")
            if not states:
                return [], stats
        stats["joined"] = len(states)
        hits = []
        for cl, bl, f, sp in states:
            types = []
            for i, o in enumerate(opts):
                present = [k for k in range(n1) if cl[k][i] != o.absent]
                types.append(flats_with_presence(n1, present))
            for Wsel in itertools.product(*types):
                stats["types"] += 1
                hits.extend(self._complete(distinct, x0, opts, blocks, cl, bl, Wsel, f, sp, coord, composite, stats))
        hits = self._dedupe(hits)
        stats["hits"] = len(hits)
        return hits, stats

    def _join_blocks(self, fam, arrays, blocks, combo, Ssel, rhs, splits):
        """The blocks' part of a join: with a pinned family the residual must
        use every chosen translate (else the solution duplicates a smaller
        set), and for a pair the admissible coefficient splits (c_1, c_2) with
        c_1 + c_2 = D are refined: a two-translate slice pins them up to the
        phases, a one-translate slice constrains them once pinned. Returns the
        new split lists, or None to drop the state."""
        if not blocks or fam.kappa:
            return splits
        a = self._block_coordinates(fam, arrays, blocks, combo, Ssel, rhs)
        if a is None:
            return splits                                # ambiguous split between blocks: kept
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

    def _block_coordinates(self, fam, arrays, blocks, combo, Ssel, rhs, strict=False):
        """Coordinates of the residual rhs - W d on the chosen translates
        (complex), or None when it does not lie in their span or the split
        between blocks is ambiguous (the chosen translates of different
        blocks dependent). In the join None means no pruning; at the final
        reconstruction (`strict`) it would mean a silently dropped state, so
        the run raises UnpinnedFamily there instead."""
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
            if np.linalg.norm(res) < 1e-7:
                return np.zeros(0)
            if strict:
                raise UnpinnedFamily("reconstruction: a slice residual is nonzero with no translate chosen")
            return None
        sol = _affine_solve_C(V, res)
        if sol is None:
            if strict:
                raise UnpinnedFamily("reconstruction: a slice residual lies outside the span of the chosen "
                                     "translates at the solve tolerance although the exact restriction accepted it")
            return None
        if sol[1].shape[1]:
            if strict:
                raise UnpinnedFamily(f"reconstruction: the chosen translates of the blocks are dependent "
                                     f"({sol[1].shape[1]} free coordinates), so the split of the residual between "
                                     "the blocks is a family; carrying it as unknowns is not implemented")
            return None
        return sol[0]

    def _complete(self, distinct, x0, opts, blocks, cl, bl, Wsel, fam, sp, coord, composite, stats):
        n1 = self.n1
        r = len(opts)
        rows = []
        for i in range(r):
            ccode = {k: cl[k][i] for k in range(n1) if cl[k][i] != opts[i].absent}
            rows.append(composite_codes(opts[i], Wsel[i], ccode, composite))
        exempt = tuple(sorted(b.pos for b in blocks))
        states = [([], [], fam, sp)]
        for c, x in enumerate(composite):
            rhs = self.rhs(x0, x0 ^ x)
            codes = [np.unique(rw[:, c]) for rw in rows]
            arrays = [(o.m1[cd], o.m2[cd], o.vecs[cd]) for o, cd in zip(opts, codes)]
            new = []
            for cl2, bl2, f, sp1 in states:
                sols = solve_slice(arrays, blocks, f, rhs, self.rng, stats=stats, log=self.log)
                stats["composite_solutions"] += len(sols)
                for combo, Ssel in sols:
                    f2 = f.restrict(*slice_system(arrays, blocks, combo, Ssel, rhs))
                    if f2 is None or f2.has_zero_coefficient(exempt):
                        stats["zero_coefficient"] += f2 is not None
                        continue
                    sp2 = self._join_blocks(f2, arrays, blocks, combo, Ssel, rhs, sp1)
                    if sp2 is None:
                        stats["split_pruned"] += 1
                        continue
                    new.append((cl2 + [tuple(int(codes[i][combo[i]]) for i in range(r))], bl2 + [Ssel], f2, sp2))
            states = new
            if not states:
                return []
        hits = []
        offsets = [1 << k for k in range(n1)] + composite
        for cl2, bl2, f, _ in states:
            if blocks and f.kappa > 0:
                stats["unpinned"] = stats.get("unpinned", 0) + 1
                raise UnpinnedFamily(f"{f.kappa}-parameter coefficient family left after the {(1 << n1) - 1} "
                                     f"slice equations on a base with a repeated state (blocks "
                                     f"{[b.g for b in blocks]}); the block reconstruction would use an "
                                     "arbitrary member of the family")
            ord_terms = []
            for i in range(r):
                o = opts[i]
                match = np.ones(len(rows[i]), dtype=bool)
                for c in range(len(composite)):
                    match &= rows[i][:, c] == cl2[c][i]
                if not np.any(match):
                    break
                t = np.zeros((1 << n1, 1 << self.n2), dtype=complex)
                t[x0] = o.u
                for k in range(n1):
                    t[coord[k]] = o.vecs[cl[k][i]]
                for c, x in enumerate(composite):
                    t[x0 ^ x] = o.vecs[cl2[c][i]]
                ord_terms.append(t.ravel())
            else:
                d = f.coefficients(self.rng)
                bpos = {b.pos for b in blocks}
                ords = [i for i in range(len(distinct)) if i not in bpos]
                coeffs = [d[i] for i in ords]
                if not blocks:
                    hits.append(self.confirm(ord_terms, coeffs, f.kappa))
                    continue
                # residual coordinates per slice, split over the blocks; a
                # residual the strict solve cannot place raises UnpinnedFamily
                per_block = [[] for _ in blocks]
                for s, x in enumerate(offsets):
                    if s < n1:
                        arrays_s, combo, Ssel = [o.arrays() for o in opts], cl[s], bl[s]
                    else:
                        c = s - n1
                        combo = tuple(cl2[c][i] for i in range(r))
                        arrays_s, Ssel = [o.arrays() for o in opts], bl2[c]
                    rhs = self.rhs(x0, x0 ^ x)
                    a = self._block_coordinates(f, arrays_s, blocks, combo, Ssel, rhs, strict=True)
                    pos = 0
                    for bi, (b, S) in enumerate(zip(blocks, Ssel)):
                        per_block[bi].append((x, {k: a[pos + j] for j, k in enumerate(S)}))
                        pos += len(S)
                recon = [reconstruct_block(b.o, b.g, d[b.pos], per_block[bi], n1, self.rng)
                         for bi, b in enumerate(blocks)]
                stats["reconstructions"] += 1
                for choice in itertools.product(*recon):
                    terms, cs = list(ord_terms), list(coeffs)
                    degenerate = f.kappa > 0
                    for b, (copies, deg) in zip(blocks, choice):
                        degenerate |= deg
                        for cval, cd in copies:
                            t = np.zeros((1 << n1, 1 << self.n2), dtype=complex)
                            t[x0] = b.o.u
                            for x, code in cd.items():
                                t[x0 ^ x] = b.o.vecs[code]
                            terms.append(t.ravel())
                            cs.append(cval)
                    hits.append(self.confirm(terms, cs, int(degenerate)))
        return hits

    def confirm(self, terms, coeffs, free):
        """Residual, coefficients and rank of the terms against psi^m, in
        floating point and (the span condition) mod P2."""
        m = self.n1 + self.n2
        res, c = confirm_decomposition(terms, m)
        A = np.column_stack(terms)
        rank = int(np.linalg.matrix_rank(A, tol=1e-8))
        codes = np.array([exact_codes(t)[0] for t in terms])
        U2 = self.F2.codes_to_field(codes)
        r0 = _rank_mod(U2, P2)
        r1 = _rank_mod(np.vstack([U2, self.F2.target(m)]), P2)
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


def x0_reps(n1):
    """One base point per Hamming weight."""
    return [(1 << w) - 1 for w in range(n1 + 1)]


def confirm_decomposition(terms, m):
    """Residual of psi^m against the span of the terms and the coefficients."""
    A = np.column_stack(terms)
    psi = psi_for("qubit_H", m)
    c, *_ = np.linalg.lstsq(A, psi, rcond=None)
    return float(np.linalg.norm(A @ c - psi)), c
