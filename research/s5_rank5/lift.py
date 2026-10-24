"""Lift stage (round 33, agent B): from a base (a k-multiset of (n-1)-qutrit letters w_j with a coefficient family
such that sum_j b_j w_j = S^{(x)(n-1)}) to every stabilizer decomposition of S^{(x) n} with k terms, all full on the
last qutrit, whose <1| slice is the base. By Lemma F.1 (card S0540) each term is
    s_j = zeta'_j P_j^{-1} w_j (x) |0> + w_j (x) |1> + zeta_j P_j w_j (x) |2>,
with P_j a Pauli (only its coset modulo the stabilizer group of w_j matters: 3^(n-1) rays) and zeta_j, zeta'_j cube
roots of unity, and the equations are
    E2:  sum_j b_j zeta_j P_j w_j = -t,      E0:  sum_j b_j zeta'_j P_j^{-1} w_j = 0,
with t = S^{(x)(n-1)} unnormalized (its slices are 0, t, -t). For a base with unique coefficients (kappa = 0) E2 is
solved by meeting in the middle over the options (ray, zeta) of the terms, keyed by two random functionals modulo
P (a necessary condition; every match is re-verified on the full vectors), then E0 over the 3^k phase choices. For
kappa = 1 (coefficients b0 + lambda beta) the condition is that A = -t - sum b0_j zeta_j P_j w_j and
B = sum beta_j zeta_j P_j w_j are parallel; all option tuples are enumerated with a two-functional determinant
screen and survivors are verified. Every output is verified modulo both primes and, when asked, exactly over Z[omega].
"""
import numpy as np, itertools, os, sys
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L

PACK = 1 << 31

class Rays:
    """coset rays of letters on n qutrits, cached by canonical code."""
    def __init__(self, n):
        self.n = n; self.X, self.src, self.ph = L.pauli_maps(n); self.cache = {}
    def get(self, code):
        k = code.tobytes()
        if k not in self.cache:
            rays = L.coset_rays(code, self.n, self.src, self.ph)
            self.cache[k] = (np.array([r[0] for r in rays]), np.array([r[1] for r in rays]))
        return self.cache[k]

def option_tables(base_codes, rays, P):
    """for each term j: codes of the rays P w_j (R, D) and of P^-1 w_j (R, D)."""
    fw = []; bw = []
    for c in base_codes:
        a, b = rays.get(c); fw.append(a); bw.append(b)
    return fw, bw

@njit(cache=True)
def _dot(f, v, P):
    s = 0
    for j in range(len(v)):
        if v[j]: s = (s + f[j] * v[j]) % P
    return s

@njit(cache=True)
def _funcvals(f, g, vals, P):
    """(R, 2) functional values of the rows of vals (R, D)."""
    R = vals.shape[0]; out = np.empty((R, 2), dtype=np.int64)
    for r in range(R):
        out[r, 0] = _dot(f, vals[r], P); out[r, 1] = _dot(g, vals[r], P)
    return out

@njit(cache=True)
def _mitm_k5(FV, b, w3, ftarget, P, maxsol):
    """FV: (5, R, 2) functional values of the rays; option o = 3*r + z means ray r with phase omega^z; the term value
    is b_j omega^z ray. Solutions of sum_j b_j omega^z_j ray_{j, r_j} = target (given by ftarget (2,)) under the
    two functionals. Returns (nsol, sols (maxsol, 5) option indices)."""
    R = FV.shape[1]; nopt = 3 * R
    # per-term per-option functional values
    T = np.empty((5, nopt, 2), dtype=np.int64)
    for j in range(5):
        for r in range(R):
            for z in range(3):
                for q in range(2):
                    T[j, 3 * r + z, q] = (b[j] * w3[z] % P) * FV[j, r, q] % P
    # left: terms 0, 1
    nl = nopt * nopt; keys = np.empty(nl, dtype=np.int64)
    for o0 in range(nopt):
        for o1 in range(nopt):
            k0 = (T[0, o0, 0] + T[1, o1, 0]) % P; k1 = (T[0, o0, 1] + T[1, o1, 1]) % P
            keys[o0 * nopt + o1] = k0 * PACK + k1
    order = np.argsort(keys); sk = keys[order]
    sols = np.empty((maxsol, 5), dtype=np.int64); nsol = 0
    for o2 in range(nopt):
        for o3 in range(nopt):
            a0 = (T[2, o2, 0] + T[3, o3, 0]) % P; a1 = (T[2, o2, 1] + T[3, o3, 1]) % P
            for o4 in range(nopt):
                k0 = (ftarget[0] - a0 - T[4, o4, 0]) % P; k1 = (ftarget[1] - a1 - T[4, o4, 1]) % P
                key = k0 * PACK + k1
                lo = np.searchsorted(sk, key, side='left')
                while lo < nl and sk[lo] == key:
                    idx = order[lo]
                    if nsol < maxsol:
                        sols[nsol, 0] = idx // nopt; sols[nsol, 1] = idx % nopt; sols[nsol, 2] = o2; sols[nsol, 3] = o3; sols[nsol, 4] = o4
                    nsol += 1
                    lo += 1
    return nsol, sols

@njit(cache=True)
def _mitm_k4(FV, b, w3, ftarget, P, maxsol):
    R = FV.shape[1]; nopt = 3 * R
    T = np.empty((4, nopt, 2), dtype=np.int64)
    for j in range(4):
        for r in range(R):
            for z in range(3):
                for q in range(2):
                    T[j, 3 * r + z, q] = (b[j] * w3[z] % P) * FV[j, r, q] % P
    nl = nopt * nopt; keys = np.empty(nl, dtype=np.int64)
    for o0 in range(nopt):
        for o1 in range(nopt):
            keys[o0 * nopt + o1] = ((T[0, o0, 0] + T[1, o1, 0]) % P) * PACK + (T[0, o0, 1] + T[1, o1, 1]) % P
    order = np.argsort(keys); sk = keys[order]
    sols = np.empty((maxsol, 4), dtype=np.int64); nsol = 0
    for o2 in range(nopt):
        for o3 in range(nopt):
            k0 = (ftarget[0] - T[2, o2, 0] - T[3, o3, 0]) % P; k1 = (ftarget[1] - T[2, o2, 1] - T[3, o3, 1]) % P
            key = k0 * PACK + k1
            lo = np.searchsorted(sk, key, side='left')
            while lo < nl and sk[lo] == key:
                idx = order[lo]
                if nsol < maxsol:
                    sols[nsol, 0] = idx // nopt; sols[nsol, 1] = idx % nopt; sols[nsol, 2] = o2; sols[nsol, 3] = o3
                nsol += 1
                lo += 1
    return nsol, sols

@njit(cache=True)
def _mitm_k3(FV, b, w3, ftarget, P, maxsol):
    R = FV.shape[1]; nopt = 3 * R
    T = np.empty((3, nopt, 2), dtype=np.int64)
    for j in range(3):
        for r in range(R):
            for z in range(3):
                for q in range(2):
                    T[j, 3 * r + z, q] = (b[j] * w3[z] % P) * FV[j, r, q] % P
    keys = np.empty(nopt, dtype=np.int64)
    for o0 in range(nopt): keys[o0] = T[0, o0, 0] * PACK + T[0, o0, 1]
    order = np.argsort(keys); sk = keys[order]
    sols = np.empty((maxsol, 3), dtype=np.int64); nsol = 0
    for o1 in range(nopt):
        for o2 in range(nopt):
            k0 = (ftarget[0] - T[1, o1, 0] - T[2, o2, 0]) % P; k1 = (ftarget[1] - T[1, o1, 1] - T[2, o2, 1]) % P
            key = k0 * PACK + k1
            lo = np.searchsorted(sk, key, side='left')
            while lo < nopt and sk[lo] == key:
                if nsol < maxsol:
                    sols[nsol, 0] = order[lo]; sols[nsol, 1] = o1; sols[nsol, 2] = o2
                nsol += 1
                lo += 1
    return nsol, sols

def mitm(FV, b, P, ftarget, maxsol=2000000):
    k = FV.shape[0]; w = L.cube_root(P); w3 = np.array([1, w, w * w % P], dtype=np.int64)
    fn = {5: _mitm_k5, 4: _mitm_k4, 3: _mitm_k3}[k]
    nsol, sols = fn(FV, np.asarray(b, dtype=np.int64), w3, np.asarray(ftarget, dtype=np.int64), P, maxsol)
    assert nsol <= maxsol, ('mitm overflow', nsol)
    return sols[:nsol]

class Lifter:
    """lifts of bases of (n-1)-qutrit letters to n qutrits (all terms full on the last qutrit)."""
    def __init__(self, n, P=L.PRIMES[0], seed=2026, targets=None):
        self.n = n; self.P = P; self.rays = Rays(n - 1); self.tab = L.modtab(P); self.D = 3 ** (n - 1)
        rng = np.random.default_rng(seed)
        self.f = rng.integers(1, P, size=self.D, dtype=np.int64); self.g = rng.integers(1, P, size=self.D, dtype=np.int64)
        self.t = L.target(n - 1) % P
        self.w3 = np.array([1, L.cube_root(P), L.cube_root(P) ** 2 % P], dtype=np.int64)
        # slice targets (t1, t2, t0) of the three equations E1, E2, E0; for S^n they are (t, -t, 0)
        if targets is None: self.t1, self.t2, self.t0 = self.t, (-self.t) % P, np.zeros(self.D, dtype=np.int64)
        else: self.t1, self.t2, self.t0 = [np.asarray(x, dtype=np.int64) % P for x in targets]

    def solve_coefficients(self, base_codes):
        """unique coefficients b (mod P) with sum b_j w_j = t, or None if the letters are dependent / t not in span."""
        from census import decide
        V = self.tab[np.asarray(base_codes)]; k = len(base_codes)
        idx = np.arange(k, dtype=np.int64); coef = np.zeros(max(k, 6), dtype=np.int64)
        rank, rank_t, allnz = decide(V, idx, k, self.t1, self.P, coef)
        if rank == k and rank_t == k: return coef[:k].copy(), allnz
        return None, False

    def lift(self, base_codes, b, verify_exact=False):
        """all k-term lifts of the base with coefficients b (mod P): list of dicts with the lifted term codes
        (n-qutrit, canonical), options, and phases. Requires kappa = 0 (b unique)."""
        P = self.P; k = len(base_codes); base_codes = np.asarray(base_codes)
        fw, bw = option_tables(base_codes, self.rays, P)
        R = fw[0].shape[0]
        FV = np.stack([_funcvals(self.f, self.g, self.tab[fw[j]], P) for j in range(k)])     # (k, R, 2)
        ftarget = np.array([_dot(self.f, self.t2, P), _dot(self.g, self.t2, P)], dtype=np.int64)
        sols = mitm(FV, b, P, ftarget)
        out = []
        Vfw = [self.tab[fw[j]] for j in range(k)]; Vbw = [self.tab[bw[j]] for j in range(k)]
        for s in sols:
            # full verification of E2
            acc = np.zeros(self.D, dtype=np.int64)
            for j in range(k):
                r, z = divmod(int(s[j]), 3); acc = (acc + b[j] * self.w3[z] % P * Vfw[j][r]) % P
            if not np.array_equal(acc, self.t2): continue
            # E0 over the phase choices
            for zs in itertools.product(range(3), repeat=k):
                acc0 = np.zeros(self.D, dtype=np.int64)
                for j in range(k):
                    r = int(s[j]) // 3; acc0 = (acc0 + b[j] * self.w3[zs[j]] % P * Vbw[j][r]) % P
                if not np.array_equal(acc0, self.t0): continue
                terms = []
                for j in range(k):
                    r, z = divmod(int(s[j]), 3)
                    c0 = bw[j][r].copy(); nz0 = c0 != 0; c0 = np.where(nz0, ((c0 - 1 + zs[j]) % 3) + 1, 0).astype(np.int8)
                    c2 = fw[j][r].copy(); nz2 = c2 != 0; c2 = np.where(nz2, ((c2 - 1 + z) % 3) + 1, 0).astype(np.int8)
                    terms.append(L.canon(L.assemble_last(c0, base_codes[j], c2).reshape(1, -1))[0])
                out.append(dict(terms=np.array(terms), options=[int(x) for x in s], z0=list(zs)))
        return out

    def verify(self, terms, b, P2=L.PRIMES[1], exact=False):
        """check sum_j b_j terms_j = S^{(x) n} modulo P (b given mod P) and, independently, that the n-qutrit target
        lies in the span of the terms modulo the second prime (and exactly if asked)."""
        from census import decide
        tn = L.target(self.n) if not hasattr(self, 'tn_custom') else self.tn_custom
        V = self.tab[terms]; acc = np.zeros(3 ** self.n, dtype=np.int64)
        # the lifted term with the canonical phase may differ from the assembled one by a cube root: re-solve instead
        k = len(terms); idx = np.arange(k, dtype=np.int64); coef = np.zeros(max(k, 6), dtype=np.int64)
        rank, rank_t, allnz = decide(V, idx, k, tn % self.P, self.P, coef)
        ok1 = (rank_t == rank) and allnz
        V2 = L.modtab(P2)[terms]; coef2 = np.zeros(max(k, 6), dtype=np.int64)
        rank2, rank_t2, allnz2 = decide(V2, idx, k, tn % P2, P2, coef2)
        ok2 = (rank_t2 == rank2) and allnz2
        if rank < k: ok1 = (rank_t == rank); ok2 = (rank_t2 == rank2)        # dependent terms: coefficients form a family
        res = dict(ok=bool(ok1 and ok2), rank=int(rank), rank_t=int(rank_t), allnz=bool(allnz), rank2=int(rank2), rank_t2=int(rank_t2))
        if exact:
            r1, r2 = L.exact_cover_check(terms, tn); res['exact'] = (int(r1), int(r2)); res['ok'] = res['ok'] and r1 == r2
        return res


# ---------- kappa = 1 bases: coefficient family b0 + lambda beta ----------

@njit(cache=True)
def _brute_k5_family(FA, FB, P, maxsol, o0lo, o0hi):
    """FA[j, o, q], FB[j, o, q]: functional values (q = 0, 1 for f, g) of b0_j zeta v_j and beta_j zeta v_j for option o
    of term j (the constant -t is folded into term 0 of FA). Finds all option tuples with
    det[(fA, gA), (fB, gB)] = 0 where A = sum_j FA[j, o_j], B = sum_j FB[j, o_j]. Term 0's options restricted to
    [o0lo, o0hi) for parallel runs."""
    nopt = FA.shape[1]
    sols = np.empty((maxsol, 5), dtype=np.int64); nsol = 0
    for o0 in range(o0lo, o0hi):
        fa0 = FA[0, o0, 0]; ga0 = FA[0, o0, 1]; fb0 = FB[0, o0, 0]; gb0 = FB[0, o0, 1]
        for o1 in range(nopt):
            fa1 = (fa0 + FA[1, o1, 0]) % P; ga1 = (ga0 + FA[1, o1, 1]) % P; fb1 = (fb0 + FB[1, o1, 0]) % P; gb1 = (gb0 + FB[1, o1, 1]) % P
            for o2 in range(nopt):
                fa2 = (fa1 + FA[2, o2, 0]) % P; ga2 = (ga1 + FA[2, o2, 1]) % P; fb2 = (fb1 + FB[2, o2, 0]) % P; gb2 = (gb1 + FB[2, o2, 1]) % P
                for o3 in range(nopt):
                    fa3 = (fa2 + FA[3, o3, 0]) % P; ga3 = (ga2 + FA[3, o3, 1]) % P; fb3 = (fb2 + FB[3, o3, 0]) % P; gb3 = (gb2 + FB[3, o3, 1]) % P
                    for o4 in range(nopt):
                        fa = (fa3 + FA[4, o4, 0]) % P; ga = (ga3 + FA[4, o4, 1]) % P; fb = (fb3 + FB[4, o4, 0]) % P; gb = (gb3 + FB[4, o4, 1]) % P
                        if (fa * gb - ga * fb) % P == 0 and not (fb == 0 and gb == 0 and (fa != 0 or ga != 0)):
                            if nsol < maxsol:
                                sols[nsol, 0] = o0; sols[nsol, 1] = o1; sols[nsol, 2] = o2; sols[nsol, 3] = o3; sols[nsol, 4] = o4
                            nsol += 1
    return nsol, sols

def lift_family(lifter, base_codes, b0, beta, o0range=None, maxsol=5000000, log=None):
    """all 5-term lifts of a kappa = 1 base (coefficients b0 + lambda beta, mod P): E2 requires A = -t - sum b0_j zeta_j P_j w_j
    parallel to B = sum beta_j zeta_j P_j w_j (lambda = ratio), then E0 with that lambda; every survivor verified in
    full. Returns the list of lifts (terms, options, z0, lambda, coefficients)."""
    P = lifter.P; k = 5; base_codes = np.asarray(base_codes)
    fw, bw = option_tables(base_codes, lifter.rays, P); R = fw[0].shape[0]; nopt = 3 * R
    FV = np.stack([_funcvals(lifter.f, lifter.g, lifter.tab[fw[j]], P) for j in range(k)])   # (k, R, 2)
    FA = np.zeros((k, nopt, 2), dtype=np.int64); FB = np.zeros((k, nopt, 2), dtype=np.int64)
    for j in range(k):
        for r in range(R):
            for z in range(3):
                for q in range(2):
                    FA[j, 3 * r + z, q] = (-(b0[j] * lifter.w3[z] % P) * FV[j, r, q]) % P
                    FB[j, 3 * r + z, q] = ((beta[j] * lifter.w3[z] % P) * FV[j, r, q]) % P
    ft = np.array([_dot(lifter.f, lifter.t2, P), _dot(lifter.g, lifter.t2, P)], dtype=np.int64)
    FA[0, :, 0] = (FA[0, :, 0] + ft[0]) % P; FA[0, :, 1] = (FA[0, :, 1] + ft[1]) % P
    lo, hi = (0, nopt) if o0range is None else o0range
    nsol, sols = _brute_k5_family(FA, FB, P, maxsol, lo, hi)
    assert nsol <= maxsol, ('family brute force overflow', nsol)
    Vfw = [lifter.tab[fw[j]] for j in range(k)]; Vbw = [lifter.tab[bw[j]] for j in range(k)]
    out = []; nscreen = int(nsol)
    for s in sols[:nsol]:
        A = lifter.t2.copy(); B = np.zeros(lifter.D, dtype=np.int64)
        for j in range(k):
            r, z = divmod(int(s[j]), 3)
            A = (A - b0[j] * lifter.w3[z] % P * Vfw[j][r]) % P
            B = (B + beta[j] * lifter.w3[z] % P * Vfw[j][r]) % P
        if not B.any():
            if A.any(): continue
            lams = None          # E2 holds for every lambda
        else:
            jB = int(np.flatnonzero(B)[0]); lam = A[jB] * pow(int(B[jB]), P - 2, P) % P
            if not np.array_equal(A, lam * B % P): continue
            lams = [lam]
        for zs in itertools.product(range(3), repeat=k):
            A0 = (-lifter.t0) % P; B0 = np.zeros(lifter.D, dtype=np.int64)
            for j in range(k):
                r = int(s[j]) // 3
                A0 = (A0 + b0[j] * lifter.w3[zs[j]] % P * Vbw[j][r]) % P
                B0 = (B0 + beta[j] * lifter.w3[zs[j]] % P * Vbw[j][r]) % P
            # need A0 + lambda B0 = 0  (E0: sum (b0 + lambda beta) zeta' P^-1 w = t0)
            if lams is None:
                if not B0.any():
                    if A0.any(): continue
                    cand = ['free']
                else:
                    jB = int(np.flatnonzero(B0)[0]); lam = (-A0[jB]) * pow(int(B0[jB]), P - 2, P) % P
                    if not np.array_equal((A0 + lam * B0) % P, np.zeros(lifter.D, dtype=np.int64)): continue
                    cand = [lam]
            else:
                lam = lams[0]
                if ((A0 + lam * B0) % P).any(): continue
                cand = [lam]
            for lam in cand:
                coef = None if lam == 'free' else [(b0[j] + lam * beta[j]) % P for j in range(k)]
                if coef is not None and any(c == 0 for c in coef): continue      # a zero coefficient: fewer than k terms
                terms = []
                for j in range(k):
                    r, z = divmod(int(s[j]), 3)
                    c0 = bw[j][r].copy(); nz0 = c0 != 0; c0 = np.where(nz0, ((c0 - 1 + zs[j]) % 3) + 1, 0).astype(np.int8)
                    c2 = fw[j][r].copy(); nz2 = c2 != 0; c2 = np.where(nz2, ((c2 - 1 + z) % 3) + 1, 0).astype(np.int8)
                    terms.append(L.canon(L.assemble_last(c0, base_codes[j], c2).reshape(1, -1))[0])
                out.append(dict(terms=np.array(terms), options=[int(x) for x in s], z0=list(zs), lam=lam, coef=coef))
    if log: print(f'    family brute force: {nscreen} screen survivors, {len(out)} lifts', file=log, flush=True)
    return out
