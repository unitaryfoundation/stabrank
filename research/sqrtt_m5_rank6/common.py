"""Shared pieces of the rank-6 census of sqrt(T)^{(x) 5} through the Galois-forced subspace.

Notation. zeta = exp(i pi/8); t_k = (1, zeta^k)^{(x) m}; V4 = span(t_1, t_5, t_9, t_13), four-dimensional for m >= 3;
the target is t_1 in V4. A stabilizer state is an int8 code of length 2^m: 0 off the support, 1 + e on it for the
amplitude i^e, with e = 0 at the smallest support index (the verifier's parametrization names each state once).
All arithmetic is modulo primes P == 1 (mod 16) with zeta -> g, a primitive sixteenth root of unity mod P, and
i -> g^4, a ring homomorphism on Z[zeta]; exact decisions are ranks over Q(zeta_16) with sympy's DomainMatrix.
"""
import itertools, numpy as np

PRIMES = (2013265921, 1811939329)
PRIM_ROOT = {2013265921: 31, 1811939329: 13}
CONJ = (1, 5, 9, 13)
PIVOTS = (0, 1, 3, 7)


def count_states(n):
    t = 1 << n
    for j in range(1, n + 1):
        t *= (1 << j) + 1
    return t


def rref_matrices(n, k):
    for pivots in itertools.combinations(range(n), k):
        non_pivots = [j for j in range(n) if j not in pivots]
        free = [(r, c) for r in range(k) for c in non_pivots if c > pivots[r]]
        for vals in itertools.product(range(2), repeat=len(free)):
            W = np.zeros((k, n), dtype=np.int64)
            for r in range(k):
                W[r, pivots[r]] = 1
            for (r, c), v in zip(free, vals):
                W[r, c] = v
            yield W, non_pivots


def stabilizer_codes(n):
    """All n-qubit stabilizer states, each named once, as an (N, 2^n) int8 array (qubit 0 most significant)."""
    weights = 1 << np.arange(n - 1, -1, -1)
    blocks = []
    for k in range(n + 1):
        Y = np.array(list(itertools.product(range(2), repeat=k)), dtype=np.int64).reshape(1 << k, k)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        L = np.array(list(itertools.product(range(4), repeat=k)), dtype=np.int64).reshape(4 ** k, k)
        Qc = np.array(list(itertools.product(range(2), repeat=len(pairs))), dtype=np.int64).reshape(1 << len(pairs), len(pairs))
        YY = np.array([[y[i] * y[j] for (i, j) in pairs] for y in Y], dtype=np.int64).reshape(1 << k, len(pairs))
        lin = (L @ Y.T) % 4
        quad = (2 * (Qc @ YY.T)) % 4
        phase = ((quad[:, None, :] + lin[None, :, :]) % 4).reshape(-1, 1 << k).astype(np.int8) + 1
        for W, non_pivots in rref_matrices(n, k):
            for free in itertools.product(range(2), repeat=n - k):
                x0 = np.zeros(n, dtype=np.int64)
                x0[non_pivots] = free
                pos = ((x0[None, :] + Y @ W) % 2) @ weights
                codes = np.zeros((phase.shape[0], 1 << n), dtype=np.int8)
                codes[:, pos] = phase
                blocks.append(codes)
    out = np.concatenate(blocks)
    assert len(out) == count_states(n)
    return out


def hash_rows(codes, seed=11):
    """Two 61-bit linear hashes of each row (N, 2); rows are int8 codes."""
    rng = np.random.default_rng(seed)
    M = (1 << 61) - 1
    out = np.zeros((len(codes), 2), dtype=np.int64)
    for h in range(2):
        c = rng.integers(1, M, size=codes.shape[1], dtype=np.int64)
        acc = np.zeros(len(codes), dtype=np.int64)
        for j in range(codes.shape[1]):
            acc = (acc + (codes[:, j].astype(np.int64) * c[j]) % M) % M
        out[:, h] = acc
    return out


def weights(m):
    return np.array([bin(x).count("1") for x in range(1 << m)], dtype=np.int64)


def root16(P):
    g = pow(PRIM_ROOT[P], (P - 1) // 16, P)
    assert pow(g, 8, P) == P - 1 and pow(g, 16, P) == 1
    return g


def modinv_arr(a, P):
    res = np.ones_like(a)
    base = a % P
    e = P - 2
    while e:
        if e & 1:
            res = (res * base) % P
        base = (base * base) % P
        e >>= 1
    return res


def matinv_mod(A, P):
    n = len(A)
    M = [[int(A[i][j]) % P for j in range(n)] + [int(i == j) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c])
        M[c], M[piv] = M[piv], M[c]
        inv = pow(M[c][c], P - 2, P)
        M[c] = [(x * inv) % P for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % P for x, y in zip(M[r], M[c])]
    return np.array([row[n:] for row in M], dtype=np.int64)


def rank_mod(rows, P):
    M = [[int(x) % P for x in r] for r in rows]
    rk = 0
    for c in range(len(M[0]) if M else 0):
        piv = next((r for r in range(rk, len(M)) if M[r][c]), None)
        if piv is None:
            continue
        M[rk], M[piv] = M[piv], M[rk]
        inv = pow(M[rk][c], P - 2, P)
        M[rk] = [(x * inv) % P for x in M[rk]]
        for r in range(len(M)):
            if r != rk and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % P for x, y in zip(M[r], M[rk])]
        rk += 1
        if rk == len(M):
            break
    return rk


class Field:
    """One prime: the value table of the codes, the V4 basis, the residue map, and the key functionals."""

    def __init__(self, m, P, seed):
        self.m, self.P = m, P
        g = root16(P)
        self.g = g
        self.tab = np.array([0, 1, pow(g, 4, P), P - 1, P - pow(g, 4, P)], dtype=np.int64)
        w = weights(m)
        self.B = np.array([[pow(g, int(k * ww), P) for k in CONJ] for ww in w], dtype=np.int64)
        self.Ainv = matinv_mod(self.B[list(PIVOTS)], P)
        self.f = np.random.default_rng(seed).integers(1, P, size=(2, 1 << m), dtype=np.int64)

    def values(self, codes):
        return self.tab[codes]

    def residues(self, codes, chunk=250000):
        P = self.P
        out = np.empty(codes.shape, dtype=np.int64)
        for s in range(0, len(codes), chunk):
            V = self.tab[codes[s:s + chunk]]
            C = np.zeros((len(V), 4), dtype=np.int64)
            for k in range(4):
                C = (C + (V[:, PIVOTS[k]][:, None] * self.Ainv[:, k][None, :]) % P) % P
            R = V.copy()
            for k in range(4):
                R = (R - (C[:, k][:, None] * self.B[:, k][None, :]) % P) % P
            out[s:s + chunk] = R
        assert not out[:, list(PIVOTS)].any()
        return out

    def keys_of(self, R):
        """projective keys (N, 3) of the rows of R: first nonzero index (-1 for zero rows), two functionals / pivot entry."""
        P = self.P
        nz = R != 0
        zero = ~nz.any(axis=1)
        p = np.argmax(nz, axis=1)
        rp = R[np.arange(len(R)), p]
        rp[zero] = 1
        inv = modinv_arr(rp, P)
        k = np.empty((len(R), 3), dtype=np.int64)
        k[:, 0] = np.where(zero, -1, p)
        for j in range(2):
            S = ((R * self.f[j][None, :]) % P).sum(axis=1) % P
            k[:, 1 + j] = np.where(zero, 0, (S * inv) % P)
        return k


def perm_index_maps(m):
    """For every permutation sigma of the m qubits, the index map x -> sigma(x) on {0..2^m-1} (qubit 0 most significant)."""
    maps = []
    for perm in itertools.permutations(range(m)):
        idx = np.zeros(1 << m, dtype=np.int64)
        for x in range(1 << m):
            bits = [(x >> (m - 1 - i)) & 1 for i in range(m)]
            y = 0
            for i in range(m):
                y = 2 * y + bits[perm[i]]
            idx[x] = y
        maps.append(idx)
    return maps


def canon_codes(codes):
    """phase 0 at the smallest support index."""
    nz = codes != 0
    first = np.argmax(nz, axis=1)
    ph0 = codes[np.arange(len(codes)), first] - 1
    return np.where(nz, ((codes - 1 - ph0[:, None]) % 4) + 1, 0).astype(np.int8)


def group_images(codes, m, maps):
    """Generator over the group G = S_m x <S^{(x) m}> (order m! * 4) of the image code arrays of `codes`:
    yields (label, codes_image) with codes_image canonical. S^{(x) m} multiplies the amplitude at x by i^{|x|}."""
    w = weights(m)
    for pi, idx in enumerate(maps):
        base = codes[:, idx]
        for s in range(4):
            img = np.where(base != 0, ((base - 1 + s * w[None, :]) % 4) + 1, 0).astype(np.int8)
            yield (pi, s), canon_codes(img)


def group_image_single(code, m, idx, s):
    w = weights(m)
    base = code[idx]
    img = np.where(base != 0, ((base - 1 + s * w) % 4) + 1, 0).astype(np.int8)
    return canon_codes(img[None, :])[0]


_K = None


def field_K():
    global _K
    if _K is None:
        import sympy as sp
        _K = sp.QQ.algebraic_field(sp.exp(sp.I * sp.pi / 8))
    return _K


def exact_rank(codes, m, extra=None):
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix
    K = field_K()
    xz = K.from_sympy(sp.exp(sp.I * sp.pi / 8))
    tab = [K.zero, K.one, xz ** 4, -K.one, -(xz ** 4)]
    rows = [[tab[int(c)] for c in row] for row in codes]
    if extra is not None:
        rows += [[K.from_sympy(e) for e in row] for row in extra]
    return DomainMatrix(rows, (len(rows), 1 << m), K).rank()


def exact_coefficients(codes, m):
    """exact coefficients c (sympy) with sum c_j s_j = t_1, or None if t_1 is not in the span or the rows are dependent."""
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix
    K = field_K()
    z = sp.exp(sp.I * sp.pi / 8)
    xz = K.from_sympy(z)
    tab = [K.zero, K.one, xz ** 4, -K.one, -(xz ** 4)]
    cols = [[tab[int(c)] for c in row] for row in codes]           # terms as rows
    A = DomainMatrix(cols, (len(cols), 1 << m), K).transpose()      # (2^m, r)
    b = DomainMatrix([[xz ** int(ww)] for ww in weights(m)], (1 << m, 1), K)
    _, piv = A.transpose().rref()
    if len(piv) != len(cols):
        return None
    sq = DomainMatrix([A.to_Matrix().row(i).tolist()[0] for i in piv], (len(piv), len(cols)), K) if False else None
    rows = A.rep.to_list() if hasattr(A.rep, 'to_list') else A.to_list()
    sqA = DomainMatrix([rows[i] for i in piv], (len(piv), len(cols)), K)
    brows = b.rep.to_list() if hasattr(b.rep, 'to_list') else b.to_list()
    sqb = DomainMatrix([brows[i] for i in piv], (len(piv), 1), K)
    x = sqA.lu_solve(sqb)
    if (A * x - b).to_Matrix() != sp.zeros(1 << m, 1):
        return None
    return [K.to_sympy(x[j, 0].element) for j in range(len(cols))]
