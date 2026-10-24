"""Exact machinery for qutrit stabilizer states (round 33, agent B): codes, modular evaluation, Paulis, slices, the
symmetry group of S^{(x) n} for the Strange state S = (|1> - |2>)/sqrt 2, and exact linear algebra over Z[omega].

Conventions. A state on n qutrits is indexed by x in F_3^n with index sum_i x_i 3^(n-1-i) (qutrit 0 most
significant, the last qutrit least significant). An unnormalized stabilizer state has amplitudes omega^(e_x) on an
affine support and 0 elsewhere; its code is the int8 vector with 0 outside the support and 1 + e_x inside,
canonicalized so that the first support index carries phase 0. Modular arithmetic maps omega to a primitive cube
root of unity modulo a prime P == 1 (mod 3); this is a ring homomorphism Z[omega] -> F_P.
"""
import numpy as np, itertools

PRIMES = (2013265921, 1811939329)
PRIM_ROOT = {2013265921: 31, 1811939329: 13}
OMEGA = np.exp(2j * np.pi / 3)

def cube_root(P):
    w = pow(PRIM_ROOT[P], (P - 1) // 3, P)
    assert w != 1 and pow(w, 3, P) == 1
    return w

def modtab(P):
    """code value -> residue: 0 -> 0, 1 + e -> omega^e."""
    w = cube_root(P)
    return np.array([0, 1, w, (w * w) % P], dtype=np.int64)

CPLX_TAB = np.array([0, 1, OMEGA, OMEGA ** 2], dtype=np.complex128)

def rref_subspaces(n, k):
    for piv in itertools.combinations(range(n), k):
        free = [c for c in range(n) if c not in piv]
        slots = [(i, c) for i in range(k) for c in free if c > piv[i]]
        for vals in itertools.product(range(3), repeat=len(slots)):
            W = np.zeros((k, n), dtype=np.int64)
            for i in range(k): W[i, piv[i]] = 1
            for (i, c), v in zip(slots, vals): W[i, c] = v
            yield W, free

def canon(codes):
    nz = codes != 0
    first = np.argmax(nz, axis=1)
    ph0 = codes[np.arange(len(codes)), first] - 1
    return np.where(nz, ((codes - 1 - ph0[:, None]) % 3) + 1, 0).astype(np.int8)

def letters(n):
    """all n-qutrit stabilizer states as canonical codes (N, 3^n); N = 3^n prod_{j<=n} (3^j + 1)."""
    out = []
    weights = 3 ** np.arange(n - 1, -1, -1)
    for k in range(n + 1):
        Y = np.array(list(itertools.product(range(3), repeat=k)), dtype=np.int64).reshape(3 ** k, k)
        ut = [(i, j) for i in range(k) for j in range(i, k)]
        Qs = np.array(list(itertools.product(range(3), repeat=len(ut))), dtype=np.int64).reshape(3 ** len(ut), len(ut))
        Ls = np.array(list(itertools.product(range(3), repeat=k)), dtype=np.int64).reshape(3 ** k, k)
        QY = np.zeros((len(Qs), 3 ** k), dtype=np.int64)
        for t, (i, j) in enumerate(ut): QY += np.outer(Qs[:, t], Y[:, i] * Y[:, j])
        LY = Ls @ Y.T
        QY8 = (QY % 3).astype(np.int8); LY8 = (LY % 3).astype(np.int8)          # int8 broadcast: 387 MB at k = 4 instead of 3 GB
        ph = ((QY8[:, None, :] + LY8[None, :, :]) % 3).reshape(-1, 3 ** k).astype(np.int8) + 1
        for W, free in rref_subspaces(n, k):
            for x0bits in itertools.product(range(3), repeat=len(free)):
                x0 = np.zeros(n, dtype=np.int64); x0[free] = x0bits
                idx = ((x0[None, :] + Y @ W) % 3) @ weights
                c = np.zeros((len(ph), 3 ** n), dtype=np.int8); c[:, idx] = ph
                out.append(canon(c))
    codes = np.concatenate(out); del out
    expect = 3 ** n * int(np.prod([3 ** j + 1 for j in range(1, n + 1)]))
    assert len(codes) == expect, (len(codes), expect)
    return codes

def dense(codes):
    return CPLX_TAB[codes]

def target(n):
    """unnormalized S^{(x) n}: amplitude prod_i (0, 1, -1)[x_i], as an int64 vector in {0, 1, -1}."""
    a = np.array([0, 1, -1], dtype=np.int64); v = np.array([1], dtype=np.int64)
    for _ in range(n): v = np.kron(v, a)
    return v

def code_hash(codes, seed=12345):
    """64-bit hashes of code rows (random linear hash modulo 2^61 - 1 on the int8 entries)."""
    rng = np.random.default_rng(seed); M = (1 << 61) - 1
    h = np.zeros(len(codes), dtype=np.int64)
    coef = rng.integers(1, M, size=codes.shape[1], dtype=np.int64)
    for j in range(codes.shape[1]):
        h = (h + (codes[:, j].astype(np.int64) * coef[j]) % M) % M
    return h

class Lookup:
    """index of a canonical code among the letters (by sorted hash; collisions resolved by comparison)."""
    def __init__(self, codes):
        self.codes = codes; self.h = code_hash(codes); self.order = np.argsort(self.h, kind='stable'); self.hs = self.h[self.order]
    def find(self, code):
        code = canon(code.reshape(1, -1))[0]
        hh = int(code_hash(code.reshape(1, -1))[0])
        lo = np.searchsorted(self.hs, hh, 'left'); hi = np.searchsorted(self.hs, hh, 'right')
        for k in range(lo, hi):
            i = int(self.order[k])
            if np.array_equal(self.codes[i], code): return i
        return -1
    def find_many(self, codes):
        codes = canon(codes); hh = code_hash(codes)
        pos = np.searchsorted(self.hs, hh, 'left'); out = np.full(len(codes), -1, dtype=np.int64)
        for r in range(len(codes)):
            k = pos[r]
            while k < len(self.hs) and self.hs[k] == hh[r]:
                i = int(self.order[k])
                if np.array_equal(self.codes[i], codes[r]): out[r] = i; break
                k += 1
        return out

# ---------- Paulis and slices on codes ----------

def pauli_maps(n):
    """for every a in F_3^n the source index array of X^a ((X^a v)(x) = v(x - a)) and for every b the phase vector
    b.x (so that (X^a Z^b v)(x) = omega^(b.(x-a)) v(x-a) = omega^(b.x') v(x') with x' = x - a)."""
    X = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int64); weights = 3 ** np.arange(n - 1, -1, -1)
    src = {}; ph = {}
    for a in itertools.product(range(3), repeat=n):
        src[a] = (((X - np.array(a)) % 3) @ weights)
    for b in itertools.product(range(3), repeat=n):
        ph[b] = ((X @ np.array(b)) % 3)
    return X, src, ph

def pauli_code(code, a, b, src, ph):
    """code of X^a Z^b applied to the state with the given code (not canonicalized): entry x gets code[x-a] with phase
    + b.(x-a)."""
    s = src[a]; c = code[s]; nz = c != 0
    out = np.where(nz, ((c - 1 + ph[b][s]) % 3) + 1, 0).astype(np.int8)
    return out

def coset_rays(code, n, src, ph):
    """the 3^n distinct rays P t over the Pauli group: list of (code of P t canonical, code of P^-1 t canonical), one per
    coset of the stabilizer group; P^-1 is X^(-a) Z^(-b) up to a phase."""
    seen = {}; rays = []
    for a in itertools.product(range(3), repeat=n):
        na = tuple((-x) % 3 for x in a)
        for b in itertools.product(range(3), repeat=n):
            c1 = canon(pauli_code(code, a, b, src, ph).reshape(1, -1))[0]; k = c1.tobytes()
            if k in seen: continue
            nb = tuple((-x) % 3 for x in b)
            c2 = canon(pauli_code(code, na, nb, src, ph).reshape(1, -1))[0]
            seen[k] = len(rays); rays.append((c1, c2))
    assert len(rays) == 3 ** n, len(rays)
    return rays

def slice_last(v, a):
    """slice of a vector (code or values) on the last qutrit at value a: indices 3 y + a."""
    return v[a::3]

def assemble_last(v0, v1, v2):
    """vector on n qutrits from its three slices on the last qutrit."""
    out = np.empty(3 * len(v1), dtype=v1.dtype); out[0::3] = v0; out[1::3] = v1; out[2::3] = v2; return out

# ---------- symmetry group of S^{(x) n}: L^n x| S_n, L = <F, Pgate> mod Paulis ----------

F1 = np.array([[OMEGA ** (x * y) for y in range(3)] for x in range(3)]) / np.sqrt(3)
PG1 = np.diag([1, OMEGA, OMEGA])

def local_op(U, i, n):
    out = np.array([[1.0 + 0j]])
    for q in range(n): out = np.kron(out, U if q == i else np.eye(3))
    return out

def perm_index(pi, n):
    X = np.array(list(itertools.product(range(3), repeat=n))); weights = 3 ** np.arange(n - 1, -1, -1)
    Xp = np.empty_like(X); Xp[:, list(pi)] = X
    return Xp @ weights

def generators(n):
    gens = []
    for i in range(n): gens.append(('F%d' % i, local_op(F1, i, n))); gens.append(('P%d' % i, local_op(PG1, i, n)))
    if n >= 2:
        pi = list(range(n)); pi[0], pi[1] = 1, 0; gens.append(('t01', perm_index(pi, n)))
        gens.append(('cyc', perm_index([(q + 1) % n for q in range(n)], n)))
    return gens

def apply_gen(g, V):
    return V @ g.T if g.ndim == 2 else V[:, g]

def vec_to_code(v, tol=1e-7):
    """code of a dense vector proportional to a stabilizer state (amplitudes a common modulus times cube roots);
    raises if the vector is not of that form."""
    v = np.asarray(v); m = np.abs(v); mx = m.max(); nz = m > tol * mx
    assert np.allclose(m[nz], mx, atol=tol * mx), 'not a flat-modulus vector'
    j = np.flatnonzero(nz)[0]; u = v / v[j]
    e = np.round(np.angle(u[nz]) / (2 * np.pi / 3)).astype(np.int64) % 3
    assert np.allclose(u[nz], OMEGA ** e, atol=1e-6), 'phases are not cube roots'
    code = np.zeros(len(v), dtype=np.int8); code[nz] = e + 1
    return code

def letter_maps(codes, gens, lookup):
    """permutations of the letter set induced by the generators (dense application, exact re-identification)."""
    V = dense(codes); maps = []
    for name, g in gens:
        img = apply_gen(g, V)
        imgc = np.array([vec_to_code(row) for row in img]); m = lookup.find_many(imgc)
        assert (m >= 0).all(), name
        maps.append(m)
    return maps

def orbits_uf(N, maps):
    parent = np.arange(N)
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for m in maps:
        for x in range(N):
            ra, rb = find(x), find(int(m[x]))
            if ra != rb: parent[ra] = rb
    return np.array([find(x) for x in range(N)])

# ---------- exact arithmetic in Z[omega] ----------
# elements a + b omega as pairs of python ints; omega^2 = -1 - omega

def zmul(x, y):
    a, b = x; c, d = y
    return (a * c - b * d, a * d + b * c - b * d)

def zadd(x, y): return (x[0] + y[0], x[1] + y[1])
def zsub(x, y): return (x[0] - y[0], x[1] - y[1])
def zneg(x): return (-x[0], -x[1])
def zconj(x):
    a, b = x; return (a - b, -b)          # conjugate of a + b omega is a + b omega^2 = (a - b) - b omega
def znorm(x):
    a, b = x; return a * a - a * b + b * b
def ziszero(x): return x[0] == 0 and x[1] == 0
def zdiv_exact(x, y):
    """x / y assuming y divides x in Z[omega]."""
    n = znorm(y); num = zmul(x, zconj(y))
    assert num[0] % n == 0 and num[1] % n == 0, 'inexact division'
    return (num[0] // n, num[1] // n)

def code_to_z(code):
    tab = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (-1, -1)}
    return [tab[int(c)] for c in code]

def int_to_z(v): return [(int(x), 0) for x in v]

def bareiss_rank(rows):
    """rank over Q(omega) of a list of Z[omega] rows (fraction-free elimination)."""
    M = [list(r) for r in rows]; m = len(M); ncol = len(M[0]) if m else 0
    rank = 0; prev = (1, 0); r = 0
    for c in range(ncol):
        if r >= m: break
        piv = next((i for i in range(r, m) if not ziszero(M[i][c])), None)
        if piv is None: continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(r + 1, m):
            for j in range(c + 1, ncol):
                M[i][j] = zdiv_exact(zsub(zmul(M[r][c], M[i][j]), zmul(M[i][c], M[r][j])), prev)
            M[i][c] = (0, 0)
        prev = M[r][c]; r += 1; rank += 1
    return rank

def exact_cover_check(letter_codes, tvec):
    """exact: rank of the letters, rank with the target appended, and (if the letters are independent and the target
    is in their span) the coefficients as (numerator, denominator) pairs in Z[omega] and whether all are nonzero."""
    L = [code_to_z(c) for c in letter_codes]; t = int_to_z(tvec)
    r1 = bareiss_rank(L); r2 = bareiss_rank(L + [t])
    return r1, r2
