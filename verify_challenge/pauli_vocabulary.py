"""Exact projection-lemma rank exclusion for qubit vectors with entries in Z[i].

Integer arithmetic only: vectors are pairs of int64 arrays (real, imaginary
part), and every rank decision is made modulo a split prime P = 1 (mod 4),
where i is a square root of -1 in F_P. No floating-point number is created.

Projection lemma. If psi = sum_{j<=r} c_j s_j with stabilizer states s_j and
g is a signed Pauli in the stabilizer group of s_1, then (1 - g) psi =
sum_{j>=2} c_j (1 - g) s_j, and (1 - g) maps stabilizer states to stabilizer
states or to 0, so chi((1 - g) psi) <= r - 1. Call g cheap for rank r if that
inequality is not excluded. Every non-identity element of the signed group of
every term then lies in the cheap set, so every term lies in the vocabulary:
the stabilizer states whose whole signed group is cheap. If two terms share g,
(1 - g) kills both and chi((1 - g) psi) <= r - 2; if three share g, <= r - 3.

The ranks chi((1 - g) psi) are bounded below by the same procedure one qubit
down. (1 - g) psi lies in the -1 eigenspace of g, which `restrict` identifies
with C^(2^(n-1)) by a pair of mutually inverse stabilizer maps (contraction
with <0| on a qubit where g acts as X or Y, or a CNOT ladder and contraction
for a diagonal g), so the rank is preserved and stabilizer states correspond
to stabilizer states.

Search. If psi lies in the span of r vocabulary states, fix r - 2 of them as
pivots and pass to the quotient by span(psi, pivots): the images of the other
two are linearly dependent. Images are normalised projectively mod P and
sorted, so every dependent pair is found; the converse is checked by exact
ranks. See `exclude` for why every rank used is exact.
"""

from __future__ import annotations

import itertools

import numpy as np

# ----------------------------------------------------------------- primes


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, valid for n < 3.3 * 10^24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for q in small:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def split_prime_below(bound: int) -> int:
    """Largest prime p < bound with p = 1 (mod 4)."""
    p = bound - 1
    while not (p % 4 == 1 and _is_prime(p)):
        p -= 1
    return p


def sqrt_minus_one(p: int) -> int:
    for c in range(2, p):
        t = pow(c, (p - 1) // 4, p)
        if t * t % p == p - 1:
            return t
    raise ValueError(p)


# Products of two residues stay below 2^62, so int64 never overflows.
P1 = split_prime_below(1 << 31)
P2 = split_prime_below(P1)
I1 = sqrt_minus_one(P1)
I2 = sqrt_minus_one(P2)

# ----------------------------------------------------------------- tables

POP = np.array([bin(x).count("1") for x in range(1 << 12)], dtype=np.int64)
PAR = POP & 1


def decode(idx, n):
    """Signed Pauli index -> (a, b, s): X-part a, Z-part b, sign (-1)^s.

    The index is 2u + s with u = (a << n) | b, and the operator is
    (-1)^s i^{|a & b|} X^a Z^b, which is Hermitian. Indices 0 and 1 are +I, -I."""
    idx = np.asarray(idx, dtype=np.int64)
    return idx >> (n + 1), (idx >> 1) & ((1 << n) - 1), idx & 1


def encode(a, b, s, n):
    return (((np.asarray(a, dtype=np.int64) << n) | b) << 1) | s


class Tables:
    """Lookup tables for n qubits: products, commutation, action and restriction."""

    def __init__(self, n):
        self.n, self.N, self.U = n, 1 << n, 1 << (2 * n)
        self.M = 2 * self.U
        u = np.arange(self.U, dtype=np.int64)
        a, b = u >> n, u & (self.N - 1)
        a1, b1, a2, b2 = a[:, None], b[:, None], a[None, :], b[None, :]
        self.COMM = (PAR[a1 & b2] ^ PAR[a2 & b1]) == 0
        e = (POP[a1 & b1] + POP[a2 & b2] - POP[(a1 ^ a2) & (b1 ^ b2)]
             + 2 * PAR[b1 & a2]) % 4
        self.PH = (e >> 1).astype(np.uint8)    # sign bit of a commuting product
        y = np.arange(self.N, dtype=np.int64)
        self.SRC = y[None, :] ^ a[:, None]
        self.EPH = (POP[a & b][:, None] + 2 * PAR[b[:, None] & self.SRC]) % 4
        self.KEEP = np.zeros((self.M, max(self.N // 2, 1)), dtype=np.int64)
        if n >= 1:
            for g in range(2, self.M):
                self.KEEP[g] = restrict_index(g, n)

    def pmul(self, i1, i2):
        """Signed index of the product of commuting signed Paulis (vectorised)."""
        i1 = np.asarray(i1, dtype=np.int64)
        i2 = np.asarray(i2, dtype=np.int64)
        u1, u2 = i1 >> 1, i2 >> 1
        return ((u1 ^ u2) << 1) | (((i1 ^ i2) & 1) ^ self.PH[u1, u2].astype(np.int64))

    def commute(self, i1, i2):
        return self.COMM[np.asarray(i1) >> 1, np.asarray(i2) >> 1]


_TABLES = {}


def tables(n):
    if n not in _TABLES:
        _TABLES[n] = Tables(n)
    return _TABLES[n]


def pmul(i1, i2, n):
    return tables(n).pmul(i1, i2)


def commute(i1, i2, n):
    return tables(n).commute(i1, i2)


# ----------------------------------------------------------------- Z[i] vectors


def times_i_pow(re, im, e):
    """Multiply entrywise by i^e (e an int array mod 4)."""
    e = np.asarray(e) % 4
    out_re = np.where(e == 0, re, np.where(e == 1, -im, np.where(e == 2, -re, im)))
    out_im = np.where(e == 0, im, np.where(e == 1, re, np.where(e == 2, -im, -re)))
    return out_re, out_im


def apply_pauli(re, im, g, n):
    """(g v)(y) = (-1)^s i^{|a&b|} (-1)^{b.(y^a)} v(y^a), on the last axis."""
    T = tables(n)
    u, s = int(g) >> 1, int(g) & 1
    return times_i_pow(re[..., T.SRC[u]], im[..., T.SRC[u]], T.EPH[u] + 2 * s)


def restrict_index(g, n):
    """Coordinates identifying the -1 eigenspace of g with n-1 qubits.

    For x' in [0, 2^(n-1)) returns the n-qubit index x it is read from: the
    entries with x_j = 0 when g acts as X or Y on qubit j (lowest such j), and
    for g = (-1)^s Z^b the entries with x_j = 1 + s + b'.x (mod 2), j the
    lowest bit of b and b' = b without it."""
    a, b, s = (int(t) for t in decode(g, n))
    xp = np.arange(1 << (n - 1), dtype=np.int64)
    if a:
        j = (a & -a).bit_length() - 1
        low = xp & ((1 << j) - 1)
        return ((xp >> j) << (j + 1)) | low
    j = (b & -b).bit_length() - 1
    bp = b & ~(1 << j)
    low = xp & ((1 << j) - 1)
    x0 = ((xp >> j) << (j + 1)) | low
    bit = (1 + s + PAR[bp & x0]) & 1          # s + b.x odd: eigenvalue -1
    return x0 | (bit << j)


def kill_project(re, im, g, n):
    """The (n-1)-qubit vector of (1 - g) v: the projection killing g-stabilised states."""
    gre, gim = apply_pauli(re, im, g, n)
    keep = tables(n).KEEP[int(g)]
    return (re - gre)[..., keep], (im - gim)[..., keep]


def kill_project_all(re, im, n):
    """kill_project for every signed Pauli g at once: arrays (2 * 4^n, 2^(n-1));
    rows 0 and 1 (+-I) are zero."""
    T = tables(n)
    bre, bim = times_i_pow(re[T.SRC], im[T.SRC], T.EPH)
    gre = np.empty((T.M, T.N), dtype=np.int64)
    gim = np.empty((T.M, T.N), dtype=np.int64)
    gre[0::2], gim[0::2] = bre, bim
    gre[1::2], gim[1::2] = -bre, -bim
    dre = np.take_along_axis(re[None, :] - gre, T.KEEP, axis=1)
    dim = np.take_along_axis(im[None, :] - gim, T.KEEP, axis=1)
    dre[:2] = 0
    dim[:2] = 0
    return dre, dim


def canon_rows_exact(re, im):
    """Projective normal form of each row: multiply by the conjugate of the first
    nonzero entry and divide by the gcd of all integer parts. Rows that are
    Q(i)-multiples of each other get identical output; zero rows stay zero."""
    re = np.atleast_2d(re)
    im = np.atleast_2d(im)
    nzm = (re != 0) | (im != 0)
    has = nzm.any(axis=1)
    f = nzm.argmax(axis=1)
    rows = np.arange(re.shape[0])
    a, b = re[rows, f][:, None], im[rows, f][:, None]
    wr, wi = re * a + im * b, im * a - re * b
    g = np.gcd.reduce(np.abs(np.concatenate([wr, wi], axis=1)), axis=1)
    g[g == 0] = 1
    return np.concatenate([wr // g[:, None], wi // g[:, None]], axis=1), has


def zkey(re, im):
    """Exact projective key: v and lambda v (lambda in Q(i)^*) get the same bytes."""
    c, has = canon_rows_exact(re, im)
    return c[0].tobytes() if has[0] else b"zero"


def is_zero(re, im):
    return not (np.any(re) or np.any(im))


def norm2(re, im):
    return int(np.sum(re.astype(object) ** 2 + im.astype(object) ** 2))


# ----------------------------------------------------------------- stabilizer states


def is_stabilizer(re, im, n):
    """Exact test: v = c * sum_{x in A} i^{l(x)} (-1)^{q(x)} |x> with A affine,
    l linear and q quadratic over F_2 (Dehaene and De Moor)."""
    nz = np.flatnonzero((re != 0) | (im != 0))
    k = nz.size.bit_length() - 1
    if nz.size == 0 or nz.size != 1 << k:
        return False
    x0 = int(nz[0])
    d = nz ^ x0
    basis, piv = [], []                        # F_2 row reduction of the support
    for v in d:
        v = int(v)
        for bv, pb in zip(basis, piv):
            if (v >> pb) & 1:
                v ^= bv
        if v:
            pb = v.bit_length() - 1
            for t in range(len(basis)):
                if (basis[t] >> pb) & 1:
                    basis[t] ^= v
            basis.append(v)
            piv.append(pb)
    if len(basis) != k:
        return False
    a0, b0 = int(re[x0]), int(im[x0])
    m0 = a0 * a0 + b0 * b0
    e = np.zeros(1 << k, dtype=np.int64)
    for y in range(1 << k):
        x = x0
        for t in range(k):
            if (y >> t) & 1:
                x ^= basis[t]
        xr, xi = int(re[x]), int(im[x])
        pr, pi = xr * a0 + xi * b0, xi * a0 - xr * b0   # v(x) conj(v(x0))
        if (pr, pi) == (m0, 0):
            e[y] = 0
        elif (pr, pi) == (0, m0):
            e[y] = 1
        elif (pr, pi) == (-m0, 0):
            e[y] = 2
        elif (pr, pi) == (0, -m0):
            e[y] = 3
        else:
            return False
    lin = e & 1
    for y in range(1 << k):
        if lin[y] != (sum(lin[1 << t] for t in range(k) if (y >> t) & 1) & 1):
            return False
    q = ((e - lin) // 2) & 1
    anf = q.copy()                             # Moebius transform
    for t in range(k):
        step = 1 << t
        for y in range(1 << k):
            if y & step:
                anf[y] ^= anf[y ^ step]
    return not any(anf[y] and POP[y] >= 3 for y in range(1 << k))


def state_of_group(gens, n):
    """Exact state (first nonzero entry 1) fixed by commuting signed Paulis `gens`."""
    N = 1 << n
    re = np.eye(N, dtype=np.int64)
    im = np.zeros((N, N), dtype=np.int64)
    for g in gens:                             # rows <- (1 + g) rows
        gre, gim = apply_pauli(re, im, int(g), n)
        re, im = re + gre, im + gim
    row = int(np.argmax(np.any((re != 0) | (im != 0), axis=1)))
    r, i = re[row], im[row]
    nz = np.flatnonzero((r != 0) | (i != 0))
    a0, b0 = int(r[nz[0]]), int(i[nz[0]])
    m0 = a0 * a0 + b0 * b0
    pr, pi = r * a0 + i * b0, i * a0 - r * b0
    if np.any(pr % m0) or np.any(pi % m0):
        raise AssertionError("group state is not a phase vector")
    return pr // m0, pi // m0


def group_elements(gens, n):
    """All 2^k signed elements generated by commuting independent `gens`."""
    el = np.zeros(1, dtype=np.int64)            # +I has index 0
    for g in gens:
        el = np.concatenate([el, pmul(np.full(el.shape, int(g)), el, n)])
    return el


def groups_in(n, cheap):
    """Generators of every signed stabilizer group on n qubits whose non-identity
    elements all lie in `cheap` (bool over signed indices), each exactly once.

    A group is listed through its greedy basis g_1 < g_2 < ..., with g_{k+1} the
    smallest element outside <g_1, ..., g_k>. Equivalently each new generator is
    at most every element of its coset, and later cosets only contain larger
    elements, which the same test enforces at their own step. Two cheap elements
    can lie in one group only if they commute, are distinct as unsigned Paulis
    and their product is cheap, so candidates are the common neighbours of the
    current elements in that graph."""
    T = tables(n)
    cheap = np.asarray(cheap, dtype=bool).copy()
    cheap[[0, 1]] = False                      # +I, -I
    cand = np.flatnonzero(cheap)
    if cand.size == 0:
        return []
    pos = np.full(cheap.size, -1, dtype=np.int64)
    pos[cand] = np.arange(cand.size)
    cu = cand >> 1
    adj = T.COMM[cu[:, None], cu[None, :]] & (cu[:, None] != cu[None, :])
    prod = T.pmul(cand[:, None], cand[None, :])
    adj &= cheap[prod]
    out = []

    def rec(gens, elems, mask):
        if len(gens) == n:
            out.append(tuple(gens))
            return
        last = gens[-1] if gens else -1
        for t in np.flatnonzero(mask & (cand > last)):
            g = int(cand[t])
            coset = T.pmul(g, elems)
            if np.any(coset < g):
                continue
            new_mask = mask & np.logical_and.reduce(adj[pos[coset]], axis=0)
            rec(gens + [g], np.concatenate([elems, coset]), new_mask)

    rec([], np.zeros(1, dtype=np.int64), np.ones(cand.size, dtype=bool))
    return out


# ----------------------------------------------------------------- arithmetic mod P


def to_modp(re, im, p, ip):
    return (re % p + ip * (im % p) % p) % p


def inv_mod(x, p):
    r = np.ones_like(x)
    b = x % p
    e = p - 2
    while e:
        if e & 1:
            r = r * b % p
        b = b * b % p
        e >>= 1
    return r


def rref(rows, p):
    """Row-reduced echelon form mod p of a small matrix: (R, pivot columns)."""
    M = np.array(rows, dtype=np.int64) % p
    piv, r = [], 0
    for c in range(M.shape[1]):
        if r == M.shape[0]:
            break
        nz = np.flatnonzero(M[r:, c])
        if nz.size == 0:
            continue
        k = r + int(nz[0])
        M[[r, k]] = M[[k, r]]
        M[r] = M[r] * pow(int(M[r, c]), p - 2, p) % p
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        piv.append(c)
        r += 1
    return M[:r], piv


def rank_mod(rows, p):
    return len(rref(rows, p)[1])


def quotient_images(Vm, R, piv, p):
    """Images of the rows of Vm in F_p^N / rowspace(R), as vectors vanishing on the
    pivot columns of the reduced basis R."""
    out = Vm.copy()
    for i, c in enumerate(piv):
        out = (out - Vm[:, c:c + 1] * R[i] % p) % p
    return out


def canon_rows(Phi, p):
    nzm = Phi != 0
    has = nzm.any(axis=1)
    first = nzm.argmax(axis=1)
    lead = Phi[np.arange(Phi.shape[0]), first]
    inv = inv_mod(np.where(has, lead, 1), p)
    return Phi * inv[:, None] % p, has


_HASH = np.random.default_rng(20261119).integers(0, 1 << 20, size=(2, 64))


def parallel_pairs(C, has, p):
    """Index pairs of nonzero rows of C (projectively normalised) that are equal."""
    idx = np.flatnonzero(has)
    if idx.size < 2:
        return []
    N = C.shape[1]
    h1 = (C[idx] @ _HASH[0, :N]) % p          # entries < 2^31, weights < 2^20:
    h2 = (C[idx] @ _HASH[1, :N]) % p          # sums stay below 2^57
    key = h1 * p + h2
    order = np.argsort(key, kind="stable")
    ks = key[order]
    out = []
    start = 0
    for t in range(1, ks.size + 1):
        if t == ks.size or ks[t] != ks[start]:
            if t - start > 1:
                grp = idx[order[start:t]]
                for u, v in itertools.combinations(grp.tolist(), 2):
                    if np.array_equal(C[u], C[v]):
                        out.append((u, v))
            start = t
    return out


# ----------------------------------------------------------------- exclusion


class Found(Exception):
    """A genuine decomposition was found (the bound being proved is false)."""


class Inconsistent(Exception):
    """A step contradicts an exclusion proved earlier: a bug, never a result."""


def pair_compatibility(members, allowed):
    """ok[i, j]: groups i and j share no signed element outside `allowed`."""
    m = members.shape[0]
    bad = members & ~allowed[None, :]
    ok = np.ones((m, m), dtype=bool)
    for i in range(m):
        cols = np.flatnonzero(bad[i])
        if cols.size:
            ok[i] = ~members[:, cols].any(axis=1)
    return ok


def exclude(psi, V, r, allowed_pair=None, allowed_trip=None,
            pivot_list=None, stats=None):
    """Prove that psi is not in the span of r states of V, given that it is not in
    the span of r - 1 of them (proved earlier by the caller).

    Search. Fix r - 2 pivots and map every other state to F_P^N / span(psi,
    pivots). If psi = sum of the pivots and t, t' (with nonzero coefficients),
    the images of t and t' are linearly dependent mod P: a relation over Z[i]
    with coprime coefficients reduces to a nontrivial relation mod P, and its
    t, t' part cannot vanish because psi and the pivots stay independent mod P
    (checked). Images are scaled to lead with 1 and sorted, so every dependent
    pair is listed (zero images included), and each listed pair is decided.

    Exactness. `dependent` decides linear dependence over Q(i) of psi and some
    states. Independence mod P1 or mod P2 is independence over Q(i) (a nonzero
    minor mod P is nonzero). Dependence mod P is dependence over Q(i) when every
    maximal minor is too small to be divisible by the prime: Hadamard's bound
    |det|^2 <= prod of squared column norms, below P (or below P1 P2 for
    dependence modulo both). Without that bound the call fails loudly.

    Pivots. Without `pivot_list`, every (r-2)-subset s_1 < ... < s_{r-2} is a
    pivot set and the remaining two terms are searched among larger indices. A
    caller may pass orbit representatives of pivot tuples under a symmetry of
    psi and V instead; then the other two terms range over all indices.

    Constraints. If `allowed_pair` is given, a candidate decomposition is pruned
    when two of its terms share a signed element outside `allowed_pair` or three
    share one outside `allowed_trip`; the caller certifies those sets."""
    st = stats if stats is not None else {}
    m = V.m
    if m == 0:
        return True
    A = norm2(*psi)
    S = V.S
    if r == 1:
        canon, _ = canon_rows_exact(V.re, V.im)
        mine, _ = canon_rows_exact(*psi)
        if np.any(np.all(canon == mine[0], axis=1)):
            raise Found("psi is a vocabulary state")
        return True
    pm1 = to_modp(*psi, P1, I1)
    pm2 = to_modp(*psi, P2, I2)
    Vm1, Vm2 = V.M1, V.M2

    def dependent(states, with_psi=True):
        rows1 = ([pm1] if with_psi else []) + [Vm1[i] for i in states]
        k = len(rows1)
        if rank_mod(rows1, P1) == k:
            return False
        bound = (A if with_psi else 1) * S ** len(states)
        if bound < P1:
            return True
        rows2 = ([pm2] if with_psi else []) + [Vm2[i] for i in states]
        st["second_prime"] = st.get("second_prime", 0) + 1
        if rank_mod(rows2, P2) == k:
            return False
        if bound < P1 * P2:
            return True
        raise AssertionError("Hadamard bound exceeds P1 * P2")

    use_c = allowed_pair is not None
    if use_c:
        members = V.members
        ok2 = pair_compatibility(members, allowed_pair)

    def compatible(piv):
        mask = np.ones(m, dtype=bool)
        if use_c:
            for i in piv:
                mask &= ok2[i]
            for i, j in itertools.combinations(piv, 2):
                cols = np.flatnonzero(members[i] & members[j] & ~allowed_trip)
                if cols.size:
                    mask &= ~members[:, cols].any(axis=1)
        mask[list(piv)] = False
        return mask

    def pivots():
        if pivot_list is not None:
            for piv in pivot_list:
                yield tuple(int(t) for t in piv), False
            return
        k = r - 2

        def rec(prefix, allowed):
            if len(prefix) == k:
                yield tuple(prefix), True
                return
            for t in np.flatnonzero(allowed):
                nxt = allowed.copy()
                nxt[: t + 1] = False
                if use_c:
                    nxt &= ok2[t]
                    for i in prefix:
                        sh = np.flatnonzero(members[i] & members[t] & ~allowed_trip)
                        if sh.size:
                            nxt &= ~members[:, sh].any(axis=1)
                yield from rec(prefix + [int(t)], nxt)

        yield from rec([], np.ones(m, dtype=bool))

    for piv, ordered in pivots():
        st["pivots"] = st.get("pivots", 0) + 1
        R, pc = rref([pm1] + [Vm1[i] for i in piv], P1)
        if len(pc) < r - 1:                    # dependent mod P1
            if dependent(list(piv), with_psi=False):
                continue                       # dependent pivots: fewer terms
            if dependent(list(piv)):
                raise Inconsistent(f"psi in the span of pivots {piv}")
            raise AssertionError(f"pivots {piv} independent but degenerate mod P1")
        cmask = compatible(piv)
        if ordered and piv:
            cmask[: piv[-1] + 1] = False
        cand = np.flatnonzero(cmask)
        if cand.size == 0:
            continue
        C, has = canon_rows(quotient_images(Vm1[cand], R, pc, P1), P1)
        st["images"] = st.get("images", 0) + cand.size
        for t in cand[~has]:                   # t in span(psi, pivots) mod P1
            full = list(piv) + [int(t)]
            if not dependent(full):
                # every pair (t, t') would look dependent mod P1; never silently
                raise AssertionError(f"image of {t} vanishes mod P1 only")
            if not dependent(full, with_psi=False):
                raise Inconsistent(f"psi in the span of {full}")
            # pivots and t dependent: a decomposition using them has fewer terms
        for u, v in parallel_pairs(C, has, P1):
            t1, t2 = int(cand[u]), int(cand[v])
            if use_c and not ok2[t1, t2]:
                continue
            full = list(piv) + [t1, t2]
            if not dependent(full):
                continue                       # independent over Q(i)
            if not dependent(full, with_psi=False):
                raise Found(f"psi in the span of states {full}")
            st["dependent"] = st.get("dependent", 0) + 1
    return True


# ----------------------------------------------------------------- recursive prover


class Vocab:
    """Stabilizer states (rows, first nonzero entry 1) with their signed groups and
    their residues mod P1 and P2."""

    def __init__(self, n, re, im, elems):
        self.n, self.re, self.im = n, re, im
        self.elems = elems                     # (m, 2^n - 1) non-identity elements
        self.m = re.shape[0]
        self.M1 = to_modp(re, im, P1, I1)
        self.M2 = to_modp(re, im, P2, I2)
        self.S = int(np.max(np.sum(re * re + im * im, axis=1))) if self.m else 0
        self._members = None

    @property
    def members(self):
        if self._members is None:
            mem = np.zeros((self.m, 2 * 4 ** self.n), dtype=bool)
            if self.m:
                mem[np.arange(self.m)[:, None], self.elems] = True
            self._members = mem
        return self._members

    def subset(self, rows):
        return Vocab(self.n, self.re[rows], self.im[rows], self.elems[rows])


def vocab_from_groups(n, gens_list):
    N = 1 << n
    m = len(gens_list)
    re = np.zeros((m, N), dtype=np.int64)
    im = np.zeros((m, N), dtype=np.int64)
    elems = np.zeros((m, N - 1), dtype=np.int64)
    for t, g in enumerate(gens_list):
        re[t], im[t] = state_of_group(g, n)
        elems[t] = group_elements(g, n)[1:]
    return Vocab(n, re, im, elems)


_FULL = {}


def full_vocab(n):
    """All n-qubit stabilizer states (n <= 4), checked against the known count."""
    if n not in _FULL:
        V = vocab_from_groups(n, groups_in(n, np.ones(2 * 4 ** n, dtype=bool)))
        expect = 1 << n
        for k in range(1, n + 1):
            expect *= (1 << k) + 1
        if V.m != expect or len({zkey(V.re[i], V.im[i]) for i in range(V.m)}) != expect:
            raise AssertionError(f"stabilizer enumeration on {n} qubits")
        _FULL[n] = V
    return _FULL[n]


def vocabulary(n, cheap):
    """The vocabulary of a cheap set: every stabilizer state whose non-identity
    signed elements are all cheap."""
    cheap = np.asarray(cheap, dtype=bool)
    if n <= 4:
        F = full_vocab(n)
        return F.subset(np.flatnonzero(cheap[F.elems].all(axis=1)))
    return vocab_from_groups(n, groups_in(n, cheap))


class Prover:
    """Lower bounds chi(psi) >= k, each one a complete exact exclusion.

    lower_bound(psi, n, kmax) returns the largest k <= kmax it proves. On n <= 3
    qubits the vocabulary is the full list of stabilizer states; above, it is the
    vocabulary of the projection lemma, with the projected ranks bounded by the
    same method one qubit down."""

    def __init__(self, log=None, direct_n=3):
        self.memo = {}
        self.log = log
        self.direct_n = direct_n
        self.stats = {}

    def lower_bound(self, re, im, n, kmax, orbit=None):
        if is_zero(re, im):
            return 0
        if kmax <= 1:
            return 1
        key = (zkey(re, im), n)
        return self._lb(key, re, im, n, kmax, orbit)

    def _lb(self, key, re, im, n, kmax, orbit=None):
        lb, tried = self.memo.get(key, (1, 1))
        if tried >= kmax:
            return min(lb, kmax)
        if tried == 1:
            if is_stabilizer(re, im, n):
                self.memo[key] = (1, 10 ** 9)
                return 1
            lb, tried = 2, 2
        while tried < kmax and lb == tried:
            k = tried + 1
            ok = self.prove_ge(re, im, n, k, orbit)
            tried = k
            if ok:
                lb = k
        self.memo[key] = (lb, tried)
        return min(lb, kmax)

    def projected_bounds(self, re, im, n, r, orbit=None, sub_orbit=None):
        """lb[g] = proven lower bound (capped at r) on chi of the (n-1)-qubit
        projection killing g, for every signed Pauli g; -1 on +-I. With `orbit`
        (orbit[g] = representative of g under a symmetry of psi), only
        representatives are computed."""
        M = 2 * 4 ** n
        dre, dim = kill_project_all(re, im, n)
        rows = np.arange(2, M) if orbit is None else np.unique(orbit[2:])
        canon, has = canon_rows_exact(dre[rows], dim[rows])
        uniq, first, inv = np.unique(canon, axis=0, return_index=True, return_inverse=True)
        vals = np.zeros(uniq.shape[0], dtype=np.int64)
        for t in range(uniq.shape[0]):
            g = rows[first[t]]
            if not has[first[t]]:
                vals[t] = 0
                continue
            so = sub_orbit(int(g)) if sub_orbit is not None else None
            vals[t] = self._lb((uniq[t].tobytes(), n - 1), dre[g], dim[g], n - 1, r, so)
        lbg = np.full(M, -1, dtype=np.int64)
        lbg[rows] = vals[np.ravel(inv)]
        if orbit is not None:
            lbg[2:] = lbg[orbit[2:]]
        return lbg

    def prove_ge(self, re, im, n, k, orbit=None):
        """chi(psi) >= k, given chi(psi) >= k - 1 already proved."""
        r = k - 1
        psi = (re, im)
        try:
            if n <= self.direct_n:
                return exclude(psi, full_vocab(n), r, stats=self.stats)
            lbg = self.projected_bounds(re, im, n, r, orbit)
            cheap = lbg < r
            V = vocabulary(n, cheap)
            if self.log:
                self.log(f"  n={n} k={k}: cheap {int(cheap[2:].sum())}, vocabulary {V.m}")
            return exclude(psi, V, r, lbg <= r - 2, lbg <= r - 3, stats=self.stats)
        except Found:
            return False


# ----------------------------------------------------------------- monomial symmetries


class MonomialOp:
    """x-basis monomial map U|x> = i^{phi(x)} |pi(x)>, antilinear if `conj`:
    U(sum_x c_x |x>) = sum_x c~_x i^{phi(x)} |pi(x)>, c~ = conj(c) or c."""

    def __init__(self, pi, phi, conj=False, name=""):
        self.pi = np.asarray(pi, dtype=np.int64)
        self.phi = np.asarray(phi, dtype=np.int64) % 4
        self.conj = conj
        self.name = name

    def apply(self, re, im):
        if self.conj:
            im = -im
        out_re = np.empty_like(re)
        out_im = np.empty_like(im)
        r2, i2 = times_i_pow(re, im, self.phi)
        out_re[..., self.pi] = r2
        out_im[..., self.pi] = i2
        return out_re, out_im

    def on_paulis(self, n):
        """Permutation of signed Pauli indices g -> U g U^{-1} (asserted Pauli)."""
        T = tables(n)
        N = T.N
        pi, phi = self.pi, self.phi
        pinv = np.empty_like(pi)
        pinv[pi] = np.arange(N)
        x = np.arange(N, dtype=np.int64)
        g = np.arange(T.M, dtype=np.int64)
        a, b, s = decode(g, n)
        eP = (POP[a & b][:, None] + 2 * s[:, None] + 2 * PAR[b[:, None] & x[None, :]])
        xa = x[None, :] ^ a[:, None]
        E = (-eP if self.conj else eP) - phi[None, :] + phi[xa]
        F = E[:, pinv] % 4                     # U g U^-1 |y> = i^F(y) |pi(pi^-1(y) ^ a)>
        ap = pi[a] ^ pi[0]
        if not np.array_equal(pi[xa], pi[None, :] ^ ap[:, None]):
            raise AssertionError(f"{self.name}: not affine")
        bp = np.zeros_like(a)
        for k in range(n):
            d = (F[:, 1 << k] - F[:, 0]) % 4
            if np.any(d & 1):
                raise AssertionError(f"{self.name}: not a Pauli")
            bp |= (d >> 1) << k
        d0 = (F[:, 0] - POP[ap & bp]) % 4
        if np.any(d0 & 1):
            raise AssertionError(f"{self.name}: not Hermitian")
        if not np.array_equal(F, (F[:, :1] + 2 * PAR[bp[:, None] & x[None, :]]) % 4):
            raise AssertionError(f"{self.name}: phases not of Pauli form")
        img = encode(ap, bp, d0 >> 1, n)
        if not np.array_equal(np.sort(img), g):
            raise AssertionError(f"{self.name}: not a bijection")
        return img


def orbit_labels(perms, size):
    """label[i] = smallest index in the orbit of i under the group the perms generate."""
    lab = np.arange(size, dtype=np.int64)
    while True:
        old = lab.copy()
        for p in perms:
            m = np.minimum(lab, lab[p])
            np.minimum.at(m, p, lab)            # i ~ p[i] both ways
            lab = m
        lab = lab[lab]
        if np.array_equal(lab, old):
            return lab


def schreier_generators(perms, point):
    """Generators of the stabilizer of `point` in <perms>, as permutations."""
    trans = {point: np.arange(perms[0].size, dtype=np.int64)}
    queue = [point]
    out = {}
    while queue:
        x = queue.pop()
        tx = trans[x]
        for p in perms:
            y = int(p[x])
            ptx = p[tx]                        # p o t_x : point -> y
            if y not in trans:
                trans[y] = ptx
                queue.append(y)
            else:
                ty = trans[y]
                inv = np.empty_like(ty)
                inv[ty] = np.arange(ty.size)
                h = inv[ptx]                   # t_y^-1 o p o t_x fixes point
                if h[point] != point:
                    raise AssertionError("Schreier generator moves the point")
                out[h.tobytes()] = h
    return list(out.values()), len(trans)


def transport_paulis(g, n):
    """For q commuting with g, the signed (n-1)-qubit Pauli acting as q on the -1
    eigenspace of g in the coordinates of `restrict_index`; -1 elsewhere."""
    T = tables(n)
    Tm = tables(n - 1)
    keep = T.KEEP[g]
    Nh = T.N // 2
    L_re = np.zeros((Nh, T.N), dtype=np.int64)    # lifted basis: e_x - g e_x
    L_re[np.arange(Nh), keep] = 1
    L_im = np.zeros_like(L_re)
    gr, gi = apply_pauli(L_re, L_im, g, n)
    L_re, L_im = L_re - gr, L_im - gi
    scale = int(L_re[0, keep[0]])                 # 1, or 2 for a diagonal g
    out = np.full(T.M, -1, dtype=np.int64)
    y = np.arange(Nh, dtype=np.int64)
    for q in range(2, T.M):
        if not T.commute(q, g) or (q >> 1) == (g >> 1):
            continue
        qr, qi = apply_pauli(L_re, L_im, q, n)
        Wr, Wi = qr[:, keep], qi[:, keep]          # row x' = q~ e_x' (times scale)
        tgt = np.argmax((Wr != 0) | (Wi != 0), axis=1)
        if np.any(Wr % scale) or np.any(Wi % scale):
            raise AssertionError("transport: scale")
        vr, vi = Wr[y, tgt] // scale, Wi[y, tgt] // scale
        e = np.where(vr == 1, 0, np.where(vi == 1, 1, np.where(vr == -1, 2, 3)))
        if np.any(np.abs(vr) + np.abs(vi) != 1) or np.any(((Wr != 0) | (Wi != 0)).sum(1) != 1):
            raise AssertionError("transport: not monomial")
        ap = int(tgt[0])                          # q~ e_x' = i^e(x') e_{x' ^ a'}
        if not np.array_equal(tgt, y ^ ap):
            raise AssertionError("transport: not a Pauli")
        bp = 0
        for k in range(n - 1):
            d = (e[1 << k] - e[0]) % 4
            if d & 1:
                raise AssertionError("transport: phase")
            bp |= (d >> 1) << k
        d0 = (e[0] - POP[ap & bp]) % 4
        if d0 & 1 or np.any(e % 4 != (e[0] + 2 * PAR[bp & y]) % 4):
            raise AssertionError("transport: phase form")
        out[q] = int(encode(ap, bp, d0 >> 1, n - 1))
    return out


def transported_orbits(g, n, stab_perms):
    """orbit[q~] on (n-1)-qubit signed Paulis induced by perms fixing g."""
    qt = transport_paulis(g, n)
    Mm = 2 * 4 ** (n - 1)
    dom = np.flatnonzero(qt >= 0)
    if np.unique(qt[dom]).size != Mm - 2:
        raise AssertionError("transport does not cover the (n-1)-qubit Paulis")
    small = []
    for h in stab_perms:
        if not np.all(qt[h[dom]] >= 0):
            raise AssertionError("stabilizer perm leaves the commutant")
        p = np.arange(Mm, dtype=np.int64)
        p[qt[dom]] = qt[h[dom]]
        small.append(p)
    uniq = {p.tobytes(): p for p in small}
    lab = orbit_labels(list(uniq.values()), Mm)
    return lab, qt


# ----------------------------------------------------------------- symmetric top level


class Symmetry:
    """A group of monomial symmetries of psi (U psi proportional to psi, asserted),
    acting on signed Paulis and on vocabularies."""

    def __init__(self, ops, re, im, n):
        self.ops, self.n = ops, n
        k0 = zkey(re, im)
        for op in ops:
            if zkey(*op.apply(re, im)) != k0:
                raise AssertionError(f"{op.name} is not a symmetry")
        self.perms = [op.on_paulis(n) for op in ops]
        self.orbit = orbit_labels(self.perms, 2 * 4 ** n)
        self._sub = {}

    def sub_orbit(self, g):
        """Orbits on (n-1)-qubit signed Paulis of the stabilizer of g, transported."""
        if g not in self._sub:
            hp, _ = schreier_generators(self.perms, g)
            self._sub[g] = transported_orbits(g, self.n, hp)[0]
        return self._sub[g]

    def on_vocab(self, V):
        """Permutations of the vocabulary rows induced by the generators."""
        canon, _ = canon_rows_exact(V.re, V.im)
        index = {canon[i].tobytes(): i for i in range(V.m)}
        out = []
        for op in self.ops:
            c2, _ = canon_rows_exact(*op.apply(V.re, V.im))
            p = np.array([index.get(c2[i].tobytes(), -1) for i in range(V.m)])
            if np.any(p < 0) or np.unique(p).size != V.m:
                raise AssertionError(f"vocabulary not closed under {op.name}")
            out.append(p)
        return out


def pivot_lists(vperms, m, r, ok2=None, members=None, allowed_trip=None):
    """Pivot tuples for exclude(..., pivot_list=...) reduced by a symmetry group
    acting on the vocabulary by `vperms`. Every r-set is mapped by the group to
    one containing the listed pivots, the rest being free: r = 3 uses orbit
    representatives of states, r = 4 of ordered pairs, r = 5 pair representatives
    followed by every third state."""
    if r <= 2:
        return [()]
    single = orbit_labels(vperms, m)
    if r == 3:
        return [(int(i),) for i in np.unique(single)]
    pidx = np.arange(m * m, dtype=np.int64)
    pperms = [(p[pidx // m] * m + p[pidx % m]) for p in vperms]
    plab = orbit_labels(pperms, m * m)
    reps = [int(t) for t in np.unique(plab) if t // m != t % m]
    pairs = [(t // m, t % m) for t in reps]
    if ok2 is not None:
        pairs = [(i, j) for i, j in pairs if ok2[i, j]]
    if r == 4:
        return pairs
    out = []
    for i, j in pairs:
        mask = np.ones(m, dtype=bool) if ok2 is None else ok2[i] & ok2[j]
        if members is not None:
            cols = np.flatnonzero(members[i] & members[j] & ~allowed_trip)
            if cols.size:
                mask &= ~members[:, cols].any(axis=1)
        mask[[i, j]] = False
        out += [(i, j, int(k)) for k in np.flatnonzero(mask)]
    return out
