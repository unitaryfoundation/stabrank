"""Slice-and-lift for the magic cat family, in the even-weight compressed picture.

Target claim (if every step passes): CERTIFIED chi(cat_7) >= 4
(and CERTIFIED chi(cat_8) >= 4 by the <0| monotonicity of
cert_cat_m5_rank2.py).

Setting. c_m(y) = cat_m(y, parity(y)) on m-1 qubits is the even-weight
compression of |cat_m>; chi(cat_m) = chi(c_m) (cert_cat_m5_rank2.py,
step 1). All entries of c_m are fourth roots of unity, c_m(y) = i^{(|y| +
par(y))/2}. Slicing c_{m+1} along its first qubit gives, exactly,
    c_{m+1}(0, y) = c_m(y),      c_{m+1}(1, y) = D(y) c_m(y),
with D(y) = i if |y| is even and 1 if |y| is odd; D = i * (-i)^{y_1 xor ... },
and (-i)^{xor} = prod_j (-i)^{y_j} prod_{j<k} (-1)^{y_j y_k} is a diagonal
Clifford, so chi(D c_m) = chi(c_m). (Both identities are asserted below.)

Lemma (qubit slice-and-lift, as in verify_challenge/slice_lift.py,
`lifts_qubit`). If c_{m+1} = sum_{j<=r} e_j s_j with r = chi(c_m), slicing
gives c_m = sum e_j a_j and D c_m = sum e_j b_j, with a_j = (<0| (x) I) s_j,
b_j = (<1| (x) I) s_j. Each slice needs r nonzero terms, so every a_j and b_j
is nonzero, {a_j} is a rank-r decomposition of c_m, and for a qubit
stabilizer state with both slices nonzero b_j = mu_j P_j a_j with P_j an
X/Z Pauli string and mu_j a fourth root of unity. So
    chi(c_{m+1}) = r  =>  some rank-r decomposition (e_j, a_j) of c_m has
                          sum_j e_j mu_j P_j a_j = D c_m.
If no rank-r decomposition of c_m admits such a matching, chi(c_{m+1}) > r.

Symmetry. A Clifford g on the m-1 slice qubits with g c_m = nu c_m and
g D c_m = nu D c_m (same nu) maps decompositions to decompositions and
liftable ones to liftable ones (I (x) g fixes c_{m+1} up to nu). Used here:
all permutations of the qubits (c_m depends only on |y|), and R (x) R on a
pair, R = (X + Y)/sqrt 2, which maps |00> -> i|11>, |11> -> -i|00>,
|01> <-> |10> and preserves y_a xor y_b. Both properties are asserted
exactly for every generator. So it suffices to take pivots from one state
per orbit of the group they generate: every decomposition has a term, and
moving that term to its orbit representative moves the whole decomposition
to one containing the representative.

Enumeration of rank-3 decompositions (exact). For the pivot s1, with
q(s) = s - (s_0 / t_0) t (kernel span{t}) and
q2(s) = q(s1)_k q(s) - q(s)_k q(s1) (kernel span{t, s1}), a set {s1, s2, s3}
of distinct rays with t in its span and no two spanning t (rank 2 is
excluded exactly) is exactly a pair with q2(s2), q2(s3) nonzero and parallel.
Parallelism is screened modulo the prime P = 2013265921 (P = 1 mod 4, i -> a
square root of -1). This is exact here, by norms that the script asserts:
every entry of q is a Gaussian integer with |q| <= 2 (checked exactly on the
whole dictionary), so |q2| <= 8 (norm <= 64) and every 2 x 2 minor of two q2
rows has norm <= 128^2 = 16384, far below P. A nonzero Gaussian integer of
norm < P is nonzero mod P, so zero patterns survive reduction, and two rows
are parallel over Q(i) exactly when they are parallel mod P. The same holds
for the independence test on the quotients by the pivot (entries of modulus
<= 2, minors <= 8). Every candidate is then also confirmed in exact
Gaussian-rational arithmetic. The lift matching is screened the same
way (meet in the middle over the 4 * 2^(m-1) values mu P a_j per term) and
every hit confirmed exactly. Nothing in a decision rests on floating point.

Premises re-established in the run: chi(c_6) >= 3 (the exact rank-2
exclusion of cert_cat_m5_rank2.py at m = 5, then the <0| slice identity),
and the slice identity (I (x) <0|) cat_8 = 2^{-1/2} cat_7 before the m = 8
claim is printed.

Controls, same code path:
  (+) c_5 -> c_6: chi(cat_5) = chi(cat_6) = 3 (bounds/cat-m6-upper-3.json),
      so some rank-3 decomposition of c_5 must lift; the run fails otherwise.
  (-) c_4 -> c_5 at rank 2: chi(cat_4) = 2 < 3 = chi(cat_5), so no rank-2
      decomposition of c_4 may lift.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from fractions import Fraction

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

P = 2013265921
RI = pow(31, (P - 1) // 4, P)        # a square root of -1 mod P
assert RI * RI % P == P - 1
POW_I = np.array([1, RI, P - 1, P - RI], dtype=np.int64)
UNIT = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def fail(msg):
    print(f"FAILED: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


# ------------------------------------------------ stabilizer dictionary ----

def rref_subspaces(n, k):
    for piv in itertools.combinations(range(n), k):
        free = [(r, c) for r in range(k) for c in range(n) if c not in piv and c > piv[r]]
        for vals in itertools.product((0, 1), repeat=len(free)):
            W = np.zeros((k, n), np.int64)
            for r in range(k):
                W[r, piv[r]] = 1
            for (r, c), a in zip(free, vals):
                W[r, c] = a
            yield piv, W


def grid(vals, k):
    rows = list(itertools.product(vals, repeat=k))
    return np.array(rows, np.int64).reshape(len(rows), k)


def stabilizer_exponents(n):
    """All n-qubit stabilizer states as int8 exponent rows: e[x] in {0..3}
    (amplitude i^e) or -1 (off support), normalized so the smallest support
    index carries exponent 0. Exact; count asserted."""
    dim = 1 << n
    rows = []
    for k in range(n + 1):
        ys = grid((0, 1), k)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        quad = np.array([[y[i] * y[j] for (i, j) in pairs] for y in ys], np.int64).reshape(len(ys), len(pairs))
        Qs = grid((0, 1), len(pairs))
        Ls = grid(range(4), k)
        # exponent table over (Q, l) for each y: l.y + 2 Q(y)
        ph = (2 * (Qs @ quad.T))[:, None, :] + (Ls @ ys.T)[None, :, :]        # (nQ, nL, 2^k)
        ph = (ph % 4).reshape(-1, len(ys))
        weights = 1 << (n - 1 - np.arange(n))
        for piv, W in rref_subspaces(n, k):
            nonpiv = [c for c in range(n) if c not in piv]
            for x0f in itertools.product((0, 1), repeat=n - k):
                x0 = np.zeros(n, np.int64)
                x0[nonpiv] = x0f
                pts = ((x0[None, :] + ys @ W) % 2) @ weights
                E = np.full((ph.shape[0], dim), -1, np.int8)
                E[:, pts] = ph
                rows.append(E)
    E = np.concatenate(rows)
    expect = 1 << n
    for j in range(1, n + 1):
        expect *= (1 << j) + 1
    if E.shape[0] != expect:
        fail(f"{E.shape[0]} stabilizer states on {n} qubits, expected {expect}")
    return renormalize(E)


def renormalize(E):
    first = np.argmax(E >= 0, axis=1)
    base = E[np.arange(E.shape[0]), first].astype(np.int16)
    out = np.where(E >= 0, (E.astype(np.int16) - base[:, None]) % 4, -1).astype(np.int8)
    return out


def keys_of(E):
    v = np.ascontiguousarray(E).view(np.dtype((np.void, E.shape[1])))
    return v.ravel()


def to_modp(E):
    M = np.where(E >= 0, POW_I[np.clip(E, 0, 3)], 0)
    return M.astype(np.int64)


def to_gauss(e_row):
    return [(0, 0) if v < 0 else UNIT[v] for v in e_row]


# --------------------------------------------------------- the targets -----

def c_exponents(m):
    """compressed cat_m on m-1 qubits: exponent (|y| + par)/2 mod 4."""
    n = m - 1
    w = np.array([bin(y).count("1") for y in range(1 << n)])
    return (((w + (w & 1)) // 2) % 4).astype(np.int8)


def d_exponents(n):
    """D(y): exponent 1 (i) if |y| even, 0 if odd."""
    w = np.array([bin(y).count("1") for y in range(1 << n)])
    return np.where(w % 2 == 0, 1, 0).astype(np.int8)


def check_slices(m):
    big, small = c_exponents(m + 1), c_exponents(m)
    half = 1 << (m - 1)
    if not np.array_equal(big[:half], small):
        fail(f"slice 0 of c_{m + 1} is not c_{m}")
    if not np.array_equal(big[half:], (small + d_exponents(m - 1)) % 4):
        fail(f"slice 1 of c_{m + 1} is not D c_{m}")


# ------------------------------------------------------------ symmetry -----

def perm_index_map(n, perm):
    """index map for permuting qubit positions: new[x] = old[x'] ."""
    dim = 1 << n
    src = np.zeros(dim, np.int64)
    for x in range(dim):
        bits = [(x >> (n - 1 - j)) & 1 for j in range(n)]
        nb = [bits[perm[j]] for j in range(n)]
        src[x] = sum(b << (n - 1 - j) for j, b in enumerate(nb))
    return src


def apply_perm(E, src):
    return E[:, src]


def apply_rr(E, n, a=0, b=1):
    """R (x) R on qubits a, b: |00> -> i|11>, |11> -> -i|00>, |01> <-> |10>."""
    dim = 1 << n
    x = np.arange(dim)
    ba, bb = (x >> (n - 1 - a)) & 1, (x >> (n - 1 - b)) & 1
    flip = x ^ (1 << (n - 1 - a)) ^ (1 << (n - 1 - b))
    add = np.where((ba == 0) & (bb == 0), 1, np.where((ba == 1) & (bb == 1), 3, 0))
    # new[flip(x)] = i^add(x) old[x]
    out = np.full_like(E, -1)
    newv = np.where(E >= 0, (E.astype(np.int16) + add[None, :]) % 4, -1).astype(np.int8)
    out[:, flip] = newv
    return out


def parity_swap_maps(n):
    """Swap of the last slice qubit with the dropped parity qubit of the
    uncompressed cat state. On slice 0 the parity qubit is par(y), on slice 1
    it is 1 xor par(y); new[y] = old[y_1 .. y_{n-1}, that parity bit]."""
    dim = 1 << n
    ev, od = np.zeros(dim, np.int64), np.zeros(dim, np.int64)
    for y in range(dim):
        b = bin(y).count("1") & 1
        ev[y] = (y & ~1) | b
        od[y] = (y & ~1) | (1 - b)
    return ev, od


def generators(n, full=True):
    """Symmetries I (x) g of the uncompressed cat_{m+1} on its qubits 2..m+1,
    as (kind, map on slice 0, map on slice 1): qubit permutations (with
    `full`, including the dropped parity qubit) and R (x) R on a pair. The
    slice-0 map acts on the dictionary; both maps are checked on c_{m+1}."""
    gens = []
    if n >= 2:
        for perm in ([1, 0] + list(range(2, n)), list(range(1, n)) + [0]):
            src = perm_index_map(n, perm)
            gens.append(("perm", src, src))
        gens.append(("rr", None, None))
    if full and n >= 1:
        ev, od = parity_swap_maps(n)
        gens.append(("perm", ev, od))
    return gens


def act(g, E, n, odd=False):
    kind, ev, od = g
    if kind == "perm":
        return apply_perm(E, od if odd else ev)
    return apply_rr(E, n)


def check_symmetry(gens, m):
    """I (x) g must fix c_{m+1} = |0> c_m + |1> D c_m up to one scalar."""
    n = m - 1
    t = c_exponents(m)[None, :]
    dt = ((c_exponents(m) + d_exponents(n)) % 4)[None, :]
    for g in gens:
        gt, gdt = act(g, t, n), act(g, dt, n, odd=True)
        if (gt < 0).any() or (gdt < 0).any():
            fail("a generator moved the target off its support")
        nu1 = (gt[0] - t[0]) % 4
        nu2 = (gdt[0] - dt[0]) % 4
        if len(set(nu1.tolist())) != 1 or len(set(nu2.tolist())) != 1 or nu1[0] != nu2[0]:
            fail(f"generator {g[0]} does not fix c_{m + 1} up to one scalar")


def orbit_reps(E, gens, n):
    N = E.shape[0]
    if not gens:
        return np.arange(N)
    keys = keys_of(E)
    order = np.argsort(keys)
    sk = keys[order]
    parent = np.arange(N)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for g in gens:
        img = renormalize(act(g, E, n))
        ik = keys_of(img)
        pos = np.searchsorted(sk, ik)
        if (pos >= N).any() or (sk[np.minimum(pos, N - 1)] != ik).any():
            fail("a generator does not permute the dictionary")
        j = order[pos]
        for a, b in zip(range(N), j):
            ra, rb = find(a), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(N)])
    return roots


# ------------------------------------------------------ modular helpers ----

def inv_mod(a):
    """vectorized a^(P-2) mod P."""
    res = np.ones_like(a)
    base = a % P
    e = P - 2
    while e:
        if e & 1:
            res = res * base % P
        base = base * base % P
        e >>= 1
    return res


def proj_normal(V):
    """rows divided by their first nonzero entry mod P (zero rows stay zero)."""
    nz = V != 0
    has = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    lead = V[np.arange(V.shape[0]), first]
    lead = np.where(has, lead, 1)
    return V * inv_mod(lead)[:, None] % P, has


def proj_hash(V, rng_vecs, W=None):
    """projective hash of rows (zero rows get -1)."""
    if W is None:
        W, has = proj_normal(V)
    else:
        has = (W != 0).any(axis=1)
    h = [(W * r[None, :] % P).sum(axis=1) % P for r in rng_vecs]
    h1 = np.where(has, h[0], -1)
    h2 = np.where(has, h[1], -1)
    return h1, h2


# ------------------------------------------------- exact verification ------

def gmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def gdiv(a, b):
    d = b[0] * b[0] + b[1] * b[1]
    n = gmul(a, (b[0], -b[1]))
    return (Fraction(n[0], d), Fraction(n[1], d))


def solve_exact(cols, t):
    """Gaussian-rational solve of sum_j d_j cols[j] = t; None if inconsistent."""
    r = len(cols)
    A = [[(Fraction(c[i][0]), Fraction(c[i][1])) for c in cols] + [(Fraction(t[i][0]), Fraction(t[i][1]))]
         for i in range(len(t))]
    rows, piv_cols, ri = len(A), [], 0
    for cj in range(r):
        p = next((i for i in range(ri, rows) if A[i][cj] != (0, 0)), None)
        if p is None:
            return None
        A[ri], A[p] = A[p], A[ri]
        pv = A[ri][cj]
        A[ri] = [gdiv(x, pv) for x in A[ri]]
        for i in range(rows):
            if i != ri and A[i][cj] != (0, 0):
                f = A[i][cj]
                A[i] = [(x[0] - (f[0] * y[0] - f[1] * y[1]), x[1] - (f[0] * y[1] + f[1] * y[0]))
                        for x, y in zip(A[i], A[ri])]
        piv_cols.append(cj)
        ri += 1
    for i in range(ri, rows):
        if A[i][r] != (0, 0):
            return None
    return [A[i][r] for i in range(r)]


def combo_exact(ds, vecs):
    out = [(Fraction(0), Fraction(0))] * len(vecs[0])
    for d, v in zip(ds, vecs):
        out = [(o[0] + d[0] * x[0] - d[1] * x[1], o[1] + d[0] * x[1] + d[1] * x[0]) for o, x in zip(out, v)]
    return out


# --------------------------------------------------------- the search ------

def assert_norms(E, t, Q1):
    """Exact check that every entry of q(s) = s - s_0 t is a Gaussian integer
    of modulus <= 2, that the mod-P table equals its reduction, and that the
    derived bounds on q2 and on all 2 x 2 minors stay below P."""
    U = np.array(UNIT + [(0, 0)], np.int64)            # exponent -1 -> index 4
    se = np.where(E >= 0, E, 4).astype(np.int64)
    sr, si = U[se, 0], U[se, 1]
    tr, ti = U[t.astype(np.int64), 0], U[t.astype(np.int64), 1]
    s0r, s0i = sr[:, :1], si[:, :1]
    qr = sr - (s0r * tr[None, :] - s0i * ti[None, :])
    qi = si - (s0r * ti[None, :] + s0i * tr[None, :])
    if (qr * qr + qi * qi).max() > 4:
        fail("an entry of q has modulus > 2")
    if not np.array_equal((qr + qi * RI) % P, Q1):
        fail("mod-P quotient table disagrees with the exact Gaussian quotient")
    q2_norm = (2 * 2 + 2 * 2) ** 2          # |q2| <= |q1_k||q| + |q_k||q1| <= 8
    minor_norm = (2 * 8 * 8) ** 2           # |a d - b c| <= 2 * 8 * 8
    dep_minor_norm = (2 * 2 * 2) ** 2       # quotients by the pivot: entries <= 2
    if not (q2_norm < P and minor_norm < P and dep_minor_norm < P):
        fail("norm bound not below P")


def decompositions(m, rank, E, M, reps, log_every=200):
    """All rank-`rank` (2 or 3) decompositions of c_m containing a pivot from
    `reps`, as sorted index tuples; exact."""
    n = m - 1
    t = c_exponents(m)
    tm = POW_I[t]
    rng = np.random.default_rng(12345)
    rv = [rng.integers(1, P, size=1 << n) for _ in range(2)]
    # q(s) = s - (s_0 / t_0) t ; t_0 = 1 (exponent 0 at y = 0)
    if t[0] != 0:
        fail("c_m(0) != 1")
    Q1 = (M - M[:, :1] * tm[None, :]) % P
    assert_norms(E, t, Q1)
    found = set()
    if rank == 2:
        h1, h2 = proj_hash(Q1, rv)
        if (h1 < 0).any():
            fail("target is a stabilizer state")
        key = h1 * P + h2
        order = np.argsort(key, kind="stable")
        sk = key[order]
        same = np.flatnonzero(sk[1:] == sk[:-1])
        groups = {}
        for a in same:
            groups.setdefault(int(sk[a]), set()).update((int(order[a]), int(order[a + 1])))
        for g in groups.values():
            for i, j in itertools.combinations(sorted(g), 2):
                found.add((i, j))
        return confirm(found, E, t)
    t0 = time.time()
    # reps = orbit root of every state. Sort states by root; a pivot (an orbit
    # representative) only needs partners whose orbit root is >= its own
    # (move the term with the smallest orbit root of a decomposition to its
    # representative; the other terms keep their orbits).
    roots = reps
    order_r = np.argsort(roots, kind="stable")
    sroots = roots[order_r]
    pivots = np.unique(roots)
    reps = pivots
    for c, s1 in enumerate(pivots):
        start = int(np.searchsorted(sroots, roots[s1]))
        sub = order_r[start:]
        q1 = Q1[s1]
        k = int(np.flatnonzero(q1)[0])
        Q2 = (q1[k] * Q1[sub] - Q1[sub, k:k + 1] * q1[None, :]) % P
        W2, _ = proj_normal(Q2)
        a1 = M[s1]
        k1 = int(np.flatnonzero(a1)[0])
        h1, h2 = proj_hash(None, rv, W2)
        if int((h1 < 0).sum()) != 1:
            fail("a state other than the pivot lies in span{t, pivot}: rank 2 is not excluded")
        key = np.where(h1 < 0, -1, h1 * P + h2)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        same = np.flatnonzero((sk[1:] == sk[:-1]) & (sk[1:] >= 0))
        groups = {}
        for a in same:
            groups.setdefault(int(sk[a]), set()).update((int(order[a]), int(order[a + 1])))
        for g in groups.values():
            for li, lj in itertools.combinations(sorted(g), 2):
                i, j = int(sub[li]), int(sub[lj])
                if s1 in (i, j):
                    continue
                # exact (small Gaussian integers, see docstring): q2 parallel,
                # and {s1, si, sj} independent (their quotients by s1 not parallel)
                if not np.array_equal(W2[li], W2[lj]):
                    continue
                WR, _ = proj_normal((a1[k1] * M[[i, j]] - M[[i, j], k1:k1 + 1] * a1[None, :]) % P)
                if np.array_equal(WR[0], WR[1]):
                    continue
                found.add(tuple(sorted((int(s1), i, j))))
        if log_every and (c + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  pivots {c + 1}/{len(reps)}  candidates {len(found)}  "
                  f"{el:.0f}s elapsed, ~{el / (c + 1) * (len(reps) - c - 1):.0f}s left", flush=True)
    return confirm(found, E, t)


def confirm(cands, E, t):
    tg = to_gauss(t)
    out = []
    for c in cands:
        cols = [to_gauss(E[i]) for i in c]
        d = solve_exact(cols, tg)
        if d is not None and all(x != (0, 0) for x in d):
            out.append((c, d))
    return out


def pauli_images(e_row, n):
    """the 2^n rays P a for P = X^u Z^v, as exponent rows (renormalized not)."""
    dim = 1 << n
    x = np.arange(dim)
    imgs = {}
    for u in range(dim):
        for v in range(dim):
            par = np.array([bin(v & xx).count("1") & 1 for xx in x])
            new = np.full(dim, -1, np.int8)
            src = e_row
            val = np.where(src >= 0, (src.astype(np.int16) + 2 * par) % 4, -1)
            new[x ^ u] = val
            k = renormalize(new[None, :])[0].tobytes()
            imgs.setdefault(k, new)
    if len(imgs) != dim:
        fail(f"{len(imgs)} Pauli rays, expected {dim}")
    return list(imgs.values())


def lifts(dec, E, m):
    """Exact test of sum_j d_j mu_j P_j a_j = D c_m for one decomposition."""
    idx, ds = dec
    n = m - 1
    rhs_e = (c_exponents(m) + d_exponents(n)) % 4
    rhs = POW_I[rhs_e]
    dmod = []
    for d in ds:
        if any(x.denominator % P == 0 for x in d):
            fail("a coefficient denominator is divisible by P")
        re = d[0].numerator * pow(d[0].denominator, P - 2, P) % P
        im = d[1].numerator * pow(d[1].denominator, P - 2, P) % P
        dmod.append((re + im * RI) % P)
    opts = []
    for j, i in enumerate(idx):
        rows = []
        for img in pauli_images(E[i], n):
            for mu in range(4):
                e = np.where(img >= 0, (img.astype(np.int16) + mu) % 4, -1)
                rows.append(e.astype(np.int8))
        rows = np.array(rows)
        opts.append(rows)
    mods = [to_modp(o) * dm % P for o, dm in zip(opts, dmod)]
    r = len(idx)
    # meet in the middle: last term against the sum of the others
    rng = np.random.default_rng(99)
    hv = rng.integers(1, P, size=1 << n)
    last = mods[-1]
    table = {}
    for k, row in enumerate(last):
        table.setdefault(int((row * hv % P).sum() % P), []).append(k)
    hits = []
    for combo in itertools.product(*[range(len(o)) for o in mods[:-1]]):
        s = rhs.copy()
        for j, k in enumerate(combo):
            s = (s - mods[j][k]) % P
        h = int((s * hv % P).sum() % P)
        for k in table.get(h, []):
            full = combo + (k,)
            vecs = [to_gauss(opts[j][kk]) for j, kk in enumerate(full)]
            lhs = combo_exact(ds, vecs)
            want = [(Fraction(a), Fraction(b)) for a, b in to_gauss(rhs_e)]
            if lhs == want:
                hits.append(full)
    return hits


def run_cell(m, rank, label):
    n = m - 1
    t0 = time.time()
    check_slices(m)
    E = stabilizer_exponents(n)
    M = to_modp(E)
    gens = generators(n)
    check_symmetry(gens, m)
    reps = orbit_reps(E, gens, n) if rank == 3 else None
    print(f"[{label}] c_{m} on {n} qubits: {E.shape[0]} stabilizer states"
          + (f", {len(np.unique(reps))} pivot orbits" if reps is not None else "")
          + f" ({time.time() - t0:.1f}s)", flush=True)
    decs = decompositions(m, rank, E, M, reps)
    print(f"[{label}] rank-{rank} decompositions of c_{m} found (pivot reps): {len(decs)} "
          f"({time.time() - t0:.1f}s)", flush=True)
    n_lift = 0
    for dec in decs:
        h = lifts(dec, E, m)
        n_lift += len(h)
    print(f"[{label}] decompositions that lift to c_{m + 1}: {n_lift} ({time.time() - t0:.1f}s)", flush=True)
    return len(decs), n_lift


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "controls"):
        nd, nl = run_cell(4, 2, "control -")
        if nd == 0:
            fail("negative control: no rank-2 decomposition of c_4 found, where chi(cat_4) = 2")
        if nl != 0:
            fail("negative control: a rank-2 decomposition of c_4 lifts, but chi(cat_5) = 3")
        print("negative control passed: c_4 rank 2 does not lift (chi(cat_5) = 3)", flush=True)
        nd, nl = run_cell(5, 3, "control +")
        if nd == 0 or nl == 0:
            fail("positive control: no rank-3 decomposition of c_5 lifts, but chi(cat_6) = 3")
        print("positive control passed: c_5 rank 3 lifts to c_6 (chi(cat_6) = 3)", flush=True)
    if which in ("all", "target"):
        import cert_cat_m5_rank2 as r2
        r2.exclude_rank2_cat5(verbose=False)
        r2.check_slice_identity(6)
        print("premise: chi(cat_5) >= 3 re-established exactly, hence chi(c_6) >= 3 "
              "by the <0| slice identity", flush=True)
        nd, nl = run_cell(6, 3, "target")
        if nd == 0:
            fail("no rank-3 decomposition of c_6 found, but chi(cat_6) = 3")
        if nl:
            print(f"c_6 rank-3 decompositions lift: chi(cat_7) = 3 (a record rank)", flush=True)
            return 2
        print("no rank-3 decomposition of c_6 lifts to c_7", flush=True)
        print("CERTIFIED chi(cat_7) >= 4")
        r2.check_slice_identity(8)
        print("(I (x) <0|) cat_8 = 2^(-1/2) cat_7 exactly; projection monotonicity")
        print("CERTIFIED chi(cat_8) >= 4")
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
