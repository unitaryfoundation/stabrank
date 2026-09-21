"""Count the six-qubit stabilizer states that can be a term of a rank-5
decomposition of |H>^6, by the slice argument of PR #87 and its closure
under the symmetry group of |H>^6.

Stabilizer groups. A six-qubit stabilizer state has support x_0 + V with V a
subspace of F_2^6 of dimension j, and its stabilizer group, as a Lagrangian
subspace L of F_2^12, is {(v, z) : v in V, z . v_a = B(v, v_a) for a basis
v_1, ..., v_j of V} for a symmetric bilinear form B on V; 64 states share
each L, and the count sum_V 2^(j(j+1)/2) = 4,922,775 Lagrangians times 64 is
the 315,057,600 states.

Property P (PR #87): every four-qubit slice of a rank-5 decomposition has
all five terms nonzero, so every term's V projects onto F_2^2 for every pair
of coordinates, that is, V^perp (the pure-Z part of L) has no word of weight
1 or 2.

Property P* (closure). The set of possible terms is invariant under the
unitary symmetry group G of |H>^6 (Hadamards on any subset of qubits and
permutations), so a term satisfies P after every element of G. A Hadamard
on qubit k swaps the x and z coordinates of qubit k in L, so the pure-Z
words of H^a L H^a are the elements of L that have no X on qubits outside
a and no Z on qubits inside a; ranging over a, P* says: L contains no
non-identity element of weight at most 2 all of whose non-identity tensor
factors are X or Z (no Y). Elements with a Y factor are allowed.

The script counts the Lagrangians and states satisfying P and P*, per
support dimension. With --orbits it also counts the orbits of G on the P*
states by an exact union-find (keys are the exact amplitude patterns).
"""

from __future__ import annotations

import itertools
import sys
import time

import numpy as np

N_QUBITS = 6


def subspaces(n):
    """Every subspace of F_2^n as an RREF basis tuple (rows as ints, most
    significant pivot first)."""
    out = []
    for j in range(n + 1):
        for pivots in itertools.combinations(range(n - 1, -1, -1), j):
            free = [[c for c in range(p) if c not in pivots] for p in pivots]
            for choice in itertools.product(*[range(1 << len(f)) for f in free]):
                rows = []
                for p, f, ch in zip(pivots, free, choice):
                    r = 1 << p
                    for bit, c in enumerate(f):
                        if (ch >> bit) & 1:
                            r |= 1 << c
                    rows.append(r)
                out.append(tuple(rows))
    return out


def weight(x):
    return bin(x).count("1")


def dual_distance_ok(basis, n, dmin=3):
    """V^perp has no nonzero word of weight below dmin."""
    for w in range(1, dmin):
        for supp in itertools.combinations(range(n), w):
            z = sum(1 << c for c in supp)
            if all(weight(z & v) % 2 == 0 for v in basis):
                return False
    return True


def forbidden_paulis(n):
    """(v, z) with disjoint supports and 1 <= |v| + |z| <= 2."""
    out = []
    for w in (1, 2):
        for supp in itertools.combinations(range(n), w):
            for kinds in itertools.product((0, 1), repeat=w):   # 0 = X, 1 = Z
                v = sum(1 << c for c, k in zip(supp, kinds) if k == 0)
                z = sum(1 << c for c, k in zip(supp, kinds) if k == 1)
                out.append((v, z))
    return out


def coords(basis, v):
    """t with sum t_a v_a = v, or None if v is not in the span."""
    t = 0
    for a, b in enumerate(basis):
        piv = b.bit_length() - 1
        if (v >> piv) & 1:
            v ^= b
            t |= 1 << a
    return None if v else t


def popcount(a):
    a = a.astype(np.uint64)
    c = np.zeros_like(a)
    while np.any(a):
        c += a & np.uint64(1)
        a >>= np.uint64(1)
    return c


def good_forms(basis, forb):
    """Symmetric forms B on V, as bitmasks over the (a <= b) index pairs,
    whose Lagrangian avoids every forbidden element."""
    j = len(basis)
    pairs = [(a, b) for a in range(j) for b in range(a, j)]
    bit = {p: i for i, p in enumerate(pairs)}
    nB = 1 << len(pairs)
    B = np.arange(nB, dtype=np.uint64)
    bad = np.zeros(nB, dtype=bool)
    for v, z in forb:
        t = coords(basis, v)
        if t is None:
            continue
        cond = np.ones(nB, dtype=bool)          # for all a: sum_b B_ab t_b = z . v_a
        for a in range(j):
            mask = 0
            for b in range(j):
                if (t >> b) & 1:
                    mask |= 1 << bit[(min(a, b), max(a, b))]
            rhs = weight(z & basis[a]) % 2
            par = popcount(B & np.uint64(mask)) & np.uint64(1)
            cond &= par == rhs
        bad |= cond
    return B[~bad], pairs


def counts(n=N_QUBITS, verbose=True):
    subs = subspaces(n)
    assert len(subs) == 2825, len(subs)
    forb = forbidden_paulis(n)
    assert len(forb) == 72
    total_states = total_L = P_states = P_L = Ps_states = Ps_L = 0
    per_dim = {}
    keep = []
    for basis in subs:
        j = len(basis)
        nL = 1 << (j * (j + 1) // 2)
        total_L += nL
        total_states += 64 * nL
        if not dual_distance_ok(basis, n):
            continue
        P_L += nL
        P_states += 64 * nL
        good, pairs = good_forms(basis, forb)
        c = len(good)
        Ps_L += c
        Ps_states += 64 * c
        d = per_dim.setdefault(j, [0, 0, 0])
        d[0] += 1
        d[1] += nL
        d[2] += c
        if c:
            keep.append((basis, good, pairs))
    assert total_states == 315057600, total_states
    if verbose:
        print(f"six-qubit stabilizer states: {total_states} ({total_L} stabilizer groups)")
        print(f"property P (V^perp has distance >= 3): {P_states} states, {P_L} groups")
        print(f"property P* (no XZ-only element of weight <= 2): {Ps_states} states, {Ps_L} groups")
        print("per support dimension j: subspaces with P, groups with P, groups with P*, states with P*")
        for j in sorted(per_dim):
            d = per_dim[j]
            print(f"  j={j}: {d[0]} subspaces, {d[1]} groups with P, {d[2]} groups with P*, {64 * d[2]} states")
    return keep, dict(total=total_states, P=P_states, Pstar=Ps_states, P_L=P_L, Pstar_L=Ps_L)


# ------------------------------------------------------------- orbits -----

def pattern_batches(keep, n, max_batch=4096):
    """Yield (start, patterns) over all P* states in a fixed order: for each
    (basis, forms) block, forms in blocks of `max_batch`, and for each form
    the 2^j choices of the upper bits of l and the 2^(n-j) coset points x_0.

    The state with data (x_0, l, q) is sum_t i^(l.t) (-1)^(q(t)) |x_0 + W t>
    with q the off-diagonal part of B and l = diag(B) mod 2 plus twice a
    free bit vector; x_0 ranges over the 2^(n-j) coset representatives with
    zeros at the pivot columns. Patterns are int8 phase codes (0 = zero,
    1..4 = 1, i, -1, -i), canonical (first nonzero entry is 1)."""
    dim = 1 << n
    start = 0
    for basis, forms, pairs in keep:
        j = len(basis)
        pivcols = [b.bit_length() - 1 for b in basis]
        freecols = [c for c in range(n) if c not in pivcols]
        x0s = np.array([sum(1 << c for bit, c in enumerate(freecols) if (f >> bit) & 1)
                        for f in range(1 << len(freecols))], dtype=np.int64)
        T = np.arange(1 << j)
        tb = ((T[:, None] >> np.arange(j)[None, :]) & 1).astype(np.int64)      # (2^j, j)
        pos = np.zeros(1 << j, dtype=np.int64)
        for a, b in enumerate(basis):
            pos ^= tb[:, a] * b
        off = [(idx, a, b) for idx, (a, b) in enumerate(pairs) if a != b]
        dia = [(idx, a) for idx, (a, b) in enumerate(pairs) if a == b]
        prod = np.array([tb[:, a] * tb[:, b] for _, a, b in off], dtype=np.int64).reshape(len(off), 1 << j)
        offidx = np.array([i for i, _, _ in off], dtype=np.uint64)
        diaidx = np.array([i for i, _ in dia], dtype=np.uint64)
        for s0 in range(0, len(forms), max_batch):
            Bs = forms[s0:s0 + max_batch]
            nb = len(Bs)
            offbits = ((Bs[:, None] >> offidx[None, :]) & np.uint64(1)).astype(np.int64)   # (nb, n_off)
            q = (offbits @ prod) % 2 if len(off) else np.zeros((nb, 1 << j), dtype=np.int64)
            diag = ((Bs[:, None] >> diaidx[None, :]) & np.uint64(1)).astype(np.int64)      # (nb, j)
            l = diag[:, None, :] + 2 * tb[None, :, :]                                        # (nb, 2^j, j)
            ph = (l @ tb.T + 2 * q[:, None, :]) % 4                                          # (nb, 2^j hi, 2^j t)
            codes = (ph + 1).astype(np.int8)
            pat = np.zeros((nb, 1 << j, len(x0s), dim), dtype=np.int8)
            for xi, x0 in enumerate(x0s):
                pat[:, :, xi, pos ^ x0] = codes
            pat = canon(pat.reshape(-1, dim))
            yield start, pat
            start += len(pat)


def canon(P):
    """Canonical phase-code rows: rotate so the first nonzero entry is 1."""
    nz = P > 0
    first = np.argmax(nz, axis=1)
    ref = P[np.arange(len(P)), first] - 1
    Q = np.where(nz, (P - 1 - ref[:, None]) % 4 + 1, 0).astype(np.int8)
    return Q


def apply_hadamard(P, k, n):
    """H on qubit k, exactly, on phase-code rows; the result must again be a
    stabilizer pattern (it is, up to the overall 1/sqrt 2), returned canonical."""
    dim = 1 << n
    idx = np.arange(dim)
    lo = idx[(idx >> k) & 1 == 0]
    hi = lo | (1 << k)
    # complex integer arithmetic on small ints: codes -> (re, im)
    re = np.array([0, 1, 0, -1, 0], dtype=np.int8)[P]
    im = np.array([0, 0, 1, 0, -1], dtype=np.int8)[P]
    re2 = np.empty_like(re)
    im2 = np.empty_like(im)
    re2[:, lo] = re[:, lo] + re[:, hi]
    im2[:, lo] = im[:, lo] + im[:, hi]
    re2[:, hi] = re[:, lo] - re[:, hi]
    im2[:, hi] = im[:, lo] - im[:, hi]
    # entries are in {0, +-1, +-i, +-2, +-2i, +-1+-i}; a stabilizer output has one modulus
    mod2 = re2.astype(np.int64) ** 2 + im2.astype(np.int64) ** 2
    mx = mod2.max(axis=1)
    assert np.all((mod2 == 0) | (mod2 == mx[:, None])), "Hadamard image is not a stabilizer pattern"
    out = np.zeros_like(P)
    for code, (r, i) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)], start=1):
        out[(re2 == r) & (im2 == i)] = code
    # modulus 2 case: entries +-2, +-2i
    for code, (r, i) in enumerate([(2, 0), (0, 2), (-2, 0), (0, -2)], start=1):
        out[(re2 == r) & (im2 == i)] = code
    # modulus sqrt 2 case: entries (+-1 +- i); phase of 1+i is pi/4: divide by (1+i)
    m = (np.abs(re2) == 1) & (np.abs(im2) == 1)
    if np.any(m):
        # (r + i s)/(1 + i) = ((r + s) + i (s - r))/2
        rr = (re2[m].astype(np.int16) + im2[m]) // 2
        ii = (im2[m].astype(np.int16) - re2[m]) // 2
        code = np.zeros(rr.shape, dtype=np.int8)
        for c, (r, i) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)], start=1):
            code[(rr == r) & (ii == i)] = c
        out[m] = code
    return canon(out)


def apply_perm(P, perm, n):
    """Permute qubits: new qubit k carries old qubit perm[k]."""
    dim = 1 << n
    src = np.arange(dim)
    dst = np.zeros(dim, dtype=np.int64)
    for k in range(n):
        dst |= ((src >> perm[k]) & 1) << k
    out = np.empty_like(P)
    out[:, dst] = P
    return canon(out)


HASH_SEED = 0x5EED6


def _hash(P, r1, r2):
    Q = P.astype(np.uint64)
    return (Q * r1[None, :]).sum(axis=1), (Q * r2[None, :]).sum(axis=1)


def orbit_count(keep, n=N_QUBITS, total=None, verbose=True):
    """Orbits of the unitary symmetry group of |H>^n (Hadamards on any
    subset of qubits, permutations) on the P* states, by union-find over
    exact keys: every state is hashed to 128 bits, the hashes are checked
    to be distinct (so the map state -> hash is injective on the set, and
    a generator image is identified exactly), and every generator is
    checked to map the set into itself."""
    t0 = time.time()
    rng = np.random.default_rng(HASH_SEED)
    dim = 1 << n
    r1 = rng.integers(1, 2 ** 63, size=dim, dtype=np.uint64) | np.uint64(1)
    r2 = rng.integers(1, 2 ** 63, size=dim, dtype=np.uint64) | np.uint64(1)
    N = total
    h1 = np.empty(N, dtype=np.uint64)
    h2 = np.empty(N, dtype=np.uint64)
    for start, pat in pattern_batches(keep, n):
        a, b = _hash(pat, r1, r2)
        h1[start:start + len(pat)] = a
        h2[start:start + len(pat)] = b
    assert start + len(pat) == N
    order = np.lexsort((h2, h1))
    s1, s2 = h1[order], h2[order]
    dup = (s1[1:] == s1[:-1]) & (s2[1:] == s2[:-1])
    assert not np.any(dup), "hash collision among P* states; change HASH_SEED"
    if verbose:
        print(f"  {N} P* states hashed, all distinct [{time.time() - t0:.1f}s]", flush=True)

    def lookup(Q):
        a, b = _hash(Q, r1, r2)
        lo = np.searchsorted(s1, a, side="left")
        hi = np.searchsorted(s1, a, side="right")
        # within equal-h1 runs (almost always length 1) match h2
        out = np.full(len(a), -1, dtype=np.int64)
        for k in range(int((hi - lo).max())):
            cand = lo + k
            ok = (cand < hi)
            ok[ok] &= s2[cand[ok]] == b[ok]
            out[ok] = order[cand[ok]]
        assert np.all(out >= 0), "a symmetry image is not a P* state"
        return out

    perms = [(1, 0) + tuple(range(2, n)), tuple(range(1, n)) + (0,)]
    ngen = n + len(perms)
    gens = np.empty((ngen, N), dtype=np.int32)
    for start, pat in pattern_batches(keep, n):
        sl = slice(start, start + len(pat))
        for k in range(n):
            gens[k, sl] = lookup(apply_hadamard(pat, k, n))
        for t, perm in enumerate(perms):
            gens[n + t, sl] = lookup(apply_perm(pat, perm, n))
    if verbose:
        print(f"  generator images computed [{time.time() - t0:.1f}s]", flush=True)
    labels = np.arange(N, dtype=np.int32)
    while True:
        prev = labels
        for k in range(ngen):
            labels = np.minimum(labels, labels[gens[k]])
        for _ in range(3):
            labels = labels[labels]
        if np.array_equal(labels, prev):
            break
    roots, sizes = np.unique(labels, return_counts=True)
    if verbose:
        hist = {}
        for s in sizes.tolist():
            hist[s] = hist.get(s, 0) + 1
        print(f"  orbits: {len(roots)}; orbit size -> number of orbits: {dict(sorted(hist.items()))} "
              f"[{time.time() - t0:.1f}s]", flush=True)
    return len(roots), sizes


def main(argv):
    t0 = time.time()
    keep, c = counts()
    print(f"[{time.time() - t0:.1f}s]")
    if "--orbits" in argv:
        orbit_count(keep, total=c["Pstar"])
        print(f"[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main(sys.argv[1:])
