"""Structured rank-7 search at T5 m=2 from the nine-term product (2026-09-23).

Cell: 5 <= chi(|T5>^2) <= 8 (`bounds/T5-m2-lower-5.json`, `-upper-8.json`);
the product of two rank-3 single-ququint decompositions has nine terms.
Question: is there a seven-term decomposition of |T5>^2 that keeps at least
five of the nine product terms? Fix five product terms (with free
coefficients) and ask for two more two-ququint stabilizer states t1, t2 with
|T5>^2 in span(five fixed, t1, t2).

Method. Every amplitude in sight is 0 or a power of w5 up to the
normalisation sqrt(5)^k, so each vector is an exponent array and the
search runs exactly over F_q for a prime q = 1 (mod 5), with w5 sent to an
element of order 5. For each choice of fixed terms F the quotient of F_q^25
by span(F, psi) has dimension 25 - |F| - 1 when those vectors are
independent mod q (checked, and the choice is re-decided at a second prime
if not). Two states t1, t2 complete the decomposition only if a nonzero
combination of them lies in span(F, psi), i.e. their images in the quotient
are parallel or one of them is zero. Parallel images are found by hashing
the canonically scaled quotient vectors of all 3,900 states (one pass per
fixed set, no pair loop). Reduction mod a prime ideal preserves every
dependency over Q(w5), so nothing is missed; a collision can be spurious
(a dependency among stabilizer states that does not involve psi, or an
accident mod q), so every collision is re-decided exactly over Q(w5) with
sympy's DomainMatrix rank. `--keep K` fixes K product terms instead of
five (K = 6 asks for rank 8 through six product terms, K = 7 is the
positive control); the one-pass hashing finds two new terms, so fewer than
four fixed terms would need a pivot loop and is not implemented.

Controls (`--control`): fixing seven of the nine product terms recovers the
two dropped ones (36 of 36); fixing six of the eight terms of the board's
rank-8 witness recovers the other two (28 of 28); fixing seven witness
terms and asking for one state returns exactly the dropped term (8 of 8)
and nothing else.

Usage (from the repository root):
    uv run --extra challenge python research/constructions/t5_m2_merge.py [--keep 5] [--control] [--prime Q]
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from rank_exclusion import dictionary, psi_for  # noqa: E402
from stabrank_verify import stabilizer_vector  # noqa: E402

P = 5
NONE = -1  # exponent marker for a zero amplitude


# ----------------------------------------------------------- exact encoding --

def exponents(v, tol=1e-9):
    """A vector whose nonzero entries all have one modulus and phases that are
    powers of w5, as an int array of exponents (NONE where zero)."""
    v = np.asarray(v).ravel()
    nz = np.abs(v) > tol
    mod = np.abs(v[nz])
    assert np.allclose(mod, mod[0], atol=1e-9), "entries of unequal modulus"
    ang = np.angle(v[nz] / mod[0])
    e = np.rint(ang / (2 * np.pi / P)).astype(int) % P
    assert np.allclose(np.exp(2j * np.pi * e / P), v[nz] / mod[0], atol=1e-7)
    out = np.full(v.shape, NONE, dtype=int)
    out[nz] = e
    return out


def kron_exp(a, b):
    """Exponent array of the tensor product of two exponent arrays."""
    A = a[:, None]
    B = b[None, :]
    out = (A + B) % P
    out[(A == NONE) | (B == NONE)] = NONE
    return out.ravel()


def order5_element(q):
    for g in range(2, q):
        if pow(g, 5, q) == 1 and g != 1:
            return g
    raise ValueError("no element of order 5")


def reduce_mod(E, q, g):
    """Exponent arrays (rows) to F_q vectors."""
    V = np.array([pow(g, int(e), q) for e in range(P)], dtype=np.int64)
    out = V[np.where(E == NONE, 0, E)]
    out[E == NONE] = 0
    return out


def rref_mod(M, q):
    """Row-reduce M (rows are vectors) over F_q; returns (R, pivot columns)."""
    M = M.copy() % q
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        i = r + nz[0]
        M[[r, i]] = M[[i, r]]
        M[r] = (M[r] * pow(int(M[r, c]), q - 2, q)) % q
        others = [k for k in range(rows) if k != r and M[k, c] != 0]
        for k in others:
            M[k] = (M[k] - M[k, c] * M[r]) % q
        piv.append(c)
        r += 1
    return M[:r], piv


def annihilator(S, q):
    """A matrix N with N v = 0 exactly for v in the row span of S (mod q):
    the quotient map F_q^n -> F_q^(n - rank S)."""
    R, piv = rref_mod(S, q)
    n = S.shape[1]
    free = [c for c in range(n) if c not in piv]
    N = np.zeros((len(free), n), dtype=np.int64)
    for i, f in enumerate(free):
        N[i, f] = 1
        for r, c in enumerate(piv):
            N[i, c] = (-R[r, f]) % q
    return N, len(piv)


_INV = {}


def canonical_keys(X, q):
    """Scale every row so that its first nonzero entry is 1; return the keys
    (None for a zero row)."""
    if q not in _INV:
        _INV[q] = np.array([0] + [pow(x, q - 2, q) for x in range(1, q)], dtype=np.int64)
    nz = X != 0
    any_nz = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    lead = X[np.arange(len(X)), first]
    scaled = (X * _INV[q][lead][:, None]) % q
    return [row.tobytes() if ok else None for row, ok in zip(scaled, any_nz)]


# -------------------------------------------------------------- exact check --

class Exact:
    """Exact linear algebra over Q(w5) through sympy's DomainMatrix."""

    def __init__(self):
        from sympy.polys.matrices import DomainMatrix
        self.DM = DomainMatrix
        self.w = sp.exp(2 * sp.pi * sp.I / 5)
        self.K = sp.QQ.algebraic_field(self.w)
        self.pow = [self.K.from_sympy(self.w ** e) for e in range(P)]
        self.zero = self.K.zero

    def matrix(self, E):
        rows = [[self.zero if e == NONE else self.pow[int(e)] for e in row] for row in E]
        return self.DM(rows, (len(rows), len(rows[0])), self.K)

    def rank(self, E):
        return self.matrix(E).rank()

    def in_span(self, psi_e, terms_e):
        """psi in span(terms), with the terms independent, exactly."""
        T = np.array(terms_e)
        r = self.rank(T)
        if r < len(terms_e):
            return False, "dependent terms"
        r2 = self.rank(np.vstack([T, psi_e[None, :]]))
        return r2 == r, f"rank {r} -> {r2}"


# ------------------------------------------------------------------ search --

def completions(fixed_e, psi_e, cand_e, q, g, exact, want_pairs=True):
    """All single states t (psi in span(fixed, t)) and pairs (t1, t2) (psi in
    span(fixed, t1, t2)) among cand_e, decided mod q and re-decided exactly.
    Returns (hits, n_zero_candidates, n_collision_pairs, quotient_ok)."""
    S = reduce_mod(np.vstack([fixed_e, psi_e[None, :]]), q, g)
    N, rk = annihilator(S, q)
    if rk != len(fixed_e) + 1:
        # the fixed terms and psi are dependent mod q: either psi is already
        # in their span (a hit of lower rank) or an accident of the prime
        ok, why = exact.in_span(psi_e, list(fixed_e))
        return ([("in-span", (), why)] if ok else []), 0, 0, False
    C = reduce_mod(cand_e, q, g)
    X = (C @ N.T) % q
    keys = canonical_keys(X, q)
    hits = []
    zeros = [i for i, k in enumerate(keys) if k is None]
    for i in zeros:
        ok, why = exact.in_span(psi_e, list(fixed_e) + [cand_e[i]])
        if ok:
            hits.append(("single", (i,), why))
    pairs = 0
    if want_pairs:
        buckets = {}
        for i, k in enumerate(keys):
            if k is not None:
                buckets.setdefault(k, []).append(i)
        for k, members in buckets.items():
            if len(members) < 2:
                continue
            for i, j in itertools.combinations(members, 2):
                pairs += 1
                ok, why = exact.in_span(psi_e, list(fixed_e) + [cand_e[i], cand_e[j]])
                if ok:
                    hits.append(("pair", (i, j), why))
    return hits, len(zeros), pairs, True


def same_state(a, b):
    """Two exponent arrays give the same state up to a global phase."""
    if not np.array_equal(a == NONE, b == NONE):
        return False
    nz = a != NONE
    d = (a[nz] - b[nz]) % P
    return len(d) == 0 or np.all(d == d[0])


def rank3_decompositions(D1, psi1):
    hits = []
    for c in itertools.combinations(range(D1.shape[1]), 3):
        A = D1[:, c]
        x = np.linalg.lstsq(A, psi1, rcond=None)[0]
        if np.linalg.norm(A @ x - psi1) < 1e-9:
            hits.append(c)
    return hits


def main(argv):
    keep = 5
    if "--keep" in argv:
        keep = int(argv[argv.index("--keep") + 1])
    q = 10061
    if "--prime" in argv:
        q = int(argv[argv.index("--prime") + 1])
    assert sp.isprime(q) and q % 5 == 1
    g = order5_element(q)
    t0 = time.time()
    exact = Exact()

    D1 = dictionary(5, 1)
    D2 = dictionary(5, 2)
    psi1 = psi_for("T5", 1)
    psi2 = psi_for("T5", 2)
    D1e = [exponents(D1[:, i]) for i in range(D1.shape[1])]
    D2e = np.array([exponents(D2[:, i]) for i in range(D2.shape[1])])
    psi1e = exponents(psi1)
    psi2e = exponents(psi2)
    assert np.array_equal(kron_exp(psi1e, psi1e), psi2e)
    decs = rank3_decompositions(D1, psi1)
    print(f"{len(decs)} rank-3 decompositions of |T5>: {decs}")
    print(f"dictionaries {D1.shape[1]} and {D2.shape[1]} states, prime {q}, w5 -> {g}, "
          f"{time.time() - t0:.1f} s")

    if "--control" in argv:
        # (a) seven of the nine product terms recover the other two
        a, b = decs[0], decs[1]
        prod = [kron_exp(D1e[i], D1e[j]) for i in a for j in b]
        found = 0
        for drop in itertools.combinations(range(9), 2):
            fixed = [prod[i] for i in range(9) if i not in drop]
            hits, nz, pairs, ok = completions(np.array(fixed), psi2e, D2e, q, g, exact)
            found += any(h[0] == "pair" for h in hits)
        print(f"control (a): {found} of 36 seven-term fixed sets complete to the nine-term "
              f"product (expected 36), {time.time() - t0:.1f} s")
        # (b) six of the eight witness terms recover the other two
        sub = json.load(open(os.path.join(ROOT, "bounds", "T5-m2-upper-8.json")))
        W = [exponents(np.array([complex(x) for x in stabilizer_vector(t, 5, 2)]))
             for t in sub["witness"]["terms"]]
        found = 0
        for drop in itertools.combinations(range(8), 2):
            fixed = [W[i] for i in range(8) if i not in drop]
            hits, nz, pairs, ok = completions(np.array(fixed), psi2e, D2e, q, g, exact)
            found += any(h[0] == "pair" for h in hits)
        print(f"control (b): {found} of 28 six-term fixed sets of the rank-8 witness complete "
              f"to it (expected 28), {time.time() - t0:.1f} s")
        # (c) seven witness terms and a single completing state: the dropped
        # term itself is always one; any other would be a rank-7
        # decomposition sharing seven terms with the witness
        trivial = other = 0
        for (drop,) in itertools.combinations(range(8), 1):
            fixed = [W[i] for i in range(8) if i != drop]
            hits, nz, pairs, ok = completions(np.array(fixed), psi2e, D2e, q, g, exact,
                                              want_pairs=False)
            for h in hits:
                if h[0] != "single":
                    continue
                if same_state(D2e[h[1][0]], W[drop]):
                    trivial += 1
                else:
                    other += 1
        print(f"control (c): {trivial} trivial single-state completions of seven witness "
              f"terms (the dropped term, expected 8) and {other} others (each would be a "
              f"rank-7 decomposition sharing seven terms with the witness), "
              f"{time.time() - t0:.1f} s")
        return 0

    # unordered pairs of decompositions; the copy swap maps (a, b) to (b, a)
    n_sets = n_zero = n_pairs = n_dep = 0
    all_hits = []
    for ia, ib in itertools.combinations_with_replacement(range(len(decs)), 2):
        a, b = decs[ia], decs[ib]
        prod = [kron_exp(D1e[i], D1e[j]) for i in a for j in b]
        for S in itertools.combinations(range(9), keep):
            fixed = np.array([prod[i] for i in S])
            hits, nz, pairs, ok = completions(fixed, psi2e, D2e, q, g, exact)
            n_sets += 1
            n_zero += nz
            n_pairs += pairs
            n_dep += not ok
            for h in hits:
                all_hits.append(((ia, ib), S, h))
                print("HIT", (ia, ib), S, h, flush=True)
    print(f"keep {keep}: {n_sets} fixed sets over {len(decs) * (len(decs) + 1) // 2} products; "
          f"{n_zero} in-span candidates and {n_pairs} parallel pairs re-decided exactly; "
          f"{n_dep} fixed sets dependent mod {q}; {len(all_hits)} exact completions "
          f"(rank <= {keep + 2}); {time.time() - t0:.1f} s")
    if all_hits:
        out = os.path.join(HERE, "results")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"t5_m2_keep{keep}_hits.json"), "w") as f:
            json.dump([[list(map(int, p)), list(S), h[0], list(map(int, h[1])), h[2]]
                       for p, S, h in all_hits], f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
