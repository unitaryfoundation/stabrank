"""Rank-r exclusion for the compressed cat state c_m = cat_m(y, parity(y)) on
n = m - 1 qubits by a two-qubit base slice, exact (screens mod P, every
candidate re-decided in Gaussian-rational arithmetic).

Slice c_m along its first two qubits: c_m(x, y) for x in F_2^2, y in F_2^{n-2}.
A rank-r decomposition c_m = sum_j e_j s_j restricts at the base point x = 00
to sum_j e_j a_j = c_m(00, .) = c_{m-2} with a_j = (<00| (x) I) s_j, so
k >= chi(c_{m-2}) terms are visible there. Stages, for r = chi(c_{m-2}) + 1:
  A     k = r, independent base states (full r-covers of c_{m-2}),
  B     k = r, dependent base states (a rank-(r-1) decomposition plus a state
        of its span; the coefficients form a one-parameter family),
  C     k = r, a repeated base state (a rank-(r-1) decomposition with one state
        doubled; the two copies share the base coefficient),
  beta  k = r - 1, a rank-(r-1) decomposition, the last term invisible at 00
        and nonzero on one of the six flats of F_2^2 that miss 00.
At the other three points every visible term is zero or i^l Q a_j with Q one of
2^{n-2} Pauli classes (qubit slice structure lemma, verify_challenge/slice_cover.py
and slice_lift.py), and the point 11 slice of a term present at 01 and 10 is a
sign times the product of its two class representatives (we allow any fourth
root, a superset). Each point is solved by meet in the middle on two random
functionals mod P, every hit is re-decided exactly, and the points are combined
under the flat rule. A surviving candidate is confirmed as a decomposition of
c_m exactly and its terms checked to be stabilizer states.

Reuses the exact dictionary, symmetry, hashing and solving code of the board's
verify_challenge/cert_cat_m7_lift.py (PR 170).
"""
import itertools, os, sys, time, json
from fractions import Fraction
import numpy as np

def _find_root():
    """the public repository root: STABRANK_PUB, else the nearest ancestor of
    this file holding verify_challenge/stabrank_verify.py, else the worktree."""
    env = os.environ.get("STABRANK_PUB")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "verify_challenge", "stabrank_verify.py")):
            return d
        d = os.path.dirname(d)
    return "/Users/vincent.russo/Projects/genesis/stabrank-r31j"

PUB = _find_root()
sys.path.insert(0, os.path.join(PUB, "verify_challenge"))
import cert_cat_m7_lift as L  # noqa: E402
P, RI, POW_I = L.P, L.RI, L.POW_I

# ----------------------------------------------------------- targets ------

def slices(m):
    """the four two-qubit slices of c_m as exponent rows (x = 2 x1 + x2)."""
    t = L.c_exponents(m)
    q = 1 << (m - 3)
    return [t[x * q:(x + 1) * q] for x in range(4)]


def base_symmetry(nb):
    """generators acting on the nb unsliced qubits: transposition, cycle, R(x)R."""
    gens = []
    if nb >= 2:
        for perm in ([1, 0] + list(range(2, nb)), list(range(1, nb)) + [0]):
            src = L.perm_index_map(nb, perm)
            gens.append(("perm", src, src))
        gens.append(("rr", None, None))
    return gens


def check_symmetry_slices(gens, m):
    """I (x) g must fix every slice of c_m up to one common scalar."""
    nb = m - 3
    sl = slices(m)
    for g in gens:
        nus = []
        for s in sl:
            gs = L.act(g, s[None, :], nb)
            if (gs < 0).any():
                L.fail("generator moves a slice off its support")
            d = (gs[0] - s) % 4
            if len(set(d.tolist())) != 1:
                L.fail("generator does not fix a slice up to scalar")
            nus.append(int(d[0]))
        if len(set(nus)) != 1:
            L.fail(f"generator {g[0]} fixes the slices with different scalars {nus}")


# ------------------------------------------------- exact helpers ----------

def exact_row(e_row):
    return [(Fraction(a), Fraction(b)) for a, b in L.to_gauss(e_row)]


def frac_mod(x):
    """Gaussian rational (Fraction, Fraction) -> F_P."""
    for c in x:
        if c.denominator % P == 0:
            L.fail("denominator divisible by P")
    re = x[0].numerator * pow(x[0].denominator, P - 2, P) % P
    im = x[1].numerator * pow(x[1].denominator, P - 2, P) % P
    return (re + im * RI) % P


def gsub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def gscale(c, v):
    return [(c[0] * x[0] - c[1] * x[1], c[0] * x[1] + c[1] * x[0]) for x in v]


def parallel_exact(u, v):
    """u = lam v exactly? returns lam or None (v != 0 required)."""
    j = next((i for i, x in enumerate(v) if x != (0, 0)), None)
    if j is None:
        return None
    lam = L.gdiv(u[j], v[j])
    return lam if gscale(lam, v) == list(u) else None


def first_nonzero_col(M):
    nz = M != 0
    return np.argmax(nz, axis=1), nz.any(axis=1)


# ------------------------------------------- census of full 4-covers ------

def census4(E, M, tb, roots, pivots=None, log_every=10, fsum=None):
    """Full 4-covers of the base target tb (exponent row) over the dictionary
    E/M: 4-sets with tb in their span, all four coefficients nonzero, up to
    the symmetry whose orbit roots are `roots`. Returns (A, B) where A are
    the independent 4-sets with their exact coefficients and B the dependent
    4-sets (rank 3) found on the way; exact by the norm bounds of
    cert_cat_m7_lift.assert_norms extended one level (|q3| <= 128, minors
    <= 2 * 128^2 < P)."""
    N, dim = M.shape
    tbm = POW_I[tb]
    j0 = int(np.flatnonzero(tb >= 0)[0]) if (tb < 0).any() else 0
    inv_t0 = POW_I[(-int(tb[j0])) % 4]
    Q1 = (M - M[:, j0:j0 + 1] * (tbm[None, :] * inv_t0 % P)) % P
    rng = np.random.default_rng(2026)
    f, g = rng.integers(1, P, size=dim), rng.integers(1, P, size=dim)
    order_r = np.argsort(roots, kind="stable")
    sroots = roots[order_r]
    allp = np.unique(roots) if pivots is None else np.asarray(pivots)
    A, B = {}, {}
    t0 = time.time()
    npairs = 0
    for c, s1 in enumerate(allp):
        start = int(np.searchsorted(sroots, roots[s1]))
        sub = order_r[start:]
        if int(sub[0]) != int(s1):
            # the pivot is not the first of its root block (pivot subset mode)
            pos = int(np.flatnonzero(sub == s1)[0])
            sub = np.concatenate([[s1], sub[:pos], sub[pos + 1:]])
        q1 = Q1[s1]
        nzq = np.flatnonzero(q1)
        if nzq.size == 0:
            L.fail("the base target is a stabilizer state")
        k = int(nzq[0])
        Q2 = (q1[k] * Q1[sub] - Q1[sub, k:k + 1] * q1[None, :]) % P
        F2, G2 = ((Q2 * f) % P).sum(axis=1) % P, ((Q2 * g) % P).sum(axis=1) % P   # reduce before the sum (referee, 2026-10-17)
        kcol, has2 = first_nonzero_col(Q2)
        n = len(sub)
        for p in range(1, n):
            if not has2[p]:
                continue          # s2 in span(tb, s1): a rank-2 relation, handled elsewhere
            kk = int(kcol[p])
            Acoef = Q2[p, kk]
            pool = np.arange(p + 1, n)
            if pool.size < 2:
                break
            Bcol = Q2[pool, kk]
            fk = (Acoef * F2[pool] - Bcol * F2[p]) % P
            gk = (Acoef * G2[pool] - Bcol * G2[p]) % P
            npairs += 1
            key = np.where(gk != 0, fk * L.inv_mod(np.where(gk != 0, gk, 1)) % P,
                           np.where(fk != 0, P, P + 1))
            order = np.argsort(key, kind="stable")
            sk = key[order]
            same = np.flatnonzero(sk[1:] == sk[:-1])
            if same.size == 0:
                continue
            groups = {}
            for a in same:
                groups.setdefault(int(sk[a]), set()).update((int(order[a]), int(order[a + 1])))
            for kv, gset in groups.items():
                idx = sorted(gset)
                rows = (Acoef * Q2[pool[idx]] - Q2[pool[idx], kk:kk + 1] * Q2[p][None, :]) % P
                if kv == P + 1:
                    # both functionals vanish: either q3 = 0 (state in span(tb, s1, s2)) or a coincidence
                    zero = ~(rows != 0).any(axis=1)
                    for li in np.flatnonzero(zero):
                        B.setdefault(tuple(sorted((int(s1), int(sub[p]), int(sub[pool[idx[li]]])))), None)
                    idx = [i for i, z in zip(idx, zero) if not z]
                    rows = rows[~zero]
                    if len(idx) < 2:
                        continue
                W, _ = L.proj_normal(rows)
                for a_, b_ in itertools.combinations(range(len(idx)), 2):
                    if not np.array_equal(W[a_], W[b_]):
                        continue
                    quad = tuple(sorted((int(s1), int(sub[p]), int(sub[pool[idx[a_]]]), int(sub[pool[idx[b_]]]))))
                    A.setdefault(quad, None)
        if log_every and (c + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  census4: pivot {c + 1}/{len(allp)} pairs {npairs} cands {len(A)} dep {len(B)} "
                  f"{el:.0f}s, ~{el / (c + 1) * (len(allp) - c - 1):.0f}s left", flush=True)
    return A, B


def classify_covers(cands, E, tb):
    """exact: keep 4-sets with tb in the span and all coefficients nonzero;
    split by rank (4 -> A with point coefficients, 3 -> B)."""
    tg = exact_row(tb)
    A, B = [], []
    for quad in cands:
        cols = [exact_row(E[i]) for i in quad]
        d = L.solve_exact(cols, tg)
        if d is None:
            # dependent set: tb may still be in the span (solve_exact fails on a dependent system)
            r = rank_exact(cols)
            if r == 3 and L.solve_exact(cols[:3], tg) is not None or in_span_exact(cols, tg):
                B.append(quad)
            continue
        if all(x != (0, 0) for x in d):
            A.append((quad, d))
        # a zero coefficient: tb in the span of three -> a rank-3 decomposition plus a state; stage B covers it
    return A, B


def rank_exact(cols):
    A = [list(c) for c in cols]
    n = len(A); m = len(A[0]); r = 0; rows = [[A[j][i] for j in range(n)] for i in range(m)]
    for cj in range(n):
        p = next((i for i in range(r, m) if rows[i][cj] != (0, 0)), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        pv = rows[r][cj]
        rows[r] = [L.gdiv(x, pv) for x in rows[r]]
        for i in range(m):
            if i != r and rows[i][cj] != (0, 0):
                fct = rows[i][cj]
                rows[i] = [(x[0] - (fct[0] * y[0] - fct[1] * y[1]), x[1] - (fct[0] * y[1] + fct[1] * y[0]))
                           for x, y in zip(rows[i], rows[r])]
        r += 1
    return r


def in_span_exact(cols, t):
    return rank_exact(cols) == rank_exact(cols + [t])
