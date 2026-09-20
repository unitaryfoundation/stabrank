"""Shared helpers for the merge-recipe scripts.

The recipes start from a product decomposition (terms s_i (x) t_j with
coefficients a_i b_j) and ask whether fewer stabilizer states span the same
target. Everything here is numeric (complex128) and is used to screen; a
decomposition becomes a bound only through verify_challenge/to_witness.py and
fit_coeffs.py, which are exact.

The one nontrivial tool is `stabilizer_states_in_span`, an exact and complete
list of the stabilizer states inside a given subspace V of C^(p^n), found
flat by flat without a dictionary. A stabilizer state supported on the
affine flat F lies in V exactly when some vector of V vanishes off F and has
unit modulus and a quadratic phase on F. Vectors of V vanishing off F form
the null space of the rows of V outside F; with V orthonormal that null
space is the eigenvalue-1 eigenspace of V_F^H V_F, one small Hermitian
matrix per flat, batched over all flats of one shape. On the (rare) flats
with a nonzero null space of dimension d, the state's phases at d
independent points of F determine it, so the p^(d-1) (4^(d-1) for qubits)
phase patterns are enumerated and each candidate is confirmed by
to_witness.term_from_vector.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "verify_challenge"), os.path.join(ROOT, "autoresearch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "constructions_common", os.path.join(ROOT, "research", "constructions", "common.py"))
_cc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cc)
alpha, target, term_vector, load_decompositions = _cc.alpha, _cc.target, _cc.term_vector, _cc.load_decompositions
from stabrank_verify import ORBIT_P  # noqa: E402
from to_witness import term_from_vector, NotStabilizer  # noqa: E402
from run import sector_target  # noqa: E402  (autoresearch/run.py)

__all__ = ["alpha", "target", "term_vector", "load_decompositions", "ORBIT_P", "sector_target",
           "term_from_vector", "NotStabilizer", "is_stabilizer", "flats", "stabilizer_states_in_span",
           "coefficient_classes", "write_warm_file", "product_terms", "phase_group",
           "modulus_pattern", "rank2_modulus_obstruction", "rank2_excluded", "kron_all"]

HERE = os.path.dirname(os.path.abspath(__file__))
WARM = os.path.join(HERE, "warm")


def kron_all(vs):
    out = np.array([1.0 + 0j])
    for v in vs:
        out = np.kron(out, v)
    return out


def phase_group(p):
    return 4 if p == 2 else p


def is_stabilizer(v, p, n, tol=1e-7):
    """The witness term of v if v is (a multiple of) a stabilizer state, else None."""
    try:
        return term_from_vector(v, p, n, tol=tol)
    except NotStabilizer:
        return None
    except (ValueError, IndexError):
        return None


def product_terms(decA, decB):
    """Terms and coefficients of the product of two decompositions
    ((vectors, coeffs) pairs as load_decompositions returns them)."""
    sA, cA = decA
    sB, cB = decB
    terms = [np.kron(s, t) for s in sA for t in sB]
    coeffs = np.array([x * y for x in cA for y in cB])
    return terms, coeffs


def coefficient_classes(coeffs, tol=1e-9):
    """Indices grouped by equal coefficient (as complex numbers), largest first."""
    classes = []
    for i, c in enumerate(coeffs):
        for cl in classes:
            if abs(coeffs[cl[0]] - c) < tol * max(1.0, abs(c)):
                cl.append(i)
                break
        else:
            classes.append([i])
    return sorted(classes, key=lambda cl: (-len(cl), cl))


# ---------------------------------------------------------------- flats


def _rref_batch(p, n, pivots):
    """Every k x n reduced row echelon matrix over F_p with the given pivot
    columns, as an integer array of shape (B, k, n)."""
    k = len(pivots)
    free = [(r, j) for r in range(k) for j in range(pivots[r] + 1, n) if j not in pivots]
    B = p ** len(free)
    W = np.zeros((B, k, n), dtype=np.int64)
    for r, pc in enumerate(pivots):
        W[:, r, pc] = 1
    if free:
        vals = np.array(list(itertools.product(range(p), repeat=len(free))), dtype=np.int64)
        for t, (r, j) in enumerate(free):
            W[:, r, j] = vals[:, t]
    return W


def flats(p, n, dims=None):
    """Yield (k, idx) with idx an integer array of shape (Nf, p^k): the point
    indices (most significant digit first) of every affine k-flat of F_p^n
    with one fixed pivot set, one group per pivot set. Every affine flat
    appears exactly once over the whole iteration.

    A flat is x0 + row space of W with W in reduced row echelon form and x0
    zero on the pivot columns; that representation is unique."""
    weights = p ** (n - 1 - np.arange(n))
    for k in range(0, n + 1):
        if dims is not None and k not in dims:
            continue
        ys = (np.array(list(itertools.product(range(p), repeat=k)), dtype=np.int64)
              if k else np.zeros((1, 0), dtype=np.int64))
        for pivots in itertools.combinations(range(n), k):
            W = _rref_batch(p, n, pivots)                                   # (B, k, n)
            nonpiv = [j for j in range(n) if j not in pivots]
            x0 = np.zeros((p ** (n - k), n), dtype=np.int64)
            if nonpiv:
                x0[:, nonpiv] = np.array(list(itertools.product(range(p), repeat=len(nonpiv))),
                                         dtype=np.int64)
            pts = np.einsum("yk,bkn->byn", ys, W)                            # (B, p^k, n)
            pts = (pts[:, None, :, :] + x0[None, :, None, :]) % p            # (B, C, p^k, n)
            idx = pts @ weights                                              # (B, C, p^k)
            yield k, idx.reshape(-1, p ** k)


def _independent_rows(A, d):
    """Indices of d rows of A (s x d, rank d) that are linearly independent."""
    _, _, piv = _qr_pivot(A.T)
    return piv[:d]


def _qr_pivot(M):
    """Column-pivoted QR via scipy if available, else greedy Gram-Schmidt."""
    try:
        import scipy.linalg
        return scipy.linalg.qr(M, pivoting=True)
    except Exception:  # pragma: no cover
        cols = []
        basis = np.zeros((M.shape[0], 0), dtype=complex)
        remaining = list(range(M.shape[1]))
        while remaining and len(cols) < M.shape[0]:
            res = M[:, remaining] - basis @ (basis.conj().T @ M[:, remaining])
            nrm = np.linalg.norm(res, axis=0)
            j = int(np.argmax(nrm))
            if nrm[j] < 1e-10:
                break
            cols.append(remaining[j])
            basis = np.column_stack((basis, res[:, j] / nrm[j]))
            remaining.pop(j)
        return None, None, np.array(cols + remaining)


def stabilizer_states_in_span(V, p, n, tol=1e-8, max_null_dim=None, chunk=40000, verbose=False,
                              dims=None):
    """Every stabilizer state (up to scalar) in the column span of V, as a
    list of (vector, witness term). Complete and exact up to the tolerance.

    `V` is any basis (columns) of the subspace. `max_null_dim` skips flats
    whose null space has more than that many dimensions (the phase
    enumeration is p^(d-1) or 4^(d-1) patterns); skipped flats are reported
    in the returned diagnostics rather than silently dropped."""
    Q, _ = np.linalg.qr(np.asarray(V, dtype=complex))
    c = Q.shape[1]
    dim = p ** n
    g = phase_group(p)
    roots = np.exp(2j * np.pi * np.arange(g) / g)
    found = []
    diag = {"flats": 0, "flagged": 0, "skipped": [], "patterns": 0}
    for k, idx_all in flats(p, n, dims=dims):
        s = p ** k
        for start in range(0, idx_all.shape[0], max(1, chunk // max(s, 1))):
            idx = idx_all[start:start + max(1, chunk // max(s, 1))]
            diag["flats"] += idx.shape[0]
            VF = Q[idx]                                                      # (Nf, s, c)
            M = np.einsum("fsa,fsb->fab", VF.conj(), VF)
            ev = np.linalg.eigvalsh(M)                                       # ascending
            nulld = (ev > 1 - tol).sum(axis=1)
            for f in np.flatnonzero(nulld):
                d = int(nulld[f])
                diag["flagged"] += 1
                if max_null_dim is not None and d > max_null_dim:
                    diag["skipped"].append((k, int(start + f), d))
                    continue
                w, U = np.linalg.eigh(M[f])
                N = U[:, w > 1 - tol]                                        # (c, d)
                A = VF[f] @ N                                                # (s, d): values on F
                rows = _independent_rows(A, d)
                Ainv = np.linalg.inv(A[rows])
                npat = g ** (d - 1)
                diag["patterns"] += npat
                # stage 1: enforce unit modulus on a few extra points of F
                extra = np.setdiff1d(np.arange(s), rows)[: max(2 * d, 4)]
                for pstart in range(0, npat, 1 << 18):
                    pidx = np.arange(pstart, min(npat, pstart + (1 << 18)))
                    digits = (pidx[:, None] // g ** np.arange(d - 1)[None, :]) % g
                    P = np.column_stack((np.ones(len(pidx)), roots[digits])) if d > 1 \
                        else np.ones((len(pidx), 1), dtype=complex)         # (npat, d)
                    a = Ainv @ P.T                                           # (d, npat)
                    if len(extra):
                        ue = A[extra] @ a
                        ok = np.all(np.abs(np.abs(ue) - 1) < 1e-6, axis=0)
                    else:
                        ok = np.ones(len(pidx), bool)
                    for t in np.flatnonzero(ok):
                        u = Q @ (N @ a[:, t])
                        u = u / u[idx[f][rows[0]]]
                        if np.any(np.abs(np.abs(u[idx[f]]) - 1) > 1e-6):
                            continue
                        term = is_stabilizer(u, p, n)
                        if term is not None and int(term["k"]) == k:
                            found.append((u / np.linalg.norm(u), term))
        if verbose:
            print(f"  dim {k}: {diag['flats']} flats so far, {diag['flagged']} flagged, "
                  f"{len(found)} states", flush=True)
    return found, diag


# ---------------------------------------------------------------- modulus patterns


def modulus_pattern(v, tol=1e-7):
    """Distinct nonzero moduli of v (sorted) and their multiplicities."""
    a = np.abs(v)
    a = a[a > tol]
    vals = []
    for x in np.sort(a):
        if not vals or abs(x - vals[-1][0]) > tol:
            vals.append([x, 1])
        else:
            vals[-1][1] += 1
    return vals


def rank2_modulus_obstruction(v, p, tol=1e-7):
    """A cheap exact necessary condition for v = a s_1 + b s_2 with s_1, s_2
    stabilizer states with distinct flats F_1, F_2 (the same flat is the
    stabilizer test on v itself). On F_1 minus F_2 and F_2 minus F_1 the
    modulus is constant; on the intersection it is |a' + b' w^j| for the
    phase-group element w^j, at most ceil((g+1)/2) values; the support is
    contained in F_1 union F_2. Returns None when no obstruction is seen and
    a string naming the obstruction otherwise."""
    g = phase_group(p)
    pat = modulus_pattern(v, tol)
    nvals = len(pat)
    maxvals = 2 + (g // 2 + 1)
    if nvals > maxvals:
        return f"{nvals} distinct moduli, a rank-2 sum has at most {maxvals}"
    supp = int((np.abs(v) > tol).sum())
    # support sizes of a union of two flats: |F1| + |F2| - |F1 cap F2| with the cap a flat or empty
    sizes = set()
    n = int(round(np.log(len(v)) / np.log(p)))
    for k1 in range(n + 1):
        for k2 in range(k1 + 1):
            for kc in [-1] + list(range(k2 + 1)):
                cap = 0 if kc < 0 else p ** kc
                for lost in range(0, cap + 1):                # cancellation on the intersection
                    sizes.add(p ** k1 + p ** k2 - cap - lost)
    if supp not in sizes:
        return f"support size {supp} is not |F1 u F2| minus cancellations for any two flats"
    return None


# ---------------------------------------------------------------- warm-start files


def write_warm_file(path, orbit, m, p, vectors, coeffs=None, note=""):
    """A bounds-style file whose `witness.terms` seed autoresearch/run.py
    (--warm-from). It is not a submission: coefficients are numeric strings
    and `provenance.method` says so."""
    terms = []
    for v in vectors:
        t = is_stabilizer(v, p, m)
        if t is None:
            raise NotStabilizer("a warm-start vector is not a stabilizer state")
        terms.append(t)
    rec = {"schema_version": "0.1", "orbit": orbit, "m": m, "direction": "upper",
           "rank": len(terms),
           "witness": {"terms": terms,
                       "coeffs": [repr(complex(c)) for c in coeffs] if coeffs is not None else []},
           "provenance": {"author": "this repository", "reference": "research/merges",
                          "method": "warm start for autoresearch/run.py --warm-from; not a submission"},
           "notes": note}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f)
        f.write("\n")
    return path


def _phase_candidates(x, m1, m2, thetas, tol=1e-6):
    """Values phi with |m1 + m2 e^{i(phi + theta_j)}| = x for some j, or None
    when no phi works."""
    c = (x ** 2 - m1 ** 2 - m2 ** 2) / (2 * m1 * m2)
    if abs(c) > 1 + tol:
        return None
    base = np.arccos(max(-1.0, min(1.0, c)))
    out = []
    for sgn in (1, -1):
        for th in thetas:
            out.append((sgn * base - th) % (2 * np.pi))
    return np.array(out)


def _consistent_phi(sets, tol=1e-6):
    """Is there one phi (mod 2 pi) within tol of a member of every set?"""
    if not sets:
        return True
    for phi in sets[0]:
        if all(np.min(np.abs((s - phi + np.pi) % (2 * np.pi) - np.pi)) < tol for s in sets[1:]):
            return True
    return False


def _same_flat_consistent(values, zeros, g, tol=1e-6):
    """Can the distinct moduli `values` (plus a zero when `zeros`) all be of
    the form |m_1 + m_2 e^{i(phi + theta_j)}|, j in Z_g, for one (m_1, m_2,
    phi)? Squaring gives x_j^2 = R + u cos theta_j - w sin theta_j with
    R = m_1^2 + m_2^2, (u, w) = C (cos phi, sin phi), C = 2 m_1 m_2, which is
    linear in (R, u, w); every injective assignment of the values to group
    elements is solved and accepted when the residual vanishes and R >= C."""
    vals = list(values) + ([0.0] if zeros else [])
    if len(vals) > g:
        return False
    if len(vals) <= 2:
        return True
    thetas = 2 * np.pi * np.arange(g) / g
    for js in itertools.permutations(range(g), len(vals)):
        M = np.column_stack((np.ones(len(vals)), np.cos(thetas[list(js)]), -np.sin(thetas[list(js)])))
        rhs = np.array(vals) ** 2
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        R, u, w = sol
        C = np.hypot(u, w)
        if np.linalg.norm(M @ sol - rhs) < tol * max(1.0, rhs.max()) and R >= C - tol:
            return True
    return False


def _nested_consistent(m1, values, zeros, g, tol=1e-6):
    """The flat of the second state lies inside the first's: m_1 is observed
    off the intersection, m_2 and phi are not. On the intersection
    x_j^2 = (m_1^2 + m_2^2) + 2 m_1 m_2 cos(phi + theta_j); with the linear
    solve of `_same_flat_consistent` the fitted (R, C) must satisfy
    C = 2 m_1 sqrt(R - m_1^2), and a zero needs m_2 = m_1."""
    vals = list(values) + ([0.0] if zeros else [])
    if len(vals) > g:
        return False
    if len(vals) <= 2:
        return True
    thetas = 2 * np.pi * np.arange(g) / g
    rhs = np.array(vals) ** 2
    for js in itertools.permutations(range(g), len(vals)):
        M = np.column_stack((np.ones(len(vals)), np.cos(thetas[list(js)]), -np.sin(thetas[list(js)])))
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        R, u, w = sol
        C = np.hypot(u, w)
        if np.linalg.norm(M @ sol - rhs) > tol * max(1.0, rhs.max()):
            continue
        m2sq = R - m1 ** 2
        if m2sq < -tol:
            continue
        m2 = np.sqrt(max(m2sq, 0.0))
        if abs(C - 2 * m1 * m2) > tol * max(1.0, C):
            continue
        if zeros and abs(m2 - m1) > tol:
            continue
        return True
    return False


def rank2_excluded(v, p, tol=1e-6):
    """Exact exclusion of v = a s_1 + b s_2 (two stabilizer states) from the
    modulus multiset of v alone. Returns (excluded, reason). Every test is a
    necessary condition, so `excluded` is sound; `not excluded` says nothing.

    With flats F_1, F_2 of dimensions k_1 >= k_2 meeting in a flat of size
    `cap` (or not at all): the p^k_1 - cap points of F_1 minus F_2 carry one
    modulus m_1 = |a| p^(-k_1/2), the p^k_2 - cap points of F_2 minus F_1 one
    modulus m_2, and each intersection point carries |m_1 + m_2 e^(i phi)
    w^j| for one unknown phi and j in Z_g (the phase difference of two
    quadratic forms on the intersection takes values in the phase group). A
    zero on the intersection needs m_1 = m_2 and phi + theta_j = pi (the
    coefficients' relative phase is free, so this is possible for any g).
    When one flat
    contains the other (F_2 minus F_1 empty) m_2 is not observed off the
    intersection and is fitted from the intersection moduli instead. The same
    flat gives at most g distinct nonzero moduli, zeros included when the two
    moduli agree."""
    g = phase_group(p)
    n = int(round(np.log(len(v)) / np.log(p)))
    pat = modulus_pattern(v, tol)
    if not pat:
        return True, "zero vector"
    vals = [x for x, _ in pat]
    mult = {round(x, 6): q for x, q in pat}
    exact = {round(x, 6): x for x, q in pat}          # unrounded moduli for the phase test
    supp = sum(mult.values())
    thetas = 2 * np.pi * np.arange(g) / g
    # same flat
    for k in range(n + 1):
        if p ** k >= supp and _same_flat_consistent(vals, p ** k > supp, g):
            return False, f"consistent with two states on one flat of dimension {k}"
    for k1 in range(n + 1):
        for k2 in range(k1 + 1):
            for kc in [-1] + list(range(k2 + 1)):
                cap = 0 if kc < 0 else p ** kc
                if k1 == k2 == kc:
                    continue
                A, B = p ** k1 - cap, p ** k2 - cap
                for m1 in vals:
                    m2_choices = vals if B > 0 else [None]
                    for m2 in m2_choices:
                        need = {round(m1, 6): A}
                        if m2 is not None:
                            need[round(m2, 6)] = need.get(round(m2, 6), 0) + B
                        if any(mult.get(x, 0) < q for x, q in need.items()):
                            continue
                        rest = {x: mult[x] - need.get(x, 0) for x in mult}
                        rest = {x: q for x, q in rest.items() if q > 0}
                        zeros = cap - sum(rest.values())
                        if zeros < 0:
                            continue
                        if m2 is None:
                            # nested flats: m2 unobserved, fitted from the intersection moduli
                            if _nested_consistent(m1, [exact[x] for x in rest], zeros > 0, g):
                                return False, (f"consistent configuration: flat of dim {k2} inside a flat of "
                                               f"dim {k1}, {len(rest)} intersection moduli")
                            continue
                        if zeros > 0 and abs(m1 - m2) > tol:
                            continue
                        if len(rest) > g:
                            continue
                        sets = []
                        bad = False
                        for x in rest:
                            c = _phase_candidates(exact[x], m1, m2, thetas)
                            if c is None:
                                bad = True
                                break
                            sets.append(c)
                        if bad:
                            continue
                        if zeros > 0:
                            sets.append(np.array([(np.pi - th) % (2 * np.pi) for th in thetas]))
                        if _consistent_phi(sets):
                            return False, (f"consistent configuration: flats of dims {k1}, {k2} meeting in "
                                           f"{cap} points, {zeros} cancellations")
    return True, "no pair of flats and coefficients matches the modulus multiset"
