"""Exact search for decompositions whose term set is invariant under one cyclic
shift of the copies (2026-09-26).

Setting. |M>^m and |cat_m> are invariant under every permutation of the copies,
so for any sigma in S_m the term set of a decomposition may be asked to be
sigma-invariant as an ansatz. Take sigma an m-cycle. The sigma-orbits on a term
set have size dividing m, so a decomposition of rank R < m with a sigma-invariant
term set has every term fixed by sigma, and if the terms are independent (they
are in a minimal decomposition) the fixing is exact: sigma|s_i> = |s_i>, not
merely up to a phase, since the coefficients of an independent set are unique.
For m prime and R < m the search is then finite and small, and this script runs
it exactly: enumerate every sigma-fixed stabilizer state, then ask for R of them
whose span contains the target.

The ansatz is the one the literature's own record decomposition satisfies.
Qassim, Pashayan, and Gosset's three terms for |cat_6> are, term by term,
invariant under every permutation of the six qubits: the line {0^6, 1^6}, the
even-weight uniform state |E_6>, and prod_{i<j} CZ_ij |E_6>.

Enumeration. A stabilizer state fixed by sigma is supported on a sigma-invariant
affine flat and carries a sigma-invariant phase function on it. The
sigma-invariant subspaces of F_p^n are the cyclic codes of length n, a handful
(8 for p = 2, n = 7; 4 for p = 3, n = 5), and the invariant cosets of each are
read off sigma(x0) - x0 in W. On each invariant flat the phase function is
constant on the sigma-orbits of the flat's points, so it is enumerated either
orbit by orbit (g^(r-1) patterns for r orbits, g = 4 at p = 2 and g = p
otherwise) or over the quadratic coefficients, whichever is smaller; the
whole-space flat is handled by the coefficient symmetry directly, where sigma
permutes the coordinates and the invariant forms are those with l constant and
Q constant on the cyclic orbits of coordinate pairs. Every state is confirmed by
`to_witness.term_from_vector`, the repository's own exact stabilizer test.

Subset search. Having excluded every rank below R, the target lies in the span
of a set S of size R-2 together with two further states a and b exactly when the
images of a and b in V / (span(S) + span(psi)) are nonzero and parallel, so the
search runs over the (R-2)-subsets and reads the parallel pairs off one Gram
matrix instead of solving C(N, R) least-squares problems.

Usage (from the repository root):

    uv run --extra challenge python research/constructions/cyclic_symmetric.py --control
    uv run --extra challenge python research/constructions/cyclic_symmetric.py \
        --orbit cat --m 7 --rank 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank_verify import ORBIT_P, target_vector  # noqa: E402
from to_witness import term_from_vector, NotStabilizer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def phase_modulus(p):
    """The modulus of the phase exponent: Z_4 at p = 2, Z_p otherwise."""
    return 4 if p == 2 else p


# ------------------------------------------------------------------ sigma ---

def digits(x, n, p):
    out = np.empty(n, dtype=np.int64)
    for j in range(n - 1, -1, -1):
        out[j] = x % p
        x //= p
    return out


def shift_index(x, n, p):
    """sigma on a basis index: the digit at position j moves to position j+1."""
    d = digits(x, n, p)
    d = np.roll(d, 1)
    return int(d @ (p ** (n - 1 - np.arange(n))))


def shift_source(n, p):
    """src[j] = the index whose amplitude lands at j under sigma."""
    dst = np.array([shift_index(x, n, p) for x in range(p ** n)], dtype=np.int64)
    src = np.empty_like(dst)
    src[dst] = np.arange(p ** n, dtype=np.int64)
    return src


def orbits_of(perm, size):
    seen = np.zeros(size, dtype=bool)
    out = []
    for x in range(size):
        if seen[x]:
            continue
        orb = []
        y = x
        while not seen[y]:
            seen[y] = True
            orb.append(y)
            y = int(perm[y])
        out.append(orb)
    return out


# ------------------------------------------------------- invariant flats ---

def _add(a, b, n, p):
    return int((digits(a, n, p) + digits(b, n, p)) % p @ (p ** (n - 1 - np.arange(n))))


def invariant_subspaces(n, p):
    """Every sigma-invariant subspace of F_p^n, as a frozenset of indices."""
    def close(gens):
        S = {0}
        for g in gens:
            if g in S:
                continue
            new = set(S)
            for c in range(1, p):
                gc = g
                for _ in range(c - 1):
                    gc = _add(gc, g, n, p)
                new |= {_add(s, gc, n, p) for s in S}
            S = new
        return frozenset(S)

    trivial = frozenset([0])
    found = {trivial}
    frontier = [trivial]
    while frontier:
        W = frontier.pop()
        for v in range(1, p ** n):
            if v in W:
                continue
            gens = set(W)
            u = v
            for _ in range(n):
                gens.add(u)
                u = shift_index(u, n, p)
            W2 = close(sorted(gens))
            if W2 not in found:
                found.add(W2)
                frontier.append(W2)
    return sorted(found, key=len)


def invariant_flats(n, p):
    """Every sigma-invariant affine flat, as (basis, points)."""
    out = []
    for W in invariant_subspaces(n, p):
        basis = []
        spanned = {0}
        for v in sorted(W):
            if v in spanned:
                continue
            basis.append(v)
            new = set(spanned)
            for c in range(1, p):
                vc = v
                for _ in range(c - 1):
                    vc = _add(vc, v, n, p)
                new |= {_add(s, vc, n, p) for s in spanned}
            spanned = new
        seen = set()
        for x0 in range(p ** n):
            if x0 in seen:
                continue
            coset = frozenset(_add(x0, w, n, p) for w in W)
            seen |= coset
            if all(shift_index(x, n, p) in coset for x in coset):
                out.append((tuple(basis), tuple(sorted(coset)), x0))
    return out


def flat_points(basis, x0, n, p):
    """Point index of each y in F_p^k, y read as a base-p index over `basis`."""
    k = len(basis)
    pts = np.empty(p ** k, dtype=np.int64)
    for y in range(p ** k):
        x = x0
        t = y
        for i in range(k):
            c = t % p
            t //= p
            for _ in range(c):
                x = _add(x, basis[i], n, p)
        pts[y] = x
    return pts


def _ydigits(k, p):
    ys = np.arange(p ** k, dtype=np.int64)
    return np.stack([(ys // p ** i) % p for i in range(k)], axis=1) if k else \
        np.zeros((p ** k, 0), dtype=np.int64)


# ------------------------------------- stabilizer states on a single flat ---

def quadratic_filter(phi, k, p):
    """Rows of `phi` (values in Z_g indexed by y) that are stabilizer phase
    functions: phi(0) + l.y + 2 Q(y) at p = 2 with Q strictly upper triangular,
    and phi(0) + l.y + Q(y) at odd p with Q upper triangular."""
    g = phase_modulus(p)
    B = phi.shape[0]
    e = [p ** i for i in range(k)]
    bits = _ydigits(k, p)
    c = phi[:, 0]
    ok = np.ones(B, dtype=bool)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if p == 2:
        lin = np.stack([(phi[:, e[i]] - c) % g for i in range(k)], axis=1) if k else \
            np.zeros((B, 0), dtype=phi.dtype)
        quad = np.zeros((B, len(pairs)), dtype=phi.dtype)
        for t, (i, j) in enumerate(pairs):
            d = (phi[:, e[i] | e[j]] - phi[:, e[i]] - phi[:, e[j]] + c) % 4
            ok &= (d % 2) == 0
            quad[:, t] = d // 2
        if not ok.any():
            return ok
        rec = (c[:, None] + lin @ bits.T) % 4
        for t, (i, j) in enumerate(pairs):
            rec = (rec + 2 * quad[:, t][:, None] * (bits[:, i] * bits[:, j])[None, :]) % 4
    else:
        inv2 = pow(2, p - 2, p)
        a = np.stack([(phi[:, e[i]] - c) % p for i in range(k)], axis=1) if k else \
            np.zeros((B, 0), dtype=phi.dtype)
        diag = np.zeros((B, k), dtype=phi.dtype)
        for i in range(k):
            d = (phi[:, 2 * e[i]] - c - 2 * a[:, i]) % p
            diag[:, i] = (inv2 * d) % p
        lin = (a - diag) % p
        quad = np.zeros((B, len(pairs)), dtype=phi.dtype)
        for t, (i, j) in enumerate(pairs):
            quad[:, t] = (phi[:, e[i] + e[j]] - c - a[:, i] - a[:, j]) % p
        rec = (c[:, None] + lin @ bits.T) % p
        for i in range(k):
            rec = (rec + diag[:, i][:, None] * (bits[:, i] ** 2)[None, :]) % p
        for t, (i, j) in enumerate(pairs):
            rec = (rec + quad[:, t][:, None] * (bits[:, i] * bits[:, j])[None, :]) % p
    ok &= np.all(rec == phi, axis=1)
    return ok


def patterns_by_orbit(orbs, k, p, cap):
    """Phase patterns constant on `orbs`, normalized to 0 on the orbit of y = 0."""
    g = phase_modulus(p)
    r = len(orbs)
    if g ** (r - 1) > cap:
        return None
    base = next(i for i, o in enumerate(orbs) if 0 in o)
    others = [i for i in range(r) if i != base]
    n_pat = g ** len(others)
    vals = np.zeros((n_pat, r), dtype=np.int64)
    for t, i in enumerate(others):
        vals[:, i] = (np.arange(n_pat) // (g ** t)) % g
    phi = np.zeros((n_pat, p ** k), dtype=np.int64)
    for i, o in enumerate(orbs):
        phi[:, np.array(o, dtype=np.int64)] = vals[:, i][:, None]
    return phi


def count_by_coeffs(k, p):
    """How many stabilizer phase functions F_p^k carries."""
    g = phase_modulus(p)
    npairs = k * (k - 1) // 2
    return (g ** k) * ((2 if p == 2 else p) ** npairs) * (1 if p == 2 else p ** k)


def patterns_by_coeffs(k, p, chunk=1 << 17):
    """Every stabilizer phase function on F_p^k, in chunks. The caller checks
    `count_by_coeffs` against its cap first: this is a generator, so a guard
    inside it would yield nothing rather than signalling the refusal."""
    g = phase_modulus(p)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    nq = len(pairs) + (0 if p == 2 else k)
    bits = _ydigits(k, p)
    cols = [bits[:, i] for i in range(k)]
    qcols = ([bits[:, i] ** 2 for i in range(k)] if p != 2 else []) + \
            [bits[:, i] * bits[:, j] for i, j in pairs]
    lvals = itertools.product(range(g), repeat=k)
    qbase = np.array(qcols).T if qcols else np.zeros((p ** k, 0), dtype=np.int64)
    qmod = 2 if p == 2 else p
    qall = np.array(list(itertools.product(range(qmod), repeat=nq)), dtype=np.int64) \
        if nq else np.zeros((1, 0), dtype=np.int64)
    qpart = ((2 if p == 2 else 1) * (qall @ qbase.T)) % g
    out = []
    for l in lvals:
        lin = (np.array(l, dtype=np.int64) @ np.array(cols)) % g if k else \
            np.zeros(p ** k, dtype=np.int64)
        out.append((lin[None, :] + qpart) % g)
        if sum(x.shape[0] for x in out) >= chunk:
            yield np.concatenate(out)
            out = []
    if out:
        yield np.concatenate(out)


def full_flat_states(n, p):
    """The sigma-fixed stabilizer phase functions of full support: l constant,
    the diagonal of Q constant, and the off-diagonal Q constant on the cyclic
    orbits of coordinate pairs."""
    g = phase_modulus(p)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = {q: t for t, q in enumerate(pairs)}
    seen, orbs = set(), []
    for q in pairs:
        if q in seen:
            continue
        orb, r = [], q
        for _ in range(n):
            r = tuple(sorted(((r[0] + 1) % n, (r[1] + 1) % n)))
            orb.append(idx[r])
            seen.add(r)
        orbs.append(sorted(set(orb)))
    bits = np.stack([(np.arange(p ** n) // p ** (n - 1 - i)) % p for i in range(n)], axis=1)
    qcols = np.stack([bits[:, i] * bits[:, j] for i, j in pairs], axis=1)
    dcol = (bits ** 2).sum(axis=1)
    qmod = 2 if p == 2 else p
    out = []
    for a in range(g):
        for dg in range(1 if p == 2 else p):
            for coeffs in itertools.product(range(qmod), repeat=len(orbs)):
                q = np.zeros(len(pairs), dtype=np.int64)
                for t, o in enumerate(orbs):
                    q[np.array(o)] = coeffs[t]
                phi = (a * bits.sum(axis=1) + (2 if p == 2 else 1) * (qcols @ q)
                       + dg * dcol) % g
                out.append(phi)
    return np.array(out, dtype=np.int64)


def fixed_stabilizer_states(n, p, cap=1 << 22, verbose=False):
    """Every stabilizer state |s> on n qudits with sigma|s> = |s>, as a list of
    (unit vector, witness term), plus the flats that had to be skipped."""
    src = shift_source(n, p)
    sig = np.array([shift_index(x, n, p) for x in range(p ** n)], dtype=np.int64)
    g = phase_modulus(p)
    root = np.exp(2j * np.pi / g)
    out, seen, skipped = [], set(), []
    for basis, points, x0 in invariant_flats(n, p):
        k = len(basis)
        if k == n:
            pts = np.arange(p ** n, dtype=np.int64)
            streams = [full_flat_states(n, p)]
            how, prefiltered = "coefficient symmetry", True
        else:
            pts = flat_points(basis, x0, n, p)
            pos = {int(v): i for i, v in enumerate(pts)}
            perm_y = np.array([pos[int(sig[int(v)])] for v in pts], dtype=np.int64)
            orbs = orbits_of(perm_y, p ** k)
            phi = patterns_by_orbit(orbs, k, p, cap)
            if phi is not None:
                streams = [phi[quadratic_filter(phi, k, p)]]
                how, prefiltered = f"orbit patterns (r={len(orbs)})", True
            else:
                nforms = count_by_coeffs(k, p)
                if nforms > cap:
                    skipped.append((k, x0, len(orbs), nforms))
                    if verbose:
                        print(f"  flat k={k} x0={x0}: SKIPPED, {len(orbs)} orbits "
                              f"({phase_modulus(p) ** (len(orbs) - 1)} patterns) and "
                              f"{nforms} quadratic forms, both over the cap", flush=True)
                    continue
                streams = patterns_by_coeffs(k, p)
                how, prefiltered = f"all {nforms} quadratic forms", False
                perm_for_stream = perm_y
        kept = 0
        for phi in streams:
            if not prefiltered:
                phi = phi[np.all(phi[:, perm_for_stream] == phi, axis=1)]
            if phi.shape[0] == 0:
                continue
            amp = root ** phi / np.sqrt(p ** k)
            for row in amp:
                v = np.zeros(p ** n, dtype=complex)
                v[pts] = row
                if np.linalg.norm(v[src] - v) > 1e-9:
                    continue
                key = tuple(np.round(v / v[pts[0]], 6).view(np.float64))
                if key in seen:
                    continue
                try:
                    term = term_from_vector(v, p, n)
                except NotStabilizer:
                    continue
                seen.add(key)
                out.append((v, term))
                kept += 1
        if verbose:
            print(f"  flat k={k} x0={x0}: {how}, {kept} sigma-fixed states", flush=True)
    return out, skipped


# --------------------------------------------------------- subset search ---

def min_rank(states, psi, rmax, verbose=False, tol=1e-8):
    """The smallest R <= rmax with psi in the span of R of `states`, and one
    witnessing (index tuple, coefficients). Exhaustive."""
    if not states:
        return None, None
    U = np.column_stack(states)
    N = len(states)
    B = np.linalg.qr(np.column_stack([U, psi]))[0]
    A = B.conj().T @ U
    b = B.conj().T @ psi

    def verify(comb):
        M = U[:, list(comb)]
        c, *_ = np.linalg.lstsq(M, psi, rcond=None)
        return c if np.linalg.norm(M @ c - psi) < 1e-9 else None

    for R in range(1, rmax + 1):
        t0 = time.time()
        if R == 1:
            for i in range(N):
                c = verify((i,))
                if c is not None:
                    return 1, ((i,), c)
            if verbose:
                print(f"  rank 1: none, {time.time() - t0:.1f}s", flush=True)
            continue
        stats = [0]
        b0 = b / np.linalg.norm(b)
        A0 = A - np.outer(b0, b0.conj() @ A)
        cur = []

        def descend(img, depth, start):
            nrm = np.linalg.norm(img, axis=0)
            if depth == R - 2:
                stats[0] += 1
                live = np.flatnonzero(nrm > 1e-7)
                live = live[live >= start]
                if live.size < 2:
                    return None
                X = img[:, live] / nrm[live]
                G = np.abs(X.conj().T @ X)
                np.fill_diagonal(G, 0.0)
                uu, vv = np.nonzero(np.triu(G) > 1 - tol)
                for u, v in zip(uu, vv):
                    comb = tuple(sorted(cur + [int(live[u]), int(live[v])]))
                    c = verify(comb)
                    if c is not None:
                        return comb, c
                return None
            for i in range(start, N - (R - 2 - depth) + 1):
                if nrm[i] <= 1e-7:
                    continue
                q = img[:, i] / nrm[i]
                cur.append(i)
                hit = descend(img - np.outer(q, q.conj() @ img), depth + 1, i + 1)
                cur.pop()
                if hit is not None:
                    return hit
            return None

        hit = descend(A0, 0, 0)
        if hit is not None:
            return R, hit
        if verbose:
            print(f"  rank {R}: {stats[0]} subsets of size {R - 2} scanned, none "
                  f"completes, {time.time() - t0:.1f}s", flush=True)
    return None, None


# ----------------------------------------------------------------- driver ---

def run_cell(orbit, m, rmax, cap, verbose=True):
    p = ORBIT_P[orbit]
    n = m
    t0 = time.time()
    states, skipped = fixed_stabilizer_states(n, p, cap=cap, verbose=verbose)
    psi = np.array([complex(x) for x in target_vector(orbit, m)]).ravel()
    psi = psi / np.linalg.norm(psi)
    fixed_ok = np.linalg.norm(psi[shift_source(n, p)] - psi) < 1e-9
    print(f"{orbit} m={m} (p={p}): {len(states)} sigma-fixed stabilizer states, "
          f"target sigma-fixed: {fixed_ok}, {len(skipped)} flats skipped, "
          f"{time.time() - t0:.1f}s")
    if skipped:
        print(f"  skipped flats (k, x0, orbits): {skipped}")
    U = np.column_stack([s for s, _ in states]) if states else np.zeros((1, 0))
    c, *_ = np.linalg.lstsq(U, psi, rcond=None)
    res = np.linalg.norm(U @ c - psi)
    print(f"  residual of the target on the span of all of them: {res:.3e} "
          f"(span dimension {np.linalg.matrix_rank(U, tol=1e-8)})")
    if res > 1e-9:
        print(f"  VERDICT: the target is not in the span of the sigma-fixed states, "
              f"so no decomposition of rank < {m} has a sigma-invariant term set.")
        return None
    R, hit = min_rank([s for s, _ in states], psi, rmax, verbose=verbose)
    if R is None:
        print(f"  VERDICT: no sigma-invariant decomposition of rank <= {rmax}.")
        return None
    comb, coeffs = hit
    print(f"  HIT: rank {R} sigma-invariant decomposition, terms {comb}")
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"cyclic_{orbit}_m{m}_rank{R}.json")
    with open(path, "w") as f:
        json.dump({"orbit": orbit, "m": m, "rank": R,
                   "terms": [states[i][1] for i in comb],
                   "coeffs": [[float(z.real), float(z.imag)] for z in coeffs]}, f, indent=1)
    print(f"  written {path}")
    return R


CONTROLS = [("cat", 6, 3), ("cat", 5, 3), ("qubit_H", 3, 3), ("qubit_T", 3, 3),
            ("N", 3, 4), ("H3", 3, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbit")
    ap.add_argument("--m", type=int)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--cap", type=int, default=1 << 24)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.control:
        for orbit, m, rmax in CONTROLS:
            run_cell(orbit, m, rmax, a.cap, verbose=not a.quiet)
            print()
        return
    if not a.orbit or not a.m:
        raise SystemExit("--orbit and --m, or --control")
    run_cell(a.orbit, a.m, a.rank, a.cap, verbose=not a.quiet)


if __name__ == "__main__":
    main()
