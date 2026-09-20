"""Shared helpers for the structured-construction scripts.

Everything here is numeric (complex128) and is used to screen candidates.
Anything that becomes a bound goes through verify_challenge/to_witness.py
and fit_coeffs.py, which are exact.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank_verify import ORBIT_P, orbit_state  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOL = 1e-7


def alpha(orbit):
    """Single-copy amplitudes as a complex array."""
    return np.array([complex(x) for x in orbit_state(orbit)]).ravel()


def target(orbit, m):
    a = alpha(orbit)
    v = a
    for _ in range(m - 1):
        v = np.kron(v, a)
    return v


def term_vector(term, p, n):
    """Numeric rebuild of a witness term (k, x0, W, Q, l), unit norm."""
    k = int(term["k"])
    x0 = np.array(term["x0"], dtype=np.int64) % p
    W = np.array(term.get("W", []), dtype=np.int64).reshape(k, n) % p
    Q = np.array(term.get("Q", [[0] * k for _ in range(k)]), dtype=np.int64).reshape(k, k)
    lmod = 4 if p == 2 else p
    ell = np.array(term.get("l", [0] * k), dtype=np.int64) % lmod
    dim = p ** n
    v = np.zeros(dim, dtype=complex)
    weights = p ** (n - 1 - np.arange(n))
    for y in itertools.product(range(p), repeat=k):
        y = np.array(y, dtype=np.int64)
        x = (x0 + y @ W) % p
        pos = int(x @ weights)
        q = sum(int(Q[i][j]) * int(y[i]) * int(y[j]) for i in range(k) for j in range(i, k))
        lin = int(ell @ y)
        if p == 2:
            v[pos] = (1j) ** (lin % 4) * (-1) ** (q % 2)
        else:
            v[pos] = np.exp(2j * np.pi * ((q + lin) % p) / p)
    return v / np.sqrt(p ** k)


def load_decompositions(orbit, m, rank):
    """The stored list of rank-`rank` decompositions of |M>^m (term vectors and
    coefficients), each re-checked to reproduce the target."""
    path = os.path.join(DATA, f"{orbit}_m{m}_rank{rank}.json")
    with open(path) as f:
        rec = json.load(f)
    p = ORBIT_P[orbit]
    psi = target(orbit, m)
    out = []
    for dec in rec["decompositions"]:
        U = np.column_stack([term_vector(t, p, m) for t in dec])
        d, *_ = np.linalg.lstsq(U, psi, rcond=None)
        if np.linalg.norm(U @ d - psi) > 1e-9 or np.linalg.matrix_rank(U, tol=1e-8) < U.shape[1]:
            raise AssertionError(f"a stored decomposition of {orbit}^{m} does not reproduce the target")
        out.append(([U[:, i] for i in range(U.shape[1])], d))
    return out, rec


def is_stabilizer_batch(V, p, tol=1e-6):
    """Boolean mask over the rows of V (shape (batch, p^n)): rows whose nonzero
    entries all have one modulus and whose support size is a power of p.
    Necessary for a stabilizer state, not sufficient; survivors are confirmed
    by `confirm_stabilizer`."""
    A = np.abs(V)
    mx = A.max(axis=1)
    nz = A > tol * np.maximum(mx, 1e-300)[:, None]
    cnt = nz.sum(axis=1)
    ok = mx > tol
    mn = np.where(nz, A, np.inf).min(axis=1)
    ok &= (mx - mn) <= 1e-6 * mx
    powers = [p ** k for k in range(0, 20)]
    ok &= np.isin(cnt, powers)
    return ok


def confirm_stabilizer(v, p, n):
    """The witness term for v if v is a stabilizer state, else None."""
    from to_witness import term_from_vector, NotStabilizer
    try:
        return term_from_vector(v, p, n)
    except NotStabilizer:
        return None


def phase_table(p):
    if p == 2:
        return np.array([1, 1j, -1, -1j])
    return np.exp(2j * np.pi * np.arange(p) / p)
