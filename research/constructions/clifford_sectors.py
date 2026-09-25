"""Eigensector decompositions of |M>^m under non-Pauli single-qudit Cliffords.

The Pauli sector route (sectors.py, sectors_p.py) writes |M>^m as the sum of
its projections onto the joint eigenspaces of a group of Pauli strings; every
sector is a stabilizer code state one qudit smaller and the route is closed
on every record-capable cell (constructions_2026_09.md, 2026-09-24). This
script runs the same decomposition for a single-qudit Clifford C of order n
mod phase that is neither a Pauli nor a symmetry of |M>: with C normalized so
that C^n = I, the operator C^{(x) m} has order n and

    |M>^m = sum_k Pi_k |M>^m,   Pi_k |M>^m = (1/n) sum_j w_n^{-jk} |C^j M>^{(x) m},

so chi(|M>^m) <= sum_k chi(Pi_k |M>^m). For n = 2 and C = Z on the T-type
qubit state this is QPG's cat pair; for a non-Pauli C the sectors are sums of
n product states of Clifford images of |M> and do not lie in a stabilizer
code, so nothing is compressed: each sector is decided as an m-qudit state.
Exact where the m-qudit dictionary fits (qubits m <= 4 in seconds, m = 5 with
--dict5; qutrits m <= 3; ququints m <= 2): rank 1 by the stabilizer test, 2
by rank2_search, 3 by rank3_search with --rank3, else "> 3". Without a
dictionary every single-qudit slice of the sector (projection of one qudit
onto a basis state) is a state one qudit smaller whose rank is a lower bound
on the sector's, so the slices give exact lower bounds; --anneal R runs the
board's annealer on the undecided sectors for an upper bound.

Cyclic groups <C> are enumerated once each (the sectors depend only on the
group mod phase) and deduped under conjugation by the local Clifford
stabilizer of |M>; Paulis and elements fixing |M> up to phase are skipped
(the latter have a single nonzero sector).

Usage:
    clifford_sectors.py ORBIT M [--bound B] [--rank3] [--dict5] [--anneal R]
                        [--anneal-max K] [--orders 2,3,4] [--control]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import alpha  # noqa: E402
from rank_exclusion import clifford_group, dictionary, rank2_search, rank3_search  # noqa: E402
from sectors_p import ANNEAL_CFG, anneal, dict_for, is_stab  # noqa: E402
from stabrank_verify import ORBIT_P  # noqa: E402

TOL = 1e-9


def ukey(U):
    """Key of a unitary mod phase."""
    v = U.ravel()
    i = np.flatnonzero(np.abs(v) > 1e-9)[0]
    v = v / v[i]
    v = np.round(v, 6)
    # -0.0 and 0.0 differ as bytes in both parts, so add 0.0 to each
    return np.stack([v.real + 0.0, v.imag + 0.0]).tobytes()


def order_mod_phase(U, p):
    """Smallest n with U^n proportional to I, and U rescaled so U^n = I."""
    V = U.copy()
    for n in range(1, 4 * p * p + 10):
        if np.allclose(V, V[0, 0] * np.eye(p), atol=1e-8) and abs(V[0, 0]) > 1e-9:
            lam = V[0, 0]
            return n, U / lam ** (1.0 / n)
        V = V @ U
    raise ValueError("no finite order")


def is_pauli(U, p):
    X = np.roll(np.eye(p), 1, axis=0)
    Z = np.diag([np.exp(2j * np.pi * j / p) for j in range(p)])
    for P in (X, Z):
        Q = U @ P @ U.conj().T
        if ukey(Q) != ukey(P):
            return False
    return True


def cyclic_classes(orbit, p):
    """One generator per cyclic subgroup of the single-qudit Clifford group
    mod phase, non-Pauli, not fixing |M> up to phase, deduped under
    conjugation by the local Clifford stabilizer of |M>. Returns a list of
    (order, C with C^order = I, size of the conjugacy class of the group)."""
    a = alpha(orbit)
    G = clifford_group(p)
    stab = [U for U in G if abs(abs(np.vdot(a, U @ a)) - 1) < 1e-9]
    out = {}
    for U in G:
        if is_pauli(U, p):
            continue
        if abs(abs(np.vdot(a, U @ a)) - 1) < 1e-9:
            continue
        n, C = order_mod_phase(U, p)
        if n == 1:
            continue
        # key of the cyclic group mod phase
        pw = [np.linalg.matrix_power(C, j) for j in range(1, n)]
        gkey = frozenset(ukey(P) for P in pw)
        # canonical under conjugation by the stabilizer
        keys = []
        for V in stab:
            keys.append(frozenset(ukey(V @ P @ V.conj().T) for P in pw))
        canon = min(keys, key=lambda s: sorted(s))
        if canon not in out:
            out[canon] = (n, C, set())
        out[canon][2].add(gkey)
    return [(n, C, len(s)) for n, C, s in out.values()]


def sectors_of(a, C, n, m):
    """The nonzero sectors Pi_k |M>^m, k = 0..n-1, as unit vectors with weights."""
    prods = []
    v = a.copy()
    for j in range(n):
        t = v
        for _ in range(m - 1):
            t = np.kron(t, v)
        prods.append(t)
        v = C @ v
    w = np.exp(2j * np.pi / n)
    out = []
    for k in range(n):
        s = sum(w ** (-j * k) * prods[j] for j in range(n)) / n
        nrm = np.linalg.norm(s)
        if nrm > 1e-9:
            out.append((k, nrm ** 2, s / nrm))
    total = sum(wt for _, wt, _ in out)
    assert abs(total - 1) < 1e-8, total
    return out


def slices(u, p, m):
    """All single-qudit slices of u (projection of qudit i onto |a>), unit norm."""
    T = u.reshape((p,) * m)
    out = []
    for i in range(m):
        for x in range(p):
            s = np.take(T, x, axis=i).reshape(-1)
            nrm = np.linalg.norm(s)
            if nrm > 1e-9:
                out.append(s / nrm)
    return out


def exact_rank(u, p, n, D, rank3):
    """1, 2, 3, or '>3' / '>=3' in the n-qudit dictionary D."""
    if is_stab(u, p, n):
        return 1
    r2, _ = rank2_search(u, D)
    if r2 == "RANK1":
        return 1
    if r2:
        return 2
    if not rank3:
        return ">=3"
    r = rank3_search(u, D, workers=1)
    return 3 if r["found"] else ">3"


def lower_bound(u, p, m, cap, rank3, memo):
    """Exact rank when m <= cap, else the largest slice lower bound (recursive),
    as (value, exact_flag, text)."""
    if m <= cap:
        D = dict_for(p, m)
        r = exact_rank(u, p, m, D, rank3)
        if isinstance(r, int):
            return r, True, str(r)
        return {">=3": 3, ">3": 4}[r], False, r
    best = (1, "1")
    for s in slices(u, p, m):
        k = (np.round(s / s[np.flatnonzero(np.abs(s) > 1e-9)[0]], 6) + 0.0).tobytes()
        if k in memo:
            v, txt = memo[k]
        else:
            v, _, txt = lower_bound(s, p, m - 1, cap, rank3, memo)
            memo[k] = (v, txt)
        if v > best[0]:
            best = (v, txt)
    return best[0], False, f">={best[0]} (slice {best[1]})"


def control():
    """The Z (Pauli) route reproduced through the same code path: the two Z
    sectors of |T>^6 in the T basis are the QPG cat pair, both of rank 3."""
    p = 2
    w8 = np.exp(2j * np.pi / 8)
    a = np.array([1, w8]) / np.sqrt(2)
    Z = np.diag([1.0, -1.0]).astype(complex)
    secs = sectors_of(a, Z, 2, 6)
    assert len(secs) == 2
    ANNEAL_CFG.update(seeds=1, chains=4, iters=2000, cooling=0.99)
    for k, wt, u in secs:
        err, secs_, _ = anneal(u, p, 6, 3)
        print(f"  control Z^6 sector {k}: weight {wt:.3f}, rank-3 anneal residual {err:.2e} ({secs_:.0f}s)")
        assert err < 1e-7
    # sanity of the sector identity for a non-Pauli C
    C = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    n, C = order_mod_phase(C, 2)
    secs = sectors_of(alpha("qubit_H"), C, n, 3)
    psi = alpha("qubit_H")
    psi = np.kron(np.kron(psi, psi), psi)
    rec = sum(np.sqrt(wt) * u for _, wt, u in secs)
    assert np.allclose(rec, psi, atol=1e-8), "sector sum does not rebuild the target"
    print("  control: H sectors of |H>^3 rebuild the target; order", n)
    print("controls passed")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit", nargs="?")
    ap.add_argument("m", nargs="?", type=int)
    ap.add_argument("--bound", type=int, default=None, help="the cell's current upper bound")
    ap.add_argument("--rank3", action="store_true")
    ap.add_argument("--dict5", action="store_true", help="allow the five-qubit dictionary")
    ap.add_argument("--anneal", type=int, default=None, help="anneal undecided sectors at this rank")
    ap.add_argument("--anneal-max", type=int, default=6, help="anneal at most this many sectors")
    ap.add_argument("--anneal-chains", type=int, default=2)
    ap.add_argument("--anneal-iters", type=int, default=2000)
    ap.add_argument("--orders", default=None, help="comma-separated orders to keep")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--show", action="store_true", help="print the generator of every class")
    a = ap.parse_args(argv[1:])
    if a.control:
        control()
        return
    ANNEAL_CFG.update(seeds=1, chains=a.anneal_chains, iters=a.anneal_iters, cooling=0.99)
    orbit, m = a.orbit, a.m
    p = ORBIT_P[orbit]
    cap = {2: 5 if a.dict5 else 4, 3: 3, 5: 2}[p]
    vec = alpha(orbit)
    t0 = time.time()
    classes = cyclic_classes(orbit, p)
    keep = set(int(x) for x in a.orders.split(",")) if a.orders else None
    classes = [c for c in classes if keep is None or c[0] in keep]
    print(f"{orbit} m={m}: {len(classes)} cyclic classes of non-Pauli, non-symmetry Cliffords "
          f"(orders {sorted(set(c[0] for c in classes))}), dictionary cap n <= {cap}")
    rows = []
    memo = {}
    annealed = 0
    for n, C, mult in sorted(classes, key=lambda c: c[0]):
        secs = sectors_of(vec, C, n, m)
        parts = []
        total = 0
        exact_all = True
        for k, wt, u in secs:
            v, ex, txt = lower_bound(u, p, m, cap, a.rank3, memo)
            exact_all &= ex
            up = None
            if (a.anneal and not ex and annealed < a.anneal_max
                    and (a.bound is None or v < a.bound)):
                annealed += 1
                err, secs_, _ = anneal(u, p, m, a.anneal)
                up = f"anneal r={a.anneal}: {'hit' if err < 1e-7 else f'{err:.3f}'} ({secs_:.0f}s)"
            parts.append(f"k={k} w={wt:.3f} rank {txt}" + (f" [{up}]" if up else ""))
            total += v
        eig = np.round(np.linalg.eigvals(C), 3)
        rows.append((total, n, mult, exact_all, parts))
        print(f"  order {n} (x{mult}, eig {eig}): {len(secs)} nonzero sectors, "
              f"total {'=' if exact_all else '>='} {total}" + (f" (board {a.bound})" if a.bound else ""))
        for s in parts:
            print("     ", s)
        if a.show:
            print("      C =", np.array2string(np.round(C, 3), separator=", ").replace("\n", "\n         "))
    best = min(rows, key=lambda r: r[0]) if rows else None
    if best:
        print(f"smallest sector total: {'=' if best[3] else '>='} {best[0]} (order {best[1]}); "
              f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv)
