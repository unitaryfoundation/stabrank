"""Permutation-symmetric ansatz for the qubit orbits (see perm_symmetric.py).

Same method with the qubit phase group: a term on the flat x0 + W y carries
the phase i^(l.y) (-1)^(Q(y)) with l in Z_4^k and Q strictly upper
triangular over F_2, i.e. exponents (l.y + 2 Q(y)) mod 4. A permutation of
the qubits fixing the flat fixes the state up to phase iff the exponents
shift by a constant mod 4 along the induced permutation of the support.

Flats are kept only when their S_m-stabilizer has order at least m!/R, and
the phase forms on a kept flat are tested against a generating set of that
stabilizer in chunks, so the 2^15 * 4^6 forms of the full six-qubit flat are
scanned rather than stored. For m = 6 and R <= 5 the only subgroups of S_6
of order >= 144 are A_6 and S_6, so every term of an S_6-invariant
decomposition with at most five terms is itself A_6-invariant.

Usage: perm_symmetric_qubit.py ORBIT M RMAX
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ORBIT_P, target  # noqa: E402
from perm_symmetric import cost_rank_search, flats, perm_index_maps  # noqa: E402


def generators(elements):
    """A small generating set of a permutation group given as arrays."""
    ident = np.arange(len(elements[0]))
    gens, closure = [], {ident.tobytes()}
    for g in elements:
        if g.tobytes() in closure:
            continue
        gens.append(g)
        frontier = [np.array(ident)]
        closure = {ident.tobytes()}
        frontier = [ident]
        while frontier:
            nxt = []
            for h in frontier:
                for s in gens:
                    t = s[h]
                    if t.tobytes() not in closure:
                        closure.add(t.tobytes())
                        nxt.append(t)
            frontier = nxt
        if len(closure) == len(elements):
            break
    return gens


def main(argv):
    orbit, m, R = argv[1], int(argv[2]), int(argv[3])
    assert ORBIT_P[orbit] == 2
    p = 2
    psi = target(orbit, m)
    dim = 2 ** m
    perms = perm_index_maps(m, p)
    P = np.array(perms)                                   # (m!, dim)
    G = len(perms)
    min_stab = -(-G // R)
    weights = 2 ** (m - 1 - np.arange(m))
    t0 = time.time()
    orbit_vecs, costs, reps, seen = [], [], [], set()
    n_flats = n_kept = n_states = 0
    for k, W, x0 in flats(m, p):
        n_flats += 1
        ys = np.array(list(itertools.product(range(2), repeat=k)), dtype=np.int64).reshape(2 ** k, k)
        supp = ((x0[None, :] + ys @ W) % 2) @ weights
        mask = np.zeros(dim, dtype=bool)
        mask[supp] = True
        imgs = P[:, supp]                                  # (m!, 2^k)
        inv = mask[imgs].all(axis=1)
        if inv.sum() < min_stab:
            continue
        n_kept += 1
        pos = np.full(dim, -1, dtype=np.int64)
        pos[supp] = np.arange(len(supp))
        flat_perms = [pos[imgs[g]] for g in np.flatnonzero(inv)]
        gens = generators(flat_perms)
        # exponent table: l in Z_4^k (columns 0..k-1, weight 1) and Q_ij, i<j (weight 2)
        lin = ys.T                                         # (k, 2^k)
        quad = np.array([ys[:, i] * ys[:, j] for i in range(k) for j in range(i + 1, k)],
                        dtype=np.int64).reshape(-1, 2 ** k)
        nq = quad.shape[0]
        total = 4 ** k * 2 ** nq
        n_states += total
        chunk = 1 << 16
        for lo in range(0, total, chunk):
            idx = np.arange(lo, min(total, lo + chunk))
            lpart = idx // (2 ** nq)
            qpart = idx % (2 ** nq)
            L = np.stack([(lpart // 4 ** i) % 4 for i in range(k)], axis=1) if k else np.zeros((len(idx), 0), dtype=np.int64)
            Qc = np.stack([(qpart // 2 ** i) % 2 for i in range(nq)], axis=1) if nq else np.zeros((len(idx), 0), dtype=np.int64)
            E = (L @ lin + 2 * (Qc @ quad)) % 4                # (chunk, 2^k)
            ok = np.ones(len(E), dtype=bool)
            for sigma in gens:
                diff = (E[:, sigma] - E) % 4
                ok &= (diff == diff[:, :1]).all(axis=1)
            for row in np.flatnonzero(ok):
                v = np.zeros(dim, dtype=complex)
                v[supp] = (1j) ** E[row]
                v /= np.linalg.norm(v)
                # exact stabilizer within the flat's permutations, and the character
                stab = 0
                trivial = True
                for sigma in flat_perms:
                    d = (E[row, sigma] - E[row]) % 4
                    if (d == d[0]).all():
                        stab += 1
                        if d[0] != 0:
                            trivial = False
                if stab < min_stab or not trivial:
                    continue
                sym = np.zeros(dim, dtype=complex)
                for g in perms:
                    w = np.zeros(dim, dtype=complex)
                    w[g] = v
                    sym += w
                if np.linalg.norm(sym) < 1e-9:
                    continue
                sym /= np.linalg.norm(sym)
                j = np.flatnonzero(np.abs(sym) > 1e-9)[0]
                key = (np.round(sym * (abs(sym[j]) / sym[j]), 6) + 0.0).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                orbit_vecs.append(sym)
                costs.append(G // stab)
                reps.append(v)
    costs = np.array(costs)
    print(f"{orbit} m={m}: {n_flats} flats, {n_kept} with a permutation stabilizer of order >= {min_stab}, "
          f"{n_states} phase forms scanned on them, {len(orbit_vecs)} S_{m}-orbits of size <= {R} with trivial "
          f"character; sizes: {dict(zip(*np.unique(costs, return_counts=True))) if len(costs) else {}} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    if not orbit_vecs:
        print(f"0 S_{m}-invariant decompositions with at most {R} terms")
        return 0
    D = np.column_stack(orbit_vecs)
    hits, tested = cost_rank_search(D, psi, costs, R)
    for size in sorted(tested):
        print(f"  {size}-orbit unions within cost {R}: {tested[size]} pivot sets scanned", flush=True)
    print(f"{len(hits)} S_{m}-invariant decompositions with at most {R} terms")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    for h, (cols, c) in enumerate(hits):
        terms = {}
        for j in cols:
            for g in perms:
                w = np.zeros(dim, dtype=complex)
                w[g] = reps[j]
                jj = np.flatnonzero(np.abs(w) > 1e-9)[0]
                terms.setdefault((np.round(w * (abs(w[jj]) / w[jj]), 6) + 0.0).tobytes(), w)
        vecs = list(terms.values())
        A = np.column_stack(vecs)
        x, *_ = np.linalg.lstsq(A, psi, rcond=None)
        path = os.path.join(out, f"perm_{orbit}_m{m}_rank{len(vecs)}_hit{h}.json")
        with open(path, "w") as fh:
            json.dump([[[float(z.real), float(z.imag)] for z in t] for t in vecs], fh)
        print(f"  hit {h}: orbits {cols}, {len(vecs)} terms, residual {np.linalg.norm(A @ x - psi):.1e}, written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
