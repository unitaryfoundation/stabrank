"""Certificate: chi(|T3>^{ot 3}) >= 6, by eliminating every rank-5 configuration.

The Galois space V_3 = span{t_1^{ot 3}, t_4^{ot 3}, t_7^{ot 3}} has dimension 3
and sits inside the span of any exact decomposition (see cert_t3_galois.py), so
a rank-5 decomposition leaves its five stabilizer states with components off
V_3 spanning at most two dimensions.  Those five components are then five
points of a single projective line, and every rank-5 decomposition induces such
a line.  The lines are enumerated and each is rejected by least squares.

Enumerating them by pivot is quadratic: for each of the N = 30240 states, the
other N-1 are grouped by the line they span with it, and a group of four or
more is a candidate.  Done in C^27 that is 400 GB of memory traffic, which is
why the campaign script (stabrank/examples/t3m3_rank6_certificate.py, kept as
the record of the original run) takes about 26 CPU-minutes.  Collinearity
survives any linear map, so the grouping is done on a random projection to C^6
instead: a line of the full problem is still a line after projecting, and an
unlucky projection can only propose extra candidates, which the least-squares
step then rejects in full dimension.  The search cannot lose a configuration.

Six is not an arbitrary choice.  After quotienting by the pivot the directions
live in P^4, where 30240 points sit about 0.29 apart, against a parallelism
tolerance of sqrt(2 eps) = 1.4e-3.  Projecting all the way to C^3 puts them on
a projective line 0.006 apart, close enough to the tolerance that accidental
near-parallel pairs appear -- and an intruder sorting into the middle of a
genuine group splits it below the size-4 cutoff, which loses it.  At C^3 the
run reports 5004 groups instead of 45.

Canonicalising each quotient direction by a generic linear functional makes
parallel vectors identical, so a group occupies one consecutive run after
sorting. The key is constant on a hyperplane of directions rather than on a
projective point, so an unrelated direction can share it and land inside a
run; runs of near-equal key are therefore tested all pairs by overlap, which
can only enlarge a group, never split one. An earlier attempt bucketed
directions by rounding and found only 27 of the 45 groups. The size-4 cutoff
assumes five distinct off-V_3 directions, and the script checks that premise
by confirming no two stabilizer states are parallel off V_3.

Each candidate is rejected twice: by least squares, and then in exact
arithmetic over Q(w9), where the five states are vectors over Z[w3] and the
target has entries w9^(x0+x1+x2); adjoining the target must raise the rank of
every candidate. The harvest of candidate lines is numerical and rests on the
parallelism margin described above; the rejection of what it harvests is not.

Printed claim: CERTIFIED chi(T3^3) >= 6
"""

import functools
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# One thread per worker: the arrays are small enough that a threaded BLAS only
# oversubscribes the cores, and a verification run must leave the machine usable.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stabrank.examples.t3_galois_lower_bound import distinct_states

w9 = np.exp(2j * np.pi / 9)
EPS = 1e-6


def t(a):
    return np.array([1, w9 ** a, w9 ** (2 * a)], dtype=complex) / np.sqrt(3)


def tp(v, m):
    return functools.reduce(np.kron, [v] * m)


V = np.stack([tp(t(a), 3) for a in (1, 4, 7)], axis=1)
QB, _ = np.linalg.qr(V)
D = distinct_states(3)
PSI = tp(t(1), 3)
P = D - QB @ (QB.conj().T @ D)
P = (P / np.linalg.norm(P, axis=0)).astype(np.complex128)
N = P.shape[1]

_r = np.random.default_rng(2024)
DIM = 6
R = _r.normal(size=(DIM, P.shape[0])) + 1j * _r.normal(size=(DIM, P.shape[0]))
PD = R @ P
PD /= np.linalg.norm(PD, axis=0)
CAN = _r.normal(size=DIM) + 1j * _r.normal(size=DIM)
KEY = _r.normal(size=DIM) + 1j * _r.normal(size=DIM)


def check_distinct_directions():
    """The size-4 cutoff assumes the five states of a rank-5 configuration give
    five distinct directions off V_3. Two states with parallel off-V_3
    components would collapse into one and a configuration could be harvested
    with fewer than four companions of a pivot. Check the premise: canonicalise
    every off-V_3 direction and confirm no two are parallel."""
    can = _r.normal(size=P.shape[0]) + 1j * _r.normal(size=P.shape[0])
    key = _r.normal(size=P.shape[0]) + 1j * _r.normal(size=P.shape[0])
    c = can @ P
    if (np.abs(c) <= 1e-9).any():
        raise RuntimeError("canonicalising functional vanished on a direction")
    k = (key @ P) / c
    order = np.lexsort((k.imag, k.real))
    ks = k[order]
    gap = np.abs(np.diff(ks)) > 1e-8 * (1 + np.abs(ks[:-1]))
    starts = np.concatenate(([0], np.flatnonzero(gap) + 1, [N]))
    worst = 0.0
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        V = P[:, order[a:b]]
        G = np.abs(V.conj().T @ V)
        np.fill_diagonal(G, 0)
        worst = max(worst, float(G.max()))
    ov_adj = np.abs(np.sum(P[:, order[:-1]].conj() * P[:, order[1:]], axis=0))
    worst = max(worst, float(ov_adj.max()))
    if worst > 1 - EPS:
        raise RuntimeError(f"two stabilizer states are parallel off V_3 (overlap {worst})")
    print(f"off-V_3 directions pairwise distinct: closest overlap {worst:.4f}", flush=True)


def harvest(rng):
    """Groups of four or more states sharing a projective line with a pivot."""
    lo, hi = rng
    out = []
    for i in range(lo, hi):
        pi = PD[:, i]
        q = PD - np.outer(pi, pi.conj() @ PD)    # direction of the line through i and j
        nq = np.sqrt((q.real ** 2 + q.imag ** 2).sum(axis=0))
        sel = nq > 1e-9                          # drops j = i and any repeat of it
        idx = np.where(sel)[0]
        qh = q[:, sel] / nq[sel]
        c = CAN @ qh
        ok = np.abs(c) > 1e-9
        qh = qh[:, ok]
        idx = idx[ok]
        k = (KEY @ qh) / c[ok]                   # equal for parallel qh, so groups sort together
        order = np.lexsort((k.imag, k.real))
        ks = k[order]
        # The key is constant on a hyperplane of directions, not only on a
        # projective point, so an unrelated direction can share it exactly and
        # land between two parallel ones. Cut the order into runs of near-equal
        # key and test every pair inside a run, so a group can only be
        # enlarged by an intruder, never split.
        gap = np.abs(np.diff(ks)) > 1e-8 * (1 + np.abs(ks[:-1]))
        starts = np.concatenate(([0], np.flatnonzero(gap) + 1, [len(ks)]))
        for a, b in zip(starts[:-1], starts[1:]):
            if b - a < 4:
                continue
            members = order[a:b]
            V = qh[:, members]
            G = np.abs(V.conj().T @ V)
            par = G > 1 - EPS
            seen = np.zeros(b - a, bool)
            for s in range(b - a):
                if seen[s]:
                    continue
                comp, stack = [s], [s]
                seen[s] = True
                while stack:
                    v = stack.pop()
                    for w in np.flatnonzero(par[v]):
                        if not seen[w]:
                            seen[w] = True
                            comp.append(w)
                            stack.append(w)
                if len(comp) >= 4:
                    out.append((i, idx[members[comp]].tolist()))
    return out


def main():
    check_distinct_directions()
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, min(6, (os.cpu_count() or 2) // 2))
    bounds = [(N * k // nw, N * (k + 1) // nw) for k in range(nw)]
    allc = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for f in ex.map(harvest, bounds):
            allc.extend(f)
    print(f"candidate line groups of size >= 4 found: {len(allc)}", flush=True)
    sizes = {}
    for _, g in allc:
        sizes[len(g)] = sizes.get(len(g), 0) + 1
    print("group-size histogram:", dict(sorted(sizes.items())), flush=True)

    hit, tested = None, 0
    for i, g in allc:
        for c in itertools.combinations(g, 4):
            cols = [i] + list(c)
            A = D[:, cols]
            x, *_ = np.linalg.lstsq(A, PSI, rcond=None)
            tested += 1
            if np.linalg.norm(A @ x - PSI) < 1e-9:
                # a zero coefficient would mean rank 4 or less, which is a
                # stronger refutation, not a reason to look away
                hit = (cols, x)
                break
        if hit:
            break
    print(f"rank-5 candidates tested: {tested}", flush=True)
    if hit:
        print("RANK-5 DECOMPOSITION EXISTS -> chi(T3^3) <= 5")
        print("  states:", hit[0])
        print("  coeffs:", np.round(hit[1], 10))
        return 1
    # The least-squares rejection above is numerical. Repeat it in exact
    # arithmetic over Q(w9): each stabilizer state is a vector over Z[w3] once
    # divided by its first nonzero entry, |T3>^3 has entries w9^(x0+x1+x2), and
    # psi lies in the span of the five states exactly when adjoining it does not
    # raise the rank. Every candidate must raise it.
    n_exact = exact_rejection(allc)
    print(f"exact rejection over Q(w9): all {n_exact} candidates have "
          f"rank([S | psi]) = rank(S) + 1", flush=True)
    print("no rank-5 decomposition exists => chi(T3^3) >= 6")
    print("CERTIFIED chi(T3^3) >= 6")
    return 0


def exact_rejection(allc):
    """Reject every candidate 5-subset in exact arithmetic; return the count.

    Raises AssertionError if any candidate spans psi exactly, which would
    contradict the numerical pass and must not be silently absorbed.
    """
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix

    K = sp.QQ.algebraic_field(sp.exp(2 * sp.pi * sp.I / 9))
    x9 = K.from_sympy(sp.exp(2 * sp.pi * sp.I / 9))
    w3e = [K.one, x9 ** 3, x9 ** 6]

    def exact_state(v):
        nz = np.flatnonzero(np.abs(v) > 1e-9)
        r = v / v[nz[0]]
        out = []
        for z in r:
            if abs(z) < 1e-9:
                out.append(K.zero)
                continue
            k = int(round(np.angle(z) / (2 * np.pi / 3))) % 3
            if abs(z - np.exp(2j * np.pi * k / 3)) >= 1e-9:
                raise RuntimeError("entry is not in {0, 1, w3, w3^2}")
            out.append(w3e[k])
        return out

    psi_exact = [x9 ** ((idx // 9 + (idx // 3) % 3 + idx % 3) % 9) for idx in range(27)]
    cache = {}
    tested = 0
    for i, g in allc:
        for c in itertools.combinations(g, 4):
            cols = [i] + list(c)
            for j in cols:
                if j not in cache:
                    cache[j] = exact_state(D[:, j])
            rows = [[cache[j][r] for j in cols] for r in range(27)]
            rS = DomainMatrix(rows, (27, 5), K).rank()
            rows_psi = [row + [psi_exact[r]] for r, row in enumerate(rows)]
            rSp = DomainMatrix(rows_psi, (27, 6), K).rank()
            if rSp != rS + 1:
                raise RuntimeError(f"psi lies in the exact span of {cols}: a decomposition "
                                   "of rank at most 5 exists")
            tested += 1
    return tested


if __name__ == "__main__":
    raise SystemExit(main())
