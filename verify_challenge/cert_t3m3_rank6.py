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
sorting, and every adjacent pair is then checked by an exact overlap rather
than by matching rounded coordinates: an earlier attempt bucketed directions by
rounding and found only 27 of the 45 groups.

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
        order = np.argsort(k)                    # complex sorts by (real, imag)
        qs = qh[:, order]
        ids = idx[order]
        ov = np.abs(np.sum(qs[:, :-1].conj() * qs[:, 1:], axis=0))
        par = ov > 1 - EPS
        a = 0
        while a < len(par):
            if par[a]:
                b = a
                while b < len(par) and par[b]:
                    b += 1
                members = ids[a:b + 1].tolist()
                if len(members) >= 4:
                    out.append((i, members))
                a = b
            else:
                a += 1
    return out


def main():
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
            if np.linalg.norm(A @ x - PSI) < 1e-9 and np.all(np.abs(x) > 1e-9):
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
    print("no rank-5 decomposition exists => chi(T3^3) >= 6")
    print("CERTIFIED chi(T3^3) >= 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
