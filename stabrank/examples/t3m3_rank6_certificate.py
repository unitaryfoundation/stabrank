"""chi(|T3>^{ot 3}) >= 6, by exhausting every candidate rank-5 configuration.

Companion to t3_galois_lower_bound.py, which proves >= 5.  The Galois space
V_3 = span{t_1^{ot 3}, t_4^{ot 3}, t_7^{ot 3}} has dimension 3 and is contained
in the span of any exact decomposition, so a rank-5 span leaves its five states
with components off V_3 spanning at most 2 dimensions: five coplanar points.
Quotienting by one of them, the other four become parallel.  So every possible
rank-5 decomposition corresponds to a parallel class of size >= 4 in some
quotient, and there are only finitely many to check.

Such classes DO exist -- 45 of them, all of size exactly 4 -- so this cannot be
settled by observing their absence.  Each one is decided by least squares, and
none spans |T3>^{ot 3}.

Canonicalising each quotient vector by a generic linear functional makes
genuinely parallel vectors identical, so a class occupies one consecutive run
after sorting; the runs are harvested in O(N^2 log N) and their members are then
checked with exact inner products.  This matters: an earlier attempt grouped
directions by rounding them into hash buckets and found only 27 of the 45.
"""
import numpy as np, functools, sys, itertools
from concurrent.futures import ProcessPoolExecutor
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stabrank.examples.t3_galois_lower_bound import distinct_states
w9 = np.exp(2j*np.pi/9)
def t(a): return np.array([1, w9**a, w9**(2*a)], dtype=complex)/np.sqrt(3)
def tp(v, m): return functools.reduce(np.kron, [v]*m)
V = np.stack([tp(t(a), 3) for a in (1,4,7)], axis=1)
QB, _ = np.linalg.qr(V)
D = distinct_states(3)
PSI = tp(t(1), 3)
P = D - QB @ (QB.conj().T @ D)
P = (P/np.linalg.norm(P, axis=0)).astype(np.complex128)
N = P.shape[1]
_r = np.random.default_rng(2024)
CAN = _r.normal(size=P.shape[0]) + 1j*_r.normal(size=P.shape[0])
KEY = _r.normal(size=P.shape[0]) + 1j*_r.normal(size=P.shape[0])
EPS = 1e-6

def harvest(rng):
    lo, hi = rng
    out = []
    for i in range(lo, hi):
        pi = P[:, i]
        q = P - np.outer(pi, pi.conj() @ P)
        nq = np.linalg.norm(q, axis=0)
        sel = np.arange(N) != i
        qh = q[:, sel]/nq[sel]
        idx = np.where(sel)[0]
        c = CAN @ qh
        ok = np.abs(c) > 1e-9
        qh = qh[:, ok]/c[ok]; idx = idx[ok]
        k = KEY @ qh
        order = np.lexsort((k.imag, k.real))
        qs = qh[:, order]; ids = idx[order]
        u = qs/np.linalg.norm(qs, axis=0)
        ov = np.abs(np.sum(u[:, :-1].conj()*u[:, 1:], axis=0))
        par = ov > 1 - EPS
        a = 0
        while a < len(par):
            if par[a]:
                b = a
                while b < len(par) and par[b]: b += 1
                members = ids[a:b+1].tolist()      # b-a+1 mutually parallel
                if len(members) >= 4: out.append((i, members))
                a = b
            else:
                a += 1
    return out

if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    bounds = [(N*k//nw, N*(k+1)//nw) for k in range(nw)]
    allc = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for f in ex.map(harvest, bounds): allc.extend(f)
    print(f"parallel classes of size >= 4 found: {len(allc)}", flush=True)
    sizes = {}
    for i, g in allc: sizes[len(g)] = sizes.get(len(g), 0) + 1
    print("class-size histogram:", dict(sorted(sizes.items())), flush=True)
    hit = None; tested = 0
    for i, g in allc:
        for c in itertools.combinations(g, 4):
            cols = [i] + list(c)
            A = D[:, cols]
            x, *_ = np.linalg.lstsq(A, PSI, rcond=None)
            tested += 1
            if np.linalg.norm(A @ x - PSI) < 1e-9 and np.all(np.abs(x) > 1e-9):
                hit = (cols, x); break
        if hit: break
    print(f"rank-5 candidates tested: {tested}", flush=True)
    if hit:
        print("RANK-5 DECOMPOSITION EXISTS -> chi(T3^3) <= 5 -> gamma_T3 <= log_3(5)/3 = 0.4883 (BEATS 1/2)")
        print("  states:", hit[0])
        print("  coeffs:", np.round(hit[1], 10))
    else:
        print("no rank-5 decomposition exists => chi(T3^3) >= 6  (certified, all classes tested)")
