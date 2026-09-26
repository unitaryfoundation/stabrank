"""Positive control at m = 2 for the three-pivot canonical augmentation (case-C machinery).

The three states inside V_2 are removed.  Objects compared: pools.  A class (i, j, k, U)
emitted by kernel3 determines the 4-dim image space U mod ELL, and its pool is the set of all
dictionary states with image in U (independent of the index cutoffs of the class).  A seven-set
F in general position (every five images span 4 dims mod ELL) with exact rank pair (7, 7) has
images spanning a 4-space U(F), and every class containing F has that U, hence the same pool.
(a) Canonical scan: first pivot over orbit minima of the monomial group of V_2 (order 324),
    j minimal in its Stab(i)-orbit, k minimal in its Stab(i, j)-orbit (masks as in cost_c.py);
    K_can = orbit keys (minimum over the group of a hash of the pool) of all emitted pools.
(b) Naive scan for a random sample of first pivots i with all j > i, k > j: for each class
    whose pool has <= MAXPOOL states, every seven-subset is tested (exact rank pair (7, 7) and
    general position); pools with at least one such seven-set go into K_naive.
Every key in K_naive must lie in K_can (completeness of the augmentation); K_naive must be
nonempty (the control is not vacuous)."""
import itertools, os, sys, time
import numpy as np
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cert7 as C, k3, caseB
from setup_m3 import load
from decide import Decider, ELL2
from cost_c import masks_from_perms, kok_of

d = load(2)
N0, PD0, E0, T, isfree = d["N"], d["PD"], d["E"], d["T"], d["isfree"]
free = np.flatnonzero(isfree); keep = np.flatnonzero(isfree == 0)
print("states inside V_2 removed:", free.tolist(), flush=True)
PD, E = np.ascontiguousarray(PD0[keep]), E0[keep]; N = len(keep)
old_to_new = -np.ones(N0, dtype=np.int64); old_to_new[keep] = np.arange(N)
INV = C.INV
dec = Decider(E, T)
E2T = np.vstack([dec.E2, dec.T2])
elems = C.monomial_symmetries(T, 2)
index = {k: i for i, k in enumerate(C.row_keys(E0))}
perms = []
for e in elems:
    P0 = np.fromiter((index[k] for k in C.row_keys(C.apply_element(E0, e))), dtype=np.int64, count=N0)
    P = old_to_new[P0[keep]]; assert np.all(P >= 0); perms.append(P)
perms = np.stack(perms)
print("group order", len(perms), flush=True)
RND = np.random.default_rng(9).integers(1, 2**62, size=N, dtype=np.int64)
MAXPOOL = 14


@njit(cache=True)
def pool_key(PD, inv, i, j, k, l, ell, perms, RND, pool):
    """Pool of span(q_i, q_j, q_k, q_l) mod ell (states with image in it, into `pool`),
    its size, and the orbit key = min over the group of sum RND[P[s]]."""
    N, D = PD.shape
    B = np.zeros((4, D), dtype=np.int64); piv = np.zeros(4, dtype=np.int64)
    v = np.zeros(D, dtype=np.int64)
    rk = 0
    for s in (i, j, k, l):
        for t in range(D): v[t] = PD[s, t]
        for q in range(rk):
            f = v[piv[q]]
            if f != 0:
                for t in range(D): v[t] = (v[t] - f * B[q, t]) % ell
        lead = -1
        for t in range(D):
            if v[t] != 0: lead = t; break
        if lead < 0: continue
        iv = inv[v[lead]]
        for t in range(D): B[rk, t] = (v[t] * iv) % ell
        piv[rk] = lead; rk += 1
    n = 0
    for s in range(N):
        for t in range(D): v[t] = PD[s, t]
        for q in range(rk):
            f = v[piv[q]]
            if f != 0:
                for t in range(D): v[t] = (v[t] - f * B[q, t]) % ell
        z = True
        for t in range(D):
            if v[t] != 0: z = False; break
        if z:
            pool[n] = s; n += 1
    best = np.int64(0x7FFFFFFFFFFFFFFF)
    for g in range(perms.shape[0]):
        h = np.int64(0)
        for q in range(n):
            h += RND[perms[g, pool[q]]]
        if h < best: best = h
    return n, best, rk


def classes_of(i, jok, koks_i, plain_max):
    out = []
    js = [int(j) for j in np.flatnonzero(jok)]
    plain = np.zeros(N, dtype=np.int8)
    steps = 0
    for j in js:
        if (i, j) in koks_i:
            jj = np.zeros(N, dtype=np.int8); jj[j] = 1
            c, mb, ns = k3.kernel3(PD, INV, i, jj, koks_i[(i, j)], 3, 50000, 0.0, 1.0)
            assert c.shape[0] < 50000 and mb.shape[0] < 50000 * 16
            out.append((c, mb)); steps += ns
        else:
            plain[j] = 1
    if plain.any():
        c, mb, ns = k3.kernel3(PD, INV, i, plain, ones, 3, plain_max, 0.0, 1.0)
        assert c.shape[0] < plain_max and mb.shape[0] < plain_max * 16, (c.shape[0], mb.shape[0])
        out.append((c, mb)); steps += ns
    return out, steps


ones = np.ones(N, dtype=np.int8)
pool = np.zeros(N, dtype=np.int64)
gmin = perms.min(axis=0)
reps = np.flatnonzero(gmin == np.arange(N))
jokm, koks = [], {}
for i in reps:
    st = [perms[g] for g in range(len(perms)) if perms[g][i] == i]
    jok, pair_stab = masks_from_perms(N, int(i), st)
    jokm.append(jok)
    for j, gl in pair_stab.items():
        koks[(int(i), j)] = kok_of(N, j, [st[g] for g in gl])
print(f"{len(reps)} orbits, {sum(int(j.sum()) for j in jokm)} pivot pairs, {len(koks)} pairs with nontrivial Stab(i, j)", flush=True)

# (a) canonical
t0 = time.time(); K_can = set(); ncl = 0; steps = 0
for i, jok in zip(reps, jokm):
    i = int(i)
    res, ns = classes_of(i, jok, koks, 400000); steps += ns
    for c, mb in res:
        ncl += c.shape[0]
        first = {}
        for cid, s, tag in mb:
            first.setdefault(int(cid), int(s))
        for cid in range(c.shape[0]):
            j, k = int(c[cid, 0]), int(c[cid, 1])
            l = first[cid]
            n, key, rk = pool_key(PD, INV, i, j, k, l, C.ELL, perms, RND, pool)
            assert rk == 4
            K_can.add(int(key))
print(f"canonical: {steps:.3e} steps, {ncl} classes, {len(K_can)} distinct pool orbits ({time.time()-t0:.0f}s)", flush=True)

# (b) naive, sampled first pivots
rng = np.random.default_rng(3)
sample = sorted(int(x) for x in rng.choice(N - 8, 2, replace=False))
t0 = time.time(); K_naive = set(); ncl = 0; nsub = 0; nsev = 0; steps = 0; nbig = 0
for i in sample:
    jok = (np.arange(N) > i).astype(np.int8)
    res, ns = classes_of(i, jok, {}, 3000000); steps += ns
    for c, mb in res:
        ncl += c.shape[0]
        first = {}
        for cid, s, tag in mb:
            first.setdefault(int(cid), int(s))
        for cid in range(c.shape[0]):
            j, k = int(c[cid, 0]), int(c[cid, 1])
            n, key, rk = pool_key(PD, INV, i, j, k, first[cid], C.ELL, perms, RND, pool)
            if int(key) in K_naive:
                continue
            if n > MAXPOOL:
                nbig += 1; continue
            P = [int(x) for x in pool[:n]]
            sub = np.array(list(itertools.combinations(P, 7)), dtype=np.int64)
            nsub += len(sub)
            r1 = caseB.ranks_batch(dec.E2, sub, ELL2)
            r2 = caseB.ranks_batch(E2T, np.hstack([sub, N + np.arange(3)[None, :].repeat(len(sub), 0)]), ELL2)
            good = np.flatnonzero((r1 == 7) & (r2 == 7))
            if len(good) == 0:
                continue
            F5 = np.array([[sub[t][u] for u in s5] for t in good for s5 in itertools.combinations(range(7), 5)], dtype=np.int64)
            r5 = caseB.ranks_batch(PD, F5, C.ELL).reshape(len(good), 21)
            ok = np.any(np.all(r5 == 4, axis=1))
            nsev += int(np.sum(np.all(r5 == 4, axis=1)))
            if ok:
                K_naive.add(int(key))
print(f"naive (first pivots {sample}): {steps:.3e} steps, {ncl} classes, {nsub} seven-subsets tested, {nsev} in general position with rank pair (7, 7), "
      f"{nbig} classes with pools > {MAXPOOL} skipped; {len(K_naive)} pool orbits with a genuine seven-set ({time.time()-t0:.0f}s)", flush=True)
missing = K_naive - K_can
print(f"pool orbits found naively but not canonically: {len(missing)}")
assert not missing
assert K_naive, "control vacuous"
print("CONTROL OK: every general-position rank-7 span of V_2 found by the naive three-pivot scan is found by the canonical scan")
