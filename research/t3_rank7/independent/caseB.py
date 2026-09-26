"""Case B of the rank-7 exclusion: seven-state configurations with five images in a
3-dim space mod ELL (and, by case A, not six).

Stage 1.  Two-pivot scan (the rank-7 certificate's kernel2) with need = 3: classes
{i, j} + members zero-or-parallel modulo (q_i, q_j).  Any five-subset F5 of a genuine
configuration containing the pivots is independent, so it has exact rank 5 (mod ELL2, exact
for values <= 8).  Every five-subset (pivots + 3 members) is filtered by that rank; the
survivors carry their exact rank pair (5, 6) (honest: images span 3 over Q(w3)) or (5, 7)
(dishonest: images span 4 over Q(w3), coplanar only mod ELL).

Stage 2.  For an honest F5, W6 = span(F5) + V_3 is 6-dim and the two remaining states a, b
lie outside W6 with parallel images modulo W6 (span(F5, a, b) is 7-dim and contains V_3, so
it equals W6 + a = W6 + b).  Images modulo W6 are computed mod ELL2 (the annihilator of W6 is
obtained by Cramer with 6 x 6 minors, of norm < ELL2, so the reduction is faithful);
parallel classes give candidate sets F5 + class, decided exactly.  For a dishonest F5,
W7 = span(F5) + V_3 is 7-dim and both a, b lie in W7: every dictionary state in W7 outside
span(F5) is listed and the set F5 + (all of them) is decided exactly (a decomposition exists
iff that set has rank 7 and contains V_3; if its rank is 7 and it contains V_3 any two
states outside span(F5) complete F5).
"""
import itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert7 as C
from setup_m3 import load
from decide import Decider, rank_mod, ELL2

MAXOUT = 400000


@njit(cache=True)
def ranks_batch(E2, sets, ell):
    """Rank mod ell of E2[sets[t]] for every row t of sets (small matrices)."""
    M, k = sets.shape
    n = E2.shape[1]
    out = np.empty(M, dtype=np.int64)
    A = np.empty((k, n), dtype=np.int64)
    for t in range(M):
        for a in range(k):
            for c in range(n):
                A[a, c] = E2[sets[t, a], c]
        r = 0
        for c in range(n):
            pr = -1
            for a in range(r, k):
                if A[a, c] != 0:
                    pr = a
                    break
            if pr < 0:
                continue
            if pr != r:
                for cc in range(n):
                    tmp = A[r, cc]; A[r, cc] = A[pr, cc]; A[pr, cc] = tmp
            # inverse by Fermat
            base = A[r, c] % ell; e = ell - 2; inv = 1
            while e > 0:
                if e & 1:
                    inv = (inv * base) % ell
                base = (base * base) % ell
                e >>= 1
            for cc in range(n):
                A[r, cc] = (A[r, cc] * inv) % ell
            for a in range(k):
                if a != r and A[a, c] != 0:
                    f = A[a, c]
                    for cc in range(n):
                        A[a, cc] = (A[a, cc] - f * A[r, cc]) % ell
            r += 1
            if r == k:
                break
        out[t] = r
    return out


_G = {}


def _init(d):
    _G.update(d)
    _G["dec"] = Decider(d["E"], d["T"])
    _G["E2"], _G["T2"] = _G["dec"].E2, _G["dec"].T2


def stage1(args):
    i, jlist = args
    PD, isfree = _G["PD"], _G["isfree"]
    E2, T2 = _G["E2"], _G["T2"]
    N = PD.shape[0]
    jok = np.zeros(N, dtype=np.int8)
    jok[np.asarray(jlist, dtype=np.int64)] = 1
    c, mb, par = C.kernel2(PD, C.INV, int(i), jok, isfree, 0, 3, C.ELL, MAXOUT)
    assert c.shape[0] < MAXOUT and mb.shape[0] < MAXOUT * 64, "stage-1 buffer overflow"
    assert len(par) == 0
    memb = {}
    for cid, k in mb:
        memb.setdefault(int(cid), []).append(int(k))
    fives = []
    for cid in range(c.shape[0]):
        j = int(c[cid, 0])
        mem = memb.get(cid, [])
        for sub in itertools.combinations(mem, 3):
            fives.append((i, j) + sub)
    nclass = c.shape[0]
    if not fives:
        return nclass, 0, []
    S = np.array(fives, dtype=np.int64)
    r1 = ranks_batch(E2, S, ELL2)
    keep = np.flatnonzero(r1 == 5)
    out = []
    if len(keep):
        S2 = np.hstack([S[keep], np.full((len(keep), 3), N, dtype=np.int64)])   # rows N.. = T
        E2T = np.vstack([E2, T2])
        r2 = ranks_batch(E2T, S2, ELL2)
        for t, rr in zip(keep, r2):
            out.append((tuple(int(x) for x in S[t]), int(rr)))
    return nclass, len(fives), out


def annihilator(rows, ell):
    """Basis (as rows) of {y : rows y = 0} mod ell; rows is k x n of rank k."""
    A = rows.copy() % ell
    k, n = A.shape
    piv = []
    r = 0
    for c in range(n):
        nz = np.flatnonzero(A[r:, c])
        if len(nz) == 0:
            continue
        pr = r + nz[0]
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        A[r] = (A[r] * pow(int(A[r, c]), ell - 2, ell)) % ell
        f = A[:, c].copy(); f[r] = 0
        A = (A - np.outer(f, A[r])) % ell
        piv.append(c)
        r += 1
        if r == k:
            break
    free = [c for c in range(n) if c not in piv]
    B = np.zeros((len(free), n), dtype=np.int64)
    for t, fc in enumerate(free):
        B[t, fc] = 1
        for a, pc in enumerate(piv):
            B[t, pc] = (-A[a, fc]) % ell
    return B


def project(E2, B, ell):
    """E2 @ B^T mod ell, chunked so products stay below 2^63."""
    N, n = E2.shape
    out = np.zeros((N, B.shape[0]), dtype=np.int64)
    for c0 in range(0, n, 4):
        out = (out + (E2[:, c0:c0 + 4] @ B[:, c0:c0 + 4].T) % ell) % ell
    return out


def stage2(fives):
    dec, E2, T2 = _G["dec"], _G["E2"], _G["T2"]
    N = E2.shape[0]
    found = []
    stats = {"honest": 0, "dishonest": 0, "pairs": 0, "inW7": 0, "decided": 0}
    for F5, r2 in fives:
        F5 = list(F5)
        if r2 == 6:
            stats["honest"] += 1
            W = np.vstack([E2[F5], T2])          # rank 6
            B = annihilator(W, ELL2)             # 21 x 27
            img = project(E2, B, ELL2)
            zero = np.flatnonzero(np.all(img == 0, axis=1))
            extra = [int(z) for z in zero if z not in F5]
            # states inside W6 other than F5 would give a rank-6 decomposition with F5 if
            # independent; decide the whole set to be safe
            if extra:
                ok, a, b = dec.contains(F5 + extra); stats["decided"] += 1
                if ok and a <= 7:
                    found.append((F5 + extra, a))
            groups = {}
            for s in range(N):
                row = img[s]
                nz = np.flatnonzero(row)
                if len(nz) == 0:
                    continue
                key = tuple(((row * pow(int(row[nz[0]]), ELL2 - 2, ELL2)) % ELL2).tolist())
                groups.setdefault(key, []).append(s)
            for g in groups.values():
                if len(g) >= 2:
                    stats["pairs"] += 1
                    ok, a, b = dec.contains(F5 + extra + g); stats["decided"] += 1
                    if ok:
                        found.append((F5 + extra + g, a))
        else:
            stats["dishonest"] += 1
            W = np.vstack([E2[F5], T2])          # rank 7
            B = annihilator(W, ELL2)             # 20 x 27
            img = project(E2, B, ELL2)
            inW = [int(z) for z in np.flatnonzero(np.all(img == 0, axis=1)) if z not in F5]
            stats["inW7"] += len(inW)
            if len(inW) >= 2:
                ok, a, b = dec.contains(F5 + inW); stats["decided"] += 1
                if ok:
                    found.append((F5 + inW, a))
    return found, stats, dec.how


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    d = load(3)
    N = d["N"]
    jok_of = {int(i): d["jokm"][t].astype(np.int8) for t, i in enumerate(d["reps"])}
    d = dict(d)
    t0 = time.time()
    # j-chunks per rep, front-loaded (work per j is N - j)
    tasks = []
    for i in d["reps"]:
        js = np.flatnonzero(jok_of[int(i)])
        for a in range(0, len(js), 40):
            tasks.append((int(i), js[a:a + 40].tolist()))
    rng = np.random.default_rng(0); rng.shuffle(tasks)
    print(f"{len(tasks)} stage-1 tasks over {sum(len(t[1]) for t in tasks)} pivot pairs", flush=True)
    fives = []; ncl = 0; nf = 0
    with ProcessPoolExecutor(max_workers=nw, initializer=_init, initargs=(d,)) as ex:
        for k, (nc, n5, out) in enumerate(ex.map(stage1, tasks)):
            ncl += nc; nf += n5; fives.extend(out)
            if k % 60 == 0:
                print(f"[{time.time()-t0:.0f}s] task {k}/{len(tasks)}: classes {ncl}, five-subsets {nf}, survivors {len(fives)}", flush=True)
        print(f"[{time.time()-t0:.0f}s] stage 1 done: {ncl} need-3 classes, {nf} five-subsets, {len(fives)} survive the exact independence filter", flush=True)
        hist = {}
        for F, r in fives:
            hist[r] = hist.get(r, 0) + 1
        print("survivor rank(F5 + T) histogram:", hist, flush=True)
        json.dump({"survivors": [(list(F), r) for F, r in fives], "classes": ncl, "fives": nf},
                  open(os.path.join(HERE, "logs", "caseB_stage1.json"), "w"))
        chunks = [fives[a:a + 50] for a in range(0, len(fives), 50)] or [[]]
        found = []; stats = {}; how = {}
        for f, s, h in ex.map(stage2, chunks):
            found.extend(f)
            for k2, v in s.items():
                stats[k2] = stats.get(k2, 0) + v
            for k2, v in h.items():
                how[k2] = how.get(k2, 0) + v
    print(f"[{time.time()-t0:.0f}s] stage 2: {stats}; decider {how}")
    if found:
        best = min(found, key=lambda f: f[1])
        print(f"FOUND span containing V_3 of rank {best[1]}: {best[0]}")
    else:
        print("CASE B EXCLUDED: no seven-state span containing V_3 has five images in a 3-dim space mod ELL")
    json.dump({"found": found, "stats": stats, "wall": time.time() - t0}, open(os.path.join(HERE, "logs", "caseB.json"), "w"))


if __name__ == "__main__":
    main()
