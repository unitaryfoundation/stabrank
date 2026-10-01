"""Case A of the rank-7 exclusion: configurations with six images in a 3-dim space mod ELL.

Every such six-set is inside one of the rank-6 class sets C (two pivots plus the members
zero-or-parallel modulo the pivots; the dump T3m3_rank6_sym.json holds the unfiltered list
of 538,514 classes, a superset of the certificate's 259,423).  For a genuine rank-7
configuration the seven states are independent (rank 6 is excluded), so a six-subset F
containing the two pivots has rank(F) = 6 and rank(F + T) = 7 exactly (rank 6 would be a
rank-6 decomposition), and the seventh state b lies in span(F) + V_3 but not in span(F);
then span(F + b) = span(F) + V_3 contains V_3.  So: for every F with exact rank pair (6, 7),
list the dictionary states in span(F + T) (membership mod ELL2 is necessary, since the
denominators of Cramer's rule are 7 x 7 minors of norm < ELL2) and test each exactly.
"""
import itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert7 as C
from decide import Decider, rank_mod, ELL2

_G = {}

def _init(E, T):
    _G["dec"] = Decider(E, T)
    dec = _G["dec"]
    _G["E2T"] = dec.E2
    _G["T2"] = dec.T2

def nullspace_mod(M, ell):
    """Basis of {x : M x = 0} mod ell as columns; M is rows x cols."""
    A = M.copy() % ell
    rows, cols = A.shape
    piv = []
    r = 0
    for c in range(cols):
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
        if r == rows:
            break
    free = [c for c in range(cols) if c not in piv]
    B = np.zeros((cols, len(free)), dtype=np.int64)
    for t, fc in enumerate(free):
        B[fc, t] = 1
        for i, pc in enumerate(piv):
            B[pc, t] = (-A[i, fc]) % ell
    return B

def work(chunk):
    dec = _G["dec"]
    E2, T2 = _G["E2T"], _G["T2"]
    N = E2.shape[0]
    nF = 0; ndis = 0; found = []; hist = {}
    for piv, nz, ng, members in chunk:
        others = list(members)
        for sub in itertools.combinations(others, 4):
            F = list(piv) + list(sub)
            nF += 1
            r1, r2 = dec.ranks(F)
            hist[(r1, r2)] = hist.get((r1, r2), 0) + 1
            if r1 == 6 and r2 == 7:
                ndis += 1
                # states b with rank([F, T, b]) = 7 mod ELL2: b orthogonal to the null space of [F;T]
                M = np.vstack([E2[F], T2])                 # 9 x 27, rank 7
                Bn = nullspace_mod(M, ELL2)                # 27 x 20
                # chunked matmul mod ELL2 (entries < 2^31, 27 terms: < 2^67 overflow) -> split
                prod = np.zeros((N, Bn.shape[1]), dtype=np.int64)
                for c0 in range(0, 27, 4):
                    prod = (prod + (E2[:, c0:c0 + 4] @ Bn[c0:c0 + 4, :]) % ELL2) % ELL2
                cand = np.flatnonzero(np.all(prod == 0, axis=1))
                for b in cand:
                    b = int(b)
                    if b in F:
                        continue
                    ok, a, bb = dec.contains(F + [b])
                    if ok:
                        found.append((F, b, a))
    return nF, ndis, found, hist

def main():
    E = C.build_dictionary(3); T = C.t3_targets(3)
    classes = json.load(open(os.path.join(HERE, "..", "logs", "T3m3_rank6_sym.json")))["candidates"]
    print(f"{len(classes)} rank-6 classes loaded", flush=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    chunks = [classes[a:a + 2000] for a in range(0, len(classes), 2000)]
    t0 = time.time()
    tot = [0, 0]; found = []; hist = {}
    with ProcessPoolExecutor(max_workers=nw, initializer=_init, initargs=(E, T)) as ex:
        for k, (nF, ndis, f, h) in enumerate(ex.map(work, chunks)):
            tot[0] += nF; tot[1] += ndis; found.extend(f)
            for kk, v in h.items():
                hist[kk] = hist.get(kk, 0) + v
            if k % 20 == 0:
                print(f"[{time.time()-t0:.0f}s] chunk {k}/{len(chunks)}: six-subsets {tot[0]}, dishonest {tot[1]}, decompositions {len(found)}", flush=True)
    print("rank-pair histogram of six-subsets:", dict(sorted(hist.items())))
    print(f"six-subsets {tot[0]}, with exact rank pair (6,7): {tot[1]}, rank-7 decompositions found: {len(found)}")
    if found:
        print("FOUND", found[:5])
    else:
        print("CASE A EXCLUDED: no seven-state span containing V_3 has six images in a 3-dim space mod ELL")
    json.dump({"six_subsets": tot[0], "dishonest": tot[1], "found": found, "hist": {str(k): v for k, v in hist.items()}, "wall": time.time() - t0},
              open(os.path.join(HERE, "logs", "caseA.json"), "w"))

if __name__ == "__main__":
    main()
