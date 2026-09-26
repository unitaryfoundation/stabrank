"""Stage-1 control for case B with a planted configuration at m = 3.

Seven random dictionary states s_1..s_7 of exact rank 7; a target V = span(v1, v2, v3) mod
ELL with v1, v2 in span(s_1..s_5) and v3 in span(s_1..s_7).  Then the images of s_1..s_5
modulo V span 3 dims and all seven span 4: a rank-7 configuration with a coplanar
five-subset.  The need-3 kernel with first pivot s_1 and every j allowed, followed by the
exact independence filter, must return {s_1..s_5} among its survivors (several seeds)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert7 as C
import caseB
from setup_m3 import load
from decide import ELL2, rank_mod

d = load(3)
N, E = d["N"], d["E"]
El = C.to_mod(E, C.Z6, C.ELL)
E2 = caseB.Decider(E, d["T"]).E2
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
ok_all = True
for trial in range(3):
    while True:
        S = sorted(int(x) for x in rng.choice(N, 7, replace=False))
        if rank_mod(E2[S], ELL2) == 7:
            break
    a = rng.integers(1, C.ELL, size=(2, 5)); b = rng.integers(1, C.ELL, size=7)
    v12 = (a @ El[S[:5]]) % C.ELL
    v3 = (b @ El[S]) % C.ELL
    V = np.vstack([v12, v3])
    Ann = caseB.annihilator(V, C.ELL)                     # 24 x 27
    Rm = rng.integers(0, C.ELL, size=(6, 24), dtype=np.int64)
    PD = caseB.project(El, Ann, C.ELL)
    PD = caseB.project(PD, Rm, C.ELL)                    # N x 6
    # sanity: planted images have ranks 3 (first five) and 4 (all seven) mod ELL
    r5 = C.rank_mod(PD[S[:5]], C.ELL); r7 = C.rank_mod(PD[S], C.ELL)
    isfree = np.all(PD == 0, axis=1).astype(np.int8)
    caseB._init(dict(PD=PD, isfree=isfree, E=E, T=d["T"]))
    jlist = sorted(set([S[1]] + [int(x) for x in rng.choice(N, 40, replace=False)]))
    nc, n5, out = caseB.stage1((S[0], jlist))
    surv = {tuple(sorted(F)) for F, r in out}
    hit = tuple(S[:5]) in surv
    ok_all &= hit
    print(f"trial {trial}: planted {S}, image ranks {r5}/{r7}, free states {int(isfree.sum())}; "
          f"{nc} classes, {n5} five-subsets, {len(out)} survivors; planted five-set recovered: {hit}", flush=True)
print("CONTROL B (planted) OK" if ok_all else "CONTROL B FAILED")
sys.exit(0 if ok_all else 1)
