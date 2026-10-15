"""Planted control for the three-pivot kernel (case C) at m = 3.

Seven random dictionary states of exact rank 7 and a target V spanned by three random
vectors in their span, so that mod ELL the seven images span exactly 4 dims; general
position (no five images coplanar mod ELL) is checked and the trial is redrawn otherwise.
The kernel is run with first pivot s_1, second pivots {s_2} plus 5 random states, and
must emit a class containing all seven states."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cert7 as C, k3, caseB
from setup_m3 import load
from decide import ELL2, rank_mod
import itertools

d = load(3); N, E = d["N"], d["E"]
El = C.to_mod(E, C.Z6, C.ELL)
E2 = caseB.Decider(E, d["T"]).E2
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
ok_all = True
for trial in range(3):
    while True:
        S = sorted(int(x) for x in rng.choice(N, 7, replace=False))
        if rank_mod(E2[S], ELL2) != 7:
            continue
        V = (rng.integers(1, C.ELL, size=(3, 7)) @ El[S]) % C.ELL
        Ann = caseB.annihilator(V, C.ELL)
        Rm = rng.integers(0, C.ELL, size=(6, 24), dtype=np.int64)
        PD = caseB.project(caseB.project(El, Ann, C.ELL), Rm, C.ELL)
        if C.rank_mod(PD[S], C.ELL) != 4:
            continue
        if all(C.rank_mod(PD[list(F)], C.ELL) == 4 for F in itertools.combinations(S, 5)):
            break
    assert not np.any(np.all(PD == 0, axis=1))
    jlist = sorted(set([S[1]] + [int(x) for x in rng.choice(N, 5, replace=False)]))
    jok = np.zeros(N, dtype=np.int8); jok[jlist] = 1
    c, mb, ns = k3.kernel3(PD, C.INV, S[0], jok, np.ones(N, dtype=np.int8), 3, 400000, 0.0, 1.0)
    assert c.shape[0] < 400000 and mb.shape[0] < 400000 * 16
    hit = False
    for cid in range(c.shape[0]):
        if int(c[cid, 0]) == S[1]:
            mem = {int(s) for s in mb[mb[:, 0] == cid, 1]} | {S[0], S[1], int(c[cid, 1])}
            if set(S) <= mem:
                hit = True; size = len(mem); break
    ok_all &= hit
    print(f"trial {trial}: planted {S}; {ns:.2e} steps, {c.shape[0]} classes emitted; planted seven-set recovered: {hit}"
          + (f" (class set of {size} states)" if hit else ""), flush=True)
print("CONTROL C (planted) OK" if ok_all else "CONTROL C FAILED")
sys.exit(0 if ok_all else 1)
