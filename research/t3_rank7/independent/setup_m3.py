"""Build dictionary, quotient images, symmetry data for T3 at m = 2 or 3; cache to npz."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert7 as C

def load(m, D=6, seed=2024):
    cache = os.path.join(HERE, "logs", f"setup_m{m}.npz")
    E = C.build_dictionary(m)
    T = C.t3_targets(m)
    Vl, Tl = C.to_mod(E, C.Z6, C.ELL), C.to_mod(T, C.Z6, C.ELL)
    PD = C.quotient_projection(Vl, Tl, D, seed)
    isfree = np.all(PD == 0, axis=1).astype(np.int8)
    E2, T2 = C.to_mod(E, C.Z6_2, C.ELL2), C.to_mod(T, C.Z6_2, C.ELL2)
    if os.path.exists(cache):
        z = np.load(cache)
        reps, jokm = z["reps"], z["jokm"]
        nel = int(z["nel"])
    else:
        t0 = time.time()
        nel, reps, jok_of = C.symmetry(E, T, m, isfree)
        jokm = np.stack([jok_of[int(i)] for i in reps])
        np.savez(cache, reps=reps, jokm=jokm, nel=nel)
        print(f"symmetry computed in {time.time()-t0:.0f}s: order {nel}, {len(reps)} reps", flush=True)
    return dict(E=E, T=T, PD=PD, E2=E2, T2=T2, isfree=isfree, reps=reps, jokm=jokm, nel=nel, N=E.shape[0])

if __name__ == "__main__":
    m = int(sys.argv[1])
    d = load(m)
    N = d["N"]
    jokm = d["jokm"]
    # work estimate for the three-pivot scan: sum over allowed (i, j) of sum_{k>j} (N-k)
    tot = 0.0
    npairs = 0
    for row in jokm:
        js = np.flatnonzero(row)
        npairs += len(js)
        tot += np.sum((N - js - 1.0) * (N - js - 2.0) / 2)
    print(f"m={m}: N={N}, group order {d['nel']}, reps {len(d['reps'])}, pairs {npairs}, "
          f"three-pivot inner steps {tot:.3e}")
