"""Case-C pivot structure under the exact group G' (logs/sym_exact.npz) and the exact
inner-step count of the three-pivot scan.

Canonical form (proved in REPORT-symmetry.md): a seven-set F in general position has a
labelling g F, g in G', with
  i = the G'-orbit minimum of some member, i in gF;
  j = min(gF \\ {i}), minimal in its Stab(i)-orbit;
  k = min(gF \\ {i, j}), minimal in its Stab(i, j)-orbit;
  all other members > k.
The scan visits (i, j, k) over exactly these triples and members k' > k, so its inner
step count is sum over allowed (i, j, k) of (N - k - 1).
Output: logs/costC.npz with reps, jokm (per rep), and for the pairs (i, j) with a
nontrivial Stab(i, j) the allowed-k mask; pairs not listed have every k > j allowed.
"""
import os, sys, time, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from symact import Group

def masks_from_perms(N, i, stab_perms):
    """jok for rep i from the permutations of Stab(i); and for each allowed j the
    Stab(i, j) permutations (as indices into stab_perms)."""
    if len(stab_perms) == 0:
        jok = np.ones(N, dtype=np.int8); jok[i] = 0
        return jok, {}
    P = np.stack(stab_perms)                      # (g, N)
    imin = P.min(axis=0)
    jok = ((imin == np.arange(N)) & (np.arange(N) != i)).astype(np.int8)
    # pairs with nontrivial Stab(i, j): j fixed by some nontrivial element
    nontriv = [g for g in range(len(stab_perms)) if not np.array_equal(P[g], np.arange(N))]
    pair_stab = {}
    for g in nontriv:
        for j in np.flatnonzero(P[g] == np.arange(N)):
            if jok[j]:
                pair_stab.setdefault(int(j), []).append(g)
    return jok, pair_stab

def kok_of(N, j, perms):
    P = np.stack(perms)
    kmin = P.min(axis=0)
    kok = (kmin == np.arange(N)).astype(np.int8)
    kok[:j + 1] = 0
    return kok

def main():
    t0 = time.time()
    G = Group(); N = G.N
    reps = np.unique(G.roots)                    # orbit minima
    print(f"|G'| = {G.order}, {len(reps)} orbits", flush=True)
    jokm = []; koks = {}; steps = 0; npairs = 0; ntrip = 0; stab_sizes = []
    for r, i in enumerate(reps):
        i = int(i)
        sidx = G.stabilizer_indices(i)
        stab_sizes.append(len(sidx))
        assert G.order % len(sidx) == 0 and G.order // len(sidx) == int(np.sum(G.roots == i))
        perms = [G.act(*G.element(n)) for n in sidx]
        for P in perms:
            assert P[i] == i
        jok, pair_stab = masks_from_perms(N, i, perms)
        jokm.append(jok)
        js = np.flatnonzero(jok)
        npairs += len(js)
        for j in js:
            j = int(j)
            if j in pair_stab:
                kok = kok_of(N, j, [perms[g] for g in pair_stab[j]])
                koks[(i, j)] = kok
                ks = np.flatnonzero(kok)
            else:
                ks = np.arange(j + 1, N)
            ks = ks[ks != i]
            ntrip += len(ks)
            steps += int(np.sum(N - ks - 1))
        print(f"[{time.time()-t0:.0f}s] rep {r}/{len(reps)} state {i}: |Stab(i)| = {len(sidx)}, allowed j {len(js)}, "
              f"pairs with nontrivial Stab(i,j) {len(pair_stab)}; cumulative steps {steps:.4e}", flush=True)
    jokm = np.stack(jokm)
    print(f"reps {len(reps)}, pivot pairs {npairs}, pivot triples {ntrip}, inner steps {steps:.4e}; "
          f"stabilizer sizes {sorted(set(stab_sizes))}; pairs with nontrivial Stab(i,j): {len(koks)}")
    for ns in (35, 52):
        print(f"  at {ns} ns/step: {steps*ns*1e-9/3600:.1f} CPU-hours")
    np.savez(os.path.join(HERE, "logs", "costC.npz"), reps=reps, jokm=jokm, steps=steps, order=G.order)
    pickle.dump(koks, open(os.path.join(HERE, "logs", "costC_kok.pkl"), "wb"))

if __name__ == "__main__":
    main()
