"""Level 4 -> 5 (round 33, agent B): every five-term decomposition of S^5 whose terms are all full on the last qutrit
(every five-term decomposition, by S0543) as a lift of a B_4 representative (b4_reps.json). kappa = 0 by the meet in
the middle, kappa = 1 by the family brute force (243^5 option tuples per base; parallel over the first term's options).
    python lift45.py OUTDIR [K NK]      # bases K::NK
"""
import numpy as np, os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L, lift
from lift34 import family

def main(outdir, K=0, NK=1, kappa0_only=False):
    log = open(os.path.join(outdir, f'lift45_{K}.log'), 'a'); t0 = time.time()
    P = L.PRIMES[0]; tab = L.modtab(P); t4 = L.target(4) % P
    lifter = lift.Lifter(5, P=P)
    reps = json.load(open(os.path.join(outdir, 'b4_reps.json')))['reps']
    out = []; n0 = 0; n1 = 0
    for bi in range(K, len(reps), NK):
        rep = reps[bi]; T = np.array(rep['terms'], dtype=np.int8)
        if rep['kappa'] == 0:
            b = np.array(rep['coef'], dtype=np.int64)
            if not rep['allnz']:
                print(f'[lift45] base {bi}: a zero coefficient (not a five-term decomposition); skipped', file=log, flush=True); continue
            lifts = lifter.lift(T, b); n0 += 1
        else:
            if kappa0_only:
                print(f'[lift45] base {bi}: kappa 1, left for lift45_fam.py', file=log, flush=True); continue
            V = tab[T]; b0, beta = family(V, np.arange(5), t4, P)
            lifts = lift.lift_family(lifter, T, b0, beta, log=log); n1 += 1
        for d in lifts:
            v = lifter.verify(d['terms'], None, exact=True)
            out.append(dict(base=bi, terms=d['terms'].tolist(), verify=v))
            print(f'[lift45] FOUND a five-term decomposition of S^5 from base {bi}: verify {v}', file=log, flush=True)
        print(f'[lift45] base {bi+1}/{len(reps)} (kappa {rep["kappa"]}, orbit {rep["orbit"]}, kinds {rep["kinds"]}): {len(lifts)} lifts; {time.time()-t0:.0f}s', file=log, flush=True)
    fn = os.path.join(outdir, f'lift45_{K}.json'); tmp = fn + f'.tmp{os.getpid()}'
    json.dump(dict(K=K, NK=NK, n_kappa0=n0, n_kappa1=n1, found=out), open(tmp, 'w')); os.replace(tmp, fn)
    print(f'LIFT45 DONE chunk {K}/{NK}: {len(out)} decompositions found over {n0} + {n1} bases', flush=True)

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--kappa0-only']
    main(args[0], *(int(x) for x in args[1:3]), kappa0_only='--kappa0-only' in sys.argv)
