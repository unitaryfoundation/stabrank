"""Level 4 -> 5 for one kappa = 1 base (round 33, agent B): the family brute force over 243^5 option tuples, split
into NK ranges of the first term's options.
    python lift45_fam.py OUTDIR BASE K NK
"""
import numpy as np, os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L, lift
from lift34 import family

def main(outdir, bi, K, NK):
    fn = os.path.join(outdir, f'lift45_fam_{bi}_{K}.json')
    if os.path.exists(fn): print('exists'); return
    log = open(os.path.join(outdir, f'lift45_fam_{bi}_{K}.log'), 'a'); t0 = time.time()
    P = L.PRIMES[0]; tab = L.modtab(P); t4 = L.target(4) % P; lifter = lift.Lifter(5, P=P)
    rep = json.load(open(os.path.join(outdir, 'b4_reps.json')))['reps'][bi]; assert rep['kappa'] == 1
    T = np.array(rep['terms'], dtype=np.int8); V = tab[T]; b0, beta = family(V, np.arange(5), t4, P)
    nopt = 3 * 81; lo = (nopt * K) // NK; hi = (nopt * (K + 1)) // NK
    lifts = lift.lift_family(lifter, T, b0, beta, o0range=(lo, hi), log=log)
    out = []
    for d in lifts:
        v = lifter.verify(d['terms'], None, exact=True); out.append(dict(base=bi, terms=d['terms'].tolist(), verify=v, lam=(d['lam'] if d['lam'] == 'free' else int(d['lam'])))); print(f'[lift45_fam] FOUND from base {bi}: {v}', file=log, flush=True)
    tmp = fn + f'.tmp{os.getpid()}'
    json.dump(dict(base=bi, base_terms=rep['terms'], K=K, NK=NK, o0range=[lo, hi], found=out, seconds=time.time() - t0), open(tmp, 'w')); os.replace(tmp, fn)
    print(f'LIFT45_FAM DONE base {bi} range {lo}-{hi}: {len(out)} found, {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
