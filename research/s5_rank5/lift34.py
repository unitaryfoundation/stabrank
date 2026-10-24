"""Level 3 -> 4 (round 33, agent B): every five-term decomposition of S^4 all of whose terms are full on the last
qutrit, as lifts of the B_3 representatives (b3_reps.json): kappa = 0 bases by the meet in the middle of lift.py,
kappa = 1 bases (a four-term decomposition plus a span letter or a repeated letter) by the family brute force.
Writes out/lift34.json: the level-4 five-term decompositions found (terms as four-qutrit codes), with provenance.
    python lift34.py OUTDIR
"""
import numpy as np, os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L, lift
from census import decide

def family(V, idx, t, P):
    """for a dependent 5-multiset (rank 4, t in span): b0 (a particular solution with the last letter's coefficient 0
    if possible) and beta (the dependency), mod P. Letters may repeat."""
    # solve on the 4 independent letters: find an independent 4-subset containing t in its span
    import itertools
    for sub in itertools.combinations(range(5), 4):
        coef = np.zeros(6, dtype=np.int64); ii = np.array([idx[j] for j in sub] + [0, 0], dtype=np.int64)
        rank, rank_t, allnz = decide(V, ii, 4, t, P, coef)
        if rank == 4 and rank_t == 4:
            other = [j for j in range(5) if j not in sub][0]
            # express the other letter in the span of sub: solve V[other] = sum gamma_j V[sub_j]
            gam = np.zeros(6, dtype=np.int64)
            rank2, rank_t2, _ = decide(V, ii, 4, V[idx[other]] % P, P, gam)
            assert rank2 == 4 and rank_t2 == 4
            b0 = np.zeros(5, dtype=np.int64); beta = np.zeros(5, dtype=np.int64)
            for q, j in enumerate(sub): b0[j] = coef[q]; beta[j] = (-gam[q]) % P
            beta[other] = 1
            return b0, beta
    raise RuntimeError('no independent 4-subset with t in its span')

def main(outdir):
    log = open(os.path.join(outdir, 'lift34.log'), 'a'); t0 = time.time()
    P = L.PRIMES[0]; codes = L.letters(3); V = L.modtab(P)[codes]; t = L.target(3) % P
    lifter = lift.Lifter(4, P=P)
    reps = json.load(open(os.path.join(outdir, 'b3_reps.json')))['reps']
    out = []; stats = dict(indep=0, dep=0, repeat=0, lifts_indep=0, lifts_dep=0, lifts_repeat=0)
    for bi, rep in enumerate(reps):
        idx = rep['letters']; base = codes[idx]; kind = rep['kind']; stats[kind] += 1
        if kind == 'indep':
            b = np.array(rep['coef'], dtype=np.int64)
            lifts = lifter.lift(base, b)
            for d in lifts:
                v = lifter.verify(d['terms'], b); assert v['ok'], v
                out.append(dict(src=bi, kind=kind, terms=d['terms'].tolist()))
            stats['lifts_indep'] += len(lifts)
        else:
            b0, beta = family(V, idx, t, P)
            lifts = lift.lift_family(lifter, base, b0, beta, log=log)
            for d in lifts:
                v = lifter.verify(d['terms'], None); assert v['ok'], v
                out.append(dict(src=bi, kind=kind, terms=d['terms'].tolist(), lam=(d['lam'] if d['lam'] == 'free' else int(d['lam']))))
            stats['lifts_' + kind] += len(lifts)
        if bi % 10 == 0 or lifts:
            print(f'[lift34] base {bi+1}/{len(reps)} ({kind}, orbit {rep["orbit"]}): {len(lifts)} lifts; totals {stats}; {time.time()-t0:.0f}s', file=log, flush=True)
    json.dump(dict(stats=stats, decomps=out), open(os.path.join(outdir, 'lift34.json'), 'w'))
    print(f'[lift34] done: {stats}; {time.time()-t0:.0f}s', file=log, flush=True)
    print(f'LIFT34 DONE {stats}', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
