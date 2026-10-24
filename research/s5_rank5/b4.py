"""Assemble the level-4 base list B_4 (round 33, agent B): every 5-multiset of four-qutrit letters whose span contains
S^4 with a generically nonzero coefficient family, as (a) the all-full lifts of B_3 (lift34.json), (b) the product-
pattern decompositions (prod4.json), (c) a four-term decomposition of S^4 plus a letter in its span (scan of all
7,439,040 four-qutrit letters against the three orbit representatives), (d) a four-term decomposition plus a repeated
letter. Deduplicated as multisets of canonical codes and reduced under G_4 (closure through the dense action of the
generators on the terms). Writes out/b4_reps.json with kind, kappa, coefficients.
    python b4.py OUTDIR
"""
import numpy as np, os, sys, json, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L
from census import decide, basis_add

def key(T): return tuple(sorted(c.tobytes() for c in T))

def main(outdir):
    log = open(os.path.join(outdir, 'b4.log'), 'a'); t0 = time.time()
    P = L.PRIMES[0]; tab = L.modtab(P); t4 = L.target(4) % P
    bases = {}    # key -> dict(terms, kinds)
    def add(T, kind):
        T = L.canon(np.asarray(T, dtype=np.int8)); k = key(T)
        if k not in bases: bases[k] = dict(terms=T, kinds=set())
        bases[k]['kinds'].add(kind)
    d34 = json.load(open(os.path.join(outdir, 'lift34.json')))
    for d in d34['decomps']: add(d['terms'], 'lift34-' + d['kind'])
    print(f'[b4] lift34: {len(d34["decomps"])} decompositions -> {len(bases)} distinct multisets', file=log, flush=True)
    dp = json.load(open(os.path.join(outdir, 'prod4.json')))
    for d in dp['bases']: add(d['terms'], 'prod4-' + d['case'])
    print(f'[b4] prod4: {len(dp["bases"])} -> {len(bases)} distinct', file=log, flush=True)
    # (c), (d): the three orbit representatives of the four-term decompositions of S^4 (test_lift.py output)
    D4 = np.load(os.path.join(outdir, 'decomp4_codes.npy')); info = json.load(open(os.path.join(outdir, 'test_lift.json')))
    starts = np.cumsum([0] + info['orbit_sizes'][:-1]); reps4 = [D4[s] for s in starts]
    cache = os.path.join(outdir, 'dec4_span_scan.json')
    if os.path.exists(cache):
        scan = json.load(open(cache))['scan']; print(f'[b4] span scan loaded from {cache}', file=log, flush=True)
    else:
        codes4 = L.letters(4); N4 = len(codes4)
        print(f'[b4] {N4} four-qutrit letters enumerated; {time.time()-t0:.0f}s', file=log, flush=True)
        scan = []
        for oi, D in enumerate(reps4):
            B = np.zeros((6, 81), dtype=np.int64); pc = np.zeros(6, dtype=np.int64); nb = 0
            for c in D: nb = basis_add(B, pc, nb, tab[c], P)
            assert nb == 4
            found = []
            for s in range(0, N4, 200000):
                Vc = tab[codes4[s:s + 200000]]
                for k in range(nb): Vc = (Vc - np.outer(Vc[:, pc[k]], B[k])) % P
                found += [int(s + i) for i in np.flatnonzero(~Vc.any(axis=1))]
            own = set(key([c]) for c in D)
            extra = [codes4[e].tolist() for e in found if key([codes4[e]]) not in own]
            scan.append(dict(rep=oi, n_in_span=len(found), extra=extra))
            print(f'[b4] dec4 rep {oi}: {len(found)} letters in the span ({len(extra)} beyond its own); {time.time()-t0:.0f}s', file=log, flush=True)
        del codes4
        json.dump(dict(N4=N4, scan=scan), open(cache, 'w'))
    for oi, D in enumerate(reps4):
        for e in scan[oi]['extra']:
            e = np.array(e, dtype=np.int8)
            r1, r2 = L.exact_cover_check(np.vstack([D, e[None, :]]), L.target(4)); assert r1 == 4 and r2 == 4
            add(np.vstack([D, e[None, :]]), 'dec4+span')
        for i in range(4): add(np.vstack([D, D[i][None, :]]), 'dec4+repeat')
        print(f'[b4] dec4 rep {oi}: {scan[oi]["n_in_span"]} letters in the span ({len(scan[oi]["extra"])} beyond its own)', file=log, flush=True)
    # no reduction under G_4: the lifts of the level-3 representatives already contain a G_3-image of every all-full
    # level-4 base, and prod4 one of every product-pattern base, so lifting every distinct multiset is complete
    # (redundant only up to G_4 / G_3); orbits under G_4 can be of order 10^6 and are not enumerated.
    reps = []
    for k, rec in bases.items():
        T = rec['terms']; Vt = tab[T]; idx = np.arange(5, dtype=np.int64); coef = np.zeros(6, dtype=np.int64)
        rank, rank_t, allnz = decide(Vt, idx, 5, t4, P, coef)
        assert rank_t == rank, 'target not in span'
        reps.append(dict(terms=T.tolist(), orbit=None, kinds=sorted(rec['kinds']), rank=int(rank), kappa=5 - int(rank), coef=[int(x) for x in coef[:5]] if rank == 5 else None, allnz=bool(allnz)))
    json.dump(dict(n_multisets=len(bases), n_reps=len(reps), reps=reps), open(os.path.join(outdir, 'b4_reps.json'), 'w'))
    print(f'[b4] {len(bases)} distinct level-4 bases (no G_4 reduction); kappa counts {dict((kk, sum(1 for r in reps if r["kappa"] == kk)) for kk in (0, 1))}; {time.time()-t0:.0f}s', file=log, flush=True)
    print(f'B4 DONE {len(bases)} {len(reps)}', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
