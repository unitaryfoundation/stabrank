"""Assemble the level-3 base list B_3 (round 33, agent B): every 5-multiset of three-qutrit letters whose span
contains t = S^3 with a coefficient family that is generically nonzero in every entry, up to the symmetry group G_3:
  (i)   independent full 5-covers (the census through the pivot pairs, closed under G_3), kappa = 0;
  (ii)  a four-term decomposition D of S^3 plus a letter in its span, kappa = 1;
  (iii) a four-term decomposition D plus one of its own letters repeated, kappa = 1.
A 5-multiset with S^3 in its span has rank 4 or 5 (rank <= 3 is impossible since chi(S^3) = 4); rank 4 means some
four letters are independent with t in their span, that is a four-term decomposition with all coefficients nonzero,
so (ii) and (iii) are the only dependent cases.
    python b3.py OUTDIR            # needs OUTDIR/r4/chunk_0.json (the 4-cover census) and OUTDIR/r5/*.json
"""
import numpy as np, os, sys, json, glob, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L
from census import decide

def closure(sets, maps):
    seen = {}; orbs = []
    for s in sets:
        s = tuple(sorted(s))
        if s in seen: continue
        orb = [s]; seen[s] = len(orbs); k = 0
        while k < len(orb):
            x = orb[k]; k += 1
            for m in maps:
                y = tuple(sorted(int(m[i]) for i in x))
                if y not in seen: seen[y] = len(orbs); orb.append(y)
        orbs.append(orb)
    return orbs, seen

def main(outdir):
    log = open(os.path.join(outdir, 'b3.log'), 'a'); t0 = time.time()
    P = L.PRIMES[0]; codes = L.letters(3); N = len(codes); lk = L.Lookup(codes); V = L.modtab(P)[codes]; t = L.target(3) % P
    gens = L.generators(3); maps = L.letter_maps(codes, gens, lk)
    # (0) the four-term decompositions from the r = 4 census
    c4 = json.load(open(os.path.join(outdir, 'r4', 'chunk_0.json')))
    dec4_orbs, dec4_seen = closure([tuple(s) for s in c4['covers']], maps)
    print(f'[b3] 4-covers through pivot pairs: {len(c4["covers"])}; after closure {len(dec4_seen)} four-term decompositions in {len(dec4_orbs)} orbits, sizes {[len(o) for o in dec4_orbs]}', file=log, flush=True)
    assert len(dec4_seen) == 2160 and sorted(len(o) for o in dec4_orbs) == [144, 432, 432, 1152]
    dec4_reps = [list(o[0]) for o in dec4_orbs]
    # (i) independent full 5-covers
    files = sorted(glob.glob(os.path.join(outdir, 'r5', 'chunk_*.json')))
    cov5 = []; prefixes = 0; cands = 0; shorts = 0; npairs = 0; bskips = []; pairs_seen = set()
    for fn in files:
        d = json.load(open(fn)); assert d.get('kernel') == 2, ('chunk from an old kernel', fn)
        cov5 += [tuple(s) for s in d['covers']]; prefixes += d['prefixes']; cands += d['candidates']; shorts += d['n_shorts']; npairs += len(d['pairs'])
        pairs_seen |= set(tuple(p) for p in d['pairs'])
        bskips += [tuple(x) for x in d['bskips']] if 'bskips' in d else [tuple(x[:3]) for x in d['shorts'] if x[3] == -1]      # prefixes (u, a, b) skipped because b in span(t, u, a) modulo P1
    # the one one-prime skip: recheck every skipped prefix modulo P2; a prefix independent modulo P2 is rerun with P2
    P2 = [q for q in L.PRIMES if q != P][0]; V2 = L.modtab(P2)[codes]; t2 = L.target(3) % P2
    from census import run_pairs
    flagged = set()
    for (u, a, b) in set(bskips):
        coef = np.zeros(6, dtype=np.int64); ii = np.array([u, a, b, 0, 0, 0], dtype=np.int64)
        rank, rank_t, _ = decide(V2, ii, 3, t2, P2, coef)
        if rank == 3 and rank_t == 4: flagged.add((u, a))          # b independent of (t, u, a) modulo P2: the P1 skip was a rank drop
    print(f'[b3] {len(set(bskips))} skipped prefixes (b in span(t, u, a) mod P1); {len(flagged)} pivot pairs flagged by the P2 recheck', file=log, flush=True)
    if flagged:
        cov_p2, shorts_p2, totals_p2 = run_pairs(V2, t2, sorted(flagged), 5, P2, seed=77, log=log, tag='[b3 P2 rerun] ', CODES=codes)
        cov5 += [tuple(s) for s in cov_p2]
        print(f'[b3] P2 rerun of {len(flagged)} pairs: {len(cov_p2)} covers', file=log, flush=True)
    print(f'[b3] r5 census: {len(files)} chunks, {npairs} pivot pairs ({len(pairs_seen)} distinct), {prefixes} prefixes, {cands} candidates, {len(cov5)} covers (with repetition), {shorts} shorts; {time.time()-t0:.0f}s', file=log, flush=True)
    cov5 = sorted(set(cov5))
    orbs5, seen5 = closure(cov5, maps) if cov5 else ([], {})
    print(f'[b3] independent full 5-covers: {len(cov5)} distinct through pivot pairs; {len(seen5)} after closure in {len(orbs5)} orbits; {time.time()-t0:.0f}s', file=log, flush=True)
    # (ii) span letters of the four-term decompositions: letters e with e in span(D); test all letters against each orbit rep
    dep = []
    for oi, rep in enumerate(dec4_reps):
        # residues of all letters modulo span(D): reduce with a basis of D (mod P) -- a letter is in the span iff its residue vanishes
        from census import basis_add, functional_mod_basis
        B = np.zeros((6, 27), dtype=np.int64); pc = np.zeros(6, dtype=np.int64); nb = 0
        for i in rep: nb = basis_add(B, pc, nb, V[i], P)
        assert nb == 4
        # residue zero test: v - sum_k v[pc_k] B_k == 0
        Vr = V.copy()
        for k in range(nb): Vr = (Vr - np.outer(V[:, pc[k]], B[k])) % P
        inspan = np.flatnonzero(~Vr.any(axis=1))
        extra = [int(e) for e in inspan if e not in rep]
        # exact confirmation of each span letter over Z[omega]
        for e in extra:
            r1, r2 = L.exact_cover_check(codes[list(rep) + [e]], L.target(3)); assert r1 == 4 and r2 == 4, (rep, e, r1, r2)
        print(f'[b3] dec4 orbit {oi} (size {len(dec4_orbs[oi])}): {len(inspan)} letters in the span of the representative ({len(extra)} beyond its own)', file=log, flush=True)
        for e in extra: dep.append(tuple(sorted(list(rep) + [e])))
    orbsd, seend = closure(dep, maps)
    print(f'[b3] dependent bases (4-dec + span letter): {len(dep)} through reps, {len(seend)} after closure in {len(orbsd)} orbits', file=log, flush=True)
    # (iii) repeats
    rep_bases = []
    for oi, rep in enumerate(dec4_reps):
        for i in rep: rep_bases.append(tuple(sorted(list(rep) + [i])))
    orbsr, seenr = closure(rep_bases, maps)
    print(f'[b3] repeat bases (4-dec + repeated letter): {len(rep_bases)} through reps, {len(seenr)} after closure in {len(orbsr)} orbits', file=log, flush=True)
    # coefficient data for the representatives
    def coeffs(idx):
        coef = np.zeros(6, dtype=np.int64); ii = np.array(list(idx) + [0] * (6 - len(idx)), dtype=np.int64)
        rank, rank_t, allnz = decide(V, ii, len(idx), t, P, coef); return rank, rank_t, allnz, [int(x) for x in coef[:len(idx)]]
    reps_out = []
    for orb in orbs5:
        rank, rank_t, allnz, coef = coeffs(orb[0]); assert rank == 5 and rank_t == 5 and allnz
        reps_out.append(dict(kind='indep', letters=list(orb[0]), orbit=len(orb), coef=coef))
    for orb in orbsd:
        reps_out.append(dict(kind='dep', letters=list(orb[0]), orbit=len(orb)))
    for orb in orbsr:
        reps_out.append(dict(kind='repeat', letters=list(orb[0]), orbit=len(orb)))
    json.dump(dict(dec4_reps=dec4_reps, dec4_orbit_sizes=[len(o) for o in dec4_orbs], n_indep=len(seen5), n_indep_orbits=len(orbs5),
                   indep_orbit_sizes=[len(o) for o in orbs5], n_dep=len(seend), n_dep_orbits=len(orbsd), n_repeat=len(seenr), n_repeat_orbits=len(orbsr),
                   reps=reps_out, census=dict(chunks=len(files), pairs=npairs, prefixes=prefixes, candidates=cands, covers_through_pairs=len(cov5), shorts=shorts)),
              open(os.path.join(outdir, 'b3_reps.json'), 'w'))
    np.save(os.path.join(outdir, 'b3_indep_all.npy'), np.array(sorted(seen5.keys()), dtype=np.int64))
    print(f'[b3] B_3 representatives: {len(orbs5)} independent, {len(orbsd)} dependent, {len(orbsr)} repeat; {time.time()-t0:.0f}s', file=log, flush=True)
    print(f'B3 DONE {len(orbs5)} {len(orbsd)} {len(orbsr)}', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
