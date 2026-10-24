"""Exact derivation of chi(S^3) >= 4 and chi(S^5) >= 5 with the round 33 machinery (agent B, after the referee).

chi(S^3) >= 4: the rank-4 census (census.py run 4, kernel 2) records, for every pivot pair (u, a) and every letter c
with zero residue modulo span(t_3, u, a) under both functionals, the triple (u, a, c); a three-term decomposition
{u, a, c} of t_3 would contain a pivot pair up to symmetry and have c in span(t_3, u, a), hence be recorded. Every
recorded triple is decided exactly: rank(u, a, c) and rank(u, a, c, t_3) modulo P1 (exact by Hadamard's bound,
Section 3 of README.md) and over Z[omega]. No pivot pair is dependent with t_3 (the kernel would have aborted), so
chi(S^3) >= 3 as well. Then chi(S^4) >= 4 by projection monotonicity.

chi(S^5) >= 5: by Lemma F.2 with chi(S^4) = 4 a four-term decomposition of t_5 has all terms full on every qutrit,
so its <1| slice on qutrit 5 is a four-term decomposition of t_4, one of the 9,936 (S0541), G_4-equivalent to one of
three representatives; the k = 4 lift (E2 by meet in the middle over the 81 rays and three phases per term, then E0)
of each representative has no solution. Positive control: the same lift one level down gives 3, 3, 9, 9 lifts of the
four representatives of t_3 (test_lift.py).
    python rank4_exclusion5.py OUTDIR
"""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L, lift
from census import decide

def main(outdir):
    t0 = time.time(); P = L.PRIMES[0]; log = open(os.path.join(outdir, 'rank4_exclusion5.log'), 'a')
    def say(m): print(m, flush=True); print(m, file=log, flush=True)
    codes = L.letters(3); V = L.modtab(P)[codes]; t3 = L.target(3) % P
    r4 = json.load(open(os.path.join(outdir, 'r4', 'chunk_0.json')))
    triples = sorted(set((x[0], x[1], x[3]) for x in r4['shorts'] if x[2] == -1 and x[3] >= 0))
    bad = 0; ranks = {}
    for (u, a, c) in triples:
        coef = np.zeros(6, dtype=np.int64); ii = np.array([u, a, c, 0, 0, 0], dtype=np.int64)
        rk, rkt, _ = decide(V, ii, 3, t3, P, coef)
        e1, e2 = L.exact_cover_check(codes[[u, a, c]], L.target(3))
        if (rk, rkt) != (e1, e2): say(f'  modular and exact ranks differ at {(u, a, c)}: {(rk, rkt)} vs {(e1, e2)}'); bad += 1
        if e2 == e1: say(f'  THREE-TERM DECOMPOSITION of t_3: {(u, a, c)}'); bad += 1
        ranks[(e1, e2)] = ranks.get((e1, e2), 0) + 1
    say(f'[rank4_exclusion5] chi(S^3) >= 4: {len(triples)} recorded triples (u, a, c) with c in span(t_3, u, a) mod P1; exact ranks (rank, rank with t_3): {ranks}; violations {bad}')
    # chi(S^5) >= 5: k = 4 lifts of the three representatives of the four-term decompositions of t_4
    D4 = np.load(os.path.join(outdir, 'decomp4_codes.npy')); info = json.load(open(os.path.join(outdir, 'test_lift.json')))
    starts = np.cumsum([0] + info['orbit_sizes'][:-1]); reps = [D4[s] for s in starts]
    lifter = lift.Lifter(5, P=P); nl = []
    for oi, D in enumerate(reps):
        b, allnz = lifter.solve_coefficients(D); assert b is not None and allnz
        lifts = lifter.lift(D, b); nl.append(len(lifts))
        for d in lifts: say(f'  FOUR-TERM DECOMPOSITION of t_5 from representative {oi}: {lifter.verify(d["terms"], b, exact=True)}')
    say(f'[rank4_exclusion5] chi(S^5) >= 5: k = 4 lifts of the three representatives of t_4: {nl}; {time.time()-t0:.0f}s')
    json.dump(dict(triples=len(triples), exact_ranks={str(k): v for k, v in ranks.items()}, violations=bad, lifts_k4=nl, seconds=time.time() - t0), open(os.path.join(outdir, 'rank4_exclusion5.json'), 'w'), indent=1)
    print('RANK4_EXCLUSION5 DONE' if bad == 0 and sum(nl) == 0 else 'RANK4_EXCLUSION5 FAILED', flush=True)
    return bad, nl

if __name__ == '__main__':
    main(sys.argv[1])
