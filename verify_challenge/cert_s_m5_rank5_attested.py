"""Certificate: chi(|S>^5) >= 6 for the Strange state S = (|1> - |2>)/sqrt 2 (and chi(|S>^m) >= 6 for m = 6, 7, 8
by projection monotonicity), by the exclusion of every five-term stabilizer decomposition of |S>^5 through a
recursive slice-and-lift over non-minimal covers, over stored enumerations (attested tier).

The argument (research/s5_rank5/README.md in the research record, attacks/round33-strange-rank5). Write t_n for the
unnormalized |S>^n; its slices on the last qutrit are 0, t_{n-1}, -t_{n-1}. Every five-term decomposition of t_5 has
all terms full on every qutrit (board: cert_s_m5_lift.py gives chi(S^5) >= 5; the research record's Theorem F.3,
cards S0540 to S0543, which this script re-derives in the parts it uses through Lemma F.2 below). A full term is
zeta' P^-1 w (x) |0> + w (x) |1> + zeta P w (x) |2> with w its <1| slice, P a Pauli (only its coset modulo the
stabilizer of w matters: 3^(n-1) rays) and zeta, zeta' cube roots of unity. So a five-term decomposition of t_5 is a
lift of its <1| slice, a five-term decomposition of t_4 with all coefficients nonzero (a level-4 base), under
    E2: sum_j b_j zeta_j P_j w_j = -t_4,     E0: sum_j b_j zeta'_j P_j^-1 w_j = 0.
Level-4 bases are classified on qutrit 4 by Lemma F.2 (|F| + |P_1| >= chi(S^3) = 4, |F| + |P_2| >= 4,
|F| + |P_0| != 1): either all five terms are full there, and the base is a lift of a level-3 base (a 5-multiset of
three-qutrit letters with t_3 in its span and generic coefficients nonzero), or one slice is one of the 2,160
four-term decompositions of t_3 and the base is a completion of it (cases A, B, C of the research record). Level-3
bases are independent full 5-covers (the stored census) or, since chi(S^3) = 4, a four-term decomposition of t_3 plus
a letter of its span (none exist) or a repeated letter. All lifts are solved exactly: meet in the middle on two random
functionals modulo P1 = 2013265921 for unique coefficients, a brute force over all option tuples with a determinant
screen for one-parameter families; every hit is re-decided modulo P1 and P2 = 1811939329, representatives and
survivors exactly over Z[omega]. Nothing survives at level 5, so chi(S^5) >= 6.

Premises re-derived here, exactly: chi(S^3) >= 4 (the 412 triples recorded by the rank-4 census as candidates for a
three-term decomposition all have exact rank 2 without and 3 with t_3), hence chi(S^4) >= 4 by projection; chi(S^5) >= 5
(a four-term decomposition of t_5 is all full by Lemma F.2, its <1| slice is one of the 9,936 four-term decompositions
of t_4, and the k = 4 lift of each of their three representatives has no solution), so a decomposition with four or
fewer terms makes this script fail; and Theorem F.3 (S0543, a floating-point computation in round 30): the
product-pattern completions at level 5 (cases A, B, C over the three representatives of t_4 with the complete table of
7,439,040 four-qutrit letters) have no solution at any stage (prod5.py). All modular decisions in this work are exact,
not merely sound: every matrix decided has at most six rows with entries of modulus at most 1, so by Hadamard's bound
its minors have norm at most 6^6 = 46,656 against P of order 2e9, and a nonzero minor stays nonzero mod P (the
referee's observation, research record Section 3).

What this script re-runs (seed printed below): the letters, the symmetry group and the 14,638 pivot pairs; the rank-4
census over every pair and its closure (2,160 decompositions in four orbits); the stored census chunks are checked by
hash and loaded, RECHECK_CHUNKS whole chunks chosen from the seed are re-enumerated from scratch with fresh functionals
and compared cover for cover, RECHECK_SLICES further prefix slices likewise, the 412 one-prime prefix skips are
rechecked modulo P2; the covers are closed under the group (2,013,120 in 154 orbits),
every representative confirmed exactly; the repeat bases; the lifts of every level-3 representative to level 4; the
four-term decompositions of t_4 (9,936 in three orbits) and the product-pattern completions (756); the span scan of
all 7,439,040 four-qutrit letters; the assembly of the 2,609 level-4 bases, compared with the stored lists; the stored
level-5 kappa = 0 results are checked to cover every base with nothing found, RECHECK_LIFT_CHUNKS whole chunks chosen from the seed are
re-lifted from scratch and RECHECK_BASES further bases likewise; the stored kappa = 1 family results are checked to cover
all 35 x 243 ranges with nothing found and RECHECK_RANGES ranges are recomputed. The bound file's `recomputed` counts the
whole work units (RECHECK_CHUNKS + RECHECK_LIFT_CHUNKS); the slices, the extra bases and the family ranges are partial
rechecks on top of them.

Printed claims: CERTIFIED chi(S^5) >= 6
                CERTIFIED chi(S^6) >= 6, CERTIFIED chi(S^7) >= 6, CERTIFIED chi(S^8) >= 6 (projection monotonicity)
"""
import hashlib, json, os, sys, time, tempfile, shutil, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'research', 's5_rank5')
sys.path.insert(0, PKG)
import s5lib as L            # noqa: E402
import census as C           # noqa: E402
import lift                  # noqa: E402
import b3 as B3              # noqa: E402
import prod4 as P4           # noqa: E402
import lift34 as L34         # noqa: E402
import b4 as B4              # noqa: E402
import rank4_exclusion5 as R45   # noqa: E402
import prod5 as P5           # noqa: E402

SEED = 20261023
RECHECK_SLICES, SLICE_WIDTH, RECHECK_BASES, RECHECK_RANGES = 3, 100, 120, 4
RECHECK_CHUNKS, RECHECK_LIFT_CHUNKS = 1, 2     # whole work units re-run from scratch (the 'recomputed' count)
MANIFEST = os.path.join(PKG, 'batch_manifest.json')
RES = os.path.join(PKG, 'results')
T0 = time.time()

def say(msg): print(f'[{time.time()-T0:7.0f}s] {msg}', flush=True)
def fail(msg):
    print(f'FAILED: {msg}', file=sys.stderr, flush=True); raise SystemExit(1)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def main():
    print(f'seed: {SEED}', flush=True)
    rng = np.random.default_rng(SEED)
    man = json.load(open(MANIFEST)); by_stage = {}
    for b in man['batches']:
        p = os.path.join(ROOT, b['output'])
        if not os.path.isfile(p) or sha(p) != b['sha256']: fail(f'stored output {b["output"]} missing or does not hash as recorded')
        by_stage.setdefault(b['params']['stage'], []).append(b)
    say(f'manifest: {len(man["batches"])} stored outputs verified by hash ({ {k: len(v) for k, v in by_stage.items()} })')
    tmp = tempfile.mkdtemp(prefix='s5cert_')
    try:
        run(tmp, rng, by_stage)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def run(tmp, rng, by_stage):
    P1, P2 = L.PRIMES
    # 1. letters, symmetry, pivot pairs
    codes = L.letters(3); N = len(codes)
    if N != 30240: fail('letter count')
    C.stage_pairs(tmp)
    pairs = [tuple(p) for p in json.load(open(os.path.join(tmp, 'pairs.json')))['pairs']]
    stored_pairs = [tuple(p) for p in json.load(open(os.path.join(RES, 'pairs.json')))['pairs']]
    if pairs != stored_pairs: fail('pivot pairs differ from the stored list')
    say(f'{len(pairs)} pivot pairs recomputed and equal to the stored list')
    # 2. rank-4 census and closure
    C.stage_run(4, tmp, 0, 1, P1)
    # 3. stored census chunks -> tmp/r5 (b3 reads them there), re-enumerated slices, P2 recheck of the skips (inside b3)
    os.makedirs(os.path.join(tmp, 'r5'), exist_ok=True)
    chunks = sorted(by_stage['census'], key=lambda b: b['params']['K'])
    if [b['params']['K'] for b in chunks] != list(range(4096)): fail('census chunks are not 0..4095')
    for b in chunks: shutil.copy(os.path.join(ROOT, b['output']), os.path.join(tmp, 'r5', f'chunk_{b["params"]["K"]}.json'))
    covered = set()
    for b in chunks: covered |= set(tuple(p) for p in json.load(open(os.path.join(ROOT, b['output'])))['pairs'])
    if covered != set(pairs): fail('the stored chunks do not cover every pivot pair exactly once')
    V1 = L.modtab(P1)[codes]; t1 = L.target(3) % P1
    three_pair = [b['params']['K'] for b in chunks if len(b['params']['pairs']) == 3 and b['params']['covers'] > 0] or [b['params']['K'] for b in chunks if len(b['params']['pairs']) == 3]
    for s in range(RECHECK_CHUNKS):
        K = int(rng.choice(three_pair)); d = json.load(open(os.path.join(tmp, 'r5', f'chunk_{K}.json')))
        cov, shorts, tot = C.run_pairs(V1, t1, [tuple(p) for p in d['pairs']], 5, P1, seed=SEED * 7 + K, CODES=codes)
        if set(tuple(x) for x in cov) != set(tuple(x) for x in d['covers']): fail(f'whole chunk {K} re-enumerated with fresh functionals disagrees with the stored covers')
        if set(tuple(x[:3]) for x in shorts if x[3] == -1) != set(tuple(x) for x in d['bskips']): fail(f'whole chunk {K}: prefix skips disagree')
        say(f'census whole-chunk recheck {s+1}: chunk {K} ({len(d["pairs"])} pivot pairs, {int(tot[0])} prefixes, fresh functionals): {len(d["covers"])} covers, agree')
    with_covers = [b['params']['K'] for b in chunks if b['params']['covers'] > 0]
    for s in range(RECHECK_SLICES):
        # two of the three slices are drawn around a stored cover (a positive recheck), the third anywhere
        K = int(rng.choice(with_covers)) if s < 2 else int(rng.integers(0, 4096))
        d = json.load(open(os.path.join(tmp, 'r5', f'chunk_{K}.json')))
        if s < 2:
            c0 = d['covers'][int(rng.integers(0, len(d['covers'])))]
            u, a = next((tuple(p) for p in d['pairs'] if p[0] in c0 and p[1] in c0))
        else:
            u, a = [tuple(p) for p in d['pairs']][int(rng.integers(0, len(d['pairs'])))]
        own = [c for c in d['covers'] if u in c and a in c]
        if own and s < 2:
            bmid = min(x for x in own[int(rng.integers(0, len(own)))] if x not in (u, a)); lo = max(0, bmid - SLICE_WIDTH // 2)
        else:
            lo = int(rng.integers(0, N - SLICE_WIDTH))
        hi = min(N, lo + SLICE_WIDTH)
        cov, _, tot = C.run_pairs(V1, t1, [(u, a)], 5, P1, seed=C.chunk_seed(K), CODES=codes, brange=(lo, hi))
        stored = set(tuple(c) for c in own if lo <= min(x for x in c if x not in (u, a)) < hi)
        if set(tuple(c) for c in cov) != stored: fail(f're-enumerated slice {(u, a, lo, hi)} of chunk {K} disagrees with the stored covers')
        say(f'census slice recheck {s+1}: chunk {K}, pair {(u, a)}, prefixes {lo}..{hi-1}: {len(stored)} covers, agree')
    # 4. level-3 bases (closure, exact representatives, repeats, span letters, skip recheck)
    B3.main(tmp)
    b3 = json.load(open(os.path.join(tmp, 'b3_reps.json')))
    if b3['dec4_orbit_sizes'] != [432, 1152, 432, 144] and sorted(b3['dec4_orbit_sizes']) != [144, 432, 432, 1152]: fail('four-term decompositions of t_3')
    if b3['n_indep'] != 2013120 or b3['n_indep_orbits'] != 154 or b3['n_dep'] != 0 or b3['n_repeat'] != 8640: fail(f'level-3 bases: {b3["n_indep"]} / {b3["n_indep_orbits"]} / {b3["n_dep"]} / {b3["n_repeat"]}')
    log3 = open(os.path.join(tmp, 'b3.log')).read()
    if 'flagged by the P2 recheck' not in log3 or ' 0 pivot pairs flagged' not in log3: fail('P2 recheck of the prefix skips')
    for rep in b3['reps']:
        r1, r2 = L.exact_cover_check(codes[rep['letters']], L.target(3))
        if rep['kind'] == 'indep' and not (r1 == 5 and r2 == 5): fail('a representative cover is not exact')
        if rep['kind'] == 'repeat' and not (r1 == 4 and r2 == 4): fail('a repeat base is not exact')
    say(f'level-3 bases: {b3["n_indep"]} independent covers in {b3["n_indep_orbits"]} orbits, {b3["n_repeat"]} repeat multisets in {b3["n_repeat_orbits"]} orbits, no dependent base; representatives exact')
    # 5. level 3 -> 4 lifts of every representative
    L34.main(tmp)
    l34 = json.load(open(os.path.join(tmp, 'lift34.json')))
    if l34['stats']['lifts_indep'] != 1737 or l34['stats']['lifts_repeat'] != 189: fail(f'level 3 -> 4 lifts {l34["stats"]}')
    say(f'level 3 -> 4: {l34["stats"]}')
    # 6. four-term decompositions of t_4 (lifts of the four representatives, closed under G_4)
    lifter4 = lift.Lifter(4, P=P1); all_terms = []
    for rep in b3['dec4_reps']:
        base = codes[rep]; b, allnz = lifter4.solve_coefficients(base)
        if b is None or not allnz: fail('four-term representative')
        for d in lifter4.lift(base, b):
            if not lifter4.verify(d['terms'], b, exact=True)['ok']: fail('a four-term lift of t_4 does not verify')
            all_terms.append(d['terms'])
    gens4 = L.generators(4)
    def key(T): return tuple(sorted(c.tobytes() for c in T))
    seen = {}; orbs = []
    for T in all_terms:
        k = key(T)
        if k in seen: continue
        orb = [T]; seen[k] = len(orbs); i = 0
        while i < len(orb):
            cur = L.dense(orb[i]); i += 1
            for name, g in gens4:
                img = L.apply_gen(g, cur); imgc = L.canon(np.array([L.vec_to_code(v) for v in img])); kk = key(imgc)
                if kk not in seen: seen[kk] = len(orbs); orb.append(imgc)
        orbs.append(orb)
    if len(seen) != 9936 or sorted(len(o) for o in orbs) != [432, 2592, 6912]: fail('four-term decompositions of t_4')
    np.save(os.path.join(tmp, 'decomp4_codes.npy'), np.array([o for orb in orbs for o in orb]))
    json.dump(dict(total_lifts=len(all_terms), n_decomp4=len(seen), orbit_sizes=[len(o) for o in orbs]), open(os.path.join(tmp, 'test_lift.json'), 'w'))
    say('four-term decompositions of t_4: 9,936 in orbits ' + str([len(o) for o in orbs]))
    # premises re-derived exactly: chi(S^3) >= 4 and chi(S^5) >= 5 (a decomposition with at most four terms fails here)
    bad, nl = R45.main(tmp)
    if bad or sum(nl): fail(f'chi(S^3) >= 4 or chi(S^5) >= 5 does not re-derive: violations {bad}, four-term lifts {nl}')
    say(f'premises: chi(S^3) >= 4 (412 candidate triples, all of exact rank 2 without and 3 with t_3) and chi(S^5) >= 5 (k = 4 lifts of the three representatives: {nl})')
    # Theorem F.3 re-derived exactly: no five-term decomposition of t_5 with a product term on the last qutrit
    n5 = P5.main(tmp, log=sys.stdout)
    if n5: fail(f'five-term decompositions of t_5 with a product term exist: {n5}')
    say('Theorem F.3 re-derived: no five-term decomposition of t_5 with a product term on a qutrit (cases A, B, C over the three representatives, no solution at any stage)')
    # 7. product-pattern level-4 bases
    P4.main(tmp)
    p4 = json.load(open(os.path.join(tmp, 'prod4.json')))
    if p4['n'] != 756 or p4['n_ok'] != 756: fail(f'product-pattern bases {p4["n"]} / {p4["n_ok"]}')
    say('product-pattern level-4 bases: 756, all verified')
    # 8. level-4 bases with the span scan recomputed (no cache in tmp)
    B4.main(tmp)
    scan = json.load(open(os.path.join(tmp, 'dec4_span_scan.json')))
    if scan['N4'] != 7439040 or any(s['n_in_span'] != 4 or s['extra'] for s in scan['scan']): fail('span scan of the four-qutrit letters')
    b4 = json.load(open(os.path.join(tmp, 'b4_reps.json')))
    def mkey(T): return tuple(sorted(tuple(t) for t in T))
    mine = {mkey(r['terms']): r for r in b4['reps']}
    v2 = json.load(open(os.path.join(RES, 'b4_reps_v2.json')))['reps']; v1 = json.load(open(os.path.join(RES, 'b4_reps_v1.json')))['reps']
    if set(mine) != set(mkey(r['terms']) for r in v2): fail('level-4 bases differ from the stored v2 list')
    k0 = [r for r in b4['reps'] if r['kappa'] == 0]; k1 = [r for r in b4['reps'] if r['kappa'] == 1]
    if len(k0) != 2574 or len(k1) != 35 or any(not r['allnz'] for r in k0): fail(f'level-4 kappa split {len(k0)} / {len(k1)}')
    if set(mkey(r['terms']) for r in k1) != set(mkey(r['terms']) for r in v1 if r['kappa'] == 1): fail('kappa = 1 bases differ from the stored v1 list')
    for r in v2:
        if r['kappa'] == 0:
            e1, e2 = L.exact_cover_check(np.array(r['terms'], dtype=np.int8), L.target(4))
            if not (e1 == 5 and e2 == 5): fail('a level-4 base is not exact')
    say(f'level-4 bases: {len(b4["reps"])} multisets ({len(k0)} with unique coefficients, {len(k1)} one-parameter families), equal to the stored lists; the {len(k0)} independent ones exact')
    # 9. stored level-5 kappa = 0 results: coverage, nothing found, recheck of a sample
    idx2 = {mkey(r['terms']): i for i, r in enumerate(v2)}
    covered0 = set(); found = []
    for b in by_stage['lift45']:
        d = json.load(open(os.path.join(ROOT, b['output'])))
        if d['NK'] != 128: fail('lift45 chunking')
        covered0 |= set(i for i in range(d['K'], len(v2), d['NK']) if v2[i]['kappa'] == 0); found += d['found']
    if covered0 != set(i for i, r in enumerate(v2) if r['kappa'] == 0): fail('stored kappa = 0 lifts do not cover every base')
    if found: fail(f'the stored level-5 lifts record {len(found)} decompositions')
    lifter5 = lift.Lifter(5, P=P1)
    for s in range(RECHECK_LIFT_CHUNKS):
        K = int(rng.integers(0, 128)); members = [i for i in range(K, len(v2), 128) if v2[i]['kappa'] == 0]
        for i in members:
            r = v2[i]; T = np.array(r['terms'], dtype=np.int8); b = np.array(r['coef'], dtype=np.int64)
            if lifter5.lift(T, b): fail(f'level-5 lift of base {i} found a decomposition')
        say(f'level-5 whole-chunk recheck {s+1}: chunk {K}, {len(members)} bases re-lifted, none lifts')
    sample = rng.choice(sorted(covered0), size=RECHECK_BASES, replace=False)
    for i in sample:
        r = v2[int(i)]; T = np.array(r['terms'], dtype=np.int8); b = np.array(r['coef'], dtype=np.int64)
        if lifter5.lift(T, b): fail(f'level-5 lift of base {i} found a decomposition')
    say(f'level-5 kappa = 0: stored results cover all {len(covered0)} bases, none lifts; {RECHECK_BASES} bases re-lifted, none lifts')
    # 10. stored kappa = 1 family results: coverage, nothing found, recheck of sample ranges
    idx1 = {mkey(r['terms']): i for i, r in enumerate(v1)}
    done = {}; found = []
    for b in by_stage['fam']:
        d = json.load(open(os.path.join(ROOT, b['output'])))
        if d['NK'] != 243 or mkey(d['base_terms']) != mkey(v1[d['base']]['terms']): fail('family record does not match its base')
        for r in d['ranges']:
            if r['o0range'] != [r['K'], r['K'] + 1]: fail('family range bounds')
            done.setdefault(d['base'], set()).add(r['K']); found += r['found']
    k1v1 = [i for i, r in enumerate(v1) if r['kappa'] == 1]
    if any(len(done.get(i, ())) != 243 for i in k1v1) or len(done) != 35: fail('stored family results do not cover all 35 bases x 243 ranges')
    if found: fail(f'the stored family results record {len(found)} decompositions')
    tab = L.modtab(P1); t4 = L.target(4) % P1
    for s in range(RECHECK_RANGES):
        i = int(rng.choice(k1v1)); K = int(rng.integers(0, 243)); T = np.array(v1[i]['terms'], dtype=np.int8)
        b0, beta = L34.family(tab[T], np.arange(5), t4, P1)
        lo, hi = (243 * K) // 243, (243 * (K + 1)) // 243
        if lift.lift_family(lifter5, T, b0, beta, o0range=(lo, hi)): fail(f'family range {(i, K)} found a decomposition')
        say(f'family recheck {s+1}: base {i}, range {K}: nothing')
    say(f'level-5 kappa = 1: stored results cover all 35 bases x 243 ranges, nothing found')
    print('CERTIFIED chi(S^5) >= 6', flush=True)
    for m in (6, 7, 8): print(f'CERTIFIED chi(S^{m}) >= 6', flush=True)

if __name__ == '__main__':
    main()
