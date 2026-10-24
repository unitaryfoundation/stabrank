"""Exact collection, closure under the symmetry group, and lift, for the rank-6 census of sqrt(T)^m (census.py).

    python stages.py collect M OUT [--workers W]     candidate sets -> every independent six-subset whose span
                                                     contains t_1 (batched elimination mod two primes, then every
                                                     decomposition confirmed exactly over Q(zeta_16)); closure
                                                     under G = S_m x <S^(x)m>; coefficients mod P for the lift
    python stages.py lift M OUT                      every six-term decomposition of sqrt(T)^(m+1) by the slicing
                                                     lemma, each verified exactly
    python stages.py plant M OUT                     control: slice the complete list at m along qubit 0; every
                                                     all-visible decomposition must be recovered by `lift` at m - 1
"""
import sys, os, json, time, itertools, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (PRIMES, Field, hash_rows, weights, perm_index_maps, canon_codes, exact_coefficients, root16,
                    modinv_arr)
from census import load_codes, hash_dict, say, SEED

M61 = (1 << 61) - 1


def batched_solve(Vsets, t, P):
    """Vsets (n, r, D) values mod P, t (D,): solve sum_j c_j V[j] = t for each set. Returns (c (n, r), ok (n,)):
    ok is True when the r rows are independent and t lies in their span, both decided mod P."""
    n, r, D = Vsets.shape
    A = np.concatenate([np.transpose(Vsets, (0, 2, 1)) % P, np.broadcast_to(t[None, :, None] % P, (n, D, 1))], axis=2).astype(np.int64)
    ok = np.ones(n, dtype=bool)
    ar = np.arange(n)
    for c in range(r):
        col = A[:, :, c]
        cand = (col != 0)
        cand[:, :c] = False
        has = cand.any(axis=1)
        ok &= has
        piv = np.argmax(cand, axis=1)
        piv[~has] = c
        # swap rows c and piv
        rc = A[ar, c].copy(); rp = A[ar, piv].copy()
        A[ar, c] = rp; A[ar, piv] = rc
        inv = modinv_arr(np.where(A[:, c, c] == 0, 1, A[:, c, c]), P)
        A[:, c, :] = (A[:, c, :] * inv[:, None]) % P
        f = A[:, :, c].copy()
        f[:, c] = 0
        A = (A - (f[:, :, None] * A[:, c, :][:, None, :]) % P) % P
    # consistency: rows r..D-1 of the last column must vanish
    ok &= ~(A[:, r:, r].any(axis=1))
    return A[:, :r, r].copy(), ok


def confirm_exact(args):
    codes_sel, m = args
    cs = exact_coefficients(codes_sel, m)
    return None if cs is None else [str(c) for c in cs]


def cmd_collect(m, out, workers):
    import multiprocessing as mp
    codes = load_codes(m, out)
    N, D = codes.shape
    fields = [Field(m, P, SEED + 1000 * i) for i, P in enumerate(PRIMES)]
    tvals = [np.array([pow(F.g, int(ww), F.P) for ww in weights(m)], dtype=np.int64) for F in fields]
    bl = sorted(f for f in os.listdir(os.path.join(out, 'blocks')) if f.endswith('.json'))
    reps = np.load(os.path.join(out, f"reps_m{m}.npz"))['reps']
    seen = set(); sets = {}; tsec = 0.0; nres = 0
    for f in bl:
        d = json.load(open(os.path.join(out, 'blocks', f)))
        for r in d['results']:
            seen.add(r['pivot']); nres += 1; tsec += r['seconds']
            for c in r['candidates']:
                sets.setdefault(tuple(c['members']), []).append(r['pivot'])
    missing = set(int(x) for x in reps) - seen
    complete = not missing
    say(f"{len(bl)} blocks, {nres} pivot results, {len(missing)} representatives missing, {len(sets)} distinct candidate sets, "
        f"{tsec/3600:.2f} CPU-h of pivot work")
    # every independent six-subset whose span contains t_1, mod both primes
    dec = {}
    for mem, pivs in sets.items():
        sel = np.array(mem)
        subs = np.array(list(itertools.combinations(range(len(mem)), 6)), dtype=np.int64)
        okall = np.ones(len(subs), dtype=bool)
        for F, t in zip(fields, tvals):
            Vall = F.values(codes[sel])
            for s in range(0, len(subs), 200000):
                _, ok = batched_solve(Vall[subs[s:s + 200000]], t, F.P)
                okall[s:s + 200000] &= ok
        good = subs[okall]
        for sub in good:
            dec.setdefault(tuple(int(i) for i in sel[sub]), None)
        say(f"  set of {len(mem)} members ({len(pivs)} pivots): {len(subs)} six-subsets, {len(good)} decompositions mod P")
    keys = list(dec)
    say(f"decompositions from the representatives' candidate sets: {len(keys)} (screened mod {PRIMES})")
    # exact confirmation of every one of them
    ctx = mp.get_context('fork')
    with ctx.Pool(workers) as pool:
        res = pool.map(confirm_exact, [(codes[list(k)], m) for k in keys], chunksize=16)
    bad = [k for k, r in zip(keys, res) if r is None]
    if bad:
        say(f"  WARNING: {len(bad)} subsets pass mod P but are not exact decompositions; dropped")
    dec = {k: r for k, r in zip(keys, res) if r is not None}
    say(f"exactly confirmed: {len(dec)}")
    # closure under G: image index of every involved state under each group element
    involved = sorted({i for k in dec for i in k})
    inv_pos = {s: j for j, s in enumerate(involved)}
    hd, _ = hash_dict(codes)
    maps = perm_index_maps(m)
    w = weights(m)
    sub = codes[involved]
    images = []                                                    # per group element: (index map array, phase array)
    for idx in maps:
        base = sub[:, idx]
        for s in range(4):
            img = np.where(base != 0, ((base - 1 + s * w[None, :]) % 4) + 1, 0).astype(np.int8)
            first = np.argmax(img != 0, axis=1)
            e0 = img[np.arange(len(img)), first] - 1
            can = canon_codes(img)
            h = hash_rows(can)
            ids = np.array([hd[(int(a), int(b))] for a, b in h], dtype=np.int64)
            images.append((ids, e0))
    full = set(dec)
    orbit_rep = {}
    for k in dec:
        if k in orbit_rep:
            continue
        pos = [inv_pos[i] for i in k]
        orb = set()
        for ids, _ in images:
            orb.add(tuple(sorted(int(ids[p]) for p in pos)))
        for img in orb:
            orbit_rep.setdefault(img, k)
        full |= orb
    say(f"complete list after closure under G (order {len(images)}): {len(full)} decompositions, "
        f"{len({orbit_rep[k] for k in full})} orbits")
    # coefficients mod P of every decomposition (batched), for the lift dictionary
    full = sorted(full)
    arr = np.array(full, dtype=np.int64)
    coeff_mod = []
    for F, t in zip(fields, tvals):
        cm = np.empty((len(arr), 6), dtype=np.int64); okk = np.ones(len(arr), dtype=bool)
        for s in range(0, len(arr), 100000):
            c, ok = batched_solve(F.values(codes[arr[s:s + 100000]]), t, F.P)
            cm[s:s + 100000] = c; okk[s:s + 100000] = ok
        if not okk.all():
            say(f"  ERROR: {(~okk).sum()} group images are not decompositions mod {F.P}")
        coeff_mod.append(cm)
    json.dump({'m': m, 'complete': complete, 'pivots_done': len(seen), 'reps': int(len(reps)), 'candidate_sets': len(sets),
               'from_reps': len(dec), 'orbits': len({orbit_rep[k] for k in full}), 'pivot_cpu_hours': tsec / 3600,
               'exact_coeff_reps': {','.join(map(str, k)): v for k, v in dec.items()},
               'terms': arr.tolist(), 'coeff_mod': [c.tolist() for c in coeff_mod]},
              open(os.path.join(out, f'decomps_m{m}.json'), 'w'))
    say("written decomps_m%d.json" % m)


def pauli_images_all(code, m):
    """(canonical codes (K, D) int8, phase exponents e (K,)) of X^a Z^b code = i^e * canonical, all (a, b), unique rows
    kept with every exponent that occurs: returns dict bytes -> set of exponents."""
    D = 1 << m
    xs = np.arange(D)
    perms = xs[None, :] ^ xs[:, None]                      # perms[a] = x ^ a
    base = np.zeros((D, D), dtype=np.int8)
    for a in range(D):
        base[a, perms[a]] = code
    ph = np.array([[2 * (bin(x & b).count('1') & 1) for x in range(D)] for b in range(D)], dtype=np.int64)
    img = np.where(base[:, None, :] != 0, ((base[:, None, :] - 1 + ph[None, :, :]) % 4) + 1, 0).astype(np.int8).reshape(D * D, D)
    first = np.argmax(img != 0, axis=1)
    e0 = (img[np.arange(len(img)), first] - 1).astype(np.int64)
    can = canon_codes(img)
    out = {}
    for row, e in zip(can, e0):
        out.setdefault(row.tobytes(), set()).add(int(e))
    return out


def cmd_lift(m, out, dec_path=None, tag=''):
    codes = load_codes(m, out)
    D = 1 << m
    dec = json.load(open(dec_path or os.path.join(out, f'decomps_m{m}.json')))
    terms = dec['terms']; cm = dec['coeff_mod']
    say(f"lift at m={m}: {len(terms)} decompositions in the list (complete: {dec['complete']})")
    hd, _ = hash_dict(codes)
    byterm = {}
    for di, t6 in enumerate(terms):
        for j, t in enumerate(t6):
            byterm.setdefault(t, []).append((di, j))
    g = [root16(P) for P in PRIMES]
    zinv = [pow(gg, P - 2, P) for gg, P in zip(g, PRIMES)]
    i_pow = [[pow(gg, 4 * e, P) for e in range(4)] for gg, P in zip(g, PRIMES)]
    lifts = []; cache = {}
    # iterate over orbit representatives only: the group S_m x <S^(x)(m+1)> maps the six-term decompositions of
    # sqrt(T)^(m+1) to itself (permutations of the last m qubits, and the Galois argument for S^(x)(m+1)) and acts
    # on their <0| slices as G, so every such decomposition is equivalent to one whose <0| slice is a representative
    reps = sorted({tuple(int(x) for x in k.split(',')) for k in dec['exact_coeff_reps']})
    index_of = {tuple(t): i for i, t in enumerate(terms)}
    for ri, U in enumerate(reps):
        di = index_of[U]
        votes = {}
        for j, t in enumerate(U):
            if t not in cache:
                cache[t] = pauli_images_all(codes[t], m)
            for can_bytes, e0s in cache[t].items():
                v = hd[tuple(int(x) for x in hash_rows(np.frombuffer(can_bytes, dtype=np.int8)[None, :])[0])]
                for (vi, slot) in byterm.get(v, []):
                    for e in range(4):
                        if all(((cm[fi][di][j] * i_pow[fi][e]) % P) * zinv[fi] % P == cm[fi][vi][slot]
                               for fi, P in enumerate(PRIMES)):
                            votes.setdefault(vi, {}).setdefault(j, set()).add((v, e))
        for vi, slots in votes.items():
            if len(slots) == 6:
                for choice in itertools.product(*[sorted(slots[j]) for j in range(6)]):
                    vs = [c[0] for c in choice]
                    if sorted(vs) == sorted(terms[vi]):
                        lifts.append({'U': list(U), 'V': terms[vi], 'pairs': [[U[j], vs[j], choice[j][1]] for j in range(6)]})
        if ri % 500 == 0:
            say(f"  representative {ri}/{len(reps)}: {len(lifts)} lift candidates so far")
    say(f"lift candidates passing the mod-P coefficient match: {len(lifts)}")
    verified = []
    for lf in lifts:
        rows = []
        for (u, v, e) in lf['pairs']:
            cv = codes[v]
            bot = np.where(cv != 0, ((cv - 1 + e) % 4) + 1, 0).astype(np.int8)
            rows.append(canon_codes(np.concatenate([codes[u], bot])[None, :])[0])
        rows = np.array(rows)
        cs = exact_coefficients(rows, m + 1)
        if cs is not None:
            verified.append({'pairs': lf['pairs'], 'codes': [''.join('.1i-j'[c] for c in r) for r in rows], 'coeff': [str(c) for c in cs]})
    say(f"exactly verified six-term decompositions of sqrt(T)^{m+1} (up to G on the last {m} qubits): {len(verified)}")
    json.dump({'m': m, 'lift_candidates': len(lifts), 'verified': verified, 'complete_list': dec['complete']},
              open(os.path.join(out, f'lifts_m{m+1}{tag}.json'), 'w'), indent=1)
    if verified:
        say(f"RESULT: chi(sqrtT^{m+1}) <= 6, decomposition found")
    elif dec['complete']:
        say(f"RESULT: no six-term decomposition of sqrt(T)^{m+1}; chi(sqrtT^{m+1}) >= 7")
    return verified


def cmd_plant(m, out, out_lo):
    """Control of the lift: every six-term decomposition of sqrt(T)^m whose terms are all visible on qubit 0 (both
    slices nonzero) and whose <0| slices are independent is a lift of its <0| slice decomposition U; the lift run from
    exactly the list of those U must recover every one of them."""
    codes = load_codes(m, out)
    codes_lo = load_codes(m - 1, out_lo)
    hd_lo, _ = hash_dict(codes_lo)
    dec = json.load(open(os.path.join(out, f'decomps_m{m}.json')))
    D = 1 << m; half = D // 2
    fields = [Field(m - 1, P, SEED + 1000 * i) for i, P in enumerate(PRIMES)]
    tvals = [np.array([pow(F.g, int(ww), F.P) for ww in weights(m - 1)], dtype=np.int64) for F in fields]
    planted = {}; n_vis = 0; allsets = set()
    for t6 in dec['terms']:
        top = codes[t6][:, :half]; bot = codes[t6][:, half:]
        if (top != 0).any(axis=1).all() and (bot != 0).any(axis=1).all():
            n_vis += 1
            U = tuple(sorted(hd_lo[(int(a), int(b))] for a, b in hash_rows(canon_codes(top))))
            Vs = tuple(sorted(hd_lo[(int(a), int(b))] for a, b in hash_rows(canon_codes(bot))))
            planted.setdefault((U, Vs), []).append(tuple(sorted(''.join('.1i-j'[c] for c in r) for r in codes[t6])))
            allsets.add(U); allsets.add(Vs)
    arr = np.array(sorted(allsets), dtype=np.int64)
    okall = np.ones(len(arr), dtype=bool); coeff_mod = []
    for F, t in zip(fields, tvals):
        c, ok = batched_solve(F.values(codes_lo[arr]), t, F.P)
        okall &= ok; coeff_mod.append(c)
    keep = np.flatnonzero(okall)
    good = {tuple(int(x) for x in arr[i]) for i in keep}
    planted = {k: v for k, v in planted.items() if k[0] in good and k[1] in good}
    n_planted = sum(len(v) for v in planted.values())
    say(f"m={m}: {len(dec['terms'])} decompositions, {n_vis} all-visible on qubit 0, {len(arr)} distinct slice sets, "
        f"{len(keep)} of them independent six-term decompositions of sqrt(T)^{m-1}; {n_planted} planted lifts with both slices independent")
    plant = {'m': m - 1, 'complete': False, 'terms': arr[keep].tolist(), 'coeff_mod': [c[keep].tolist() for c in coeff_mod],
             'exact_coeff_reps': {','.join(map(str, k)): None for k in arr[keep].tolist()}}
    p = os.path.join(out_lo, f'plant_from_m{m}.json')
    json.dump(plant, open(p, 'w'))
    ver = cmd_lift(m - 1, out_lo, dec_path=p, tag='_plant')
    got = {tuple(sorted(v['codes'])) for v in ver}
    want = {d for v in planted.values() for d in v}
    missing = want - got
    say(f"plant control: {len(want)} planted decompositions of sqrt(T)^{m}, {len(got)} recovered by the lift, {len(missing)} missing")
    say("PLANT CONTROL FAILED" if missing else "plant control passed")


if __name__ == '__main__':
    cmd, m, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if cmd == 'collect':
        w = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
        cmd_collect(m, out, w)
    elif cmd == 'lift':
        cmd_lift(m, out)
    elif cmd == 'plant':
        cmd_plant(m, out, sys.argv[4])
