"""Exact census of the full r-covers of S^{(x) 3} over the 30,240 three-qutrit stabilizer states (round 33, agent B).

A full r-cover is a set of r linearly independent letters whose span contains the target t = S^{(x) 3} with every
coefficient nonzero. The census works modulo a prime P == 1 (mod 3) (omega -> cube root of unity mod P): in the
quotient by the span of t and a prefix of r - 2 letters, the remaining two letters of a cover have parallel
residues, so they share the projective key (f(res) : g(res)) of two fixed random functionals; letters are grouped by
that key (sorting), every pair in a group is a candidate, and candidates are decided by Gaussian elimination mod P
(rank r, t in the span, all coefficients nonzero). Parallel residues always give equal keys (or a key in the "both
zero" class, which is paired as well), so the screen is sound; false candidates are removed by the decision step.
A letter with zero residue (in the span of t and the prefix) is reported as a short set.

Prefixes: a pivot pair (u, a), one per orbit of the symmetry group G_3 = L^3 x| S_3 on ordered pairs of letters,
followed (for r = 5) by every letter b not in {u, a}; the last two letters c < d are required to satisfy c, d > b,
so that each cover through a given pivot pair is enumerated once. Every G_3-orbit of covers contains a cover through
one of the pivot pairs, and the covers found are closed under G_3 afterwards (closure.py).

    python census.py pairs OUTDIR                 # the pivot pairs (orbits of pairs under G_3)
    python census.py run R OUTDIR K NK [PRIME]    # the pairs of chunk K of NK, results to OUTDIR/r<R>/chunk_K.json
    python census.py plant R OUTDIR SEED          # planted control: random target in the span of R random letters
"""
import numpy as np, sys, os, json, time
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L

@njit(cache=True)
def modinv(a, P):
    r = 1; b = a % P; e = P - 2
    while e:
        if e & 1: r = (r * b) % P
        b = (b * b) % P; e >>= 1
    return r

@njit(cache=True)
def basis_add(B, pc, nb, vec, P):
    """append vec to the fully reduced basis B (rows 0..nb-1 with pivot columns pc); returns the new nb (unchanged if
    vec is in the span). B must have room for one more row."""
    D = B.shape[1]; r = np.empty(D, dtype=np.int64)
    for j in range(D): r[j] = vec[j] % P
    for k in range(nb):
        c = r[pc[k]]
        if c:
            for j in range(D): r[j] = (r[j] - c * B[k, j]) % P
    piv = -1
    for j in range(D):
        if r[j]: piv = j; break
    if piv < 0: return nb
    inv = modinv(r[piv], P)
    for j in range(D): r[j] = (r[j] * inv) % P
    for k in range(nb):
        c = B[k, piv]
        if c:
            for j in range(D): B[k, j] = (B[k, j] - c * r[j]) % P
    for j in range(D): B[nb, j] = r[j]
    pc[nb] = piv
    return nb + 1

@njit(cache=True)
def functional_mod_basis(f, B, pc, nb, P):
    """phi with phi . v = f . (v mod span B) for every v (B fully reduced)."""
    D = B.shape[1]; phi = f.copy()
    for k in range(nb):
        s = 0
        for j in range(D): s = (s + f[j] * B[k, j]) % P
        phi[pc[k]] = (phi[pc[k]] - s) % P
    return phi

@njit(cache=True)
def decide(V, idx, nidx, t, P, coef):
    """rank of the letters idx[0..nidx-1] mod P, rank with t appended, and the coefficients of t if the letters are
    independent and t is in their span (coef filled, returns (rank, rank_t, all_nonzero))."""
    D = V.shape[1]
    B = np.zeros((nidx + 1, D), dtype=np.int64); pc = np.zeros(nidx + 1, dtype=np.int64); nb = 0
    for i in range(nidx):
        nb = basis_add(B, pc, nb, V[idx[i]], P)
    rank = nb
    nb2 = basis_add(B, pc, nb, t, P)
    allnz = False
    if rank == nidx and nb2 == nb:
        # solve: augmented elimination on the nidx x D system
        M = np.zeros((nidx, D + nidx), dtype=np.int64)
        for i in range(nidx):
            for j in range(D): M[i, j] = V[idx[i], j] % P
            M[i, D + i] = 1
        # row-reduce the letter part; track the transform
        r = 0; pcs = np.zeros(nidx, dtype=np.int64)
        for c in range(D):
            if r >= nidx: break
            p = -1
            for i in range(r, nidx):
                if M[i, c]: p = i; break
            if p < 0: continue
            for j in range(D + nidx):
                tmp = M[r, j]; M[r, j] = M[p, j]; M[p, j] = tmp
            inv = modinv(M[r, c], P)
            for j in range(D + nidx): M[r, j] = (M[r, j] * inv) % P
            for i in range(nidx):
                if i != r and M[i, c]:
                    cc = M[i, c]
                    for j in range(D + nidx): M[i, j] = (M[i, j] - cc * M[r, j]) % P
            pcs[r] = c; r += 1
        # now rows of M: e_{pcs[r]} (+ other cols) = sum_i T[r,i] V[idx[i]]; t = sum_r t[pcs[r]] row_r
        for i in range(nidx): coef[i] = 0
        for rr in range(nidx):
            tc = t[pcs[rr]] % P
            if tc:
                for i in range(nidx): coef[i] = (coef[i] + tc * M[rr, D + i]) % P
        allnz = True
        for i in range(nidx):
            if coef[i] == 0: allnz = False
    return rank, nb2, allnz

@njit(cache=True)
def kernel(V, CODES, W1, W2, t, u, a, r, f, g, P, V2, t2, P2, out, maxout, short, maxshort, stats, bmin, bmax):
    """all full r-covers {u, a, (b,) c, d} through the pivot pair (u, a); r in {4, 5}. out rows: letter indices
    (5 columns, -1 padded); short rows: letters in the span of t and a prefix (4 columns). stats: [prefixes,
    candidates, covers, shorts, overflow]."""
    N, D = V.shape
    B0 = np.zeros((6, D), dtype=np.int64); pc0 = np.zeros(6, dtype=np.int64)
    nb0 = basis_add(B0, pc0, 0, t, P); nb0 = basis_add(B0, pc0, nb0, V[u], P); nb0 = basis_add(B0, pc0, nb0, V[a], P)
    if nb0 < 3:
        stats[4] = -1; return
    keys = np.empty(N, dtype=np.int64); who = np.empty(N, dtype=np.int64)
    idx = np.zeros(6, dtype=np.int64); coef = np.zeros(6, dtype=np.int64)
    B = np.zeros((6, D), dtype=np.int64); pc = np.zeros(6, dtype=np.int64)
    nout = 0; nshort = 0
    bvals = np.arange(bmin, bmax) if r == 5 else np.arange(1)
    for b in bvals:
        if r == 5:
            if b == u or b == a: continue
            for k in range(nb0):
                for j in range(D): B[k, j] = B0[k, j]
                pc[k] = pc0[k]
            nb = basis_add(B, pc, nb0, V[b], P)
            if nb == nb0:
                # b in span(t, u, a): short set {u, a, b}
                if nshort < maxshort:
                    short[nshort, 0] = u; short[nshort, 1] = a; short[nshort, 2] = b; short[nshort, 3] = -1; nshort += 1
                continue
            cstart = b + 1
        else:
            for k in range(nb0):
                for j in range(D): B[k, j] = B0[k, j]
                pc[k] = pc0[k]
            nb = nb0; cstart = 0
        stats[0] += 1
        phi = functional_mod_basis(f, B, pc, nb, P); gam = functional_mod_basis(g, B, pc, nb, P)
        m = 0
        for c in range(cstart, N):
            if c == u or c == a: continue
            # exponent-split dot products: value omega^e -> sums of the functionals over the entries with that exponent
            a0 = 0; a1 = 0; a2 = 0; b0 = 0; b1 = 0; b2 = 0
            for j in range(D):
                cj = CODES[c, j]
                if cj == 1:
                    a0 += phi[j]; b0 += gam[j]
                elif cj == 2:
                    a1 += phi[j]; b1 += gam[j]
                elif cj == 3:
                    a2 += phi[j]; b2 += gam[j]
            s1 = (a0 + W1 * (a1 % P) + W2 * (a2 % P)) % P; s2 = (b0 + W1 * (b1 % P) + W2 * (b2 % P)) % P
            if s1 == 0 and s2 == 0:
                # residue (probably) zero: c in span(t, u, a, b); record as short set and also keep in the zero class
                if nshort < maxshort:
                    short[nshort, 0] = u; short[nshort, 1] = a; short[nshort, 2] = b if r == 5 else -1; short[nshort, 3] = c; nshort += 1
                keys[m] = -2; who[m] = c; m += 1
            elif s2 == 0:
                keys[m] = -1; who[m] = c; m += 1
            else:
                keys[m] = (s1 * modinv(s2, P)) % P; who[m] = c; m += 1
        order = np.argsort(keys[:m])
        i = 0
        while i < m:
            j = i
            while j + 1 < m and keys[order[j + 1]] == keys[order[i]]: j += 1
            if j > i:
                for p1 in range(i, j + 1):
                    for p2 in range(p1 + 1, j + 1):
                        c = who[order[p1]]; d = who[order[p2]]
                        stats[1] += 1
                        idx[0] = u; idx[1] = a
                        if r == 5:
                            idx[2] = b; idx[3] = c; idx[4] = d; n = 5
                        else:
                            idx[2] = c; idx[3] = d; n = 4
                        rank, rank_t, allnz = decide(V, idx, n, t, P, coef)
                        ok = rank == n and rank_t == n and allnz
                        if not ok:
                            rank, rank_t, allnz = decide(V2, idx, n, t2, P2, coef)
                            ok = rank == n and rank_t == n and allnz
                        if ok:
                            if nout < maxout:
                                for q in range(5): out[nout, q] = idx[q] if q < n else -1
                                nout += 1; stats[2] += 1
                            else:
                                stats[4] += 1
            i = j + 1
    stats[3] = nshort
    return nout

def setup(P):
    codes = L.letters(3); V = L.modtab(P)[codes]; t = L.target(3) % P
    return codes, V, t

def stabilizer_elements(n=3):
    """the group L^n x| S_n as dense 3^n x 3^n matrices generated from the generators by closure modulo phase (for n = 3
    this is 82,944 elements; too many to store at once, so this returns the single-qutrit group L (24 elements) and
    the permutations, from which elements are assembled)."""
    # single-qutrit group L: closure of <F, PG> modulo global phase (exact comparison up to phase)
    def same(U, W):
        j = np.flatnonzero(np.abs(U.ravel()) > 1e-9)[0]; ph = W.ravel()[j] / U.ravel()[j]
        return abs(abs(ph) - 1) < 1e-9 and np.allclose(U * ph, W, atol=1e-9)
    Ls = [np.eye(3, dtype=complex)]; frontier = [np.eye(3, dtype=complex)]
    while frontier:
        new = []
        for U in frontier:
            for G in (L.F1, L.PG1):
                W = G @ U
                if not any(same(E, W) for E in Ls): Ls.append(W); new.append(W)
        frontier = new
    assert len(Ls) == 24, len(Ls)
    import itertools
    perms = [L.perm_index(pi, n) for pi in itertools.permutations(range(n))]
    return Ls, perms

def vec_codes_batch(Vd, tol=1e-7):
    """codes of a batch of dense vectors proportional to stabilizer states."""
    m = np.abs(Vd); mx = m.max(axis=1); nz = m > tol * mx[:, None]
    first = np.argmax(nz, axis=1); piv = Vd[np.arange(len(Vd)), first]
    U = Vd / piv[:, None]
    e = np.round(np.angle(U) / (2 * np.pi / 3)).astype(np.int64) % 3
    assert np.allclose(np.where(nz, U, 0), np.where(nz, L.OMEGA ** e, 0), atol=1e-6)
    return np.where(nz, e + 1, 0).astype(np.int8)

def stage_pairs(outdir):
    """pivot pairs: for each G_3-orbit representative u, representatives a of the orbits of Stab(u) on the other
    letters."""
    log = open(os.path.join(outdir, 'pairs.log'), 'a')
    codes = L.letters(3); N = len(codes); lk = L.Lookup(codes); Vd = L.dense(codes)
    gens = L.generators(3); maps = L.letter_maps(codes, gens, lk); roots = L.orbits_uf(N, maps)
    reps = sorted(set(roots.tolist())); sizes = {r: int((roots == r).sum()) for r in reps}
    print(f'[pairs] {N} letters, {len(reps)} orbits, sizes {[sizes[r] for r in reps]}', file=log, flush=True)
    Ls, perms = stabilizer_elements(3)
    # all 13,824 local operators as 27x27 matrices
    locs = np.array([np.kron(np.kron(A, Bm), C) for A in Ls for Bm in Ls for C in Ls])     # (13824, 27, 27)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = []; info = []
    for u in reps:
        vu = Vd[u]; stab = []
        for pi in perms:
            vp = vu[pi]                                   # permuted vector
            img = locs @ vp                               # (13824, 27)
            ov = np.abs(img @ vu.conj()) / (np.linalg.norm(vu) ** 2)
            for gi in np.flatnonzero(np.abs(ov - 1) < 1e-8):
                stab.append((int(gi), pi))
        assert len(stab) * sizes[u] == 82944, (u, len(stab), sizes[u])
        # orbits of Stab(u) on letters: edges x -> g x
        rows = []; cols = []
        for gi, pi in stab:
            img = Vd[:, pi] @ locs[gi].T
            imgc = vec_codes_batch(img); m = lk.find_many(imgc); assert (m >= 0).all()
            rows.append(np.arange(N)); cols.append(m)
        G = coo_matrix((np.ones(N * len(stab)), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        ncomp, lab = connected_components(G, directed=True, connection='weak')
        areps = sorted(set(int(np.flatnonzero(lab == c)[0]) for c in range(ncomp)))
        areps = [a for a in areps if a != u]
        # choose the smallest index in each component as representative
        comp_min = {}
        for x in range(N):
            c = int(lab[x]); comp_min[c] = min(comp_min.get(c, x), x)
        areps = sorted(set(comp_min.values()) - {u})
        for a in areps: pairs.append((int(u), int(a)))
        info.append(dict(u=int(u), orbit=sizes[u], stab=len(stab), a_reps=len(areps)))
        print(f'[pairs] u={u} orbit {sizes[u]} stabilizer {len(stab)} -> {len(areps)} pair representatives', file=log, flush=True)
    json.dump(dict(N=N, reps=reps, sizes=[sizes[r] for r in reps], info=info, pairs=pairs), open(os.path.join(outdir, 'pairs.json'), 'w'))
    print(f'[pairs] {len(pairs)} pivot pairs in all', file=log, flush=True)
    print(f'PAIRS DONE {len(pairs)}', flush=True)

KERNEL_VERSION = 2   # 2: no letter-dependency skip, decisions modulo both primes (2026-10-23 13:40 EDT)

def run_pairs(V, t, pairs, r, P, seed, log=None, tag='', CODES=None, brange=None):
    rng = np.random.default_rng(seed); D = V.shape[1]
    W1 = L.cube_root(P); W2 = W1 * W1 % P
    P2 = [q for q in L.PRIMES if q != P][0]; V2 = L.modtab(P2)[CODES]; t2 = L.target(3) % P2 if D == 27 else None
    if t2 is None or not np.array_equal(t, L.target(3) % P): t2 = None
    f = rng.integers(1, P, size=D, dtype=np.int64); g = rng.integers(1, P, size=D, dtype=np.int64)
    out = np.full((2000000, 5), -1, dtype=np.int64); short = np.full((200000, 4), -1, dtype=np.int64)
    covers = []; shorts = []; totals = np.zeros(5, dtype=np.int64); t0 = time.time()
    for k, (u, a) in enumerate(pairs):
        stats = np.zeros(5, dtype=np.int64)
        bmin, bmax = (0, V.shape[0]) if brange is None else brange
        nout = kernel(V, CODES, W1, W2, t, u, a, r, f, g, P, V2, (t2 if t2 is not None else V2[0] * 0), P2, out, len(out), short, len(short), stats, bmin, bmax)
        assert stats[4] == 0, ('overflow or dependent pivot pair', stats)
        covers.extend(sorted(int(x) for x in row if x >= 0) for row in out[:nout])
        shorts.extend([int(x) for x in row] for row in short[:stats[3]])
        totals += stats
        if log and (k % 20 == 0 or k == len(pairs) - 1):
            print(f'{tag}pair {k+1}/{len(pairs)} (u={u}, a={a}): prefixes {totals[0]}, candidates {totals[1]}, covers {totals[2]}, shorts {totals[3]}, {time.time()-t0:.0f}s', file=log, flush=True)
    return covers, shorts, totals

def chunk_seed(K): return 1000 + K

def stage_run(r, outdir, K, NK, P):
    os.makedirs(os.path.join(outdir, f'r{r}'), exist_ok=True)
    fn = os.path.join(outdir, f'r{r}', f'chunk_{K}.json')
    if os.path.exists(fn):
        print(f'chunk {K} exists'); return
    log = open(os.path.join(outdir, f'r{r}', f'chunk_{K}.log'), 'a')
    codes, V, t = setup(P)
    pairs = [tuple(p) for p in json.load(open(os.path.join(outdir, 'pairs.json')))['pairs']]
    mine = pairs[K::NK]
    print(f'[run r={r}] chunk {K}/{NK}: {len(mine)} pivot pairs, P={P}', file=log, flush=True)
    covers, shorts, totals = run_pairs(V, t, mine, r, P, seed=chunk_seed(K), log=log, tag=f'[r{r} chunk {K}] ', CODES=codes)
    tmp = fn + f'.tmp{os.getpid()}'
    json.dump(dict(r=r, K=K, NK=NK, P=P, kernel=KERNEL_VERSION, pairs=mine, covers=covers, shorts=shorts, prefixes=int(totals[0]), candidates=int(totals[1]),
                   n_covers=int(totals[2]), n_shorts=int(totals[3])), open(tmp, 'w'))
    os.replace(tmp, fn)
    print(f'RUN DONE r={r} chunk {K}: {len(covers)} covers, {len(shorts)} shorts', flush=True)

def stage_plant(r, outdir, seed):
    """planted control: t = random combination of r random independent letters; pivot pairs = the pairs of planted
    letters (the planted target has no symmetry); the kernel must find the planted set."""
    P = L.PRIMES[0]; codes, V, _ = setup(P); N = len(codes); rng = np.random.default_rng(seed)
    while True:
        pick = sorted(rng.choice(N, r, replace=False).tolist())
        c = rng.integers(1, P, size=r, dtype=np.int64)
        t = np.zeros(V.shape[1], dtype=np.int64)
        for j in range(r): t = (t + c[j] * V[pick[j]]) % P
        coef = np.zeros(6, dtype=np.int64); idx = np.array(pick + [0] * (6 - r), dtype=np.int64)
        rank, rank_t, allnz = decide(V, idx, r, t, P, coef)
        if rank == r and allnz: break
    log = open(os.path.join(outdir, f'plant_r{r}_{seed}.log'), 'a')
    print(f'[plant r={r}] planted letters {pick}', file=log, flush=True)
    # pivot pairs: (pick[0], pick[1]) and, as a harder test, (pick[0], pick[2]) etc.
    pairs = [(pick[0], pick[1]), (pick[1], pick[0]), (pick[2], pick[4 if r == 5 else 3])]
    covers, shorts, totals = run_pairs(V, t, pairs, r, P, seed=seed, log=log, tag='[plant] ', CODES=codes)
    hit = pick in covers
    print(f'[plant r={r}] covers found {len(covers)} (planted among them: {hit}); candidates {totals[1]}; shorts {len(shorts)}', file=log, flush=True)
    json.dump(dict(r=r, seed=seed, planted=pick, found=hit, n_covers=len(covers), covers=covers[:50]), open(os.path.join(outdir, f'plant_r{r}_{seed}.json'), 'w'))
    print(f'PLANT DONE r={r} found={hit} n_covers={len(covers)}', flush=True)

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'pairs': stage_pairs(sys.argv[2])
    elif stage == 'run':
        r, outdir, K, NK = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        P = int(sys.argv[6]) if len(sys.argv) > 6 else L.PRIMES[0]
        stage_run(r, outdir, K, NK, P)
    elif stage == 'plant': stage_plant(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
