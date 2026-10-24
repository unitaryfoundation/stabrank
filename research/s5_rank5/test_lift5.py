"""Planted controls of the lift stage at level 4 -> 5 (five terms, four-qutrit letters, 81 rays each): (1) kappa = 0,
random independent base and random options, the meet in the middle must recover the planted option tuple;
(2) kappa = 1, base = four independent letters plus a letter in their span, the family brute force must recover it."""
import sys, os, time, numpy as np, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L, lift
from census import decide
P = L.PRIMES[0]; tab = L.modtab(P); rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
codes = L.letters(n - 1); N = len(codes); D = 3 ** (n - 1); w3 = np.array([1, L.cube_root(P), L.cube_root(P) ** 2 % P], dtype=np.int64)
rays = lift.Rays(n - 1)

def plant(kappa):
    while True:
        pick = rng.choice(N, 5, replace=False); V = tab[codes[pick]]
        if kappa == 0:
            if np.linalg.matrix_rank(L.dense(codes[pick])) == 5: break
        else:
            if np.linalg.matrix_rank(L.dense(codes[pick[:4]])) == 4: break
    if kappa == 1:
        gam = rng.integers(1, P, size=4, dtype=np.int64)
        e = np.zeros(D, dtype=np.int64)
        for j in range(4): e = (e + gam[j] * V[j]) % P
        # e must be a letter: choose instead e = a random letter in the span? rare; so build the 5th letter as a Pauli image? Simplest: take e = one of the four letters' Pauli image that lies in the span -- instead plant a repeat
        pick = np.concatenate([pick[:4], pick[:1]]); V = tab[codes[pick]]
    base = codes[pick]
    opts = [rng.integers(0, 3 * 3 ** (n - 1)) for _ in range(5)]; z0 = [rng.integers(0, 3) for _ in range(5)]
    b = rng.integers(1, P, size=5, dtype=np.int64)
    fw, bw = lift.option_tables(base, rays, P)
    t1 = np.zeros(D, dtype=np.int64); t2 = np.zeros(D, dtype=np.int64); t0 = np.zeros(D, dtype=np.int64)
    for j in range(5):
        r, z = divmod(int(opts[j]), 3)
        t1 = (t1 + b[j] * V[j]) % P
        t2 = (t2 + b[j] * w3[z] % P * tab[fw[j][r]]) % P
        t0 = (t0 + b[j] * w3[z0[j]] % P * tab[bw[j][r]]) % P
    return base, b, opts, z0, (t1, t2, t0)

t_start = time.time()
base, b, opts, z0, targets = plant(0)
lifter = lift.Lifter(n, P=P, targets=targets)
bb, allnz = lifter.solve_coefficients(base); assert bb is not None and np.array_equal(bb, b), (bb, b)
lifts = lifter.lift(base, bb)
hit = any(d['options'] == [int(o) for o in opts] and d['z0'] == [int(z) for z in z0] for d in lifts)
print(f'kappa=0 control (n={n}): {len(lifts)} lifts, planted recovered: {hit}, {time.time()-t_start:.0f}s')
# kappa = 1 (repeat): coefficients family b0 + lambda beta with beta = e_0 - e_4 direction (letters 0 and 4 equal)
t_start = time.time()
base, b, opts, z0, targets = plant(1)
lifter = lift.Lifter(n, P=P, targets=targets)
b0 = b.copy(); b0[0] = (b[0] + b[4]) % P; b0[4] = 0; beta = np.array([P - 1, 0, 0, 0, 1], dtype=np.int64)   # b = b0 + b[4] * beta
lifts = lift.lift_family(lifter, base, b0, beta)
hit = any(d['options'] == [int(o) for o in opts] and d['z0'] == [int(z) for z in z0] for d in lifts)
lam_ok = any(d['options'] == [int(o) for o in opts] and d['lam'] == int(b[4]) for d in lifts)
print(f'kappa=1 control (n={n}): {len(lifts)} lifts, planted recovered: {hit}, lambda recovered: {lam_ok}, {time.time()-t_start:.0f}s')
