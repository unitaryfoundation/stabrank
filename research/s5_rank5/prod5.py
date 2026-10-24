"""Theorem F.3 (S0543) re-derived exactly (round 33 agent B, after the referee): five-term decompositions of S^5 with
a product term on the last qutrit. By Lemma F.2 with chi(S^4) = 4 the slice tuple is (4,1,0,0), (4,0,1,0), (4,0,0,1)
or (3,0,1,1); after the negation of qutrit 5 the <1| slice is one of the 9,936 four-term decompositions of t_4
(S0541), G_4-equivalent to one of three representatives. Cases A, B, C as in prod4.py one level up: the full terms
range over the 81 rays and three phases of each base letter, the remaining product letter must be a nonzero
four-qutrit stabilizer state, tested against the complete table of 7,439,040 codes. Modulus P1 (exact by Hadamard's
bound). The result "no solution at every stage" proves that every five-term decomposition of S^5 is all full.
    python prod5.py OUTDIR
"""
import numpy as np, itertools, os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L, lift
from lift import _funcvals, _dot, mitm, option_tables
from prod4 import letter_of, product_code

def main(outdir, log=None):
    t0 = time.time(); P = L.PRIMES[0]; lifter = lift.Lifter(5, P=P); w3 = lifter.w3
    if log is None: log = open(os.path.join(outdir, 'prod5.log'), 'a')
    def say(m): print(m, flush=True); print(m, file=log, flush=True)
    codes4 = L.letters(4); lk = L.Lookup(codes4); say(f'[prod5] {len(codes4)} four-qutrit letters, lookup built, {time.time()-t0:.0f}s')
    D4 = np.load(os.path.join(outdir, 'decomp4_codes.npy')); info = json.load(open(os.path.join(outdir, 'test_lift.json')))
    starts = np.cumsum([0] + info['orbit_sizes'][:-1]); reps = [D4[s] for s in starts]
    out = []; stats = []
    for oi, base in enumerate(reps):
        c, allnz = lifter.solve_coefficients(base); assert c is not None and allnz
        fw, bw = option_tables(base, lifter.rays, P)
        Vfw = [lifter.tab[fw[j]] for j in range(4)]; Vbw = [lifter.tab[bw[j]] for j in range(4)]
        FVf = np.stack([_funcvals(lifter.f, lifter.g, Vfw[j], P) for j in range(4)]); FVb = np.stack([_funcvals(lifter.f, lifter.g, Vbw[j], P) for j in range(4)])
        ftm = np.array([(-_dot(lifter.f, lifter.t, P)) % P, (-_dot(lifter.g, lifter.t, P)) % P], dtype=np.int64); f0 = np.zeros(2, dtype=np.int64)
        D = 81
        def assemble(j, r, z0, z2):
            c0 = bw[j][r].copy(); nz0 = c0 != 0; c0 = np.where(nz0, ((c0 - 1 + z0) % 3) + 1, 0).astype(np.int8)
            c2 = fw[j][r].copy(); nz2 = c2 != 0; c2 = np.where(nz2, ((c2 - 1 + z2) % 3) + 1, 0).astype(np.int8)
            return L.canon(L.assemble_last(c0, base[j], c2).reshape(1, -1))[0]
        solsA = mitm(FVf, c, P, ftm); nA = 0; nAv = 0
        for s in solsA:
            acc = np.zeros(D, dtype=np.int64)
            for j in range(4):
                r, z = divmod(int(s[j]), 3); acc = (acc + c[j] * w3[z] % P * Vfw[j][r]) % P
            if not np.array_equal(acc, (-lifter.t) % P): continue
            nAv += 1
            for zs in itertools.product(range(3), repeat=4):
                q = np.zeros(D, dtype=np.int64)
                for j in range(4):
                    r = int(s[j]) // 3; q = (q + c[j] * w3[zs[j]] % P * Vbw[j][r]) % P
                if not q.any(): continue
                li = letter_of(q, lk, P, w3)
                if li < 0: continue
                out.append(dict(case='A', rep=oi, terms=[assemble(j, int(s[j]) // 3, zs[j], int(s[j]) % 3).tolist() for j in range(4)] + [product_code(codes4[li], 0).tolist()])); nA += 1
        solsB = mitm(FVb, c, P, f0); nB = 0; nBv = 0
        for s in solsB:
            acc = np.zeros(D, dtype=np.int64)
            for j in range(4):
                r, z = divmod(int(s[j]), 3); acc = (acc + c[j] * w3[z] % P * Vbw[j][r]) % P
            if acc.any(): continue
            nBv += 1
            for zs in itertools.product(range(3), repeat=4):
                p = (-lifter.t) % P
                for j in range(4):
                    r = int(s[j]) // 3; p = (p - c[j] * w3[zs[j]] % P * Vfw[j][r]) % P
                if not p.any(): continue
                li = letter_of(p, lk, P, w3)
                if li < 0: continue
                out.append(dict(case='B', rep=oi, terms=[assemble(j, int(s[j]) // 3, int(s[j]) % 3, zs[j]).tolist() for j in range(4)] + [product_code(codes4[li], 2).tolist()])); nB += 1
        nC = 0; nsolC = 0; nCv = 0
        for kk in range(4):
            J = [j for j in range(4) if j != kk]
            solsC = mitm(np.stack([FVb[j] for j in J]), c[J], P, f0); nsolC += len(solsC)
            for s in solsC:
                acc = np.zeros(D, dtype=np.int64)
                for i, j in enumerate(J):
                    r, z = divmod(int(s[i]), 3); acc = (acc + c[j] * w3[z] % P * Vbw[j][r]) % P
                if acc.any(): continue
                nCv += 1
                for zs in itertools.product(range(3), repeat=3):
                    p = (-lifter.t) % P
                    for i, j in enumerate(J):
                        r = int(s[i]) // 3; p = (p - c[j] * w3[zs[i]] % P * Vfw[j][r]) % P
                    if not p.any(): continue
                    li = letter_of(p, lk, P, w3)
                    if li < 0: continue
                    out.append(dict(case='C', rep=oi, k=kk, terms=[assemble(j, int(s[i]) // 3, int(s[i]) % 3, zs[i]).tolist() for i, j in enumerate(J)] + [product_code(base[kk], 1).tolist(), product_code(codes4[li], 2).tolist()])); nC += 1
        stats.append(dict(rep=oi, A_screen=len(solsA), A_verified=nAv, A=nA, B_screen=len(solsB), B_verified=nBv, B=nB, C_screen=nsolC, C_verified=nCv, C=nC))
        say(f'[prod5] rep {oi}: case A {len(solsA)} screen hits, {nAv} E2 solutions, {nA} decompositions; case B {len(solsB)} / {nBv} / {nB}; case C {nsolC} / {nCv} / {nC}; {time.time()-t0:.0f}s')
    json.dump(dict(n=len(out), stats=stats, found=out, seconds=time.time() - t0), open(os.path.join(outdir, 'prod5.json'), 'w'))
    say(f'[prod5] {len(out)} five-term decompositions of S^5 with a product term on the last qutrit')
    print('PROD5 DONE 0' if not out else f'PROD5 FOUND {len(out)}', flush=True)
    return len(out)

if __name__ == '__main__':
    main(sys.argv[1])
