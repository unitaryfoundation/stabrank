"""Level-4 bases with a product term on qutrit 4 (round 33, agent B): every five-term stabilizer decomposition of
S^4 (all coefficients nonzero) whose slice tuple (|F|, |P_0|, |P_1|, |P_2|) on the last qutrit is not (5, 0, 0, 0).
By Lemma F.2 (S0540) with chi(S^3) = 4 the tuple is (4,1,0,0), (4,0,1,0), (4,0,0,1), or (3,0,1,1); after the
negation of the last qutrit (in G_4) one of the <1| or <2| slices is a four-term decomposition D of S^3, one of the
four orbit representatives under G_3 (acting on qutrits 1 to 3), with unique coefficients c. Cases, as in round 30's
stage five one level down:
  A (4,1,0,0): E2 sum_j c_j zeta_j P_j w_j = -t (MITM); the product term q (x) |0> with q proportional to
     sum_j c_j zeta'_j P_j^-1 w_j, which must be a nonzero letter;
  B (4,0,0,1): E0 sum_j c_j zeta'_j P_j^-1 w_j = 0 (MITM); the product term p (x) |2> with p proportional to
     -t - sum_j c_j zeta_j P_j w_j, a nonzero letter;
  C (3,0,1,1): one base letter w_k stays as the product w_k (x) |1>; E0 over the other three; the product p_2 (x) |2>
     proportional to -t - sum_{j != k} c_j zeta_j P_j w_j, a nonzero letter.
Outputs the level-4 bases (5-multisets of four-qutrit codes) with their coefficient vectors mod P.
    python prod4.py OUTDIR
"""
import numpy as np, itertools, os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L, lift
from lift import _funcvals, _dot, mitm, option_tables

def letter_of(vec, lk, P, w3):
    """index of the letter proportional (mod P) to vec, or -1; vec nonzero."""
    nz = np.flatnonzero(vec)
    if len(nz) == 0: return -1
    inv = pow(int(vec[nz[0]]), P - 2, P); u = vec * inv % P
    code = np.zeros(len(vec), dtype=np.int8)
    for val, e in ((1, 1), (int(w3[1]), 2), (int(w3[2]), 3)):
        code[u == val] = e
    if int((code != 0).sum()) != len(nz): return -1
    return lk.find(code)

def product_code(letter_code, a):
    """code of letter (x) |a> on one more qutrit (last)."""
    out = np.zeros(3 * len(letter_code), dtype=np.int8); out[a::3] = letter_code; return out

def main(outdir):
    log = open(os.path.join(outdir, 'prod4.log'), 'a'); t0 = time.time()
    P = L.PRIMES[0]; lifter = lift.Lifter(4, P=P); w3 = lifter.w3
    codes3 = L.letters(3); lk = L.Lookup(codes3)
    reps4 = json.load(open(os.path.join(outdir, 'b3_reps.json')))['dec4_reps']     # four orbit reps of the 4-decs of S^3 (letter indices)
    out = []
    for oi, rep in enumerate(reps4):
        base = codes3[rep]; c, allnz = lifter.solve_coefficients(base); assert c is not None and allnz
        fw, bw = option_tables(base, lifter.rays, P); R = fw[0].shape[0]
        Vfw = [lifter.tab[fw[j]] for j in range(4)]; Vbw = [lifter.tab[bw[j]] for j in range(4)]
        FVf = np.stack([_funcvals(lifter.f, lifter.g, Vfw[j], P) for j in range(4)])
        FVb = np.stack([_funcvals(lifter.f, lifter.g, Vbw[j], P) for j in range(4)])
        ftm = np.array([(-_dot(lifter.f, lifter.t, P)) % P, (-_dot(lifter.g, lifter.t, P)) % P], dtype=np.int64)
        f0 = np.zeros(2, dtype=np.int64)
        def assemble(j, r, z0, z2):
            c0 = bw[j][r].copy(); nz0 = c0 != 0; c0 = np.where(nz0, ((c0 - 1 + z0) % 3) + 1, 0).astype(np.int8)
            c2 = fw[j][r].copy(); nz2 = c2 != 0; c2 = np.where(nz2, ((c2 - 1 + z2) % 3) + 1, 0).astype(np.int8)
            return L.canon(L.assemble_last(c0, base[j], c2).reshape(1, -1))[0]
        # case A
        solsA = mitm(FVf, c, P, ftm); nA = 0
        for s in solsA:
            acc = np.zeros(27, dtype=np.int64)
            for j in range(4):
                r, z = divmod(int(s[j]), 3); acc = (acc + c[j] * w3[z] % P * Vfw[j][r]) % P
            if not np.array_equal(acc, (-lifter.t) % P): continue
            for zs in itertools.product(range(3), repeat=4):
                q = np.zeros(27, dtype=np.int64)
                for j in range(4):
                    r = int(s[j]) // 3; q = (q + c[j] * w3[zs[j]] % P * Vbw[j][r]) % P
                if not q.any(): continue
                li = letter_of(q, lk, P, w3)
                if li < 0: continue
                terms = [assemble(j, int(s[j]) // 3, zs[j], int(s[j]) % 3) for j in range(4)] + [product_code(codes3[li], 0)]
                out.append(dict(case='A', rep=oi, terms=[t.tolist() for t in terms])); nA += 1
        # case B
        solsB = mitm(FVb, c, P, f0); nB = 0
        for s in solsB:
            acc = np.zeros(27, dtype=np.int64)
            for j in range(4):
                r, z = divmod(int(s[j]), 3); acc = (acc + c[j] * w3[z] % P * Vbw[j][r]) % P
            if acc.any(): continue
            for zs in itertools.product(range(3), repeat=4):
                p = (-lifter.t) % P
                for j in range(4):
                    r = int(s[j]) // 3; p = (p - c[j] * w3[zs[j]] % P * Vfw[j][r]) % P
                if not p.any(): continue
                li = letter_of(p, lk, P, w3)
                if li < 0: continue
                terms = [assemble(j, int(s[j]) // 3, int(s[j]) % 3, zs[j]) for j in range(4)] + [product_code(codes3[li], 2)]
                out.append(dict(case='B', rep=oi, terms=[t.tolist() for t in terms])); nB += 1
        # case C
        nC = 0; nsolC = 0
        for kk in range(4):
            J = [j for j in range(4) if j != kk]
            solsC = mitm(np.stack([FVb[j] for j in J]), c[J], P, f0); nsolC += len(solsC)
            for s in solsC:
                acc = np.zeros(27, dtype=np.int64)
                for i, j in enumerate(J):
                    r, z = divmod(int(s[i]), 3); acc = (acc + c[j] * w3[z] % P * Vbw[j][r]) % P
                if acc.any(): continue
                for zs in itertools.product(range(3), repeat=3):
                    p = (-lifter.t) % P
                    for i, j in enumerate(J):
                        r = int(s[i]) // 3; p = (p - c[j] * w3[zs[i]] % P * Vfw[j][r]) % P
                    if not p.any(): continue
                    li = letter_of(p, lk, P, w3)
                    if li < 0: continue
                    terms = [assemble(j, int(s[i]) // 3, int(s[i]) % 3, zs[i]) for i, j in enumerate(J)] + [product_code(base[kk], 1), product_code(codes3[li], 2)]
                    out.append(dict(case='C', rep=oi, k=kk, terms=[t.tolist() for t in terms])); nC += 1
        print(f'[prod4] rep {oi}: case A {len(solsA)} E2 solutions -> {nA} bases; case B {len(solsB)} E0 solutions -> {nB}; case C {nsolC} E0 solutions -> {nC}; {time.time()-t0:.0f}s', file=log, flush=True)
    # verify every output: five-term decomposition of S^4 with all coefficients nonzero (mod both primes)
    nok = 0
    for d in out:
        T = np.array(d['terms'], dtype=np.int8); v = lifter.verify(T, None); d['verify'] = v; nok += v['ok']
    json.dump(dict(n=len(out), n_ok=nok, bases=out), open(os.path.join(outdir, 'prod4.json'), 'w'))
    print(f'[prod4] {len(out)} level-4 bases with a product term, {nok} verified', file=log, flush=True)
    print(f'PROD4 DONE {len(out)} bases, {nok} verified', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
