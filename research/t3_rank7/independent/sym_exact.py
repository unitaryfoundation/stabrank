"""Exact stabilizer of V_3 = span{psi_0, psi_1, psi_2} in the three-qutrit Clifford group
(mod phase), together with complex conjugation, and its action on the 30240-state
dictionary.

Conventions.  Weyl operators W(a, b) = w^{-a.b} X^a Z^b (w = w3), so that
W(u) W(v) = w^{omega(u, v)} W(u + v), omega((a, b), (a', b')) = a.b' - a'.b, and
W(t) W(u) W(t)^dag = w^{-omega(t, u)} W(u).  For odd p the Weil representation gives, for
every S in Sp(6, 3), a unitary C_S with C_S W(v) C_S^dag = W(S v) exactly, unique mod
phase; every Clifford is W(t) C_S mod phase.  The Pauli expansion Pi = (1/27) sum_v c(v)
W(v) of the projector onto V_3 has c(v) = tr(W(-v) Pi) = d(v) / 9 with d(v) in Z[w3], and

    W(t) C_S  preserves V_3   iff   d(S v) = w^{-omega(t, S v)} d(v)  for all v,

an exact identity in Z[w3] (integer pairs).  Necessary: the norm N(d(Sv)) = N(d(v)).
The search enumerates all symplectic S whose generator images preserve the norms of d on
the partial spans (a complete backtracking: the condition is necessary for any solution),
then solves for t exactly.  Complex conjugation K maps W(a, b) to W(a, -b) and d to its
conjugate; K preserves V_3 iff d(D v) = conj(d(v)), checked exactly.

Action on states.  A stabilizer state is (L, e) with L a Lagrangian subspace and
W(v) psi = w^{e(v)} psi for v in L, e linear; W(t) C_S maps it to (S L, e') with
e'(S v) = e(v) + omega(t, S v), and K maps it to (D L, -e).  The 1120 * 27 = 30240 pairs
(L, e) are exactly the dictionary, so every element permutes it; the permutation is
computed from the (L, e) data (found exactly from the exponent arrays) and verified by
an exact comparison of the transformed data with the dictionary entry.
"""
import itertools, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert7 as C

NOSUPP = 6
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------- F_3^6 tables ----
VEC = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int64)   # code -> (a, b)
def code_of(v):
    v = np.asarray(v) % 3
    return int(v[0]*243 + v[1]*81 + v[2]*27 + v[3]*9 + v[4]*3 + v[5])
CODE = (VEC * np.array([243, 81, 27, 9, 3, 1])).sum(1)
assert np.all(CODE == np.arange(729))
ADD = ((VEC[:, None, :] + VEC[None, :, :]) % 3 * np.array([243, 81, 27, 9, 3, 1])).sum(2).astype(np.int16)
NEG = (((-VEC) % 3) * np.array([243, 81, 27, 9, 3, 1])).sum(1).astype(np.int16)
A_, B_ = VEC[:, :3], VEC[:, 3:]
OMEGA = ((A_[:, None, :] * B_[None, :, :]).sum(2) - (A_[None, :, :] * B_[:, None, :]).sum(2)) % 3   # omega[u, v]
OMEGA = OMEGA.astype(np.int8)
DCONJ = (np.concatenate([A_, (-B_) % 3], axis=1) * np.array([243, 81, 27, 9, 3, 1])).sum(1).astype(np.int16)

def smul(S, code):
    """S: 6x6 matrix over F_3 (columns = images of basis vectors); returns S v codes."""
    v = VEC[code]
    return ((v @ S.T) % 3 * np.array([243, 81, 27, 9, 3, 1])).sum(-1)

# ---------------------------------------------------------- dictionary data ----
m = 3; dim = 27
E = C.build_dictionary(m); N = E.shape[0]
T = C.t3_targets(m)
X = np.array(list(itertools.product(range(3), repeat=3)), dtype=np.int64)   # pos -> x (x1 most significant)
POS = X @ np.array([9, 3, 1])
assert np.all(POS == np.arange(27))

def weyl_data(a, b):
    """(W(a,b) psi)[y] = w^{b.y + a.b} psi[y - a]: returns (source index of y, phase exponent mod 6)."""
    src = (((X - a) % 3) @ np.array([9, 3, 1]))
    ph = (2 * ((X @ b) + a @ b)) % 6
    return src, ph

def stabilizer_data(Emat):
    """For each state, sorted codes of its Lagrangian L (27) and e(v) (W(v) psi = w^{e} psi)."""
    n = Emat.shape[0]
    supp = Emat != NOSUPP
    Lc = np.full((n, 27), -1, dtype=np.int16)
    Le = np.full((n, 27), -1, dtype=np.int8)
    cnt = np.zeros(n, dtype=np.int64)
    Ei = Emat.astype(np.int64)
    for code in range(729):
        a, b = VEC[code, :3], VEC[code, 3:]
        src, ph = weyl_data(a, b)
        Wp = Ei[:, src]                          # (W psi)[y] before phase, as exponents
        okS = np.all((Wp != NOSUPP) == supp, axis=1)
        val = (Wp + ph[None, :]) % 6
        diff = (val - Ei) % 6                    # meaningful on support
        diff = np.where(supp, diff, -1)
        # constant on support and even
        first = diff[np.arange(n), np.argmax(supp, axis=1)]
        okC = np.all((diff == first[:, None]) | ~supp, axis=1) & (first % 2 == 0) & okS
        idx = np.flatnonzero(okC)
        Lc[idx, cnt[idx]] = code
        Le[idx, cnt[idx]] = (first[idx] // 2) % 3   # W psi = w^{e} psi with w = w6^2: diff = 2e
        cnt[idx] += 1
    assert np.all(cnt == 27), np.unique(cnt)
    return Lc, Le

Lc, Le = stabilizer_data(E)
log(f"stabilizer data for {N} states: every state has |L| = 27")
# check isotropy and linearity of e for a sample
for s in (0, 1234, 30239):
    L = Lc[s].astype(int)
    assert np.all(OMEGA[np.ix_(L, L)] == 0)
    em = dict(zip(L, Le[s].astype(int)))
    for u in L:
        for v in L:
            assert em[int(ADD[u, v])] == (em[u] + em[v]) % 3

# keys: (L, e) -> index via exact sorted bytes (codes sorted, e follows)
def pack(Lc_, Le_):
    order = np.argsort(Lc_, axis=1, kind="stable")
    Ls = np.take_along_axis(Lc_, order, axis=1).astype(np.int32)
    Es = np.take_along_axis(Le_, order, axis=1).astype(np.int32)
    return Ls * 3 + Es      # (n, 27) int32, sorted by code
KEY = pack(Lc, Le)
keyd = {row.tobytes(): i for i, row in enumerate(KEY)}
assert len(keyd) == N
RND = np.random.default_rng(7).integers(0, 2**63 - 1, size=729 * 3, dtype=np.int64)
HASH = RND[KEY].sum(axis=1)                 # wraps mod 2^64
hord = np.argsort(HASH); HS = HASH[hord]
assert len(np.unique(HS)) == N, "hash collision in dictionary keys; change RND seed"

def act(S, t, K=False):
    """Permutation P of the dictionary induced by W(t) C_S (then K if K): state s -> P[s]."""
    Lnew = smul(S, Lc.astype(np.int64))                   # (N, 27) codes S v
    Enew = (Le.astype(np.int64) + OMEGA[t, Lnew].astype(np.int64)) % 3
    if K:
        Lnew = DCONJ[Lnew].astype(np.int64)
        Enew = (-Enew) % 3
    key = pack(Lnew.astype(np.int16), Enew.astype(np.int8))
    h = RND[key].sum(axis=1)
    pos = np.searchsorted(HS, h)
    assert np.all(pos < N) and np.all(HS[pos] == h), "image not in dictionary"
    P = hord[pos]
    assert np.array_equal(KEY[P], key), "exact key mismatch"      # exact verification
    assert len(np.unique(P)) == N
    return P

# ----------------------------------------------------------- d(v) exactly ----
# psi_r entries w3^q (exponent array T[r] in w6 units: 2q), support {f = r mod 3}
Ti = T.astype(np.int64)
Tsupp = Ti != NOSUPP
def d_of(code):
    """d(v) = sum_r <psi_r| W(-v) |psi_r> in Z[w3] as (a, b) meaning a + b w3."""
    a, b = VEC[NEG[code], :3], VEC[NEG[code], 3:]
    src, ph = weyl_data(a, b)
    cnt = np.zeros(3, dtype=np.int64)
    for r in range(3):
        Wp = Ti[r, src]
        ok = (Wp != NOSUPP) & Tsupp[r]
        # conj(psi[y]) * (W psi)[y]: exponent (Wp + ph - T) in w6 units, must be even
        ex = ((Wp + ph - Ti[r]) % 6)[ok]
        assert np.all(ex % 2 == 0)
        for q in range(3):
            cnt[q] += int(np.sum(ex == 2 * q))
    return int(cnt[0] - cnt[2]), int(cnt[1] - cnt[2])
D = np.array([d_of(c) for c in range(729)], dtype=np.int64)     # (729, 2)
NORM = D[:, 0] ** 2 - D[:, 0] * D[:, 1] + D[:, 1] ** 2
assert tuple(D[0]) == (27, 0), D[0]     # tr Pi * 9 = 27 (three vectors of norm 9)
from collections import Counter
log(f"d(v) computed; norm classes: {sorted(Counter(NORM.tolist()).items())}; support size {int(np.sum(NORM > 0))}")

def mulw(d, k):
    """d * w3^k for d = (a, b): w3 (a + b w) = a w + b w^2 = -b + (a - b) w."""
    a, b = d
    for _ in range(k % 3):
        a, b = -b, a - b
    return (a, b)
W3POW = {}   # d -> {k: d w^k}

# Antiunitary elements K W(t) C_S: they preserve V_3 iff W(t) C_S Pi (W(t) C_S)^dag = conj(Pi),
# whose Weyl coefficients are d~(u) = conj(d(D u)) (conj(a + b w) = (a - b) - b w).
DT = np.stack([D[DCONJ, 0] - D[DCONJ, 1], -D[DCONJ, 1]], axis=1)
NORMT = DT[:, 0] ** 2 - DT[:, 0] * DT[:, 1] + DT[:, 1] ** 2
Kalone = np.array_equal(DT, D)
log(f"complex conjugation alone preserves V_3: {Kalone}")

# ------------------------------------------------------ symplectic search ----
GENS = [code_of([1,0,0,0,0,0]), code_of([0,0,0,1,0,0]), code_of([0,1,0,0,0,0]),
        code_of([0,0,0,0,1,0]), code_of([0,0,1,0,0,0]), code_of([0,0,0,0,0,1])]
def span_codes(gs):
    out = np.zeros(1, dtype=np.int64)
    for g in gs:
        out = np.concatenate([out, ADD[out, g], ADD[ADD[out, g], g]])
    return out
SPANS = [span_codes(GENS[:t]) for t in range(7)]          # in a fixed coefficient order

def search(NORM_img):
    """All symplectic S with NORM_img(S v) = NORM(v) for all v (complete backtracking)."""
    sols = []
    def bt(imgs):
        t = len(imgs)
        if t == 6:
            sols.append(list(imgs)); return
        g = GENS[t]
        cand = np.arange(1, 729)
        for s in range(t):
            cand = cand[OMEGA[imgs[s], cand] == OMEGA[GENS[s], g]]
        cand = cand[NORM_img[cand] == NORM[g]]
        base = SPANS[t]
        img_base = span_codes(imgs) if t else np.zeros(1, dtype=np.int64)
        tgt = np.concatenate([NORM[base], NORM[ADD[base, g]], NORM[ADD[ADD[base, g], g]]])
        for u in cand:
            got = np.concatenate([NORM_img[img_base], NORM_img[ADD[img_base, u]], NORM_img[ADD[ADD[img_base, u], u]]])
            if np.array_equal(got, tgt):
                bt(imgs + [int(u)])
    bt([])
    return sols

GEN_COL = [int(np.flatnonzero(VEC[g])[0]) for g in GENS]      # coordinate of each generator
def S_matrix(imgs):
    """Matrix of S: column GEN_COL[s] (the coordinate of generator s) is the image of generator s."""
    S = np.zeros((6, 6), dtype=np.int64)
    for s, u in enumerate(imgs):
        S[:, GEN_COL[s]] = VEC[u]
    assert np.array_equal(smul(S, np.array(GENS)), np.array(imgs))
    return S
DW = np.zeros((729, 3, 2), dtype=np.int64)
for v in range(729):
    for k in range(3):
        DW[v, k] = mulw((int(D[v, 0]), int(D[v, 1])), k)

def lift_all(sols, Dtarget):
    """Elements (S, t) with Dtarget[S v] = w^{-omega(t, S v)} D[v] for all v."""
    out = []; nS = 0
    for imgs in sols:
        S = S_matrix(imgs)
        Sv = smul(S, np.arange(729))
        assert len(np.unique(Sv)) == 729
        ok_t = []
        for t in range(729):
            k = (-OMEGA[t, Sv].astype(np.int64)) % 3
            if np.array_equal(Dtarget[Sv], DW[np.arange(729), k]):
                ok_t.append(t)
        if ok_t:
            nS += 1
            out.extend((S, t) for t in ok_t)
    return out, nS

sols = search(NORM)
log(f"symplectic candidates (unitary) preserving the norm of d: {len(sols)}")
elemU, nSU = lift_all(sols, D)
log(f"unitary: symplectic parts that lift {nSU}; elements W(t) C_S preserving V_3 (mod phase): {len(elemU)}")
solsA = search(NORMT)
log(f"symplectic candidates (antiunitary) matching the norm of d~: {len(solsA)}")
elemA, nSA = lift_all(solsA, DT)
log(f"antiunitary: symplectic parts that lift {nSA}; elements K W(t) C_S preserving V_3: {len(elemA)}")
elements = [(S, t, 0) for S, t in elemU] + [(S, t, 1) for S, t in elemA]
nU = len(elemU); order = len(elements)
log(f"|G'| = {order} (unitary {nU}, antiunitary {len(elemA)}); known group 2916")

# ---------------------------------------------------- closure verification ----
DMAT = np.diag([1, 1, 1, 2, 2, 2])
def compose(e1, e2):
    """K^k1 W(t1) C_S1 . K^k2 W(t2) C_S2 = K^(k1+k2) W(D^k2 t1 + D^k2 S1 D^k2 t2) C_(D^k2 S1 D^k2 S2)."""
    S1, t1, k1 = e1; S2, t2, k2 = e2
    if k2:
        S1 = (DMAT @ S1 @ DMAT) % 3; t1 = int(DCONJ[t1])
    return ((S1 @ S2) % 3, int(ADD[t1, smul(S1, t2)]), (k1 + k2) % 2)
def ekey(e):
    return e[0].tobytes() + bytes([e[1] // 256, e[1] % 256, e[2]])
eset = {ekey(e): n for n, e in enumerate(elements)}
assert len(eset) == order
rng = np.random.default_rng(1)
gens = []
ID = (np.eye(6, dtype=np.int64), 0, 0)
reach = {ekey(ID)}
while len(reach) < order:
    cand_idx = [n for n in range(order) if ekey(elements[n]) not in reach]
    gens.append(elements[int(rng.choice(cand_idx))])
    todo = [ID]; reach = {ekey(ID)}
    while todo:
        e = todo.pop()
        for h in gens:
            f = compose(e, h); k = ekey(f)
            assert k in eset, "product of two group elements is not in the set (closure fails)"
            if k not in reach:
                reach.add(k); todo.append(f)
log(f"set closed under multiplication; generated by {len(gens)} elements, size {len(reach)} = {order}")

# -------------------------------------------------------- permutation action ----
gen_perms = [act(S, t, K=bool(k)) for S, t, k in gens]
log(f"generator permutations computed and verified exactly ({len(gen_perms)} generators)")
# orbits by union-find
parent = np.arange(N)
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for P in gen_perms:
    for s in range(N):
        a, b = find(s), find(int(P[s]))
        if a != b: parent[max(a, b)] = min(a, b)
roots = np.array([find(s) for s in range(N)])
orbits = np.unique(roots)
sizes = Counter(roots.tolist())
log(f"orbits of G' on the dictionary: {len(orbits)}; sizes: {sorted(Counter(sizes.values()).items())}")
# sanity: orbit-stabilizer with |G'|: each orbit size must divide |G'|
assert all(order % sz == 0 for sz in sizes.values())
np.savez(os.path.join(HERE, "logs", "sym_exact.npz"),
         S=np.stack([e[0] for e in elements]), t=np.array([e[1] for e in elements]), k=np.array([e[2] for e in elements]),
         order=order, gens_S=np.stack([g[0] for g in gens]), gens_t=np.array([g[1] for g in gens]), gens_k=np.array([g[2] for g in gens]),
         roots=roots, Lc=Lc, Le=Le)
log("saved logs/sym_exact.npz")
