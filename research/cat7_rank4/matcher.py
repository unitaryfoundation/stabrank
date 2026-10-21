"""Completion of a base slice of c_7 to a rank-4 decomposition: stages A, B, C
(four visible terms at the base point 00) and beta (three visible terms and
one term on a flat missing 00). See cat_slice2.py for the argument. Every
screen is mod P on exact Gaussian-rational data; every hit is re-decided
exactly, and a surviving candidate is checked to be a decomposition of the
target into stabilizer states.
"""
import os, sys, json, time, itertools
from fractions import Fraction
import numpy as np
def _find_root():
    """the public repository root: STABRANK_PUB, else the nearest ancestor of
    this file holding verify_challenge/stabrank_verify.py, else the worktree."""
    env = os.environ.get("STABRANK_PUB")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "verify_challenge", "stabrank_verify.py")):
            return d
        d = os.path.dirname(d)
    return "/Users/vincent.russo/Projects/genesis/stabrank-r31j"

PUB = _find_root()
sys.path.insert(0, os.path.join(PUB, "verify_challenge"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cert_cat_m7_lift as L
import cat_slice2 as C
P, RI, POW_I = L.P, L.RI, L.POW_I
NB = 4                      # unsliced qubits
DIM = 1 << NB
NOPT = 65                   # absent + 16 classes x 4 phases
POINTS = (1, 2, 3)          # 01, 10, 11
FLATS = {"p01": (1,), "p10": (2,), "p11": (3,), "l0110": (1, 2), "l0111": (1, 3), "l1011": (2, 3)}
RNG = np.random.default_rng(7)
HV = [RNG.integers(1, P, size=DIM) for _ in range(2)]
ZERO = (Fraction(0), Fraction(0))


# ------------------------------------------------------------- exact ------

def gadd(a, b): return (a[0] + b[0], a[1] + b[1])
def gsub(a, b): return (a[0] - b[0], a[1] - b[1])
def gmul(a, b): return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])
def gfrac(x): return (Fraction(x[0]), Fraction(x[1]))

def vadd(u, v): return [gadd(a, b) for a, b in zip(u, v)]
def vsub(u, v): return [gsub(a, b) for a, b in zip(u, v)]
def vscale(c, v): return [gmul(c, a) for a in v]
def vzero(v): return all(a == ZERO for a in v)

def parallel(u, v):
    """lam with u = lam v, or None; v must be nonzero."""
    j = next((i for i, x in enumerate(v) if x != ZERO), None)
    if j is None:
        return None
    lam = L.gdiv(u[j], v[j])
    return lam if vscale(lam, v) == list(u) else None

def exact_vec(e_row):
    return [gfrac(x) for x in L.to_gauss(e_row)]

def mod_of_exact(v):
    return np.array([C.frac_mod(x) for x in v], dtype=np.int64)

def dotp(R, h):
    """(R . h) mod P without int64 overflow: products reduced before the sum."""
    return ((R * h) % P).sum(axis=-1) % P

def hash_rows(R):
    """two-functional hash of mod-P rows -> packed int64 keys."""
    return dotp(R, HV[0]) * P + dotp(R, HV[1])


# --------------------------------------------------- option tables --------

_OPT = {}

def pauli_class_table():
    """class index c = 16-bit (u, v) -> list of (u, v) with the XOR rule."""
    return [(c >> NB, c & (DIM - 1)) for c in range(DIM * DIM)]

def options(E, i):
    """65 x 16 exponent rows (row 0 = absent) for state i: row 1 + 4 r + l is
    i^l Q_r a, Q_r one representative of the r-th Pauli class modulo Stab(a)
    (16 distinct rays). Also returns reps (uv label of each ray) and cls
    (256-entry table: uv label -> ray), so that class products are XORs."""
    if i in _OPT:
        return _OPT[i]
    a = E[i]
    x = np.arange(DIM)
    rays, reps, keys = [], [], {}
    cls = np.zeros(DIM * DIM, np.int64)
    for u in range(DIM):
        for v in range(DIM):
            par = np.array([bin(v & xx).count("1") & 1 for xx in x])
            val = np.where(a >= 0, (a.astype(np.int16) + 2 * par) % 4, -1)
            new = np.full(DIM, -1, np.int16)
            new[x ^ u] = val
            key = L.renormalize(new.astype(np.int8)[None, :])[0].tobytes()
            if key not in keys:
                keys[key] = len(rays)
                rays.append(new); reps.append((u << NB) | v)
            cls[(u << NB) | v] = keys[key]
    if len(rays) != DIM:
        L.fail(f"{len(rays)} Pauli rays of a 4-qubit state, expected {DIM}")
    rows = [np.full(DIM, -1, np.int8)]
    for new in rays:
        for l in range(4):
            rows.append(np.where(new >= 0, (new + l) % 4, -1).astype(np.int8))
    rows = np.array(rows)
    _OPT[i] = (rows, L.to_modp(rows), reps, cls)
    return _OPT[i]

def opt_class(o):
    """option index -> (ray r, phase l) or None for absent."""
    return None if o == 0 else ((o - 1) // 4, (o - 1) % 4)


# ------------------------------------------------------ the target --------

def target_slices_c7():
    sl = C.slices(7)
    return [(L.to_modp(s[None, :])[0], exact_vec(s)) for s in sl]

def target_slices_from_vec(v64):
    """planted target: list of 64 exact Gaussian rationals -> four slices."""
    out = []
    for x in range(4):
        ex = v64[x * DIM:(x + 1) * DIM]
        out.append((mod_of_exact(ex), ex))
    return out


# -------------------------------------------------- per-point solves ------

def solve_point_A(E, base, coefs_mod, coefs_exact, rhs):
    """all option tuples (o1..o4) with sum_j e_j w_j(o_j) = rhs at one point;
    base: 4 state indices with point coefficients."""
    rmod, rex = rhs
    mods = [options(E, i)[1] * cm % P for i, cm in zip(base, coefs_mod)]
    left = (mods[0][:, None, :] + mods[1][None, :, :]) % P                # (65,65,16)
    right = (rmod[None, None, :] - mods[2][:, None, :] - mods[3][None, :, :]) % P
    kl = hash_rows(left.reshape(-1, DIM)); kr = hash_rows(right.reshape(-1, DIM))
    common, il, ir = np.intersect1d(kl, kr, return_indices=True)
    sols = []
    if common.size == 0:
        return sols
    # keys may repeat; expand all pairs
    posl = {}
    for idx, k in enumerate(kl):
        if k in set(common.tolist()):
            posl.setdefault(int(k), []).append(idx)
    for idx, k in enumerate(kr):
        for a in posl.get(int(k), []):
            o1, o2 = divmod(a, NOPT); o3, o4 = divmod(idx, NOPT)
            tup = (o1, o2, o3, o4)
            if check_exact(E, base, coefs_exact, tup, rex):
                sols.append(tup)
    return sols

def check_exact(E, base, coefs_exact, tup, rex):
    acc = [ZERO] * DIM
    for i, c, o in zip(base, coefs_exact, tup):
        if o == 0:
            continue
        acc = vadd(acc, vscale(c, exact_vec(options(E, i)[0][o])))
    return acc == list(rex)


def solve_point_3(E, base3, coefs_mod, coefs_exact, rhs):
    """3 visible terms with point coefficients: tuples with sum = rhs."""
    rmod, rex = rhs
    mods = [options(E, i)[1] * cm % P for i, cm in zip(base3, coefs_mod)]
    left = mods[0]                                                         # (65,16)
    right = (rmod[None, None, :] - mods[1][:, None, :] - mods[2][None, :, :]) % P
    kl = hash_rows(left); kr = hash_rows(right.reshape(-1, DIM))
    common = np.intersect1d(kl, kr)
    sols = []
    if common.size == 0:
        return sols
    cs = set(common.tolist())
    posl = {}
    for idx, k in enumerate(kl):
        if int(k) in cs:
            posl.setdefault(int(k), []).append(idx)
    for idx, k in enumerate(kr):
        for a in posl.get(int(k), []):
            o2, o3 = divmod(idx, NOPT)
            tup = (a, o2, o3)
            if check_exact(E, base3, coefs_exact, tup, rex):
                sols.append(tup)
    return sols


def residuals_3(E, base3, coefs_mod, coefs_exact, rhs):
    """for every option tuple of 3 visible terms: the residual rhs - sum, as mod-P rows
    (65^3 x 16) plus the tuple index decoding."""
    rmod, _ = rhs
    mods = [options(E, i)[1] * cm % P for i, cm in zip(base3, coefs_mod)]
    R = (rmod[None, None, None, :] - mods[0][:, None, None, :] - mods[1][None, :, None, :]
         - mods[2][None, None, :, :]) % P
    return R.reshape(-1, DIM)


def dep_test(Fu, Gu, Fv, Gv):
    return (Fu * Gv - Gu * Fv) % P == 0


# ---------------------------------------------------------- stage A -------

def flat_ok(E, i, o01, o10, o11):
    """the flat rule for the term with base state i: present on {00}, a line
    through 00, or everywhere with ray(11) = ray(01) * ray(10)."""
    cl = [opt_class(o01), opt_class(o10), opt_class(o11)]
    pr = tuple(int(c is not None) for c in cl)
    if pr in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        return True
    if pr == (1, 1, 1):
        _, _, reps, cls = options(E, i)
        return cls[reps[cl[0][0]] ^ reps[cl[1][0]]] == cl[2][0]
    return False

def combine_points(E, states, sol_lists):
    """cartesian product of per-point solutions with the flat rule per term."""
    out = []
    for s1 in sol_lists[0]:
        for s2 in sol_lists[1]:
            for s3 in sol_lists[2]:
                if all(flat_ok(E, i, s1[j], s2[j], s3[j]) for j, i in enumerate(states)):
                    out.append((s1, s2, s3))
    return out


def stage_A(E, covers, slices, log_every=2000):
    hits = []
    t0 = time.time()
    for n, (quad, dstr) in enumerate(covers):
        cex = [(Fraction(a), Fraction(b)) for a, b in dstr]
        cm = [C.frac_mod(c) for c in cex]
        sols = [solve_point_A(E, quad, cm, cex, slices[x]) for x in POINTS]
        if all(sols):
            for comb in combine_points(E, quad, sols):
                hits.append(("A", quad, dstr, comb))
        if log_every and (n + 1) % log_every == 0:
            print(f"  stage A {n + 1}/{len(covers)} hits {len(hits)} {time.time() - t0:.0f}s", flush=True)
    return hits


# -------------------------------------------------------- stage beta ------

def dict_hash_table(E):
    M = L.to_modp(E)
    W, _ = L.proj_normal(M)
    return hash_rows(W)

def stage_beta(E, decs, slices, dkeys, log=True):
    """decs: rank-3 decompositions [(triple, coefficient strings)]."""
    hits = []
    for tri, dstr in decs:
        cex = [(Fraction(a), Fraction(b)) for a, b in dstr]
        cm = [C.frac_mod(c) for c in cex]
        for fname, flat in FLATS.items():
            outside = [x for x in POINTS if x not in flat]
            sols_out = {x: solve_point_3(E, tri, cm, cex, slices[x]) for x in outside}
            if not all(sols_out.values()):
                continue
            # inside the flat: residual must be a multiple of a dictionary state
            inside = {}
            for x in flat:
                R = residuals_3(E, tri, cm, cex, slices[x])
                W, has = L.proj_normal(R)
                keys = hash_rows(W)
                ok = has & np.isin(keys, dkeys)
                inside[x] = np.flatnonzero(ok)
                if inside[x].size == 0:
                    break
            if any(x not in inside or inside[x].size == 0 for x in flat):
                continue
            # combine: enumerate visible-term tuples per point, invisible term from residuals
            per_point = []
            for x in POINTS:
                if x in flat:
                    tups = [tuple(int(v) for v in np.unravel_index(i, (NOPT, NOPT, NOPT))) for i in inside[x]]
                else:
                    tups = sols_out[x]
                per_point.append(tups)
            for comb in combine_points(E, tri, per_point):
                # exact residuals on the flat
                res = {}
                good = True
                for x in flat:
                    tup = comb[POINTS.index(x)]
                    acc = [ZERO] * DIM
                    for i, c, o in zip(tri, cex, tup):
                        if o:
                            acc = vadd(acc, vscale(c, exact_vec(options(E, i)[0][o])))
                    r = vsub(list(slices[x][1]), acc)
                    if vzero(r):
                        good = False; break
                    res[x] = r
                if not good:
                    continue
                if len(flat) == 2:
                    # r_{x'} must be a Pauli image (with phase) of r_x
                    x, xp = flat
                    rx = res[x]
                    found = False
                    for u in range(DIM):
                        for v in range(DIM):
                            img = pauli_apply(rx, u, v)
                            for l in range(4):
                                ph = [(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)), (Fraction(-1), Fraction(0)), (Fraction(0), Fraction(-1))][l]
                                if vscale(ph, img) == res[xp]:
                                    found = True; break
                            if found: break
                        if found: break
                    if not found:
                        continue
                # the residual direction must be a stabilizer state (dictionary hash matched mod P; confirm exactly)
                if not is_stabilizer_exact(res[flat[0]]):
                    continue
                hits.append(("beta", tri, dstr, fname, comb))
        if log:
            print(f"  stage beta base {tri}: hits so far {len(hits)}", flush=True)
    return hits


def pauli_apply(vec, u, v):
    """X^u Z^v on an exact 16-vector."""
    out = [ZERO] * DIM
    for x in range(DIM):
        s = bin(v & x).count("1") & 1
        val = vec[x] if not s else (-vec[x][0], -vec[x][1])
        out[x ^ u] = val
    return out


def is_stabilizer_exact(vec):
    """exact test: support affine, entries (after dividing by the first nonzero)
    fourth roots of unity with a quadratic phase pattern."""
    supp = [x for x in range(len(vec)) if vec[x] != ZERO]
    if not supp or (len(supp) & (len(supp) - 1)):
        return False
    x0 = supp[0]
    piv = vec[x0]
    norm = [L.gdiv(vec[x], piv) if vec[x] != ZERO else None for x in range(len(vec))]
    units = {(Fraction(1), Fraction(0)): 0, (Fraction(0), Fraction(1)): 1,
             (Fraction(-1), Fraction(0)): 2, (Fraction(0), Fraction(-1)): 3}
    f = {}
    for x in supp:
        if norm[x] not in units:
            return False
        f[x] = units[norm[x]]
    dirs = sorted({x ^ x0 for x in supp})
    # affine check: closure of the direction set under xor
    dset = set(dirs)
    for a in dirs:
        for b in dirs:
            if (a ^ b) not in dset:
                return False
    # basis of the direction space
    basis = []
    span = {0}
    for d in dirs:
        if d not in span:
            basis.append(d)
            span = span | {s ^ d for s in span}
    k = len(basis)
    l = [(f[x0 ^ b] - f[x0]) % 4 for b in basis]
    q = {}
    for i in range(k):
        for j in range(i + 1, k):
            d = (f[x0 ^ basis[i] ^ basis[j]] - f[x0] - l[i] - l[j]) % 4
            if d not in (0, 2):
                return False
            q[(i, j)] = d // 2
    for t in itertools.product((0, 1), repeat=k):
        x = x0
        ph = 0
        for i in range(k):
            if t[i]:
                x ^= basis[i]; ph += l[i]
        for (i, j), qq in q.items():
            ph += 2 * qq * t[i] * t[j]
        if (f[x] - f[x0]) % 4 != ph % 4:
            return False
    return True


# ------------------------------------------------------ stages B, C -------



def stage_dependent(E, bases, slices, kind, log=True):
    """kind 'B': base = (tri, f_exact, a4, kappa_exact): e(lam) = (f - lam kappa, lam).
       kind 'C': base = (tri, f_exact, j): copies of tri[j] with e_A + e_B = f_j.
       Both: u(w) = lam v(w) at every point with one lam."""
    hits = []
    for base in bases:
        if kind == "B":
            tri, fex, a4, kap = base
            states = list(tri) + [a4]
            fmod = [C.frac_mod(c) for c in fex]; kmod = [C.frac_mod(c) for c in kap]
            # u = sum_{j<=3} f_j w_j - rhs ; v = sum_{j<=3} kappa_j w_j - w_4
            u_terms = [(states[j], fmod[j]) for j in range(3)]
            v_terms = [(states[j], kmod[j]) for j in range(3)] + [(a4, P - 1)]
        else:
            tri, fex, j = base
            others = [i for i in range(3) if i != j]
            states = [tri[others[0]], tri[others[1]], tri[j], tri[j]]       # w_i, w_i', w_B, w_A
            fmod = [C.frac_mod(c) for c in fex]
            # u = f_i w_i + f_i' w_i' + f_j w_B - rhs ; v = w_B - w_A   (e_A = lam)
            u_terms = [(states[0], fmod[others[0]]), (states[1], fmod[others[1]]), (states[2], fmod[j])]
            v_terms = [(states[2], 1), (states[3], P - 1)]
        per_point = []
        for x in POINTS:
            rmod, rex = slices[x]
            opts = [options(E, s)[1] for s in states]
            # features over the 4 option axes
            fu = [np.zeros(NOPT, np.int64) for _ in range(4)]; gu = [np.zeros(NOPT, np.int64) for _ in range(4)]
            fv = [np.zeros(NOPT, np.int64) for _ in range(4)]; gv = [np.zeros(NOPT, np.int64) for _ in range(4)]
            # map terms to axes: states order defines axes 0..3
            def feats(mat, cm):
                return dotp(mat, HV[0]) * cm % P, dotp(mat, HV[1]) * cm % P
            if kind == "B":
                for ax in range(3):
                    a, b = feats(opts[ax], fmod[ax]); fu[ax], gu[ax] = a, b
                    a, b = feats(opts[ax], kmod[ax]); fv[ax], gv[ax] = a, b
                a, b = feats(opts[3], P - 1); fv[3], gv[3] = a, b
            else:
                a, b = feats(opts[0], fmod[others[0]]); fu[0], gu[0] = a, b
                a, b = feats(opts[1], fmod[others[1]]); fu[1], gu[1] = a, b
                a, b = feats(opts[2], fmod[j]); fu[2], gu[2] = a, b
                a, b = feats(opts[2], 1); fv[2], gv[2] = a, b
                a, b = feats(opts[3], P - 1); fv[3], gv[3] = a, b
            rh = (int(dotp(rmod, HV[0])), int(dotp(rmod, HV[1])))
            FU = (fu[0][:, None, None, None] + fu[1][None, :, None, None] + fu[2][None, None, :, None] + fu[3][None, None, None, :] - rh[0]) % P
            GU = (gu[0][:, None, None, None] + gu[1][None, :, None, None] + gu[2][None, None, :, None] + gu[3][None, None, None, :] - rh[1]) % P
            FV = (fv[0][:, None, None, None] + fv[1][None, :, None, None] + fv[2][None, None, :, None] + fv[3][None, None, None, :]) % P
            GV = (gv[0][:, None, None, None] + gv[1][None, :, None, None] + gv[2][None, None, :, None] + gv[3][None, None, None, :]) % P
            ok = dep_test(FU, GU, FV, GV)
            idx = np.flatnonzero(ok)
            sols = []
            for i in idx:
                tup = tuple(int(v) for v in np.unravel_index(i, (NOPT,) * 4))
                lam = exact_lambda(E, states, tup, rex, kind, fex, kap if kind == "B" else None, j if kind == "C" else None)
                if lam is not False:
                    sols.append((tup, lam))
            per_point.append(sols)
            if not sols:
                break
        if len(per_point) < 3 or not all(per_point):
            if log:
                print(f"  stage {kind} base {base[0]}{'' if kind == 'B' else ' copy ' + str(base[2])}: no completion", flush=True)
            continue
        # combine with one lambda (None = free) and the flat rule per term
        for s1 in per_point[0]:
            for s2 in per_point[1]:
                for s3 in per_point[2]:
                    lams = [s for s in (s1[1], s2[1], s3[1]) if s is not None]
                    if lams and any(l != lams[0] for l in lams):
                        continue
                    comb = (s1[0], s2[0], s3[0])
                    if all(flat_ok(E, states[t], comb[0][t], comb[1][t], comb[2][t]) for t in range(4)):
                        hits.append((kind, base, comb, lams[0] if lams else None))
        if log:
            print(f"  stage {kind} base {base[0]}: per-point solutions {[len(s) for s in per_point]}, hits {len(hits)}", flush=True)
    return hits


def exact_lambda(E, states, tup, rex, kind, fex, kap, j):
    """exact: u = lam v? return lam (None if v = 0 and u = 0), or False."""
    def vec_of(ax):
        o = tup[ax]
        return None if o == 0 else exact_vec(options(E, states[ax])[0][o])
    vecs = [vec_of(ax) for ax in range(4)]
    u = [ZERO] * DIM; v = [ZERO] * DIM
    if kind == "B":
        for ax in range(3):
            if vecs[ax] is not None:
                u = vadd(u, vscale(fex[ax], vecs[ax]))
                v = vadd(v, vscale(kap[ax], vecs[ax]))
        if vecs[3] is not None:
            v = vsub(v, vecs[3])
    else:
        others = [i for i in range(3) if i != j]
        if vecs[0] is not None: u = vadd(u, vscale(fex[others[0]], vecs[0]))
        if vecs[1] is not None: u = vadd(u, vscale(fex[others[1]], vecs[1]))
        if vecs[2] is not None:
            u = vadd(u, vscale(fex[j], vecs[2])); v = vadd(v, vecs[2])
        if vecs[3] is not None: v = vsub(v, vecs[3])
    u = vsub(u, list(rex))
    if vzero(v):
        return None if vzero(u) else False
    lam = parallel(u, v)
    return lam if lam is not None else False


# ------------------------------------------------- final confirmation -----

def term_vector(E, i, opts4):
    """exact 64-vector of a term with base state i and option indices at (00, 01, 10, 11)."""
    rows = options(E, i)[0]
    v = []
    for x in range(4):
        o = opts4[x]
        for y in range(DIM):
            if o == 0 or rows[o][y] < 0:
                v.append(ZERO)
            else:
                v.append([(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)),
                          (Fraction(-1), Fraction(0)), (Fraction(0), Fraction(-1))][int(rows[o][y])])
    return v


def confirm_hit(E, hit, slices, target64):
    """Rebuild the four six-qubit terms of a hit and decide exactly whether they
    are distinct stabilizer states whose combination is the target."""
    kind = hit[0]
    terms, coefs = [], []
    if kind == "A":
        quad, dstr, comb = hit[1], hit[2], hit[3]
        for j, i in enumerate(quad):
            terms.append(term_vector(E, i, (1, comb[0][j], comb[1][j], comb[2][j])))
            coefs.append((Fraction(dstr[j][0]), Fraction(dstr[j][1])))
    elif kind == "beta":
        tri, dstr, fname, comb = hit[1], hit[2], hit[3], hit[4]
        flat = FLATS[fname]
        for j, i in enumerate(tri):
            terms.append(term_vector(E, i, (1, comb[0][j], comb[1][j], comb[2][j])))
            coefs.append((Fraction(dstr[j][0]), Fraction(dstr[j][1])))
        t4 = [ZERO] * 64
        for x in flat:
            acc = [ZERO] * DIM
            for j, i in enumerate(tri):
                o = comb[POINTS.index(x)][j]
                if o:
                    acc = vadd(acc, vscale(coefs[j], exact_vec(options(E, i)[0][o])))
            r = vsub(list(slices[x][1]), acc)
            t4[x * DIM:(x + 1) * DIM] = r
        terms.append(t4); coefs.append((Fraction(1), Fraction(0)))
    elif kind == "B":
        (tri, fex, a4, kap), comb, lam = hit[1], hit[2], hit[3]
        if lam is None:
            lam = ZERO      # a free parameter: lambda = 0 puts the fourth state out; the common check decides
        states = list(tri) + [a4]
        cs = [gsub(fex[j], gmul(lam, kap[j])) for j in range(3)] + [lam]
        for j, i in enumerate(states):
            terms.append(term_vector(E, i, (1, comb[0][j], comb[1][j], comb[2][j])))
            coefs.append(cs[j])
    elif kind == "C":
        (tri, fex, jj), comb, lam = hit[1], hit[2], hit[3]
        if lam is None:
            lam = ZERO      # identical copies at every point: lambda = 0 merges them; the common check decides
        others = [i for i in range(3) if i != jj]
        states = [tri[others[0]], tri[others[1]], tri[jj], tri[jj]]
        cs = [fex[others[0]], fex[others[1]], gsub(fex[jj], lam), lam]
        for j, i in enumerate(states):
            terms.append(term_vector(E, i, (1, comb[0][j], comb[1][j], comb[2][j])))
            coefs.append(cs[j])
    # Finish the exact check on every candidate (referee, 2026-10-17): terms with a
    # zero coefficient drop out (this covers the free-parameter paths of stages B and
    # C, taken at lambda = 0), parallel terms merge, and whatever is left must be
    # stabilizer states combining exactly to the target. A survivor with fewer than
    # four distinct terms would be a decomposition of c_7 with at most three terms,
    # which the caller treats as a failure of the premise chi(c_7) >= 4 rather than
    # discarding it unexamined.
    live = [(c, t) for c, t in zip(coefs, terms) if c != ZERO]
    merged = []                                        # (coefficient, normalized term)
    for c, t in live:
        j0 = next((i for i, x in enumerate(t) if x != ZERO), None)
        if j0 is None:
            return False, "a zero term"
        piv = t[j0]
        unit = [L.gdiv(x, piv) for x in t]
        cc = gmul(c, piv)
        for k, (c2, u2) in enumerate(merged):
            if u2 == unit:
                merged[k] = (gadd(c2, cc), u2)
                break
        else:
            merged.append((cc, unit))
    merged = [(c, u) for c, u in merged if c != ZERO]
    if not merged:
        return False, "every coefficient vanishes"
    if not all(is_stabilizer_exact(u) for _, u in merged):
        return False, "a term is not a stabilizer state"
    acc = [ZERO] * 64
    for c, u in merged:
        acc = vadd(acc, vscale(c, u))
    if acc != list(target64):
        return False, "combination is not the target"
    return True, ([u for _, u in merged], [c for c, _ in merged])
