"""Certificate: chi(|cat_7>) >= 5 (and chi(|cat_8>) >= 5 by the <0| slice
identity), by the exclusion of every rank-4 decomposition of the even-weight
compression c_7 of |cat_7> through a two-qubit base slice, over a stored
census of the full 4-covers of c_5 (attested tier).

The argument (research/cat7_rank4/REPORT.md in the research record;
cat_slice2.py and matcher.py here). c_7(y) = cat_7(y, parity(y)) on six qubits
has chi(c_7) = chi(cat_7) (cert_cat_m5_rank2.py, step 1). Slice c_7 along its
first two qubits: the 00 slice is c_5, the 11 slice is i c_5 and the 01, 10
slices are i (-i)^{par(y)} c_5(y), a diagonal Clifford image (asserted). In a
rank-4 decomposition c_7 = sum e_j s_j the base point 00 sees k >= chi(c_5) = 3
terms (the exact rank-2 exclusion of cert_cat_m5_rank2.py, re-run here):
  k = 4: the base is a 4-multiset of four-qubit stabilizer states with all
         coefficients nonzero and c_5 in its span: independent (stage A, the
         stored census), dependent (stage B: a rank-3 decomposition of c_5 plus
         a state of its span), or with a repeated state (stage C);
  k = 3: a rank-3 decomposition of c_5, the fourth term invisible at 00 and
         supported on one of the six affine flats of F_2^2 missing 00 (beta).
A term visible at 00 is nonzero on an affine flat through 00 and its slice at
00 + v is i^l Q a_j with Q one of 16 Pauli classes modulo the stabilizer of a_j
(the qubit slice structure lemma of slice_cover.py and slice_lift.py); at 11 a
term present at 01 and 10 carries the product class. Each of the other three
points is a finite equation solved by meet in the middle on two random
functionals modulo P = 2013265921, every hit re-decided in Gaussian-rational
arithmetic, the points combined under the flat rule, and every surviving
candidate rebuilt as four six-qubit vectors that must be distinct stabilizer
states combining exactly to c_7. Nothing survives, so chi(cat_7) >= 5.

Symmetry: permutations of the four unsliced qubits, R (x) R on a pair of them,
and the parity swap of cat_7 fix every slice of c_7 with one common scalar
(asserted), so one base per orbit suffices (100 pivot orbits of the 36,720
four-qubit states; 16,705 covers from 2,292,776 pivot pairs). Exactness: the census quotients are Gaussian integers with
|q1| <= 2, |q2| <= 8, |q3| <= 128 and minors below P (cert_cat_m7_lift.py,
assert_norms, one level further), base coefficients are ratios of minors of
unit-entry matrices, so every identity over Q(i) survives reduction mod P and
the modular screens have no false negatives.

Census functionals. The random functionals of the census are evaluated with
every product reduced modulo P before the sum (census4_run.dotp); the form
(Q2 @ f) % P wraps int64 on 16-term rows (16 P^2 > 2^63), which made an
earlier census list only a fifth of the covers (found by the referee of the
research record, 2026-10-17). research/cat7_rank4/census_control.py plants
random four-term targets through the census's TARGET path: the corrected
functional lists every planted 4-set, the wrapped one none of ten.

Candidates whose rebuilt terms have a zero coefficient or coincide are not
discarded: their surviving terms are checked exactly, and a decomposition of
c_7 with at most three terms would fail the run (it would contradict the
board's chi(cat_7) >= 4, cert_cat_m7_lift.py).

What this script re-runs: the manifest of research/cat7_rank4/batch_manifest.json
is checked (every stored census batch present with its hash), RECHECK work
units (pivot orbit, partner stride class) chosen from the printed seed are
re-enumerated from scratch and their
cover lists compared with the stored ones, the premises and the rank-3 list
are recomputed, stages beta, B and C run in full, and stage A runs over every
stored cover. The completeness of the census rests on the stored outputs and
on the committed enumerator (research/cat7_rank4/census4_run.py), which
regenerates any batch.

Printed claims: CERTIFIED chi(cat_7) >= 5
                CERTIFIED chi(cat_8) >= 5 (projection monotonicity)
"""
import hashlib
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "cat7_rank4"))
os.environ.setdefault("STABRANK_PUB", ROOT)
import cert_cat_m7_lift as L            # noqa: E402
import cert_cat_m5_rank2 as r2          # noqa: E402
import cat_slice2 as C                  # noqa: E402
import matcher as Mt                    # noqa: E402
import census4_run as CR                # noqa: E402

MANIFEST = os.path.join(ROOT, "research", "cat7_rank4", "batch_manifest.json")
RECHECK, SEED = 2, 20261017
CLAIM = "CERTIFIED chi(cat_7) >= 5"
CLAIM8 = "CERTIFIED chi(cat_8) >= 5"


def fail(msg):
    print(f"FAILED: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def check_parity_swap():
    ev, od = L.parity_swap_maps(4)
    sl = C.slices(7)
    nus = []
    for x, s in enumerate(sl):
        mp = od if bin(x).count("1") % 2 else ev
        d = (s[mp] - s) % 4
        if len(set(d.tolist())) != 1:
            fail("parity swap does not fix a slice up to scalar")
        nus.append(int(d[0]))
    if len(set(nus)) != 1:
        fail("parity swap fixes the slices with different scalars")


def load_manifest():
    """the stored census: one batch per (pivot orbit, partner stride class)."""
    man = json.load(open(MANIFEST))
    covers, per_unit = [], {}
    for b in man["batches"]:
        path = os.path.join(ROOT, b["output"])
        txt = open(path).read()
        if hashlib.sha256(txt.encode()).hexdigest() != b["sha256"]:
            fail(f"stored batch {b['id']} does not hash as recorded")
        body = json.loads(txt)
        prm = b["params"]
        if (body["pivot"], body["stride"], body["offset"]) != (prm["pivot"], prm["stride"], prm["offset"]) \
                or body["n"] != len(body["covers"]) or body["n"] != prm["covers"]:
            fail(f"stored batch {b['id']} disagrees with its manifest entry")
        per_unit[(body["pivot"], body["offset"])] = body["covers"]
        covers += body["covers"]
    stride = man["stride"]
    return man, covers, per_unit, stride


def main():
    t0 = time.time()
    print(f"seed: {SEED}")
    L.check_slices(6)
    r2.check_slice_identity(8)
    E = L.stabilizer_exponents(4)
    M = L.to_modp(E)
    gens = C.base_symmetry(4)
    C.check_symmetry_slices(gens, 7)
    check_parity_swap()
    ev, od = L.parity_swap_maps(4)
    roots = L.orbit_reps(E, gens + [("perm", ev, od)], 4)
    pivots = [int(p) for p in np.unique(roots)]
    sl = C.slices(7)
    if not np.array_equal(sl[0], L.c_exponents(5)):
        fail("slice 00 of c_7 is not c_5")
    par = np.array([bin(y).count("1") & 1 for y in range(16)])
    if not (np.array_equal(sl[3], (L.c_exponents(5) + 1) % 4) and np.array_equal(sl[1], sl[2])
            and np.array_equal(sl[1], (L.c_exponents(5) + 1 - par) % 4)):
        fail("slices of c_7 are not the stated Clifford images of c_5")
    print(f"c_7 sliced along two qubits: 36720 base states, {len(pivots)} pivot orbits ({time.time()-t0:.0f}s)", flush=True)
    # the stored census
    man, covers, per_unit, stride = load_manifest()
    if sorted({p for p, _ in per_unit}) != pivots or len(per_unit) != len(pivots) * stride:
        fail("the manifest's units are not the orbit representatives times the stride classes")
    print(f"stored census: {len(covers)} full 4-covers of c_5 in {len(per_unit)} batches "
          f"({len(pivots)} pivot orbits x {stride} partner classes), hashes verified", flush=True)
    # re-run RECHECK units from scratch
    rng = np.random.default_rng(SEED)
    keys = sorted(per_unit)
    chosen = sorted(keys[i] for i in rng.choice(len(keys), RECHECK, replace=False))
    for p, r in chosen:
        t1 = time.time()
        res = CR.census_pivots(("c5", True, [p], 0, None, (stride, r)))
        got = sorted([list(q), d] for q, d in res["covers"])
        if got != sorted(per_unit[(p, r)]):
            fail(f"re-enumerated unit (pivot {p}, class {r}) differs from the stored batch")
        print(f"re-enumerated pivot {p}, partner class {r} of {stride}: {res['pairs']} pairs, {len(got)} covers, "
              f"matches the stored batch ({time.time()-t1:.0f}s)", flush=True)
    # premises
    r2.exclude_rank2_cat5(verbose=False)
    print("premise: chi(c_5) >= 3 (exact rank-2 exclusion), so the base point sees 3 or 4 terms", flush=True)
    decs = L.decompositions(5, 3, E, M, roots, log_every=0)
    decs = [(list(c), [[str(x[0]), str(x[1])] for x in d]) for c, d in decs]
    if not decs:
        fail("no rank-3 decomposition of c_5 found, but chi(cat_5) = 3")
    print(f"rank-3 decompositions of c_5, one per orbit: {len(decs)}", flush=True)
    slices = Mt.target_slices_c7()
    target64 = []
    for x in range(4):
        target64 += list(slices[x][1])
    hits = []
    dkeys = Mt.dict_hash_table(E)
    hb = Mt.stage_beta(E, decs, slices, dkeys, log=False)
    print(f"stage beta: candidates {len(hb)} ({time.time()-t0:.0f}s)", flush=True)
    hits += hb
    basesB = []
    for tri, dstr in decs:
        f = [(Fraction(a), Fraction(b)) for a, b in dstr]
        cols = [C.exact_row(E[i]) for i in tri]
        R = M.copy(); piv = []
        for r in range(3):
            q = M[tri[r]].copy()
            for (pc, pr) in piv:
                q = (q - q[pc] * pr) % L.P
            k = int(np.flatnonzero(q)[0]); q = q * pow(int(q[k]), L.P - 2, L.P) % L.P
            piv.append((k, q)); R = (R - R[:, k:k + 1] * q[None, :]) % L.P
        for s in np.flatnonzero(~(R != 0).any(axis=1)):
            if int(s) in tri:
                continue
            kap = L.solve_exact(cols, C.exact_row(E[int(s)]))
            if kap is None:
                fail("a mod-P span member is not in the exact span")
            basesB.append((tri, f, int(s), kap))
    hB = Mt.stage_dependent(E, basesB, slices, "B", log=False) if basesB else []
    print(f"stage B: dependent bases {len(basesB)}, candidates {len(hB)} ({time.time()-t0:.0f}s)", flush=True)
    hits += hB
    basesC = [(tri, [(Fraction(a), Fraction(b)) for a, b in dstr], j) for tri, dstr in decs for j in range(3)]
    hC = Mt.stage_dependent(E, basesC, slices, "C", log=False)
    print(f"stage C: repeated-state bases {len(basesC)}, candidates {len(hC)} ({time.time()-t0:.0f}s)", flush=True)
    hits += hC
    hA = Mt.stage_A(E, covers, slices, log_every=0)
    print(f"stage A: {len(covers)} independent bases, candidates {len(hA)} ({time.time()-t0:.0f}s)", flush=True)
    hits += hA
    reasons, confirmed, short = {}, 0, 0
    for h in hits:
        ok, info = Mt.confirm_hit(E, h, slices, target64)
        if ok and len(info[0]) < 4:
            short += 1            # a decomposition with at most three terms: against chi(c_7) >= 4
        elif ok:
            confirmed += 1
        else:
            reasons[info] = reasons.get(info, 0) + 1
    print(f"candidates {len(hits)}, rejected exactly {reasons}, rank-4 decompositions {confirmed}, "
          f"decompositions with fewer than four terms {short} ({time.time()-t0:.0f}s)", flush=True)
    if short:
        fail("a candidate is a decomposition of c_7 with at most three stabilizer terms (chi(cat_7) >= 4 is on the board)")
    if confirmed:
        print("a rank-4 decomposition of c_7 exists: chi(cat_7) = 4")
        return 2
    print(CLAIM)
    print("(I (x) <0|) cat_8 = 2^(-1/2) cat_7 exactly; projection monotonicity")
    print(CLAIM8)
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
