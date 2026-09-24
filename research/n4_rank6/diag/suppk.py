"""Support of the dependency vector over the B6 kappa = 1 orbit reps, and
raw feature zeros against pass1 on a sample."""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research", "n4_rank6"))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
from cover_census import CoverEnumerator3
from matcher import Matcher, psi_target, E1, add, P1, P2, _split_sides
from degenerate6 import decode
from filters6 import Filters, ALL
os.nice(19)
E = CoverEnumerator3("N", 2)
Mt = Matcher(E.D, 2, E.F1, E.F2)
target = psi_target("N", 2, Mt.F1, Mt.F2)
reps = np.load(sys.argv[1])
B6 = decode(reps["B6_orbit_codes"], 6, E.N)
kB = reps["B6_orbit_kappa"]
idx = np.flatnonzero(kB == 1)
t = time.time()
supp = np.zeros(len(idx), dtype=int)
for n, i in enumerate(idx):
    A = E.C[:, B6[i]]
    _, s, vh = np.linalg.svd(A)
    K = vh[-1].conj()
    supp[n] = int(np.sum(np.abs(K) > 1e-8))
print("support of K over", len(idx), "kappa=1 orbit reps:", dict(zip(*np.unique(supp, return_counts=True))), f"[{time.time()-t:.0f}s]")
# raw against pass1 on a sample, by support
Fl = Filters(Mt)
x0 = (2, 2)
for s_val in sorted(set(supp)):
    sel = np.flatnonzero(supp == s_val)
    pick = [tuple(int(u) for u in B6[idx[j]]) for j in sel[np.linspace(0, len(sel) - 1, min(8, len(sel))).astype(int)]]
    rows = []
    for cover in pick:
        distinct = sorted(cover)
        fam = Fl.family(distinct, target, x0)
        j, parts, slack = Fl.plan_b6(distinct, fam)
        d01, K1 = fam.parts[0][0], fam.parts[0][1][:, 0]
        d02, K2 = fam.parts[1][0], fam.parts[1][1][:, 0]
        ords = [i for i in range(6) if i != j]
        tb = Fl.basis(distinct[j])
        co = [tb.coords(Mt.options(distinct[i]), distinct[i]) for i in ords]
        r1, r2, rC = target.rhs(add(x0, E1))
        R1, R2 = tb.vec(r1, r2)
        sides = _split_sides([28] * 5)
        raw = p1 = surv = 0
        t0 = time.time()
        for P in parts:
            G = [c for c in ALL if c not in P]
            o1 = [np.ascontiguousarray(c[0][:, G]) for c in co]
            o2 = [np.ascontiguousarray(c[1][:, G]) for c in co]
            combos, nraw, np1 = Fl.dense(o1, o2, np.ascontiguousarray(d01[ords] % P1), np.ascontiguousarray(K1[ords] % P1).reshape(5, 1),
                np.ascontiguousarray(d02[ords] % P2), np.ascontiguousarray(K2[ords] % P2).reshape(5, 1),
                np.ascontiguousarray(R1[G] % P1), np.ascontiguousarray(R2[G] % P2),
                [int(i) for i in sides[0]], [int(i) for i in sides[1]], 2_000_000, 5)
            raw += int(nraw); p1 += int(np1); surv += len(np.asarray(combos))
        rows.append((raw, p1, surv, time.time() - t0))
    r = np.array(rows)
    print(f"|supp K| = {s_val}: {len(sel)} reps; sample of {len(pick)}: raw mean {r[:,0].mean():.0f}, pass1 mean {r[:,1].mean():.1f}, "
          f"survivors mean {r[:,2].mean():.1f}, dense seconds mean {r[:,3].mean():.3f} (min {r[:,3].min():.3f}, max {r[:,3].max():.3f})")
