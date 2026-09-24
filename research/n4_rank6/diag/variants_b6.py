"""B6 filter variants on real orbit representatives: m = 2 against m = 3
partitions, the kernel's raw feature-zero counts, and the time split."""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research", "n4_rank6"))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
from cover_census import CoverEnumerator3
from matcher import Matcher, psi_target, E1, E2, add
from degenerate6 import decode
from filters6 import Filters
os.nice(19)
E = CoverEnumerator3("N", 2)
Mt = Matcher(E.D, 2, E.F1, E.F2)
target = psi_target("N", 2, Mt.F1, Mt.F2)
reps = np.load(sys.argv[1])
B6 = decode(reps["B6_orbit_codes"], 6, E.N)
kB = reps["B6_orbit_kappa"]
idx = np.flatnonzero(kB == 1)
pick = [tuple(int(u) for u in B6[idx[t]]) for t in np.linspace(0, len(idx) - 1, 25).astype(int)]
x0 = (2, 2)
for m in ((3,), (2,), (2, 3), (9,)):
    Fl = Filters(Mt, m_choices=m)
    # warm caches
    Fl.b6_slice(pick[0], x0, target)
    secs, dsec, plan, surv, raw, sols, slack, parts = [], [], [], [], [], [], [], []
    for cover in pick:
        t = time.time()
        out, st = Fl.b6_slice(cover, x0, target)
        secs.append(time.time() - t)
        dsec.append(st["seconds_dense"]); plan.append(st["seconds_plan"]); surv.append(st["survivors5"])
        raw.append(st["dense_raw"]); sols.append(len(out)); slack.append(st["slack"]); parts.append(st["parts"])
    print(f"m={m}: total {np.mean(secs):.3f} s (plan {np.mean(plan):.3f}, dense {np.mean(dsec):.3f}, max {np.max(secs):.2f}); "
          f"parts {sorted(set(parts))}, slack {sorted(set(slack))}; raw feature zeros mean {np.mean(raw):.0f} max {np.max(raw)}; "
          f"survivors mean {np.mean(surv):.1f} max {np.max(surv)}; items with solutions {sum(s > 0 for s in sols)}")
# second slice too, for the items with first-slice solutions
Fl = Filters(Mt, m_choices=(3,))
for cover in pick:
    out, st = Fl.b6_slice(cover, x0, target)
    if out:
        t = time.time()
        out2, st2 = Fl.b6_slice(cover, x0, target, e=E2)
        print(f"  {cover}: first slice {len(out)} solutions, second slice {len(out2)} in {time.time() - t:.2f} s")
