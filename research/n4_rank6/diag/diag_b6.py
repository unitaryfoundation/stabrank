"""Diagnose the B6 filter's structural survivors: supports of the option
vectors in the fresh term's translate basis, and survivors for several
coordinate-group choices."""
import itertools, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research", "n4_rank6"))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
from cover_census import CoverEnumerator3
from matcher import Matcher, psi_target, E1, add, P1, P2, _split_sides
from degenerate6 import decode
from filters6 import Filters
os.nice(19)
E = CoverEnumerator3("N", 2)
Mt = Matcher(E.D, 2, E.F1, E.F2)
Fl = Filters(Mt)
target = psi_target("N", 2, Mt.F1, Mt.F2)
reps = np.load(sys.argv[1])
B6 = decode(reps["B6_orbit_codes"], 6, E.N)
kB = reps["B6_orbit_kappa"]
idx = np.flatnonzero(kB == 1)
pick = [tuple(int(u) for u in B6[i]) for i in np.linspace(0, len(idx) - 1, 12).astype(int)]
x0 = (2, 2)
ALL = list(range(9))
def partitions3():
    out = []
    rest = ALL
    for a in itertools.combinations(rest[1:], 2):
        A = (0,) + a
        r2 = [c for c in rest if c not in A]
        for b in itertools.combinations(r2[1:], 2):
            B = (r2[0],) + b
            C = tuple(c for c in r2 if c not in B)
            out.append((A, B, C))
    return out
P3 = partitions3()
for cover in pick:
    distinct = sorted(cover)
    fam = Fl.family(distinct, target, x0)
    KC = fam.parts[2][1][:, 0]
    r1, r2, rC = target.rhs(add(x0, E1))
    line = []
    for j in range(6):
        if abs(KC[j]) < 1e-9:
            continue
        tb = Fl.basis(distinct[j])
        supports = []
        for i in range(6):
            if i == j:
                continue
            o = Mt.options(distinct[i])
            P = (o.m1[:27] @ tb.inv1.T) % P1          # (27, 9)
            supports.append([frozenset(np.flatnonzero(row)) for row in P])
        sizes = sorted(set(len(s) for sup in supports for s in sup))
        # slack for the 2-group choice and the best 3-partition: options with support inside a part
        def slack(parts):
            return sum(sum(1 for s in sup if any(s <= set(p) for p in parts)) for sup in supports)
        s2 = slack([(0, 1, 2, 3), (4, 5, 6, 7, 8)])
        s3 = min(slack(p) for p in P3)
        s9 = slack([(k,) for k in ALL])
        line.append(f"j={distinct[j]}: sizes {sizes}, slack2 {s2}, slack3 {s3}, slack9 {s9}")
    print(cover, "kappa", fam.kappa)
    for l in line:
        print("   ", l)
