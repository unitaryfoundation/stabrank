"""Control for case B stage 1 on the eight-term witness: find its five-subsets with exact
image rank 3 (rank(F5 + T) = 6, rank(F5) = 5) and check that the need-3 kernel plus the
independence filter, run with each member of such a subset as first pivot and every j
allowed, returns that five-subset."""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert7 as C
from setup_m3 import load
from decide import Decider, ELL2
import caseB

d = load(3)
N, E, T, PD, isfree = d["N"], d["E"], d["T"], d["PD"], d["isfree"]
dec = Decider(E, T)
w = json.load(open(os.path.join(HERE, "..", "T3-m3-upper-8.json")))
# locate witness states in the dictionary by exponent key
terms = list(C.enumerate_terms(3))
index = {k: i for i, k in enumerate(C.row_keys(E))}
def term_key(t):
    return C.row_keys(C.term_exponents(t["k"], t["x0"], t["W"], t["Q"], t["l"], 3)[None, :])[0]
wit = []
for t in w["witness"]["terms"] if "witness" in w else w["terms"]:
    tt = {"k": t["k"], "x0": t["x0"], "W": t["W"], "Q": t["Q"], "l": t["l"]}
    wit.append(index[term_key(tt)])
print("witness indices:", wit)
ok, r1, r2 = dec.contains(wit)
print("witness: contains V_3:", ok, "rank", r1, r2)
assert ok and r1 == 8
good = []
for F5 in itertools.combinations(wit, 5):
    a, b = dec.ranks(list(F5))
    if a == 5 and b == 6:
        good.append(F5)
print(f"five-subsets of the witness with image rank 3 over Q(w3): {len(good)}")
assert good, "control not applicable: no coplanar five-subset in the witness"
F5 = good[0]
print("using", F5)
caseB._init(dict(PD=PD, isfree=isfree, E=E, T=T, jok_of={i: np.ones(N, dtype=np.int8) for i in F5}))
hit = False
for i in F5:
    nc, n5, out = caseB.stage1((i, 0, N))
    surv = {tuple(sorted(F)) for F, r in out}
    print(f"pivot {i}: {nc} classes, {n5} five-subsets, {len(out)} survivors; contains control set: {tuple(sorted(F5)) in surv}")
    hit |= tuple(sorted(F5)) in surv
assert hit
print("CONTROL B OK: stage 1 recovers the witness's coplanar five-subset")
