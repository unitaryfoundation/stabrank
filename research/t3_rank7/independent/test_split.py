"""Check that caseC.task with a tiny output buffer (forcing splits) returns the same class
sets, step count, and rank histogram as with the full buffer, on pair (reps[0], 14)."""
import os, sys, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import caseC
from setup_m3 import load
d = load(3); N = d["N"]
z = np.load(os.path.join(HERE, "logs", "costC.npz"))
koks = pickle.load(open(os.path.join(HERE, "logs", "costC_kok.pkl"), "rb"))
caseC._init(dict(d), koks)
i = int(z["reps"][0]); j = int(np.flatnonzero(z["jokm"][0])[min(5, int(np.sum(z["jokm"][0])) - 1)])
args = (0, i, j, 0.0, 1.0)
t0 = time.time(); full = caseC.task(args); t1 = time.time()
split = caseC.task(args, maxout=1500); t2 = time.time()
key = lambda r: (r["steps"], r["classes"], r["sizes"], r["rhist"], sorted(map(tuple, (c[0] for c in r["cands"]))))
print(f"pair ({i}, {j}): full {t1-t0:.1f}s, {full['classes']} classes, {full['steps']} steps, splits {full['splits']}")
print(f"           split {t2-t1:.1f}s, {split['classes']} classes, {split['steps']} steps, splits {split['splits']}")
print("IDENTICAL" if key(full) == key(split) else "MISMATCH")
