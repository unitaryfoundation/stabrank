"""Case A of the rank-7 exclusion, self-contained: regenerate the coplanar six-set
classes with the certificate's two-pivot kernel (need = 4, the rank-6 scan of
cert_t3m3_rank7.py) and apply caseA.work to them.

caseA.py reads the class list from a stored dump of an earlier rank-6 scan
(stabrank-work/logs/T3m3_rank6_sym.json, 538,514 classes, not tracked).  This
script needs no input file: it runs kernel2 over the 45 orbit representatives with
the Stab(i)-minimal second pivots and need = 4 (every class is two pivots plus at
least four members zero or parallel modulo the pivots), which is complete for
canonically labelled six-sets with projected images in a 3-dim space, then decides
the exact rank pair of every six-subset (pivots plus four members).  Prints the same
verdict line as caseA.py.  About 30 s wall on 8 workers.
"""
import itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cert7 as C
from setup_m3 import load
import caseA

MAXOUT = 400000


def scan(args):
    i, jlist = args
    PD, isfree = caseA._G["PD"], caseA._G["isfree"]
    N = PD.shape[0]
    jok = np.zeros(N, dtype=np.int8); jok[np.asarray(jlist, dtype=np.int64)] = 1
    c, mb, par = C.kernel2(PD, C.INV, int(i), jok, isfree, 0, 4, C.ELL, MAXOUT)
    assert c.shape[0] < MAXOUT and mb.shape[0] < MAXOUT * 64, "buffer overflow"
    assert len(par) == 0
    memb = {}
    for cid, k in mb:
        memb.setdefault(int(cid), []).append(int(k))
    return [((int(i), int(c[cid, 0])), int(c[cid, 1]), int(c[cid, 2]), memb.get(cid, [])) for cid in range(c.shape[0])]


def init(d):
    caseA._init(d["E"], d["T"])
    caseA._G["PD"], caseA._G["isfree"] = d["PD"], d["isfree"]


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    d = load(3); N = d["N"]
    t0 = time.time()
    tasks = []
    for t, i in enumerate(d["reps"]):
        js = np.flatnonzero(d["jokm"][t]); js = js[js != int(i)]
        for a in range(0, len(js), 40):
            tasks.append((int(i), js[a:a + 40].tolist()))
    classes = []
    dd = dict(d)
    with ProcessPoolExecutor(max_workers=nw, initializer=init, initargs=(dd,)) as ex:
        for out in ex.map(scan, tasks):
            classes.extend(out)
        print(f"[{time.time()-t0:.0f}s] {len(classes)} need-4 classes from {len(tasks)} pivot tasks", flush=True)
        chunks = [classes[a:a + 2000] for a in range(0, len(classes), 2000)]
        tot = [0, 0]; found = []; hist = {}
        for nF, ndis, f, h in ex.map(caseA.work, chunks):
            tot[0] += nF; tot[1] += ndis; found.extend(f)
            for kk, v in h.items():
                hist[kk] = hist.get(kk, 0) + v
    print("rank-pair histogram of six-subsets:", dict(sorted(hist.items())))
    print(f"[{time.time()-t0:.0f}s] six-subsets {tot[0]}, with exact rank pair (6,7): {tot[1]}, rank-7 decompositions found: {len(found)}")
    json.dump({"classes": len(classes), "six_subsets": tot[0], "dishonest": tot[1], "found": found,
               "hist": {str(k): v for k, v in hist.items()}, "wall": time.time() - t0},
              open(os.path.join(HERE, "logs", "caseA_scan.json"), "w"))
    if found:
        print("FOUND", found[:5])
    else:
        print("CASE A EXCLUDED: no seven-state span containing V_3 has six images in a 3-dim space mod ELL")


if __name__ == "__main__":
    main()
