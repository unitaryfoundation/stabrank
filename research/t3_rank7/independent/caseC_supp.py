"""Supplementary case-C scan: the pivot triples the scan of 2026-09-22 to 26 skipped.

cost_c.kok_of excluded, for every pair (i, j) with nontrivial Stab(i, j), the third
pivots that are the least element of a free Stab(i, j)-orbit (kok_fix.py has the
defect and the correction).  This runner scans exactly those triples: for each of
the affected pairs the kernel runs with the mask of missing third pivots, and every
emitted class set is decided exactly as in caseC.py (same kernel, same decision,
same record format).  Resumable: finished tasks are appended to
logs/caseC_supp_progress.jsonl; the union of that file with the original progress
file covers every canonical triple (i, j, k) once, which check_caseC.py verifies.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import caseC
from setup_m3 import load
import kok_fix


def build_tasks(missing):
    keys = sorted(missing)
    return [(t, i, j, 0.0, 1.0) for t, (i, j) in enumerate(keys)]


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    d = load(3); N = d["N"]
    t0 = time.time()
    fx = kok_fix.load_all(log=lambda m: print(m, flush=True))
    missing = fx["missing"]
    pred = fx["steps_missing"]
    tasks = build_tasks(missing)
    prog = os.path.join(HERE, "logs", "caseC_supp_progress.jsonl")
    done = {}
    if os.path.exists(prog):
        for line in open(prog):
            rec = json.loads(line); done[rec["tid"]] = rec
    todo = [t for t in tasks if t[0] not in done]
    rng = np.random.default_rng(0); rng.shuffle(todo)
    print(f"{len(tasks)} supplementary tasks ({len(done)} done, {len(todo)} to do) over the pairs with missing third "
          f"pivots; predicted inner steps {pred:.4e}; setup {time.time()-t0:.0f}s", flush=True)
    t0 = time.time(); nsteps = sum(r["steps"] for r in done.values()); ncl = sum(r["classes"] for r in done.values())
    ncand = sum(len(r["cands"]) for r in done.values())
    with ProcessPoolExecutor(max_workers=nw, initializer=caseC._init, initargs=(dict(d), missing)) as ex, open(prog, "a") as fp:
        for n, rec in enumerate(ex.map(caseC.task, todo, chunksize=1)):
            fp.write(json.dumps(rec) + "\n"); fp.flush()
            nsteps += rec["steps"]; ncl += rec["classes"]; ncand += len(rec["cands"])
            if rec["cands"]:
                print(f"  CANDIDATE class sets with rank(S) = rank(S + V_3): {rec['cands']}", flush=True)
            if n % 100 == 0 or n == len(todo) - 1:
                el = time.time() - t0; frac = nsteps / pred
                print(f"[{el:.0f}s] task {n+1}/{len(todo)}: steps {nsteps:.4e} ({100*frac:.2f}%), classes {ncl}, "
                      f"candidates {ncand}; ETA {el*(1-frac)/max(frac,1e-9)/3600:.2f} h", flush=True)
    recs = [json.loads(l) for l in open(prog)]
    tot = sum(r["steps"] for r in recs); ncl = sum(r["classes"] for r in recs)
    cands = [c for r in recs for c in r["cands"]]
    rhist = {}; sizes = {}
    for r in recs:
        for k, v in r["rhist"].items(): rhist[k] = rhist.get(k, 0) + v
        for k, v in r["sizes"].items(): sizes[k] = sizes.get(k, 0) + v
    print(f"total inner steps {tot} (predicted {pred}, the difference being the third pivots on the line through "
          f"the two pivots, which the kernel does not count); classes {ncl}")
    print(f"class-set size histogram: {dict(sorted((int(k), v) for k, v in sizes.items()))}")
    print(f"rank pair histogram (rank S, rank S + V_3): {dict(sorted(rhist.items()))}")
    print(f"candidates (rank equality): {len(cands)}")
    json.dump(dict(steps=tot, predicted=pred, classes=ncl, sizes=sizes, rhist=rhist, cands=cands, tasks=len(tasks),
                   wall=time.time() - t0), open(os.path.join(HERE, "logs", "caseC_supp.json"), "w"))
    if not cands:
        print("CASE C SUPPLEMENT EXCLUDED: no class set of the skipped pivot triples has V_3 in its span")
    else:
        best = min(cands, key=lambda c: c[1][0])
        print(f"class sets containing V_3 exist; minimal rank {best[1][0]}: {best[0]} (examine seven-subsets)")


if __name__ == "__main__":
    main()
