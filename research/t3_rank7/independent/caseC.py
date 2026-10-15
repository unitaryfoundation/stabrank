"""Case C of the rank-7 exclusion: seven states whose images mod ELL are in general
position (no five in a 3-dim space), scanned with three pivots under the exact symmetry
group G' of V_3 (sym_exact.py, cost_c.py).

For each canonical pivot triple (i, j, k) (see cost_c.py) the kernel k3.kernel3 emits the
class sets {i, j, k} + (members parallel modulo span(q_i, q_j, q_k), index > k) + (members
zero modulo that span).  A genuine seven-set F in general position with canonical labelling
(i, j, k, ...) is contained in the class set S of its triple, and then V_3 lies in span(F)
and hence in span(S): rank(S) = rank(S + V_3) over Q(w3).  Every class set is decided
exactly (mod 2^31 - 1 for ranks <= 8, then a second prime, then Q(w3)); class sets that
fail the test are rejected, those that pass are recorded for exact examination of their
seven-subsets.  Resumable: finished tasks are appended to logs/caseC_progress.jsonl.
"""
import itertools, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cert7 as C, k3
from setup_m3 import load
from decide import Decider, ELL2
from caseB import ranks_batch

MAXOUT = 400000
WORK_PER_TASK = 6.0e9
_G = {}


def _init(d, koks):
    _G.update(d); _G["koks"] = koks
    _G["dec"] = Decider(d["E"], d["T"])
    _G["E2"], _G["T2"] = _G["dec"].E2, _G["dec"].T2
    _G["E2T"] = np.vstack([_G["E2"], _G["T2"]])
    _G["ones"] = np.ones(d["N"], dtype=np.int8)


def decide_sets(sets):
    """Exact rank pairs (rank S, rank S + V_3) for a list of class sets (lists of states)."""
    E2, E2T, N, dec = _G["E2"], _G["E2T"], _G["N"], _G["dec"]
    out = [None] * len(sets)
    small = [(n, S) for n, S in enumerate(sets) if len(S) <= 8]
    big = [(n, S) for n, S in enumerate(sets) if len(S) > 8]
    if small:
        L = max(len(S) for _, S in small)
        A = np.array([S + [S[0]] * (L - len(S)) for _, S in small], dtype=np.int64)
        r1 = ranks_batch(E2, A, ELL2)
        r2 = ranks_batch(E2T, np.hstack([A, N + np.arange(3)[None, :].repeat(len(A), 0)]), ELL2)
        for (n, S), a, b in zip(small, r1, r2):
            out[n] = (int(a), int(b))         # exact: values <= 8
    for n, S in big:
        r1, r2 = dec.ranks(S)
        out[n] = (int(r1), int(r2))
    return out


def scan(PD, N, i, j, jok, kok, lo, hi, maxout=MAXOUT):
    """kernel3 over the third-pivot fraction [lo, hi); on a buffer overflow the k range is
    halved and both halves scanned, since the kernel's output for a range is the union of
    its outputs per third pivot k and the step count is additive.  Returns the list of
    (cand rows, members) pieces, the inner-step count, and the number of splits."""
    c, mb, ns = k3.kernel3(PD, C.INV, i, jok, kok, 3, maxout, lo, hi)
    if c.shape[0] < maxout and mb.shape[0] < maxout * 16:
        return [(c, mb)], int(ns), 0
    k_of = lambda f: j + 1 + int(f * (N - j - 1))     # the kernel's klo / khi
    klo, khi = k_of(lo), k_of(hi)
    assert khi - klo >= 2, "buffer overflow on a single third pivot"
    kmid = (klo + khi) // 2
    mid = (kmid + 0.5 - j - 1) / (N - j - 1)
    assert k_of(mid) == kmid
    a, na, sa = scan(PD, N, i, j, jok, kok, lo, mid, maxout)
    b, nb, sb = scan(PD, N, i, j, jok, kok, mid, hi, maxout)
    return a + b, na + nb, sa + sb + 1


def task(args, maxout=MAXOUT):
    tid, i, j, lo, hi = args
    PD, N = _G["PD"], _G["N"]
    jok = np.zeros(N, dtype=np.int8); jok[j] = 1
    kok = _G["koks"].get((i, j), _G["ones"])
    pieces, ns, nsplit = scan(PD, N, i, j, jok, kok, lo, hi, maxout)
    sets = []
    for c, mb in pieces:
        memb = {}
        for cid, s, tag in mb:
            memb.setdefault(int(cid), []).append(int(s))
        sets += [[i, j, int(c[cid, 1])] + memb.get(cid, []) for cid in range(c.shape[0])]
    sizes = {}
    for S in sets:
        sizes[len(S)] = sizes.get(len(S), 0) + 1
    ranks = decide_sets(sets) if sets else []
    cands = [(S, r) for S, r in zip(sets, ranks) if r[0] == r[1]]
    rhist = {}
    for r in ranks:
        rhist[f"{r[0]},{r[1]}"] = rhist.get(f"{r[0]},{r[1]}", 0) + 1
    return dict(tid=tid, i=i, j=j, lo=lo, hi=hi, steps=int(ns), classes=len(sets), sizes=sizes,
                rhist=rhist, cands=cands, how=dict(_G["dec"].how), splits=nsplit)


def build_tasks(reps, jokm, koks, N):
    tasks = []
    for r, i in enumerate(reps):
        i = int(i)
        for j in np.flatnonzero(jokm[r]):
            j = int(j)
            if (i, j) in koks:
                ks = np.flatnonzero(koks[(i, j)])
            else:
                ks = np.arange(j + 1, N)
            ks = ks[ks != i]
            w = float(np.sum(N - ks - 1))
            npieces = max(1, int(np.ceil(w / WORK_PER_TASK)))
            if npieces == 1:
                tasks.append((i, j, 0.0, 1.0)); continue
            # split the k range (j, N) into pieces of roughly equal work sum (N - k - 1)
            kk = np.arange(j + 1, N); cw = np.cumsum(N - kk - 1.0); cw /= cw[-1]
            cuts = [0.0] + [float((np.searchsorted(cw, q / npieces)) / (N - j - 1)) for q in range(1, npieces)] + [1.0]
            for a in range(npieces):
                if cuts[a + 1] > cuts[a]:
                    tasks.append((i, j, cuts[a], cuts[a + 1]))
    return [(t,) + tk for t, tk in enumerate(tasks)]


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    d = load(3); N = d["N"]
    z = np.load(os.path.join(HERE, "logs", "costC.npz"))
    reps, jokm, steps_pred = z["reps"], z["jokm"], int(z["steps"])
    koks = pickle.load(open(os.path.join(HERE, "logs", "costC_kok.pkl"), "rb"))
    tasks = build_tasks(reps, jokm, koks, N)
    prog = os.path.join(HERE, "logs", "caseC_progress.jsonl")
    done = {}
    if os.path.exists(prog):
        for line in open(prog):
            rec = json.loads(line); done[rec["tid"]] = rec
    todo = [t for t in tasks if t[0] not in done]
    rng = np.random.default_rng(0); rng.shuffle(todo)
    print(f"{len(tasks)} tasks ({len(done)} done, {len(todo)} to do), predicted inner steps {steps_pred:.4e}, "
          f"{len(koks)} pairs with a k-mask", flush=True)
    t0 = time.time(); nsteps = sum(r["steps"] for r in done.values()); ncl = sum(r["classes"] for r in done.values())
    ncand = sum(len(r["cands"]) for r in done.values())
    with ProcessPoolExecutor(max_workers=nw, initializer=_init, initargs=(dict(d), koks)) as ex, open(prog, "a") as fp:
        for n, rec in enumerate(ex.map(task, todo, chunksize=1)):
            fp.write(json.dumps(rec) + "\n"); fp.flush()
            nsteps += rec["steps"]; ncl += rec["classes"]; ncand += len(rec["cands"])
            if rec["cands"]:
                print(f"  CANDIDATE class sets with rank(S) = rank(S + V_3): {rec['cands']}", flush=True)
            if n % 10 == 0 or n == len(todo) - 1:
                el = time.time() - t0
                frac = nsteps / steps_pred
                print(f"[{el:.0f}s] task {n+1}/{len(todo)}: steps {nsteps:.4e} ({100*frac:.2f}%), classes {ncl}, "
                      f"candidates {ncand}; ETA {el*(1-frac)/max(frac,1e-9)/3600:.2f} h", flush=True)
    recs = [json.loads(l) for l in open(prog)]
    tot = sum(r["steps"] for r in recs); ncl = sum(r["classes"] for r in recs)
    cands = [c for r in recs for c in r["cands"]]
    rhist = {}; sizes = {}; how = {}
    for r in recs:
        for k, v in r["rhist"].items(): rhist[k] = rhist.get(k, 0) + v
        for k, v in r["sizes"].items(): sizes[k] = sizes.get(k, 0) + v
    print(f"total inner steps {tot} (predicted {steps_pred}; equal: {tot == steps_pred}); classes {ncl}")
    print(f"class-set size histogram: {dict(sorted((int(k), v) for k, v in sizes.items()))}")
    print(f"rank pair histogram (rank S, rank S + V_3): {dict(sorted(rhist.items()))}")
    print(f"candidates (rank equality): {len(cands)}")
    json.dump(dict(steps=tot, predicted=steps_pred, classes=ncl, sizes=sizes, rhist=rhist, cands=cands,
                   tasks=len(tasks), wall=time.time() - t0), open(os.path.join(HERE, "logs", "caseC.json"), "w"))
    if not cands:
        print("CASE C EXCLUDED: no class set of the three-pivot scan has V_3 in its span")
    else:
        best = min(cands, key=lambda c: c[1][0])
        print(f"class sets containing V_3 exist; minimal rank {best[1][0]}: {best[0]} (examine seven-subsets)")


if __name__ == "__main__":
    main()
