"""Completeness checks of the finished case-C scan (T3, m = 3, rank-7 exclusion).

Reads the per-task progress file written by caseC.py (default
logs/pod/caseC_progress.jsonl, the file pulled from the pod) and checks, against
data rebuilt here (dictionary, projection, symmetry masks, exact ranks):

  (a) coverage: the records are the task list of caseC.build_tasks, every task id
      exactly once, each with the task's pivot pair and full third-pivot range;
  (b) step accounting: for every task, the recorded inner steps plus the steps
      of the third pivots the kernel skips (states above j whose projected image
      lies on the line through the two pivots, which kernel3 does not count)
      equal the exact prediction of cost_c.py, and the totals agree;
  (c) decisions: every class set has rank(S) < rank(S + V_3), no candidates, the
      per-task rank histograms sum to the class counts, and which arithmetic
      level decided each set;
  (e) symmetry data: the 45 first pivots are the orbit minima of G', the second-
      pivot masks equal the certificate's (up to the diagonal j = i, which the
      certificate's mask never excluded and kernel2 skips), and every third-pivot
      mask is recomputed from the full Stab(i, j) (kok_fix.py), which exposes the
      defect of cost_c.kok_of and measures the triples the scan skipped;
  (g) the supplementary scan (caseC_supp.py) of the third pivots that
      cost_c.kok_of wrongly excluded (kok_fix.py): its records cover every pair
      with missing third pivots once, its step accounting closes against the
      corrected total, and its decisions are all strict;
  (f) the two geometric facts the canonical-augmentation argument of
      REPORT-rank7.md needs from the projected images: no state is parallel to a
      first pivot mod ell, and every state whose projected image lies on the
      line through a canonical pivot pair lies on that line over Q(w3) as well
      (exact rank), so the kernel's choice of third pivot is the canonical one.

Prints PASS or GAP per item and writes logs/check_caseC.json.
"""
import json, os, pickle, sys, time
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cert7 as C, k3, caseC
from setup_m3 import load
from caseB import ranks_batch
from decide import ELL2

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

PROG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "logs", "pod", "caseC_progress.jsonl")
OUT = os.path.join(HERE, "logs", "check_caseC.json")
res = {}
gaps = []


def verdict(name, ok, detail):
    res[name] = dict(ok=bool(ok), detail=detail)
    print(f"{'PASS' if ok else 'GAP '} ({name}): {detail}", flush=True)
    if not ok:
        gaps.append(name)


@njit(cache=True)
def online_above(Q1, inv, i, j, ell):
    """States k > j, k != i whose row of Q1 (images reduced modulo q_i) is a multiple of
    Q1[j], i.e. whose projected image lies in span(q_i, q_j)."""
    N, D1 = Q1.shape
    qj = Q1[j]
    b = 0
    while b < D1 and qj[b] == 0:
        b += 1
    out = np.empty(N, dtype=np.int64)
    n = 0
    if b == D1:
        return out[:0]
    ib = inv[qj[b]]
    for k in range(j + 1, N):
        if k == i:
            continue
        c = (Q1[k, b] * ib) % ell
        z = True
        for d in range(D1):
            if d != b and (Q1[k, d] - c * qj[d]) % ell != 0:
                z = False
                break
        if z:
            out[n] = k
            n += 1
    return out[:n]


# ------------------------------------------------------------------ records ----
recs = [json.loads(l) for l in open(PROG)]
log(f"{len(recs)} records read from {PROG}")

d = load(3); N = d["N"]; PD = d["PD"]
z = np.load(os.path.join(HERE, "logs", "costC.npz"))
reps, jokm, steps_pred = z["reps"], z["jokm"], int(z["steps"])
koks = pickle.load(open(os.path.join(HERE, "logs", "costC_kok.pkl"), "rb"))
tasks = caseC.build_tasks(reps, jokm, koks, N)
log(f"task list rebuilt: {len(tasks)} tasks, {len(koks)} pairs with a third-pivot mask, predicted steps {steps_pred}")

# (a) coverage
tids = np.array([r["tid"] for r in recs])
ok_ids = len(recs) == len(tasks) and np.array_equal(np.sort(tids), np.arange(len(tasks)))
bad = 0
for r in recs:
    t = tasks[r["tid"]]
    if (r["i"], r["j"], r["lo"], r["hi"]) != (t[1], t[2], t[3], t[4]):
        bad += 1
full = all(r["lo"] == 0.0 and r["hi"] == 1.0 for r in recs)
split_hist = {}
for r in recs:
    split_hist[r.get("splits", 0)] = split_hist.get(r.get("splits", 0), 0) + 1
verdict("a_coverage", ok_ids and bad == 0 and full,
        f"{len(recs)} records, task ids 0..{len(tasks)-1} each once: {ok_ids}; records whose (i, j, lo, hi) "
        f"differ from the task list: {bad}; every task covers its full third-pivot range: {full}; "
        f"buffer-overflow halvings per task: {dict(sorted(split_hist.items()))}")

# (b) step accounting, and (f) the on-line triples for the collision check
Q1_cache = {}
def Q1_of(i):
    if i not in Q1_cache:
        Q1_cache.clear()
        Q1_cache[i] = k3.reduce_one(PD, C.INV, int(i))
    return Q1_cache[i]

by_rep = {}
for r in recs:
    by_rep.setdefault(r["i"], []).append(r)
tot_rec = 0; tot_pred = 0; tot_def = 0; mism = []
online_triples = []          # (i, j, m) with the projected image of m on the line (i, j)
online_kok = 0
for i in sorted(by_rep):
    Q1 = Q1_of(i)
    for r in by_rep[i]:
        j = r["j"]
        kok = koks.get((i, j))
        if kok is not None:
            ks = np.flatnonzero(kok)
        else:
            ks = np.arange(j + 1, N)
        ks = ks[ks != i]
        pred = int(np.sum(N - ks - 1))
        onl = online_above(Q1, C.INV, i, j, C.ELL)
        for m in onl:
            online_triples.append((i, j, int(m)))
        if kok is not None:
            onl_allowed = onl[kok[onl] != 0]
        else:
            onl_allowed = onl
        deficit = int(np.sum(N - onl_allowed - 1))
        online_kok += len(onl_allowed)
        tot_rec += r["steps"]; tot_pred += pred; tot_def += deficit
        if r["steps"] + deficit != pred:
            mism.append((r["tid"], i, j, r["steps"], deficit, pred))
    log(f"rep {i}: {len(by_rep[i])} tasks done; on-line triples so far {len(online_triples)}; mismatches {len(mism)}")
verdict("b_steps", not mism and tot_pred == steps_pred and tot_rec + tot_def == steps_pred,
        f"recorded steps {tot_rec} + skipped on-line third pivots {tot_def} = {tot_rec + tot_def}; predicted "
        f"{steps_pred} (cost_c.py), {tot_pred} (recomputed here); tasks where the identity fails: {len(mism)}; "
        f"on-line third pivots skipped by the kernel: {online_kok} (allowed by the mask) of {len(online_triples)} on-line states above j")
if mism:
    print("  first mismatches:", mism[:5])

# (c) decisions
rhist = {}; sizes = {}; how = {}; ncl = 0; cands = 0; badsum = 0
for r in recs:
    s = 0
    for k, v in r["rhist"].items():
        rhist[k] = rhist.get(k, 0) + v; s += v
    for k, v in r["sizes"].items():
        sizes[int(k)] = sizes.get(int(k), 0) + v
    for k in r.get("how", {}):
        how[k] = how.get(k, 0) + 1          # the field is a per-worker cumulative counter; only its keys mean anything
    ncl += r["classes"]; cands += len(r["cands"])
    if s != r["classes"]:
        badsum += 1
pairs = {tuple(int(x) for x in k.split(",")): v for k, v in rhist.items()}
strict = all(a < b for a, b in pairs)
verdict("c_decisions", strict and cands == 0 and badsum == 0,
        f"class sets {ncl}; rank pairs {dict(sorted(pairs.items()))}; every pair has rank(S) < rank(S + V_3): {strict}; "
        f"candidates {cands}; tasks whose histogram does not sum to its class count: {badsum}; "
        f"arithmetic levels that decided a class set of more than 8 states (keys, over tasks): {sorted(how)}; size histogram {dict(sorted(sizes.items()))}")

# (e) symmetry data against the certificate's cache
zc = np.load(os.path.join(HERE, "logs", "setup_m3.npz"))
reps_c, jokm_c = zc["reps"], zc["jokm"]
same_reps = np.array_equal(reps, reps_c)
jok_c_nodiag = jokm_c.copy()
for t, i in enumerate(reps_c):
    jok_c_nodiag[t, int(i)] = 0
same_jok = np.array_equal(jokm.astype(np.int8), jok_c_nodiag.astype(np.int8))
npairs = int(jokm.sum()); npairs_c = int(jokm_c.sum())
# the third-pivot masks: recomputed from the full Stab(i, j) (kok_fix.py); the stored
# masks of cost_c.py were computed over the nontrivial elements only and are subsets
import kok_fix
fx = kok_fix.load_all(log=log)
G = fx["G"]
missing = fx["missing"]
n_missing_k = sum(int(m.sum()) for m in missing.values())
stored_correct = all(np.array_equal(fx["stored"][k], fx["correct"][k]) for k in fx["correct"])
verdict("e_symmetry", same_reps and same_jok and stored_correct,
        f"|G'| = {G.order}; first pivots equal the certificate's 45 orbit minima: {same_reps}; second-pivot masks equal "
        f"the certificate's after removing j = i: {same_jok} ({npairs} pairs against the certificate's {npairs_c}, "
        f"difference {npairs_c - npairs}); third-pivot masks of the {len(fx['correct'])} pairs with nontrivial "
        f"Stab(i, j) equal the Stab(i, j)-orbit minima: {stored_correct}; pairs with excluded canonical third pivots "
        f"{len(missing)}, excluded third pivots {n_missing_k}, inner steps not scanned {fx['steps_missing']} "
        f"(corrected scan total {fx['steps_correct_total']} against {steps_pred} as run)")

# (g) the supplementary scan of the excluded triples
SUPP = os.path.join(HERE, "logs", "caseC_supp_progress.jsonl")
supp_ok = False; supp_detail = "no supplementary progress file"
supp_rec = 0; supp_def = 0; supp_pairs = {}; supp_cands = 0; supp_ncl = 0
if os.path.exists(SUPP):
    srecs = [json.loads(l) for l in open(SUPP)]
    skeys = sorted(missing)
    stids = np.array([r["tid"] for r in srecs])
    ok_ids = len(srecs) == len(skeys) and np.array_equal(np.sort(stids), np.arange(len(skeys)))
    bad = 0; mism_s = []; nonstrict = 0
    for r in srecs:
        i, j = skeys[r["tid"]] if r["tid"] < len(skeys) else (None, None)
        if (r["i"], r["j"], r["lo"], r["hi"]) != (i, j, 0.0, 1.0):
            bad += 1; continue
        m = missing[(i, j)]
        pred = kok_fix.steps_of_pair(N, i, j, m)
        onl = online_above(Q1_of(i), C.INV, i, j, C.ELL)
        onl = onl[m[onl] != 0]
        deficit = int(np.sum(N - onl - 1))
        supp_rec += r["steps"]; supp_def += deficit
        if r["steps"] + deficit != pred:
            mism_s.append((r["tid"], i, j))
        s = 0
        for k, v in r["rhist"].items():
            a, b = (int(x) for x in k.split(","))
            supp_pairs[(a, b)] = supp_pairs.get((a, b), 0) + v; s += v
            if a >= b:
                nonstrict += v
        supp_ncl += r["classes"]; supp_cands += len(r["cands"])
        if s != r["classes"]:
            bad += 1
    complete = ok_ids and bad == 0
    closes = complete and not mism_s and (tot_rec + tot_def + supp_rec + supp_def == fx["steps_correct_total"])
    supp_ok = closes and nonstrict == 0 and supp_cands == 0
    supp_detail = (f"{len(srecs)} records for {len(skeys)} pairs, each once with the right pair: {complete}; steps recorded "
                   f"{supp_rec} + on-line skips {supp_def}; identity failures {len(mism_s)}; original + supplement + skips = "
                   f"{tot_rec + tot_def + supp_rec + supp_def} against the corrected total {fx['steps_correct_total']}: "
                   f"{tot_rec + tot_def + supp_rec + supp_def == fx['steps_correct_total']}; class sets {supp_ncl}, rank pairs "
                   f"{dict(sorted(supp_pairs.items()))}, non-strict {nonstrict}, candidates {supp_cands}")
verdict("g_supplement", supp_ok, supp_detail)

# (f) projected geometry facts used by the augmentation argument
isfree = d["isfree"]
npar_total = 0
for i in reps:
    Q1 = Q1_of(int(i))
    par = np.flatnonzero(np.all(Q1 == 0, axis=1))
    npar_total += int(np.sum(par != i))
E2T = np.vstack([d["E2"], d["T2"]])
tri = np.array(online_triples, dtype=np.int64)
if len(tri):
    S = np.hstack([tri, np.repeat(np.arange(N, N + 3)[None, :], len(tri), axis=0)])
    rk = ranks_batch(E2T, S, ELL2)                      # exact for values <= 8
    r3 = ranks_batch(d["E2"], tri, ELL2)
    exact_online = int(np.sum(rk <= 5))
    rk_hist = {int(k): int(v) for k, v in zip(*np.unique(rk, return_counts=True))}
    r3_hist = {int(k): int(v) for k, v in zip(*np.unique(r3, return_counts=True))}
else:
    exact_online = 0; rk_hist = {}; r3_hist = {}
verdict("f_geometry", int(isfree.sum()) == 0 and npar_total == 0 and exact_online == len(tri),
        f"states with zero projected image: {int(isfree.sum())}; states parallel to a first pivot mod ell: {npar_total}; "
        f"states above j on the projected line through a canonical pivot pair: {len(tri)}, of which on the line over "
        f"Q(w3) (exact rank of the two pivots, the state, and V_3 at most 5): {exact_online}; "
        f"rank histogram with V_3 {rk_hist}, without {r3_hist}")

json.dump(dict(results=res, gaps=gaps, records=len(recs), tasks=len(tasks), steps_recorded=tot_rec,
               steps_skipped_online=tot_def, steps_predicted=steps_pred, rank_pairs={f"{a},{b}": v for (a, b), v in pairs.items()},
               sizes={str(k): v for k, v in sorted(sizes.items())}, decider_levels=how, splits={str(k): v for k, v in split_hist.items()},
               online_triples=len(online_triples), online_triples_exact=exact_online, pairs=npairs, pairs_certificate=npairs_c,
               steps_missing=fx['steps_missing'], steps_corrected_total=fx['steps_correct_total'], supplement_steps=supp_rec,
               supplement_skipped_online=supp_def, supplement_rank_pairs={f'{a},{b}': v for (a, b), v in supp_pairs.items()}),
          open(OUT, "w"), indent=1)
if not gaps:
    print("ALL CHECKS PASS")
elif gaps == ["e_symmetry"] and res["g_supplement"]["ok"]:
    print("ALL CHECKS PASS with the third-pivot gap of (e) closed by the supplementary scan (g)")
else:
    print(f"GAPS: {gaps}")
log(f"written {OUT}")
