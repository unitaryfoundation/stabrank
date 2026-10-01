"""Correct third-pivot masks for the case-C scan, and the triples the scan of
2026-09-22 to 26 did not visit.

Defect.  cost_c.kok_of computed the mask of admissible third pivots for a pair
(i, j) with nontrivial Stab(i, j) as the states k > j with min_h h(k) = k, the
minimum taken over the NONTRIVIAL elements h of Stab(i, j) only.  A state k that
no nontrivial element fixes and that is the least element of its orbit has
min_h h(k) > k over the nontrivial elements, so it was excluded, although it is
the canonical third pivot of every configuration whose canonical labelling has
it in third place.  The correct mask is min over all of Stab(i, j), identity
included: kok[k] = 1 iff k > j and k is the least element of its Stab(i, j)-orbit.
The stored masks (logs/costC_kok.pkl) are subsets of the correct ones; the
missing states are the orbit minima that are free points of Stab(i, j).  Pairs
with trivial Stab(i, j) carry no mask and were scanned in full.

This module recomputes every mask from the exact group action (symact.Group,
logs/sym_exact.npz), returns the missing masks (correct and not stored) per
pair, and the corrected inner-step total.  check_caseC.py uses it to report the
gap and to verify that the supplementary scan (caseC_supp.py) closes it.
"""
import os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from symact import Group


def correct_masks(G, reps, jokm, log=print):
    """{(i, j): kok} for every canonical pair with nontrivial Stab(i, j), from the full
    Stab(i, j) (identity included); pairs with trivial Stab(i, j) are absent."""
    N = G.N
    out = {}
    for r, i in enumerate(reps):
        i = int(i)
        sidx = G.stabilizer_indices(i)
        P = np.stack([G.act(*G.element(n)) for n in sidx])          # (|Stab(i)|, N), identity included
        assert np.all(P[:, i] == i)
        nontriv = ~np.all(P == np.arange(N)[None, :], axis=1)
        assert int(np.sum(~nontriv)) == 1, "exactly one identity expected"
        for j in np.flatnonzero(jokm[r]):
            j = int(j)
            rows = P[:, j] == j
            if int(np.sum(rows & nontriv)) == 0:
                continue                                              # trivial Stab(i, j): no mask
            kmin = P[rows].min(axis=0)
            kok = (kmin == np.arange(N)).astype(np.int8)
            kok[:j + 1] = 0
            out[(i, j)] = kok
        log(f"rep {r}/{len(reps)} state {i}: |Stab(i)| = {P.shape[0]}, masked pairs so far {len(out)}")
    return out


def steps_of_pair(N, i, j, kok):
    ks = np.flatnonzero(kok) if kok is not None else np.arange(j + 1, N)
    ks = ks[ks != i]
    return int(np.sum(N - ks - 1))


def missing_masks(correct, stored, N):
    """Per pair, the mask of admissible third pivots the stored mask excluded, and the
    step totals (stored, correct, missing) over the masked pairs."""
    assert set(correct) == set(stored), (len(correct), len(stored), len(set(correct) ^ set(stored)))
    miss = {}; st_steps = 0; co_steps = 0; mi_steps = 0
    for key, kok in correct.items():
        i, j = key
        s = stored[key]
        assert np.all(s <= kok), f"stored mask of {key} is not a subset of the correct one"
        m = ((kok == 1) & (s == 0)).astype(np.int8)
        st_steps += steps_of_pair(N, i, j, s); co_steps += steps_of_pair(N, i, j, kok)
        if m.any():
            miss[key] = m
            mi_steps += steps_of_pair(N, i, j, m)
    assert co_steps == st_steps + mi_steps
    return miss, st_steps, co_steps, mi_steps


def load_all(log=print):
    G = Group(); N = G.N
    z = np.load(os.path.join(HERE, "logs", "costC.npz"))
    reps, jokm, steps_pred = z["reps"], z["jokm"], int(z["steps"])
    stored = pickle.load(open(os.path.join(HERE, "logs", "costC_kok.pkl"), "rb"))
    t0 = time.time()
    correct = correct_masks(G, reps, jokm, log=lambda m: None)
    log(f"correct masks for {len(correct)} pairs in {time.time()-t0:.0f}s")
    miss, st, co, mi = missing_masks(correct, stored, N)
    return dict(G=G, N=N, reps=reps, jokm=jokm, steps_pred=steps_pred, stored=stored, correct=correct,
                missing=miss, steps_stored_masked=st, steps_correct_masked=co, steps_missing=mi,
                steps_correct_total=steps_pred - st + co)


if __name__ == "__main__":
    d = load_all()
    print(f"masked pairs {len(d['correct'])}; pairs with missing third pivots {len(d['missing'])}; "
          f"missing third pivots {sum(int(m.sum()) for m in d['missing'].values())}")
    print(f"steps over masked pairs: stored {d['steps_stored_masked']}, correct {d['steps_correct_masked']}, "
          f"missing {d['steps_missing']}")
    print(f"scan total: as run {d['steps_pred']}, corrected {d['steps_correct_total']}")
