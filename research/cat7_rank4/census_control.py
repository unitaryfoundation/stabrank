"""Census completeness control (referee, 2026-10-17): plant a random four-term
target t' = sum e_j a_j of independent four-qubit stabilizer states, run the
census through its TARGET path on the planted pivot over a random
sub-dictionary that contains the four states, and check that the planted
4-set is listed. CENSUS_OLD_HASH=1 selects the pre-referee functional
(R @ f) % P, which wraps int64 and misses plants.

    python census_control.py N SEED OUT.json
"""
import os, sys, json, time
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import census4_run as CR
import cat_slice2 as C
L = CR.L


def main():
    n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed)
    E = L.stabilizer_exponents(4)
    results = []
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"plant_census_{os.getpid()}.json")
    for k in range(n):
        while True:
            quad = sorted(int(x) for x in rng.choice(E.shape[0], 4, replace=False))
            cols = [C.exact_row(E[i]) for i in quad]
            if C.rank_exact(cols) == 4:
                break
        coefs = []
        for _ in range(4):
            while True:
                a, b = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
                if (a, b) != (0, 0):
                    coefs.append((Fraction(a), Fraction(b))); break
        t = [(Fraction(0), Fraction(0))] * 16
        for c, col in zip(coefs, cols):
            t = [(x[0] + c[0] * y[0] - c[1] * y[1], x[1] + c[0] * y[1] + c[1] * y[0]) for x, y in zip(t, col)]
        json.dump({"exact": [[str(x[0]), str(x[1])] for x in t]}, open(tmp, "w"))
        CR._G.clear()
        subset = sorted(set(int(x) for x in rng.choice(E.shape[0], 3000, replace=False)) | set(quad))
        t0 = time.time()
        res = CR.census_pivots((tmp, False, [quad[0]], 0, subset))
        found = any(list(q) == quad for q, d in res["covers"])
        results.append({"quad": quad, "coefs": [[str(c[0]), str(c[1])] for c in coefs], "found": found,
                        "covers_listed": len(res["covers"]), "pairs": res["pairs"], "seconds": round(time.time() - t0, 1)})
        print(f"plant {k}: quad {quad} found={found} ({len(res['covers'])} covers listed, {res['pairs']} pairs, {time.time()-t0:.0f}s)", flush=True)
    os.remove(tmp)
    summary = {"old_hash": CR.OLD_HASH, "plants": n, "found": sum(r["found"] for r in results), "results": results}
    json.dump(summary, open(out, "w"), indent=1)
    print(f"hash={'old (R @ f) % P' if CR.OLD_HASH else 'reduce-before-sum'}: {summary['found']} of {n} planted 4-covers listed")


if __name__ == "__main__":
    main()
