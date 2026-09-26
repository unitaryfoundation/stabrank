"""Product closure of the board's upper bounds (2026-09-26).

chi(|M>^{a+b}) <= chi(|M>^a) chi(|M>^b), so the board's upper bounds are only
meaningful if no split of a cell beats the bound filed there. This prints, for
every copy orbit and every m up to `--mmax`, the best value reachable by
splitting m into parts whose cells the ledger already holds, and flags the
cells where that value is below the filed bound (a bound to rewrite) or where
the ledger holds no bound at all (a cell to fill with a product witness).

The cat track is excluded: |cat_m> is one state per m, not a tensor power, so
there is no split.

    uv run --extra challenge python research/constructions/ledger_closure.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank_verify import ORBIT_P, FAMILY  # noqa: E402

PUBLISHED = {"S": 0.3155, "N": 0.4206, "H3": 0.4206, "T3": 0.5000,
             "qubit_H": 0.3962, "qubit_T": 0.3962, "T5": 0.5000}


def board(path):
    rows = json.load(open(path))["rows"]
    up, lo = {}, {}
    for r in rows:
        key = (r["orbit"], r["m"])
        if r["direction"] == "upper":
            up[key] = min(up.get(key, 10 ** 9), r["rank"])
        else:
            lo[key] = max(lo.get(key, 0), r["rank"])
    return up, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(ROOT, "docs", "ledger.json"))
    ap.add_argument("--mmax", type=int, default=12)
    a = ap.parse_args()
    up, lo = board(a.ledger)
    orbits = sorted({o for o, _ in up} - set(FAMILY))
    print(f"{'orbit':9s} {'m':>3s} {'filed':>6s} {'split':>6s} {'how':>16s} {'gamma(split)':>13s}  verdict")
    for orbit in orbits:
        p = ORBIT_P[orbit]
        best = {}
        for m in range(1, a.mmax + 1):
            cand = {}
            if (orbit, m) in up:
                cand[up[(orbit, m)]] = "filed"
            for k in range(1, m // 2 + 1):
                if k in best and (m - k) in best:
                    v = best[k][0] * best[m - k][0]
                    cand.setdefault(v, f"{k}+{m - k}")
            if not cand:
                continue
            v = min(cand)
            best[m] = (v, cand[v])
            filed = up.get((orbit, m))
            split = min((x for x in cand if cand[x] != "filed"), default=None)
            if split is None and filed is None:
                continue
            g = math.log(v, p) / m
            if filed is None:
                verdict = "NO CELL: a product witness would fill it"
            elif split is not None and split < filed:
                verdict = "SPLIT BEATS THE FILED BOUND"
            else:
                verdict = "closed"
            if g < PUBLISHED[orbit] - 1e-9 and filed is not None:
                verdict += "; BELOW THE PUBLISHED EXPONENT"
            print(f"{orbit:9s} {m:3d} {str(filed):>6s} {str(split):>6s} "
                  f"{cand[v]:>16s} {g:13.4f}  {verdict}")
    print()
    print("Lower bounds carry up by projection monotonicity: chi(|M>^{m+1}) >= chi(|M>^m).")
    for orbit in orbits:
        run = 0
        bad = []
        for m in range(1, a.mmax + 1):
            if (orbit, m) in lo:
                if lo[(orbit, m)] < run:
                    bad.append((m, lo[(orbit, m)], run))
                run = max(run, lo[(orbit, m)])
        if bad:
            print(f"  {orbit}: lower bounds below the value carried up from a smaller m: {bad}")
    print("  (a cell listed above holds a lower bound weaker than one already "
          "implied by a smaller m)")


if __name__ == "__main__":
    main()
