"""Cost per verified discovery, from the run log and the board's ledger.

Reads `autoresearch/runs.jsonl` (one line per annealing run) and
`docs/ledger.json` (one row per bound with its tier) and prints, per cell, the
number of runs, the CPU-hours spent, how many runs solved the cell, and the
CPU-hours per exact solution. The last column is the quantity the Phase I
review asks for: it should fall over time if the loop is learning anything.

Usage: summary.py [--since YYYY-MM-DD] [--json]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "autoresearch", "runs.jsonl")
LEDGER = os.path.join(ROOT, "docs", "ledger.json")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv[1:])

    cells = collections.defaultdict(lambda: {"runs": 0, "cpu_s": 0.0, "wall_s": 0.0,
                                             "solved": 0, "exact": 0, "best": 1.0})
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            if a.since and r["when"][:10] < a.since:
                continue
            c = cells[(r["orbit"], r["m"], r["rank"])]
            c["runs"] += 1
            c["cpu_s"] += r["cpu_s"]
            c["wall_s"] += r["wall_s"]
            c["solved"] += int(bool(r.get("solved")))
            c["exact"] += int(bool(r.get("exact")))
            c["best"] = min(c["best"], r["residual"])
    tiers = {}
    if os.path.exists(LEDGER):
        for row in json.load(open(LEDGER))["rows"]:
            if row["direction"] == "upper":
                tiers[(row["orbit"], row["m"], row["rank"])] = row["tier"]
    rows = []
    for (orbit, m, rank), c in sorted(cells.items()):
        cpu_h = c["cpu_s"] / 3600
        rows.append({"orbit": orbit, "m": m, "rank": rank, "runs": c["runs"],
                     "cpu_hours": round(cpu_h, 3), "exact_solutions": c["exact"],
                     "cpu_hours_per_solution": round(cpu_h / c["exact"], 3) if c["exact"] else None,
                     "best_residual": round(c["best"], 4),
                     "board_tier": tiers.get((orbit, m, rank), "not on board")})
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    if not rows:
        print("no runs logged")
        return 0
    print(f"{'cell':<18}{'runs':>5}{'CPU-h':>8}{'exact':>6}{'CPU-h/soln':>12}{'best':>8}  tier")
    for r in rows:
        per = f"{r['cpu_hours_per_solution']:.3f}" if r["cpu_hours_per_solution"] is not None else "-"
        print(f"{r['orbit'] + ' m=' + str(r['m']) + ' r=' + str(r['rank']):<18}{r['runs']:>5}"
              f"{r['cpu_hours']:>8.3f}{r['exact_solutions']:>6}{per:>12}{r['best_residual']:>8.4f}"
              f"  {r['board_tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
