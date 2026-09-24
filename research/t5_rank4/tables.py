"""The residual tables behind the structural facts of the rank-4 exclusion
of |T>^5 (docs/notes/t5_rank4_exclusion.md, section 1), recomputed with the
committed machinery of research/constructions/two_qubit_slice.py and stored
under results/tables.json.

    tables.py [--J 2] [--out PATH]

For a stored minimal decomposition (d_i, u_i) of |T>^{n_2} and a slice
ratio tau^j (tau = a_1 / a_0 = e^{i pi/4} tan(beta) the ratio between
neighbouring points of F_2^{n_1}), the table counts, over every code
combination of the visible terms (4 2^{n_2} + 1 options each: the phased
Pauli translates of u_i and absence), the combinations whose residual
tau^j psi_{n_2} - sum_i d_i w_i is zero ("exact") and those whose residual
is a nonzero multiple of a stabilizer state ("stabilizer").

  t4: the unique rank-3 decomposition of |T>^4 (qubit_T_m4_rank3.json) at
      |j| <= J. Fact 1 (every term of a rank-4 decomposition of |T>^5 is
      full along every qubit) needs (exact, stabilizer) = (0, 0) at j = -1
      and j = 1: a one-qubit slice with exactly three visible terms would
      make the fourth term local and the equation at the other value a
      stabilizer-residual row at ratio tau^{+-1}.
  t3: the 8 stored rank-3 decompositions of |T>^3 (qubit_T_m3_rank3.json)
      at |j| <= J. The configuration with one line term on each diagonal
      of F_2^2 needs an exact row at j = 2 (or, from the other base point,
      j = -2), so 0 exact combinations there excludes it.

This script runs on its own path (research/constructions first, so that
two_qubit_slice's `common` is the constructions module) and imports nothing
from research/t5_rank4; driver.py runs it as a subprocess.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from common import alpha, load_decompositions, target  # noqa: E402  (research/constructions/common.py)
from two_qubit_slice import Dict, PauliEnv, TermShapes, residual_table  # noqa: E402

ORBIT = "qubit_T"


def ratio_tables(n2, rank, J, log):
    """Per stored decomposition and ratio tau^j the (exact, stabilizer)
    counts, and the totals."""
    a = alpha(ORBIT)
    tau = a[1] / a[0]
    psi = target(ORBIT, n2)
    dic = Dict(n2)
    env = PauliEnv(n2)
    decs, rec = load_decompositions(ORBIT, n2, rank)
    rows, totals = [], {j: [0, 0] for j in range(-J, J + 1)}
    for di, (u, d) in enumerate(decs):
        Vs = [TermShapes(env, ui, 0, 1).slice_vectors() for ui in u]
        row = {}
        for j in range(-J, J + 1):
            tab = residual_table(Vs, d, (tau ** j) * psi, dic)
            row[j] = (int((tab == 0).sum()), int((tab == 1).sum()))
            totals[j][0] += row[j][0]
            totals[j][1] += row[j][1]
        rows.append({str(j): list(v) for j, v in row.items()})
        log(f"  dec {di}: " + " ".join(f"j={j}:{row[j]}" for j in range(-J, J + 1)))
    log("  totals (exact, stabilizer) over all decompositions: "
        + " ".join(f"j={j}:{tuple(totals[j])}" for j in range(-J, J + 1)))
    return {"orbit": ORBIT, "n2": n2, "rank": rank, "decompositions": len(decs), "source": rec.get("source"),
            "options_per_term": 4 * (1 << n2) + 1, "rows": rows,
            "totals": {str(j): list(v) for j, v in totals.items()}}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--J", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "tables.json"))
    a = ap.parse_args(argv[1:])
    try:
        os.nice(19)
    except OSError:
        pass
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)

    out = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "git": None, "runs": {}}
    try:
        out["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                             stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    ok = True
    log("t4: the rank-3 decomposition of |T>^4, one sliced qubit, ratios tau^j")
    t1 = time.time()
    r4 = ratio_tables(4, 3, a.J, log)
    r4["seconds"] = time.time() - t1
    good = r4["totals"]["-1"] == [0, 0] and r4["totals"]["1"] == [0, 0]
    r4["fact"] = "Fact 1: (exact, stabilizer) = (0, 0) at j = -1 and j = 1"
    r4["pass"] = bool(good)
    log(f"  Fact 1: {r4['totals']['-1']} at j = -1, {r4['totals']['1']} at j = 1: {'OK' if good else 'FAIL'}")
    ok &= good
    out["runs"]["t4"] = r4
    log("t3: the rank-3 decompositions of |T>^3, one sliced qubit, ratios tau^j")
    t1 = time.time()
    r3 = ratio_tables(3, 3, a.J, log)
    r3["seconds"] = time.time() - t1
    good = r3["totals"]["-2"][0] == 0 and r3["totals"]["2"][0] == 0
    r3["fact"] = "the (1, 1) configuration: 0 exact combinations at j = -2 and j = 2"
    r3["pass"] = bool(good)
    log(f"  (1, 1) exclusion: exact combinations {r3['totals']['-2'][0]} at j = -2, {r3['totals']['2'][0]} "
        f"at j = 2: {'OK' if good else 'FAIL'}")
    ok &= good
    out["runs"]["t3"] = r3
    out["pass"] = bool(ok)
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {os.path.relpath(a.out, ROOT)}; tables {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
