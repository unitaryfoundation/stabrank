"""Exact tensor-product witnesses from bound files already on the board.

|M>^(a+b) = |M>^a (x) |M>^b, so the terms s_i (x) t_j of two verified
decompositions, with coefficients a_i b_j, are a decomposition of the
(a+b)-copy state. In the board's parametrisation the product of two terms
is the term with x0 and l concatenated, W and Q block diagonal, and k added,
so the product witness is written exactly, without a numerical detour, and
the coefficients are the sympy products of the factor coefficients.

The bound is only chi(|M>^(a+b)) <= chi_a chi_b, the sub-multiplicative
value the board already implies, and it never moves an exponent; it fills a
cell the board holds no witness for, and it is the warm start the merge
anneals of docs/notes/merge_recipes_2026_09.md begin from.

Usage (from the repository root):
    uv run --extra challenge python research/constructions/product_witness.py ORBIT M1 M2 [M3 ...]
reads bounds/ORBIT-m{Mi}-upper-*.json (the smallest rank at each Mi) and
writes bounds/ORBIT-m{sum Mi}-upper-{prod ranks}.json.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank_verify import ORBIT_P, implied_gamma  # noqa: E402


def best_upper(orbit, m):
    paths = glob.glob(os.path.join(ROOT, "bounds", f"{orbit}-m{m}-upper-*.json"))
    ranked = sorted((int(re.search(r"-upper-(\d+)\.json$", p).group(1)), p) for p in paths)
    if not ranked:
        raise FileNotFoundError(f"no upper bound file for {orbit} m={m}")
    return ranked[0][1]


def product_term(s, t, n_s, n_t):
    ks, kt = int(s["k"]), int(t["k"])
    Ws = [list(r) for r in s.get("W", [])]
    Wt = [list(r) for r in t.get("W", [])]
    Qs = s.get("Q", [[0] * ks for _ in range(ks)])
    Qt = t.get("Q", [[0] * kt for _ in range(kt)])
    W = [r + [0] * n_t for r in Ws] + [[0] * n_s + r for r in Wt]
    Q = [list(Qs[i]) + [0] * kt for i in range(ks)] + [[0] * ks + list(Qt[i]) for i in range(kt)]
    return {"k": ks + kt, "x0": list(s["x0"]) + list(t["x0"]), "W": W, "Q": Q,
            "l": list(s.get("l", [0] * ks)) + list(t.get("l", [0] * kt))}


def product_witness(wa, wb, n_a, n_b):
    terms, coeffs = [], []
    for s, ca in zip(wa["terms"], wa["coeffs"]):
        for t, cb in zip(wb["terms"], wb["coeffs"]):
            terms.append(product_term(s, t, n_a, n_b))
            c = sp.nsimplify(sp.radsimp(sp.expand(sp.sympify(ca) * sp.sympify(cb))))
            coeffs.append(str(c))
    return {"terms": terms, "coeffs": coeffs}


def main(argv):
    orbit = argv[1]
    ms = [int(a) for a in argv[2:]]
    p = ORBIT_P[orbit]
    files = [best_upper(orbit, m) for m in ms]
    subs = [json.load(open(f)) for f in files]
    w, n = subs[0]["witness"], ms[0]
    for sub, m in zip(subs[1:], ms[1:]):
        w = product_witness(w, sub["witness"], n, m)
        n += m
    rank = len(w["terms"])
    factors = " x ".join(f"chi(|{orbit}>^{m}) <= {s['rank']}" for s, m in zip(subs, ms))
    gamma = implied_gamma(p, rank, n)
    out = {
        "schema_version": "0.1", "orbit": orbit, "m": n, "direction": "upper", "rank": rank,
        "provenance": {
            "author": "this repository",
            "reference": "stabrank research/constructions/product_witness.py",
            "method": "structured construction",
            "date": "2026-09-22",
            "github": ["vprusso"],
            "compute": {"cpu_hours": 0.001, "wall_clock_hours": 0.001, "runs": 1,
                        "hardware": "Apple silicon laptop, one core; direct construction, no search"},
        },
        "notes": (
            f"Tensor product of the board's witnesses {', '.join(os.path.basename(f) for f in files)}: "
            f"{factors} gives chi(|{orbit}>^{n}) <= {rank}, exponent {gamma:.4f}. First witness "
            "for this cell; it is the sub-multiplicative value the board already implied and does "
            "not move the exponent. Terms are the exact products (x0 and l concatenated, W and Q "
            "block diagonal) and coefficients the exact products, built by "
            "research/constructions/product_witness.py."),
        "witness": w,
    }
    path = os.path.join(ROOT, "bounds", f"{orbit}-m{n}-upper-{rank}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"written {os.path.relpath(path, ROOT)}: rank {rank}, gamma {gamma:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
