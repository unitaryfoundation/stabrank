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

For an orbit with no published exponent (T5, measured against the
single-copy product bound log_p chi(|M>)) a product of cells that are
themselves below that bound lands below it too, and the notes say so; the
site does not count that as a beaten exponent.

Usage (from the repository root):
    uv run --extra challenge python research/constructions/product_witness.py ORBIT M1 M2 [M3 ...] [--date YYYY-MM-DD]
reads, for each Mi, the smallest-rank witness-bearing bounds/ORBIT-m{Mi}-upper-*.json
or, when the stored decomposition lists under data/ hold a smaller rank (the
m <= 3 cells of N, H3, and T3 are Lean or literature entries without a
witness), data/ORBIT_m{Mi}_rank*.json with coefficients fitted exactly, and
writes bounds/ORBIT-m{sum Mi}-upper-{prod ranks}.json. The date defaults to
today.
"""

from __future__ import annotations

import datetime
import glob
import json
import math
import os
import re
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

from stabrank_verify import ORBIT_P, implied_gamma  # noqa: E402

# Orbits measured against a single-copy product bound rather than a published
# exponent (CONTRIBUTING.md); the baseline is log_p of the m=1 rank.
UNPUBLISHED_BASELINE = {"T5": 3}


def best_upper(orbit, m):
    """The smallest-rank bound file at (orbit, m) that carries a witness, or
    None (the m <= 3 qutrit cells of N, H3, and T3 are literature or Lean
    entries without one)."""
    paths = glob.glob(os.path.join(ROOT, "bounds", f"{orbit}-m{m}-upper-*.json"))
    ranked = sorted((int(re.search(r"-upper-(\d+)\.json$", p).group(1)), p) for p in paths)
    for rank, path in ranked:
        if "witness" in json.load(open(path)):
            return rank, path
    return None


def best_data(orbit, m):
    """The smallest-rank stored decomposition list under data/, or None."""
    paths = glob.glob(os.path.join(HERE, "data", f"{orbit}_m{m}_rank*.json"))
    ranked = sorted((int(re.search(r"_rank(\d+)\.json$", p).group(1)), p) for p in paths)
    return ranked[0] if ranked else None


def factor(orbit, m):
    """An exact witness for |M>^m: the best witness-bearing bound file, or, if
    the stored decomposition lists of `data/` hold a smaller rank (or the only
    one), their first decomposition with coefficients fitted exactly by
    fit_coeffs.fit. Returns (witness, rank, description)."""
    from fit_coeffs import fit
    bound = best_upper(orbit, m)
    data = best_data(orbit, m)
    if data is not None and (bound is None or data[0] < bound[0]):
        rank, path = data
        rec = json.load(open(path))
        terms = rec["decompositions"][0]
        coeffs = fit(orbit, m, terms)
        if coeffs is None:
            raise ValueError(f"no exact coefficients for {path}")
        rel = os.path.relpath(path, ROOT)
        return ({"terms": terms, "coeffs": [str(c) for c in coeffs]}, rank,
                f"{rel} (decomposition 0 of {rec['count']}, coefficients fitted exactly by "
                f"fit_coeffs.fit)")
    if bound is None:
        raise FileNotFoundError(f"no witness for {orbit} m={m} in bounds/ or data/")
    rank, path = bound
    return json.load(open(path))["witness"], rank, os.path.basename(path)


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
    date = datetime.date.today().isoformat()
    if "--date" in argv:
        i = argv.index("--date")
        date = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    orbit = argv[1]
    ms = [int(a) for a in argv[2:]]
    p = ORBIT_P[orbit]
    facs = [factor(orbit, m) for m in ms]
    w, n = facs[0][0], ms[0]
    for (wf, _, _), m in zip(facs[1:], ms[1:]):
        w = product_witness(w, wf, n, m)
        n += m
    rank = len(w["terms"])
    factors = " x ".join(f"chi(|{orbit}>^{m}) <= {r}" for (_, r, _), m in zip(facs, ms))
    seen = {}
    for _, _, d in facs:
        seen[d] = seen.get(d, 0) + 1
    sources = ", ".join(d if c == 1 else f"{d} (taken {c} times)" for d, c in seen.items())
    gamma = implied_gamma(p, rank, n)
    if orbit in UNPUBLISHED_BASELINE:
        base = math.log(UNPUBLISHED_BASELINE[orbit], p)
        standing = (
            f"First witness for this cell. The orbit has no published exponent and is measured "
            f"against the single-copy product bound log_{p}({UNPUBLISHED_BASELINE[orbit]}) = "
            f"{base:.4f}; this exponent is {'below' if gamma < base else 'not below'} that "
            f"baseline because the m=2 factor already is, so it beats no published exponent "
            f"and the site does not count it as one. It is the sub-multiplicative value the "
            f"board already implied.")
    else:
        standing = ("First witness for this cell; it is the sub-multiplicative value the board "
                    "already implied and does not move the exponent.")
    out = {
        "schema_version": "0.1", "orbit": orbit, "m": n, "direction": "upper", "rank": rank,
        "provenance": {
            "author": "this repository",
            "reference": "stabrank research/constructions/product_witness.py",
            "method": "structured construction",
            "date": date,
            "github": ["vprusso"],
            "compute": {"cpu_hours": 0.001, "wall_clock_hours": 0.001, "runs": 1,
                        "hardware": "Apple silicon laptop, one core; direct construction, no search"},
        },
        "notes": (
            f"Tensor product of the exact decompositions {sources}: "
            f"{factors} gives chi(|{orbit}>^{n}) <= {rank}, exponent {gamma:.4f}. {standing} "
            "Terms are the exact products (x0 and l concatenated, W and Q "
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
