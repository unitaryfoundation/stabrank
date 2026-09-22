"""Nine-copy witnesses for the qubit H-type orbit from the cat-state machinery.

Every route through the constructions of Qassim, Pashayan, and Gosset
(arXiv:2106.07740) and Kissinger, van de Wetering, and Vilmart
(arXiv:2202.09202) lands on 18 terms at nine copies:

  project     (I^9 (x) <b|)|cat_10> is proportional to |cat_9> for b = 0 and
              to |cat_9^-> = (|T>^9 - |T_perp>^9)/sqrt 2 for b = 1, so the
              nine glued cat_10 terms give |T>^9 = (|cat_9> + |cat_9^->)/
              sqrt 2 in at most 18 terms.
  glue        |cat_9> is proportional to (I (x) <cat_2| (x) I)(|cat_6> (x)
              |cat_5>) with |cat_5> = sqrt 2 (I (x) <0|)|cat_6> in three
              terms: 3 x 3 = 9 terms for |cat_9>, 18 for |T>^9.
  partial     |T>^5 = sqrt 2 (I (x) <T|)|cat_6> leaves one |T> in each of
              three terms, so |T>^9 costs 3 chi(T^5) = 18 with the board's
              m = 5 witness.
  products    |T>^6 (x) |T>^3 (6 x 3) and |T>^7 (x) |T>^2 (9 x 2).

This script builds all of them numerically in the T basis, checks each
against |T>^9, pools the distinct stabilizer states they use, reports how
many there are and whether iterative pruning finds a shorter combination
inside the pool, and writes the projected-cat_10 witness as
bounds/qubit_H-m9-upper-18.json in exact arithmetic.

Usage (from the repository root):
    uv run --extra challenge python research/constructions/kvv_cat_m9.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

import kvv_cat  # noqa: E402
from kvv_cat import (C, T, apply_on, cat, cat6_terms, cat10_terms, lstsq_check, power,  # noqa: E402
                     split_T_power, to_H_basis)
from stabrank_verify import stabilizer_vector  # noqa: E402
from to_witness import witness_from_vectors  # noqa: E402


def board_terms_T_basis(m):
    """The board's best |H>^m witness terms, mapped to the T basis (kvv_cat's
    version assumes the file is named -upper-{m}, which fails at m = 5)."""
    import glob
    import re
    paths = glob.glob(os.path.join(ROOT, "bounds", f"qubit_H-m{m}-upper-*.json"))
    path = min(paths, key=lambda q: int(re.search(r"-upper-(\d+)\.json$", q).group(1)))
    sub = json.load(open(path))
    out = []
    for t in sub["witness"]["terms"]:
        v = np.array(stabilizer_vector(t, 2, m).evalf(20), dtype=complex).ravel()
        for q in range(m):
            v = apply_on(C.conj().T, q, m, v)
        out.append(v)
    lstsq_check(power(T, m), out, f"board m={m} witness ({os.path.basename(path)}) in the T basis")
    return out


kvv_cat.board_terms_T_basis = board_terms_T_basis


def project_last(vec, n, b):
    return vec.reshape(2 ** (n - 1), 2)[:, b].copy()


def distinct(vectors, tol=1e-9):
    """One representative per ray (vectors equal up to a scalar)."""
    reps = []
    for v in vectors:
        nv = np.linalg.norm(v)
        if nv < tol:
            continue
        u = v / nv
        if not any(abs(abs(np.vdot(r, u)) - 1) < tol for r in reps):
            reps.append(u)
    return reps


def route_project():
    c10 = cat10_terms()
    out = []
    for b in (0, 1):
        proj = [project_last(t, 10, b) for t in c10]
        proj = [v for v in proj if np.linalg.norm(v) > 1e-12]
        tgt = project_last(cat(10), 10, b)
        lstsq_check(tgt, proj, f"cat_10 projected onto |{b}> on qubit 10 ({len(proj)} terms)")
        out += proj
    return out


def route_glue():
    c6 = cat6_terms()
    c5 = [project_last(t, 6, 0) for t in c6]
    lstsq_check(cat(5), c5, "cat_5 from cat_6 projected onto |0>")
    bra = np.zeros(4, dtype=complex)
    bra[0], bra[3] = 1, -1j
    out = []
    for si in c6:
        for sj in c5:
            v = np.kron(si, sj).reshape(32, 4, 16)
            u = np.einsum("a,iaj->ij", bra, v).reshape(-1)
            if np.linalg.norm(u) > 1e-12:
                out.append(u)
    lstsq_check(cat(9), out, f"cat_9 glued from cat_6 and cat_5 ({len(out)} terms)")
    return split_T_power(out, 9)


def route_partial():
    from kvv_cat import S
    r = 4
    choi_terms = [apply_on(S.conj().T, 0, r + 1, v) for v in board_terms_T_basis(r + 1)]
    out = []
    for s in cat6_terms():
        s = s.reshape(32, 2)
        for ch in choi_terms:
            M = ch.reshape(2, 2 ** r).T
            out.append(np.einsum("xa,ba->xb", s, M).reshape(-1))
    return out


def route_products():
    t6 = split_T_power(cat6_terms(), 6)
    t3 = board_terms_T_basis(3)
    t2 = board_terms_T_basis(2)
    from kvv_cat import partial_product_terms
    t7 = partial_product_terms(2)
    a = [np.kron(x, y) for x in t6 for y in t3]
    b = [np.kron(x, y) for x in t7 for y in t2]
    return a, b


def prune(target, pool):
    """Greedy pruning: drop the pool vector whose removal raises the
    least-squares residual least, while the residual stays zero."""
    keep = list(range(len(pool)))
    A = np.stack(pool, axis=1)
    while True:
        best = None
        for j in keep:
            cols = [c for c in keep if c != j]
            c, *_ = np.linalg.lstsq(A[:, cols], target, rcond=None)
            res = np.linalg.norm(A[:, cols] @ c - target)
            if res < 1e-9 and (best is None or res < best[1]):
                best = (j, res)
        if best is None:
            return keep
        keep.remove(best[0])


def main(argv):
    m = 9
    tgt = power(T, m)
    routes = {}
    routes["project cat_10"] = route_project()
    routes["glue cat_6 with cat_5"] = route_glue()
    routes["partial 3 x chi(T^5)"] = route_partial()
    a, b = route_products()
    routes["product T^6 x T^3"] = a
    routes["product T^7 x T^2"] = b
    pool = []
    for name, terms in routes.items():
        lstsq_check(tgt, terms, name)
        d = distinct(terms)
        print(f"    {name}: {len(terms)} terms, {len(d)} distinct rays; "
              f"span rank {np.linalg.matrix_rank(np.stack(d, axis=1), tol=1e-8)}")
        pool += d
    pool = distinct(pool)
    A = np.stack(pool, axis=1)
    print(f"  pool over all routes: {len(pool)} distinct stabilizer states, span rank "
          f"{np.linalg.matrix_rank(A, tol=1e-8)}")
    t0 = time.time()
    kept = prune(tgt, pool)
    print(f"  greedy pruning inside the pool: {len(kept)} terms suffice [{time.time() - t0:.0f}s]")
    # A few randomised pruning orders, in case the greedy order is unlucky.
    rng = np.random.default_rng(1)
    best = len(kept)
    for _ in range(20):
        order = rng.permutation(len(pool))
        keep = list(order)
        for j in list(order):
            cols = [c for c in keep if c != j]
            if len(cols) < 1:
                continue
            c, *_ = np.linalg.lstsq(A[:, cols], tgt, rcond=None)
            if np.linalg.norm(A[:, cols] @ c - tgt) < 1e-9:
                keep = cols
        best = min(best, len(keep))
    print(f"  best over 20 random pruning orders: {best} terms")

    vecs = [to_H_basis(t, m) for t in routes["project cat_10"]]
    assert len(vecs) == 18
    w = witness_from_vectors("qubit_H", m, vecs)
    sub = {
        "schema_version": "0.1", "orbit": "qubit_H", "m": m, "direction": "upper", "rank": 18,
        "provenance": {
            "author": "this repository",
            "reference": "arXiv:2106.07740, arXiv:2202.09202",
            "method": "structured construction",
            "date": "2026-09-22",
            "github": ["vprusso"],
            "compute": {"cpu_hours": 0.01, "wall_clock_hours": 0.01, "runs": 1,
                        "hardware": "Apple silicon laptop, one core; direct construction, no search"},
        },
        "notes": (
            "chi(|H>^9) <= 18, the first entry for this cell; exponent log_2(18)/9 = 0.4633, "
            "above the published log_2(3)/4. Construction: the nine-term glued |cat_10> of "
            "Qassim, Pashayan, and Gosset (as restated by Kissinger, van de Wetering, and "
            "Vilmart) projected onto |0> and onto |1> on the tenth qubit gives |cat_9> and "
            "|cat_9^-> = (|T>^9 - |T_perp>^9)/sqrt 2 in nine terms each, and |T>^9 = (|cat_9> + "
            "|cat_9^->)/sqrt 2. Every other route through the same machinery (gluing cat_6 with "
            "cat_5, the 4-to-3 partial decomposition times the board's m=5 witness, the products "
            "T^6 x T^3 and T^7 x T^2) also gives 18, and the pool of distinct stabilizer states "
            "the routes use admits no shorter combination by pruning "
            "(research/constructions/kvv_cat_m9.py). Built here and verified in exact arithmetic."),
        "witness": w,
    }
    out = os.path.join(ROOT, "bounds", "qubit_H-m9-upper-18.json")
    with open(out, "w") as f:
        json.dump(sub, f, indent=2)
        f.write("\n")
    print(f"  written {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
