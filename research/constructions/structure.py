"""Slice structure of known decompositions: how many terms are local along
each qudit, and whether the full/local dichotomy holds.

A term is local along qudit q if it is nonzero on exactly one slice, i.e. it
is |k> (x) v up to reordering of the qudits; otherwise every slice is
nonzero (full). The stored minimal decompositions under data/ and the
board's witnesses under bounds/ are both reported.

Usage: structure.py            (all stored lists and bound witnesses)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ORBIT_P, ROOT, term_vector  # noqa: E402


def slice_counts(v, p, m, q):
    """Number of nonzero slices of v along qudit q."""
    T = v.reshape((p,) * m)
    T = np.moveaxis(T, q, 0).reshape(p, -1)
    return int((np.linalg.norm(T, axis=1) > 1e-9).sum())


def describe(terms, p, m):
    out = []
    for q in range(m):
        counts = [slice_counts(t, p, m, q) for t in terms]
        bad = [c for c in counts if c not in (1, p)]
        out.append({"qudit": q, "local": sum(c == 1 for c in counts),
                    "full": sum(c == p for c in counts), "other": len(bad)})
    return out


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        rec = json.load(open(path))
        p, m = ORBIT_P[rec["orbit"]], rec["m"]
        pattern = {}
        for dec in rec["decompositions"]:
            terms = [term_vector(t, p, m) for t in dec]
            d = describe(terms, p, m)
            key = tuple(sorted((x["local"], x["other"]) for x in d))
            pattern[key] = pattern.get(key, 0) + 1
        print(f"{rec['orbit']} m={m} rank {rec['rank']}: {rec['count']} decompositions; "
              f"(local terms per qudit, sorted) -> count: {pattern}")
        rows.append((rec["orbit"], m, rec["rank"], pattern))
    print()
    for path in sorted(glob.glob(os.path.join(ROOT, "bounds", "*-upper-*.json"))):
        sub = json.load(open(path))
        p, m = ORBIT_P[sub["orbit"]], int(sub["m"])
        if m < 2 or "witness" not in sub:
            continue
        terms = [term_vector(t, p, m) for t in sub["witness"]["terms"]]
        d = describe(terms, p, m)
        print(f"{os.path.basename(path)}: local per qudit {[x['local'] for x in d]}, "
              f"non-dichotomy {sum(x['other'] for x in d)}")


if __name__ == "__main__":
    main()
