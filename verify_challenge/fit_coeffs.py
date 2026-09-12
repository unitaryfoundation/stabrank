"""Solve for the exact coefficients of a proposed decomposition.

Submitters supply the stabilizer terms; this recovers the coefficients in exact
arithmetic, or reports that no exact combination of those terms reproduces the
target. Running it before submitting turns a "close" decomposition into either
a submission or a clear no, without burning a verification slot.

The terms are linearly independent in every case that matters (a dependent set
would mean the true rank is lower), so the exact normal equations
(A^H A)^{-1} A^H b give the unique candidate, and it is then checked against the
target on the nose rather than by residual size.
"""

from __future__ import annotations

import json
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stabrank_verify import ORBIT_P, is_zero, stabilizer_vector, target_vector


def fit(orbit, m, terms):
    """Exact coefficients, or None if no exact combination reproduces the target."""
    p = ORBIT_P[orbit]
    cols = [stabilizer_vector(t, p, m) for t in terms]
    A = sp.Matrix.hstack(*cols)
    b = target_vector(orbit, m)

    if A.rank() < A.cols:
        raise ValueError("the supplied terms are linearly dependent, so the true "
                         "rank is below the number of terms; drop the redundant ones")
    G = sp.simplify(A.H * A)
    x = sp.simplify(G.inv() * (A.H * b))
    if any(not is_zero(v)[0] for v in (A * x - b)):
        return None
    return [sp.nsimplify(sp.radsimp(sp.simplify(v))) for v in x]


def main(argv):
    if len(argv) < 2:
        print("usage: fit_coeffs.py <submission.json> [--write]")
        return 2
    path = argv[1]
    sub = json.load(open(path))
    try:
        c = fit(sub["orbit"], int(sub["m"]), sub["witness"]["terms"])
    except ValueError as exc:
        print(f"rejected: {exc}")
        return 1
    if c is None:
        print("no exact combination of these terms reproduces the target")
        return 1
    strs = [str(v) for v in c]
    print("exact coefficients:")
    for s in strs:
        print(f"  {s}")
    if "--write" in argv:
        sub["witness"]["coeffs"] = strs
        with open(path, "w") as f:
            json.dump(sub, f, indent=2)
            f.write("\n")
        print(f"written into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
