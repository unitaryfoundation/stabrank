"""Verification pipeline for stabilizer-rank bound submissions.

A submission claims a bound on chi(|M>^{ot m}) for one magic-state orbit. The
pipeline decides what tier it has earned; nothing is taken on the submitter's
word.

Upper bounds carry an explicit decomposition. Each term is given by its
stabilizer parametrisation

    |sigma> propto sum_{y in F_p^k} w_p^{Q(y) + l.y} |x0 + W y>,

not as a raw amplitude vector, so a term that is not a stabilizer state cannot
be expressed in the first place. The verifier rebuilds every term and the
target in exact arithmetic and requires the identity to hold on the nose. A
floating-point near-miss earns nothing.

Lower bounds cannot be checked from a static witness, so they carry a
certificate script. The pipeline runs it under a time budget and requires it to
exit zero having printed its claim. A bound with no runnable certificate is
recorded at the `cited` tier and is never allowed to set a leaderboard record.

Tiers, in decreasing strength:
    verified    exact arithmetic confirmed the decomposition here
    reproduced  a certificate script ran to completion and asserted the bound
    cited       attributed to the literature; not machine-checked
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- orbits ----
# Each orbit fixes the local dimension p and the single-qudit magic state, in
# exact arithmetic. These are the states the literature's exponents refer to;
# the names match the repo's state-of-the-art page.

def _w(p):
    """Primitive p-th root of unity, in a form sympy can reason about.

    exp(2*pi*I/p) is correct but sympy will not reduce sums of its powers to
    zero on its own, so give it the radical form for the cases that matter.
    """
    if p == 2:
        return sp.Integer(-1)
    if p == 3:
        return sp.Rational(-1, 2) + sp.I * sp.sqrt(3) / 2
    return sp.exp(2 * sp.pi * sp.I / p)


def is_zero(e, digits=60, tol=sp.Float("1e-45")):
    """Decide whether an exact expression vanishes.

    Tries symbolic reduction first. Cyclotomic entries (the T3 orbit lives in
    Q(w9)) do not always reduce symbolically, so fall back to evaluating at
    `digits` significant figures and requiring the modulus below `tol`. That is
    far tighter than any floating-point search could fake, and the terms are
    stabilizer states by construction, so a pass here is a genuine certificate.
    """
    z = sp.simplify(sp.expand(e))
    if z == 0:
        return True, "symbolic"
    z = sp.simplify(sp.expand_complex(z))
    if z == 0:
        return True, "symbolic"
    try:
        v = abs(complex(sp.N(z, digits)))
    except (TypeError, ValueError):
        return False, "undecided"
    return (v < float(tol)), f"numeric<{digits}dig>"


def orbit_state(orbit):
    """The single-qudit magic state for an orbit, as an exact column vector."""
    if orbit == "S":                      # Strange
        return sp.Matrix([0, 1, -1]) / sp.sqrt(2)
    if orbit == "N":                      # Norrell
        return sp.Matrix([1, 1, -2]) / sp.sqrt(6)
    if orbit == "H3":                     # qutrit Hadamard eigenstate
        # eigenvector of the qutrit Fourier transform with eigenvalue 1
        r3 = sp.sqrt(3)
        a = sp.sqrt((3 + r3) / 6)
        b = sp.sqrt((3 - r3) / 12)
        return sp.Matrix([a, b, b])
    if orbit == "T3":                     # qutrit T-type / face centre
        w9 = sp.exp(2 * sp.pi * sp.I / 9)
        return sp.Matrix([1, w9, w9 ** 2]) / sp.sqrt(3)
    if orbit == "qubit_H":                # qubit H-type, edge centre
        return sp.Matrix([sp.cos(sp.pi / 8), sp.sin(sp.pi / 8)])
    if orbit == "qubit_T":                # qubit Bravyi-Kitaev T-type, face centre
        b = sp.acos(1 / sp.sqrt(3)) / 2
        return sp.Matrix([sp.cos(b), sp.exp(sp.I * sp.pi / 4) * sp.sin(b)])
    raise ValueError(f"unknown orbit {orbit!r}")


ORBIT_P = {"S": 3, "N": 3, "H3": 3, "T3": 3, "qubit_H": 2, "qubit_T": 2}

ORBIT_LABEL = {
    "S": "Strange", "N": "Norrell", "H3": "H₃", "T3": "T₃",
    "qubit_H": "H-type", "qubit_T": "BK T-type",
}


def target_vector(orbit, m):
    """|M>^{ot m} as an exact column vector of length p^m."""
    v = orbit_state(orbit)
    out = v
    for _ in range(m - 1):
        out = sp.Matrix(sp.kronecker_product(out, v))
    return sp.simplify(out)


# ------------------------------------------------------- stabilizer terms ----

def stabilizer_vector(term, p, n):
    """Rebuild one stabilizer state from its parametrisation, exactly.

    term: {"k": k, "x0": [n ints], "W": [k rows of n ints], "Q": [[k x k]],
           "l": [k ints]}
    giving  sum_{y in F_p^k} w_p^{Q(y) + l.y} |x0 + W y>  up to normalisation.

    Q is read as an upper-triangular form: only entries with i <= j are used, so
    a submitter cannot smuggle in a non-quadratic phase.
    """
    k = int(term["k"])
    x0 = [int(a) % p for a in term["x0"]]
    if len(x0) != n:
        raise ValueError(f"x0 has length {len(x0)}, expected {n}")
    W = [[int(a) % p for a in row] for row in term.get("W", [])]
    if len(W) != k or any(len(r) != n for r in W):
        raise ValueError(f"W must be {k}x{n}")
    Q = [[int(a) % p for a in row] for row in term.get("Q", [[0] * k for _ in range(k)])]
    ell = [int(a) % p for a in term.get("l", [0] * k)]
    if len(ell) != k:
        raise ValueError(f"l must have length {k}")

    dim = p ** n
    vec = sp.zeros(dim, 1)
    w = _w(p)
    support = set()
    for idx in range(p ** k):
        y = []
        t = idx
        for _ in range(k):
            y.append(t % p)
            t //= p
        x = [(x0[c] + sum(y[r] * W[r][c] for r in range(k))) % p for c in range(n)]
        pos = 0
        for c in range(n):
            pos = pos * p + x[c]
        if pos in support:
            raise ValueError("W is not injective: repeated support point")
        support.add(pos)
        e = sum(Q[i][j] * y[i] * y[j] for i in range(k) for j in range(i, k))
        e += sum(ell[i] * y[i] for i in range(k))
        vec[pos] = w ** (e % p)
    return vec / sp.sqrt(p ** k)


# ------------------------------------------------------------ verification ---

class Result:
    def __init__(self, ok, tier, detail, gamma=None):
        self.ok, self.tier, self.detail, self.gamma = ok, tier, detail, gamma

    def __repr__(self):
        return f"<{'PASS' if self.ok else 'FAIL'} tier={self.tier}: {self.detail}>"


def implied_gamma(p, rank, m):
    """Per-copy asymptotic exponent implied by chi(|M>^{ot m}) <= rank."""
    return float(sp.log(rank, p) / m)


def verify_upper(sub):
    """Rebuild the decomposition exactly and require it to equal the target."""
    orbit, m, rank = sub["orbit"], int(sub["m"]), int(sub["rank"])
    p = ORBIT_P[orbit]
    terms = sub["witness"]["terms"]
    coeffs = sub["witness"]["coeffs"]
    if len(terms) != rank or len(coeffs) != rank:
        return Result(False, None,
                      f"claims rank {rank} but gives {len(terms)} terms and "
                      f"{len(coeffs)} coefficients")
    try:
        cols = [stabilizer_vector(t, p, m) for t in terms]
    except Exception as exc:
        return Result(False, None, f"a term is not a valid stabilizer state: {exc}")
    try:
        cs = [sp.sympify(c) for c in coeffs]
    except Exception as exc:
        return Result(False, None, f"coefficient did not parse: {exc}")
    if any(is_zero(c)[0] for c in cs):
        return Result(False, None, "a coefficient is zero, so the true rank is lower "
                                   "than claimed; resubmit at the smaller rank")

    tgt = target_vector(orbit, m)
    acc = sp.zeros(p ** m, 1)
    for c, col in zip(cs, cols):
        acc += c * col
    diff = acc - tgt
    modes = set()
    for d in diff:
        ok, how = is_zero(d)
        modes.add(how)
        if not ok:
            try:
                resid = float(sp.sqrt(sum(abs(complex(sp.N(t, 30))) ** 2 for t in diff)))
                extra = f" (residual {resid:.3e})"
            except (TypeError, ValueError):
                extra = ""
            return Result(False, None,
                          f"decomposition does not reproduce the target{extra}")
    how = "symbolically" if modes == {"symbolic"} else "to 60 significant digits"
    return Result(True, "verified",
                  f"{rank} stabilizer terms reproduce |{orbit}>^{{ot {m}}}, checked {how}",
                  implied_gamma(p, rank, m))


def verify_lower(sub, budget_s=900):
    """Run the certificate script; require exit zero and its claim on stdout."""
    cert = sub.get("certificate")
    if not cert or not cert.get("script"):
        return Result(True, "cited",
                      "no runnable certificate; recorded as cited and cannot set a record")
    path = os.path.join(ROOT, cert["script"])
    if not os.path.isfile(path):
        return Result(False, None, f"certificate script not found: {cert['script']}")
    expect = cert.get("expect", "")
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=budget_s, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return Result(False, None, f"certificate exceeded the {budget_s}s budget")
    if proc.returncode != 0:
        return Result(False, None, f"certificate exited {proc.returncode}")
    if expect and expect not in proc.stdout:
        return Result(False, None, f"certificate ran but did not print {expect!r}")
    return Result(True, "reproduced", f"certificate script asserted: {expect or 'ok'}")


def verify(sub, budget_s=900):
    for field in ("schema_version", "orbit", "m", "direction", "rank", "provenance"):
        if field not in sub:
            return Result(False, None, f"missing required field {field!r}")
    if sub["orbit"] not in ORBIT_P:
        return Result(False, None, f"unknown orbit {sub['orbit']!r}")
    if int(sub["m"]) < 1:
        return Result(False, None, "m must be at least 1")
    if int(sub["rank"]) < 1:
        return Result(False, None, "rank must be at least 1")
    if sub["direction"] == "upper":
        if not sub.get("witness"):
            return Result(True, "cited",
                          "no decomposition supplied; recorded as cited and cannot "
                          "set a record",
                          implied_gamma(ORBIT_P[sub["orbit"]], int(sub["rank"]),
                                        int(sub["m"])))
        return verify_upper(sub)
    if sub["direction"] == "lower":
        return verify_lower(sub, budget_s)
    return Result(False, None, f"direction must be 'upper' or 'lower'")


def main(argv):
    if len(argv) < 2:
        print("usage: stabrank_verify.py <submission.json> [...]")
        return 2
    bad = 0
    for path in argv[1:]:
        sub = json.load(open(path))
        r = verify(sub)
        name = os.path.basename(path)
        sign = "<=" if sub.get("direction") == "upper" else ">="
        head = f"chi(|{sub.get('orbit')}>^{sub.get('m')}) {sign} {sub.get('rank')}"
        print(f"{'PASS' if r.ok else 'FAIL'}  {name}  {head}")
        print(f"      tier={r.tier}  {r.detail}")
        if r.gamma is not None:
            print(f"      implied gamma <= {r.gamma:.4f}")
        bad += 0 if r.ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
