"""The rank-5 decomposition of |T5>^2 that the census found, worked out.

The census of research/t5q_m2_rank5 (docs/notes/t5q_m2_rank5_exclusion.md)
was built to exclude rank 5 and instead returned one 5-set, the dictionary
states [525, 563, 591, 619, 637] of batch 31. This script rebuilds that set
from the dictionary and shows what it is:

* the five states are the lines x + y = c of F_5^2, c = 0..4, parametrized
  by x, with the quadratic phase w^(3c x^2 - 3c^2 x); on such a line the
  cubic phase of |T5>^2 is quadratic, x^3 + (c - x)^3 = c^3 + 3c x^2 -
  3c^2 x, so the projection of |T5>^2 onto the eigenspace Z(x)Z = w^c is
  w^(c^3)/sqrt(5) times a stabilizer state (the Pauli eigensector
  construction of research/constructions/sectors.py with P = Z(x)Z);
* the identity w^(x^3 + y^3) = w^(c^3) w^(3c x^2 - 3c^2 x) on x + y = c is
  checked in Z[w] (an element is zero iff its five coefficients on 1, w,
  ..., w^4 agree, the zero test of cert_t5_m1_rank2.py) and numerically;
* the coefficients in the board's normalization are w^(c^3)/sqrt(5), and
  fit_coeffs.fit recovers them independently; the witness verifies
  symbolically;
* the 5-set is fixed by every generator of the unitary symmetry group of
  |T5>^2 (order 50) and by the antiunitary one, so its orbit is the set
  itself: with ranks 2 to 4 excluded and the census complete over orbits,
  it is the only rank-5 decomposition of |T5>^2 over stabilizer states;
* the five states share the stabilizer <Z(x)Z> (eigenphase w^c on the
  sector c) and nothing else among the 625 Weyl operators.

Usage (from the repository root):
    uv run --extra challenge python research/t5q_m2_rank5/decomposition.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from fit_coeffs import fit  # noqa: E402
from rank_exclusion import dictionary, psi_for, symmetry_orbit_reps  # noqa: E402
from stabrank_verify import verify_upper  # noqa: E402
from to_witness import term_from_vector  # noqa: E402

P = 5
HIT = [525, 563, 591, 619, 637]          # research/t5q_m2_rank5/results/batch_31.json
W5 = np.exp(2j * np.pi / P)


def sector_terms():
    """The five Z(x)Z eigensectors as witness terms: line x + y = c through
    (0, c) with direction (1, -1), phase 3c x^2 - 3c^2 x in the parameter x."""
    return [{"k": 1, "x0": [0, c], "W": [[1, P - 1]], "Q": [[(3 * c) % P]],
             "l": [(-3 * c * c) % P]} for c in range(P)]


def identity_in_Zw():
    """w^(x^3 + y^3) = w^(c^3 + 3c x^2 - 3c^2 x) on x + y = c, decided in Z[w]."""
    for x in range(P):
        for y in range(P):
            c = (x + y) % P
            acc = [0] * P
            acc[(x ** 3 + y ** 3) % P] += 1
            acc[(c ** 3 + 3 * c * x * x - 3 * c * c * x) % P] -= 1
            if len(set(acc)) != 1:
                return False
    return True


def weyl_stabilizers(v):
    """The two-ququint Weyl operators X^a1 Z^b1 (x) X^a2 Z^b2 fixing v up to a
    phase, with the phase exponent."""
    def w1(a, b):
        return np.roll(np.eye(P), a, axis=0) @ np.diag([W5 ** ((b * j) % P) for j in range(P)])
    out = {}
    for a1, b1, a2, b2 in itertools.product(range(P), repeat=4):
        ov = np.vdot(v, np.kron(w1(a1, b1), w1(a2, b2)) @ v)
        if abs(abs(ov) - 1) < 1e-9:
            out[(a1, b1, a2, b2)] = int(round(np.angle(ov) / (2 * np.pi / P))) % P
    return out


def main():
    D = dictionary(P, 2)
    psi = psi_for("T5", 2)
    terms = [term_from_vector(D[:, i], P, 2) for i in HIT]
    want = sector_terms()
    if terms != want:
        print("the hit's states are not the Z(x)Z sectors:", terms, file=sys.stderr)
        return 1
    print("the five states are the lines x + y = c (c = 0..4) with phase w^(3c x^2 - 3c^2 x):")
    for i, t in zip(HIT, terms):
        print(f"  {i}: {t}")
    if not identity_in_Zw():
        print("the Z[w] identity fails", file=sys.stderr)
        return 1
    print("Z[w]: w^(x^3 + y^3) = w^(c^3) w^(3c x^2 - 3c^2 x) on every line x + y = c")

    w = sp.exp(2 * sp.pi * sp.I / P)
    coeffs = [w ** ((c ** 3) % P) / sp.sqrt(P) for c in range(P)]
    fitted = fit("T5", 2, terms)
    if fitted is None or any(abs(complex(sp.N(a - b, 30))) > 1e-25 for a, b in zip(coeffs, fitted)):
        print("fit_coeffs.fit disagrees with w^(c^3)/sqrt 5", file=sys.stderr)
        return 1
    print("coefficients w^(c^3)/sqrt(5) (c = 0..4), recovered independently by fit_coeffs.fit")
    r = verify_upper({"orbit": "T5", "m": 2, "rank": 5,
                      "witness": {"terms": terms, "coeffs": [str(c) for c in coeffs]}})
    print("verify_upper:", r)
    if not r.ok:
        return 1
    A = D[:, HIT]
    c_num, *_ = np.linalg.lstsq(A, psi, rcond=None)
    print(f"numerical residual {np.linalg.norm(A @ c_num - psi):.2e}, rank {np.linalg.matrix_rank(A)}")

    start = tuple(sorted(HIT))
    for anti in (False, True):
        _, info = symmetry_orbit_reps("T5", 2, D, antiunitary=anti)
        orbit = {start}
        frontier = [start]
        while frontier:
            nxt = []
            for s in frontier:
                for pm in info["perms"]:
                    t = tuple(sorted(int(pm[x]) for x in s))
                    if t not in orbit:
                        orbit.add(t)
                        nxt.append(t)
            frontier = nxt
        print(f"orbit of the 5-set under the symmetry group of order {info['order']} "
              f"(antiunitary={anti}): {len(orbit)} set(s)")
        for gi, pm in enumerate(info["perms"]):
            print(f"  generator {gi}: {list(start)} -> {[int(pm[x]) for x in start]}")

    stabs = [weyl_stabilizers(D[:, i]) for i in HIT]
    common = set(stabs[0])
    for s in stabs[1:]:
        common &= set(s)
    print("Weyl stabilizer sizes", [len(s) for s in stabs], "common:",
          sorted(common), "with eigenphase exponents",
          {k: [s[k] for s in stabs] for k in sorted(common) if any(k)})
    for c in range(P):
        ZZ = np.kron(np.diag(W5 ** np.arange(P)), np.diag(W5 ** np.arange(P)))
        proj = sum(W5 ** (-c * j) * np.linalg.matrix_power(ZZ, j) for j in range(P)) / P
        v = proj @ psi
        ov = abs(np.vdot(D[:, HIT[c]], v)) / np.linalg.norm(v)
        print(f"sector Z(x)Z = w^{c}: |Pi psi| = {np.linalg.norm(v):.4f} = 1/sqrt 5, "
              f"overlap with state {HIT[c]}: {ov:.6f}")
    print("chi(T5^2) = 5: the five Z(x)Z eigensectors of |T5>^2 are stabilizer states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
