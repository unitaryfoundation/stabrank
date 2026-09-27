"""The single-orbit shape under a cyclic shift of the copies, decided at the
constant points.

Setting (2026-09-26 section of `docs/notes/constructions_2026_09.md`): sigma
is the m-cycle on the copies, and a decomposition of |M>^m whose term set is
sigma-invariant splits into sigma-orbits of size 1 and, at prime m, of size
m. The all-fixed shape is what `cyclic_symmetric.py` enumerates. The shape
left over at R = m is a single orbit {s, sigma s, ..., sigma^(m-1) s}. There
sigma-invariance of the target and independence of the terms force the m
coefficients equal, so

    |M>^m = m d Pi s,     Pi = (1/m) sum_j sigma^j,

and Pi s is the orbit average of the amplitude function of s.

The constant points settle this without a search. The fixed points of sigma
on F_p^m are the p constant vectors c 1 = (c, ..., c), and they are exactly
the diagonal line L = {c 1 : c in F_p}, a one-dimensional subspace. At a
fixed point the orbit average is the value itself, so

    s(c 1) = <c 1 | M>^m / (m d)     for every c in F_p.

Two facts about a stabilizer state s now decide the shape:

  (i)  supp(s) is an affine flat, so it meets the line L in 0, in 1, or in
       all p of its points;
  (ii) the nonzero amplitudes of s all have the same modulus.

For a tensor power every constant point carries the amplitude alpha_c^m, and
a magic state has at least two nonzero amplitudes, so |M>^m is nonzero at at
least two constant points and (i) forces all p of them into supp(s). Then
a constant point where |M>^m vanishes is a contradiction outright, and
otherwise (ii) forces

    |<0 1|M>^m| = |<1 1|M>^m| = ... = |<(p-1) 1|M>^m|,

which for a tensor power is |alpha_0| = ... = |alpha_{p-1}|: the single-copy
state must be unbiased in the computational basis. It does not depend on m.
A family track such as cat, where the target is nonzero at only one constant
point, escapes the test: there the line condition is vacuous.

The shape needs R = m, since the orbit has exactly m terms. Of the
record-capable cells only S m=5 has its record rank equal to m, so that is
the only cell where the verdict bites.

This script prints the constant-point moduli and the verdict for every board
orbit, and checks the two facts it rests on numerically.

Usage: single_orbit.py [--control]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import alpha  # noqa: E402

from stabrank_verify import FAMILY, ORBIT_P, target_vector  # noqa: E402

ORBITS = [("S", 5), ("N", 5), ("H3", 5), ("T3", 5), ("qubit_H", 7),
          ("qubit_T", 7), ("T5", 3), ("cat", 7)]


def target_numeric(orbit, m):
    if orbit in FAMILY:
        v = np.array([complex(x) for x in target_vector(orbit, m)]).ravel()
        return v / np.linalg.norm(v)
    a = alpha(orbit)
    a = a / np.linalg.norm(a)
    v = a
    for _ in range(m - 1):
        v = np.kron(v, a)
    return v


def constant_points(p, m):
    """The indices of the p constant vectors c 1 in F_p^m."""
    step = sum(p ** j for j in range(m))
    return [c * step for c in range(p)]


def symmetrize(v, p, m):
    """(1/m) sum_j sigma^j v for the m-cycle sigma on the copies."""
    base = np.arange(p ** m).reshape((p,) * m)
    out = np.zeros_like(v)
    pi = tuple(range(1, m)) + (0,)
    q = tuple(range(m))
    for _ in range(m):
        out = out + v[np.transpose(base, q).ravel()]
        q = tuple(q[pi[i]] for i in range(m))
    return out / m


def report():
    print("%-9s %-3s %-34s %s" % ("orbit", "m", "|<c 1|M>^m| for c in F_p", "single m-orbit"))
    for orbit, m in ORBITS:
        p = ORBIT_P[orbit]
        psi = target_numeric(orbit, m)
        vals = [abs(psi[i]) for i in constant_points(p, m)]
        nz = [v for v in vals if v > 1e-12]
        if len(nz) < 2:
            verdict = "not excluded (only %d constant point carries amplitude, so the line condition is vacuous)" % len(nz)
        elif len(nz) < p:
            verdict = "impossible (a constant point has zero amplitude, the line forces it into supp)"
        elif max(nz) - min(nz) > 1e-9 * max(nz):
            verdict = "impossible (the constant-point moduli differ, ratio %.4f)" % (max(nz) / min(nz))
        else:
            verdict = "not excluded (the moduli agree)"
        print("%-9s %-3d %-34s %s" % (orbit, m, " ".join("%.6f" % v for v in vals), verdict))


def random_stabilizer(p, n, rng):
    """A uniform-ish stabilizer state as (support flat, phases), built from a
    random affine flat and a random quadratic phase, in the parametrization
    of verify_challenge/to_witness.py."""
    k = rng.integers(0, n + 1)
    W = rng.integers(0, p, size=(k, n))
    x0 = rng.integers(0, p, size=n)
    Q = rng.integers(0, p, size=(k, k))
    ell = rng.integers(0, p, size=k)
    w = np.exp(2j * np.pi / p)
    v = np.zeros(p ** n, dtype=complex)
    weights = p ** (n - 1 - np.arange(n))
    for y in itertools.product(range(p), repeat=int(k)):
        y = np.array(y, dtype=np.int64)
        x = (x0 + y @ W) % p
        q = sum(int(Q[i][j]) * int(y[i]) * int(y[j]) for i in range(int(k)) for j in range(i, int(k)))
        v[int(x @ weights)] = w ** ((q + int(ell @ y)) % p)
    return v / np.linalg.norm(v)


def control():
    ok = True

    def show(name, good, detail=""):
        nonlocal ok
        ok = ok and good
        print("  %-62s %s %s" % (name, "ok" if good else "FAIL", detail))

    rng = np.random.default_rng(20260927)
    p, n = 3, 5
    pts = constant_points(p, n)
    met2, moduli_ok, flat_ok = 0, True, True
    for _ in range(400):
        v = random_stabilizer(p, n, rng)
        on = [i for i in pts if abs(v[i]) > 1e-12]
        if len(on) >= 2:
            met2 += 1
            if len(on) != p:
                flat_ok = False
            vals = [abs(v[i]) for i in on]
            if max(vals) - min(vals) > 1e-9:
                moduli_ok = False
    show("a flat meeting the diagonal line twice contains all of it", flat_ok,
         "(%d of 400 random states met it twice)" % met2)
    show("the amplitudes on the line have one modulus", moduli_ok)

    # the symmetrizer really is the orbit average and fixes the target
    for orbit, m in (("S", 5), ("T3", 5), ("qubit_H", 7)):
        p = ORBIT_P[orbit]
        psi = target_numeric(orbit, m)
        show("sigma fixes |%s>^%d" % (orbit, m),
             np.linalg.norm(symmetrize(psi, p, m) - psi) < 1e-9)
    p, n = 3, 5
    v = random_stabilizer(p, n, rng)
    s = symmetrize(v, p, n)
    show("Pi s is sigma invariant", np.linalg.norm(symmetrize(s, p, n) - s) < 1e-9)
    show("Pi s agrees with s at the constant points",
         all(abs(s[i] - v[i]) < 1e-9 for i in constant_points(p, n)))

    # |S>^5: the premise of the argument, read off the target
    psi = target_numeric("S", 5)
    vals = [abs(psi[i]) for i in constant_points(3, 5)]
    show("|S>^5 vanishes at 00000 and not at 11111 or 22222",
         vals[0] < 1e-12 and vals[1] > 1e-6 and vals[2] > 1e-6,
         "(%s)" % " ".join("%.4f" % x for x in vals))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv[1:])
    if a.control:
        return control()
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
