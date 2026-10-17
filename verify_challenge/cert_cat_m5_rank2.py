"""Certificate: chi(|cat_m>) >= 3 for m = 5, 6, 7, 8, exact throughout.

|cat_m> = 2^{-(m-1)/2} sum_{|x| even} i^{|x|/2} |x>  (Qassim, Pashayan, and
Gosset, arXiv:2106.07740, Eq. 3; `stabrank_verify.cat_vector`).

Step 1, compression. |cat_m> lies in the stabilizer code Z^{(x) m} = +1 (its
support is the even-weight strings). If |cat_m> = sum_j c_j s_j with s_j
stabilizer states, apply the code projector P = (I + Z^{(x) m})/2: P s_j is
zero or proportional to a stabilizer state (projecting a stabilizer state by
(I + g)/2 for a Pauli g gives 0, the state itself, or a stabilizer state), so
|cat_m> = sum_j c_j P s_j has at most as many stabilizer terms, all inside the
code. The Clifford U = CNOT(1->m) ... CNOT(m-1->m) maps the code onto
C^{2^{m-1}} (x) |0>, so each code term becomes s'_j (x) |0> with s'_j an
(m-1)-qubit stabilizer state (slicing <0| on the last qubit). Hence
chi(|cat_m>) equals the rank of the compressed (m-1)-qubit state
c_m(y) = cat_m(y, parity(y)) over (m-1)-qubit stabilizer states.

Step 2, exact rank-2 exclusion for c_5 (4 qubits). All 36,720 four-qubit
stabilizer states are generated exactly as Gaussian-integer vectors (entries
in {0, 1, -1, i, -i} after dividing by the first nonzero entry, checked by
exact divisibility), by breadth-first closure under H, S and CNOT, with the
count 2^4 * 3 * 5 * 9 * 17 asserted. The target c_5 has unit entries
(fourth roots of unity). Fix a coordinate j0; the linear map
q(s) = s - (s_j0 / t_j0) t has kernel span{t} and keeps Gaussian integers
(t_j0 is a unit). For distinct rays s1, s2:
    t in span{s1, s2} with t not a multiple of s1 or s2
        <=>  q(s1), q(s2) nonzero and parallel over Q(i),
and q(s) = 0 <=> s is a multiple of t (rank 1). Parallel classes are found
by an exact projective key (divide by the first nonzero entry, as reduced
Gaussian rationals). No floating point is used anywhere in the decision.

Positive control in the same code path: c_4 (cat_4 compressed to 3 qubits,
1080 states) must show at least one parallel class, since chi(cat_4) = 2
(bounds/cat-m4-upper-2.json); the script exits non-zero otherwise. A second
control checks that rank 1 is detected: a stabilizer state used as target
must be flagged RANK1.

Step 3, monotonicity. (I (x) <0|) |cat_m> = 2^{-1/2} |cat_{m-1}> exactly
(checked here on the exact exponent arrays for m = 6, 7, 8 and against
stabrank_verify.cat_vector), and <0| on one qubit maps a stabilizer state to
a stabilizer state or zero, so chi(cat_m) >= chi(cat_{m-1}); the same
projection lemma as verify_challenge/cert_s_m4_from_m3.py.

Printed claims: CERTIFIED chi(cat_5) >= 3
                CERTIFIED chi(cat_6) >= 3
                CERTIFIED chi(cat_7) >= 3
                CERTIFIED chi(cat_8) >= 3
"""

from __future__ import annotations

import os
import sys
import time
from math import gcd

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fail(msg):
    print(f"FAILED: {msg}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------- exact Gaussian vectors -----
# A vector is a pair (re, im) of int64 arrays of length 2^n, MSB-first indexing.

def normalize(re, im):
    """Divide by the first nonzero entry; every ratio must be a unit."""
    nz = np.flatnonzero((re != 0) | (im != 0))
    a, b = int(re[nz[0]]), int(im[nz[0]])
    d = a * a + b * b
    # z / z0 = z * conj(z0) / |z0|^2
    nr = re * a + im * b
    ni = im * a - re * b
    if (nr % d).any() or (ni % d).any():
        fail("non-unit amplitude ratio in a stabilizer state")
    nr //= d
    ni //= d
    if not np.all((np.abs(nr) + np.abs(ni) <= 1)):
        fail("amplitude ratio outside {0, +-1, +-i}")
    return nr, ni


def key(re, im):
    return re.astype(np.int8).tobytes() + im.astype(np.int8).tobytes()


def stabilizer_states(n):
    dim = 1 << n
    idx = np.arange(dim)
    gates = []
    for j in range(n):
        bit = 1 << (n - 1 - j)
        lo = idx[(idx & bit) == 0]
        hi = lo | bit
        gates.append(("H", lo, hi))
        gates.append(("S", hi, None))
    for c in range(n):
        for t in range(n):
            if c != t:
                cb, tb = 1 << (n - 1 - c), 1 << (n - 1 - t)
                perm = np.where(idx & cb, idx ^ tb, idx)
                gates.append(("C", perm, None))

    def apply(g, re, im):
        kind, a, b = g
        re, im = re.copy(), im.copy()
        if kind == "H":
            r0, i0, r1, i1 = re[a].copy(), im[a].copy(), re[b].copy(), im[b].copy()
            re[a], im[a], re[b], im[b] = r0 + r1, i0 + i1, r0 - r1, i0 - i1
        elif kind == "S":
            re[a], im[a] = -im[a], re[a].copy()
        else:
            re, im = re[a], im[a]
        return normalize(re, im)

    re0 = np.zeros(dim, np.int64); re0[0] = 1
    im0 = np.zeros(dim, np.int64)
    seen = {key(re0, im0): (re0, im0)}
    frontier = [(re0, im0)]
    while frontier:
        nxt = []
        for re, im in frontier:
            for g in gates:
                r2, i2 = apply(g, re, im)
                k = key(r2, i2)
                if k not in seen:
                    seen[k] = (r2, i2)
                    nxt.append((r2, i2))
        frontier = nxt
    expect = 1 << n
    for k in range(1, n + 1):
        expect *= (1 << k) + 1
    if len(seen) != expect:
        fail(f"{len(seen)} stabilizer states on {n} qubits, expected {expect}")
    R = np.stack([v[0] for v in seen.values()])
    I = np.stack([v[1] for v in seen.values()])
    return R, I


# ---------------------------------------------------------- the target -----

def cat_exponents(m):
    """e[x] = |x|/2 mod 4 on even-weight x, -1 (amplitude 0) on odd weight;
    the amplitude is i^e (times the common scale 2^{-(m-1)/2})."""
    x = np.arange(1 << m)
    w = np.array([bin(v).count("1") for v in x])
    return np.where(w % 2 == 0, (w // 2) % 4, -1)


def compressed_cat(m):
    """c_m(y) = cat_m(y, parity(y)) as an exact Gaussian vector on m-1 qubits."""
    e = cat_exponents(m)
    n = m - 1
    re = np.zeros(1 << n, np.int64); im = np.zeros(1 << n, np.int64)
    for y in range(1 << n):
        x = (y << 1) | (bin(y).count("1") & 1)
        k = e[x]
        if k < 0:
            fail("compressed point off the even-weight support")
        re[y], im[y] = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    return re, im


def check_against_verifier(m):
    """The exponent closed form agrees with stabrank_verify.cat_vector exactly."""
    import sympy as sp
    from stabrank_verify import cat_vector
    v = cat_vector(m)
    scale = sp.Integer(2) ** sp.Rational(-(m - 1), 2)
    e = cat_exponents(m)
    for x in range(1 << m):
        want = 0 if e[x] < 0 else scale * sp.I ** int(e[x])
        if sp.simplify(v[x] - want) != 0:
            fail(f"closed form disagrees with cat_vector({m}) at x={x}")


# ------------------------------------------------ exact rank-2 exclusion ---

def rank2_classes(tre, tim, R, I):
    """Exact: returns ("RANK1", idx) or the list of parallel classes of size >= 2
    among the quotients q(s) = s - (s_j0/t_j0) t."""
    j0 = 0
    a, b = int(tre[j0]), int(tim[j0])
    if a * a + b * b != 1:
        fail("target entry at j0 is not a unit")
    # s_j0 / t_j0 = s_j0 * conj(t_j0)   (t_j0 a unit)
    cr = R[:, j0] * a + I[:, j0] * b
    ci = I[:, j0] * a - R[:, j0] * b
    # q = s - c t  (complex multiply c * t)
    qr = R - (cr[:, None] * tre[None, :] - ci[:, None] * tim[None, :])
    qi = I - (cr[:, None] * tim[None, :] + ci[:, None] * tre[None, :])
    zero = ~((qr != 0) | (qi != 0)).any(axis=1)
    if zero.any():
        return "RANK1", np.flatnonzero(zero)
    classes = {}
    for s in range(R.shape[0]):
        r, i = qr[s], qi[s]
        k = int(np.flatnonzero((r != 0) | (i != 0))[0])
        p, q = int(r[k]), int(i[k])
        d = p * p + q * q
        nr = r * p + i * q          # (r + i i) * conj(p + q i)
        ni = i * p - r * q
        g = d
        for v in np.concatenate([nr, ni]):
            g = gcd(g, int(v))
        kk = (d // g, (nr // g).tobytes(), (ni // g).tobytes())
        classes.setdefault(kk, []).append(s)
    return [c for c in classes.values() if len(c) >= 2]


def exclude_rank2_cat5(verbose=True):
    """Controls, then the exact rank-2 exclusion for compressed cat_5.
    Returns normally only if chi(cat_5) >= 3 is established; fails loudly
    (exit 1) otherwise. Reused by cert_cat_m7_lift.py for its premise."""
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    for m in (4, 5, 6):
        check_against_verifier(m)
    say("closed form of |cat_m> matches stabrank_verify.cat_vector exactly for m = 4, 5, 6")

    # positive control: cat_4 compressed to 3 qubits has rank 2
    R3, I3 = stabilizer_states(3)
    tre, tim = compressed_cat(4)
    res = rank2_classes(tre, tim, R3, I3)
    if isinstance(res, tuple) or not res:
        fail("positive control: no rank-2 pair for compressed cat_4, where chi(cat_4) = 2")
    say(f"positive control: compressed cat_4 (3 qubits, {R3.shape[0]} states) has "
        f"{len(res)} parallel class(es), i.e. rank-2 decompositions, as expected")

    R4, I4 = stabilizer_states(4)
    say(f"4-qubit stabilizer states generated exactly: {R4.shape[0]} ({time.time() - t0:.1f}s)")

    # rank-1 control: a stabilizer state as target must be flagged RANK1
    res1 = rank2_classes(R4[1234], I4[1234], R4, I4)
    if not (isinstance(res1, tuple) and res1[0] == "RANK1"):
        fail("rank-1 control: a stabilizer target was not flagged")
    say("rank-1 control: a stabilizer target is flagged, as expected")

    tre, tim = compressed_cat(5)
    res = rank2_classes(tre, tim, R4, I4)
    if isinstance(res, tuple):
        fail(f"compressed cat_5 is a stabilizer state? indices {res[1][:5]}")
    if res:
        fail(f"compressed cat_5 has {len(res)} rank-2 decompositions")
    say("compressed cat_5: no stabilizer state is a multiple of it, and no two "
        "quotients are parallel over Q(i): rank >= 3, exact")


def check_slice_identity(m):
    """(I (x) <0|) cat_m = 2^{-1/2} cat_{m-1}, on the exact exponent arrays."""
    e, e1 = cat_exponents(m), cat_exponents(m - 1)
    if not np.array_equal(e[0::2], e1):    # last qubit (least significant bit) = 0
        fail(f"<0| slice of cat_{m} is not cat_{m - 1}")


def main():
    t0 = time.time()
    exclude_rank2_cat5()
    print("CERTIFIED chi(cat_5) >= 3")
    # monotonicity: (I (x) <0|) cat_m = 2^{-1/2} cat_{m-1}
    for m in (6, 7, 8):
        check_slice_identity(m)
        print(f"(I (x) <0|) cat_{m} = 2^(-1/2) cat_{m - 1} exactly; projection monotonicity")
        print(f"CERTIFIED chi(cat_{m}) >= 3")
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
