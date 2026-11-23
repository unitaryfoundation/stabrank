"""Exact certificate: chi(cat_7) >= 6.

Printed claim (when every check passes):
    CERTIFIED chi(cat_7) >= 6

Together with bounds/cat-m7-upper-6.json this gives chi(cat_7) = 6. Integer arithmetic throughout: vectors over
Z[i] as int64 pairs, every rank decided modulo split primes P1, P2 just below
2^31 (verify_challenge/pauli_vocabulary.py). No floating-point number is
created anywhere.

1. Target. c_7(y) = cat_7(y, par(y)) = i^{(|y| + par(y))/2} on six qubits is the
   image of |cat_7> under a CNOT ladder, so chi(cat_7) = chi(c_7). The
   compression is checked symbolically against stabrank_verify.target_vector.

2. Projection lemma (Wstate / tclass note). If c_7 = sum_{j<=5} c_j s_j and the
   signed Pauli g stabilises s_1, then (1 - g) c_7 is a sum of four stabilizer
   states. So every non-identity element of the signed group of every term lies
   in the cheap set {g : chi((1 - g) c_7) <= 4}, every term lies in the
   vocabulary (states whose whole group is cheap), two terms can share only g
   with chi((1 - g) c_7) <= 3, and three terms only g with chi <= 2.

3. Symmetry. (01), the 6-cycle, the CNOT map that swaps qubit 0 with the parity
   qubit (together S_7 on the seven cat qubits), J_0 J_1 with J = X S^dagger,
   and the antiunitary K = S^{(x)7} o conj all fix c_7 up to a scalar
   (asserted). Their action on the 8190 non-identity signed Paulis has 12 orbits,
   of sizes 14, 42, 70, 84, 84, 448, 448, 560, 560, 840, 1680, 3360.

4. Projected ranks, one orbit representative each, by the same exclusion one
   qubit down (and recursively to three qubits, where the full list of 1080
   stabilizer states is used): chi((1 - g) c_7) >= 5 on five classes (6524
   Paulis), >= 4 on four (1526) and >= 3 on three (140). The stabilizer of g in
   the symmetry group, through Schreier generators transported to the 5-qubit
   picture, reduces the 2046 five-qubit projections to 13 to 88 orbits.

5. Top level. Ranks 2 and 3 have empty cheap sets; for rank 4 the 140 cheap
   Paulis contain no stabilizer group. For rank 5 the cheap set has 1666
   elements and the vocabulary 1088 states (enumerated as all signed groups in
   the cheap set, each once). Pivot triples: an orbit representative of an
   ordered compatible pair under the symmetry group, then every compatible
   third state; the last two terms are found by sorting quotient images. No
   decomposition exists.

6. Controls. The same prover gives chi(c_5) = chi(c_6) = 3 exactly (it proves
   >= 3 and finds a decomposition when asked for >= 4) on 4 and 5 qubits, and a
   planted sum of five compatible vocabulary states on six qubits is excluded at
   rank 4 and found at rank 5 by the same search code.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import sympy as sp

import pauli_vocabulary as pv
from stabrank_verify import target_vector

N_QUBITS = 6
# projected-rank lower bound required for each orbit representative (min index)
TARGET = {2: 3, 6: 3, 128: 3, 14: 4, 896: 4, 3968: 4, 3969: 4,
          129: 5, 132: 5, 140: 5, 897: 5, 912: 5}
ORBIT_SIZES = [14, 42, 70, 84, 84, 448, 448, 560, 560, 840, 1680, 3360]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def c_vector(m: int):
    """c_m(y) = i^{(|y| + par(y))/2} on m - 1 qubits, as (re, im) int64 arrays."""
    n = m - 1
    y = np.arange(1 << n)
    e = ((pv.POP[y] + pv.PAR[y]) // 2) % 4
    return pv.times_i_pow(np.ones(1 << n, np.int64), np.zeros(1 << n, np.int64), e)


def check_target() -> None:
    cat7 = target_vector("cat", 7)
    re, im = c_vector(7)
    scale = sp.sqrt(2) ** 6
    for x in range(128):
        if bin(x).count("1") % 2:
            if sp.simplify(cat7[x]) != 0:
                fail(f"cat_7 has an odd-weight amplitude at {x}")
            continue
        y = x >> 1                              # x = (y, par(y)), parity bit last
        want = int(re[y]) + sp.I * int(im[y])
        if sp.simplify(scale * cat7[x] - want) != 0:
            fail(f"compressed c_7 mismatch at y={y}")


def symmetry_ops():
    n, x = N_QUBITS, np.arange(64)

    def bit(k):
        return (x >> k) & 1

    zero = 0 * x
    return [
        pv.MonomialOp(x ^ ((bit(0) ^ bit(1)) * 3), zero, name="(01)"),
        pv.MonomialOp(sum(bit(k) << ((k + 1) % n) for k in range(n)), zero, name="6-cycle"),
        pv.MonomialOp((x & ~1) | pv.PAR[x], zero, name="swap with parity qubit"),
        pv.MonomialOp(x ^ 3, 3 * (bit(0) + bit(1)), name="J_0 J_1"),
        pv.MonomialOp(x, pv.POP[x] + pv.PAR[x], conj=True, name="K"),
    ]


def controls(log) -> None:
    pr = pv.Prover()
    for m in (5, 6):
        re, im = c_vector(m)
        lb = pr.lower_bound(re, im, m - 1, 4)
        if lb != 3:
            fail(f"control: chi(c_{m}) bound {lb}, expected exactly 3")
    log("controls: chi(c_5) = chi(c_6) = 3 recovered (>= 3 proved, rank-3 decompositions found)")


def planted_control(V, lbg, log) -> None:
    """Five pairwise and triple-compatible, independent vocabulary states, summed
    with unit coefficients, among 40 others: rank 4 excluded, rank 5 found."""
    rng = np.random.default_rng(20261119)
    ok2 = pv.pair_compatibility(V.members, lbg <= 3)
    trip = lbg <= 2
    for _ in range(200):
        chosen = [int(rng.integers(V.m))]
        for t in rng.permutation(V.m):
            t = int(t)
            if t in chosen or not all(ok2[t, c] for c in chosen):
                continue
            if any(np.any(V.members[t] & V.members[a] & V.members[b] & ~trip)
                   for a in chosen for b in chosen if a < b):
                continue
            if pv.rank_mod([V.M1[c] for c in chosen + [t]], pv.P1) < len(chosen) + 1:
                continue
            chosen.append(t)
            if len(chosen) == 5:
                break
        if len(chosen) == 5:
            break
    else:
        fail("planted control: no five compatible states")
    coef = [0, 1, 2, 3, 0]                     # 1, i, -1, -i, 1
    pre = np.zeros(V.re.shape[1], np.int64)
    pim = np.zeros_like(pre)
    for c, e in zip(chosen, coef):
        r2, i2 = pv.times_i_pow(V.re[c], V.im[c], np.full(pre.shape, e))
        pre, pim = pre + r2, pim + i2
    others = [int(t) for t in rng.permutation(V.m) if int(t) not in chosen][:40]
    rows = np.array(sorted(chosen + others))
    sub = V.subset(rows)
    try:
        pv.exclude((pre, pim), sub, 4, lbg <= 2, lbg <= 1)
    except (pv.Found, pv.Inconsistent) as e:
        fail(f"planted control: rank 4 not excluded ({e})")
    try:
        pv.exclude((pre, pim), sub, 5, lbg <= 3, lbg <= 2)
    except pv.Found:
        log("planted control: rank 4 excluded, the planted rank-5 decomposition found")
        return
    fail("planted control: planted decomposition missed")


def main() -> int:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    check_target()
    log("target: c_7 is the compressed cat_7")
    pv.full_vocab(3)
    pv.full_vocab(4)
    log("stabilizer states: 1080 on 3 qubits, 36720 on 4 qubits (counts checked)")
    controls(log)

    n = N_QUBITS
    re, im = c_vector(7)
    sym = pv.Symmetry(symmetry_ops(), re, im, n)
    reps, sizes = np.unique(sym.orbit[2:], return_counts=True)
    if sorted(sizes.tolist()) != ORBIT_SIZES or set(reps.tolist()) != set(TARGET):
        fail(f"unexpected Pauli orbits {dict(zip(reps.tolist(), sizes.tolist()))}")
    log("symmetry: 5 generators fix c_7; 12 orbits on the 8190 signed Paulis")

    pr = pv.Prover()
    lbg = np.full(2 * 4 ** n, -1, dtype=np.int64)
    for g in reps.tolist():
        tr, ti = pv.kill_project(re, im, g, n)
        lb = pr.lower_bound(tr, ti, n - 1, TARGET[g], orbit=sym.sub_orbit(g))
        if lb != TARGET[g]:
            fail(f"class {g}: proved chi >= {lb}, needed {TARGET[g]}")
        lbg[sym.orbit == g] = lb
    lbg[:2] = -1
    count = {k: int(np.sum(lbg[2:] == k)) for k in (3, 4, 5)}
    log(f"projected ranks: chi >= 5 on {count[5]}, >= 4 on {count[4]}, >= 3 on {count[3]} "
        f"signed Paulis")
    if count != {3: 140, 4: 1526, 5: 6524}:
        fail(f"unexpected class counts {count}")

    if pv.is_stabilizer(re, im, n):
        fail("c_7 is a stabilizer state")
    for k in range(3, 7):
        r = k - 1
        cheap = lbg < r
        cheap[:2] = False
        V = pv.vocabulary(n, cheap)
        st = {}
        if V.m:
            vperms = sym.on_vocab(V)
            ok2 = pv.pair_compatibility(V.members, lbg <= r - 2)
            piv = pv.pivot_lists(vperms, V.m, r, ok2, V.members, lbg <= r - 3)
            if r == 5:
                planted_control(V, lbg, log)
            try:
                pv.exclude((re, im), V, r, lbg <= r - 2, lbg <= r - 3, pivot_list=piv, stats=st)
            except pv.Found as e:
                fail(f"rank {r} decomposition of c_7: {e}")
            except pv.Inconsistent as e:
                fail(f"inconsistent with rank {r - 1} exclusion: {e}")
            st["pivot tuples"] = len(piv)
        log(f"chi(c_7) >= {k}: cheap set {int(cheap.sum())}, vocabulary {V.m}, {st}")

    print("CERTIFIED chi(cat_7) >= 6")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
