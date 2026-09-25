"""Magic code states of QPG's Theorem 4 at every prime on the board.

Qassim, Pashayan, and Gosset (arXiv:2106.07740, Theorem 4) define, for a
linear [m, k] code L over F_2, the code state |L_hat> = 2^{-k/2} sum_{x in L}
|x_hat> with |0_hat> = |T>, |1_hat> = |T_perp> = Z|T>, and prove that if
chi(T^n) has an exponent gamma then gamma <= log_2 chi(L_hat) / (m - 2k)
for every k < m/2. The proof contracts an information set of l copies of
L_hat with <T|^{kl} (each product of the code kills every codeword but 0,
since <T|T_perp> = 0), which gives T^{l(m - k)} from chi(L_hat)^l chi(T^{kl})
stabilizer terms. Since sum_{x in L} Z^x is |L| times the projector onto the
span of the basis states of L^perp, |L_hat> is the restriction of |T>^m to
L^perp, and the Clifford that maps L^perp to |0>^k (x) F_2^{m-k} compresses
it to an (m - k)-qubit state whose rank is chi(L_hat). The repetition code
gives the cat states; k >= 2 is what this script scans.

The same construction and proof hold verbatim for any single-qudit state
|M> over F_p whose Z-translates Z^j |M> are orthonormal, that is, whose
amplitudes all have modulus p^{-1/2}: T3 and T5 qualify, and so does the
H-type qubit state in its T-basis representative (1, e^{i pi/4}) / sqrt 2.
N, H3, and S do not, in the Z basis or in any other Pauli eigenbasis (the
script checks all p + 1 of them), so the route does not apply to them in
this form. The two-translate state (|M>^m + |P M>^m) / sqrt 2 at odd p is
also checked: it is QPG's cat shape with only two of the p translates, and
the contraction lemma would need its m = 2 member to be a stabilizer state.

Codes are enumerated up to monomial equivalence (coordinate permutations and
scalings, both local Cliffords on the code state) as multisets of m points
of PG(k-1, p), zero columns allowed, up to PGL_k(p). For each code the
compressed state is built from a basis of L^perp, its rank is decided in
the (m-k)-qudit dictionary where that fits (rank 1, 2, 3, else "> 3"), else
bounded below by single-qudit slices and, with --anneal, searched at the
rank that would beat the baseline. The exponent log_p(chi) / (m - 2k) is
compared with the orbit's baseline.

Usage:
    code_states_p.py ORBIT M K [--rank3] [--dict5] [--anneal] [--anneal-max N]
                     [--chains 2] [--iters 2000] [--unbiased] [--twocat M]
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import alpha  # noqa: E402
from rank_exclusion import clifford_group, rank2_search, rank3_search  # noqa: E402
from sectors_p import ANNEAL_CFG, anneal, dict_for, is_stab  # noqa: E402
from stabrank_verify import ORBIT_P  # noqa: E402

BASE = {"S": math.log(2, 3) / 2, "N": math.log(4, 3) / 3, "H3": math.log(4, 3) / 3, "T3": 0.5,
        "qubit_H": math.log(3, 2) / 4, "qubit_T": math.log(3, 2) / 4, "T5": 0.5}


def rep_vector(orbit):
    """The single-qudit amplitudes used for the code state: the T-basis
    representative for qubit_H, the board's vector otherwise."""
    if orbit == "qubit_H":
        return np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
    return alpha(orbit)


def unbiased_bases(orbit):
    """Which Pauli eigenbases (Z, X, X Z^c) see |M> with all |<e|M>|^2 = 1/p."""
    p = ORBIT_P[orbit]
    a = alpha(orbit)
    w = np.exp(2j * np.pi / p)
    F = np.array([[w ** (j * k) for k in range(p)] for j in range(p)]) / np.sqrt(p)
    out = []
    bases = {"Z": np.eye(p)}
    bases["X"] = F
    for c in range(1, p):
        S = np.diag([w ** ((c * j * (j - 1) // 2) % p) if p != 2 else (1j) ** (c * j) for j in range(p)])
        bases[f"X Z^{c}"] = S @ F        # eigenvectors of S X S^-1
    for name, B in bases.items():
        probs = np.abs(B.conj().T @ a) ** 2
        if np.allclose(probs, 1 / p, atol=1e-9):
            out.append(name)
    return out


# ------------------------------------------------------------- codes -----

def projective_points(p, k):
    pts = []
    seen = set()
    for v in itertools.product(range(p), repeat=k):
        if not any(v):
            continue
        i = next(j for j in range(k) if v[j])
        inv = pow(v[i], -1, p)
        c = tuple((x * inv) % p for x in v)
        if c not in seen:
            seen.add(c)
            pts.append(c)
    return pts


def gl(p, k):
    for rows in itertools.product(itertools.product(range(p), repeat=k), repeat=k):
        M = np.array(rows, dtype=int)
        if rank_mod(M, p) == k:
            yield M


def rank_mod(M, p):
    M = M.copy() % p
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), -1, p)) % p
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
    return r


def code_classes(p, k, m):
    """Multisets of m points of PG(k-1, p) plus a zero point, spanning F_p^k,
    one per PGL_k(p) orbit, as generator matrices (k x m)."""
    pts = projective_points(p, k)
    idx = {c: i for i, c in enumerate(pts)}
    group = list(gl(p, k))
    perms = []
    for M in group:
        perm = []
        for c in pts:
            v = tuple(int(x) for x in (M @ np.array(c)) % p)
            i = next(j for j in range(k) if v[j])
            inv = pow(v[i], -1, p)
            perm.append(idx[tuple((x * inv) % p for x in v)])
        perms.append(tuple(perm))
    perms = sorted(set(perms))
    npts = len(pts)
    seen = set()
    out = []
    for mult in itertools.combinations_with_replacement(range(npts + 1), m):
        cnt = [0] * (npts + 1)
        for i in mult:
            cnt[i] += 1
        if cnt[npts] > 0 and sum(cnt[:npts]) < k:
            continue
        key = min(tuple(cnt[perm[i]] for i in range(npts)) + (cnt[npts],) for perm in perms)
        if key in seen:
            continue
        seen.add(key)
        cols = []
        for i in range(npts):
            cols += [pts[i]] * key[i]
        cols += [(0,) * k] * key[npts]
        G = np.array(cols, dtype=int).T
        if rank_mod(G, p) < k:
            continue
        out.append(G)
    return out


def nullspace_mod(G, p):
    """A basis of {x : G x = 0 mod p} as rows."""
    k, m = G.shape
    M = G.copy() % p
    pivots = []
    r = 0
    for c in range(m):
        piv = next((i for i in range(r, k) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), -1, p)) % p
        for i in range(k):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        pivots.append(c)
        r += 1
    free = [c for c in range(m) if c not in pivots]
    basis = []
    for f in free:
        x = np.zeros(m, dtype=int)
        x[f] = 1
        for i, c in enumerate(pivots):
            x[c] = (-M[i, f]) % p
        basis.append(x)
    return np.array(basis, dtype=int)


def compressed_state(a, G, p):
    """The restriction of |M>^m to L^perp in the coordinates of a basis of
    L^perp: phi(y) = prod_i a[x_i] with x = y B."""
    k, m = G.shape
    B = nullspace_mod(G, p)
    n = m - k
    assert B.shape == (n, m)
    phi = np.zeros(p ** n, dtype=complex)
    for idx, y in enumerate(itertools.product(range(p), repeat=n)):
        x = (np.array(y) @ B) % p
        phi[idx] = np.prod(a[x])
    return phi / np.linalg.norm(phi)


def slices(u, p, m):
    T = u.reshape((p,) * m)
    out = []
    for i in range(m):
        for x in range(p):
            s = np.take(T, x, axis=i).reshape(-1)
            nrm = np.linalg.norm(s)
            if nrm > 1e-9:
                out.append(s / nrm)
    return out


def exact_rank(u, p, n, D, rank3):
    if is_stab(u, p, n):
        return 1
    r2, _ = rank2_search(u, D)
    if r2 == "RANK1":
        return 1
    if r2:
        return 2
    if not rank3:
        return ">=3"
    r = rank3_search(u, D, workers=1)
    return 3 if r["found"] else ">3"


def lower_bound(u, p, n, cap, rank3, memo):
    if n <= cap:
        r = exact_rank(u, p, n, dict_for(p, n), rank3)
        if isinstance(r, int):
            return r, True, str(r)
        return {">=3": 3, ">3": 4}[r], False, r
    best = (1, "1")
    for s in slices(u, p, n):
        key = (np.round(s / s[np.flatnonzero(np.abs(s) > 1e-9)[0]], 6) + 0.0).tobytes()
        if key not in memo:
            v, _, txt = lower_bound(s, p, n - 1, cap, rank3, memo)
            memo[key] = (v, txt)
        v, txt = memo[key]
        if v > best[0]:
            best = (v, txt)
    return best[0], False, f">={best[0]} (slice {best[1]})"


def two_translate(orbit, m):
    """Rank facts for (|M>^m + |P M>^m)/sqrt 2 over the nonidentity Paulis P."""
    p = ORBIT_P[orbit]
    a = alpha(orbit)
    w = np.exp(2j * np.pi / p)
    X = np.roll(np.eye(p), 1, axis=0)
    Z = np.diag([w ** j for j in range(p)])
    cap = {2: 4, 3: 3, 5: 2}[p]
    seen = set()
    for ax in range(p):
        for cz in range(p):
            if ax == 0 and cz == 0:
                continue
            P = np.linalg.matrix_power(X, ax) @ np.linalg.matrix_power(Z, cz)
            b = P @ a
            if abs(abs(np.vdot(a, b)) - 1) < 1e-9:
                continue
            u = np.ones(1, dtype=complex)
            v = np.ones(1, dtype=complex)
            for _ in range(m):
                u = np.kron(u, a)
                v = np.kron(v, b)
            c = u + v
            c /= np.linalg.norm(c)
            key = (np.round(np.abs(c), 6)).tobytes()
            if key in seen:
                continue
            seen.add(key)
            memo = {}
            r, ex, txt = lower_bound(c, p, m, cap, True, memo)
            print(f"  {orbit} two-translate cat with P = X^{ax} Z^{cz} at m={m}: |<M|PM>| = "
                  f"{abs(np.vdot(a, b)):.3f}, rank {txt}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit", nargs="?")
    ap.add_argument("m", nargs="?", type=int)
    ap.add_argument("k", nargs="?", type=int)
    ap.add_argument("--rank3", action="store_true")
    ap.add_argument("--dict5", action="store_true")
    ap.add_argument("--anneal", action="store_true", help="anneal undecided codes at the record rank")
    ap.add_argument("--anneal-rank", type=int, default=None, help="override the anneal rank")
    ap.add_argument("--anneal-max", type=int, default=20)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--unbiased", action="store_true", help="report the unbiased Pauli bases per orbit")
    ap.add_argument("--twocat", type=int, default=None, help="two-translate cat ranks at this m")
    a = ap.parse_args(argv[1:])
    if a.unbiased:
        for orbit in ("S", "N", "H3", "T3", "T5", "qubit_H", "qubit_T"):
            print(f"  {orbit}: unbiased Pauli eigenbases {unbiased_bases(orbit) or 'none'}")
        return
    if a.twocat:
        for orbit in ("S", "N", "H3", "T3", "T5"):
            two_translate(orbit, a.twocat)
        return
    ANNEAL_CFG.update(seeds=1, chains=a.chains, iters=a.iters, cooling=0.99)
    orbit, m, k = a.orbit, a.m, a.k
    p = ORBIT_P[orbit]
    base = BASE[orbit]
    vec = rep_vector(orbit)
    assert np.allclose(np.abs(vec) ** 2, 1 / p), "the representative is not Z-unbiased"
    cap = {2: 5 if a.dict5 else 4, 3: 3, 5: 2}[p]
    n = m - k
    denom = m - 2 * k
    assert denom >= 1
    rec = math.floor(p ** (base * denom) - 1e-9)
    if abs(p ** (base * denom) - rec) < 1e-9:
        rec -= 1
    t0 = time.time()
    codes = code_classes(p, k, m)
    print(f"{orbit} [{m}, {k}]_{p}: {len(codes)} codes up to monomial equivalence; compressed "
          f"state on {n} qudits; record needs chi(L_hat) <= {rec} (exponent < {base:.4f}); "
          f"tie at chi = {p ** (base * denom):.3f}")
    memo = {}
    annealed = 0
    best = None
    for G in codes:
        phi = compressed_state(vec, G, p)
        v, ex, txt = lower_bound(phi, p, n, cap, a.rank3, memo)
        cols = ["".join(str(int(x)) for x in G[:, j]) for j in range(m)]
        up = ""
        R = a.anneal_rank or rec
        if a.anneal and not ex and v <= R and annealed < a.anneal_max:
            annealed += 1
            err, secs_, _ = anneal(phi, p, n, R)
            up = f"; anneal r={R}: {'HIT' if err < 1e-7 else f'{err:.3f}'} ({secs_:.0f}s)"
        g = math.log(v, p) / denom if v > 0 else 0
        print(f"  cols {' '.join(cols)}: rank {txt} -> exponent {'=' if ex else '>='} {g:.4f}{up}")
        if best is None or v < best[0]:
            best = (v, ex, cols)
    if best:
        print(f"smallest code-state rank {'=' if best[1] else '>='} {best[0]} "
              f"(exponent {'=' if best[1] else '>='} {math.log(best[0], p) / denom:.4f}); {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv)
