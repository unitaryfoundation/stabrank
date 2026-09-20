"""Pauli eigensector decompositions of |M>^m and the ranks of the sectors.

For a Pauli string P = P_1 (x) ... (x) P_m with every P_i nonidentity
(rescaled so P^3 = I), |M>^m = sum_s Pi_s |M>^m with Pi_s the projector on the
eigenvalue w^s, so chi(|M>^m) <= sum_s chi(Pi_s |M>^m). Each sector state is
(1/3) sum_j w^{-sj} (P^j |M>)^(x m), a sum of three product states inside the
stabilizer code {P = w^s}. Any stabilizer decomposition of a sector state can
be projected into the code term by term, so its rank is computed inside the
code: a Clifford C with C P C^-1 = Z_1 turns the code into |k> (x) C^(m-1),
and the rank of the (m-1)-qutrit factor is decided by the rank-2 and rank-3
searches of rank_exclusion.py over the (m-1)-qutrit dictionary.

For T3 and P = Z^(x m) this is the cat/sector construction behind
bounds/T3-m5-upper-18.json, and at m = 3, 4 the sectors have rank 3
(exhaustive, control below). Here every Pauli string with full support is
tried, up to the local Clifford stabilizer of |M> on each copy and the copy
permutations, for the N, H3, S and T3 orbits.

Only Pauli strings are used: the eigenspaces of a non-Pauli Clifford are not
stabilizer codes, so a sector's rank would have to be decided in the full
m-qutrit dictionary, which at m = 4 (7.4 million states, 81 amplitudes each)
does not fit in memory on a shared machine.

Usage: sectors.py ORBIT M [--rank3]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import alpha, target  # noqa: E402
from rank_exclusion import clifford_group, dictionary, rank2_search, rank3_search  # noqa: E402

W3 = np.exp(2j * np.pi / 3)
X1 = np.roll(np.eye(3), 1, axis=0)                 # X|x> = |x+1>
Z1 = np.diag(W3 ** np.arange(3))
F1 = np.array([[W3 ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(3)
S1 = np.diag([1, 1, W3])                           # S X S^-1 = XZ
M2 = np.zeros((3, 3))                              # |x> -> |2x>
for _x in range(3):
    M2[(2 * _x) % 3, _x] = 1


def pauli1(a, c):
    return np.linalg.matrix_power(X1, a) @ np.linalg.matrix_power(Z1, c)


def local(U, i, m):
    """U on qutrit i, identity elsewhere."""
    out = np.array([[1.0]])
    for j in range(m):
        out = np.kron(out, U if j == i else np.eye(3))
    return out


def sum_gate(i, j, m):
    """SUM_{i->j}: |x_i, x_j> -> |x_i, x_j + x_i>."""
    dim = 3 ** m
    G = np.zeros((dim, dim))
    for src in range(dim):
        x = [(src // 3 ** (m - 1 - k)) % 3 for k in range(m)]
        y = list(x)
        y[j] = (x[j] + x[i]) % 3
        dst = sum(y[k] * 3 ** (m - 1 - k) for k in range(m))
        G[dst, src] = 1
    return G


def swap_gate(i, j, m):
    dim = 3 ** m
    G = np.zeros((dim, dim))
    for src in range(dim):
        x = [(src // 3 ** (m - 1 - k)) % 3 for k in range(m)]
        x[i], x[j] = x[j], x[i]
        dst = sum(x[k] * 3 ** (m - 1 - k) for k in range(m))
        G[dst, src] = 1
    return G


def reduce_to_z0(avec, cvec, m):
    """A Clifford C (dense) with C P C^-1 proportional to Z on qutrit 0, for
    P = (x)_i X^{a_i} Z^{c_i} with full support."""
    a, c = list(avec), list(cvec)
    dim = 3 ** m
    C = np.eye(dim, dtype=complex)
    for i in range(m):
        if a[i] == 0:
            C = local(F1, i, m) @ C                 # Z -> X^-1
            a[i], c[i] = (-c[i]) % 3, 0
    for i in range(m):
        if c[i] != 0:
            k = (-c[i] * pow(a[i], -1, 3)) % 3      # S^k: X^a Z^c -> X^a Z^{c + k a}
            C = np.linalg.matrix_power(local(S1, i, m), k) @ C
            c[i] = 0
    for i in range(m):
        if a[i] == 2:
            C = local(M2, i, m) @ C                 # X^2 -> X
            a[i] = 1
    for i in range(1, m):
        C = sum_gate(0, i, m).conj().T @ C          # X_0 X_i -> X_0
    C = local(F1, 0, m) @ C                         # X_0 -> Z_0
    return C


def pauli_string(avec, cvec, m):
    P = np.array([[1.0 + 0j]])
    for a, c in zip(avec, cvec):
        P = np.kron(P, pauli1(a, c))
    # rescale so that P^3 = I
    P3 = P @ P @ P
    lam = P3[0, 0]
    P = P / lam ** (1 / 3)
    assert np.allclose(P @ P @ P, np.eye(3 ** m)), "P^3 is not the identity after rescaling"
    return P


def sector_states(orbit, m, avec, cvec):
    """The three sector states of |M>^m for P, each as (m-1)-qutrit vectors
    after the Clifford reduction, with their norms (the weights in |M>^m)."""
    psi = target(orbit, m)
    P = pauli_string(avec, cvec, m)
    C = reduce_to_z0(avec, cvec, m)
    out = []
    for s in range(3):
        v = sum(W3 ** (-s * j) * np.linalg.matrix_power(P, j) @ psi for j in range(3)) / 3
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            out.append((s, 0.0, None))
            continue
        w = (C @ v).reshape(3, -1)
        blocks = np.linalg.norm(w, axis=1)
        k = int(np.argmax(blocks))
        if not np.allclose(np.delete(blocks, k), 0, atol=1e-9):
            raise AssertionError("the Clifford reduction did not isolate one block")
        out.append((s, float(nrm), w[k] / blocks[k]))
    return out


def pauli_classes(orbit, m):
    """Pauli strings with full support, one per orbit of the local stabilizer
    of |M> acting on each copy and the copy permutations: multisets over the
    orbits of nonidentity single-qutrit Paulis under the stabilizer."""
    a1 = alpha(orbit)
    G = clifford_group(3)
    stab = [U for U in G if abs(abs(np.vdot(a1, U @ a1)) - 1) < 1e-9]
    paulis = [(a, c) for a in range(3) for c in range(3) if (a, c) != (0, 0)]

    def key(Pm):
        v = Pm.ravel()
        v = v / v[np.flatnonzero(np.abs(v) > 1e-9)[0]]
        return (np.round(v, 6) + 0.0).tobytes()

    index = {key(pauli1(a, c)): (a, c) for a, c in paulis}
    parent = {pc: pc for pc in paulis}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for U in stab:
        for a, c in paulis:
            img = index[key(U @ pauli1(a, c) @ U.conj().T)]
            ra, rb = find((a, c)), find(img)
            if ra != rb:
                parent[ra] = rb
    reps = sorted({find(pc) for pc in paulis})
    classes = list(itertools.combinations_with_replacement(reps, m))
    return classes, len(stab), reps


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("orbit")
    ap.add_argument("m", type=int)
    ap.add_argument("--rank3", action="store_true", help="also run the rank-3 search on sectors of rank > 2")
    a = ap.parse_args(argv[1:])
    m = a.m
    classes, order, reps = pauli_classes(a.orbit, m)
    print(f"{a.orbit} m={m}: local stabilizer of order {order}, {len(reps)} orbits of nonidentity "
          f"Paulis {reps}, {len(classes)} Pauli-string classes with full support")
    D = dictionary(3, m - 1)
    t0 = time.time()
    summary = []
    for cls in classes:
        avec = [p[0] for p in cls]
        cvec = [p[1] for p in cls]
        secs = sector_states(a.orbit, m, avec, cvec)
        ranks = []
        for s, nrm, v in secs:
            if v is None:
                ranks.append(0)
                continue
            r2, _ = rank2_search(v, D)
            if r2 == "RANK1":
                ranks.append(1)
            elif r2:
                ranks.append(2)
            else:
                if a.rank3:
                    r = rank3_search(v, D, workers=1)
                    ranks.append(3 if r["found"] else 4)   # 4 means "> 3"
                else:
                    ranks.append(3)                          # ">= 3"
        label = " ".join(f"X^{p[0]}Z^{p[1]}" for p in cls)
        weights = [round(nrm ** 2, 4) for _, nrm, _ in secs]
        print(f"  {label}: sector weights {weights}, sector ranks "
              f"{[('>=3' if r == 3 and not a.rank3 else ('>3' if r == 4 else r)) for r in ranks]} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        summary.append((label, ranks))
    best = min(sum(r for r in ranks) for _, ranks in summary)
    print(f"smallest sector-rank total (with >=3 counted as 3): {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
