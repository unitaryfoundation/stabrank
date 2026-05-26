"""Numerical audit of the seven qutrit stabilizer decompositions in the paper.

For each qutrit (orbit, m) in the paper's Appendix~\ref{app:decomps},
this script:
  1. Encodes the canonical-form parameters (k, x_0, W, Q) and
     coefficient list as stated in main.tex.
  2. Builds each stabilizer state numerically.
  3. Builds the target |M>^{otimes m}.
  4. Verifies the claimed identity sum_i c_i |s_i> = |M>^{otimes m}
     to floating-point precision.

The qubit decompositions of Appendix~\ref{app:qubit-t} (T-type m=2,
m=3, m=4) are checked by verify_qubit_decompositions.py in the same
directory.

A PASS means the algebraic identity in the appendix holds exactly. A
FAIL means there's a transcription or proof error in the paper.

Self-contained: only needs numpy.
"""
from __future__ import annotations

import itertools

import numpy as np


P = 3
OMEGA3 = np.exp(2j * np.pi / 3)


def trit_index(x_vec, n):
    """Standard kronecker index, MSB-first (matches np.kron convention)."""
    idx = 0
    for trit in x_vec:
        idx = idx * P + int(trit)
    return idx


def build_stab(n, k, x0, W, Q_eval):
    """Build (1/sqrt(3^k)) sum_y omega^{Q(y)} |x_0 + W y> over F_3^k.

    Args:
      n: number of qutrits
      k: dimension of subspace (number of free trits y)
      x0: length-n integer array (coset rep)
      W: n x k integer matrix (columns are basis vectors of subspace)
      Q_eval: callable y -> int, the quadratic phase polynomial mod 3
    """
    state = np.zeros(P ** n, dtype=complex)
    x0 = np.asarray(x0, dtype=int)
    W = np.asarray(W, dtype=int)
    if k == 0:
        idx = trit_index(x0 % P, n)
        state[idx] = 1.0
        return state
    for y_tuple in itertools.product(range(P), repeat=k):
        y = np.asarray(y_tuple, dtype=int)
        x = (x0 + W @ y) % P
        idx = trit_index(x, n)
        q = Q_eval(y) % P
        state[idx] += OMEGA3 ** q
    state /= np.sqrt(P ** k)
    return state


def kron_n(v, n):
    out = v
    for _ in range(n - 1):
        out = np.kron(out, v)
    return out


def strange_state():
    return np.array([0, 1, -1], dtype=complex) / np.sqrt(2)


def h3_state():
    return np.array([1, (np.sqrt(3) - 1) / 2, (np.sqrt(3) - 1) / 2], dtype=complex) / np.sqrt(3 - np.sqrt(3))


def norrell_state():
    return np.array([1, 1, -2], dtype=complex) / np.sqrt(6)


def t3_state():
    o9 = np.exp(2j * np.pi / 9)
    return np.array([1, o9, o9 ** 2], dtype=complex) / np.sqrt(3)


def cols(*es):
    return np.column_stack(es)


def e(i, n):
    v = np.zeros(n, dtype=int)
    v[i] = 1
    return v


def report(name, target, terms):
    """terms: list of (coeff, state)."""
    s = np.zeros_like(target)
    for c, v in terms:
        s = s + c * v
    res = np.linalg.norm(s - target)
    nrm = np.linalg.norm(target)
    rel = res / nrm if nrm > 0 else float("inf")
    status = "PASS" if rel < 1e-9 else "FAIL"
    print(f"  [{status}]  {name}:  ||sum - target|| = {res:.3e}  "
          f"(rel {rel:.3e})  chi <= {len(terms)}")
    return rel < 1e-9


# === Decomposition encodings ===

def verify_strange_m2():
    n, k = 2, 2
    x0 = np.zeros(n, dtype=int)
    W = np.eye(n, dtype=int)
    Q1 = lambda y: y[0] ** 2 + y[0] * y[1] + y[1] ** 2
    Q2 = lambda y: y[0] ** 2 + 2 * y[0] * y[1] + y[1] ** 2
    S1 = build_stab(n, k, x0, W, Q1)
    S2 = build_stab(n, k, x0, W, Q2)
    coef = -1j * np.sqrt(3) / 2 * OMEGA3
    target = kron_n(strange_state(), 2)
    return report("Strange m=2 (chi=2)", target, [(coef, S1), (-coef, S2)])


def verify_strange_m3():
    n = 3
    W_e02 = cols(e(0, n), e(2, n))  # n x 2
    Q1 = lambda w: w[0] ** 2 + w[0] * w[1]
    Q3 = lambda w: w[0] ** 2 + 2 * w[0] * w[1]
    Q2 = lambda w: 2 * w[0] ** 2 + w[0] * w[1] + w[1] ** 2
    Q4 = lambda w: 2 * w[0] ** 2 + 2 * w[0] * w[1] + w[1] ** 2
    x0_13 = np.array([0, 2, 0])
    x0_24 = np.array([0, 1, 0])
    S1 = build_stab(n, 2, x0_13, W_e02, Q1)
    S3 = build_stab(n, 2, x0_13, W_e02, Q3)
    S2 = build_stab(n, 2, x0_24, W_e02, Q2)
    S4 = build_stab(n, 2, x0_24, W_e02, Q4)
    alpha = (3 * np.sqrt(2) - 1j * np.sqrt(6)) / 8
    beta = -1j * np.sqrt(6) / 4
    target = kron_n(strange_state(), 3)
    return report("Strange m=3 (chi<=4)", target,
                  [(alpha, S1), (-alpha, S3), (beta, S2), (-beta, S4)])


def verify_h3_m2():
    n = 2
    # |S_1> = |0, +>:  k=1, x_0=0, W=e_1, Q=0
    S1 = build_stab(n, 1, np.zeros(n, dtype=int), e(1, n).reshape(n, 1), lambda y: 0)
    # |S_2> = |+, 0>:  k=1, x_0=0, W=e_0, Q=0
    S2 = build_stab(n, 1, np.zeros(n, dtype=int), e(0, n).reshape(n, 1), lambda y: 0)
    # |S_3>: k=2, x_0=0, W=I_2, Q_3(y) = 2 y_0^2 + y_1^2
    S3 = build_stab(n, 2, np.zeros(n, dtype=int), np.eye(n, dtype=int),
                    lambda y: 2 * y[0] ** 2 + y[1] ** 2)
    c = (np.sqrt(3) - 1) / 2
    N = np.sqrt(3 - np.sqrt(3))
    a1 = c * np.sqrt(3) / N ** 2 * (1 - c * OMEGA3)
    a2 = c * np.sqrt(3) / N ** 2 * (1 - c * OMEGA3 ** 2)
    a3 = 3 * c ** 2 / N ** 2
    target = kron_n(h3_state(), 2)
    return report("H_3 m=2 (chi<=3)", target,
                  [(a1, S1), (a2, S2), (a3, S3)])


def verify_h3_m3():
    n = 3
    # S_1, S_2: k=3, x_0=0, W=I_3
    Q1 = lambda y: 2 * y[0] ** 2 + 2 * y[1] ** 2 + y[2] ** 2
    Q2 = lambda y: y[0] ** 2 + y[1] ** 2 + 2 * y[2] ** 2
    S1 = build_stab(n, 3, np.zeros(n, dtype=int), np.eye(n, dtype=int), Q1)
    S2 = build_stab(n, 3, np.zeros(n, dtype=int), np.eye(n, dtype=int), Q2)
    # S_3 = |0, 0, +>: k=1, x_0=0, W=e_2, Q=0
    S3 = build_stab(n, 1, np.zeros(n, dtype=int), e(2, n).reshape(n, 1), lambda y: 0)
    # S_4 = |+, +, 0>: k=2, x_0=0, W=(e_0, e_1), Q=0
    S4 = build_stab(n, 2, np.zeros(n, dtype=int), cols(e(0, n), e(1, n)), lambda y: 0)
    c = (np.sqrt(3) - 1) / 2
    N = np.sqrt(3 - np.sqrt(3))
    a1 = 3 * c / (4 * N) * (1 + 1j)
    a2 = 3 * c / (4 * N) * (1 - 1j)
    a3 = a4 = 3 / (4 * N)
    target = kron_n(h3_state(), 3)
    return report("H_3 m=3 (chi<=4)", target,
                  [(a1, S1), (a2, S2), (a3, S3), (a4, S4)])


def verify_norrell_m2():
    n = 2
    # S_1: k=2, x_0=0, W=I_2, Q_1(y) = y_0 + y_1 + y_0 y_1
    Q1 = lambda y: y[0] + y[1] + y[0] * y[1]
    S1 = build_stab(n, 2, np.zeros(n, dtype=int), np.eye(n, dtype=int), Q1)
    # S_2 = |2, 2>: k=0, x_0=(2,2)
    S2 = build_stab(n, 0, np.array([2, 2]), np.zeros((n, 0), dtype=int), lambda y: 0)
    # S_3: k=2, x_0=0, W=I_2, Q_3 = 2*Q_1
    Q3 = lambda y: 2 * (y[0] + y[1] + y[0] * y[1])
    S3 = build_stab(n, 2, np.zeros(n, dtype=int), np.eye(n, dtype=int), Q3)
    alpha = -OMEGA3 / 2
    target = kron_n(norrell_state(), 2)
    return report("Norrell m=2 (chi<=3)", target,
                  [(alpha, S1), (1.0, S2), (np.conj(alpha), S3)])


def verify_norrell_m3():
    n = 3
    # S_1: k=2, x_0=(2,0,0), W=(e_1,e_2), Q_1 = w_0 + 2 w_0^2 + 2 w_1 + w_1^2
    S1 = build_stab(n, 2, np.array([2, 0, 0]), cols(e(1, n), e(2, n)),
                    lambda w: w[0] + 2 * w[0] ** 2 + 2 * w[1] + w[1] ** 2)
    # S_2: k=2, x_0=(0,2,0), W=(e_0,e_2), Q_2 = 2 w_0 + w_0^2 + w_1 + 2 w_1^2
    S2 = build_stab(n, 2, np.array([0, 2, 0]), cols(e(0, n), e(2, n)),
                    lambda w: 2 * w[0] + w[0] ** 2 + w[1] + 2 * w[1] ** 2)
    # S_3 = |+,+,+>: k=3, x_0=0, W=I_3, Q=0
    S3 = build_stab(n, 3, np.zeros(n, dtype=int), np.eye(n, dtype=int), lambda y: 0)
    # S_4: k=2, x_0=(0,0,2), W=(e_0,e_1), Q_4 = w_0 + 2 w_0^2 + 2 w_1 + w_1^2
    S4 = build_stab(n, 2, np.array([0, 0, 2]), cols(e(0, n), e(1, n)),
                    lambda w: w[0] + 2 * w[0] ** 2 + 2 * w[1] + w[1] ** 2)
    a124 = -np.sqrt(6) / 4
    a3 = np.sqrt(2) / 4
    target = kron_n(norrell_state(), 3)
    return report("Norrell m=3 (chi<=4)", target,
                  [(a124, S1), (a124, S2), (a3, S3), (a124, S4)])


def verify_norrell_m4():
    n = 4
    # S_0: k=3, x_0=(0,2,0,0), W=(e_0,e_2,e_3),
    #      Q_0 = 2 w_0 + w_0^2 + w_1 + 2 w_1^2 + 2 w_2 + w_2^2
    S0 = build_stab(n, 3, np.array([0, 2, 0, 0]), cols(e(0, n), e(2, n), e(3, n)),
                    lambda w: 2 * w[0] + w[0] ** 2 + w[1] + 2 * w[1] ** 2 + 2 * w[2] + w[2] ** 2)
    # S_1: k=4, x_0=0, W=I_4, Q_1 = sum_{i=0,1}(2 y_i + y_i^2) + sum_{i=2,3}(y_i + 2 y_i^2)
    S1 = build_stab(n, 4, np.zeros(n, dtype=int), np.eye(n, dtype=int),
                    lambda y: (2 * y[0] + y[0] ** 2) + (2 * y[1] + y[1] ** 2)
                              + (y[2] + 2 * y[2] ** 2) + (y[3] + 2 * y[3] ** 2))
    # S_2: k=2, x_0=(2,0,0,2), W=(e_1,e_2), Q_2 = w_0 + 2 w_0^2 + w_1 + 2 w_1^2
    S2 = build_stab(n, 2, np.array([2, 0, 0, 2]), cols(e(1, n), e(2, n)),
                    lambda w: w[0] + 2 * w[0] ** 2 + w[1] + 2 * w[1] ** 2)
    # S_3: k=2, x_0=(0,0,2,2), W=(e_0,e_1), Q_3 = 2 w_0 + w_0^2 + w_1 + 2 w_1^2
    S3 = build_stab(n, 2, np.array([0, 0, 2, 2]), cols(e(0, n), e(1, n)),
                    lambda w: 2 * w[0] + w[0] ** 2 + w[1] + 2 * w[1] ** 2)
    # S_4: k=3, x_0=(2,0,0,0), W=(e_1,e_2,e_3),
    #      Q_4 = 2 w_0 + w_0^2 + 2 w_1 + w_1^2 + w_2 + 2 w_2^2
    S4 = build_stab(n, 3, np.array([2, 0, 0, 0]), cols(e(1, n), e(2, n), e(3, n)),
                    lambda w: 2 * w[0] + w[0] ** 2 + 2 * w[1] + w[1] ** 2 + w[2] + 2 * w[2] ** 2)
    # S_5: k=2, x_0=(0,2,2,0), W=(e_0,e_3), Q_5 = w_0 + 2 w_0^2 + 2 w_1 + w_1^2
    S5 = build_stab(n, 2, np.array([0, 2, 2, 0]), cols(e(0, n), e(3, n)),
                    lambda w: w[0] + 2 * w[0] ** 2 + 2 * w[1] + w[1] ** 2)
    # S_6 = |+>^{otimes 4}: k=4, x_0=0, W=I_4, Q=0
    S6 = build_stab(n, 4, np.zeros(n, dtype=int), np.eye(n, dtype=int), lambda y: 0)

    sigma = np.sqrt(3) / 4
    xi = np.exp(1j * np.pi / 6)
    target = kron_n(norrell_state(), 4)
    return report("Norrell m=4 (chi<=7)", target, [
        (sigma * OMEGA3, S0),
        (sigma * xi, S1),
        (sigma * xi, S2),
        (sigma * xi, S3),
        (sigma * OMEGA3, S4),
        (sigma * xi, S5),
        (OMEGA3 ** 2 / 4, S6),
    ])


def main() -> int:
    print("Verifying paper appendix decompositions:")
    results = []
    for fn in [verify_strange_m2, verify_strange_m3,
               verify_h3_m2, verify_h3_m3,
               verify_norrell_m2, verify_norrell_m3, verify_norrell_m4]:
        try:
            ok = fn()
        except Exception as exc:
            print(f"  [ERROR]  {fn.__name__}: {exc}")
            ok = False
        results.append((fn.__name__, ok))

    print("\nSummary:")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name}")
    print(f"\n  {n_pass}/{len(results)} decompositions verified.")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
