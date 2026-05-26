"""Decode saved stabilizer-rank decomposition npz files into the canonical
(k, x_0, W, Q) form used in the paper appendix:

    |S> = (1/sqrt(p^k)) sum_{y in F_p^k} omega^{Q(y)} |x_0 + W y>

where Q(y) is a quadratic polynomial in y_0,...,y_{k-1} over F_p (p=3 here),
W is an m x k generator matrix over F_p, and x_0 in F_p^m.

For each basis vector we print:
- k, x_0, W
- Q: the quadratic monomial coefficients (constant, linear, pure square, mixed)
- linear-combination coefficient on the basis state

Usage:
    python extract_decomposition.py solution_strange_m3_chi4_*.npz
"""

import argparse
import glob
import itertools
import sys

import numpy as np

P = 3
OMEGA = np.exp(2j * np.pi / P)


def trit_index(x_vec):
    idx = 0
    for trit in x_vec:
        idx = idx * P + int(trit)
    return idx


def index_to_trit(idx, m):
    x = np.zeros(m, dtype=int)
    for i in range(m - 1, -1, -1):
        x[i] = idx % P
        idx //= P
    return x


def rref_fp(M, p=P):
    """Row-reduced echelon form over F_p, in place. Returns (M, pivots)."""
    M = M.copy() % p
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nonzero = [i for i in range(r, rows) if M[i, c] != 0]
        if not nonzero:
            continue
        pivot = nonzero[0]
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        inv = pow(int(M[r, c]), -1, p)
        M[r] = (M[r] * inv) % p
        for i in range(rows):
            if i != r and M[i, c] != 0:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        pivots.append(c)
        r += 1
    return M, pivots


def find_generators(deltas, m, p=P):
    """Given a set of delta vectors in F_p^m spanning a k-dim subspace,
    return W (m x k) whose columns generate that subspace."""
    if len(deltas) == 0:
        return np.zeros((m, 0), dtype=int), 0
    A = np.array(deltas, dtype=int) % p
    rref, pivots = rref_fp(A, p)
    rank = len([r for r in range(rref.shape[0]) if not np.all(rref[r] == 0)])
    if rank == 0:
        return np.zeros((m, 0), dtype=int), 0
    basis_rows = rref[:rank]
    W = basis_rows.T  # m x rank
    return W, rank


def solve_W_y_eq_delta(W, delta, p=P):
    """Solve W y = delta over F_p for y in F_p^k. W is m x k.
    Assumes a solution exists (delta is in column span of W)."""
    m, k = W.shape
    aug = np.column_stack([W % p, delta % p])
    rref, pivots = rref_fp(aug, p)
    y = np.zeros(k, dtype=int)
    pivot_to_var = [pi for pi in pivots if pi < k]
    for r, c in enumerate(pivot_to_var):
        y[c] = int(rref[r, k])
    return y


def fit_quadratic_polynomial(y_vals, q_vals, k, p=P):
    """Given samples (y, q) with y in F_p^k and q in F_p, fit a quadratic
    polynomial Q(y) = c_0 + sum_i a_i y_i + sum_i b_i y_i^2 + sum_{i<j} c_ij y_i y_j
    over F_p. Returns dict {monomial_string: coefficient}.

    Solves the linear system M c = q over F_p via Gaussian elimination on
    augmented [M | q]. A solution exists iff the underlying state is a
    quadratic stabilizer state (which all SA-engine outputs are by
    construction).
    """
    monomials = []
    monomials.append(("1", lambda y: 1))
    for i in range(k):
        monomials.append((f"y_{i}", lambda y, i=i: int(y[i])))
    for i in range(k):
        monomials.append((f"y_{i}^2", lambda y, i=i: int(y[i]) ** 2))
    for i, j in itertools.combinations(range(k), 2):
        monomials.append((f"y_{i}y_{j}", lambda y, i=i, j=j: int(y[i]) * int(y[j])))

    n_eqs = len(y_vals)
    n_unk = len(monomials)
    M = np.zeros((n_eqs, n_unk), dtype=int)
    q = np.zeros(n_eqs, dtype=int)
    for r, (y, qv) in enumerate(zip(y_vals, q_vals)):
        for c, (_, fn) in enumerate(monomials):
            M[r, c] = fn(y) % p
        q[r] = qv % p

    aug = np.column_stack([M, q])
    rref, pivots = rref_fp(aug, p)
    bad = [r for r in range(rref.shape[0])
           if np.all(rref[r, :n_unk] == 0) and rref[r, n_unk] != 0]
    if bad:
        raise RuntimeError("polynomial system inconsistent — state not quadratic stabilizer?")

    coeffs = np.zeros(n_unk, dtype=int)
    for r, c in enumerate([pi for pi in pivots if pi < n_unk]):
        coeffs[c] = int(rref[r, n_unk])

    return {name: int(coeffs[i]) for i, (name, _) in enumerate(monomials)
            if int(coeffs[i]) != 0}


def decode_stabilizer_state(v, m, p=P, tol=1e-7):
    """Decode unit-norm length-p^m vector v as a stabilizer state.
    Returns (k, x_0, W, Q_dict, global_phase)."""
    abs_v = np.abs(v)
    max_amp = abs_v.max()
    support = np.where(abs_v > tol * max_amp)[0]
    n_sup = len(support)
    k = int(round(np.log(n_sup) / np.log(p)))
    if p ** k != n_sup:
        raise ValueError(f"support size {n_sup} is not {p}^k for any k")

    x0_idx = int(support[0])
    x0 = index_to_trit(x0_idx, m)

    deltas = []
    for s_idx in support[1:]:
        s_vec = index_to_trit(int(s_idx), m)
        deltas.append((s_vec - x0) % p)

    W, rank = find_generators(deltas, m, p)
    if rank != k:
        raise RuntimeError(f"generator rank {rank} != expected k={k}")

    # global phase fixes Q(0) = 0
    expected_amp = 1.0 / np.sqrt(p ** k)
    global_phase = v[x0_idx] / expected_amp

    y_vals = []
    q_vals = []
    for s_idx in support:
        s_vec = index_to_trit(int(s_idx), m)
        delta = (s_vec - x0) % p
        y = solve_W_y_eq_delta(W, delta, p)
        # check
        check = (W @ y) % p
        if not np.array_equal(check, delta):
            raise RuntimeError(f"y={y} doesn't satisfy W y = delta={delta}")
        amp = v[int(s_idx)] * np.sqrt(p ** k) / global_phase
        # amp should be on unit circle; phase = omega^q
        ang = np.angle(amp) / (2 * np.pi / p)
        q = int(round(ang)) % p
        residual = abs(amp / OMEGA ** q - 1.0)
        if residual > 1e-5:
            raise RuntimeError(
                f"amplitude at y={y} not omega^q (residual {residual:.2e}, |amp|={abs(amp):.4f})")
        y_vals.append(y)
        q_vals.append(q)

    Q = fit_quadratic_polynomial(y_vals, q_vals, k, p)
    return k, x0, W, Q, complex(global_phase)


def format_W(W):
    if W.shape[1] == 0:
        return "(empty)"
    rows = [", ".join(str(int(x)) for x in row) for row in W]
    return "[" + "; ".join(rows) + "]"


def format_Q(Q, k):
    if not Q:
        return "0"
    parts = []
    for name, coeff in sorted(Q.items()):
        if name == "1":
            parts.append(str(coeff))
        else:
            parts.append(f"{coeff}{name}" if coeff != 1 else name)
    return " + ".join(parts)


def report_npz(path):
    data = np.load(path)
    n = int(data["n"]) if "n" in data.files else None
    p = int(data["p"]) if "p" in data.files else 3
    chi = int(data["k"]) if "k" in data.files else None
    orbit = str(data["orbit"]) if "orbit" in data.files else "?"
    coeffs = np.asarray(data["linear_coeffs"])

    print(f"\n=== {path} ===")
    print(f"orbit={orbit}  m={n}  p={p}  chi={chi}")
    print(f"final_error = {float(data['final_error']):.3e}")
    print()

    target = np.asarray(data["target_function"])
    target = target / np.linalg.norm(target)

    recon = np.zeros_like(target)
    for i in range(chi):
        b = np.asarray(data[f"basis_func_{i}"])
        try:
            kk, x0, W, Q, gphase = decode_stabilizer_state(b, n, p=p)
        except Exception as e:
            print(f"basis {i}: DECODE FAIL — {e}")
            continue
        c = complex(coeffs[i])
        recon = recon + c * b
        print(f"basis {i}:")
        print(f"  k = {kk}")
        print(f"  x_0 = {x0.tolist()}")
        print(f"  W   = {format_W(W)}")
        print(f"  Q(y) = {format_Q(Q, kk)}  (mod {p})")
        print(f"  global phase = {gphase:.6f}  |.|={abs(gphase):.4f}")
        print(f"  alpha = {c:.6f}")
        print()
    err = np.linalg.norm(target - recon)
    print(f"sum_i alpha_i |S_i> recon err = {err:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="npz files (globs ok)")
    args = parser.parse_args()

    paths = []
    for arg in args.paths:
        m = sorted(glob.glob(arg))
        if not m:
            print(f"WARN: no match for {arg}", file=sys.stderr)
        paths.extend(m)

    for p in paths:
        report_npz(p)


if __name__ == "__main__":
    main()
