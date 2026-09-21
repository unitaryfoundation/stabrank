"""Cat-state witnesses for the qubit H-type orbit at m = 6, 7, 8, 10.

Kissinger, van de Wetering, and Vilmart (arXiv:2202.09202, Section 4)
restate two constructions of Qassim, Pashayan, and Gosset (arXiv:2106.07740)
in the ZX-calculus and add a third. Written in the phase basis, with
|T> = (|0> + e^{i pi/4}|1>)/sqrt 2 and |T_perp> = Z|T>,

    |cat_n> = (|T>^n + |T_perp>^n)/sqrt 2
            = 2^{-(n-1)/2} sum_{|x| even} i^{|x|/2} |x>,

and the three constructions are

  cat_6     |cat_6> = a(|0^6> - i|1^6>) + b|E_6> + c|K_6>, where |E_6> is the
            uniform superposition of even-weight strings and |K_6> the same
            with sign (-1)^{|x|/2}; three stabilizer terms, so |T>^6 has six.
  gluing    |cat_{a+b-2}> is proportional to (I (x) <cat_2| (x) I)(|cat_a>
            (x) |cat_b>), since the bra <00| - i<11| kills the odd-odd and
            mixed parity sectors and shifts the phase of the odd-odd one by i.
            Two copies of |cat_6> give |cat_10> in 9 terms and |T>^10 in 18.
  partial   |T>^5 = sqrt 2 (I (x) <T|)|cat_6>, because <T|Z|T> = 0. Each of
            the three terms <T|_6 s_j is a Clifford image of (stabilizer (x)
            |T>): one T state is left per term, so four are removed at a cost
            of three terms. Tensoring with |T>^2 or |T>^3 and regrouping the
            leftover |T> with them, the bra and kets on the extra qubits form
            the operator M = |T>^{r}<T|, whose Choi vector (S^dagger (x) I)
            |T>^{r+1} has a known stabilizer decomposition (the board's m = 3
            and m = 4 witnesses). That gives |T>^7 in 3 x 3 = 9 terms and
            |T>^8 in 3 x 4 = 12.

Everything is built numerically in the T basis, mapped to the board's H basis
by the Clifford C = HSH (C|T> is proportional to cos(pi/8)|0> + sin(pi/8)|1>),
and handed to `verify_challenge/to_witness.witness_from_vectors`, which
recovers the exact (k, x0, W, Q, l) form of each term and fits the coefficients
in exact arithmetic. |T>^n itself is split as (|cat_n> + G_1|cat_n>)/sqrt 2
with G = (X + Y)/sqrt 2 on the first qubit, the Clifford that fixes |T> and
negates |T_perp>.

Usage (from the repository root):
    uv run --extra challenge python research/constructions/kvv_cat.py [m ...]
writes bounds/qubit_H-m{m}-upper-{rank}.json for the requested m in
{7, 8, 10} (default: all, plus the m = 6 control, which goes to the system
temp directory since the board's m = 6 cell already carries QPG's witness)
with the witness and provenance filled in, and prints the numerical residual
of each construction before conversion.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from stabrank_verify import stabilizer_vector  # noqa: E402
from to_witness import witness_from_vectors  # noqa: E402

# ------------------------------------------------------------- one qubit ----
w8 = np.exp(1j * np.pi / 4)
T = np.array([1, w8]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
G = (X + Y) / np.sqrt(2)          # fixes |T>, negates Z|T>
C = H @ S @ H                     # C|T> is proportional to |H>
HSTATE = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])
assert abs(abs(HSTATE.conj() @ C @ T) - 1) < 1e-12


def kron(*vs):
    out = np.array([1.0 + 0j])
    for v in vs:
        out = np.kron(out, v)
    return out


def power(v, n):
    return kron(*([v] * n))


def apply_on(op, qubit, n, vec):
    """(I (x) ... op ... (x) I) vec on an n-qubit vector."""
    t = vec.reshape([2] * n)
    t = np.moveaxis(np.tensordot(op, t, axes=([1], [qubit])), 0, qubit)
    return t.reshape(-1)


def weights(n):
    return np.array([bin(i).count("1") for i in range(2 ** n)])


def cat(n):
    wt = weights(n)
    v = np.where(wt % 2 == 0, 1j ** (wt // 2), 0).astype(complex)
    return v / np.sqrt(2 ** (n - 1))


def cat6_terms():
    wt = weights(6)
    even = (wt % 2 == 0).astype(complex)
    s1 = np.zeros(64, dtype=complex)
    s1[0], s1[63] = 1, -1j
    s2 = even.copy()
    s3 = even * (-1.0) ** (wt // 2)
    return [s1, s2, s3]


def lstsq_check(target, terms, label):
    A = np.stack(terms, axis=1)
    c, *_ = np.linalg.lstsq(A, target, rcond=None)
    resid = np.linalg.norm(A @ c - target)
    print(f"  {label}: {len(terms)} terms, residual {resid:.2e}")
    assert resid < 1e-10, label
    return c


def split_T_power(cat_terms, n):
    """Terms for |T>^n from terms for |cat_n>, via (cat + G_1 cat)/sqrt 2."""
    return cat_terms + [apply_on(G, 0, n, t) for t in cat_terms]


def to_H_basis(vec, n):
    for q in range(n):
        vec = apply_on(C, q, n, vec)
    return vec


def board_terms_T_basis(m):
    """The board's |H>^m witness terms, mapped to the T basis."""
    path = os.path.join(ROOT, "bounds", f"qubit_H-m{m}-upper-{m}.json")
    sub = json.load(open(path))
    out = []
    for t in sub["witness"]["terms"]:
        v = np.array(stabilizer_vector(t, 2, m).evalf(20), dtype=complex).ravel()
        for q in range(m):
            v = apply_on(C.conj().T, q, m, v)
        out.append(v)
    lstsq_check(power(T, m), out, f"board m={m} witness in the T basis")
    return out


def cat10_terms():
    """(I^5 (x) <cat_2| (x) I^5)(s_i (x) s_j) over the 3 x 3 pairs."""
    bra = np.zeros(4, dtype=complex)
    bra[0], bra[3] = 1, -1j          # conjugate of |00> + i|11>
    out = []
    for si in cat6_terms():
        for sj in cat6_terms():
            v = np.kron(si, sj).reshape(32, 4, 32)
            u = np.einsum("a,iaj->ij", bra, v).reshape(-1)
            if np.linalg.norm(u) > 1e-12:
                out.append(u)
    return out


def partial_terms_T5():
    """The three terms <T|_6 s_j with |T>^5 = sqrt 2 sum_j c_j <T|_6 s_j."""
    return [np.tensordot(s.reshape(32, 2), T.conj(), axes=([1], [0]))
            for s in cat6_terms()]


def partial_product_terms(r):
    """Terms for |T>^{5+r}: (I^5 (x) M_l) s_j with sum_l d_l M_l = |T>^r <T|."""
    choi_terms = [apply_on(S.conj().T, 0, r + 1, v)
                  for v in board_terms_T_basis(r + 1)]
    choi = apply_on(S.conj().T, 0, r + 1, power(T, r + 1))
    assert np.allclose(choi, np.kron(T.conj(), power(T, r)))
    out = []
    for s in cat6_terms():
        s = s.reshape(32, 2)
        for ch in choi_terms:
            M = ch.reshape(2, 2 ** r).T           # M[b, a] from |a>|b>
            out.append(np.einsum("xa,ba->xb", s, M).reshape(-1))
    return out


def construct(m):
    if m == 6:
        c6 = cat6_terms()
        lstsq_check(cat(6), c6, "cat_6 from the three QPG terms")
        terms = split_T_power(c6, 6)
    elif m == 10:
        c10 = cat10_terms()
        lstsq_check(cat(10), c10, "cat_10 from glued cat_6 pairs")
        terms = split_T_power(c10, 10)
    elif m in (7, 8):
        lstsq_check(power(T, 5), partial_terms_T5(), "partial decomposition of T^5")
        terms = partial_product_terms(m - 5)
    else:
        raise ValueError(m)
    lstsq_check(power(T, m), terms, f"T^{m} in the T basis")
    return [to_H_basis(t, m) for t in terms]


NOTES = {
    6: ("chi(|H>^6) <= 6 from the cat_6 decomposition of Qassim, Pashayan, and "
        "Gosset, in the ZX form of Kissinger, van de Wetering, and Vilmart. "
        "Witness built here by research/constructions/kvv_cat.py as a control "
        "for the m = 7, 8, 10 cells: the three stabilizer terms of |cat_6> = "
        "(|T>^6 + |T_perp>^6)/sqrt 2 and their images under the Clifford that "
        "fixes |T> and negates |T_perp> on one qubit, mapped to the H basis."),
    7: ("chi(|H>^7) <= 9 from the partial decomposition of Kissinger, van de "
        "Wetering, and Vilmart (arXiv:2202.09202, Section 4.2): |T>^5 = sqrt 2 "
        "(I (x) <T|)|cat_6> has three terms, each a Clifford image of "
        "(stabilizer (x) |T>), so four T states are removed at a cost of three "
        "terms and chi(T^t) <= 3 chi(T^{t-4}). At t = 7 the leftover |T> of "
        "each term joins the remaining |T>^2, and |T>^3 has rank 3, giving 3 x "
        "3 = 9. Improves the 12 that QPG's table lists at seven copies. "
        "Witness built here by research/constructions/kvv_cat.py from the "
        "three cat_6 terms and the board's m=3 witness, and verified in exact "
        "arithmetic; exponent log_2(9)/7 = 0.4528."),
    8: ("chi(|H>^8) <= 12 from the partial decomposition of Kissinger, van de "
        "Wetering, and Vilmart (arXiv:2202.09202, Section 4.2), chi(T^t) <= 3 "
        "chi(T^{t-4}), with chi(T^4) = 4: the leftover |T> of each of the three "
        "terms joins the remaining |T>^3, and |T>^4 has rank 4. Same value as "
        "the product 6 x 2 of the m=6 and m=2 cells and as QPG's table. Witness "
        "built here by research/constructions/kvv_cat.py from the three cat_6 "
        "terms and the board's m=4 witness, and verified in exact arithmetic; "
        "exponent log_2(12)/8 = 0.4481."),
    10: ("chi(|H>^10) <= 18 from the cat_{4k+2} gluing of Qassim, Pashayan, and "
         "Gosset as restated by Kissinger, van de Wetering, and Vilmart "
         "(arXiv:2202.09202, Section 4.2) at k = 2: |cat_10> is proportional to "
         "(I (x) <cat_2| (x) I)(|cat_6> (x) |cat_6>), so it has at most 3 x 3 = "
         "9 stabilizer terms and |T>^10 = (|cat_10> + |cat_10^->)/sqrt 2 has 18. "
         "Exponent log_2(18)/10 = 0.4170, the best finite-m value on the board "
         "for this orbit. QPG's own nine-term |cat_10> gives the same rank. "
         "Witness built here by research/constructions/kvv_cat.py and verified "
         "in exact arithmetic."),
}

RANK = {6: 6, 7: 9, 8: 12, 10: 18}


def main(argv):
    ms = [int(a) for a in argv[1:]] or [6, 7, 8, 10]
    for m in ms:
        print(f"m = {m}")
        t0 = time.time()
        vecs = construct(m)
        assert len(vecs) == RANK[m], (m, len(vecs))
        w = witness_from_vectors("qubit_H", m, vecs)
        dt = time.time() - t0
        print(f"  exact witness with {len(w['terms'])} terms in {dt:.1f} s")
        sub = {
            "schema_version": "0.1", "orbit": "qubit_H", "m": m,
            "direction": "upper", "rank": RANK[m],
            "provenance": {
                "author": "Kissinger, van de Wetering, and Vilmart",
                "reference": "arXiv:2202.09202",
                "method": "literature",
                "date": "2022-02-18",
                "github": ["vprusso"],
                "compute": {
                    "cpu_hours": 0.01, "wall_clock_hours": 0.01, "runs": 1,
                    "hardware": "Apple silicon laptop, one core; direct "
                                "construction, no search",
                },
            },
            "notes": NOTES[m],
            "witness": w,
        }
        out = os.path.join(ROOT, "bounds", f"qubit_H-m{m}-upper-{RANK[m]}.json")
        if m == 6:
            # The board's m=6 cell already carries QPG's witness; this one is
            # the control for the construction and goes to the temp directory.
            out = os.path.join(tempfile.gettempdir(),
                               "qubit_H-m6-upper-6-kvv-control.json")
        with open(out, "w") as f:
            json.dump(sub, f, indent=2)
            f.write("\n")
        print(f"  written {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
