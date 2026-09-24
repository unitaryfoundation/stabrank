"""The cat-track witnesses: |cat_m> for m = 2 to 8 from the published terms.

Qassim, Pashayan, and Gosset (arXiv:2106.07740, QPG) define, with
|T> = (|0> + e^{i pi/4}|1>)/sqrt 2 and |T_perp> = Z|T>,

    |cat_m> = (|T>^m + |T_perp>^m)/sqrt 2 = 2^{-(m-1)/2} sum_{|x| even} i^{|x|/2} |x>

(their Eq. 3), and write out these stabilizer decompositions:

  m = 2   |cat_2> = 2^{-1/2}(|00> + i|11>), a stabilizer state (Eq. 5).
  m = 4   |cat_4> = i|E_4> + ((1-i)/2) 2^{-1/2}(|0^4> - i|1^4>), with |E_4> the
          uniform superposition of the even-weight strings (appendix).
  m = 6   |cat_6> = 2^{-3/2}(|0^6> - i|1^6>) + 2^{-1/2} e^{3 i pi/4}(|E_6> + i|K_6>),
          |K_6> = prod_{i<j} CZ_ij |E_6> (Eq. 5).
  m = 8   |cat_8> is proportional to <cat_2|_{4,5}(|cat_4> (x) |cat_6>), so the
          2 x 3 contracted products are a decomposition (appendix).
  m = 3, 5, 7   <0|_m |cat_m> is proportional to |cat_{m-1}>, so the terms of
          the decomposition one size up, projected, decompose the smaller cat
          (appendix; Kissinger, van de Wetering, and Vilmart, arXiv:2202.09202,
          Section 4.1, draw the m = 3 and m = 5 results in ZX).

Everything here is a mechanical replay of those statements: the terms are
built numerically exactly as written, the contraction and projections are
applied as stated, and `to_witness.witness_from_vectors` recovers the exact
(k, x0, W, Q, l) form of each term and fits the coefficients in exact
arithmetic, so the verifier checks the identity symbolically. The script
asserts that the contraction gives six nonzero, linearly independent terms
and that each projection keeps every term, so the filed ranks are the ranks
the papers claim; if either ever failed, the cell would have to stay cited at
the paper's value rather than record a smaller one.

Usage (from the repository root):
    uv run --extra challenge python research/constructions/qpg_cat.py [m ...]
writes bounds/cat-m{m}-upper-{rank}.json for the requested m in 2..8
(default: all) and prints the numerical residual of each construction.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from stabrank_verify import target_vector  # noqa: E402
from to_witness import witness_from_vectors  # noqa: E402

RANK = {2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 6, 8: 6}


def weights(n):
    return np.array([bin(i).count("1") for i in range(2 ** n)])


def cat(n):
    w = weights(n)
    return np.where(w % 2 == 0, 1j ** (w // 2), 0).astype(complex) / np.sqrt(2 ** (n - 1))


def even_state(n):
    return (weights(n) % 2 == 0).astype(complex) / np.sqrt(2 ** (n - 1))


def ghz_minus_i(n):
    """2^{-1/2}(|0^n> - i|1^n>)."""
    v = np.zeros(2 ** n, dtype=complex)
    v[0], v[-1] = 1, -1j
    return v / np.sqrt(2)


def cat2_terms():
    v = np.zeros(4, dtype=complex)
    v[0], v[3] = 1, 1j
    return [v / np.sqrt(2)]


def cat4_terms():
    return [even_state(4), ghz_minus_i(4)]


def cat6_terms():
    w = weights(6)
    # prod_{i<j} CZ_ij multiplies |x> by (-1)^{C(|x|, 2)}, which is (-1)^{|x|/2}
    # on the even-weight strings that carry |E_6>.
    K6 = even_state(6) * (-1.0) ** (w // 2)
    return [ghz_minus_i(6), even_state(6), K6]


def contract_cat2(u, a, v, b):
    """(I (x) <cat_2| (x) I)(u (x) v) on the last qubit of u and the first of v."""
    bra = np.conj(cat2_terms()[0])
    U = u.reshape(2 ** (a - 1), 2)
    V = v.reshape(2, 2 ** (b - 1))
    out = np.zeros((2 ** (a - 1), 2 ** (b - 1)), dtype=complex)
    for p in range(2):
        for q in range(2):
            out += bra[2 * p + q] * np.outer(U[:, p], V[q, :])
    return out.reshape(-1)


def cat8_terms():
    terms = [contract_cat2(x, 4, y, 6) for x in cat4_terms() for y in cat6_terms()]
    assert all(np.linalg.norm(t) > 1e-9 for t in terms), "a contracted term vanished"
    return terms


def project_first_zero(terms, n):
    """<0|_1 applied to every term of an n-qubit decomposition; the first qubit
    is the most significant bit, and QPG's <0|_m on any one qubit gives the
    same state up to the qubit ordering, which the cat is symmetric under."""
    out = [t.reshape(2, 2 ** (n - 1))[0] for t in terms]
    assert all(np.linalg.norm(t) > 1e-9 for t in out), "a projected term vanished"
    return out


def lstsq_check(target, terms, label):
    A = np.stack(terms, axis=1)
    assert np.linalg.matrix_rank(A, tol=1e-9) == A.shape[1], f"{label}: dependent terms"
    c, *_ = np.linalg.lstsq(A, target, rcond=None)
    resid = np.linalg.norm(A @ c - target)
    print(f"  {label}: {len(terms)} terms, residual {resid:.2e}")
    assert resid < 1e-10, label
    return c


def construct(m):
    if m == 2:
        terms = cat2_terms()
    elif m == 4:
        terms = cat4_terms()
    elif m == 6:
        terms = cat6_terms()
    elif m == 8:
        terms = cat8_terms()
    elif m in (3, 5, 7):
        terms = project_first_zero(construct(m + 1), m + 1)
    else:
        raise ValueError(m)
    lstsq_check(cat(m), terms, f"cat_{m}")
    return terms


NOTES = {
    2: ("chi(|cat_2>) = 1: |cat_2> = 2^{-1/2}(|00> + i|11>) is a stabilizer state "
        "(Qassim, Pashayan, and Gosset, Eq. 5 and Table 1). The one-term witness is "
        "the state itself. The m = 2 cell implies no exponent, since the gluing "
        "identity has denominator m - 2."),
    3: ("chi(|cat_3>) <= 2 from <0|_4 |cat_4> proportional to |cat_3> applied to the "
        "two-term decomposition of |cat_4> (Qassim, Pashayan, and Gosset, appendix, "
        "chi(cat_3) <= chi(cat_4) <= 2; Kissinger, van de Wetering, and Vilmart, "
        "Section 4.1, draw the projected decomposition in ZX). Exact with the "
        "rank-1 exclusion of bounds/cat-m3-lower-2.json. Witness built by "
        "research/constructions/qpg_cat.py by projecting the two cat_4 terms, and "
        "verified in exact arithmetic. Implied H-type exponent log_2(2)/1 = 1."),
    4: ("chi(|cat_4>) <= 2 from |cat_4> = i|E> + ((1-i)/2) 2^{-1/2}(|0^4> - i|1^4>), "
        "|E> the uniform superposition of the even-weight four-bit strings (Qassim, "
        "Pashayan, and Gosset, appendix; Kissinger, van de Wetering, and Vilmart, "
        "Section 4.1, use it as the 2^{0.25 t} decomposition). Exact with "
        "bounds/cat-m4-lower-2.json. Witness built by "
        "research/constructions/qpg_cat.py from the two terms as written, and "
        "verified in exact arithmetic. Implied H-type exponent log_2(2)/2 = 0.5."),
    5: ("chi(|cat_5>) <= 3 from <0|_6 |cat_6> proportional to |cat_5> applied to the "
        "three-term decomposition of |cat_6> (Qassim, Pashayan, and Gosset, "
        "appendix; Kissinger, van de Wetering, and Vilmart, Section 4.1, draw it in "
        "ZX). Exact with their Lemma 3, chi(cat_5) > 2, cited in "
        "bounds/cat-m5-lower-3.json. Witness built by "
        "research/constructions/qpg_cat.py by projecting the three cat_6 terms, and "
        "verified in exact arithmetic. Implied H-type exponent log_2(3)/3 = 0.5283."),
    6: ("chi(|cat_6>) <= 3 from |cat_6> = 2^{-3/2}(|0^6> - i|1^6>) + 2^{-1/2} "
        "e^{3 i pi/4}(|E_6> + i|K_6>), |E_6> the uniform superposition of the "
        "even-weight strings and |K_6> = prod_{i<j} CZ_ij |E_6> (Qassim, Pashayan, "
        "and Gosset, Eq. 5). Through chi(T^6) <= 2 chi(cat_6) this is the rank-6 "
        "cell of the qubit_H orbit, bounds/qubit_H-m6-upper-6.json. Exact with the "
        "appendix's chi(cat_6) >= chi(cat_5) > 2, cited in "
        "bounds/cat-m6-lower-3.json. Witness built by "
        "research/constructions/qpg_cat.py from the three terms as written, and "
        "verified in exact arithmetic. Implied H-type exponent log_2(3)/4 = 0.3963, "
        "the published qubit exponent."),
    7: ("chi(|cat_7>) <= 6 from <0|_8 |cat_8> proportional to |cat_7> applied to the "
        "six-term contraction that decomposes |cat_8> (Qassim, Pashayan, and "
        "Gosset, appendix, chi(cat_7) <= chi(cat_8) <= 6). Witness built by "
        "research/constructions/qpg_cat.py by projecting the six cat_8 terms, and "
        "verified in exact arithmetic; the six projected terms are linearly "
        "independent. Implied H-type exponent log_2(6)/5 = 0.5170."),
    8: ("chi(|cat_8>) <= 6 from <cat_2|_{4,5}(|cat_4> (x) |cat_6>) proportional to "
        "|cat_8> (Qassim, Pashayan, and Gosset, appendix): <cat_2| is a stabilizer "
        "state, so the 2 x 3 contracted products of the cat_4 and cat_6 terms are "
        "a decomposition. Through chi(T^8) <= 2 chi(cat_8) this gives the 12 of "
        "their Table 1 at eight copies. Witness built by "
        "research/constructions/qpg_cat.py; the six contracted terms are nonzero "
        "and linearly independent, and the identity is verified in exact "
        "arithmetic. Implied H-type exponent log_2(6)/6 = 0.4308; rank 5 here would "
        "give 0.3870 and beat the published exponent."),
}


def main(argv):
    ms = [int(a) for a in argv[1:]] or sorted(RANK)
    for m in ms:
        print(f"m = {m}")
        t0 = time.time()
        vecs = construct(m)
        assert len(vecs) == RANK[m], (m, len(vecs))
        tgt = np.array([complex(x) for x in target_vector("cat", m)]).ravel()
        assert np.allclose(tgt, cat(m)), "closed form disagrees with the verifier's target"
        w = witness_from_vectors("cat", m, vecs)
        print(f"  exact witness with {len(w['terms'])} terms in {time.time() - t0:.1f} s")
        sub = {
            "schema_version": "0.1", "orbit": "cat", "m": m,
            "direction": "upper", "rank": RANK[m],
            "provenance": {
                "author": "Qassim, Pashayan, and Gosset",
                "reference": "arXiv:2106.07740",
                "method": "literature",
                "date": "2021-06-14",
                "github": [],
            },
            "notes": NOTES[m],
            "witness": w,
        }
        out = os.path.join(ROOT, "bounds", f"cat-m{m}-upper-{RANK[m]}.json")
        with open(out, "w") as f:
            json.dump(sub, f, indent=2)
            f.write("\n")
        print(f"  written {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
