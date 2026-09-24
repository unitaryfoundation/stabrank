"""The cat family track: target, exponent rule, bound files, and certificate.

|cat_m> = (|T>^m + |T_perp>^m)/sqrt 2 is one m-qubit state per m, not a
tensor power, so the verifier builds it directly and the board shows the
exponent it implies for the qubit orbits, log_2(rank)/(m - 2), instead of a
per-copy exponent. These tests pin the definition to the paper's (Qassim,
Pashayan, and Gosset, arXiv:2106.07740, Eq. 3), the exponent rule, the
gluing identity behind it, the consistency of the cat cells with the
qubit_H cells through chi(T^m)/2 <= chi(cat_m) <= chi(T^m) (their Eq. 4),
and the thirteen committed bound files.
"""

import glob
import json
import os
import sys

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "site_challenge"))

from stabrank_verify import (FAMILY, ORBIT_P, cat_vector, exponent_copies,  # noqa: E402
                             implied_exponent, orbit_state, target_vector, verify)

CAT_FILES = sorted(os.path.basename(p)[:-5]
                   for p in glob.glob(os.path.join(ROOT, "bounds", "cat-*.json")))
UPPER = {2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 6, 8: 6}
LOWER = {3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3}


def _load(name):
    return json.load(open(os.path.join(ROOT, "bounds", name + ".json")))


def _power(v, m):
    out = v
    for _ in range(m - 1):
        out = sp.Matrix(sp.kronecker_product(out, v))
    return out


# ------------------------------------------------------------ the target ---

@pytest.mark.parametrize("m", range(1, 7))
def test_cat_closed_form_matches_the_definition(m):
    """2^{-(m-1)/2} sum_{|x| even} i^{|x|/2}|x> equals (|T>^m + (Z|T>)^m)/sqrt 2
    symbolically, with |T> = (|0> + e^{i pi/4}|1>)/sqrt 2 (QPG Eq. 3)."""
    T = sp.Matrix([1, sp.exp(sp.I * sp.pi / 4)]) / sp.sqrt(2)
    Tp = sp.Matrix([1, -sp.exp(sp.I * sp.pi / 4)]) / sp.sqrt(2)
    defn = (_power(T, m) + _power(Tp, m)) / sp.sqrt(2)
    diff = (cat_vector(m) - defn).applyfunc(lambda z: sp.simplify(sp.expand_complex(z)))
    assert diff == sp.zeros(2 ** m, 1)
    assert sp.simplify(sum(abs(z) ** 2 for z in cat_vector(m))) == 1


def test_target_vector_dispatches_to_the_family():
    assert "cat" in FAMILY and ORBIT_P["cat"] == 2
    assert target_vector("cat", 4) == cat_vector(4)
    with pytest.raises(ValueError):
        orbit_state("cat")
    # the copy orbits are untouched
    assert target_vector("qubit_H", 2) == sp.simplify(_power(orbit_state("qubit_H"), 2))


def test_small_cats_are_stabilizer_states():
    """|cat_1> = |0> and |cat_2> = (|00> + i|11>)/sqrt 2 (QPG Eq. 5)."""
    assert list(cat_vector(1)) == [1, 0]
    assert list(cat_vector(2)) == [1 / sp.sqrt(2), 0, 0, sp.I / sp.sqrt(2)]


# ---------------------------------------------------------- the exponent ---

def test_exponent_rule():
    assert exponent_copies("cat", 2) is None
    assert exponent_copies("cat", 6) == 4
    assert exponent_copies("qubit_H", 6) == 6
    assert implied_exponent("cat", 1, 2) is None
    assert abs(implied_exponent("cat", 3, 6) - np.log2(3) / 4) < 1e-12
    assert abs(implied_exponent("cat", 5, 8) - np.log2(5) / 6) < 1e-12
    assert abs(implied_exponent("qubit_H", 6, 6) - np.log2(6) / 6) < 1e-12
    assert abs(implied_exponent("T5", 8, 2) - np.log(8) / np.log(5) / 2) < 1e-12


def _cat_np(m):
    return np.array([complex(z) for z in cat_vector(m)])


@pytest.mark.parametrize("a, b", [(2, 2), (2, 4), (4, 4), (4, 6), (6, 6)])
def test_gluing_identity(a, b):
    """(I (x) <cat_2| (x) I)(|cat_a> (x) |cat_b>) = (1/2)|cat_{a+b-2}>, the
    identity behind the implied exponent (QPG, proof of Theorem 1)."""
    bra = np.conj(_cat_np(2))
    U = _cat_np(a).reshape(2 ** (a - 1), 2)
    V = _cat_np(b).reshape(2, 2 ** (b - 1))
    out = sum(bra[2 * p + q] * np.outer(U[:, p], V[q, :]) for p in range(2) for q in range(2))
    assert np.allclose(out.reshape(-1), _cat_np(a + b - 2) / 2)


def test_projection_and_bra_relations():
    """<0|_1 |cat_m> is proportional to |cat_{m-1}> (monotonicity) and
    |cat_m> = 2^{-1/2}(I + Z^m)|T>^m (the upper half of QPG Eq. 4)."""
    for m in range(3, 8):
        proj = _cat_np(m).reshape(2, -1)[0]
        assert np.allclose(proj, _cat_np(m - 1) / np.sqrt(2))
    m = 5
    T = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
    Tm = np.array([1.0 + 0j])
    for _ in range(m):
        Tm = np.kron(Tm, T)
    par = np.array([(-1) ** bin(i).count("1") for i in range(2 ** m)])
    assert np.allclose((Tm + par * Tm) / np.sqrt(2), _cat_np(m))


# --------------------------------------------------------- the bound files ---

def test_cat_files_present():
    assert CAT_FILES == sorted([f"cat-m{m}-upper-{r}" for m, r in UPPER.items()]
                               + [f"cat-m{m}-lower-{r}" for m, r in LOWER.items()])


@pytest.mark.parametrize("name", CAT_FILES)
def test_cat_files_validate(name):
    import jsonschema
    from validate_bounds import SCHEMA, tier_requirements
    schema = json.load(open(SCHEMA))
    assert "cat" in schema["properties"]["orbit"]["enum"]
    sub = _load(name)
    assert sub["orbit"] == "cat"
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(sub))
    errors, _ = tier_requirements(sub)
    assert errors == []
    assert "2106.07740" in sub["provenance"]["reference"] or \
        sub["provenance"]["reference"] == "stabrank verify_challenge"


def test_cat_m_min():
    from validate_bounds import tier_requirements
    sub = {"schema_version": "0.1", "orbit": "cat", "m": 1, "direction": "upper",
           "rank": 1, "provenance": {"author": "x", "date": "2026-09-24"}}
    errors, _ = tier_requirements(sub)
    assert any("m = 2" in e for e in errors)
    sub["m"] = 2
    assert tier_requirements(sub)[0] == []


@pytest.mark.parametrize("m", sorted(UPPER))
def test_cat_upper_witnesses_verify(m):
    """The published decompositions, replayed: every upper file carries a
    witness that the verifier confirms symbolically at the paper's rank."""
    sub = _load(f"cat-m{m}-upper-{UPPER[m]}")
    assert sub["provenance"]["method"] == "literature"
    r = verify(sub)
    assert r.ok and r.tier == "verified", r
    assert len(sub["witness"]["terms"]) == UPPER[m]
    if m == 2:
        assert r.gamma is None
    else:
        assert abs(r.gamma - np.log2(UPPER[m]) / (m - 2)) < 1e-12


def test_cat_lower_tiers():
    """m = 3, 4 are exact certificates; m = 5 to 8 are citations of QPG's
    appendix and cannot hold a record."""
    for m in (3, 4):
        sub = _load(f"cat-m{m}-lower-2")
        assert sub["certificate"]["exact"] is True
        assert sub["certificate"]["script"] == "verify_challenge/cert_family_rank1.py"
        assert sub["certificate"]["expect"] == f"CERTIFIED chi(cat_{m}) >= 2"
    for m in (5, 6, 7, 8):
        sub = _load(f"cat-m{m}-lower-3")
        assert "certificate" not in sub and "lean" not in sub
        assert verify(sub).tier == "cited"


def test_cat_cells_consistent_with_qubit_H():
    """chi(T^m)/2 <= chi(cat_m) <= chi(T^m) (QPG Eq. 4) with |T> and |H> one
    Clifford orbit: the committed intervals must satisfy
    cat_lower <= H_upper and H_lower <= 2 cat_upper at every common m."""
    best = {}
    for path in glob.glob(os.path.join(ROOT, "bounds", "*.json")):
        sub = json.load(open(path))
        if sub["orbit"] not in ("cat", "qubit_H"):
            continue
        key = (sub["orbit"], int(sub["m"]), sub["direction"])
        r = int(sub["rank"])
        cur = best.get(key)
        if cur is None or (sub["direction"] == "upper" and r < cur) or \
                (sub["direction"] == "lower" and r > cur):
            best[key] = r
    common = sorted({k[1] for k in best if k[0] == "cat"} & {k[1] for k in best if k[0] == "qubit_H"})
    assert common == [2, 3, 4, 5, 6, 7, 8]
    for m in common:
        cat_lo, cat_up = best.get(("cat", m, "lower"), 1), best[("cat", m, "upper")]
        h_lo, h_up = best.get(("qubit_H", m, "lower"), 1), best[("qubit_H", m, "upper")]
        assert cat_lo <= h_up, (m, cat_lo, h_up)
        assert h_lo <= 2 * cat_up, (m, h_lo, cat_up)
        assert cat_up <= h_up, (m, cat_up, h_up)   # best-known uppers respect Eq. 4 too


# --------------------------------------------------------- the certificate ---

def test_rank1_certificate_refutations():
    import cert_family_rank1 as c
    assert c.refutation(cat_vector(1)) is None
    assert c.refutation(cat_vector(2)) is None
    assert "quadratic form" in c.refutation(cat_vector(3))
    assert "quadratic form" in c.refutation(cat_vector(4))
    assert "quadratic form" in c.refutation(cat_vector(5))
    # a W-like support of three points is refuted on the support test
    v = sp.Matrix([0, 1, 1, 0, 1, 0, 0, 0]) / sp.sqrt(3)
    assert "not a power of two" in c.refutation(v)
    # a non-fourth-root ratio is refuted first
    v = sp.Matrix([1, sp.exp(sp.I * sp.pi / 4), 0, 0]) / sp.sqrt(2)
    assert "fourth root" in c.refutation(v)
    # a genuine stabilizer state with Y-type phases passes
    y = sp.Matrix([1, sp.I]) / sp.sqrt(2)
    assert c.refutation(_power(y, 2)) is None
    src = open(os.path.join(ROOT, "verify_challenge", "cert_family_rank1.py")).read()
    for m in (3, 4):
        assert f"CERTIFIED chi(cat_{m}) >= 2" in src


# ----------------------------------------------------------------- the site ---

def test_site_constants_and_ket():
    import build as site
    assert "cat" in site.ORBIT_ORDER
    for table in (site.BASELINE, site.SYSTEM, site.ORBIT_TEX, site.BASE_TEX,
                  site.ORBIT_DEF, site.COLOR):
        assert "cat" in table
    tensor = "&#x02297;"                       # the MathML entity for \otimes
    k = site.ket("cat", 6, "&le;", 3)
    assert "<mtext>cat</mtext>" in k and tensor not in k
    assert tensor in site.ket("qubit_H", 6, "&le;", 6)
    # the target rule: at m the candidate rank is the largest below
    # 2^{base (m-2)} that is not under the board's lower bound at m, and the
    # lowest implied exponent among those wins; with the QPG lower bounds in
    # place that is chi(cat_7) <= 3 (log_2(3)/5 = 0.317). Without them the
    # rule would name the impossible chi(cat_6) <= 2.
    base = site.BASELINE["cat"][0]
    cells = {("cat", m, "lower"): {"sub": {"rank": r}} for m, r in LOWER.items()}
    tgt = site.next_target("cat", base, cells=cells, lead="")
    assert "m=7" in tgt and "&le; 3" in tgt and "0.3170" in tgt
    assert "m=6" in site.next_target("cat", base, cells=None)
