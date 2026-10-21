"""The sqrt(T) and CS orbits: states, width plumbing, certificates, and cells.

`qubit_sqrtT` is a third single-qubit orbit, (|0> + e^{i pi/8}|1>)/sqrt 2,
and `CS` is the first width-2 orbit, CS|++> on two qubits with m copies on
2m qubits. These tests pin the two states, the width table and every place
the pipeline derives a qubit count from m (the verifier, the coefficient
fit, the witness converter, the symmetry reduction, the dictionaries the
certificate driver picks, the slice-and-lift dispatch), the block
slice-and-lift against its controls, the exact arguments behind the lower
bounds (Gaussian-integer pair search for CS, Galois descent for sqrt(T)),
and the committed bound files.
"""

import glob
import itertools
import json
import os
import sys

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, ROOT)

from stabrank_verify import (ORBIT_P, ORBIT_WIDTH, orbit_qudits, orbit_state,  # noqa: E402
                             stabilizer_vector, target_vector, verify)
from rank_exclusion import (_copy_permutation, dictionary, local_clifford_group,  # noqa: E402
                            psi_for, qubit_clifford_order, rank2_search,
                            symmetry_orbit_reps)


def _load(name):
    return json.load(open(os.path.join(ROOT, "bounds", name + ".json")))


def _num(v):
    return np.array([complex(sp.N(z, 20)) for z in v]).ravel()


# ------------------------------------------------------------- the states ---

def test_sqrtt_state():
    v = orbit_state("qubit_sqrtT")
    assert ORBIT_P["qubit_sqrtT"] == 2 and orbit_qudits("qubit_sqrtT", 3) == 3
    assert v.shape == (2, 1)
    assert sp.simplify(sum(abs(z) ** 2 for z in v) - 1) == 0
    assert np.allclose(_num(v), np.array([1, np.exp(1j * np.pi / 8)]) / np.sqrt(2))
    # not a stabilizer state, and not in the H-type or T-type orbits: its
    # Bloch vector is (cos(pi/8), sin(pi/8), 0), which no signed coordinate
    # permutation carries to an edge center (1, 1, 0)/sqrt 2 or a face center
    D = dictionary(2, 1)
    assert np.max(np.abs(_num(v).conj() @ D)) < 1 - 1e-6
    bloch = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0])
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            img = np.array(signs) * bloch[list(perm)]
            assert not np.allclose(np.abs(img), [1, 1, 0] / np.sqrt(2))
            assert not np.allclose(np.abs(img), [1, 1, 1] / np.sqrt(3))


def test_cs_state_and_width():
    v = orbit_state("CS")
    assert ORBIT_P["CS"] == 2 and ORBIT_WIDTH["CS"] == 2
    assert list(v) == [sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2), sp.I / 2]
    assert orbit_qudits("CS", 3) == 6 and orbit_qudits("qubit_H", 3) == 3
    assert target_vector("CS", 2).shape == (16, 1)
    assert target_vector("CS", 3).shape == (64, 1)
    # CS|++>: the phase i^(x1 x2) on the full support
    cs = np.diag([1, 1, 1, 1j]) @ (np.ones(4) / 2)
    assert np.allclose(_num(v), cs)
    D = dictionary(2, 2)
    assert np.max(np.abs(_num(v).conj() @ D)) < 1 - 1e-6


# ------------------------------------------------------------ the plumbing ---

def test_verify_upper_uses_width():
    """A CS witness is parametrized over 2m qubits; the committed two-copy
    witness verifies and a term of the wrong length is refused."""
    sub = _load("CS-m2-upper-3")
    r = verify(sub)
    assert r.ok and r.tier == "verified" and "symbolically" in r.detail
    assert abs(r.gamma - np.log2(3) / 2) < 1e-12
    bad = json.loads(json.dumps(sub))
    bad["witness"]["terms"][0]["x0"] = [0, 0]           # length m, not 2m
    assert not verify(bad).ok


def test_fit_and_witness_converter_use_width():
    from fit_coeffs import fit
    from to_witness import witness_from_vectors
    sub = _load("CS-m1-upper-2")
    coeffs = fit("CS", 1, sub["witness"]["terms"])
    assert [str(c) for c in coeffs] == sub["witness"]["coeffs"]
    pp = np.ones(4) / 2
    cz = np.array([1, 1, 1, -1]) / 2
    w = witness_from_vectors("CS", 1, [pp, cz])
    assert len(w["terms"]) == 2 and all(len(t["x0"]) == 2 for t in w["terms"])


def test_sqrtt_fit_is_cyclotomic():
    """The sqrt(T) coefficients are solved inside Q(zeta_16): short
    polynomials in exp(I*pi/8), and the four-copy witness, which the generic
    sympy route mis-simplifies, verifies symbolically."""
    from fit_coeffs import CYCLOTOMIC, fit
    assert CYCLOTOMIC["qubit_sqrtT"] == (2, 4, 16)
    sub = _load("qubit_sqrtT-m4-upper-5")
    coeffs = fit("qubit_sqrtT", 4, sub["witness"]["terms"])
    assert [str(c) for c in coeffs] == sub["witness"]["coeffs"]
    r = verify(sub)
    assert r.ok and r.tier == "verified" and "symbolically" in r.detail


def test_local_clifford_group_and_block_permutation():
    assert qubit_clifford_order(1) == 24 and qubit_clifford_order(2) == 11520
    G = local_clifford_group("CS")
    assert len(G) == 11520
    assert len(local_clifford_group("qubit_sqrtT")) == 24
    cs = _num(orbit_state("CS"))
    uni = [U for U in G if abs(abs(np.vdot(cs, U @ cs)) - 1) < 1e-9]
    assert len(uni) == 24
    # swapping two blocks of width 2 is the qubit permutation (2, 3, 0, 1)
    P = _copy_permutation((1, 0), 2, 2, width=2)
    psi = np.kron(np.arange(1, 5), np.arange(5, 9)).astype(float)
    assert np.allclose(P @ psi, np.kron(np.arange(5, 9), np.arange(1, 5)))
    # width 1 is unchanged
    assert np.allclose(_copy_permutation((1, 0), 2, 2), _copy_permutation((1, 0), 2, 2, width=1))


def test_cs_symmetry_reduction():
    """The symmetry group of |CS>^m: 24 local Cliffords per copy, the block
    permutations and the antiunitary CZ o conjugation; every generator is
    checked inside symmetry_orbit_reps to fix the target and permute the
    dictionary, so building the representatives is the test."""
    D2 = dictionary(2, 2)
    reps, info = symmetry_orbit_reps("CS", 1, D2)
    assert info["local"] == 24 and info["antiunitary"] and info["order"] == 48
    assert info["orbits"] == 6
    with pytest.raises(ValueError):
        symmetry_orbit_reps("CS", 2, D2)        # two copies need the 4-qubit dictionary


def test_certificate_driver_picks_width_dictionary(capsys):
    """run_certificate's dictionary for (CS, m=1) is the two-qubit one."""
    from rank_exclusion import run_certificate
    code = run_certificate(cells=[], controls=[], workers=1)
    assert code == 0
    # a cell with a known rank-2 decomposition fails the rank-3 exclusion, as it should
    code = run_certificate(cells=[("CS", 1)], controls=[], workers=1)
    out = capsys.readouterr().out
    assert code == 1 and "2 qubits: 60 stabilizer states" in out


# ------------------------------------------------------- the block lift -----

def test_block_lift_agrees_with_qubit_lift_at_width_one():
    from slice_lift import all_decompositions, lifts_qubit, lifts_qubit_block
    for orbit, m, rank, want in (("qubit_T", 3, 3, True), ("qubit_H", 2, 2, False)):
        D = dictionary(2, m)
        alpha, psi = _num(orbit_state(orbit)), psi_for(orbit, m)
        decs, _ = all_decompositions(orbit, m, rank, D, verbose=False)
        assert decs
        plain = block = 0
        for cols in decs:
            states = [D[:, c] for c in cols]
            d, *_ = np.linalg.lstsq(np.column_stack(states), psi, rcond=None)
            plain += len(lifts_qubit(states, d, alpha, psi, m)[0])
            block += len(lifts_qubit_block(states, d, alpha, psi, m, 1)[0])
        assert plain == block and (plain > 0) == want


def test_block_lift_controls():
    """|T>(x)|+> as a width-2 state keeps rank 2 at two copies, so its rank-2
    decompositions lift; |CS> has rank 3 at two copies, so its do not."""
    from slice_lift import confirm_lift, decompositions_with_pivot, lifts_qubit_block
    D2 = dictionary(2, 2)
    tplus = np.kron(_num(orbit_state("qubit_T")), np.ones(2) / np.sqrt(2))
    cs = _num(orbit_state("CS"))
    for alpha, want in ((tplus, True), (cs, False)):
        decs = set()
        for i in range(D2.shape[1]):
            decs.update(decompositions_with_pivot(alpha, D2, i, 2))
        assert decs
        lifts = 0
        for cols in sorted(decs):
            states = [D2[:, c] for c in cols]
            d, *_ = np.linalg.lstsq(np.column_stack(states), alpha, rcond=None)
            L, _ = lifts_qubit_block(states, d, alpha, alpha, 2, 2)
            for terms in L:
                assert confirm_lift(terms, alpha, alpha) < 1e-8
            lifts += len(L)
        assert (lifts > 0) == want


def test_block_lift_needs_nonzero_amplitudes():
    from slice_lift import lifts_qubit_block
    D2 = dictionary(2, 2)
    with pytest.raises(ValueError):
        lifts_qubit_block([D2[:, 0]], [1.0], np.array([1, 0, 0, 1]) / np.sqrt(2), D2[:, 0], 2, 2)


def test_lift_all_dispatches_on_width():
    from slice_lift import all_decompositions, lift_all
    D2 = dictionary(2, 2)
    decs, _ = all_decompositions("CS", 1, 2, D2, verbose=False)
    assert decs
    assert lift_all("CS", 1, decs, D2, verbose=False) == []


def test_block_lift_for_cs_is_decided_by_the_stabilizer_test():
    """For CS the amplitude ratios 1, 1, i are fourth roots of unity, so the
    identity assignment solves every slice equation of every base
    decomposition; the matcher must find exactly that solution per slice and
    the assembled candidate |CS> (x) u_i must be rejected as not a stabilizer
    state. That rejection, not the slice equations, is what excludes the lift,
    and the statistics lift_all reports must show it."""
    from slice_lift import all_decompositions, lift_all
    D2 = dictionary(2, 2)
    decs, _ = all_decompositions("CS", 1, 2, D2, verbose=False)
    stats = []
    assert lift_all("CS", 1, decs, D2, verbose=False, stats_out=stats) == []
    assert len(stats) == len(decs) >= 1
    for st in stats:
        assert st["matches"] == [1, 1, 1] and st["identity_only"]
        assert st["candidates"] == 1 and len(st["rejected"]) == 1
        assert "not a Z_4 quadratic form" in st["rejected"][0][2]
    # the positive control keeps a candidate: the T-type state with |+> lifts
    from slice_lift import decompositions_with_pivot, lifts_qubit_block
    tplus = np.kron(_num(orbit_state("qubit_T")), np.ones(2) / np.sqrt(2))
    kept = 0
    for i in range(D2.shape[1]):
        for cols in decompositions_with_pivot(tplus, D2, i, 2):
            states = [D2[:, c] for c in cols]
            d, *_ = np.linalg.lstsq(np.column_stack(states), tplus, rcond=None)
            st = {}
            L, _ = lifts_qubit_block(states, d, tplus, tplus, 2, 2, stats=st)
            kept += len(L)
            assert len(L) == st["candidates"] - len({c for c, _, _ in st["rejected"]})
    assert kept > 0


# ------------------------------------------------------ the exact arguments ---

def test_cs_exact_enumeration_and_pair_search():
    import cert_cs_m2_rank2 as c
    S2 = c.stabilizer_states(2)
    assert len(S2) == c.count(2) == 60
    assert len({c.canonical(s) for s in S2}) == 60
    assert len(c.stabilizer_states(3)) == c.count(3) == 1080
    cs = ((1, 0), (1, 0), (1, 0), (0, 1))
    rank1, pairs = c.exact_search(cs, S2)
    assert rank1 == [] and len(pairs) == 15
    plus, cz = ((1, 0),) * 4, ((1, 0), (1, 0), (1, 0), (-1, 0))
    assert tuple(sorted((S2.index(plus), S2.index(cz)))) in {tuple(sorted(p)) for p in pairs}
    rank1, _ = c.exact_search(plus, S2)
    assert rank1 == [S2.index(plus)]
    # the numerical pair search agrees on both counts
    num, _ = rank2_search(psi_for("CS", 1), dictionary(2, 2))
    assert len(num) == 15


def test_sqrtt_galois_rank():
    import cert_sqrtt_galois as g
    K = sp.QQ.algebraic_field(g.ZETA)
    for m in range(1, 6):
        assert g.conjugate_matrix(K, m).rank() == min(4, m + 1)
    assert sorted(g.CONJ) == [k for k in range(1, 16, 2) if (4 * k) % 16 == 4]


# ------------------------------------------------------------- the cells ----

NEW_FILES = sorted(os.path.basename(p)[:-5] for pat in ("CS-*.json", "qubit_sqrtT-*.json")
                   for p in glob.glob(os.path.join(ROOT, "bounds", pat)))
CELLS = {("CS", 1): (2, 2), ("CS", 2): (3, 3), ("CS", 3): (4, 6),
         ("qubit_sqrtT", 1): (2, 2), ("qubit_sqrtT", 2): (3, 3),
         ("qubit_sqrtT", 3): (4, 4), ("qubit_sqrtT", 4): (4, 5)}


def test_new_bound_files_exist():
    assert NEW_FILES == [
        "CS-m1-lower-2", "CS-m1-upper-2", "CS-m2-lower-3", "CS-m2-upper-3",
        "CS-m3-lower-3", "CS-m3-lower-4", "CS-m3-upper-6",
        "qubit_sqrtT-m1-lower-2", "qubit_sqrtT-m1-upper-2", "qubit_sqrtT-m2-lower-3",
        "qubit_sqrtT-m2-upper-3", "qubit_sqrtT-m3-lower-4", "qubit_sqrtT-m3-upper-4",
        "qubit_sqrtT-m4-lower-4", "qubit_sqrtT-m4-upper-5"]


@pytest.mark.parametrize("name", NEW_FILES)
def test_new_bound_files_validate(name):
    import jsonschema
    from validate_bounds import SCHEMA, tier_requirements
    schema = json.load(open(SCHEMA))
    assert {"qubit_sqrtT", "CS"} <= set(schema["properties"]["orbit"]["enum"])
    sub = _load(name)
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(sub))
    errors, _ = tier_requirements(sub)
    assert errors == []
    lo, up = CELLS[(sub["orbit"], sub["m"])]
    if sub["direction"] == "upper":
        assert sub["rank"] == up
    else:
        assert sub["rank"] <= lo


def test_best_bounds_per_cell():
    best = {}
    for name in NEW_FILES:
        sub = _load(name)
        key = (sub["orbit"], sub["m"])
        lo, up = best.get(key, (0, 10 ** 9))
        if sub["direction"] == "upper":
            up = min(up, sub["rank"])
        else:
            lo = max(lo, sub["rank"])
        best[key] = (lo, up)
    assert best == CELLS


@pytest.mark.parametrize("name", [n for n in NEW_FILES if "upper" in n])
def test_new_upper_witnesses_verify_symbolically(name):
    r = verify(_load(name))
    assert r.ok and r.tier == "verified" and "symbolically" in r.detail


@pytest.mark.parametrize("name", [n for n in NEW_FILES if "lower" in n])
def test_new_certificate_claims_match(name):
    """Each lower bound's claim string is printed by its script, which
    names it in its docstring, and `exact` is declared exactly where the
    argument is exact."""
    sub = _load(name)
    cert = sub["certificate"]
    src = open(os.path.join(ROOT, cert["script"])).read()
    assert cert["expect"] in src.split('"""')[1], "the docstring must list the claim"
    expect = cert["expect"]
    assert (f'print("{expect}")' in src
            or (expect.startswith("CERTIFIED chi(qubit_sqrtT^")
                and 'print(f"CERTIFIED chi(qubit_sqrtT^{m}) >= {r}")' in src))
    assert cert.get("exact", False) == (cert["script"] != "verify_challenge/cert_cs_m3_lift.py")


def test_cs_m_cap():
    from validate_bounds import tier_requirements
    sub = {"schema_version": "0.1", "orbit": "CS", "m": 5, "direction": "upper",
           "rank": 100, "provenance": {"author": "x", "date": "2026-10-21"}}
    errors, _ = tier_requirements(sub)
    assert any("m <= 4" in e for e in errors)
    sub["m"] = 4
    assert tier_requirements(sub)[0] == []
    sub["orbit"] = "qubit_sqrtT"
    sub["m"] = 10
    assert tier_requirements(sub)[0] == []


def test_site_constants_cover_the_orbits():
    sys.path.insert(0, os.path.join(ROOT, "site_challenge"))
    import build as site
    for orbit in ("qubit_sqrtT", "CS"):
        assert orbit in site.ORBIT_ORDER
        for table in (site.BASELINE, site.SYSTEM, site.ORBIT_TEX, site.BASE_TEX,
                      site.ORBIT_DEF, site.COLOR, site.ORB_TEX_NAME):
            assert orbit in table
    assert site.BASELINE["qubit_sqrtT"][0] == 0.5 and site.BASELINE["CS"][0] == 1.0
    assert "CS" in site.UNPUBLISHED and "qubit_sqrtT" not in site.UNPUBLISHED
    assert site.SYSTEM["CS"] == "two-qubit"
    # the next cell that would beat the sqrt(T) baseline is rank 5 at m = 5
    cells = {("qubit_sqrtT", m, "lower"): {"sub": {"rank": r}}
             for m, r in ((1, 2), (2, 3), (3, 4), (4, 4))}
    tgt = site.next_target("qubit_sqrtT", 0.5, cells, lead="")
    assert "5 at m=5" in tgt
