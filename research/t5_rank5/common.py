"""Shared setup for the batch pipeline of the rank-5 exclusion of |T>^5
(docs/notes/t5_rank5_exclusion.md): the cell (n_1 = 2 sliced qubits, the
single base point 00, the qubit_T orbit over Q(zeta_24)), the six flats of
F_2^2 that miss 00 and the 21 flat pairs, the lists every stage draws its
bases from (the kernel census of the full 5-covers of |T>^3 per pivot pair,
the degenerate 5-multisets, the full 4-covers of the rank-4 record, the
full 3-covers), canonical hashing, the partition loaders with their hash
checks, and the exact re-decision of a stored hit against psi_5 = |T>^5.

This module is imported as `common` by driver.py, batch.py and aggregate.py
from research/t5_rank5 only; no other research directory is put on the
path, and the constructions module research/constructions/common.py is
loaded by file.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from slice_cover import (FOURTH, P2, CoverEnumerator, Field, _rank_mod,  # noqa: E402
                         confirm_decomposition, exact_codes)


def _constructions_common():
    """research/constructions/common.py, loaded by file so that its name
    never collides with this module."""
    path = os.path.join(ROOT, "research", "constructions", "common.py")
    spec = importlib.util.spec_from_file_location("stabrank_constructions_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cc = _constructions_common()
target = _cc.target                            # |M>^m as a vector
term_vector = _cc.term_vector                  # a witness term (k, x0, W, Q, l) as a vector
load_decompositions = _cc.load_decompositions  # the stored rank-r decompositions of |M>^m

N1 = 2                      # sliced qubits
N2 = 3                      # unsliced qubits (the base slice is a cover of psi_3)
M = 5                       # copies of |T>
RANK = 5
ORBIT = "qubit_T"
X0 = 0b00                   # the single base point of the design
X0S = (X0,)
STAGES = ("A", "B", "C", "beta", "gamma")
PARTITION = os.path.join(HERE, "partition.json")
RESULTS = os.path.join(HERE, "results")
CENSUS = os.path.join(RESULTS, "census5_full.json")            # the 5-cover census of psi_3 per pivot pair
DEGENERATE = os.path.join(HERE, "degenerate5.json")            # dependent and repeated 5-covers (stages B, C)
COVERS4 = os.path.join(ROOT, "research", "t5_rank4", "covers4.json")   # full 4-covers of psi_3 (stage beta)
RATES = os.path.join(RESULTS, "rates.json")
NUM_TOL = 1e-8
CLAIM = "CERTIFIED chi(qubit_T^5) >= 6"
CLAIM_M6 = "CERTIFIED chi(qubit_T^6) >= 6"

# Points of F_2^2 along the sliced qubits 1, 2 are x = 2 x_1 + x_2, so 1 = 01,
# 2 = 10, 3 = 11. The six affine flats missing 00: three points and three
# lines (B the diagonal {01, 10}, C1 the line x_1 = 1, D1 the line x_2 = 1).
FLATS = {"p01": (1,), "p10": (2,), "p11": (3,), "B": (1, 2), "C1": (2, 3), "D1": (1, 3)}
FLAT_NAMES = list(FLATS)                                          # the order used everywhere
FLAT_PAIRS = [(a, b) for i, a in enumerate(FLAT_NAMES) for b in FLAT_NAMES[i:]]   # 21 multisets
RUNS_PER_COVER = {"A": 1, "B": 1, "C": 1, "beta": len(FLAT_NAMES), "gamma": len(FLAT_PAIRS)}


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_json(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lower_priority():
    """nice 19 for this process; a no-op when the priority is already at the
    floor (macOS raises PermissionError there, Linux clamps)."""
    try:
        os.nice(19)
    except OSError:
        pass


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_hashed(path, body):
    """Write `body` with its own sha256 appended, atomically."""
    body = dict(body)
    body.pop("sha256", None)
    body["sha256"] = sha256_json(body)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(body, f)
        f.write("\n")
    os.replace(tmp, path)
    return body["sha256"]


def load_hashed(path, what):
    with open(path) as f:
        doc = json.load(f)
    body = {k: v for k, v in doc.items() if k != "sha256"}
    if sha256_json(body) != doc.get("sha256"):
        raise ValueError(f"{what} {path}: sha256 does not match its content")
    return doc


def load_partition(path=PARTITION):
    part = load_hashed(path, "partition")
    for key in ("batch_geometry", "census", "degenerate", "covers4", "covers3"):
        if key not in part:
            raise ValueError(f"{path} lacks `{key}`; rerun driver.py partition")
    return part


def load_degenerate(path=DEGENERATE, expect_sha256=None):
    """The stored dependent and repeated 5-covers (stages B and C) as sorted
    tuples, with the record; the sha256 must equal the partition's."""
    doc = load_hashed(path, "degenerate cover list")
    if expect_sha256 is not None and doc["sha256"] != expect_sha256:
        raise ValueError(f"{path}: sha256 {doc['sha256'][:16]} differs from the partition's "
                         f"{expect_sha256[:16]}")
    return [tuple(int(x) for x in c) for c in doc["covers"]], doc


def load_covers4(path=COVERS4, expect_sha256=None):
    """The stage (beta) bases: every full 4-cover of psi_3 up to G_3 from the
    rank-4 record (4,697 of distinct independent states, 12 with a repeated
    state), as sorted tuples with their kinds and the record."""
    doc = load_hashed(path, "4-cover census")
    if expect_sha256 is not None and doc["sha256"] != expect_sha256:
        raise ValueError(f"{path}: sha256 {doc['sha256'][:16]} differs from the partition's "
                         f"{expect_sha256[:16]}")
    covers = [tuple(int(x) for x in c) for c in doc["covers"]]
    if len(doc["kinds"]) != len(covers):
        raise ValueError(f"{path}: kinds and covers differ in length")
    return covers, doc


def covers3_of(doc4):
    """The full 3-covers stored with the 4-cover census (the stage (gamma)
    bases), as sorted tuples."""
    return [tuple(int(x) for x in T) for T in doc4["covers3"]]


def load_census(path=CENSUS, expect_sha256=None):
    """The 5-cover census of psi_3 (covers, candidates and kernel seconds per
    pivot pair; the covers themselves are regenerated by the kernel), checked
    against the file hash the partition records."""
    if expect_sha256 is not None:
        got = sha256_file(path)
        if got != expect_sha256:
            raise ValueError(f"{path}: file sha256 {got[:16]} differs from the partition's "
                             f"{expect_sha256[:16]}")
    with open(path) as f:
        return json.load(f)


def batch_geometry(part, index):
    """The geometry entry with the given batch index."""
    geos = part["batch_geometry"]
    if 0 <= index < len(geos) and geos[index]["index"] == index:
        return geos[index]
    for geo in geos:
        if geo["index"] == index:
            return geo
    raise IndexError(f"batch index {index} not in the partition (0..{len(geos) - 1})")


def make_enumerator(native=True):
    return CoverEnumerator(N2, native=native, orbit=ORBIT)


def pairs_of(E):
    """Every (pivot, partner, member count) unit of the 5-cover enumeration,
    in the pivot order of the enumerator."""
    out = []
    for i in E.reps:
        i = int(i)
        members, partners = E.pivot_plan(i)
        for j in partners:
            j = int(j)
            Mc = int(np.count_nonzero(members > j)) - (1 if i > j else 0)
            out.append((i, j, Mc))
    return out


def multiplicity_pattern(cover):
    return tuple(sorted((cover.count(u) for u in set(cover)), reverse=True))


def stage_of(cover):
    """B for a dependent 5-cover of five distinct states, C for a 5-cover
    with a repeated state, beta for a 4-multiset, gamma for a 3-cover (A, a
    5-cover of distinct independent states, is never stored)."""
    if len(cover) == 3:
        return "gamma"
    if len(cover) == 4:
        return "beta"
    return "B" if len(set(cover)) == len(cover) else "C"


def kind_of(E, cover):
    """A for a cover of distinct independent states, B for distinct
    dependent states, C for a repeated state. Independence is decided mod
    P2 and numerically and the two must agree."""
    if len(set(cover)) < len(cover):
        return "C"
    r2 = E.rank_mod2(tuple(cover), False)
    A = E.C[:, list(cover)]
    rC = int(np.linalg.matrix_rank(A, tol=1e-8))
    if (r2 == len(cover)) != (rC == len(cover)):
        raise AssertionError(f"modular and numeric independence disagree on {cover}")
    return "A" if r2 == len(cover) else "B"


# ------------------------------------------------------------ hits ------

def term_from_codes(codes):
    c = np.asarray(codes, dtype=np.int64)
    return np.where(c > 0, FOURTH[(c - 1) % 4], 0).astype(complex)


def decide_terms(codes_list, m=M):
    """Exact re-decision of a candidate decomposition given as phase-code
    patterns: whether psi^m = |T>^m lies in the span of the terms mod P2
    (the Q(zeta_24) field) and numerically, the numerical rank, and the
    coefficients."""
    terms = [term_from_codes(c) for c in codes_list]
    res, coeffs = confirm_decomposition(terms, m, ORBIT)
    A = np.column_stack(terms)
    rank = int(np.linalg.matrix_rank(A, tol=1e-8))
    F2 = Field(P2, ORBIT)
    U2 = F2.codes_to_field(np.array([np.asarray(c, dtype=np.int8) for c in codes_list]))
    r0 = _rank_mod(U2, P2)
    r1 = _rank_mod(np.vstack([U2, F2.target(m)]), P2)
    exact = bool(r0 == r1)
    numeric = bool(res < NUM_TOL)
    return {"exact_mod_p2": exact, "numeric": numeric, "agree": exact == numeric,
            "decomposition": exact and numeric, "rank": rank, "terms_count": len(terms),
            "independent": rank == len(terms),
            "nonzero": bool(np.all(np.abs(coeffs) > 1e-9)),
            "residual": float(res), "coeffs": [[float(z.real), float(z.imag)] for z in coeffs]}


def hit_record(h, cover, x0, flats=None):
    """(deterministic part, numeric part) of one matcher hit: the base cover
    and point, the invisible flats of the run (stages beta and gamma), the
    terms as phase codes, and the exact re-decision."""
    codes = [exact_codes(t)[0].tolist() for t in h["terms"]]
    d = decide_terms(codes)
    det = {"cover": [int(u) for u in cover], "x0": int(x0), "terms": codes,
           "free_parameters": int(h["free_parameters"]),
           **{k: d[k] for k in ("exact_mod_p2", "numeric", "agree", "decomposition", "rank",
                                "terms_count", "independent", "nonzero")}}
    if flats is not None:
        det["flats"] = list(flats)
    return det, {"residual": d["residual"], "coeffs": d["coeffs"]}


def genuine(h, rank):
    """A matcher hit that is a decomposition of the stated rank."""
    return h["rank"] == rank and h["exact"] and h["independent"] and h["nonzero"] and h["residual"] < NUM_TOL


def codes_key(terms):
    return sorted(exact_codes(t)[0].tobytes() for t in terms)
