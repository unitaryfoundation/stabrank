"""Shared setup for the batch pipeline of the rank-5 exclusion of |H>^6
(docs/notes/h6_rank5_exclusion.md): paths, canonical hashing, the partition
and degenerate-cover loaders with their hash checks, the pivot-pair units of
stage A, and the exact re-decision of a stored hit."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
from slice_cover import (FOURTH, P2, Field, _rank_mod, confirm_decomposition,  # noqa: E402
                         exact_codes)

N1 = 3                      # sliced qubits
M = 6                       # copies of |H>
RANK = 5
ORBIT = "qubit_H"
PARTITION = os.path.join(HERE, "partition.json")
REPAIR = os.path.join(HERE, "partition_stage_c_v2.json")      # the stage C repair partition, when present
DEGENERATE = os.path.join(HERE, "degenerate_covers.json")
RESULTS = os.path.join(HERE, "results")
CENSUS = os.path.join(RESULTS, "kernel_census.json")
DEGENERATE_SAMPLE = os.path.join(RESULTS, "degenerate_sample.json")
STAGES = ("A", "B", "C")
NUM_TOL = 1e-8


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
    if "batch_geometry" not in part:
        raise ValueError(f"{path} is a stage A partition without stages B and C; rerun "
                         "driver.py partition")
    return part


def load_degenerate(path=DEGENERATE, expect_sha256=None):
    """The stored degenerate covers (stages B and C), as a list of sorted
    tuples, with the record; the sha256 must equal the partition's."""
    doc = load_hashed(path, "degenerate cover list")
    if expect_sha256 is not None and doc["sha256"] != expect_sha256:
        raise ValueError(f"{path}: sha256 {doc['sha256'][:16]} differs from the partition's "
                         f"{expect_sha256[:16]}")
    return [tuple(int(x) for x in c) for c in doc["covers"]], doc


def batch_geometry(part, index):
    """The geometry entry with the given batch index (a repair partition's
    indices continue after the original partition's, so the position in
    the list and the index differ)."""
    geos = part["batch_geometry"]
    first = geos[0]["index"] if geos else 0
    if 0 <= index - first < len(geos) and geos[index - first]["index"] == index:
        return geos[index - first]
    for geo in geos:
        if geo["index"] == index:
            return geo
    raise IndexError(f"batch index {index} not in the partition "
                     f"({first}..{geos[-1]['index'] if geos else first - 1})")


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
    """B for a dependent cover of five distinct states, C for a cover with a
    repeated state."""
    return "B" if len(set(cover)) == len(cover) else "C"


# ------------------------------------------------------------ hits ------

def term_from_codes(codes):
    c = np.asarray(codes, dtype=np.int64)
    return np.where(c > 0, FOURTH[(c - 1) % 4], 0).astype(complex)


def decide_terms(codes_list, m=M):
    """Exact re-decision of a candidate decomposition given as phase-code
    patterns: whether psi^m lies in the span of the terms mod P2 and
    numerically, the numerical rank, and the coefficients."""
    terms = [term_from_codes(c) for c in codes_list]
    res, coeffs = confirm_decomposition(terms, m)
    A = np.column_stack(terms)
    rank = int(np.linalg.matrix_rank(A, tol=1e-8))
    F2 = Field(P2)
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


def hit_record(h, cover, x0):
    """(deterministic part, numeric part) of one matcher hit: the base cover
    and point, the terms as phase codes, and the exact re-decision."""
    codes = [exact_codes(t)[0].tolist() for t in h["terms"]]
    d = decide_terms(codes)
    det = {"cover": [int(u) for u in cover], "x0": int(x0), "terms": codes,
           "free_parameters": int(h["free_parameters"]),
           **{k: d[k] for k in ("exact_mod_p2", "numeric", "agree", "decomposition", "rank",
                                "terms_count", "independent", "nonzero")}}
    return det, {"residual": d["residual"], "coeffs": d["coeffs"]}
