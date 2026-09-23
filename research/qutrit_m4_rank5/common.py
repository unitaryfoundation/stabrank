"""Shared setup for the batch pipeline of the rank-5 exclusion of |N>^4 and
|H3>^4 by a two-qutrit all-visible base slice
(docs/notes/qutrit_m4_rank5_exclusion.md): the two cells, paths, canonical
hashing, the partition and degenerate-cover loaders with their hash checks,
the pivot-pair units of stage A, and the exact re-decision of a stored hit."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from cover_census import P2, Field3, rank_mod  # noqa: E402
from matcher import constructions_common, exact_codes, term_from_codes  # noqa: E402

N1 = 2                      # sliced qutrits
N2 = 2                      # base qutrits
M = N1 + N2                 # copies of |M>
RANK = 5
ORBITS = ("N", "H3")
X0 = {"N": (0, 0), "H3": (1, 1)}      # base point (note, section 2)
STAGES = ("A", "B", "C")
NUM_TOL = 1e-8
RESULTS_ROOT = os.path.join(HERE, "results")


def check_orbit(orbit):
    if orbit not in ORBITS:
        raise ValueError(f"orbit must be one of {ORBITS}, got {orbit!r}")
    return orbit


def partition_path(orbit):
    return os.path.join(HERE, f"partition_{check_orbit(orbit)}.json")


def degenerate_path(orbit):
    return os.path.join(HERE, f"degenerate_covers_{check_orbit(orbit)}.json")


def results_dir(orbit):
    return os.path.join(RESULTS_ROOT, check_orbit(orbit))


def census_path(orbit):
    return os.path.join(results_dir(orbit), "kernel_census.json")


def degenerate_sample_path(orbit):
    return os.path.join(results_dir(orbit), "degenerate_sample.json")


def manifest_path(orbit, complete=True):
    return os.path.join(HERE, f"batch_manifest_{check_orbit(orbit)}{'' if complete else '.partial'}.json")


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def load_partition(orbit, path=None):
    part = load_hashed(path or partition_path(orbit), "partition")
    if part.get("orbit") != orbit:
        raise ValueError(f"partition is for {part.get('orbit')}, not {orbit}")
    if "batch_geometry" not in part or "degenerate" not in part:
        raise ValueError("partition lacks stages B and C; rerun driver.py partition after "
                         "driver.py degenerate --write")
    return part


def load_census(orbit, path=None):
    """The stage A census: per pivot pair the covers, candidates and
    seconds, and the hashed full cover lists."""
    return load_hashed(path or census_path(orbit), "census")


def load_degenerate(orbit, path=None, expect_sha256=None):
    """The stored degenerate covers (stages B and C) as sorted tuples, with
    the record; the sha256 must equal the partition's."""
    doc = load_hashed(path or degenerate_path(orbit), "degenerate cover list")
    if doc.get("orbit") != orbit:
        raise ValueError(f"degenerate list is for {doc.get('orbit')}, not {orbit}")
    if expect_sha256 is not None and doc["sha256"] != expect_sha256:
        raise ValueError(f"degenerate list sha256 {doc['sha256'][:16]} differs from the partition's "
                         f"{expect_sha256[:16]}")
    return [tuple(int(x) for x in c) for c in doc["covers"]], doc


def batch_geometry(part, index):
    if not 0 <= index < len(part["batch_geometry"]):
        raise IndexError(f"batch index {index} outside 0..{len(part['batch_geometry']) - 1}")
    geo = part["batch_geometry"][index]
    assert geo["index"] == index
    return geo


def pairs_of(E):
    """Every (pivot, partner, member count) unit of the 5-cover enumeration,
    in the pivot order of the enumerator (CoverEnumerator3.units)."""
    return [list(u) for u in E.units()]


def multiplicity_pattern(cover):
    return tuple(sorted((cover.count(u) for u in set(cover)), reverse=True))


def stage_of(cover):
    """B for a dependent cover of distinct states, C for a cover with a
    repeated state."""
    return "B" if len(set(cover)) == len(cover) else "C"


# ------------------------------------------------------------ hits ------

def target_mod(orbit, m, F):
    a = F.alpha(orbit)
    v = a
    for _ in range(m - 1):
        v = np.kron(v, a) % F.p
    return v


def decide_terms(orbit, codes_list, m=M):
    """Exact re-decision of a candidate decomposition given as phase-code
    patterns: whether |M>^m lies in the span of the terms mod P2 and
    numerically, the numerical rank, and the coefficients."""
    terms = [term_from_codes(c) for c in codes_list]
    A = np.column_stack(terms)
    psi = constructions_common().target(orbit, m)
    coeffs, *_ = np.linalg.lstsq(A, psi, rcond=None)
    res = float(np.linalg.norm(A @ coeffs - psi))
    rank = int(np.linalg.matrix_rank(A, tol=1e-8))
    F2 = Field3(P2)
    U2 = F2.codes_to_field(np.array([np.asarray(c, dtype=np.int64) for c in codes_list]))
    r0 = rank_mod(U2, P2)
    r1 = rank_mod(np.vstack([U2, target_mod(orbit, m, F2)[None, :]]), P2)
    exact = bool(r0 == r1)
    numeric = bool(res < NUM_TOL)
    return {"exact_mod_p2": exact, "numeric": numeric, "agree": exact == numeric,
            "decomposition": exact and numeric, "rank": rank, "terms_count": len(terms),
            "independent": rank == len(terms),
            "nonzero": bool(np.all(np.abs(coeffs) > 1e-9)),
            "residual": res, "coeffs": [[float(z.real), float(z.imag)] for z in coeffs]}


def hit_record(orbit, h, cover, x0):
    """(deterministic part, numeric part) of one matcher hit: the base cover
    and point, the terms as phase codes, and the exact re-decision."""
    codes = [exact_codes(t)[0].astype(int).tolist() for t in h["terms"]]
    d = decide_terms(orbit, codes)
    det = {"cover": [int(u) for u in cover], "x0": [int(v) for v in x0], "terms": codes,
           "free_parameters": int(h["free_parameters"]),
           **{k: d[k] for k in ("exact_mod_p2", "numeric", "agree", "decomposition", "rank",
                                "terms_count", "independent", "nonzero")}}
    return det, {"residual": d["residual"], "coeffs": d["coeffs"]}
