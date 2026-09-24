"""Shared setup for the batch pipeline of the rank-6 exclusion of |N>^4
(docs/notes/n4_rank6_design.md, docs/notes/n4_rank6_exclusion.md): the
cell (n_1 = 2 sliced qutrits, the single base point (2, 2), the qutrit N
orbit over Q(omega_3)), the 16 affine flats of F_3^2 that miss (2, 2) and
the 136 flat multisets, the lists every stage draws its bases from (the
kernel census of the full 6-covers of |N>^2 per pivot pair, the G_2 orbit
representatives of the dependent and repeated 6-multisets, of the full
5-multisets and of the full 4-multisets), canonical hashing, the partition
loaders with their hash checks, and the exact re-decision of a stored hit
against psi_4 = |N>^4.

Imported as `common` by driver.py, batch.py, aggregate.py, invisible3.py
and the prototypes of this directory. The research/qutrit_m4_rank5
directory is on the path for cover_census.py and matcher.py; its own
common.py is loaded by file as `qcommon`, so that the two modules named
common never collide.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
QUTRIT = os.path.join(ROOT, "research", "qutrit_m4_rank5")
for _p in (QUTRIT, os.path.join(ROOT, "verify_challenge"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from cover_census import P1, P2, CoverEnumerator3, Field3, rank_mod  # noqa: E402
from matcher import (COMP, E1, E2, OFFSETS, PTS, Matcher, add, exact_codes, pidx, psi_target,  # noqa: E402
                     term_from_codes)


def _load_by_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qcommon = _load_by_file("qutrit_m4_common", os.path.join(QUTRIT, "common.py"))

N1 = 2                      # sliced qutrits
N2 = 2                      # unsliced qutrits (the base slice is a cover of psi_2 = |N>^2)
M = 4                       # copies of |N>
RANK = 6
ORBIT = "N"
X0 = (2, 2)                 # the single base point of the design
STAGES = ("A6", "B6", "C6", "beta", "gamma")
PARTITION = os.path.join(HERE, "partition.json")
RESULTS = os.path.join(HERE, "results")
CENSUS6 = os.path.join(RESULTS, "census6_N_full.json")      # the 6-cover census of psi_2 per pivot pair
REPS = os.path.join(HERE, "reps_N.json")                   # the orbit representatives of every other list
RATES = os.path.join(RESULTS, "rates.json")
WITNESS7 = os.path.join(QUTRIT, "N_m4_rank7_witness.json")
NUM_TOL = 1e-8
CLAIM = "CERTIFIED chi(N^4) >= 7"
POD_FACTOR_COMPILED = 1.3
POD_FACTOR_PYTHON = 4.0

# ------------------------------------------------------------- flats -------
#
# Points of F_3^2 along the sliced qutrits 1, 2 as pairs (x_1, x_2); the
# offsets x - X0 are the matcher's OFFSETS. The 16 affine flats missing
# X0 = (2, 2): the 8 other points and the 8 lines not through X0, named as
# in research/n4_rank6/results/invisible3_geometry.json.

DIRS = [(0, 1), (1, 0), (1, 1), (1, 2)]
COORD_POINTS = (add(X0, E1), add(X0, E2))          # (0, 2) and (2, 0)


def _flats_missing(x0=X0):
    out = []
    for y in PTS:
        if y != x0:
            out.append((f"p{y[0]}{y[1]}", (y,)))
    seen = set()
    for v in DIRS:
        for b in PTS:
            pts = tuple(sorted({add(b, (t * v[0] % 3, t * v[1] % 3)) for t in range(3)}))
            if pts in seen or x0 in pts:
                continue
            seen.add(pts)
            out.append((f"L{v[0]}{v[1]}_" + "".join(f"{p[0]}{p[1]}" for p in pts), pts))
    assert len(out) == 16 and sum(1 for _, pts in out if len(pts) == 3) == 8
    return out


FLATS = dict(_flats_missing())                     # name -> points (a tuple of (x_1, x_2) pairs)
FLAT_NAMES = list(FLATS)                           # the order used everywhere
FLAT_PAIRS = list(itertools.combinations_with_replacement(FLAT_NAMES, 2))   # 136 multisets
assert len(FLAT_PAIRS) == 136
RUNS_PER_ITEM = {"A6": 1, "B6": 1, "C6": 1, "beta": len(FLAT_NAMES), "gamma": len(FLAT_PAIRS)}


def offset_index(y):
    """Index of the point y in the matcher's OFFSETS order (the eight
    nonzero offsets from X0: e_1, e_2, then COMP)."""
    return OFFSETS.index(((y[0] - X0[0]) % 3, (y[1] - X0[1]) % 3))


# ----------------------------------------------------------- hashing -------

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
        json.dump(body, f, separators=(",", ":"))
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


# ------------------------------------------------------------- codes -------

N_DICT = 360                                        # two-qutrit stabilizer states


def encode(sets, N=N_DICT):
    """Sorted rows (n, k) to one int per row, base N (k <= 6 fits int64)."""
    sets = np.asarray(sets, dtype=np.int64)
    code = np.zeros(len(sets), dtype=np.int64)
    for c in range(sets.shape[1]):
        code = code * N + sets[:, c]
    return code


def decode(codes, k, N=N_DICT):
    codes = np.asarray(codes, dtype=np.int64)
    out = np.zeros((len(codes), k), dtype=np.int64)
    c = codes.copy()
    for j in range(k - 1, -1, -1):
        out[:, j] = c % N
        c //= N
    return out


def multiplicity_pattern(cover):
    return tuple(sorted((list(cover).count(u) for u in set(cover)), reverse=True))


# ----------------------------------------------------------- loaders -------

def load_partition(path=PARTITION):
    part = load_hashed(path, "partition")
    for key in ("batch_geometry", "census", "reps", "x0", "flats"):
        if key not in part:
            raise ValueError(f"{path} lacks `{key}`; rerun driver.py partition")
    if tuple(part["x0"]) != X0:
        raise ValueError(f"{path}: base point {part['x0']} is not the cell's {X0}")
    return part


def load_reps(path=REPS, expect_sha256=None):
    """The orbit-representative lists: B6 (dependent 6-sets with kappa),
    C6 (repeated 6-multisets), k5 (full 5-multisets), k4 (full
    4-multisets), each as an (n, k) int array of sorted dictionary indices
    in the order of the stored codes; with the record."""
    doc = load_hashed(path, "orbit representative lists")
    if expect_sha256 is not None and doc["sha256"] != expect_sha256:
        raise ValueError(f"{path}: sha256 {doc['sha256'][:16]} differs from the partition's {expect_sha256[:16]}")
    lists = {}
    for key, k in (("B6", 6), ("C6", 6), ("k5", 5), ("k4", 4)):
        lists[key] = decode(doc[key]["codes"], k)
        if len(lists[key]) != doc[key]["count"]:
            raise ValueError(f"{path}: {key} holds {len(lists[key])} codes, the record says {doc[key]['count']}")
    lists["B6_kappa"] = np.asarray(doc["B6"]["kappa"], dtype=np.int64)
    if len(lists["B6_kappa"]) != len(lists["B6"]):
        raise ValueError(f"{path}: B6 kappa list and codes differ in length")
    return lists, doc


def load_census6(path=CENSUS6, expect_sha256=None):
    """The 6-cover census of psi_2 (per pivot pair: members, full 6-covers,
    independent ones, candidates, seconds), checked against the file hash
    the partition records."""
    if expect_sha256 is not None:
        got = sha256_file(path)
        if got != expect_sha256:
            raise ValueError(f"{path}: file sha256 {got[:16]} differs from the partition's {expect_sha256[:16]}")
    with open(path) as f:
        cen = json.load(f)
    if cen.get("pairs_done") != cen.get("pairs") or len(cen["rows"]) != cen["pairs"]:
        raise ValueError(f"{path}: the census is incomplete ({cen.get('pairs_done')} of {cen.get('pairs')} pairs)")
    return cen


def batch_geometry(part, index):
    geos = part["batch_geometry"]
    if 0 <= index < len(geos) and geos[index]["index"] == index:
        return geos[index]
    for geo in geos:
        if geo["index"] == index:
            return geo
    raise IndexError(f"batch index {index} not in the partition (0..{len(geos) - 1})")


def make_enumerator(native=True):
    return CoverEnumerator3(ORBIT, N2, native=native)


def pairs_of(E):
    """Every (pivot, partner, member count) unit of the cover enumeration,
    in the pivot order of the enumerator (CoverEnumerator3.units)."""
    return [list(u) for u in E.units()]


def new_matcher(E, native=True, seed=29):
    return Matcher(E.D, N2, E.F1, E.F2, native=native, seed=seed)


def target_of(E):
    return psi_target(ORBIT, N2, E.F1, E.F2)


def item_class(E, item, stage):
    """The cost class of an item of stage B6, C6, beta or gamma: for B6
    "kappa K"; for the others the multiplicity pattern, with " dependent"
    appended when the distinct states are dependent."""
    item = [int(u) for u in item]
    distinct = sorted(set(item))
    dep = np.linalg.matrix_rank(E.C[:, distinct], tol=1e-8) < len(distinct)
    if stage == "B6":
        return f"kappa {len(item) - np.linalg.matrix_rank(E.C[:, distinct], tol=1e-8)}"
    key = str(multiplicity_pattern(item))
    if dep:
        key += " dependent"
    return key


# ------------------------------------------------------------ hits ------

def decide_terms(codes_list, m=M):
    """Exact re-decision of a candidate decomposition given as phase-code
    patterns: whether psi_4 = |N>^4 lies in the span of the terms mod P2
    and numerically, the numerical rank, and the coefficients."""
    return qcommon.decide_terms(ORBIT, codes_list, m=m)


def hit_record(h, item, x0, flats=None):
    """(deterministic part, numeric part) of one matcher hit: the base
    multiset and point, the invisible flats of the run (stages beta and
    gamma), the terms as phase codes, and the exact re-decision."""
    codes = [exact_codes(t)[0].astype(int).tolist() for t in h["terms"]]
    d = decide_terms(codes)
    det = {"cover": [int(u) for u in item], "x0": [int(v) for v in x0], "terms": codes,
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


def witness_terms():
    """The seven terms of the Lean rank-7 witness of |N>^4
    (research/qutrit_m4_rank5/N_m4_rank7_witness.json, rebuilt from
    lean_proofs/LeanProofs/NorrellM4Pointwise.lean by the rank-5 driver) as
    complex vectors on four qutrits, the sliced qutrits first."""
    with open(WITNESS7) as f:
        doc = json.load(f)
    if doc["orbit"] != ORBIT or doc["m"] != M or doc["rank"] != 7:
        raise ValueError(f"{WITNESS7} is not the rank-7 witness of |N>^4")
    return [term_from_codes(c) for c in doc["terms"]], doc
