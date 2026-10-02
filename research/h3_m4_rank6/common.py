"""Shared setup for the rank-6 exclusion of |H3>^4
(docs/notes/h3_m4_rank6_design.md): the cell (n_1 = 2 sliced qutrits, the
single base point (0, 0), the qutrit H3 orbit over Q(omega_3, sqrt 3)),
the 16 affine flats of F_3^2 that miss (0, 0) and the 136 flat multisets,
the slice ratios, the lists the stages draw their bases from (the attested
rank-5 census of |H3>^2 and its degenerate multisets), canonical hashing,
and the exact re-decision of a stored hit against psi_4 = |H3>^4.

Imported as `common` by invisible_p3.py, stages.py, filters6.py,
degenerate6.py, driver.py, batch.py, aggregate.py and probe.py. The
research/qutrit_m4_rank5 directory is on the path for cover_census.py and
matcher.py; its own common.py is loaded by file as `qcommon`, so that the
two modules named common never collide (the arrangement of
research/n4_rank6/common.py, whose API this module keeps so that the N^4
pipeline files port with their stage logic unchanged).

Why (0, 0). |H3> has amplitudes proportional to (1 + sqrt 3, 1, 1), so
along qutrits 1, 2 the slice at x of psi_4 is alpha_x psi_2 with alpha_x =
a_{x_1} a_{x_2}; from (0, 0) the ratio alpha_x / alpha_{(0, 0)} is
1 / (1 + sqrt 3) at the four points with one zero coordinate and
1 / (1 + sqrt 3)^2 at the four points with none, never 1. From the rank-5
run's base point (1, 1) three points have ratio 1, where the trivial
translate always solves the exact slice equation; (0, 0) is to H3 what
(2, 2) is to N (docs/notes/next_exclusion_feasibility_2.md, section 5.2).
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
    # this directory first, so that `common` resolves here and not to the
    # rank-5 pipeline's module of the same name
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
from cover_census import P1, P2, CoverEnumerator3, Field3, rank_mod  # noqa: E402
from matcher import (  # noqa: E402
    COMP,
    E1,
    E2,
    OFFSETS,
    PTS,
    Matcher,
    add,
    exact_codes,
    pidx,
    psi_target,
    term_from_codes,
)


def _load_by_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qcommon = _load_by_file("qutrit_m4_common", os.path.join(QUTRIT, "common.py"))
while HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

N1 = 2                      # sliced qutrits
N2 = 2                      # unsliced qutrits (the base slice is a cover of psi_2 = |H3>^2)
M = 4                       # copies of |H3>
RANK = 6
ORBIT = "H3"
X0 = (0, 0)                 # the single base point of the design
STAGES = ("A6", "B6", "C6", "beta", "gamma")
RESULTS = os.path.join(HERE, "results")
PARTITION = os.path.join(HERE, "partition.json")
CENSUS6 = os.path.join(RESULTS, "census6_H3_full.json")    # the 6-cover census of psi_2 per pivot pair
REPS = os.path.join(HERE, "reps_H3.json")                  # the orbit representatives of every other list
RATES = os.path.join(RESULTS, "rates.json")
WITNESS8 = os.path.join(ROOT, "bounds", "H3-m4-upper-8.json")
CLAIM = "CERTIFIED chi(H3^4) >= 7"
NUM_TOL = 1e-8
N_DICT = 360                # two-qutrit stabilizer states
GROUP_ORDER = 32            # |G_2| for |H3>^2: the 4-element Clifford stabilizer of |H3> on each copy and the swap
POD_FACTOR_COMPILED = 1.3
POD_FACTOR_PYTHON = 4.0

# ------------------------------------------------------------- flats -------
#
# Points of F_3^2 along the sliced qutrits 1, 2 as pairs (x_1, x_2); the
# offsets x - X0 are the matcher's OFFSETS. The 16 affine flats missing X0:
# the 8 other points and the 8 lines not through X0, named p{x_1}{x_2} for
# a point and L{d_1}{d_2}_{points} for a line of direction (d_1, d_2).

DIRS = [(0, 1), (1, 0), (1, 1), (1, 2)]
SQRT3 = np.sqrt(3.0)
ALPHA_C = {0: 1.0 + SQRT3, 1: 1.0, 2: 1.0}       # single-copy amplitudes up to the common scalar


def flats_missing(x0):
    """The 16 affine flats of F_3^2 missing x0 as (name, points) pairs: the
    8 points first, then the 8 lines, each line's points sorted."""
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


def ratio(y, x0=X0):
    """alpha_y / alpha_x0 over C (the slice of psi_4 at y is this multiple
    of psi_2 relative to the slice at x0)."""
    num = ALPHA_C[y[0]] * ALPHA_C[y[1]]
    den = ALPHA_C[x0[0]] * ALPHA_C[x0[1]]
    return num / den


class Cell:
    """The geometry of one base point: the point, the eight other points,
    the flats missing it, the 136 flat multisets, and the offset index of
    every point in the matcher's OFFSETS order."""

    def __init__(self, x0=X0):
        self.x0 = tuple(int(v) for v in x0)
        self.others = [y for y in PTS if y != self.x0]
        self.flats = dict(flats_missing(self.x0))
        self.flat_names = list(self.flats)
        self.flat_pairs = list(itertools.combinations_with_replacement(self.flat_names, 2))
        assert len(self.flat_pairs) == 136
        self.coord_points = (add(self.x0, E1), add(self.x0, E2))

    def offset_index(self, y):
        """Index of the point y in OFFSETS (the eight nonzero offsets from
        x0: e_1, e_2, then COMP)."""
        return OFFSETS.index(((y[0] - self.x0[0]) % 3, (y[1] - self.x0[1]) % 3))

    def offset(self, y):
        return ((y[0] - self.x0[0]) % 3, (y[1] - self.x0[1]) % 3)

    def point(self, offset):
        return add(self.x0, offset)

    def flat_kind(self, name):
        pts = self.flats[name]
        if len(pts) == 1:
            return "point"
        c1, c2 = self.coord_points
        n = sum(1 for c in (c1, c2) if c in pts)
        return f"line, {n} coordinate point{'s' if n != 1 else ''}"


CELL = Cell(X0)
FLATS = CELL.flats
FLAT_NAMES = CELL.flat_names
FLAT_PAIRS = CELL.flat_pairs
RUNS_PER_ITEM = {"A6": 1, "B6": 1, "C6": 1, "beta": len(FLAT_NAMES), "gamma": len(FLAT_PAIRS)}


def offset_index(y):
    return CELL.offset_index(y)


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


def write_hashed(path, body, indent=None):
    """Write `body` with its own sha256 appended, atomically."""
    body = dict(body)
    body.pop("sha256", None)
    body["sha256"] = sha256_json(body)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        if indent is None:
            json.dump(body, f, separators=(",", ":"))
        else:
            json.dump(body, f, indent=indent)
        f.write("\n")
    os.replace(tmp, path)
    return body["sha256"]


def write_record(path, body):
    return write_hashed(path, body, indent=1)


def load_hashed(path, what):
    with open(path) as f:
        doc = json.load(f)
    body = {k: v for k, v in doc.items() if k != "sha256"}
    if sha256_json(body) != doc.get("sha256"):
        raise ValueError(f"{what} {path}: sha256 does not match its content")
    return doc


# ------------------------------------------------------------- codes -------

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


# ----------------------------------------------------------- the cell ------

def make_enumerator(native=True):
    return CoverEnumerator3(ORBIT, N2, native=native)


def new_matcher(E, native=True, seed=29):
    return Matcher(E.D, N2, E.F1, E.F2, native=native, seed=seed)


def target_of(E):
    return psi_target(ORBIT, N2, E.F1, E.F2)


def rank5_lists():
    """The attested rank-5 lists of |H3>^2 (research/qutrit_m4_rank5): the
    full 3-, 4- and 5-covers of distinct independent states (one per G_2
    orbit as far as the pivot and partner reductions go), the dependent
    5-covers and the repeated 5-multisets (the cancel-at-base multisets
    T_3 + (b, b) included), as sorted tuples."""
    cen = qcommon.load_census(ORBIT)
    deg, ddoc = qcommon.load_degenerate(ORBIT)
    out = {"covers3": [tuple(c) for c in cen["covers3"]], "covers4": [tuple(c) for c in cen["covers4"]],
           "covers5": [tuple(c) for c in cen["covers5"]],
           "dep5": [c for c in deg if len(set(c)) == 5], "rep5": [c for c in deg if len(set(c)) < 5],
           "census_sha256": cen["sha256"], "degenerate_sha256": ddoc["sha256"]}
    return out


def item_class(E, item, stage=None):
    """The cost class of an item: for stage B6 "kappa K"; otherwise the
    multiplicity pattern, with " dependent" appended when the distinct
    states are dependent."""
    item = [int(u) for u in item]
    distinct = sorted(set(item))
    rank = np.linalg.matrix_rank(E.C[:, distinct], tol=1e-8)
    if stage == "B6":
        return f"kappa {len(item) - rank}"
    key = str(multiplicity_pattern(item))
    if rank < len(distinct):
        key += " dependent"
    return key


def pairs_of(E):
    """Every (pivot, partner, member count) unit of the cover enumeration,
    in the pivot order of the enumerator (CoverEnumerator3.units)."""
    return [list(u) for u in E.units()]


# ----------------------------------------------------------- loaders -------

def load_partition(path=PARTITION):
    part = load_hashed(path, "partition")
    for key in ("batch_geometry", "census", "reps", "x0", "flats"):
        if key not in part:
            raise ValueError(f"{path} lacks `{key}`; rerun driver.py partition")
    if tuple(part["x0"]) != X0:
        raise ValueError(f"{path}: base point {part['x0']} is not the cell's {X0}")
    if part.get("orbit") != ORBIT:
        raise ValueError(f"{path}: orbit {part.get('orbit')} is not {ORBIT}")
    return part


def load_reps(path=REPS, expect_sha256=None):
    """The orbit-representative lists: B6 (dependent 6-sets with kappa),
    C6 (repeated 6-multisets), k5 (full 5-multisets), k4 (full
    4-multisets), each as an (n, k) int array of sorted dictionary indices
    in the order of the stored codes; with the record."""
    doc = load_hashed(path, "orbit representative lists")
    if expect_sha256 is not None and doc["sha256"] != expect_sha256:
        raise ValueError(f"{path}: sha256 {doc['sha256'][:16]} differs from the partition's {expect_sha256[:16]}")
    if doc.get("orbit") != ORBIT:
        raise ValueError(f"{path}: lists are for {doc.get('orbit')}, not {ORBIT}")
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
    """The 6-cover census of psi_2 (per pivot pair: pivot, partner, members,
    full 6-covers, independent ones, candidates, seconds), checked against
    the file hash the partition records."""
    if expect_sha256 is not None:
        got = sha256_file(path)
        if got != expect_sha256:
            raise ValueError(f"{path}: file sha256 {got[:16]} differs from the partition's {expect_sha256[:16]}")
    with open(path) as f:
        cen = json.load(f)
    if cen.get("orbit") != ORBIT:
        raise ValueError(f"{path}: census is for {cen.get('orbit')}, not {ORBIT}")
    if cen.get("pairs_done") != cen.get("pairs") or len(cen["rows"]) != cen["pairs"]:
        raise ValueError(f"{path}: the census is incomplete ({cen.get('pairs_done')} of {cen.get('pairs')} pairs)")
    if "full6_independent" not in cen:
        cen["full6_independent"] = int(sum(r[4] for r in cen["rows"]))
    return cen


def batch_geometry(part, index):
    geos = part["batch_geometry"]
    if 0 <= index < len(geos) and geos[index]["index"] == index:
        return geos[index]
    for geo in geos:
        if geo["index"] == index:
            return geo
    raise IndexError(f"batch index {index} not in the partition (0..{len(geos) - 1})")


def run_units(stage, geo=None):
    """The (label, unit) runs of one item of a stage: one run for A6, B6,
    C6; one per flat for beta and one per flat multiset for gamma, or the
    subset the batch geometry names (`flats`, `pairs`: the dry-run
    partition of driver.py partition --tiny)."""
    if stage == "beta":
        names = FLAT_NAMES if geo is None or "flats" not in geo else list(geo["flats"])
        return [(f, f) for f in names]
    if stage == "gamma":
        pairs = FLAT_PAIRS if geo is None or "pairs" not in geo else [tuple(p) for p in geo["pairs"]]
        return [(list(p), p) for p in pairs]
    return [(None, None)]


def runs_per_item(stage, geo=None):
    return len(run_units(stage, geo))


def witness_terms():
    """The eight terms of the Lean rank-8 witness of |H3>^4
    (bounds/H3-m4-upper-8.json: the four terms of the rank-4 decomposition
    of |H3>^3 tensored with the rank-2 decomposition of |H3>) as complex
    vectors on four qutrits in the bound file's qutrit order."""
    with open(WITNESS8) as f:
        doc = json.load(f)
    if doc["orbit"] != ORBIT or int(doc["m"]) != M or int(doc["rank"]) != 8:
        raise ValueError(f"{WITNESS8} is not the rank-8 witness of |H3>^4")
    from matcher import constructions_common
    cc = constructions_common()
    return [cc.term_vector(t, 3, M) for t in doc["witness"]["terms"]], doc


# ------------------------------------------------------------ hits ------

def decide_terms(codes_list, m=M):
    """Exact re-decision of a candidate decomposition given as phase-code
    patterns: whether psi_4 = |H3>^4 lies in the span of the terms mod P2
    and numerically, the numerical rank, and the coefficients."""
    return qcommon.decide_terms(ORBIT, codes_list, m=m)


def hit_record(h, item, x0, flats=None):
    """(deterministic part, numeric part) of one matcher hit."""
    codes = [exact_codes(t)[0].astype(int).tolist() for t in h["terms"]]
    d = decide_terms(codes)
    det = {"cover": [int(u) for u in item], "x0": [int(v) for v in x0], "terms": codes,
           "free_parameters": int(h["free_parameters"]),
           **{k: d[k] for k in ("exact_mod_p2", "numeric", "agree", "decomposition", "rank",
                                "terms_count", "independent", "nonzero")}}
    if flats is not None:
        det["flats"] = list(flats)
    return det, {"residual": d["residual"], "coeffs": d["coeffs"]}


def genuine(h, rank=RANK):
    """A matcher hit that is a decomposition of the stated rank."""
    return h["rank"] == rank and h["exact"] and h["independent"] and h["nonzero"] and h["residual"] < NUM_TOL


def codes_key(terms):
    return sorted(exact_codes(t)[0].tobytes() for t in terms)


__all__ = ["CELL", "CENSUS6", "CLAIM", "COMP", "Cell", "E1", "E2", "FLATS", "FLAT_NAMES", "FLAT_PAIRS", "HERE", "M",
           "N1", "N2", "OFFSETS", "ORBIT", "P1", "P2", "PARTITION", "PTS", "POD_FACTOR_COMPILED", "POD_FACTOR_PYTHON",
           "RANK", "RATES", "REPS", "RESULTS", "ROOT", "RUNS_PER_ITEM", "STAGES", "X0", "Field3", "add",
           "batch_geometry", "codes_key", "decide_terms", "decode", "encode", "flats_missing", "genuine",
           "git_commit", "hit_record", "item_class", "load_census6", "load_hashed", "load_partition", "load_reps",
           "lower_priority", "make_enumerator", "multiplicity_pattern", "new_matcher", "offset_index", "pairs_of",
           "pidx", "qcommon", "rank5_lists", "rank_mod", "ratio", "run_units", "runs_per_item", "target_of",
           "term_from_codes", "witness_terms", "write_hashed", "write_record"]
