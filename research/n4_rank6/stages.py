"""The per-item runs of stages A6, B6 and C6, shared by batch.py and
driver.py (docs/notes/n4_rank6_exclusion.md, section 4).

A6: a full 6-cover of distinct independent states through the compiled
stage A kernel SliceMatch3Kernel directly (the kernel's hits confirmed in
Python, its coordinate-slice counts recorded as matcher.Matcher._run_native
records them); a cover the kernel declines goes through Matcher.run, which
decides it on the reference path.

B6: a dependent 6-set. With kappa = 1 and the compiled dense solve
available, the fresh-term filter of filters6.Filters lists the exact
solutions of the first coordinate slice (the reference's own list, by the
design note's lemma and its planted control); an empty list decides the
base, else the second coordinate slice is filtered the same way, and a base
with solutions at both goes through the unchanged reference Matcher.run,
whose raw coordinate counts must equal the filter's. A base the filter
cannot take (no fresh term with a nonzero dependency modulo both primes,
or kappa outside 1) falls back to the reference, and the fallback is
counted outside the deterministic part. With kappa >= 2 the reference runs
at the raised candidate cap (set_max_cand).

C6: a repeated state through Matcher.run (the block paths, and
BlockOnlyMatcher for a base without an ordinary term).

The deterministic per-run key of every stage is the raw coordinate-slice
solution count `coord_raw` ([n_1] when the first slice has no solution,
else [n_1, n_2] with both counted against the initial family), which the
filter path and the reference path produce alike.
"""
from __future__ import annotations

import functools
import time

import numpy as np

import common  # noqa: E402  (research/n4_rank6/common.py)
from common import X0, pidx
import matcher as mmod  # noqa: E402
from matcher import E1, E2, term_from_codes  # noqa: E402

_BASE_SOLVE = mmod.solve_slice3 if not isinstance(mmod.solve_slice3, functools.partial) else mmod.solve_slice3.func
_CAP = {"value": 2_000_000}


def set_max_cand(cap):
    """The candidate cap of the dense family solve for every Matcher.run in
    this process (matcher.solve_slice3 is looked up at call time)."""
    cap = int(cap)
    _CAP["value"] = cap
    if cap == 2_000_000:
        mmod.solve_slice3 = _BASE_SOLVE
    else:
        mmod.solve_slice3 = functools.partial(_BASE_SOLVE, max_cand=cap)


def max_cand():
    return _CAP["value"]


# ------------------------------------------------------------------ A6 ----

def a6_kernel(M, target):
    """The compiled stage A kernel for the target, or None."""
    if M.native_cls is None:
        return None
    return M._native_kernel(target)


def run_a6(M, kernel, target, cover):
    """(hits, stats) for a cover of distinct states through the kernel
    directly; the reference path when the kernel is missing or declines."""
    if kernel is not None:
        res = kernel.run([int(c) for c in cover], pidx(X0))
        if res["status"] == 0:
            hits = M._dedupe([M.confirm([term_from_codes(c) for c in h], 0, target) for h in np.asarray(res["hits"])])
            raw = [int(v) for v in res["coord_solutions"]]
            if raw[-1] == 0:
                coord_raw, coord_solutions = raw, []
            else:
                n1, n12 = raw
                coord_raw, coord_solutions = [n1, n12 // n1], [n1, n12]
            return hits, {"coord_raw": coord_raw, "coord_solutions": coord_solutions, "refused": False,
                          "native": True, "candidates": int(res["candidates"]), "hits": len(hits), "kappa": 0}
    return M.run(cover, X0, target)


# ------------------------------------------------------------------ B6 ----

def run_b6(M, Fl, target, cover, kappa):
    """(hits, stats, extra) for a dependent 6-set: the filter path for
    kappa = 1 when `Fl` (a filters6.Filters) is given, the reference
    otherwise. `extra` holds the path, the filter's raw feature-zero
    counts and the dense seconds, outside the deterministic part."""
    extra = {"path": "reference", "dense_raw": 0, "dense_raw_s": 0.0, "fallback": None}
    if kappa == 1 and Fl is not None:
        t0 = time.time()
        sols1, st1 = Fl.b6_slice(cover, X0, target, E1)
        if sols1 is None:
            if st1.get("refused"):
                return [], {"coord_raw": [], "coord_solutions": [], "refused": True, "native": False,
                            "kappa": kappa, "hits": 0}, {**extra, "path": "filter"}
            extra["fallback"] = st1.get("outside", "outside the filter")
        else:
            extra["path"] = "filter"
            extra["dense_raw"] += int(st1["dense_raw"])
            extra["dense_raw_s"] += st1["seconds_dense"]
            if not sols1:
                return [], {"coord_raw": [0], "coord_solutions": [], "refused": False, "native": True,
                            "kappa": kappa, "hits": 0, "filter_s": time.time() - t0}, extra
            sols2, st2 = Fl.b6_slice(cover, X0, target, E2)
            if sols2 is None:
                extra["fallback"] = st2.get("outside", "outside the filter at the second slice")
            else:
                extra["dense_raw"] += int(st2["dense_raw"])
                extra["dense_raw_s"] += st2["seconds_dense"]
                if not sols2:
                    return [], {"coord_raw": [len(sols1), 0], "coord_solutions": [], "refused": False,
                                "native": True, "kappa": kappa, "hits": 0, "filter_s": time.time() - t0}, extra
                hits, st = M.run(cover, X0, target)
                extra["path"] = "filter+reference"
                extra["dense_raw"] += int(st.get("dense_raw", 0))
                if not st["refused"] and st["coord_raw"] != [len(sols1), len(sols2)]:
                    raise AssertionError(f"filter and reference disagree on the coordinate-slice counts of {cover}: "
                                         f"{[len(sols1), len(sols2)]} against {st['coord_raw']}")
                st["filter_s"] = time.time() - t0
                return hits, st, extra
    hits, st = M.run(cover, X0, target)
    extra["dense_raw"] += int(st.get("dense_raw", 0))
    return hits, st, extra


# ------------------------------------------------------------------ C6 ----

def run_c6(M, target, cover):
    return M.run(cover, X0, target)


def coord_key(st):
    """The deterministic histogram key of an A6, B6 or C6 run."""
    return ",".join(str(b) for b in st.get("coord_raw", []))


__all__ = ["a6_kernel", "run_a6", "run_b6", "run_c6", "set_max_cand", "max_cand", "coord_key"]
