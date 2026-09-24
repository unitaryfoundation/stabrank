"""G_2 orbit counts of the stage (beta') and (gamma) base lists of the
rank-6 exclusion of |N>^4: the full 5-multisets of the rank-5 pipeline
(census 5-covers, dependent and repeated 5-multisets) and the full
4-multisets (4-covers, dependent 4-covers, repeated 4-multisets), reduced
by the canonical form of degenerate6.canonical_codes.

    orbits5.py [--out FILE]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "qutrit_m4_rank5"))
sys.path.insert(0, HERE)
from cover_census import CoverEnumerator3  # noqa: E402
from common import qcommon  # noqa: E402  (research/n4_rank6/common.py loads the rank-5 common by file)
from degenerate6 import canonical_codes, group_perms, span_states  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        os.nice(19)
    except OSError:
        pass
    t0 = time.time()
    E = CoverEnumerator3("N", 2)
    G = group_perms(E)
    cen = qcommon.load_census("N")
    covers3 = [tuple(c) for c in cen["covers3"]]
    covers4 = [tuple(c) for c in cen["covers4"]]
    covers5 = [tuple(c) for c in cen["covers5"]]
    deg, _ = qcommon.load_degenerate("N")
    dep5 = [c for c in deg if len(set(c)) == 5]
    rep5 = [c for c in deg if len(set(c)) < 5]
    rec = {"group_order": int(len(G)), "k5": {}, "k4": {}}

    def orbits(lst, k):
        if not lst:
            return 0
        codes = canonical_codes(np.array(lst, dtype=np.int64), G, E.N)
        return int(len(np.unique(codes)))

    for name, lst in (("full 5-covers (census)", covers5), ("dependent 5-covers", dep5), ("repeated 5-multisets", rep5)):
        rec["k5"][name] = {"multisets": len(lst), "orbits": orbits(lst, 5)}
        print(f"{name}: {len(lst)} -> {rec['k5'][name]['orbits']} orbits", flush=True)
    all5 = covers5 + dep5 + rep5
    rec["k5"]["all"] = {"multisets": len(all5), "orbits": orbits(all5, 5)}
    print(f"all stage (beta') bases: {len(all5)} -> {rec['k5']['all']['orbits']} orbits", flush=True)
    dep4 = []
    for T in covers3:
        for x in span_states(E, T).tolist():
            S = tuple(sorted(T + (x,)))
            if E.is_cover(S) and E.is_full(S):
                dep4.append(S)
    dep4 = sorted(set(dep4))
    rep4 = sorted({tuple(sorted(T + (u,))) for T in covers3 for u in T})
    for name, lst in (("full 4-covers (census)", covers4), ("dependent 4-covers", dep4), ("repeated 4-multisets", rep4)):
        rec["k4"][name] = {"multisets": len(lst), "orbits": orbits(lst, 4)}
        print(f"{name}: {len(lst)} -> {rec['k4'][name]['orbits']} orbits", flush=True)
    all4 = covers4 + dep4 + rep4
    rec["k4"]["all"] = {"multisets": len(all4), "orbits": orbits(all4, 4)}
    print(f"all stage (gamma) bases: {len(all4)} -> {rec['k4']['all']['orbits']} orbits", flush=True)
    rec["covers3_orbits"] = orbits(covers3, 3)
    rec["seconds"] = time.time() - t0
    out = a.out or os.path.join(RESULTS, "orbits5_N.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {os.path.relpath(out, ROOT)} [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
