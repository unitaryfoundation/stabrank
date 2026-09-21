"""One-off: write the stored minimal-decomposition lists as witness terms.

The lists come from verify_challenge/slice_lift.all_decompositions (one
member per unitary-symmetry orbit, as dictionary indices) or from lift_all
(term vectors). Both are converted to the dictionary-independent
(k, x0, W, Q, l) form so the scripts here do not depend on the dictionary
order. Usage: export_data.py SCRATCH_DIR, where SCRATCH_DIR holds the
pickles named below, or export_data.py --small [ORBIT,M,RANK] for the small
cells recomputed in seconds.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ORBIT_P, target  # noqa: E402
from rank_exclusion import dictionary  # noqa: E402
from to_witness import term_from_vector  # noqa: E402

SOURCES = {
    # (orbit, m, rank): (pickle, kind)
    ("N", 3, 4): ("N_m3_rank4_decs.pkl", "indices"),
    ("H3", 3, 4): ("H3_m3_rank4_decs.pkl", "indices"),
    ("S", 3, 4): ("S_m3_rank4_decs.pkl", "indices"),
    ("qubit_H", 4, 4): ("qubit_H_m4_rank4_decs.pkl", "indices"),
    ("qubit_T", 4, 3): ("qubit_T_m4_rank3_decs.pkl", "indices"),
    ("S", 4, 4): ("S_m4_rank4_lifts.pkl", "vectors"),
}


SMALL = [("N", 2, 3), ("H3", 2, 3), ("S", 2, 2), ("T3", 2, 3), ("qubit_H", 3, 3), ("qubit_T", 3, 3),
         ("N", 1, 2), ("H3", 1, 2), ("S", 1, 2), ("T3", 1, 3), ("qubit_H", 1, 2), ("qubit_T", 1, 2),
         ("qubit_H", 2, 2)]


def export_small(only=None):
    """Complete lists at the small cells, recomputed here (seconds each), used
    as controls for the constructors. `only` restricts to one (orbit, m, rank)."""
    from slice_lift import all_decompositions
    os.makedirs(DATA, exist_ok=True)
    for orbit, m, rank in SMALL:
        if only is not None and (orbit, m, rank) != only:
            continue
        p = ORBIT_P[orbit]
        D = dictionary(p, m)
        decs, _ = all_decompositions(orbit, m, rank, D, verbose=False)
        psi = target(orbit, m)
        out = []
        for cols in decs:
            vecs = [D[:, c] for c in cols]
            U = np.column_stack(vecs)
            d, *_ = np.linalg.lstsq(U, psi, rcond=None)
            assert np.linalg.norm(U @ d - psi) < 1e-9
            out.append([term_from_vector(v, p, m) for v in vecs])
        rec = {"orbit": orbit, "m": m, "rank": rank, "count": len(out),
               "source": "verify_challenge/slice_lift.all_decompositions, recomputed by export_data.py --small",
               "decompositions": out}
        path = os.path.join(DATA, f"{orbit}_m{m}_rank{rank}.json")
        with open(path, "w") as f:
            json.dump(rec, f)
            f.write("\n")
        print(f"{path}: {len(out)} decompositions")


def main(scratch, only=None):
    if scratch == "--small":
        if only is not None:
            o, m, r = only.split(",")
            only = (o, int(m), int(r))
        return export_small(only)
    os.makedirs(DATA, exist_ok=True)
    dicts = {}
    for (orbit, m, rank), (name, kind) in SOURCES.items():
        p = ORBIT_P[orbit]
        raw = pickle.load(open(os.path.join(scratch, name), "rb"))
        psi = target(orbit, m)
        decs = []
        for item in raw:
            if kind == "indices":
                if (p, m) not in dicts:
                    dicts[(p, m)] = dictionary(p, m)
                vecs = [dicts[(p, m)][:, i] for i in item]
            else:
                vecs = list(item)
            U = np.column_stack(vecs)
            d, *_ = np.linalg.lstsq(U, psi, rcond=None)
            assert np.linalg.norm(U @ d - psi) < 1e-9, (orbit, m, item)
            decs.append([term_from_vector(v, p, m) for v in vecs])
        rec = {"orbit": orbit, "m": m, "rank": rank, "count": len(decs),
               "source": ("verify_challenge/slice_lift.py "
                          f"({'all_decompositions' if kind == 'indices' else 'lift_all'}), "
                          f"stored as {name} on 2026-09-18"),
               "decompositions": decs}
        path = os.path.join(DATA, f"{orbit}_m{m}_rank{rank}.json")
        with open(path, "w") as f:
            json.dump(rec, f)
            f.write("\n")
        print(f"{path}: {len(decs)} decompositions")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
