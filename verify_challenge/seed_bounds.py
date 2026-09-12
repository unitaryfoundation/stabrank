"""Seed the ledger from the published state of the art.

Entries with no machine-checkable witness are recorded at the `cited` tier.
That is deliberate: the leaderboard shows them so the picture is complete, but
they cannot set a record, so the only way to top a cell is to submit something
the pipeline can check.
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bounds")

LIT = [
    # orbit, m, direction, rank, author, reference, note
    ("S", 1, "upper", 2, "Labib and Russo", "arXiv:2605.28586", "single copy"),
    ("S", 3, "upper", 4, "Labib and Russo", "arXiv:2605.28586", "exhaustive small-m value; tight"),
    ("S", 3, "lower", 4, "Labib and Russo", "arXiv:2605.28586", "exhaustive"),
    ("S", 4, "upper", 4, "Labib and Russo", "arXiv:2605.28586", "tight"),
    ("S", 4, "lower", 4, "Labib and Russo", "arXiv:2605.28586", "projection monotonicity from m=3"),
    ("S", 6, "upper", 8, "Labib and Russo", "arXiv:2605.28586", "2^3 from the m=2 identity; rank 7 plateaus at sqrt(8/27)"),
    ("N", 2, "upper", 3, "Labib and Russo", "arXiv:2605.28586", ""),
    ("N", 3, "upper", 4, "Labib and Russo", "arXiv:2605.28586", "tight"),
    ("N", 3, "lower", 4, "Labib and Russo", "arXiv:2605.28586", "exhaustive"),
    ("N", 4, "upper", 7, "Labib and Russo", "arXiv:2605.28586", "rank 6 floors at sqrt(211/4043)"),
    ("H3", 2, "upper", 3, "Labib and Russo", "arXiv:2605.28586", ""),
    ("H3", 3, "upper", 4, "Labib and Russo", "arXiv:2605.28586", "tight"),
    ("H3", 3, "lower", 4, "Labib and Russo", "arXiv:2605.28586", "exhaustive"),
    ("H3", 4, "upper", 8, "Labib and Russo", "arXiv:2605.28586", "rank 6 plateaus at sqrt(70-37 sqrt 3)/12"),
    ("T3", 1, "upper", 3, "Kocia and Sarovar", "arXiv:2003.01130", ""),
    ("T3", 2, "upper", 3, "Kocia and Sarovar", "arXiv:2003.01130", "tight; lower bound certified in this repo"),
    ("T3", 3, "upper", 8, "Labib and Russo", "arXiv:2605.28586", "explicit eight-term witness; Kocia-Sarovar recorded this as 8?"),
    ("T3", 4, "upper", 9, "Labib and Russo", "arXiv:2605.28586", "nine-term product witness from the m=2 carry decomposition"),
    ("qubit_H", 2, "upper", 2, "Bravyi, Smith and Smolin", "arXiv:1506.01396", ""),
    ("qubit_H", 3, "upper", 3, "Bravyi, Smith and Smolin", "arXiv:1506.01396", ""),
    ("qubit_H", 4, "upper", 4, "Labib and Russo", "arXiv:2605.28586", ""),
    ("qubit_H", 6, "upper", 6, "Bravyi and Gosset", "arXiv:1601.07601", "underpins the standard 2^{0.47n} figure"),
    ("qubit_H", 6, "lower", 4, "Labib and Russo", "arXiv:2605.28586", "projection monotonicity"),
    ("qubit_T", 4, "upper", 3, "Labib and Russo", "arXiv:2605.28586", "matches the QPG exponent by direct algebra"),
    ("qubit_T", 6, "upper", 6, "Labib and Russo", "arXiv:2605.28586", "rank 5 plateaus at sqrt(5/6) sin(pi/12)"),
]

for orbit, m, direction, rank, author, ref, note in LIT:
    sub = {
        "schema_version": "0.1", "orbit": orbit, "m": m,
        "direction": direction, "rank": rank,
        "provenance": {"author": author, "reference": ref, "method": "literature"},
        "notes": note,
    }
    name = f"{orbit}-m{m}-{direction}-{rank}.json"
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        print(f"  skip {name} (already present)")
        continue
    with open(path, "w") as f:
        json.dump(sub, f, indent=2); f.write("\n")
    print(f"  wrote {name}")
