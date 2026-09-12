"""Validate submissions against schema/bound.schema.json.

The schema is linked from every page of the site, so it has to be the real
contract rather than documentation that drifted: additionalProperties is false
throughout, which means a field the schema does not know about is an error, not
a silent pass.  It is checked in CI for exactly that reason.

Schema conformance is necessary and nowhere near sufficient.  It says a
submission is well formed, not that the bound holds; stabrank_verify.py decides
that.  Run this first anyway, since a malformed file wastes a verification run.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import jsonschema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "schema", "bound.schema.json")


def tier_requirements(sub):
    """Cross-field rules the schema cannot express, as (errors, notes).

    A note is not a defect. A submission with no witness and no certificate is
    a perfectly good citation of a published bound; it just cannot hold a
    record, and saying so once here is cheaper than a surprise on the board.
    """
    errors, notes = [], []
    d = sub.get("direction")
    if d == "upper" and "certificate" in sub:
        errors.append("an upper bound is settled by a decomposition, not a "
                      "certificate script; drop 'certificate'")
    if d == "lower" and "witness" in sub:
        errors.append("a lower bound cannot carry a witness; drop 'witness'")
    w = sub.get("witness")
    if w and len(w.get("terms", [])) != len(w.get("coeffs", [])):
        errors.append(f"{len(w.get('terms', []))} terms against "
                      f"{len(w.get('coeffs', []))} coefficients")
    if w and len(w.get("terms", [])) != sub.get("rank"):
        errors.append(f"rank is {sub.get('rank')} but the witness has "
                      f"{len(w.get('terms', []))} terms")
    for c in (w or {}).get("coeffs", []):
        if any(ch in c for ch in ".eE") and "sqrt" not in c and "exp" not in c:
            errors.append(f"coefficient {c!r} looks like a float; "
                          "coefficients must be exact")

    if d == "upper" and not w and "lean" not in sub:
        notes.append("no decomposition and no Lean proof, so this records as "
                     "cited and cannot hold a record")
    if d == "lower" and "certificate" not in sub and "lean" not in sub:
        notes.append("no certificate and no Lean proof, so this records as "
                     "cited and cannot hold a record")
    return errors, notes


def main(argv):
    schema = json.load(open(SCHEMA))
    paths = argv[1:] or sorted(glob.glob(os.path.join(ROOT, "bounds", "*.json")))
    bad = 0
    for path in paths:
        name = os.path.basename(path)
        try:
            sub = json.load(open(path))
        except json.JSONDecodeError as exc:
            print(f"FAIL  {name}  not valid JSON: {exc}")
            bad += 1
            continue
        problems = [f"{'.'.join(str(x) for x in e.path) or '<root>'}: {e.message}"
                    for e in jsonschema.Draft202012Validator(schema).iter_errors(sub)]
        errors, notes = tier_requirements(sub)
        problems += errors
        if problems:
            bad += 1
            print(f"FAIL  {name}")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"OK    {name}")
        for n in notes:
            print(f"      note: {n}")
    print(f"\n{len(paths) - bad} of {len(paths)} valid")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
