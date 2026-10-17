"""Verify the given submissions and record receipts under certs/ for the ones
that pass at a record-holding tier, with the same grouped, parallel
verification site_challenge/build.py uses when it rebuilds the board. Used
by the Submissions workflow on a push to main, so that a bound is verified
once, by the run that follows its merge, and the Site rebuild finds the
receipt instead of running the certificate a second time. Prints one line
per bound as it finishes and exits 1 if any fails.

    uv run --extra challenge python site_challenge/record_receipts.py bounds/X.json ...
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "site_challenge"))
from build import RECEIPT_TIERS, receipt_for, slug, verify_subs  # noqa: E402


def main(paths, jobs=None):
    items = []
    for path in paths:
        sub = json.load(open(path))
        if receipt_for(sub) is not None:
            print(f"SKIP  {os.path.basename(path)}  receipt already current", flush=True)
            continue
        items.append((path, sub))
    results = verify_subs(items, jobs) if items else {}
    bad = sum(1 for r in results.values() if not r["ok"])
    written = [f"certs/{slug(sub)}.json" for path, sub in items
               if results[path]["ok"] and results[path]["tier"] in RECEIPT_TIERS]
    print("receipts written:", " ".join(written) if written else "none", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
