"""Verify the given submissions and record receipts under certs/ for the ones
that pass at a record-holding tier, exactly as site_challenge/build.py does
when it rebuilds the board. Used by the Submissions workflow on a push to
main, so that a bound is verified once, by the run that follows its merge,
and the Site rebuild finds the receipt instead of running the certificate a
second time. Prints one summary line per file and exits 1 if any fails.

    uv run --extra challenge python site_challenge/record_receipts.py bounds/X.json ...
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "site_challenge"))
from stabrank_verify import verify  # noqa: E402
from build import CERTS, content_hash, slug  # noqa: E402


def main(paths):
    bad = 0
    written = []
    for path in paths:
        sub = json.load(open(path))
        s, h = slug(sub), content_hash(sub)
        cpath = os.path.join(CERTS, f"{s}.json")
        if os.path.exists(cpath) and json.load(open(cpath)).get("content_hash") == h:
            print(f"SKIP  {os.path.basename(path)}  receipt already current")
            continue
        r = verify(sub)
        print(f"{'PASS' if r.ok else 'FAIL'}  {os.path.basename(path)}  tier={r.tier}  {r.detail}")
        if not r.ok:
            bad += 1
            continue
        if r.tier in ("verified", "reproduced", "attested"):
            os.makedirs(CERTS, exist_ok=True)
            res = {"ok": r.ok, "tier": r.tier, "detail": r.detail,
                   "gamma": r.gamma, "content_hash": h}
            with open(cpath, "w") as f:
                json.dump(res, f, indent=2)
                f.write("\n")
            written.append(os.path.relpath(cpath, ROOT))
    print("receipts written:", " ".join(written) if written else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
