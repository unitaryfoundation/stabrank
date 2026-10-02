"""Fill the bound draft T5-m3-lower-6.json.draft into a submission file.

    fill_draft.py [--out PATH] [--manifest research/t5q_m2_rank5/batch_manifest.json]
                  [--date YYYY-MM-DD] [--hardware TEXT] [--cpu-hours H] [--wall-hours H] [--runs N]

The date defaults to today (UTC); the compute block defaults to what the
stored lift-stage records under results/ report (the cell record's
seconds and every control's seconds, summed, one run each) with the
hostname of the cell record as the hardware. The manifest must exist (the
listing aggregate writes it) and must be the file the draft's attested
block names. The result is validated against schema/bound.schema.json
when jsonschema is installed, and written to --out (default
results/T5-m3-lower-6.json; move it to bounds/ by hand).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DRAFT = os.path.join(HERE, "T5-m3-lower-6.json.draft")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "T5-m3-lower-6.json"))
    ap.add_argument("--manifest", default=os.path.join("research", "t5q_m2_rank5", "batch_manifest.json"))
    ap.add_argument("--date", default=None)
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--cpu-hours", type=float, default=None)
    ap.add_argument("--wall-hours", type=float, default=None)
    ap.add_argument("--runs", type=int, default=None)
    a = ap.parse_args(argv[1:])
    with open(DRAFT) as f:
        doc = json.load(f)
    if os.path.relpath(os.path.join(ROOT, a.manifest), ROOT) != doc["certificate"]["attested"]["batches"]:
        print("the manifest path differs from the draft's attested.batches", file=sys.stderr)
        return 1
    if not os.path.exists(os.path.join(ROOT, a.manifest)):
        print(f"{a.manifest} does not exist; run aggregate.py --list first", file=sys.stderr)
        return 1
    records = sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
    seconds, runs, host = 0.0, 0, None
    for path in records:
        with open(path) as f:
            rec = json.load(f)
        runs += 1
        seconds += float(rec.get("seconds", rec.get("stats", {}).get("seconds", 0.0)) or 0.0)
        for plant in rec.get("plants", []):
            seconds += float(plant.get("stats", {}).get("seconds", 0.0))
        if os.path.basename(path) == "cell.json":
            host = rec.get("hostname")
    comp = doc["provenance"]["compute"]
    comp["cpu_hours"] = round(a.cpu_hours if a.cpu_hours is not None else seconds / 3600, 3)
    comp["wall_clock_hours"] = round(a.wall_hours if a.wall_hours is not None else seconds / 3600, 3)
    comp["runs"] = a.runs if a.runs is not None else runs
    comp["hardware"] = a.hardware or (f"{host}, one process at nice 19; the lift stage and its controls "
                                      "(the census behind it is the attested block)")
    doc["provenance"]["date"] = a.date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    try:
        import jsonschema
        with open(os.path.join(ROOT, "schema", "bound.schema.json")) as f:
            schema = json.load(f)
        jsonschema.validate(doc, schema)
        print("schema: valid")
    except ImportError:
        print("schema: jsonschema not installed, not validated")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(a.out, ROOT)}: date {doc['provenance']['date']}, compute {comp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
