"""Fill the bound draft H3-m4-lower-7.json.draft into a submission file.

    fill_draft.py [--out PATH] [--manifest research/h3_m4_rank6/batch_manifest.json]
                  [--date YYYY-MM-DD] [--hardware TEXT] [--cpu-hours H] [--wall-hours H]

The compute block defaults to what the stored batch records under
results/ report (the sum of their cpu_s, the span from the earliest
`started` to the latest `ended`, the hostname and platform of the first
record); the stage counts in the attested note come from the partition
and the manifest. The manifest must exist (aggregate.py writes it after
every stored check passes) and must be the file the draft's attested
block names. The result is validated against schema/bound.schema.json
when jsonschema is installed and written to --out (default
results/H3-m4-lower-7.json; move it to bounds/ by hand after the
certificate passes).
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
DRAFT = os.path.join(HERE, "H3-m4-lower-7.json.draft")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "H3-m4-lower-7.json"))
    ap.add_argument("--manifest", default=os.path.join("research", "h3_m4_rank6", "batch_manifest.json"))
    ap.add_argument("--partition", default=os.path.join(HERE, "partition.json"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--date", default=None)
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--cpu-hours", type=float, default=None)
    ap.add_argument("--wall-hours", type=float, default=None)
    a = ap.parse_args(argv[1:])
    with open(DRAFT) as f:
        doc = json.load(f)
    if os.path.relpath(os.path.join(ROOT, a.manifest), ROOT) != doc["certificate"]["attested"]["batches"]:
        print("the manifest path differs from the draft's attested.batches", file=sys.stderr)
        return 1
    if not os.path.exists(os.path.join(ROOT, a.manifest)):
        print(f"{a.manifest} does not exist; run aggregate.py first", file=sys.stderr)
        return 1
    with open(os.path.join(ROOT, a.manifest)) as f:
        man = json.load(f)
    if not man.get("complete"):
        print("the manifest is partial", file=sys.stderr)
        return 1
    with open(a.partition) as f:
        part = json.load(f)
    cpu = 0.0
    starts, ends = [], []
    host = plat = None
    by_stage = {}
    for path in sorted(glob.glob(os.path.join(a.results_dir, "batch_*.json"))):
        with open(path) as f:
            rec = json.load(f)
        cpu += float(rec.get("cpu_s", 0.0))
        starts.append(rec.get("started"))
        ends.append(rec.get("ended"))
        st = rec["batch"]["stage"]
        by_stage[st] = by_stage.get(st, 0) + 1
        if host is None:
            host = rec.get("hostname")
            plat = f"{rec.get('machine')}, Python {rec.get('python')}, numpy {rec.get('numpy')}"
    starts = sorted(s for s in starts if s)
    ends = sorted(e for e in ends if e)
    wall = None
    if starts and ends:
        t0 = datetime.datetime.fromisoformat(starts[0])
        t1 = datetime.datetime.fromisoformat(ends[-1])
        wall = (t1 - t0).total_seconds() / 3600
    sb = part["stage_batches"]
    reps = part["reps"]
    fill = {"A6_BATCHES": f"{sb['A6']:,}", "B6_BATCHES": f"{sb['B6']:,}", "C6_BATCHES": f"{sb['C6']:,}",
            "BETA_BATCHES": f"{sb['beta']:,}", "GAMMA_BATCHES": f"{sb['gamma']:,}",
            "COVERS": f"{part['census']['covers']:,}", "B6_ORBITS": f"{reps['B6']:,}", "C6_ORBITS": f"{reps['C6']:,}",
            "K5_ORBITS": f"{reps['k5']:,}", "K4_ORBITS": f"{reps['k4']:,}", "BATCHES": f"{part['batches']:,}"}
    for key in ("note",):
        for k, v in fill.items():
            doc["certificate"]["attested"][key] = doc["certificate"]["attested"][key].replace("{" + k + "}", v)
    comp = doc["provenance"]["compute"]
    cpu_h = a.cpu_hours if a.cpu_hours is not None else cpu / 3600
    comp["cpu_hours"] = round(cpu_h, 1)
    comp["wall_clock_hours"] = round(a.wall_hours if a.wall_hours is not None else (wall or 0.0), 1)
    comp["runs"] = 1
    span = f"{starts[0][:16].replace('T', ' ')} to {ends[-1][:16].replace('T', ' ')} UTC" if starts and ends else "DATES"
    hw = a.hardware or (f"RunPod CPU pod {host} ({plat}), batches in parallel at nice 19, {span} for the "
                        f"{part['batches']:,}-batch loop; the aggregate's two seeded re-runs on the same pod")
    comp["hardware"] = hw
    doc["certificate"]["attested"]["hardware"] = hw
    doc["certificate"]["attested"]["compute_hours"] = round(cpu_h, 1)
    doc["provenance"]["date"] = a.date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    left = [k for k in ("DATE", "HARDWARE") if k in json.dumps(doc)]
    if left:
        print(f"placeholders left: {left}", file=sys.stderr)
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
    print(f"wrote {os.path.relpath(a.out, ROOT)}: date {doc['provenance']['date']}, cpu {comp['cpu_hours']} h, "
          f"wall {comp['wall_clock_hours']} h, batches by stage {by_stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
