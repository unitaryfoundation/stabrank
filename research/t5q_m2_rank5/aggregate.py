"""Certificate-side check of the batch outputs of the rank-5 census of
|T5>^2.

    aggregate.py [--partition P] [--results-dir D] [--dry-run] [--partial]
                 [--no-low-census] [--recheck N] [--recheck-seed S] [--recheck-dir D]

Checks, in order: the partition's own hash; that its units are exactly
the (pivot, partner, member count) units of a fresh enumeration over the
two-ququint dictionary in the enumerator's order, that the dictionary's
phase codes hash as the partition says, and that the batch geometry tiles
the units exactly once in order; the k = 1, 2, 3, 4 censuses of psi_2 =
|T5>^2 through the same reductions, all empty (ranks 1 to 4 excluded
exactly, which the rank-5 argument needs; about 100 s; --no-low-census
trusts research/t5q_m2_rank5/results/control_rank4.json instead); every
batch file present; each file's two hashes (deterministic part, whole
record); geometry, hashes and cell equal to the partition's; unit counts
consistent (units run plus undecided units equal the batch's units, one
member count per unit); every `undecided` list empty; every hit re-decided
here from its states (psi_2 against the span of the states modulo
2013265921 in the Q(zeta_5) field, and numerically) with no hit a
decomposition.

Then, unless --dry-run, it re-runs N batches from scratch (indices drawn
from numpy's default_rng(S) over the batches present, S printed on a
`seed:` line) into a scratch directory and compares the deterministic hash
of each with the stored one, bit for bit. When the stored checks pass it
writes batch_manifest.json (one entry per batch with its parameters,
output path and SHA-256, the file `certificate.attested.batches` names),
or batch_manifest.partial.json while batches are missing.

Prints `CERTIFIED chi(T5^2) >= 6` only when every batch is present, every
check above holds, no hit exists and the re-runs match. --dry-run reports
the same checks without the re-runs and never certifies. --partial reports
progress over the batches present instead of failing on the missing ones.
Exit 0 when certified or (dry run) when every stored check passes, 2 when
a stored batch reports a decomposition, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (research/t5q_m2_rank5/common.py)
from common import CLAIM, M, ORBIT, RANK  # noqa: E402

DETERMINISTIC_KEY = "deterministic_sha256"
FULL_KEY = "sha256"


def split_record(rec):
    """(deterministic part, full part) of a batch record, by key order."""
    det, full = {}, {}
    before_det = True
    for k, v in rec.items():
        if k == FULL_KEY:
            break
        full[k] = v
        if k == DETERMINISTIC_KEY:
            before_det = False
        elif before_det:
            det[k] = v
    return det, full


def check_batch(rec, part, geo, problems):
    """Append every discrepancy of one stored batch to `problems`."""
    idx = geo["index"]
    det, full = split_record(rec)
    if common.sha256_json(det) != rec.get(DETERMINISTIC_KEY):
        problems.append(f"batch {idx}: deterministic hash does not match its content")
    if common.sha256_json(full) != rec.get(FULL_KEY):
        problems.append(f"batch {idx}: record hash does not match its content")
    for key in ("native_runs", "candidates"):
        if key in det:
            problems.append(f"batch {idx}: `{key}` sits inside the deterministic part")
    b = rec.get("batch", {})
    want_units = [list(u) for u in part["units"][geo["start"]:geo["end"]]]
    if b.get("index") != idx or b.get("start") != geo["start"] or b.get("end") != geo["end"] \
            or b.get("n_units") != geo["n_units"]:
        problems.append(f"batch {idx}: geometry {b.get('index')}/{b.get('start')}/{b.get('end')} differs from "
                        f"the partition's {idx}/{geo['start']}/{geo['end']}")
    if b.get("units") != want_units:
        problems.append(f"batch {idx}: units differ from the partition's")
    for key, want in (("orbit", ORBIT), ("m", M), ("rank", RANK), ("N", part["N"]),
                      ("group_order", part["group_order"]), ("dictionary_sha256", part["dictionary_sha256"]),
                      ("plan_sha256", part["plan_sha256"]), ("partition_sha256", part["sha256"])):
        if rec.get(key) != want:
            problems.append(f"batch {idx}: {key} = {str(rec.get(key))[:16]}, expected {str(want)[:16]}")
    und = rec.get("undecided", [])
    n_units_und = sum("hit" not in u for u in und)
    if rec.get("units_run", 0) + n_units_und != geo["n_units"]:
        problems.append(f"batch {idx}: units run ({rec.get('units_run')}) plus undecided units ({n_units_und}) "
                        f"do not equal the batch's {geo['n_units']} units")
    members = rec.get("members", [])
    if len(members) != geo["n_units"] or sum(m is not None for m in members) != rec.get("units_run", 0):
        problems.append(f"batch {idx}: one member count per unit run is missing")
    if und:
        problems.append(f"batch {idx}: {len(und)} undecided unit(s) ({und[0].get('reason', '')[:80]})")
    hits = rec.get("hits", [])
    if rec.get("hit_count") != len(hits):
        problems.append(f"batch {idx}: hit_count disagrees with the hits list")
    if rec.get("decompositions") != sum(h.get("decomposition", False) for h in hits):
        problems.append(f"batch {idx}: decompositions disagrees with the hits' decisions")
    unit_set = {(u[0], u[1]) for u in want_units}
    for h_i, h in enumerate(hits):
        u = tuple(h.get("unit", ()))
        if u not in unit_set or not (u[0] in h.get("states", []) and u[1] in h.get("states", [])):
            problems.append(f"batch {idx} hit {h_i}: not a set through one of the batch's pivot pairs")


def redecide_hits(E, records, problems):
    """Every stored hit re-decided from its states. Returns the list of
    (batch index, hit) that are decompositions of psi_2."""
    found = []
    for idx, rec in sorted(records.items()):
        for h_i, h in enumerate(rec.get("hits", [])):
            d = E.decide(h["states"])
            for key in ("exact_mod_p2", "numeric", "decomposition", "rank", "terms_count", "independent",
                        "nonzero"):
                if d[key] != h.get(key):
                    problems.append(f"batch {idx} hit {h_i}: stored {key} = {h.get(key)}, re-decided {d[key]}")
            if not d["agree"]:
                problems.append(f"batch {idx} hit {h_i}: modular and numeric decisions disagree "
                                f"(residual {d['residual']:.2e})")
            if len(set(h["states"])) != len(h["states"]):
                problems.append(f"batch {idx} hit {h_i}: a repeated state")
            if d["decomposition"]:
                found.append((idx, {**h, "residual": d["residual"]}))
    return found


def check_plan(part, problems):
    """A fresh enumeration against the partition: dictionary hash, units,
    plan hash, geometry tiling. Returns the enumerator."""
    t0 = time.time()
    E = common.make_enumerator()
    units = [list(u) for u in E.units()]
    if E.N != part["N"] or E.info["order"] != part["group_order"] or len(E.reps) != part["pivots"]:
        problems.append("dictionary size, symmetry order or pivot count differs from the partition's")
    if E.dictionary_sha256() != part["dictionary_sha256"]:
        problems.append("the dictionary's phase codes do not hash as the partition says")
    if units != part["units"]:
        problems.append("the partition's units are not the enumerator's pivot pairs in its order")
    if common.sha256_json(part["units"]) != part["plan_sha256"] or len(part["units"]) != part["units_count"]:
        problems.append("the partition's plan hash or unit count does not match its units")
    pos = 0
    for k, geo in enumerate(part["batch_geometry"]):
        if geo["index"] != k or geo["start"] != pos or geo["end"] <= geo["start"] \
                or geo["n_units"] != geo["end"] - geo["start"]:
            problems.append(f"batch geometry {k} does not continue the tiling at unit {pos}")
            break
        pos = geo["end"]
    if pos != len(part["units"]) or len(part["batch_geometry"]) != part["batches"]:
        problems.append("the batch geometry does not tile the units exactly once")
    print(f"plan re-enumerated in {time.time() - t0:.0f}s: {E.N} states, group order {E.info['order']}, "
          f"{len(E.reps)} pivots, {len(units)} units, dictionary sha256 {E.dictionary_sha256()[:16]}"
          + (" (equal to the partition's)" if not problems else ""))
    return E


def check_low_census(E, problems):
    t0 = time.time()
    tot = {}
    for k in (1, 2, 3, 4):
        r = E.census_low(k)
        tot[k] = (len(r["hits"]), r["candidates"], r["units"])
        if r["hits"]:
            problems.append(f"the k = {k} census lists {len(r['hits'])} set(s) with psi_2 in their span: "
                            f"{r['hits'][0]['states']}")
    print(f"low-rank censuses in {time.time() - t0:.0f}s: "
          + ", ".join(f"k = {k}: {h} hit(s), {c} candidates over {u} unit(s)" for k, (h, c, u) in tot.items()))
    return tot


def write_manifest(path, part, records, results_dir, complete):
    """The batch manifest behind the attested bound: one entry per stored
    batch with its parameters, output path and file hash (the fields
    verify_challenge/stabrank_verify.load_batch_manifest requires)."""
    entries = []
    for idx in sorted(records):
        geo = common.batch_geometry(part, idx)
        out = os.path.join(results_dir, f"batch_{idx}.json")
        params = {"units": geo["n_units"], "start": geo["start"], "end": geo["end"],
                  "units_run": records[idx]["units_run"], "hits": records[idx]["hit_count"]}
        entries.append({"id": f"batch_{idx}", "params": params,
                        "output": os.path.relpath(out, common.ROOT), "sha256": common.sha256_file(out)})
    doc = {"partition": os.path.relpath(common.PARTITION, common.ROOT), "partition_sha256": part["sha256"],
           "dictionary_sha256": part["dictionary_sha256"], "plan_sha256": part["plan_sha256"],
           "complete": complete, "batches": entries}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(path, common.ROOT)}: {len(entries)} batch(es)" + ("" if complete else " (partial)"))


def rerun(part_path, index, out_dir):
    cmd = ["nice", "-n", "19", sys.executable, os.path.join(common.HERE, "batch.py"), str(index),
           "--partition", part_path, "--out-dir", out_dir, "--force"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=common.ROOT, capture_output=True, text=True)
    return proc, time.time() - t0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--partition", default=common.PARTITION)
    ap.add_argument("--results-dir", default=common.RESULTS)
    ap.add_argument("--dry-run", action="store_true", help="stored checks only, no re-runs")
    ap.add_argument("--partial", action="store_true", help="report over the batches present")
    ap.add_argument("--no-low-census", action="store_true",
                    help="skip the k = 1..4 censuses (about 100 s); the stored control_rank4.json stands in")
    ap.add_argument("--manifest", default=None,
                    help="where to write the batch manifest (default batch_manifest.json, or "
                         "batch_manifest.partial.json when batches are missing)")
    ap.add_argument("--no-manifest", action="store_true", help="do not write a manifest")
    ap.add_argument("--recheck", type=int, default=2, help="batches to re-run from scratch")
    ap.add_argument("--recheck-seed", type=int, default=20260925)
    ap.add_argument("--recheck-dir", default=None)
    a = ap.parse_args(argv[1:])
    common.lower_priority()
    t_all = time.time()
    problems = []
    try:
        part = common.load_partition(a.partition)
    except (OSError, ValueError) as exc:
        print(f"NOT CERTIFIED: {exc}")
        return 1
    print(f"partition {os.path.relpath(a.partition, common.ROOT)}: {ORBIT} m={M} rank={RANK}, {part['units_count']} "
          f"units in {part['batches']} batches, sha256 {part['sha256'][:16]}")
    E = check_plan(part, problems)
    if not a.no_low_census:
        check_low_census(E, problems)
    else:
        print("low-rank censuses skipped (--no-low-census)")
    geometry = part["batch_geometry"]
    records, missing = {}, []
    for geo in geometry:
        idx = geo["index"]
        path = os.path.join(a.results_dir, f"batch_{idx}.json")
        if not os.path.exists(path):
            missing.append(idx)
            continue
        with open(path) as f:
            records[idx] = json.load(f)
        check_batch(records[idx], part, geo, problems)
    present = set(records)
    B = len(geometry)
    tot = {"units": 0, "units_run": 0, "hits": 0, "undecided": 0, "cpu_s": 0.0, "wall_s": 0.0, "kernel_s": 0.0,
           "candidates": 0, "m2": 0.0}
    for idx, r in records.items():
        geo = common.batch_geometry(part, idx)
        tot["units"] += geo["n_units"]
        tot["units_run"] += r.get("units_run", 0)
        tot["hits"] += r.get("hit_count", 0)
        tot["undecided"] += len(r.get("undecided", []))
        tot["candidates"] += r.get("candidates", 0)
        tot["m2"] += float(sum(u[2] ** 2 for u in part["units"][geo["start"]:geo["end"]]))
        for k in ("cpu_s", "wall_s", "kernel_s"):
            tot[k] += r.get(k, 0.0)
    found = redecide_hits(E, records, problems)
    print(f"batches present {len(records)}/{B}" + (f", missing {len(missing)}" if missing else ""))
    if records:
        rate = tot["kernel_s"] / max(tot["m2"], 1.0)
        proj = rate * part["cost_model"]["sum_m2"]
        print(f"{tot['units_run']}/{tot['units']} units run in the batches present, {tot['candidates']} modular "
              f"candidates, {tot['hits']} hits, {tot['undecided']} undecided; {tot['kernel_s'] / 3600:.2f} kernel "
              f"CPU-h ({tot['cpu_s'] / 3600:.2f} process CPU-h, {tot['kernel_s'] / len(records):.0f} kernel s per "
              f"batch, {rate:.3g} s per M^2); projected census {proj / 3600:.1f} CPU-h at this rate (partition "
              f"estimate {part['estimated_pod_s'] / 3600:.1f} pod CPU-h)")
    for p in problems[:50]:
        print(f"PROBLEM: {p}")
    if len(problems) > 50:
        print(f"... {len(problems) - 50} more problems")
    if found:
        idx, h = found[0]
        print(f"DECOMPOSITION FOUND: psi_2 = |T5>^2 lies in the span of the {h['terms_count']} stabilizer states "
              f"{h['states']} (batch {idx}, numerical rank {h['rank']}, residual {h['residual']:.2e}); "
              f"chi(T5^2) <= {h['rank']}; {len(found)} such hit(s)")
        return 2
    if missing and not a.partial:
        print(f"NOT CERTIFIED: {len(missing)} batch(es) missing, first {missing[:10]}")
        return 1
    if problems:
        print(f"NOT CERTIFIED: {len(problems)} problem(s)")
        return 1
    if missing:
        print(f"partial: every stored check passes for the {len(records)} batches present; {len(missing)} to go")
    else:
        print("every stored check passes: all batches present, hashes verify, the units are the enumerator's "
              "pivot pairs tiled exactly once, the dictionary hashes as recorded, "
              + ("ranks 1 to 4 are excluded by the exact censuses, " if not a.no_low_census else "")
              + "no undecided unit, and no 5-set has |T5>^2 in its span")
    if records and not a.no_manifest:
        manifest = a.manifest or os.path.join(common.HERE, "batch_manifest.json" if not missing
                                              else "batch_manifest.partial.json")
        write_manifest(manifest, part, records, a.results_dir, complete=not missing)
    if a.dry_run:
        print("dry run: re-runs skipped; NOT CERTIFIED")
        return 1 if missing else 0
    pool = sorted(present)
    n = min(a.recheck, len(pool))
    rng = np.random.default_rng(a.recheck_seed)
    picks = sorted(int(pool[x]) for x in rng.choice(len(pool), size=n, replace=False)) if n else []
    out_dir = a.recheck_dir or tempfile.mkdtemp(prefix="t5q_m2_rank5_recheck_")
    print(f"seed: {a.recheck_seed}")
    print(f"re-running {n} batch(es) from scratch: {picks} -> {out_dir}")
    mismatches = []
    for idx in picks:
        proc, dt = rerun(a.partition, idx, out_dir)
        path = os.path.join(out_dir, f"batch_{idx}.json")
        if proc.returncode not in (0, 2) or not os.path.exists(path):
            mismatches.append(idx)
            print(f"  batch {idx}: re-run failed (exit {proc.returncode}); stderr tail: "
                  f"{(proc.stderr or '').strip().splitlines()[-1:]}")
            continue
        with open(path) as f:
            new = json.load(f)
        same = new.get(DETERMINISTIC_KEY) == records[idx].get(DETERMINISTIC_KEY)
        print(f"  batch {idx}: {dt:.0f}s, deterministic hash {'matches' if same else 'DIFFERS'} "
              f"({new.get(DETERMINISTIC_KEY, '')[:16]})")
        if not same:
            mismatches.append(idx)
    if mismatches:
        print(f"NOT CERTIFIED: re-run mismatch on batch(es) {mismatches}")
        return 1
    if missing:
        print(f"partial: the {n} re-run(s) match; NOT CERTIFIED ({len(missing)} batches missing); "
              f"{time.time() - t_all:.0f}s")
        return 1
    print(f"aggregate complete in {time.time() - t_all:.0f}s")
    print(CLAIM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
