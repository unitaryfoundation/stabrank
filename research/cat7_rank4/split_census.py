"""Split a census file (census4_run.py output) into one batch per work unit
(pivot orbit, partner stride class) and write the batch manifest:

    python split_census.py CENSUS.json [STRIDE]

A cover's unit is determined by the enumeration order: its pivot is the
member with the smallest orbit root, its partner s2 the next member in the
root-sorted order, and its class is the position of s2 modulo STRIDE.
"""
import os, sys, json, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import census4_run as CR


def main():
    cen = json.load(open(sys.argv[1]))
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    G = CR.setup("c5", True)
    roots = G["roots"]
    order_r = np.argsort(roots, kind="stable")
    pos = np.empty_like(order_r); pos[order_r] = np.arange(len(order_r))
    pivots = [int(p) for p in np.unique(roots)]
    units = {(p, r): [] for p in pivots for r in range(stride)}
    for quad, d in cen["covers"]:
        piv = min(int(roots[i]) for i in quad)
        start = int(np.searchsorted(roots[order_r], roots[piv]))
        others = sorted(int(pos[i]) - start for i in quad if i != piv)
        units[(piv, others[0] % stride)].append([quad, d])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for f in os.listdir(os.path.join(HERE, "results")):
        os.remove(os.path.join(HERE, "results", f))
    batches = []
    for (p, r), covers in units.items():
        body = {"pivot": p, "stride": stride, "offset": r, "covers": sorted(covers), "n": len(covers)}
        name = f"pivot_{p:05d}_r{r:02d}.json"
        txt = json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
        open(os.path.join(HERE, "results", name), "w").write(txt)
        batches.append({"id": name[:-5], "params": {"pivot": p, "stride": stride, "offset": r, "covers": len(covers)},
                        "output": f"research/cat7_rank4/results/{name}", "sha256": hashlib.sha256(txt.encode()).hexdigest()})
    man = {"source": os.path.basename(sys.argv[1]), "pairs": cen["pairs"], "pivots": len(pivots), "stride": stride,
           "covers": len(cen["covers"]), "census_seconds": cen["seconds"], "batches": batches}
    json.dump(man, open(os.path.join(HERE, "batch_manifest.json"), "w"), indent=1)
    print(f"{len(batches)} batches ({len(pivots)} pivots x stride {stride}), {len(cen['covers'])} covers, manifest written")


if __name__ == "__main__":
    main()
