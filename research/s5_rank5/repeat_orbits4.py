"""Orbits under G_4 = L^4 x| S_4 of the twelve level-4 repeat multisets (a four-term decomposition of S^4 plus one of
its own letters), round 33 agent B: closure by the dense action of the generators on the five terms.
    python repeat_orbits4.py OUTDIR
"""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s5lib as L

def main(outdir):
    t0 = time.time()
    D4 = np.load(os.path.join(outdir, 'decomp4_codes.npy')); info = json.load(open(os.path.join(outdir, 'test_lift.json')))
    starts = np.cumsum([0] + info['orbit_sizes'][:-1]); reps = [D4[s] for s in starts]
    gens = L.generators(4)
    def key(T): return tuple(sorted(c.tobytes() for c in T))
    seen = {}; orbs = []; src = []
    for oi, D in enumerate(reps):
        for i in range(4):
            T = L.canon(np.vstack([D, D[i][None, :]])); k = key(T)
            if k in seen: src[seen[k]].append((oi, i)); continue
            orb = [T]; seen[k] = len(orbs); j = 0
            while j < len(orb):
                cur = L.dense(orb[j]); j += 1
                for name, g in gens:
                    img = L.apply_gen(g, cur); imgc = L.canon(np.array([L.vec_to_code(v) for v in img])); kk = key(imgc)
                    if kk not in seen: seen[kk] = len(orbs); orb.append(imgc)
            orbs.append(orb); src.append([(oi, i)])
    sizes = [len(o) for o in orbs]
    res = dict(n_orbits=len(orbs), sizes=sizes, total=sum(sizes), through_representatives=[[list(x) for x in s] for s in src], seconds=time.time() - t0)
    json.dump(res, open(os.path.join(outdir, 'repeat_orbits4.json'), 'w'), indent=1)
    print(f'REPEAT_ORBITS4 {len(orbs)} orbits, sizes {sizes}, {sum(sizes)} multisets, {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
