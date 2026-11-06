"""Clean generation of dictionary, minimum-three gate and all retained pairs.

The enlarged-group point-stabilizer/reversal parametrization adapts the accepted
unitary parametrization directly to the2,304-element unitary/antiunitary group.
No stored scientific input is read.
"""
from itertools import permutations
from math import comb
import json, time
import numpy as np
import cert_cs3_dictionary as d
import cert_cs3_pair_primitives as p


def tick(deadline):
    if time.perf_counter() >= deadline: raise TimeoutError('portable preparation deadline')


def orbit_partition(maps, n):
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for mapping in maps:
        for x, y in enumerate(mapping):
            a, b = find(x), find(int(y))
            if a != b: parent[max(a,b)] = min(a,b)
    parts = {}
    for x in range(n): parts.setdefault(find(x), []).append(x)
    return [{'representative': min(v), 'members': v, 'size': len(v)} for _, v in sorted(parts.items())]


def local_clifford_check():
    # All permutations of F2² are affine; the target correction has even
    # mixed Z4 coefficient and hence is a diagonal Clifford up to global phase.
    for pi in permutations(range(4)):
        assert pi[0]^pi[1]^pi[2]^pi[3] == 0
        a,b = pi[1]^pi[0], pi[2]^pi[0]
        assert a and b and a != b
        ph = [(int(x==3)-int(pi[x]==3))%4 for x in range(4)]
        assert (ph[3]-ph[2]-ph[1]+ph[0])%2 == 0
    return {'all24_affine_permutations_and_even_mixed_phases': True}


def build(folder, deadline):
    tick(deadline); pool, stats = d.dictionary(deadline); target = d.target(); n = len(pool)
    lookup = {row.tobytes(): i for i,row in enumerate(pool)}
    assert len(lookup) == n == 36720
    specs = d.generators(target); maps=[]; phases=[]
    for spec in specs:
        tick(deadline)
        moved = pool[:,spec['permutation']].astype(np.int16)
        raw = np.where(moved>=0, (moved+np.asarray(spec['phase']))%4, -1)
        gamma = raw[np.arange(n),np.argmax(raw>=0,axis=1)]
        canon = np.where(raw>=0,(raw-gamma[:,None])%4,-1).astype(np.int8)
        mapping = np.array([lookup[row.tobytes()] for row in canon],dtype=np.uint16)
        assert np.array_equal(np.sort(mapping),np.arange(n))
        assert np.array_equal(mapping[mapping],np.arange(n))
        assert np.all((gamma+gamma[mapping])%4==0)
        maps.append(mapping); phases.append(gamma.astype(np.int8))
    gm,gp=np.array(maps),np.array(phases)
    unitary_orbits=orbit_partition(gm,n)
    # Complete minimum-three enumeration supplies the analytic slice lemmas.
    controls=d.controls(); records=[]
    for index, orbit in enumerate(unitary_orbits):
        tick(deadline); u=orbit['representative']
        hits, info, done, reason=d.scan(pool[u],target,pool,deadline)
        assert done and reason is None, ('minimum_three_incomplete_or_shorter_cover',reason)
        records.append({'orbit_index':index,'representative':u,'hits':hits,'stats':info})
    solutions, solution_orbits=d.close_hits(records,pool,target,gm,gp,deadline)
    assert len(solutions)==16 and len(solution_orbits)==1
    known_ids=tuple(sorted(lookup[r.tobytes()] for r in d.known_rays()))
    assert known_ids in {tuple(row['ids']) for row in solutions}
    with (folder/'minimum_three.json').open('x',encoding='utf-8') as f:
        json.dump({'controls':controls,'complete_pivot_cases':records,'covers':solutions,'orbits':solution_orbits},f)
    # Build the unitary action, then append its commuting antiunitary coset.
    unitary, inverse, _=p.generate_group({'generators':specs},gm,deadline)
    unitary=unitary.astype(np.uint16)
    raw=np.where(pool>=0,(2*target[None,:]-pool)%4,-1)
    delta=raw[np.arange(n),np.argmax(raw>=0,axis=1)]
    canon=np.where(raw>=0,(raw-delta[:,None])%4,-1).astype(np.int8)
    amap=np.array([lookup[row.tobytes()] for row in canon],dtype=np.uint16)
    assert np.array_equal(amap[amap],np.arange(n)) and np.all((-delta+delta[amap])%4==0)
    for g in range(7):
        assert np.array_equal(amap[gm[g]],gm[g,amap])
        assert np.all((-gp[g].astype(np.int16)+delta[gm[g]]-delta-gp[g,amap])%4==0)
    actions=np.concatenate((unitary,unitary[:,amap]))
    del unitary
    inverse=np.concatenate((inverse,inverse+1152)).astype(np.uint16)
    group=2304
    allmaps=np.concatenate((gm,amap[None,:]))
    allphases=np.concatenate((gp,delta.astype(np.int8)[None,:]))
    orbits=orbit_partition(allmaps,n)
    classes=np.empty(n,dtype=np.int16)
    sizes=np.array([o['size'] for o in orbits],dtype=np.int64)
    for i,o in enumerate(orbits):classes[o['members']]=i
    parts=[]; weightparts=[]; checks=[]
    for i,o in enumerate(orbits):
        tick(deadline); u,size=o['representative'],o['size']
        stabilizer=np.flatnonzero(actions[:,u]==u)
        assert len(stabilizer)*size==group
        labels=np.arange(n,dtype=np.uint16)
        for h in stabilizer:np.minimum(labels,actions[h],out=labels)
        assert np.array_equal(classes[labels],classes)
        counts=np.bincount(labels,minlength=n).astype(np.int64)
        reps=np.flatnonzero(counts)
        assert np.array_equal(labels[reps],reps) and int(counts.sum())==n
        images,first=np.unique(actions[:,u],return_index=True)
        assert images.tolist()==sorted(o['members'])
        transversal=np.full(n,-1,dtype=np.int32);transversal[images]=first
        later=reps[classes[reps]>i];same=reps[(classes[reps]==i)&(reps!=u)]
        reverse=labels[actions[inverse[transversal[same]],u]]
        assert np.array_equal(labels[actions[inverse[transversal[reverse]],u]],same)
        assert np.array_equal(counts[same],counts[reverse])
        keep=same<=reverse;same,reverse=same[keep],reverse[keep]
        weights=size*counts[same];selfreverse=same==reverse
        assert np.all(weights[selfreverse]%2==0);weights[selfreverse]//=2
        vs=np.concatenate((same,later));ws=np.concatenate((weights,size*counts[later]))
        order=np.argsort(vs);vs,ws=vs[order],ws[order]
        assert np.all(vs>u) and np.all((ws>0)&(ws<=group))
        coverage=np.zeros(len(orbits),dtype=np.int64)
        np.add.at(coverage,classes[vs],ws)
        expected=np.zeros(len(orbits),dtype=np.int64)
        expected[i]=comb(size,2);expected[i+1:]=size*sizes[i+1:]
        assert np.array_equal(coverage,expected)
        parts.append(np.column_stack((np.full(len(vs),u),vs)).astype(np.uint16))
        weightparts.append(ws.astype(np.uint16))
        checks.append({'first':u,'classes_covered_exactly':True,'same_class_reversal_exact':True})
    pairs=np.concatenate(parts);weights=np.concatenate(weightparts)
    keys=pairs[:,0].astype(np.int64)*n+pairs[:,1]
    assert np.all(keys[1:]>keys[:-1]) and int(weights.astype(np.int64).sum())==comb(n,2)
    # Independent direct checks of endpoints of the deterministic parametrization.
    for j in sorted({k*(len(pairs)-1)//31 for k in range(32)}):
        x,y=actions[:,pairs[j,0]].astype(np.int64),actions[:,pairs[j,1]].astype(np.int64)
        assert int(np.min(np.minimum(x,y)*n+np.maximum(x,y)))==int(keys[j])
    del actions
    np.savez_compressed(folder/'prepared.npz',rays=pool,target=target,generator_maps=allmaps,
                        generator_phases=allphases,pairs=pairs,pair_weights=weights)
    meta={'dictionary':stats,'local_clifford_check':local_clifford_check(),'unitary_generators':specs,
          'minimum_three_covers':16,'minimum_three_orbits':1,'unitary_pivot_orbits':len(unitary_orbits),
          'enlarged_ray_orbits':len(orbits),'retained_pair_classes':len(pairs),
          'weighted_unordered_pairs':int(weights.astype(np.int64).sum()),'pair_class_checks':checks,
          'direct_enlarged_group_order':2304,'no_stored_scientific_inputs':True}
    with (folder/'preparation.json').open('x',encoding='utf-8') as f:json.dump(meta,f,indent=2)
    return meta
