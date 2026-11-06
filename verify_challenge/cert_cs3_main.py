"""Proposed clean-folder CS-cube certificate driver; not yet validated.

Regenerates all data and tests every cover orbit, without family/fidelity filters.
Prints the certificate claim only after every complete gate succeeds. See the
included analytic slice lemma and README; preparing this source is not a run.
"""
import sys
if sys.flags.optimize:
    raise RuntimeError('This proof program requires assertions; python -O/-OO is forbidden.')
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from fractions import Fraction
from math import lcm
import argparse, ast, gzip, hashlib, json, multiprocessing, os, time
import numpy as np
import cert_cs3_dictionary as d
import cert_cs3_projected as projected
import cert_cs3_cover_orbits as orbit
import cert_cs3_fibers as fiber
from cert_cs3_prepare import build
from cert_cs3_parameters import PARAMETERS

HERE=Path(__file__).resolve().parent
ENGINE=POOL=PAIRS=None
DEADLINE=0.0


def tick():
    if time.perf_counter()>=DEADLINE:raise TimeoutError('portable total runtime budget')


def save(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def layout():
    from cert_cs3_manifest import MANIFEST
    manifest=MANIFEST
    assert manifest['parameters']==PARAMETERS
    for name,digest in manifest['sha256'].items():
        path=HERE/name
        assert path.resolve().is_relative_to(HERE.resolve()) and path.is_file()
        assert sha(path)==digest,name
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
    return manifest


def preflight(work):
    """New metadata and small controls only; no complete four-cover/lift scan."""
    prep=build(work,DEADLINE)
    with np.load(work/'prepared.npz',allow_pickle=False) as z:pool,target=z['rays'],z['target']
    projected.DEADLINE=DEADLINE
    inv=np.zeros(projected.P,dtype=np.int64);inv[1]=1
    for j in range(2,len(inv)):inv[j]=(-(projected.P//j)*int(inv[projected.P%j]))%projected.P
    controls=projected.controls(pool,target,np.asarray(PARAMETERS['projection']),inv)
    fiber.tick=tick;fiber.DEADLINE=DEADLINE;fiber.SCALE=2
    fixtures,fixture_target=fiber.bases()
    fibers=fiber.preflight(fixtures,fixture_target)
    tick()
    return {'status':'preflight_complete_not_a_rank_certificate','preparation':prep,
            'projected_controls':controls,'fiber_controls':fibers,
            'complete_four_cover_census_run':False,'all_cover_lifts_run':False}


def worker_init(path,deadline):
    global ENGINE,POOL,PAIRS,DEADLINE
    DEADLINE=deadline;projected.DEADLINE=deadline
    with np.load(path,allow_pickle=False) as z:POOL,target,PAIRS=z['rays'],z['target'],z['pairs']
    inv=np.zeros(projected.P,dtype=np.int64);inv[1]=1
    for j in range(2,len(inv)):inv[j]=(-(projected.P//j)*int(inv[projected.P%j]))%projected.P
    assert np.all(np.arange(1,len(inv),dtype=np.int64)*inv[1:]%projected.P==1)
    ENGINE=projected.Engine(POOL,target,np.asarray(PARAMETERS['projection']),inv)


def scan_batch(indices):
    rows=[];hitcount=hitbytes=0
    for index in indices:
        tick();u,v=map(int,PAIRS[index]);identities=[]
        def emit(record):
            nonlocal hitcount,hitbytes
            item={'retained_ids':[u,v],**record}
            hitcount+=1;hitbytes+=len(json.dumps(item,separators=(',',':')).encode())
            assert hitcount<=PARAMETERS['batch_identity_cap'],'batch identity count cap'
            assert hitbytes<=PARAMETERS['batch_identity_byte_cap'],'batch identity byte cap'
            identities.append(item)
        stats=ENGINE.scan(POOL[u],POOL[v],emit)
        assert stats['complete'] and stats['base_rank']==3 and stats['all_pairs_accounted']
        assert sum(stats['all_free_pair_accounting'].values())==674160840
        rows.append({'index':int(index),'retained_ids':[u,v],'stats':stats,'identities':identities})
    return rows


def complete_four_census(work,workers):
    with np.load(work/'prepared.npz',allow_pickle=False) as z:pool,target,pairs=z['rays'],z['target'],z['pairs']
    projected.DEADLINE=DEADLINE
    inv=np.zeros(projected.P,dtype=np.int64);inv[1]=1
    for j in range(2,len(inv)):inv[j]=(-(projected.P//j)*int(inv[projected.P%j]))%projected.P
    save(work/'projected_controls.json',projected.controls(pool,target,np.asarray(PARAMETERS['projection']),inv))
    covers={};seen=set();counts=Counter();submitted=0;last_print=0.0
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    with gzip.open(work/'four_census.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as log:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_init,initargs=(str(work/'prepared.npz'),DEADLINE)) as executor:
            active={}
            try:
                while submitted<len(pairs) or active:
                    tick()
                    while submitted<len(pairs) and len(active)<PARAMETERS['inflight_batches']:
                        ids=list(range(submitted,min(submitted+PARAMETERS['case_batch'],len(pairs))))
                        active[executor.submit(scan_batch,ids)]=ids;submitted+=len(ids)
                    done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                    for future in done:
                        expected=active.pop(future);rows=future.result()
                        assert [r['index'] for r in rows]==expected
                        for row in rows:
                            i=row['index'];assert i not in seen and row['retained_ids']==pairs[i].astype(int).tolist()
                            seen.add(i);stats=row['stats']
                            kinds=Counter(r['kind'] for r in row['identities'])
                            assert kinds['independent_four_cover']==stats['accepted_independent_four_covers']
                            assert kinds['rejected_zero_retained_coefficient']==stats['rejected_zero_retained']
                            assert kinds['zero_residual_three_cover']==stats['zero_residual_three_covers']
                            assert len(row['identities'])==stats['candidate_pairs']+stats['zero_residual_three_covers']
                            counts.update(kinds)
                            for hit in row['identities']:
                                if hit['kind']!='independent_four_cover':continue
                                ids=hit['retained_ids']+hit['free_ids'];cf=tuple(map(orbit.parse,hit['coefficients']))
                                order=sorted(range(4),key=lambda j:ids[j]);key=tuple(ids[j] for j in order)
                                coeff=tuple(cf[j] for j in order)
                                assert len(set(key))==4 and all(c!=orbit.ZERO for c in coeff)
                                if key in covers:assert covers[key]==coeff
                                else:covers[key]=coeff
                                assert len(covers)<=PARAMETERS['unique_cover_cap'],'unique cover cap'
                            log.write(json.dumps(row,separators=(',',':'))+'\n')
                    if time.perf_counter()-last_print>=5:
                        print(json.dumps({'stage':'four_cover_census','complete_cases':len(seen),'total':len(pairs),
                                          'unique_input_covers':len(covers)}),flush=True);last_print=time.perf_counter()
            finally:
                for future in active:future.cancel()
    assert seen==set(range(len(pairs)))
    save(work/'four_census_summary.json',{'all_complete':True,'retained_pair_classes':len(pairs),
          'identity_counts':dict(counts),'unique_input_covers':len(covers),'no_family_or_fidelity_filters':True})
    save(work/'unique_covers.json',[{'ids':ids,'coefficients':[orbit.serial(c) for c in cf]} for ids,cf in sorted(covers.items())])
    return covers


def all_cover_orbits(work,covers):
    with np.load(work/'prepared.npz',allow_pickle=False) as z:
        pool,target,maps,phases=z['rays'],z['target'],z['generator_maps'],z['generator_phases']
    geometry=orbit.ExactGeometry(pool,target,tick)
    specs=json.loads((work/'preparation.json').read_text(encoding='utf-8'))['unitary_generators']
    physical={'unitary':specs,'phase_row_to_id':{tuple(map(int,row)):i for i,row in enumerate(pool)}}
    unresolved=set(covers);reps=[];coverage={}
    while unresolved:
        tick();ids=min(unresolved);cf=covers[ids];geometry.verify(ids,cf)
        canonical,values,words=orbit.full_orbit(ids,cf,maps,phases,geometry,tick)
        members=set(values)&set(covers)
        assert members and members<=unresolved
        for key in members:
            assert values[key]==covers[key]
            orbit.replay_physical(key,covers[key],words[key],canonical,values[canonical],pool,target,physical,tick)
            coverage[str(key)]={'canonical':canonical,'word':words[key]}
        unresolved-=members
        reps.append({'ids':canonical,'coefficients':[orbit.serial(c) for c in values[canonical]],
                     'full_orbit_size':len(values),'input_members':len(members)})
        if len(reps)%16==0:print(json.dumps({'stage':'all_cover_orbits','complete_orbits':len(reps),'unassigned':len(unresolved)}),flush=True)
    reps.sort(key=lambda r:r['ids'])
    assert len({tuple(r['ids']) for r in reps})==len(reps) and sum(r['input_members'] for r in reps)==len(covers)
    save(work/'all_cover_orbits.json',reps);save(work/'cover_transports.json',coverage)
    return reps


def all_lifts(work,reps):
    with np.load(work/'prepared.npz',allow_pickle=False) as z:pool,target=z['rays'],z['target']
    fiber.tick=tick;fiber.DEADLINE=DEADLINE;fiber.SECONDS=PARAMETERS['seconds']
    fiber.TRANSITION_CAP=PARAMETERS['transition_cap'];fiber.ORDERED_PAIR_CAP=PARAMETERS['ordered_pair_cap']
    fiber.SCALE=2;fixture,fixture_target=fiber.bases();save(work/'fiber_controls.json',fiber.preflight(fixture,fixture_target))
    folder=work/'lifts';folder.mkdir();checked=[];controls={}
    for index,rep in enumerate(reps):
        tick();rows=pool[list(rep['ids'])];cf=tuple(map(orbit.parse,rep['coefficients']))
        scale=lcm(*(x.denominator for c in cf for x in c))
        scaled=[(int(c[0]*scale),int(c[1]*scale)) for c in cf]
        assert scale+sum(abs(a)+abs(b) for a,b in scaled)<2**50
        coefficients=np.array(scaled,dtype=np.int64);fiber.SCALE=scale
        assert fiber.rank_phase(rows)==4
        assert all(fiber.stabilizer_check(fiber.decoded(row)) for row in rows)
        assert np.array_equal(sum(fiber.weighted(row,c) for row,c in zip(rows,coefficients)),fiber.target_scaled(target))
        case=folder/('orbit_'+str(index))
        base=fiber.prepare_base('portable_orbit_'+str(index),rows,coefficients,target,case)
        assert base['coverage']['complete'],'INCOMPLETE: transition cap reached'
        if scale not in controls:
            positive=fiber.joint_test(base,target,0,'plus_control',case)
            assert positive['status']=='exact_lift_witness'
            assert positive['witness']['all64_amplitudes_exact'] and positive['witness']['all_four_global_stabilizers_checked']
            controls[scale]=index
        result=fiber.joint_test(base,target,1,'CS_target',case)
        assert result['status']=='complete_no_lift_for_fixed_base',('NOT_CERTIFIED',index,result['status'])
        assert result['complete_negative'] and result['transition_list_complete']
        assert result['ordered_pairs_examined']==result['ordered_pairs_required']==len(base['transitions'])**2
        assert result['sign_vectors_evaluated']==16*result['ordered_pairs_examined']
        checked.append({'orbit_index':index,'ids':rep['ids'],'transitions':len(base['transitions']),
                        'pairs':result['ordered_pairs_examined'],'signs':result['sign_vectors_evaluated']})
        if (index+1)%16==0:print(json.dumps({'stage':'unfiltered_lifts','complete':index+1,'total':len(reps)}),flush=True)
    assert len(checked)==len(reps)
    save(work/'all_lifts.json',{'all_complete_negative':True,'cases':checked,'denominator_controls':controls})
    return checked


def main():
    global DEADLINE
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-layout',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--work-dir',type=Path,default=HERE/'portable_run')
    parser.add_argument('--workers',type=int,default=PARAMETERS['workers'])
    args=parser.parse_args();manifest=layout()
    if sum((args.validate_layout,args.preflight,args.run))!=1:
        raise SystemExit('Select exactly one of --validate-layout, --preflight or --run.')
    if args.validate_layout:
        print(json.dumps({'layout_hashes_and_AST_pass':True,'scientific_computation_run':False,'package_status':'staged_not_runtime_validated'}));return
    if not (args.run or args.preflight):raise SystemExit('No scientific mode selected.')
    assert 1<=args.workers<=PARAMETERS['workers']<=4
    began=time.perf_counter();DEADLINE=began+(PARAMETERS['preflight_seconds'] if args.preflight else PARAMETERS['seconds'])
    work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    save(work/'started.json',{'parameters':PARAMETERS,'workers':args.workers,'preflight_only':args.preflight,
                            'embedded_manifest_sha256':sha(HERE/'cert_cs3_manifest.py')})
    try:
        if args.preflight:
            result=preflight(work);result['elapsed_seconds']=time.perf_counter()-began
            save(work/'result.json',result)
            print(json.dumps({'preflight_passed':True,'rank_certificate_issued':False}),flush=True)
            return
        prep=build(work,DEADLINE)
        covers=complete_four_census(work,args.workers)
        reps=all_cover_orbits(work,covers)
        lifts=all_lifts(work,reps)
        tick()
        result={'status':'complete_unfiltered_portable_gate','all_gates_passed':True,
                'dictionary_rays':36720,'minimum_three_covers':prep['minimum_three_covers'],
                'minimum_three_orbits':prep['minimum_three_orbits'],
                'retained_pair_classes':prep['retained_pair_classes'],'unique_input_covers':len(covers),
                'all_cover_orbits':len(reps),'complete_negative_lift_orbits':len(lifts),
                'no_family_or_fidelity_filters':True,'no_precomputed_scientific_inputs':True,
                'analytic_prerequisite':'cert_cs3_analytic.py (also supplied as proofs/slice_reduction.md)',
                'elapsed_seconds':time.perf_counter()-began}
    except BaseException as error:
        save(work/'result.json',{'status':'INCOMPLETE_OR_FAILED_NOT_CERTIFIED','all_gates_passed':False,
            'error_type':type(error).__name__,'error':str(error),'elapsed_seconds':time.perf_counter()-began})
        raise
    save(work/'result.json',result)
    print('CERTIFIED chi(CS^3) >= 5',flush=True)


if __name__=='__main__':
    multiprocessing.freeze_support();main()
