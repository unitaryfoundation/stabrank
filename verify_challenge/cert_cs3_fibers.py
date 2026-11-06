"""Portable staging copy of accepted exact routines; see PROVENANCE.json.
Not executed or certified by source staging.
"""
from pathlib import Path
from itertools import product
from collections import defaultdict
import hashlib,json,time
import numpy as np
from cert_cs3_exact import stabilizer_check
TRANSITION_CAP=1024; ORDERED_PAIR_CAP=65536
SECONDS=3600; SCALE=2; DEADLINE=0
REAL=np.array([0,1,0,-1,0],dtype=np.int64)
IMAG=np.array([0,0,1,0,-1],dtype=np.int64)
I4=((1,0),(0,1),(-1,0),(0,-1))
SIGNS=np.array([[1-2*((k>>j)&1) for j in range(4)] for k in range(16)],dtype=np.int64)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def tick():
    if time.perf_counter()>=DEADLINE:raise TimeoutError('global60_second_cap')

def decoded(row):return tuple((0,0) if int(e)<0 else I4[int(e)] for e in row)

def phase_product(a,b):
    return np.array([-1 if a[x&3]<0 or b[x>>2]<0 else (a[x&3]+b[x>>2])%4 for x in range(16)],dtype=np.int8)

def phased(row,k):return np.where(row>=0,(row+k)%4,-1).astype(np.int8)

def weighted(rows,c):
    re=REAL[np.asarray(rows,dtype=np.int64)+1];im=IMAG[np.asarray(rows,dtype=np.int64)+1]
    return np.concatenate((int(c[0])*re-int(c[1])*im,int(c[1])*re+int(c[0])*im),axis=-1)

def target_scaled(row):return weighted(row,(SCALE,0))

def pauli(row,rep):
    a,b,k=map(int,rep);x=np.arange(len(row),dtype=np.int64);before=x^a
    old=row[before]
    parity=np.array([(int(t)&b).bit_count()&1 for t in before],dtype=np.int8)
    return np.where(old>=0,(old+2*parity+k)%4,-1).astype(np.int8)

def compose(left,right):
    a,b,k=map(int,left);c,d,l=map(int,right)
    return (a^c,b^d,(k+l+2*((b&c).bit_count()&1))%4)

def rank_phase(rows):
    p,i=1009,469
    a=np.array([0,1,i,p-1,p-i],dtype=np.int64)[rows.astype(np.int64)+1].T.copy()
    rank=0
    for col in range(a.shape[1]):
        nz=np.flatnonzero(a[rank:,col])
        if not len(nz):continue
        at=rank+int(nz[0]);a[[rank,at]]=a[[at,rank]]
        a[rank]=a[rank]*pow(int(a[rank,col]),-1,p)%p
        for j in range(rank+1,len(a)):a[j]=(a[j]-int(a[j,col])*a[rank])%p
        rank+=1
        if rank==a.shape[1]:break
    return rank

def bases():
    a=np.array([-1,-1,-1,0],dtype=np.int8)
    b=np.array([0,0,0,0],dtype=np.int8)
    c=np.array([0,0,0,2],dtype=np.int8)
    target=np.array([int((x&3)==3)+int((x>>2)==3) for x in range(16)],dtype=np.int8)
    data=[('rectangle',np.array([phase_product(b,b),phase_product(a,a),phase_product(a,b),phase_product(b,a)]),
           np.array([[2,0],[0,-4],[-2,2],[-2,2]],dtype=np.int64)),
          ('nonrigid_two_row',np.array([phase_product(a,b),phase_product(a,a),phase_product(c,b),phase_product(c,c)]),
           np.array([[2,2],[-4,0],[1,1],[1,-1]],dtype=np.int64))]
    for name,rows,coeff in data:
        assert rows.shape==(4,16) and rank_phase(rows)==4
        assert all(stabilizer_check(decoded(row)) for row in rows)
        assert all(np.any(cc) for cc in coeff)
        assert np.array_equal(sum(weighted(row,cc) for row,cc in zip(rows,coeff)),target_scaled(target))
    return data,target

def single_global(row,pa,pb,negative=False):
    return np.concatenate((row,pauli(row,pa),pauli(row,pb),phased(pauli(row,compose(pb,pa)),2 if negative else 0)))

def preflight(data,target):
    assert 469*469%1009==1008 and 1009>4**4
    delta=np.full(16,-1,dtype=np.int8);delta[0]=0
    px,pz=(1,0,0),(0,1,0)
    zx=pauli(delta,compose(pz,px));xz=pauli(delta,compose(px,pz))
    assert np.array_equal(zx,phased(xz,2))
    for negative in (False,True):assert stabilizer_check(decoded(single_global(delta,px,pz,negative)))
    # A true nonrigid T transition, with actual Pauli phases retained.
    _,rows,coeff=data[1]
    transition=np.array([pauli(row,(3,0,0)) if j in (0,2) else row for j,row in enumerate(rows)])
    assert not np.array_equal(transition,rows)
    assert np.array_equal(sum(weighted(row,cc) for row,cc in zip(transition,coeff)),target_scaled(target))
    global_rows=np.array([single_global(row,(3,0,0) if j in (0,2) else (0,0,0),(0,0,0)) for j,row in enumerate(rows)])
    assert all(stabilizer_check(decoded(row)) for row in global_rows)
    plus_target=np.tile(target,4)
    assert np.array_equal(sum(weighted(row,cc) for row,cc in zip(global_rows,coeff)),target_scaled(plus_target))
    return {'all_passed':True,'two_base_identities_exact':True,'two_bases_rank4':True,
            'noncommuting_actual_phase_control':True,'both_CZ_signs_stabilizer':True,
            'explicit_nonrigid_transition_exact':True,'nontrivial_plus_factor64_amplitude_control':True}

def actual_images(row):
    images=[];reps=[];lookup={};multiplicity=[];stabilizers=[]
    for a,b,k in product(range(16),range(16),range(4)):
        tick();rep=(a,b,k);image=pauli(row,rep);key=image.tobytes()
        if key not in lookup:
            lookup[key]=len(images);images.append(image);reps.append(rep);multiplicity.append(0)
        multiplicity[lookup[key]]+=1
        if np.array_equal(image,row):stabilizers.append(rep)
    assert len(images)==64 and set(multiplicity)=={16} and len(stabilizers)==16
    assert np.array_equal(images[0],row) and reps[0]==(0,0,0)
    return np.array(images,dtype=np.int8),np.array(reps,dtype=np.int16),lookup,stabilizers

def build_products(row,images,reps,lookup,stabilizers):
    table=np.empty((64,64),dtype=np.uint8)
    for ia in range(64):
        tick()
        for ib in range(64):
            direct=pauli(images[ia],reps[ib])
            formula=pauli(row,compose(reps[ib],reps[ia]))
            assert np.array_equal(direct,formula)
            table[ia,ib]=lookup[direct.tobytes()]
    # Every exact stabilizer representative variation changes the product at most by sign.
    ia,ib=17,39;original=images[int(table[ia,ib])]
    for stabilizer in stabilizers:
        altered=compose(reps[ib],stabilizer)
        assert np.array_equal(pauli(row,altered),images[ib])
        changed=pauli(row,compose(altered,reps[ia]))
        assert np.array_equal(changed,original) or np.array_equal(changed,phased(original,2))
    return table

def enumerate_transitions(weighted_images,target):
    left=(weighted_images[0][:,None,:]+weighted_images[1][None,:,:]).reshape(4096,32)
    right=(target-(weighted_images[2][:,None,:]+weighted_images[3][None,:,:])).reshape(4096,32)
    groups=defaultdict(list)
    for index,row in enumerate(left):groups[np.ascontiguousarray(row,dtype='<i8').tobytes()].append(index)
    transitions=[];raw_matches=0;processed=0
    for r,row in enumerate(right):
        tick();processed+=1
        for l in groups.get(np.ascontiguousarray(row,dtype='<i8').tobytes(),[]):
            raw_matches+=1
            if len(transitions)>=TRANSITION_CAP:
                return np.array(transitions,dtype=np.uint8),{'complete':False,'reason':'transition_cap',
                    'right_sums_processed':processed,'raw_matches_seen':raw_matches,'stored_transitions':len(transitions)}
            tr=(l//64,l%64,r//64,r%64)
            assert np.array_equal(sum(weighted_images[j,tr[j]] for j in range(4)),target)
            transitions.append(tr)
    assert len(set(transitions))==len(transitions) and (0,0,0,0) in transitions
    return np.array(transitions,dtype=np.uint8),{'complete':True,'left_sums':4096,'right_sums_processed':processed,
        'left_distinct_integer_keys':len(groups),'raw_matches_seen':raw_matches,'stored_transitions':len(transitions),
        'all_matching_pairs_retained':True,'transition_rank_filter_used':False}

def prepare_base(name,rows,coeff,target,folder):
    folder.mkdir(exist_ok=False);all_images=[];all_reps=[];tables=[];lookups=[]
    for row in rows:
        images,reps,lookup,stabilizers=actual_images(row)
        tables.append(build_products(row,images,reps,lookup,stabilizers))
        all_images.append(images);all_reps.append(reps);lookups.append(lookup)
    images=np.array(all_images);reps=np.array(all_reps);tables=np.array(tables)
    wi=np.array([weighted(im,c) for im,c in zip(images,coeff)])
    transitions,coverage=enumerate_transitions(wi,target_scaled(target))
    expected=None
    if name=='nonrigid_two_row':
        expected=tuple(lookups[j][(pauli(row,(3,0,0)) if j in (0,2) else row).tobytes()] for j,row in enumerate(rows))
        if coverage['complete']:assert expected in set(map(tuple,transitions))
    if name=='rectangle' and coverage['complete']:assert transitions.tolist()==[[0,0,0,0]]
    np.savez_compressed(folder/'prepared.npz',base_rows=rows,coefficients_scaled=coeff,target=target,
        actual_images=images,pauli_representatives=reps,product_tables=tables,transitions=transitions)
    save(folder/'preparation.json',{'base':name,'coverage':coverage,'scale':SCALE,
        'base_phase_rows':rows.tolist(),'coefficients_scaled':coeff.tolist(),
        'expected_nonrigid_transition':list(expected) if expected is not None else None,
        'prepared_npz_sha256':sha(folder/'prepared.npz')})
    return {'name':name,'rows':rows,'coeff':coeff,'images':images,'reps':reps,'tables':tables,
            'weighted':wi,'transitions':transitions,'coverage':coverage}

def validate_positive(base,target,a,b,sign_index,phase11):
    rows=[];representatives=[]
    for j in range(4):
        pa=base['reps'][j,int(a[j])];pb=base['reps'][j,int(b[j])]
        row=single_global(base['rows'][j],pa,pb,bool((sign_index>>j)&1))
        assert np.array_equal(row[:16],base['rows'][j])
        assert stabilizer_check(decoded(row));rows.append(row)
        representatives.append({'P_A':pa.astype(int).tolist(),'P_B':pb.astype(int).tolist(),
                                'CZ_sign':int(SIGNS[sign_index,j])})
    rows=np.array(rows);expected=np.concatenate((target,target,target,phased(target,phase11)))
    assert np.array_equal(sum(weighted(row,c) for row,c in zip(rows,base['coeff'])),target_scaled(expected))
    return {'transition_A':list(map(int,a)),'transition_B':list(map(int,b)),'sign_index':sign_index,
        'term_representatives':representatives,'global_phase_rows':rows.tolist(),
        'all64_amplitudes_exact':True,'all_four_global_stabilizers_checked':True,
        'target_phase11':phase11,'coefficients_scaled':base['coeff'].tolist(),'scale':SCALE}

def joint_test(base,target,phase11,taskname,folder):
    started=time.perf_counter();transitions=base['transitions'];m=len(transitions)
    if not base['coverage']['complete']:
        result={'task':taskname,'status':'incomplete_transition_list','complete_negative':False,
                'transition_coverage':base['coverage'],'ordered_pairs_examined':0,'sign_vectors_evaluated':0}
        save(folder/(taskname+'.json'),result);return result
    wanted=target_scaled(phased(target,phase11));examined=0;witness=None;reason=None
    for ia,a in enumerate(transitions):
        for ib,b in enumerate(transitions):
            tick()
            if examined>=ORDERED_PAIR_CAP:reason='ordered_pair_cap';break
            vector=np.array([base['weighted'][j,int(base['tables'][j,int(a[j]),int(b[j])])] for j in range(4)])
            totals=SIGNS@vector;hits=np.flatnonzero(np.all(totals==wanted,axis=1));examined+=1
            if len(hits):
                witness=validate_positive(base,target,a,b,int(hits[0]),phase11)
                witness['transition_indices']=[ia,ib];break
            if examined%1024==0:
                save(folder/'progress.json',{'active_task':taskname,'ordered_pairs_examined':examined,
                    'ordered_pairs_required':m*m,'sign_vectors_evaluated':16*examined})
        if witness is not None or reason is not None:break
    if witness is not None:status='exact_lift_witness'
    elif reason is not None:status='incomplete_ordered_pair_scan'
    else:status='complete_no_lift_for_fixed_base';assert examined==m*m
    result={'task':taskname,'base':base['name'],'target_phase11':phase11,'status':status,
        'transition_count':m,'transition_list_complete':True,'ordered_pairs_required':m*m,
        'ordered_pairs_examined':examined,'sign_vectors_evaluated':16*examined,'stopping_reason':reason,
        'complete_negative':status=='complete_no_lift_for_fixed_base','witness':witness,
        'elapsed_seconds':time.perf_counter()-started,
        'scope':'This exact base and its fixed coefficients only; no transfer to another type-pattern member.'}
    save(folder/(taskname+'.json'),result);return result
