"""Portable staging copy of accepted exact routines; see PROVENANCE.json.
Not executed or certified by source staging.
"""
from itertools import combinations,product
from collections import Counter
from math import comb,isqrt
import time
import numpy as np
import cert_cs3_exact as exact
from cert_cs3_exact import stabilizer_check
P=65521; I=41224
VALUES=np.array([0,1,I,P-1,P-I],dtype=np.int64)


def codes(v):return np.asarray(v,dtype=np.int8)

def phase_guard(a):
    assert a.ndim==2 and a.shape[1]==16 and np.all((a>=-1)&(a<=3))
    assert np.all(np.any(a>=0,axis=1))

def target():return codes([sum(((x>>(2*j))&1)*((x>>(2*j+1))&1) for j in range(2))%4 for x in range(16)])

def known_rays():
    return codes([[0 if x==0 else -1 for x in range(16)],
        [x.bit_count()%4 for x in range(16)],
        [(-x.bit_count()+2*sum(((x>>(2*j))&1)*((x>>(2*j+1))&1) for j in range(2)))%4 for x in range(16)]])

def pivots(a):
    q=np.asarray(a,dtype=np.int64).T.copy()%P; rank=0; out=[]
    for col in range(q.shape[1]):
        nz=np.flatnonzero(q[rank:,col])
        if not len(nz):continue
        row=rank+int(nz[0]);q[[rank,row]]=q[[row,rank]]
        q[rank]=q[rank]*pow(int(q[rank,col]),-1,P)%P
        f=q[:,col].copy();f[rank]=0;q=(q-f[:,None]*q[rank:rank+1])%P
        out.append(col);rank+=1
        if rank==q.shape[0]:break
    return out

def inverse(a):
    k=len(a);q=np.concatenate((np.asarray(a,dtype=np.int64),np.eye(k,dtype=np.int64)),axis=1)%P
    for j in range(k):
        nz=np.flatnonzero(q[j:,j]);assert len(nz)
        r=j+int(nz[0]);q[[j,r]]=q[[r,j]];q[j]=q[j]*pow(int(q[j,j]),-1,P)%P
        f=q[:,j].copy();f[j]=0;q=(q-f[:,None]*q[j:j+1])%P
    assert np.array_equal(q[:,:k],np.eye(k,dtype=np.int64))
    return q[:,k:]

def exact_cover(rows,t):
    vs=[exact.decode(r) for r in rows];tt=exact.decode(t)
    cc=exact.exact_solve(vs,tt)
    assert cc is not None and exact.exact_identity(vs,cc,tt)
    assert all(stabilizer_check(v) for v in vs)
    return cc

def dictionary(deadline):
    rays=[];support_masks=set();counts=Counter();linear_counts=Counter()
    for d in range(5):
        for ps in combinations(range(4),d):
            free=[j for j in range(4) if j not in ps]
            places=[(r,j) for r,p in enumerate(ps) for j in free if j>p]
            for bits in product(range(2),repeat=len(places)):
                basis=[1<<p for p in ps]
                for bit,(r,j) in zip(bits,places):basis[r]|=bit<<j
                assert all((b&((1<<ps[r])-1))==0 and all(((b>>p)&1)==(r==s) for s,p in enumerate(ps)) for r,b in enumerate(basis))
                linear_counts[d]+=1
                offsets=[]
                for mask in range(1<<d):
                    v=0
                    for j,b in enumerate(basis):
                        if mask>>j&1:v^=b
                    offsets.append(v)
                assert len(set(offsets))==1<<d
                for shiftbits in product(range(2),repeat=len(free)):
                    if time.perf_counter()>=deadline:raise TimeoutError('dictionary preparation cap')
                    origin=sum(bit<<j for bit,j in zip(shiftbits,free))
                    points=[origin^z for z in offsets];mask=sum(1<<z for z in points)
                    assert mask not in support_masks;support_masks.add(mask);counts[d]+=1
                    first=points.index(min(points));qp=list(combinations(range(d),2))
                    for lin in product(range(4),repeat=d):
                        for quad in product(range(2),repeat=len(qp)):
                            phase=[(sum(lin[j] for j in range(d) if m>>j&1)+2*sum(b for b,(j,k) in zip(quad,qp) if m>>j&1 and m>>k&1))%4 for m in range(1<<d)]
                            row=[-1]*16;shift=phase[first]
                            for x,z in zip(points,phase):row[x]=(z-shift)%4
                            rays.append(row)
    a=codes(rays);phase_guard(a)
    assert [linear_counts[d] for d in range(5)]==[1,15,35,15,1]
    assert [counts[d] for d in range(5)]==[16,120,140,30,1]
    assert len(a)==sum(counts[d]*4**d*2**comb(d,2) for d in range(5))==36720
    assert len({r.tobytes() for r in a})==36720
    assert np.all(a[np.arange(len(a)),np.argmax(a>=0,axis=1)]==0)
    return a,{'linear_support_counts':[linear_counts[d] for d in range(5)],'affine_support_counts':[counts[d] for d in range(5)],'all36720_distinct_canonical_rays':True}

def generators(t):
    specs=[]
    for block in range(2):
        for j in range(3):
            pi=list(range(4));pi[j],pi[j+1]=pi[j+1],pi[j]
            perm=[(x&~(3<<(2*block)))|(pi[(x>>(2*block))&3]<<(2*block)) for x in range(16)]
            specs.append({'name':f'block{block}_transpose{j}_{j+1}','permutation':perm,'phase':[(int(t[x])-int(t[perm[x]]))%4 for x in range(16)]})
    specs.append({'name':'copy_swap','permutation':[((x&3)<<2)|(x>>2) for x in range(16)],'phase':[0]*16})
    for s in specs:
        assert sorted(s['permutation'])==list(range(16))
        assert all((s['phase'][x]+int(t[s['permutation'][x]]))%4==int(t[x]) for x in range(16))
    return specs

def scan(u,t,pool,deadline):
    phase_guard(pool);phase_guard(codes([u,t]))
    bm=VALUES[codes([u,t]).astype(np.int64).T+1];pv=pivots(bm)
    if len(pv)<2:
        cc=exact_cover([u],t)
        return [],{'base_rank':len(pv)},False,{'kind':'rank_at_most_one','coefficients':[exact.serial_coeff(c) for c in cc]}
    inv=inverse(bm[pv]);dm=VALUES[pool.astype(np.int64).T+1]
    co=inv@dm[pv]%P;res=(dm-bm@co)%P
    assert np.all(res[pv]==0)
    nz=np.any(res!=0,axis=0);zero=np.flatnonzero(~nz)
    for j in zero:
        if co[1,j]:
            cc=exact_cover([u,pool[j]],t)
            return [],{'base_rank':2},False,{'kind':'rank_at_most_two_zero_residual','free_id':int(j),'coefficients':[exact.serial_coeff(c) for c in cc]}
    ids=np.flatnonzero(nz);nonpv=[j for j in range(16) if j not in pv]
    rr=res[:,ids];leading=np.argmax(rr!=0,axis=0);lv=rr[leading,np.arange(len(ids))]
    ii=np.array([pow(int(v),-1,P) for v in lv],dtype=np.int64)
    keys=np.ascontiguousarray((rr[nonpv]*ii).T%P,dtype='<u2').view(np.dtype((np.void,2*len(nonpv)))).reshape(-1)
    sigma=co[1,ids]*ii%P;order=np.argsort(keys,kind='stable');ks=keys[order]
    bounds=np.r_[0,np.flatnonzero(ks[1:]!=ks[:-1])+1,len(ids)]
    hits=[];same_pairs=0;sigma_pairs=0;max_bucket=0
    for start,end in zip(bounds[:-1],bounds[1:]):
        if time.perf_counter()>=deadline:return hits,{'processed_buckets':int(np.searchsorted(bounds,start))},False,{'kind':'time_limit'}
        size=int(end-start);same_pairs+=size*(size-1)//2;max_bucket=max(max_bucket,size)
        if size<2:continue
        members=order[start:end];labels=sigma[members];_,counts=np.unique(labels,return_counts=True)
        different=size*(size-1)//2-sum(int(n)*(int(n)-1)//2 for n in counts)
        sigma_pairs+=different
        if not different:continue
        label_groups={}
        for a in members:label_groups.setdefault(int(sigma[a]),[]).append(int(a))
        for ga,gb in combinations(label_groups.values(),2):
            for a,b in product(ga,gb):
                if time.perf_counter()>=deadline:return hits,{'base_rank':2},False,{'kind':'time_limit_during_hit_enumeration'}
                ia,ib=int(ids[a]),int(ids[b]);cc=exact_cover([u,pool[ia],pool[ib]],t)
                if any(c==exact.GZERO for c in cc):
                    return hits,{'base_rank':2},False,{'kind':'rank_at_most_two_zero_coefficient','free_ids':[ia,ib],'coefficients':[exact.serial_coeff(c) for c in cc]}
                hits.append({'free_ids':[ia,ib],'coefficients':[exact.serial_coeff(c) for c in cc]})
    n0=len(zero);nr=len(ids);n=len(pool)
    accounting={'zero_zero':n0*(n0-1)//2,'zero_nonzero':n0*nr,'same_nonzero_bucket':same_pairs,'different_nonzero_bucket':nr*(nr-1)//2-same_pairs}
    assert sum(accounting.values())==n*(n-1)//2 and len(hits)==sigma_pairs
    return hits,{'base_rank':2,'pivot_rows':pv,'zero_residual_rays':int(n0),'nonzero_residual_rays':int(nr),
        'projective_buckets':len(bounds)-1,'maximum_bucket':max_bucket,'different_sigma_pairs':sigma_pairs,
        'pair_accounting':accounting,'all_pairs_accounted':True,'every_hit_exact_Qi_checked':True},True,None

def controls():
    assert P>4**4 and all(P%d for d in range(2,isqrt(P)+1)) and I*I%P==P-1
    kr=known_rays();kt=target();hit,_,done,reason=scan(kr[0],kt,kr,float('inf'))
    assert done and reason is None and len(hit)==1 and set(hit[0]['free_ids'])=={1,2}
    def ray(ds):return codes([ds.get(x,-1) for x in range(16)])
    e0=ray({0:0});e1=ray({1:0});e2=ray({2:0});e3=ray({3:0})
    tt=ray({0:0,1:0,2:0});pp=codes([e1,e2,ray({0:0,1:1}),ray({0:0,2:1})])
    hit,_,done,reason=scan(e0,tt,pp,float('inf'))
    assert done and reason is None and {tuple(sorted(h['free_ids'])) for h in hit}=={(0,1),(0,3),(1,2),(2,3)}
    hit,_,done,reason=scan(e0,tt,codes([e3,ray({0:0,3:1})]),float('inf'))
    assert done and not hit and reason is None
    _,_,done,reason=scan(e0,ray({0:0,1:0}),codes([e1]),float('inf'))
    assert not done and reason['kind']=='rank_at_most_two_zero_residual'
    _,_,done,reason=scan(e0,e0,codes([e1]),float('inf'))
    assert not done and reason['kind']=='rank_at_most_one'
    return {'all_passed':True,'known_three_recovered':True,'all_four_planted_completions_enumerated':True,
        'constant_sigma_rejected':True,'rank_two_and_rank_one_guards_exactly_checked':True}

def close_hits(records,pool,t,gm,gp,deadline):
    solutions={};todo=[]
    def add(ids,cc):
        key=tuple(sorted(map(int,ids)));ordered={int(i):c for i,c in zip(ids,cc)};cv=tuple(ordered[i] for i in key)
        assert len(key)==len(set(key))==3 and all(c!=exact.GZERO for c in cv)
        if key in solutions:
            assert solutions[key]==cv;return
        if time.perf_counter()>=deadline:raise TimeoutError('symmetry closure cap')
        checked=exact_cover([pool[i] for i in key],t);assert checked==cv
        solutions[key]=cv;todo.append(key)
    for row in records:
        for hit in row['hits']:
            add([row['representative']]+hit['free_ids'],[exact.read_coeff(c) for c in hit['coefficients']])
    cursor=0
    while cursor<len(todo):
        if time.perf_counter()>=deadline:raise TimeoutError('symmetry closure cap')
        ids=todo[cursor];cc=solutions[ids];cursor+=1
        for mapping,phases in zip(gm,gp):
            newids=[int(mapping[i]) for i in ids]
            newcc=[exact.gmul(c,exact.PHASES[int(phases[i])]) for i,c in zip(ids,cc)]
            add(newids,newcc)
    for row in records:
        found={tuple(sorted([row['representative']]+h['free_ids'])) for h in row['hits']}
        assert found=={s for s in solutions if row['representative'] in s}
    orbit_rows=[];remaining=set(solutions)
    while remaining:
        seed=min(remaining);members={seed};queue=[seed]
        for ids in queue:
            for mapping in gm:
                new=tuple(sorted(int(mapping[i]) for i in ids));assert new in solutions
                if new not in members:members.add(new);queue.append(new)
        remaining-=members;orbit_rows.append({'representative':list(seed),'size':len(members),'members':[list(s) for s in sorted(members)]})
    return [{'ids':list(ids),'coefficients':[exact.serial_coeff(c) for c in cc]} for ids,cc in sorted(solutions.items())],orbit_rows
