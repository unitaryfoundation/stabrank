"""Portable staging copy of accepted exact routines; see PROVENANCE.json.
Not executed or certified by source staging.
"""
from itertools import combinations,product
from collections import Counter
from math import comb,lcm,isqrt
import time
import cert_cs3_pair_primitives as baseline
prior=baseline.prior; exact=prior.exact; np=prior.np
P,I=prior.P,prior.I
VALUES=prior.VALUES
DEADLINE=0
RE=np.array([0,1,0,-1,0],dtype=np.int64)
IM=np.array([0,0,1,0,-1],dtype=np.int64)
PATTERNS=np.array(list(product(range(5),repeat=3)),dtype=np.int64)
QVALUES=(exact.GZERO,)+tuple(exact.PHASES)


def tick(where='projected calibration deadline'):
    if time.perf_counter()>=DEADLINE:raise TimeoutError(where)

def serial(cc):return [exact.serial_coeff(c) for c in cc]

def gdot(a,b):
    s=exact.GZERO
    for x,y in zip(a,b):s=exact.gadd(s,exact.gmul(x,y))
    return s

class Engine:
    def __init__(self,pool,target,L,inverses):
        prior.phase_guard(pool);prior.phase_guard(np.array([target],dtype=np.int8))
        self.pool=pool;self.target=target;self.L=np.asarray(L,dtype=np.int64)
        assert self.L.shape==(3,16) and np.all((self.L>=0)&(self.L<P))
        self.inverses=inverses;self.dm=VALUES[pool.astype(np.int64).T+1]
        self.tm=VALUES[target.astype(np.int64)+1]
        self.ld=self.L@self.dm%P
        self.pattern_cache={};self.decoded_cache={}
        self.real=RE[pool.astype(np.int64)+1];self.imag=IM[pool.astype(np.int64)+1]
        self.tr=RE[target.astype(np.int64)+1];self.ti=IM[target.astype(np.int64)+1]

    def rayq(self,j):
        if j not in self.decoded_cache:self.decoded_cache[j]=exact.decode(self.pool[j])
        return self.decoded_cache[j]

    def scan(self,u,v,emit):
        started=time.perf_counter();tick()
        B=VALUES[np.asarray([u,v,self.target],dtype=np.int64).T+1]
        assert len(prior.pivots(B[:,:2]))==2
        pv=prior.pivots(B)
        if len(pv)!=3:
            cc=prior.exact_cover([u,v],self.target)
            emit({'kind':'unexpected_target_in_retained_span','coefficients':serial(cc)})
            return {'complete':False,'stop':'unexpected_target_in_retained_span','base_rank':len(pv)}
        pvt=tuple(pv)
        if pvt not in self.pattern_cache:
            codes=self.pool[:,pv].astype(np.int16)+1
            self.pattern_cache[pvt]=(25*codes[:,0]+5*codes[:,1]+codes[:,2]).astype(np.uint8)
        bins=self.pattern_cache[pvt]
        invB=prior.inverse(B[pv])
        a125=invB@VALUES[PATTERNS.T]%P
        correction=B@a125%P
        projected_correction=(self.L@B%P)@a125%P
        prep_seconds=time.perf_counter()-started
        t=time.perf_counter()
        z=(self.ld-projected_correction[:,bins])%P
        n=len(self.pool);positions=np.arange(n)
        leadpos=np.argmax(z!=0,axis=0);lead=z[leadpos,positions]
        znon=np.flatnonzero(lead);zzero=np.flatnonzero(lead==0)
        invlead=self.inverses[lead[znon]]
        normalized=z[:,znon]*invlead%P
        first=leadpos[znon]
        K=np.where(first==0,normalized[1]+P*normalized[2],
                   np.where(first==1,P*P+normalized[2],P*P+P)).astype(np.uint64)
        sigma=(a125[2,bins[znon]]*invlead%P).astype(np.uint64)
        joint=K*np.uint64(P)+sigma
        assert (P*P+P)*P+(P-1)<2**48
        projection_seconds=time.perf_counter()-t
        t=time.perf_counter()
        order=np.argsort(joint,kind='stable');sj=joint[order]
        sk=sj//np.uint64(P);ss=sj%np.uint64(P)
        bounds=np.r_[0,np.flatnonzero(sk[1:]!=sk[:-1])+1,len(order)] if len(order) else np.array([0])
        jb=np.r_[0,np.flatnonzero(sj[1:]!=sj[:-1])+1,len(order)] if len(order) else np.array([0])
        sizes=np.diff(bounds);jsizes=np.diff(jb)
        same=int(np.sum(sizes*(sizes-1)//2));same_sigma=int(np.sum(jsizes*(jsizes-1)//2))
        active=np.flatnonzero((sizes>=2)&(ss[bounds[:-1]]!=ss[bounds[1:]-1])) if len(order) else []
        sort_seconds=time.perf_counter()-t
        stats=Counter();full_cache={};exact_a={};exact_B=None;exact_inv=None
        full_check_seconds=exact_seconds=0.0
        uvq=(exact.decode(u),exact.decode(v));tq=exact.decode(self.target)
        uvreal=RE[np.asarray([u,v],dtype=np.int64)+1];uvimag=IM[np.asarray([u,v],dtype=np.int64)+1]

        def full(j):
            if j not in full_cache:full_cache[j]=(self.dm[:,j]-correction[:,int(bins[j])])%P
            return full_cache[j]

        def exact_coordinates(j):
            nonlocal exact_B,exact_inv
            code=int(bins[j])
            if exact_inv is None:
                exact_B=(*uvq,tq)
                exact_inv=exact.exact_inverse([[vec[x] for vec in exact_B] for x in pv])
            if code not in exact_a:
                pp=[QVALUES[int(k)] for k in PATTERNS[code]]
                exact_a[code]=tuple(gdot(row,pp) for row in exact_inv)
            return exact_a[code]

        def check_integer_identity(freeids,cc):
            denominator=lcm(*(exact.F(x).denominator for c in cc for x in c))
            cr=np.array([int(c[0]*denominator) for c in cc],dtype=np.int64)
            ci=np.array([int(c[1]*denominator) for c in cc],dtype=np.int64)
            assert denominator+sum(abs(int(a))+abs(int(b)) for a,b in zip(cr,ci))<2**50
            rr=np.vstack((uvreal,self.real[list(freeids)]));ii=np.vstack((uvimag,self.imag[list(freeids)]))
            assert np.array_equal(cr@rr-ci@ii,denominator*self.tr)
            assert np.array_equal(cr@ii+ci@rr,denominator*self.ti)

        def exact_pair(ia,ib):
            nonlocal exact_seconds
            t0=time.perf_counter();tick('exact pair reconstruction deadline')
            aa=exact_coordinates(ia);ab=exact_coordinates(ib)
            k=int(np.flatnonzero(full(ia))[0])
            ra=exact.gsub(self.rayq(ia)[k],gdot([v[k] for v in exact_B],aa))
            rb=exact.gsub(self.rayq(ib)[k],gdot([v[k] for v in exact_B],ab))
            assert ra!=exact.GZERO and rb!=exact.GZERO
            lam=exact.gdiv(rb,ra)
            d=tuple(exact.gsub(b,exact.gmul(lam,a)) for a,b in zip(aa,ab))
            assert d[2]!=exact.GZERO
            cc=(exact.gneg(exact.gdiv(d[0],d[2])),exact.gneg(exact.gdiv(d[1],d[2])),
                exact.gneg(exact.gdiv(lam,d[2])),exact.gdiv(exact.GONE,d[2]))
            assert cc[2]!=exact.GZERO and cc[3]!=exact.GZERO
            check_integer_identity((ia,ib),cc)
            # B_P is exactly invertible, ra!=0 makes [B,u] rank4;
            # d_target!=0 transfers that independence to [r,s,u,w].
            bad=[j for j in (0,1) if cc[j]==exact.GZERO]
            kind='rejected_zero_retained_coefficient' if bad else 'independent_four_cover'
            emit({'kind':kind,'free_ids':[ia,ib],'coefficients':serial(cc),'zero_retained_positions':bad})
            stats['rejected_zero_retained' if bad else 'accepted_independent_four_covers']+=1
            stats['candidate_pairs']+=1;exact_seconds+=time.perf_counter()-t0

        def consider(ia,ib,la,lb,branch):
            nonlocal full_check_seconds
            tick('projected full-residual decision deadline');t0=time.perf_counter()
            valid=np.array_equal(full(ia)*int(lb)%P,full(ib)*int(la)%P)
            full_check_seconds+=time.perf_counter()-t0
            stats[branch+'_pairs_checked']+=1
            if not valid:
                assert branch=='projected'
                stats['false_projected_collisions']+=1;return
            exact_pair(ia,ib)

        t=time.perf_counter()
        for run in active:
            tick('cross-sigma bucket deadline')
            start,end=int(bounds[run]),int(bounds[run+1])
            sb=ss[start:end]
            groupbounds=np.r_[0,np.flatnonzero(sb[1:]!=sb[:-1])+1,end-start]+start
            groups=[znon[order[a:b]] for a,b in zip(groupbounds[:-1],groupbounds[1:])]
            for ga,gb in combinations(groups,2):
                for x,y in product(ga,gb):
                    ia,ib=sorted((int(x),int(y)))
                    consider(ia,ib,lead[ia],lead[ib],'projected')
        projected_loop_seconds=time.perf_counter()-t
        assert stats['projected_pairs_checked']==same-same_sigma

        # Every zero projected residual is processed with the full13-coordinate
        # quotient. A true-zero full residual is not silently discarded.
        t=time.perf_counter();nonpv=[x for x in range(16) if x not in pv]
        rq=np.column_stack([full(int(j)) for j in zzero]) if len(zzero) else np.empty((16,0),dtype=np.int64)
        nonzero=np.any(rq!=0,axis=0)
        truezero=zzero[~nonzero]
        for j0 in truezero:
            j=int(j0);tick('zero residual exact bookkeeping deadline')
            if a125[2,int(bins[j])]:
                t0=time.perf_counter();a=exact_coordinates(j);assert a[2]!=exact.GZERO
                cc=(exact.gneg(exact.gdiv(a[0],a[2])),exact.gneg(exact.gdiv(a[1],a[2])),exact.gdiv(exact.GONE,a[2]))
                check_integer_identity((j,),cc)
                emit({'kind':'zero_residual_three_cover','free_id':j,'coefficients':serial(cc)})
                stats['zero_residual_three_covers']+=1;exact_seconds+=time.perf_counter()-t0
        fid=zzero[nonzero];fr=rq[:,nonzero]
        fpos=np.argmax(fr!=0,axis=0);fl=fr[fpos,np.arange(len(fid))]
        finv=self.inverses[fl]
        fkey=np.ascontiguousarray((fr[nonpv]*finv).T%P,dtype='<u2').view(np.dtype((np.void,26))).reshape(-1)
        fsigma=a125[2,bins[fid]]*finv%P
        fo=np.argsort(fkey,kind='stable');fk=fkey[fo]
        fb=np.r_[0,np.flatnonzero(fk[1:]!=fk[:-1])+1,len(fid)] if len(fid) else np.array([0])
        fsz=np.diff(fb);fsame=int(np.sum(fsz*(fsz-1)//2));fdiff=0
        for run in np.flatnonzero(fsz>=2):
            tick('full13 fallback bucket deadline')
            members=fo[fb[run]:fb[run+1]];values,counts=np.unique(fsigma[members],return_counts=True)
            fdiff+=comb(len(members),2)-sum(comb(int(c),2) for c in counts)
            groups=[members[fsigma[members]==z] for z in values]
            for ga,gb in combinations(groups,2):
                for a,b in product(ga,gb):
                    ia,ib=int(fid[a]),int(fid[b]);la,lb=int(fl[a]),int(fl[b])
                    if ia>ib:ia,ib,la,lb=ib,ia,lb,la
                    consider(ia,ib,la,lb,'fallback')
        assert stats['fallback_pairs_checked']==fdiff
        fallback_seconds=time.perf_counter()-t
        nz,n0,nf=len(znon),len(truezero),len(fid)
        accounting={'projected_zero_nonzero':len(zzero)*nz,
            'different_nonzero_projected_key':comb(nz,2)-same,
            'same_projected_key_same_sigma':same_sigma,
            'same_projected_key_different_sigma_false_full':stats['false_projected_collisions'],
            'same_projected_key_different_sigma_full_valid':stats['projected_pairs_checked']-stats['false_projected_collisions'],
            'fallback_truezero_truezero':comb(n0,2),'fallback_truezero_nonzero':n0*nf,
            'fallback_different_full_key':comb(nf,2)-fsame,'fallback_same_key_same_sigma':fsame-fdiff,
            'fallback_same_key_different_sigma':fdiff}
        assert sum(accounting.values())==comb(n,2)
        assert stats['candidate_pairs']==stats['accepted_independent_four_covers']+stats['rejected_zero_retained']
        tick()
        return {'complete':True,'base_rank':3,'pivot_rows':pv,'phase_patterns_used':int(len(np.unique(bins))),
            'zero_projection_rays':len(zzero),'zero_residual_rays':n0,'full13_fallback_nonzero_rays':nf,
            'zero_residual_three_covers':stats['zero_residual_three_covers'],
            'projected_nonzero_rays':nz,'projected_buckets':len(bounds)-1,
            'projected_cross_sigma_buckets':len(active),'maximum_projected_bucket':int(sizes.max(initial=0)),
            'projected_cross_sigma_candidates':stats['projected_pairs_checked'],
            'false_projected_collisions':stats['false_projected_collisions'],
            'fallback_cross_sigma_candidates':stats['fallback_pairs_checked'],
            'candidate_pairs':stats['candidate_pairs'],'accepted_independent_four_covers':stats['accepted_independent_four_covers'],
            'rejected_zero_retained':stats['rejected_zero_retained'],
            'all_free_pair_accounting':accounting,'all_pairs_accounted':True,'every_candidate_exact_Qi_checked':True,
            'full_residuals_materialized':len(full_cache),
            'timings':{'pivot_and125cache':prep_seconds,'projection_and_normalization':projection_seconds,
                'numeric_sort_and_run_accounting':sort_seconds,'projected_candidate_loop':projected_loop_seconds,
                'full13_fallback':fallback_seconds,'full_residual_comparisons':full_check_seconds,
                'exact_reconstruction_and_emit':exact_seconds},
            'elapsed_seconds':time.perf_counter()-started}

def controls(pool,target,L,inv):
    flags={};covers=baseline.cover_controls(pool,target)
    def tiny(u,v,t,rows,projection=L):
        records=[];eng=Engine(np.asarray(rows,dtype=np.int8),np.asarray(t,dtype=np.int8),projection,inv)
        stats=eng.scan(u,v,records.append);return stats,records
    for c in covers:
        rows=pool[c['ids']]
        s,r=tiny(rows[0],rows[1],target,rows[2:])
        assert s['complete'] and s['accepted_independent_four_covers']==1
        assert r[0]['free_ids']==[0,1]
    flags['split_and_nonrefinement_positive_controls']=True
    rows=pool[covers[0]['ids']]
    s,r=tiny(rows[0],rows[1],target,rows[2:],np.zeros((3,16),dtype=np.int64))
    assert s['complete'] and s['zero_projection_rays']==2 and s['full13_fallback_nonzero_rays']==2
    assert s['accepted_independent_four_covers']==1 and s['fallback_cross_sigma_candidates']==1
    flags['deliberate_zero_projection_full13_fallback']=True
    e=[baseline.ray({j:0}) for j in range(7)]
    falseL=np.zeros((3,16),dtype=np.int64);falseL[0,3]=falseL[0,4]=1;falseL[1,5]=falseL[2,6]=1
    s,r=tiny(e[0],e[1],e[2],[e[3],baseline.ray({2:0,4:0})],falseL)
    assert s['complete'] and s['false_projected_collisions']==1 and s['candidate_pairs']==0 and not r
    flags['deliberate_false_projected_collision_rejected']=True
    t=baseline.ray({0:0,1:0,2:0});multi=[e[3],baseline.ray({2:0,3:0}),baseline.ray({2:0,3:3})]
    s,r=tiny(e[0],e[1],t,multi,falseL)
    assert s['complete'] and s['accepted_independent_four_covers']==3
    assert {tuple(x['free_ids']) for x in r}=={(0,1),(0,2),(1,2)}
    flags['three_sigma_classes_all_cross_pairs']=True
    t=baseline.ray({j:0 for j in range(4)})
    s,r=tiny(e[0],e[1],t,[e[2],e[3],baseline.ray({0:0,2:1}),baseline.ray({1:0,3:1})])
    assert s['complete'] and s['accepted_independent_four_covers']==4
    assert {tuple(x['free_ids']) for x in r}=={(0,1),(0,3),(1,2),(2,3)}
    flags['all_four_old_synthetic_pairs']=True
    s,r=tiny(e[0],e[1],t,[e[4],baseline.ray({0:0,4:1})])
    assert s['complete'] and s['candidate_pairs']==0 and not r
    flags['same_sigma_skipped']=True
    a,f,g=prior.known_rays()
    s,r=tiny(a,e[1],target,[f,g]);assert s['rejected_zero_retained']==1 and r[0]['zero_retained_positions']==[1]
    s,r=tiny(a,f,target,[g],np.zeros((3,16),dtype=np.int64))
    assert s['zero_residual_three_covers']==1 and r[0]['kind']=='zero_residual_three_cover'
    s,r=tiny(e[0],e[1],baseline.ray({0:0,1:0}),[e[2]])
    assert not s['complete'] and s['stop']=='unexpected_target_in_retained_span'
    flags.update(zero_retained_rejected=True,true_zero_lower_cover_preserved=True,shorter_base_guard=True,all_passed=True)
    return flags
