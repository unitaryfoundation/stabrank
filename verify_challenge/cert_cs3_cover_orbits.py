"""Portable staging copy of accepted exact routines; see PROVENANCE.json.
Not executed or certified by source staging.
"""
from collections import deque
from fractions import Fraction as F
from numbers import Integral
import json,struct
ZERO=(F(0),F(0)); UNITS=((1,0),(0,1),(-1,0),(0,-1))
QUNITS=tuple(tuple(F(x) for x in u) for u in UNITS)
GROUP_ORDER=2304


def add(a,b):return a[0]+b[0],a[1]+b[1]

def sub(a,b):return a[0]-b[0],a[1]-b[1]

def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]

def divide(a,b):
    norm=b[0]*b[0]+b[1]*b[1];assert norm
    c=mul(a,(b[0],-b[1]));return c[0]/norm,c[1]/norm

def rotate(c,k):return mul(c,UNITS[int(k)%4])

def serial(c):return [[x.numerator,x.denominator] for x in c]

def parse(c):
    assert isinstance(c,list) and len(c)==2
    assert all(isinstance(x,list) and len(x)==2 and all(type(v) is int for v in x) and x[1]>0 for x in c)
    return tuple(F(*x) for x in c)

def compact(x):return json.dumps(x,separators=(',',':'),sort_keys=True)

def key_bytes(ids):
    assert len(ids)==4 and tuple(ids)==tuple(sorted(set(ids))) and all(type(i) is int and 0<=i<36720 for i in ids)
    return struct.pack('>4H',*ids)

def key_ids(blob):return struct.unpack('>4H',blob)

def coefficients_text(cf):return compact([serial(c) for c in cf])

def coefficients_from_text(text):return tuple(parse(c) for c in json.loads(text))

def phase_vector(row):
    assert len(row)==16 and all(isinstance(e,Integral) and not isinstance(e,bool) and -1<=e<=3 for e in row)
    return tuple(ZERO if int(e)<0 else QUNITS[int(e)] for e in row)

def exact_rank(matrix):
    a=[[tuple(F(x) for x in z) for z in row] for row in matrix]
    if not a:return 0
    width=len(a[0]);assert all(len(row)==width for row in a)
    rank=0
    for column in range(width):
        pivot=next((i for i in range(rank,len(a)) if a[i][column]!=ZERO),None)
        if pivot is None:continue
        a[rank],a[pivot]=a[pivot],a[rank]
        lead=a[rank][column]
        for i in range(rank+1,len(a)):
            if a[i][column]==ZERO:continue
            factor=divide(a[i][column],lead)
            for j in range(column,width):a[i][j]=sub(a[i][j],mul(factor,a[rank][j]))
        rank+=1
        if rank==len(a):break
    return rank

class ExactGeometry:
    def __init__(self,pool,target_codes,tick):
        assert len(pool)==36720 and len(target_codes)==16
        self.pool=pool;self.target=phase_vector(target_codes);self.tick=tick
        self.vectors={};self.ranks={};self.supports={}

    def vector(self,ray):
        if ray not in self.vectors:self.vectors[ray]=phase_vector(self.pool[ray])
        return self.vectors[ray]

    def schmidt_rank(self,ray):
        if ray not in self.ranks:
            self.tick();v=self.vector(ray)
            rank=exact_rank([v[4*i:4*i+4] for i in range(4)])
            assert rank in (1,2,4);self.ranks[ray]=rank
        return self.ranks[ray]

    def support(self,ray):
        if ray not in self.supports:self.supports[ray]=sum(int(e)>=0 for e in self.pool[ray])
        return self.supports[ray]

    def verify(self,ids,cf):
        self.tick();key_bytes(ids)
        assert len(cf)==4 and all(c!=ZERO for c in cf)
        columns=[self.vector(i) for i in ids]
        for x in range(16):
            total=ZERO
            for c,v in zip(cf,columns):total=add(total,mul(c,v[x]))
            assert total==self.target[x],('identity',ids,x)
        assert exact_rank([[v[x] for v in columns] for x in range(16)])==4,('dependent_cover',ids)

    def fingerprint(self,ids,cf):
        ent=sum(self.schmidt_rank(i)>1 for i in ids)
        q=tuple(sorted(F(self.support(i))*(c[0]*c[0]+c[1]*c[1]) for i,c in zip(ids,cf)))
        return ent,q

def transform(ids,cf,g,actions,phases):
    assert 0<=g<8
    image=[]
    for ray,c in zip(ids,cf):
        if g==7:c=(c[0],-c[1])
        image.append((int(actions[g][ray]),rotate(c,int(phases[g][ray]))))
    image.sort(key=lambda x:x[0])
    return tuple(r for r,c in image),tuple(c for r,c in image)

def full_orbit(ids,cf,actions,phases,geometry,tick):
    """All8generators; caller has verified actual involutions and target action."""
    values={ids:cf};words={ids:()};todo=deque([ids]);edges=0
    while todo:
        tick();current=todo.popleft();cc=values[current]
        for g in range(8):
            image,ic=transform(current,cc,g,actions,phases);edges+=1
            if image in values:assert values[image]==ic,('orbit_coefficient_disagreement',image)
            else:
                assert len(values)<GROUP_ORDER
                values[image]=ic;words[image]=words[current]+(g,);todo.append(image)
    assert GROUP_ORDER%len(values)==0 and edges==8*len(values)
    can=min(values);geometry.verify(can,values[can])
    assert geometry.fingerprint(can,values[can])==geometry.fingerprint(ids,cf)
    tocanonical={key:tuple(reversed(word))+words[can] for key,word in words.items()}
    return can,values,tocanonical

def replay_physical(ids,cf,word,expected_ids,expected_cf,pool,target_codes,generator_specs,tick):
    """Rebuild raw amplitudes, canonical phase and coefficients at each step."""
    lookup={tuple(int(e) for e in pool[r]):r for r in set(ids)|set(expected_ids)}
    # The adapter supplies the complete lookup once as an explicit special field.
    complete_lookup=generator_specs['phase_row_to_id']
    assert all(complete_lookup[k]==v for k,v in lookup.items())
    current=tuple(ids);coeff=tuple(cf)
    for g in word:
        tick();moved=[]
        for ray,c in zip(current,coeff):
            row=pool[ray]
            if g==7:
                raw=tuple(-1 if int(e)<0 else (2*int(t)-int(e))%4 for e,t in zip(row,target_codes))
                c=(c[0],-c[1])
            else:
                spec=generator_specs['unitary'][g]
                raw=tuple(-1 if int(row[p])<0 else (int(row[p])+int(a))%4 for p,a in zip(spec['permutation'],spec['phase']))
            gamma=next(e for e in raw if e>=0)
            canonical=tuple(-1 if e<0 else (e-gamma)%4 for e in raw)
            moved.append((complete_lookup[canonical],rotate(c,gamma)))
        moved.sort(key=lambda x:x[0]);current=tuple(r for r,c in moved);coeff=tuple(c for r,c in moved)
    assert current==expected_ids and coeff==expected_cf
