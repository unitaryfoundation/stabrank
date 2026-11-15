"""Portable staging copy of accepted exact routines; see PROVENANCE.json.
Not executed or certified by source staging.
"""
from fractions import Fraction as F
import numpy as np
GZERO=(F(0),F(0)); GONE=(F(1),F(0))
PHASES=((1,0),(0,1),(-1,0),(0,-1))
ZERO=GZERO; ONE=GONE; I4=PHASES


def gadd(a,b): return (a[0]+b[0], a[1]+b[1])

def gneg(a): return (-a[0], -a[1])

def gsub(a,b): return (a[0]-b[0], a[1]-b[1])

def gmul(a,b): return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])

def gdiv(a,b):
    den = b[0]*b[0]+b[1]*b[1]
    return (F(a[0]*b[0]+a[1]*b[1], den), F(a[1]*b[0]-a[0]*b[1], den))

def decode(v): return tuple(GZERO if e == -1 else PHASES[e] for e in v)

def serial_coeff(c): return [[F(x).numerator,F(x).denominator] for x in c]

def read_coeff(c): return (F(*c[0]),F(*c[1]))

def exact_identity(vectors, coeffs, target):
    for x,t in enumerate(target):
        total=GZERO
        for v,c in zip(vectors,coeffs): total=gadd(total,gmul(c,v[x]))
        if total != t: return False
    return True

def exact_solve(vectors, target):
    """RREF over Q(i); return one solution with free variables set to zero.

    This is also correct for duplicate/dependent columns. No assumptions about
    nonzero modular coordinates or fixed coefficients enter this routine.
    """
    k=len(vectors)
    rows=[[tuple(F(a) for a in v[x]) for v in vectors]+[tuple(F(a) for a in target[x])]
          for x in range(len(target))]
    pivots=[];r=0
    for c in range(k):
        pivot=next((j for j in range(r,len(rows)) if rows[j][c] != GZERO),None)
        if pivot is None: continue
        rows[r],rows[pivot]=rows[pivot],rows[r]
        z=rows[r][c]
        rows[r]=[gdiv(a,z) for a in rows[r]]
        for j in range(len(rows)):
            if j==r or rows[j][c]==GZERO: continue
            z=rows[j][c]
            rows[j]=[gsub(a,gmul(z,b)) for a,b in zip(rows[j],rows[r])]
        pivots.append(c);r+=1
    if any(all(z==GZERO for z in row[:k]) and row[k]!=GZERO for row in rows): return None
    sol=[GZERO]*k
    for j,c in enumerate(pivots): sol[c]=rows[j][k]
    assert exact_identity(vectors,sol,target)
    return tuple(sol)

def exact_inverse(a):
    n=len(a)
    q=[list(row)+[GONE if r==c else GZERO for c in range(n)] for r,row in enumerate(a)]
    for c in range(n):
        r=next(j for j in range(c,n) if q[j][c]!=GZERO)
        q[c],q[r]=q[r],q[c]
        z=q[c][c];q[c]=[gdiv(v,z) for v in q[c]]
        for j in range(n):
            if j!=c and q[j][c]!=GZERO:
                z=q[j][c];q[j]=[gsub(v,gmul(z,w)) for v,w in zip(q[j],q[c])]
    return [row[n:] for row in q]
def add(a, b): return (a[0]+b[0], a[1]+b[1])

def neg(a): return (-a[0], -a[1])

def sub(a, b): return add(a, neg(b))

def mul(a, b): return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])

def div(a, b):
    d = b[0]*b[0]+b[1]*b[1]
    return (F(a[0]*b[0]+a[1]*b[1], d), F(a[1]*b[0]-a[0]*b[1], d))

def stabilizer_check(v):
    """Independent affine-support and Z4 quadratic-phase characterization."""
    support = [i for i,z in enumerate(v) if z != ZERO]
    if not support or len(support)&(len(support)-1): return False
    x0 = support[0]
    basis, span = [], {0}
    for x in support:
        q = x ^ x0
        if q not in span:
            basis.append(q)
            span |= {r ^ q for r in tuple(span)}
    if {x ^ x0 for x in support} != span: return False
    phases = []
    for mask in range(1 << len(basis)):
        x = x0
        for j,b in enumerate(basis):
            if mask >> j & 1: x ^= b
        z = div(v[x], v[x0])
        if z not in I4: return False
        phases.append(I4.index(z))
    for j in range(len(basis)):
        for mask in range(len(phases)):
            if mask >> j & 1: phases[mask] = (phases[mask]-phases[mask ^ (1<<j)])%4
    return all((mask.bit_count()<2 or c%2==0) and (mask.bit_count()<3 or c==0)
               for mask,c in enumerate(phases))
