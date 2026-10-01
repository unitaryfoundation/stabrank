"""Three-pivot kernel: seven images in a 4-dim space mod ELL, case C (no five coplanar).

For first pivot i, second pivots j with jok[j] != 0 (j = least index among the non-i
members, minimal in its Stab(i)-orbit), third pivot k > j = the least-index member whose
image is off the line (q_i, q_j); every other member has index > k, or lies on the line
(q_i, q_j) with index in (j, k).  Modulo span(q_i, q_j, q_k) the other four members are zero
(coplanar with the pivots) or mutually parallel; in case C at most one is zero, so a class
is >= need_par parallel members with index > k, plus a fourth parallel member or a zero.

Counting is direct-indexed on the ratio of two of the three reduced coordinates (cells
0..ELL+1); cells with enough members are verified on the third coordinate.  A different
point can share a cell only by an ELL^-1 coincidence, and that only adds work, never loses
a class.
"""
import numpy as np
from numba import njit, int64

ELL = 65521
TZ = ELL + 2


@njit(cache=True, boundscheck=False)
def reduce_one(PD, inv, i):
    N, D = PD.shape
    qi = PD[i]
    a = 0
    while qi[a] == 0:
        a += 1
    ia = inv[qi[a]]
    Q1 = np.empty((N, D - 1), dtype=np.int64)
    for k in range(N):
        c = (PD[k, a] * ia) % ELL
        col = 0
        for d in range(D):
            if d != a:
                Q1[k, col] = (PD[k, d] - c * qi[d]) % ELL
                col += 1
    return Q1


@njit(cache=True, boundscheck=False)
def kernel3(PD, inv, i, jok, kok, need_par, max_out, kfrac_lo, kfrac_hi):
    """Returns cand rows (j, k, npar, nzero), members (cand id, state, tag 0 par / 1 zero),
    and the number of inner steps.  Buffers: max_out classes and 16 * max_out members; the
    caller must check that the returned shapes are strictly below these bounds.  Third pivots k are restricted to the fraction
    [kfrac_lo, kfrac_hi) of the range (j, N) for work splitting, and to kok[k] != 0
    (k minimal in its Stab(i, j)-orbit; kok is per (i, j) when Stab(i, j) is nontrivial)."""
    N, D = PD.shape
    Q1 = reduce_one(PD, inv, i)
    D1 = D - 1
    D2 = D1 - 1
    Q2 = np.empty((N, D2), dtype=np.int64)
    stamp_t = np.zeros(TZ, dtype=np.int32)
    cnt = np.zeros(TZ, dtype=np.int32)
    head = np.zeros(TZ, dtype=np.int32)
    nxt = np.zeros(N, dtype=np.int32)
    touched = np.zeros(N, dtype=np.int32)
    zij = np.zeros(N, dtype=np.int64)
    out_c = np.empty((max_out, 4), dtype=np.int64)
    out_m = np.empty((max_out * 16, 3), dtype=np.int64)
    nc = 0
    nm = 0
    stamp = 0
    nsteps = 0
    for j in range(N):
        if j == i or jok[j] == 0:
            continue
        qj = Q1[j]
        b = 0
        while b < D1 and qj[b] == 0:
            b += 1
        if b == D1:
            continue
        ib = inv[qj[b]]
        nzij = 0
        for k in range(j + 1, N):
            c = (Q1[k, b] * ib) % ELL
            col = 0
            z = True
            for d in range(D1):
                if d != b:
                    val = (Q1[k, d] - c * qj[d]) % ELL
                    Q2[k, col] = val
                    if val != 0:
                        z = False
                    col += 1
            if z and k != i:
                zij[nzij] = k
                nzij += 1
        klo = j + 1 + int64(kfrac_lo * (N - j - 1))
        khi = j + 1 + int64(kfrac_hi * (N - j - 1))
        for k in range(klo, khi):
            if k == i or kok[k] == 0:
                continue
            qk = Q2[k]
            e = 0
            while e < D2 and qk[e] == 0:
                e += 1
            if e == D2:
                continue
            ie = inv[qk[e]]
            a0 = 1 if e == 0 else 0
            a1 = a0 + 1
            if a1 == e:
                a1 += 1
            a2 = a1 + 1
            if a2 == e:
                a2 += 1
            qk0 = qk[a0]
            qk1 = qk[a1]
            qk2 = qk[a2]
            stamp += 1
            nt = 0
            nsteps += N - k - 1
            for kp in range(k + 1, N):
                if kp == i:
                    continue
                c = (Q2[kp, e] * ie) % ELL
                v0 = (Q2[kp, a0] - c * qk0) % ELL
                v1 = (Q2[kp, a1] - c * qk1) % ELL
                if v0 != 0:
                    r = (v1 * inv[v0]) % ELL
                elif v1 != 0:
                    r = ELL
                else:
                    r = ELL + 1
                if stamp_t[r] != stamp:
                    stamp_t[r] = stamp
                    cnt[r] = 1
                    head[r] = kp
                    nxt[kp] = -1
                    touched[nt] = r
                    nt += 1
                else:
                    cnt[r] += 1
                    nxt[kp] = head[r]
                    head[r] = kp
            # zeros in (j, k) on the line (i, j)
            nz_line = 0
            for u in range(nzij):
                if zij[u] < k:
                    nz_line += 1
            for t in range(nt):
                r = touched[t]
                if cnt[r] < need_par and r != ELL + 1:
                    continue
                # gather members
                tcount = cnt[r]
                mem = np.empty(tcount, dtype=np.int64)
                key = np.empty(tcount, dtype=np.int64)
                u = 0
                q = head[r]
                while q != -1:
                    mem[u] = q
                    u += 1
                    q = nxt[q]
                nzero = 0
                for u in range(tcount):
                    kk = mem[u]
                    c = (Q2[kk, e] * ie) % ELL
                    v0 = (Q2[kk, a0] - c * qk0) % ELL
                    v1 = (Q2[kk, a1] - c * qk1) % ELL
                    v2 = (Q2[kk, a2] - c * qk2) % ELL
                    if v0 != 0:
                        key[u] = (v2 * inv[v0]) % ELL
                    elif v1 != 0:
                        key[u] = (v2 * inv[v1]) % ELL
                    elif v2 != 0:
                        key[u] = ELL
                    else:
                        key[u] = -1
                        nzero += 1
                if r == ELL + 1 and tcount - nzero < need_par:
                    continue
                for u in range(tcount):
                    if key[u] < 0:
                        continue
                    kv = key[u]
                    g = 0
                    for u2 in range(tcount):
                        if key[u2] == kv:
                            g += 1
                    if g < need_par:
                        continue
                    # zeros mod (i,j,k) with index > k live in cell ELL+1
                    nzero_all = nz_line
                    zr = ELL + 1
                    if stamp_t[zr] == stamp:
                        qz = head[zr]
                        while qz != -1:
                            c = (Q2[qz, e] * ie) % ELL
                            if (Q2[qz, a0] - c * qk0) % ELL == 0 and (Q2[qz, a1] - c * qk1) % ELL == 0 \
                                    and (Q2[qz, a2] - c * qk2) % ELL == 0:
                                nzero_all += 1
                            qz = nxt[qz]
                    if g >= need_par + 1 or nzero_all >= 1:
                        if nc < max_out:
                            out_c[nc, 0] = j
                            out_c[nc, 1] = k
                            out_c[nc, 2] = g
                            out_c[nc, 3] = nzero_all
                            for u2 in range(tcount):
                                if key[u2] == kv and nm < out_m.shape[0]:
                                    out_m[nm, 0] = nc
                                    out_m[nm, 1] = mem[u2]
                                    out_m[nm, 2] = 0
                                    nm += 1
                            for u2 in range(nzij):
                                if zij[u2] < k and nm < out_m.shape[0]:
                                    out_m[nm, 0] = nc
                                    out_m[nm, 1] = zij[u2]
                                    out_m[nm, 2] = 1
                                    nm += 1
                            if stamp_t[zr] == stamp:
                                qz = head[zr]
                                while qz != -1:
                                    c = (Q2[qz, e] * ie) % ELL
                                    if (Q2[qz, a0] - c * qk0) % ELL == 0 and (Q2[qz, a1] - c * qk1) % ELL == 0 \
                                            and (Q2[qz, a2] - c * qk2) % ELL == 0 and nm < out_m.shape[0]:
                                        out_m[nm, 0] = nc
                                        out_m[nm, 1] = qz
                                        out_m[nm, 2] = 1
                                        nm += 1
                                    qz = nxt[qz]
                            nc += 1
                    for u2 in range(tcount):
                        if key[u2] == kv:
                            key[u2] = -2
    return out_c[:nc], out_m[:nm], nsteps
