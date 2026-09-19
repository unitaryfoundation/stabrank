"""One batch of the three-pivot scan that excludes rank 7 for |T3>^3.

Accepts the command line autoresearch/loop.py builds for run.py, with the seed
as the batch index:

    batch.py T3 3 7 --seeds 1 --seed0 K [annealer flags, ignored] \
        --partition research/t3_rank7/partition.json [--out-dir DIR] [--no-cache]

The batch is the (block, j-range) slice K of the partition. The scan runs in
the orbit-block pivot order with the Stab(i, j) mask on the third pivot (see
common.py and docs/notes/t3_rank7_exclusion.md, section 2), and every
candidate class set is decided inside the compiled kernel: rank of the class
mod 2^31 - 1, exact for values up to 8 by the Hadamard bound, then whether the
three descent vectors psi_r reduce to zero against the class's row space. A
class of rank at least 8 cannot lie in the 7-dimensional preimage of a
4-dimensional image space, so it is a projection collision; it is re-split in
Python against the unprojected quotient images, and a subclass that still has
rank at least 8 is decided by enumerating its 7-subsets. Only counts, the
histogram and the exceptions are stored, never the class lists.

Output: <out-dir>/batch_<K>.json. Exit 0 when the batch completes with no
class set containing V_3 (loop.py logs that as "discovery"), exit 2 when one
does (loop.py logs "miss"; the last stdout line starts with DECOMPOSITION
FOUND), and an exception (exit 1 with a traceback) when the batch could not
decide every class, when its geometry or step count disagrees with the
partition, or when the setup hash differs from the partition's.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
from numba import njit, int64, uint64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import C, ELL, ELL2, INV  # noqa: E402

HIST = 64          # histogram axes: zero members, parallel members (clamped)
MAX_FOUND = 256    # stored class sets containing V; the rest are counted
MAX_SPUR = 1024    # stored spurious class sets; more than this fails the batch
MAX_OVER = 64
N_COUNTERS = 10
(C_STEPS_NOMINAL, C_STEPS_DONE, C_CAND, C_DECIDED, C_FOUND, C_SPUR, C_OVER, C_NPAR, C_NZ,
 C_PAIRS) = range(N_COUNTERS)


# ------------------------------------------------------------- kernel -------

@njit(cache=True)
def _inv_mod(a, p):
    """Inverse of a mod a prime p by the extended Euclidean algorithm."""
    t, newt = int64(0), int64(1)
    r, newr = int64(p), int64(a % p)
    while newr != 0:
        q = r // newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    if t < 0:
        t += p
    return t


@njit(cache=True)
def _rref_inplace(M, n, dim, p, pivcol):
    """Reduced row echelon form mod p of the first n rows of M, in place.
    Returns the rank; pivcol[r] is the pivot column of row r."""
    r = 0
    for c in range(dim):
        pr = -1
        for t in range(r, n):
            if M[t, c] != 0:
                pr = t
                break
        if pr < 0:
            continue
        if pr != r:
            for d in range(dim):
                tmp = M[r, d]
                M[r, d] = M[pr, d]
                M[pr, d] = tmp
        f = _inv_mod(M[r, c], p)
        for d in range(dim):
            M[r, d] = (M[r, d] * f) % p
        for t in range(n):
            if t != r and M[t, c] != 0:
                g = M[t, c]
                for d in range(dim):
                    M[t, d] = (M[t, d] - g * M[r, d]) % p
        pivcol[r] = c
        r += 1
        if r == n:
            break
    return r


@njit(cache=True)
def _in_rowspace(M, r, pivcol, T2, dim, p, buf):
    """Whether every row of T2 reduces to zero against the RREF rows 0..r-1 of M."""
    for s in range(T2.shape[0]):
        for d in range(dim):
            buf[d] = T2[s, d] % p
        for t in range(r):
            g = buf[pivcol[t]]
            if g != 0:
                for d in range(dim):
                    buf[d] = (buf[d] - g * M[t, d]) % p
        for d in range(dim):
            if buf[d] != 0:
                return False
    return True


@njit(cache=True)
def _decide(members, n, E2, T2, ell2, work, pivcol, tbuf):
    """Rank of the class mod ell2 (exact if <= 8) and, when it is at most 7,
    whether span(T2) lies in the class's span. Returns (rank, found)."""
    dim = E2.shape[1]
    for t in range(n):
        for d in range(dim):
            work[t, d] = E2[members[t], d]
    r = _rref_inplace(work, n, dim, ell2, pivcol)
    if r >= 8:
        return r, False
    return r, _in_rowspace(work, r, pivcol, T2, dim, ell2, tbuf)


@njit(cache=True)
def scan_pairs(PD, inv, ell, E2, T2, ell2, i, jlist, kok_index, kok_rows, isfree, need,
               hist, found_buf, found_len, found_meta, spur_buf, spur_len, spur_meta,
               over_meta, counters):
    """The three-pivot scan for first pivot i over the second pivots in jlist.

    kok_index[t] < 0 means every k > jlist[t] is an admissible third pivot;
    otherwise kok_rows[kok_index[t]] marks the admissible k. States with
    isfree != 0 lie inside V and are members of every class; states whose
    image is parallel to q_i lie in span(V, s_i) and are likewise members
    for this first pivot (there are none at m = 3). A class set's members are
    the three pivots, the states above j inside span(V, s_i, s_j), the states
    above k whose image vanishes modulo span(q_i, q_j, q_k), and one group of
    states above k with mutually parallel images there.

    The inner loops are those of research/t3_rank7/scan3.py::kernel3 with the
    class lists replaced by the in-place decision of _decide.
    """
    N, D = PD.shape
    D1 = D - 1
    maxm = found_buf.shape[1]
    keys = np.zeros(N, dtype=np.uint64)
    tkey = np.zeros(C.TSIZE, dtype=np.uint64)
    tcnt = np.zeros(C.TSIZE, dtype=np.int64)
    tstamp = np.zeros(C.TSIZE, dtype=np.int64)
    slot_of = np.zeros(N, dtype=np.int64)
    stamp = 0
    Q1 = C._reduce_one(PD, inv, i, ell)
    skip1 = isfree.copy()
    fixed = np.empty(N, dtype=np.int64)
    nfixed = 0
    npar = 0
    for s in range(N):
        if isfree[s] != 0:
            fixed[nfixed] = s
            nfixed += 1
        elif s != i:
            z = True
            for d in range(D1):
                if Q1[s, d] != 0:
                    z = False
                    break
            if z:
                skip1[s] = 1
                fixed[nfixed] = s
                nfixed += 1
                npar += 1
    skip1[i] = 1
    counters[C_NPAR] = npar
    Q2 = np.empty((N, D1 - 1), dtype=np.int64)
    Q3 = np.empty((N, D1 - 2), dtype=np.int64)
    skip2 = np.empty(N, dtype=np.int8)
    zbuf = np.empty(N, dtype=np.int64)
    members = np.empty(maxm, dtype=np.int64)
    work = np.empty((maxm, E2.shape[1]), dtype=np.int64)
    pivcol = np.empty(maxm, dtype=np.int64)
    tbuf = np.empty(E2.shape[1], dtype=np.int64)
    for t in range(jlist.shape[0]):
        j = jlist[t]
        ki = kok_index[t]
        # nominal steps of the pair, as count_steps.py and the partition count them
        for k in range(j + 1, N):
            if ki < 0 or kok_rows[ki, k] != 0:
                counters[C_STEPS_NOMINAL] += N - k - 1
        counters[C_PAIRS] += 1
        if skip1[j] != 0:
            continue
        qj = Q1[j]
        b = -1
        for d in range(D1):
            if qj[d] != 0:
                b = d
                break
        ib = inv[qj[b]]
        for l in range(N):
            skip2[l] = skip1[l]
        nz = 0
        for l in range(j + 1, N):
            if skip1[l] != 0:
                continue
            c = (Q1[l, b] * ib) % ell
            col = 0
            zero = True
            for d in range(D1):
                if d == b:
                    continue
                v = (Q1[l, d] - c * qj[d]) % ell
                Q2[l, col] = v
                col += 1
                if v != 0:
                    zero = False
            if zero:
                zbuf[nz] = l
                nz += 1
                skip2[l] = 1
        counters[C_NZ] += nz
        nfree_j = nfixed + nz
        for k in range(j + 1, N):
            if ki >= 0 and kok_rows[ki, k] == 0:
                continue
            if skip2[k] != 0:
                continue
            counters[C_STEPS_DONE] += N - k - 1
            qk = Q2[k]
            b2 = -1
            for d in range(D1 - 1):
                if qk[d] != 0:
                    b2 = d
                    break
            ib2 = inv[qk[b2]]
            for l in range(k + 1, N):
                c = (Q2[l, b2] * ib2) % ell
                col = 0
                for d in range(D1 - 1):
                    if d == b2:
                        continue
                    Q3[l, col] = (Q2[l, d] - c * qk[d]) % ell
                    col += 1
            stamp += 1
            nzero = C._group_keys(Q3, inv, ell, k + 1, -1, skip2, keys, tkey, tcnt, tstamp,
                                  stamp, slot_of)
            nzero_total = nzero + nfree_j
            # classes: the all-zero class (ngroup 0) and one per parallel group
            for pass_ in range(2):
                if pass_ == 0:
                    if nzero_total < need:
                        continue
                    lo, hi = k, k + 1        # dummy range: no group scan
                else:
                    lo, hi = k + 1, N
                for l in range(lo, hi):
                    if pass_ == 1:
                        s = slot_of[l]
                        if s < 0:
                            continue
                        if not (tcnt[s] > 0 and tcnt[s] + nzero_total >= need):
                            continue
                        ngroup = tcnt[s]
                        tcnt[s] = -1
                        h = keys[l]
                    else:
                        ngroup = 0
                        h = uint64(0)
                    counters[C_CAND] += 1
                    a0 = nzero_total if nzero_total < HIST else HIST - 1
                    a1 = ngroup if ngroup < HIST else HIST - 1
                    hist[a0, a1] += 1
                    n = 0
                    over = False
                    members[0] = i
                    members[1] = j
                    members[2] = k
                    n = 3
                    for ll in range(k + 1, N):
                        sl = slot_of[ll]
                        if sl == -2 or (ngroup > 0 and sl >= 0 and keys[ll] == h):
                            if n < maxm:
                                members[n] = ll
                            else:
                                over = True
                            n += 1
                    for u in range(nz):
                        if n < maxm:
                            members[n] = zbuf[u]
                        else:
                            over = True
                        n += 1
                    for u in range(nfixed):
                        if n < maxm:
                            members[n] = fixed[u]
                        else:
                            over = True
                        n += 1
                    if over:
                        o = counters[C_OVER]
                        if o < over_meta.shape[0]:
                            over_meta[o, 0] = j
                            over_meta[o, 1] = k
                            over_meta[o, 2] = n
                        counters[C_OVER] += 1
                        continue
                    r, found = _decide(members, n, E2, T2, ell2, work, pivcol, tbuf)
                    if r >= 8:
                        o = counters[C_SPUR]
                        if o < spur_buf.shape[0]:
                            for u in range(n):
                                spur_buf[o, u] = members[u]
                            spur_len[o] = n
                            spur_meta[o, 0] = j
                            spur_meta[o, 1] = k
                            spur_meta[o, 2] = r
                        counters[C_SPUR] += 1
                        continue
                    counters[C_DECIDED] += 1
                    if found:
                        o = counters[C_FOUND]
                        if o < found_buf.shape[0]:
                            for u in range(n):
                                found_buf[o, u] = members[u]
                            found_len[o] = n
                            found_meta[o, 0] = j
                            found_meta[o, 1] = k
                            found_meta[o, 2] = r
                        counters[C_FOUND] += 1
    return 0


# -------------------------------------------------------- exact re-split ----

def _canonical_directions(Q):
    """Rows of Q (mod ELL) grouped by projective direction: (zero row indices,
    {direction key: [row indices]})."""
    zeros, groups = [], {}
    for t in range(Q.shape[0]):
        row = Q[t] % ELL
        nzc = np.flatnonzero(row)
        if len(nzc) == 0:
            zeros.append(t)
            continue
        can = (row * INV[row[nzc[0]]]) % ELL
        groups.setdefault(can.tobytes(), []).append(t)
    return zeros, groups


def _reduce_rows(Q, pivots):
    """Q reduced modulo the span of the rows `pivots` (mod ELL), by elimination."""
    P = pivots.copy() % ELL
    Q = Q.copy() % ELL
    for r in range(P.shape[0]):
        c = int(np.flatnonzero(P[r])[0])
        f = INV[P[r, c]]
        P[r] = (P[r] * f) % ELL
        for s in range(P.shape[0]):
            if s != r and P[s, c]:
                P[s] = (P[s] - P[s, c] * P[r]) % ELL
        Q = (Q - np.outer(Q[:, c], P[r])) % ELL
    return Q


def _exact_pair(E2, T2, rows):
    S = E2[rows]
    rS = C.rank_mod(S, ELL2)
    rST = C.rank_mod(np.vstack([S, T2]), ELL2)
    return rS, rST


def resplit(S, need, members_b, j, k):
    """Exact resolution of a class set of rank at least 8 mod ELL2.

    The members (new labels; the first three are the pivots) are regrouped by
    their unprojected quotient images modulo span(q_i, q_j, q_k): each
    subclass (pivots, zero images, one direction) is decided by the rank pair
    if its rank is at most 7, else by its 7-subsets. Returns a record with
    `found` (subclasses whose span contains V, original indices) and
    `undecided` (subclasses too large to enumerate).
    """
    piv = list(members_b[:3])
    rest = [int(x) for x in members_b[3:]]
    Q = _reduce_rows(S.Vq_b[rest], S.Vq_b[piv])
    zeros, groups = _canonical_directions(Q)
    zero_members = [rest[t] for t in zeros]
    subclasses = [piv + zero_members] if len(zero_members) >= need else []
    for g in groups.values():
        if len(g) + len(zero_members) >= need:
            subclasses.append(piv + zero_members + [rest[t] for t in g])
    rec = {"j": int(S.old_of[j]), "k": int(S.old_of[k]),
           "members": sorted(int(S.old_of[x]) for x in members_b), "subclasses": [],
           "found": [], "undecided": []}
    for sub in subclasses:
        orig = sorted(int(S.old_of[x]) for x in sub)
        rS, rST = _exact_pair(S.E2, S.T2, orig)
        entry = {"members": orig, "rank": int(rS), "rank_with_V": int(rST)}
        if rS <= 7:
            entry["decision"] = "rank pair"
            if rST == rS:
                rec["found"].append(entry)
        elif len(orig) <= 24:
            entry["decision"] = "7-subsets"
            hit = None
            for sub7 in itertools.combinations(orig, 7):
                a, b = _exact_pair(S.E2, S.T2, list(sub7))
                if a == b:
                    hit = list(sub7)
                    break
            if hit is not None:
                entry["members"] = hit
                entry["rank"] = 7
                rec["found"].append(entry)
        else:
            entry["decision"] = "undecided"
            rec["undecided"].append(entry)
        rec["subclasses"].append(entry)
    return rec


def minimal_terms(S, members, cap=13):
    """Smallest number of the members whose span contains V (exact, by
    enumerating subsets of increasing size); None if the members number more
    than cap."""
    if len(members) > cap:
        return None
    for r in range(1, len(members) + 1):
        for sub in itertools.combinations(members, r):
            a, b = _exact_pair(S.E2, S.T2, list(sub))
            if a == b:
                return r
    return None


# ------------------------------------------------------------ driver --------

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=common.ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_batch(part, index, S, verbose=True):
    """Run batch `index` of the partition on the setup S. Returns the result
    record (deterministic part only)."""
    geo = common.batch_geometry(part, index)
    assert part["setup"]["setup_sha256"] == S.setup_sha256, "setup differs from the partition's"
    assert part["m"] == S.m
    need = part["need"]
    block = geo["block"]
    i = S.blocks[block]["start"]
    stab = S.stabilizer(block)
    js_all = S.second_pivots(block, stab)
    js = js_all[(js_all >= geo["j_lo"]) & (js_all < geo["j_hi"])]
    assert len(js) == geo["pairs"], (len(js), geo["pairs"])
    kok_rows, kok_index, expected = [], np.full(len(js), -1, dtype=np.int64), 0
    for t, j in enumerate(js):
        kok = S.third_pivot_mask(int(j), stab)
        expected += S.pair_steps(int(j), kok)
        if kok is not None:
            kok_index[t] = len(kok_rows)
            kok_rows.append(kok)
    assert expected == geo["steps"], (expected, geo["steps"])
    kok_rows = (np.stack(kok_rows) if kok_rows else np.zeros((1, S.N), dtype=np.int8))
    hist = np.zeros((HIST, HIST), dtype=np.int64)
    found_buf = np.zeros((MAX_FOUND, common.MAX_MEMBERS), dtype=np.int64)
    found_len = np.zeros(MAX_FOUND, dtype=np.int64)
    found_meta = np.zeros((MAX_FOUND, 3), dtype=np.int64)
    spur_buf = np.zeros((MAX_SPUR, common.MAX_MEMBERS), dtype=np.int64)
    spur_len = np.zeros(MAX_SPUR, dtype=np.int64)
    spur_meta = np.zeros((MAX_SPUR, 3), dtype=np.int64)
    over_meta = np.zeros((MAX_OVER, 3), dtype=np.int64)
    counters = np.zeros(N_COUNTERS, dtype=np.int64)
    t0 = time.time()
    c0 = time.process_time()
    scan_pairs(S.PD_b, INV, ELL, S.E2_b, S.T2, ELL2, int(i), js.astype(np.int64), kok_index,
               kok_rows, S.isfree_b, int(need), hist, found_buf, found_len, found_meta,
               spur_buf, spur_len, spur_meta, over_meta, counters)
    kernel_s = time.time() - t0
    kernel_cpu = time.process_time() - c0
    assert counters[C_STEPS_NOMINAL] == geo["steps"], (int(counters[C_STEPS_NOMINAL]), geo["steps"])
    assert counters[C_PAIRS] == geo["pairs"]
    found = []
    for o in range(min(int(counters[C_FOUND]), MAX_FOUND)):
        mem = found_buf[o, :found_len[o]]
        orig = sorted(int(S.old_of[x]) for x in mem)
        found.append({"members": orig, "rank": int(found_meta[o, 2]),
                      "j": int(S.old_of[found_meta[o, 0]]), "k": int(S.old_of[found_meta[o, 1]]),
                      "min_terms": minimal_terms(S, orig)})
    spurious = []
    undecided = []
    for o in range(min(int(counters[C_SPUR]), MAX_SPUR)):
        rec = resplit(S, need, spur_buf[o, :spur_len[o]], int(spur_meta[o, 0]),
                      int(spur_meta[o, 1]))
        rec["rank_mod_ell2"] = int(spur_meta[o, 2])
        spurious.append(rec)
        for e in rec["found"]:
            found.append({"members": e["members"], "rank": e["rank"], "j": rec["j"], "k": rec["k"],
                          "min_terms": minimal_terms(S, e["members"]), "from_resplit": True})
        undecided.extend(rec["undecided"])
    if counters[C_SPUR] > MAX_SPUR:
        undecided.append({"reason": f"{int(counters[C_SPUR])} spurious classes exceed the "
                                    f"buffer of {MAX_SPUR}"})
    for o in range(min(int(counters[C_OVER]), MAX_OVER)):
        undecided.append({"reason": "class set larger than the member buffer",
                          "j": int(S.old_of[over_meta[o, 0]]), "k": int(S.old_of[over_meta[o, 1]]),
                          "members": int(over_meta[o, 2])})
    if counters[C_OVER] > MAX_OVER:
        undecided.append({"reason": f"{int(counters[C_OVER])} oversize classes"})
    histogram = {f"{a},{b}": int(hist[a, b]) for a in range(HIST) for b in range(HIST)
                 if hist[a, b]}
    rec = {
        "batch": {"index": index, "block": block, "j_lo": geo["j_lo"], "j_hi": geo["j_hi"],
                  "pairs": geo["pairs"], "steps": geo["steps"], "first_pivot": int(S.old_of[i]),
                  "orbit_size": S.blocks[block]["orbit_size"]},
        "m": S.m, "rank": part["rank"], "need": need,
        "partition_sha256": part["sha256"], "setup_sha256": S.setup_sha256,
        "dictionary_sha256": S.dictionary_sha256,
        "steps_nominal": int(counters[C_STEPS_NOMINAL]), "steps_done": int(counters[C_STEPS_DONE]),
        "pairs_scanned": int(counters[C_PAIRS]), "pairs_with_nontrivial_stab_ij": int(len(kok_rows) if kok_index.max() >= 0 else 0),
        "z_members_total": int(counters[C_NZ]), "parallel_to_pivot": int(counters[C_NPAR]),
        "candidates": int(counters[C_CAND]), "decided": int(counters[C_DECIDED]),
        "histogram": histogram,
        "found_count": int(counters[C_FOUND]) + sum(len(r["found"]) for r in spurious),
        "found": found,
        "spurious_count": int(counters[C_SPUR]), "spurious": spurious,
        "oversize_count": int(counters[C_OVER]),
        "undecided": undecided,
    }
    rec["deterministic_sha256"] = common.sha256_json(rec)
    timing = {"kernel_s": kernel_s, "kernel_cpu_s": kernel_cpu,
              "ns_per_step_done": 1e9 * kernel_s / max(1, rec["steps_done"])}
    if verbose:
        print(f"batch {index}: block {block} j in [{geo['j_lo']}, {geo['j_hi']}), {geo['pairs']} "
              f"pairs, {rec['steps_nominal']:.4e} steps ({rec['steps_done']:.4e} done) in "
              f"{kernel_s:.1f}s ({timing['ns_per_step_done']:.1f} ns/step); {rec['candidates']} "
              f"candidates, {rec['decided']} decided, {rec['found_count']} contain V, "
              f"{rec['spurious_count']} spurious, {len(undecided)} undecided", flush=True)
    return rec, timing


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit")
    ap.add_argument("m", type=int)
    ap.add_argument("rank", type=int)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, required=True, help="the batch index")
    for flag in ("--chains", "--iters", "--cooling", "--llm", "--warm-from", "--save-plateaus"):
        ap.add_argument(flag, default=None, help="accepted for run.py compatibility, ignored")
    ap.add_argument("--partition", default=os.path.join(common.HERE, "partition.json"))
    ap.add_argument("--out-dir", default=os.path.join(common.HERE, "batches"))
    ap.add_argument("--no-cache", action="store_true", help="rebuild the symmetry group")
    a = ap.parse_args(argv[1:])
    assert a.orbit == "T3", "this runner is specific to |T3>^m"
    assert a.seeds == 1, "one batch per invocation"
    started = datetime.datetime.now(datetime.timezone.utc)
    t_all = time.time()
    part = common.load_partition(a.partition)
    assert part["m"] == a.m and part["rank"] == a.rank, "cell differs from the partition's"
    index = a.seed0
    geo = common.batch_geometry(part, index)
    print(f"batch {index} of {part['n_batches']}: block {geo['block']}, j in [{geo['j_lo']}, "
          f"{geo['j_hi']}), {geo['steps']:.4e} steps", flush=True)
    t0 = time.time()
    S = common.Setup(a.m, cache=not a.no_cache)
    setup_s = time.time() - t0
    print(f"setup {setup_s:.1f}s, setup hash {S.setup_sha256[:16]}", flush=True)
    rec, timing = run_batch(part, index, S)
    ended = datetime.datetime.now(datetime.timezone.utc)
    ru = os.times()
    rec.update({
        "wall_s": time.time() - t_all, "cpu_s": ru.user + ru.system, "setup_s": setup_s,
        **timing, "started": started.isoformat(timespec="seconds"),
        "ended": ended.isoformat(timespec="seconds"), "hostname": socket.gethostname(),
        "git_commit": git_commit(), "kernel_version": f"{platform.system()} {platform.release()} "
        f"{platform.version()}", "machine": platform.machine(), "python": platform.python_version(),
        "numba": __import__("numba").__version__, "numpy": np.__version__,
        "symmetry_cache": not a.no_cache,
    })
    rec["sha256"] = common.sha256_json(rec)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"batch_{index}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    if rec["undecided"]:
        raise RuntimeError(f"batch {index}: {len(rec['undecided'])} class set(s) undecided; "
                           f"see {path}")
    if rec["found_count"]:
        best = min(rec["found"], key=lambda f: (f["min_terms"] or 99, f["rank"]))
        terms = (f"{best['min_terms']} states" if best["min_terms"] else
                 "not computed (class larger than the subset cap)")
        print(f"DECOMPOSITION FOUND batch {index}: {rec['found_count']} class set(s) contain V_{a.m}; "
              f"smallest spanning subset {terms}, class rank {best['rank']}, "
              f"members {best['members']}", flush=True)
        return 2
    print(f"batch {index} complete: no class set contains V_{a.m}; written "
          f"{os.path.relpath(path, common.ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
