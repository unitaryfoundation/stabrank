"""Shared setup for the batched three-pivot scan: dictionary, descent, projection,
symmetry group, orbit-block relabelling and the pivot masks.

Everything numerical reuses verify_challenge/cert_t3m3_rank7.py. The one
addition is the orbit-block order of docs/notes/t3_rank7_exclusion.md: the
states are relabelled so that the states inside V_m come first (they are never
pivots), then every G-orbit as a contiguous block, blocks in increasing order
of orbit size (ties by the least original index of the orbit). The first pivot
of block a is the block's first label, the second pivot j is minimal in its
Stab(i)-orbit with respect to the new labels, the third pivot k > j is minimal
in its Stab(i, j)-orbit, and every further member has label above k, or lies
in span(V_m, s_i, s_j) with label above j.

The inner-step count of a pivot pair (i, j) is the sum over admissible k of
N - k - 1, which counts the states above k regardless of whether the kernel
skips k as a member of span(V_m, s_i, s_j). It is the quantity that
count_steps.py reports, and partition.json is built from it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
import cert_t3m3_rank7 as C  # noqa: E402

ELL, ELL2, INV = C.ELL, C.ELL2, C.INV
PROJ_DIM = 6
PROJ_SEED = 2024
CACHE_DIR = os.path.join(HERE, "cache")
MAX_MEMBERS = 96          # members per class set the kernel can decide in place


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def sha256_bytes(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------ symmetry ------

def group_elements(T, m, cache=True):
    """The monomial Clifford symmetries of span(T), with conjugation, as
    (perm, sign, Qexp) triples, sorted canonically. Cached under cache/ as a
    compact npz because the enumeration is about 35 s of pure Python at m=3;
    the cache is a convenience only, every batch records a hash of the elements
    it used and the aggregator's re-runs bypass it."""
    path = os.path.join(CACHE_DIR, f"elements_m{m}.npz")
    if cache and os.path.exists(path):
        z = np.load(path)
        elems = [(z["perm"][t], int(z["sign"][t]), z["qexp"][t]) for t in range(z["perm"].shape[0])]
    else:
        elems = C.monomial_symmetries(T, m)
        assert all(C.target_image_ok(T, e) for e in elems)
        elems = sorted(elems, key=lambda e: (e[0].tolist(), e[1], e[2].tolist()))
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(path, perm=np.stack([e[0] for e in elems]).astype(np.int64),
                     sign=np.array([e[1] for e in elems], dtype=np.int64),
                     qexp=np.stack([e[2] for e in elems]).astype(np.int64))
    return elems


def _row_hashes(Ec, mult):
    return (Ec.astype(np.uint64) * mult[None, :]).sum(axis=1)


def dictionary_permutations(E, elems):
    """The permutation of the dictionary induced by every element, exactly:
    rows are matched by a 64-bit key and the match is then checked row by row,
    so a key collision cannot pass silently."""
    canon = C.canon_rows(E)
    rng = np.random.default_rng(0x5EED)
    mult = rng.integers(1, 2 ** 63, size=canon.shape[1], dtype=np.uint64)
    k0 = _row_hashes(canon, mult)
    order = np.argsort(k0, kind="stable")
    ks = k0[order]
    assert len(np.unique(ks)) == len(ks), "dictionary row keys collide; change the key multiplier"
    N = E.shape[0]
    perms = np.empty((len(elems), N), dtype=np.int32)
    for t, e in enumerate(elems):
        img = C.canon_rows(C.apply_element(E, e))
        pos = np.searchsorted(ks, _row_hashes(img, mult))
        assert pos.max() < N
        P = order[pos]
        assert np.array_equal(canon[P], img), "an element does not permute the dictionary"
        assert len(np.unique(P)) == N
        perms[t] = P
    return perms


# ------------------------------------------------------------- setup --------

class Setup:
    """Dictionary, targets, projection and symmetry of |T3>^m in the orbit-block
    labelling. Arrays suffixed `_b` are in the new labels; `old_of[t]` is the
    original index of label t and `newlabel[s]` the label of state s."""

    def __init__(self, m, cache=True):
        self.m = m
        E = C.build_dictionary(m)
        N = E.shape[0]
        T = C.t3_targets(m)
        Vl = C.to_mod(E, C.Z6, ELL)
        Tl = C.to_mod(T, C.Z6, ELL)
        E2 = C.to_mod(E, C.Z6_2, ELL2)
        T2 = C.to_mod(T, C.Z6_2, ELL2)
        PD = C.quotient_projection(Vl, Tl, PROJ_DIM, PROJ_SEED)
        Vq = quotient_images(Vl, Tl)
        isfree = np.all(PD == 0, axis=1).astype(np.int8)
        assert np.array_equal(isfree, np.all(Vq == 0, axis=1).astype(np.int8))
        elems = group_elements(T, m, cache=cache)
        perms = dictionary_permutations(E, elems)
        # orbits and blocks
        gmin = np.arange(N)
        for P in perms:
            gmin = np.minimum(gmin, P)
        reps = np.unique(gmin)
        size = {int(r): int((gmin == r).sum()) for r in reps}
        free_reps = [int(r) for r in reps if isfree[r]]
        block_reps = sorted((int(r) for r in reps if not isfree[r]), key=lambda r: (size[r], r))
        newlabel = np.empty(N, dtype=np.int64)
        pos = 0
        for r in free_reps:
            members = np.flatnonzero(gmin == r)
            newlabel[members] = np.arange(pos, pos + len(members))
            pos += len(members)
        self.nfree = pos
        blocks = []
        for a, r in enumerate(block_reps):
            members = np.flatnonzero(gmin == r)
            newlabel[members] = np.arange(pos, pos + len(members))
            blocks.append({"block": a, "rep": r, "orbit_size": len(members), "start": pos,
                           "end": pos + len(members)})
            pos += len(members)
        assert pos == N
        old_of = np.argsort(newlabel)
        assert np.array_equal(newlabel[old_of], np.arange(N))
        self.E, self.T, self.N = E, T, N
        self.E2, self.T2, self.PD, self.Vq, self.isfree = E2, T2, PD, Vq, isfree
        self.elems, self.perms, self.gmin = elems, perms, gmin
        self.blocks, self.newlabel, self.old_of = blocks, newlabel, old_of
        self.PD_b = np.ascontiguousarray(PD[old_of])
        self.E2_b = np.ascontiguousarray(E2[old_of])
        self.Vq_b = np.ascontiguousarray(Vq[old_of])
        self.isfree_b = np.ascontiguousarray(isfree[old_of])
        # permutations in the new labels
        self.perms_b = np.ascontiguousarray(newlabel[perms[:, old_of]].astype(np.int32))
        elem_bytes = np.concatenate([np.concatenate([e[0], [e[1]], e[2]]) for e in elems])
        self.dictionary_sha256 = sha256_bytes(E)
        self.setup_sha256 = sha256_bytes(E, PD, Vq, newlabel, elem_bytes.astype(np.int64))
        self.group_order = len(elems)

    def describe(self):
        return {"m": self.m, "N": self.N, "group_order": self.group_order,
                "orbits": len(self.blocks) + (1 if self.nfree else 0), "nfree": self.nfree,
                "projection_dim": PROJ_DIM, "projection_seed": PROJ_SEED, "ell": ELL,
                "ell2": ELL2, "dictionary_sha256": self.dictionary_sha256,
                "setup_sha256": self.setup_sha256,
                "blocks": [dict(b) for b in self.blocks]}

    # -- pivot masks in the new labels --
    def stabilizer(self, block):
        """Permutations (new labels) fixing the first pivot of the block."""
        i = self.blocks[block]["start"]
        S = self.perms_b[self.perms_b[:, i] == i]
        assert S.shape[0] * self.blocks[block]["orbit_size"] == self.group_order
        return S

    def second_pivots(self, block, S=None):
        """Admissible second pivots j of the block: labels above the first
        pivot, minimal in their Stab(i)-orbit, not inside V_m."""
        S = self.stabilizer(block) if S is None else S
        i = self.blocks[block]["start"]
        lab = np.arange(self.N)
        imin = S.min(axis=0)
        return np.flatnonzero((imin == lab) & (lab > i) & (self.isfree_b == 0))

    def third_pivot_mask(self, j, S):
        """kok mask for the pair (i, j): None when Stab(i, j) is trivial (every
        k > j admissible), else an int8 array marking the Stab(i, j)-minimal
        labels above j."""
        Sj = S[S[:, j] == j]
        if Sj.shape[0] <= 1:
            return None
        lab = np.arange(self.N)
        kmin = Sj.min(axis=0)
        return ((kmin == lab) & (lab > j)).astype(np.int8)

    def pair_steps(self, j, kok):
        N = self.N
        if kok is None:
            r = N - int(j) - 1
            return r * (r - 1) // 2
        ks = np.flatnonzero(kok)
        return int((N - ks - 1).sum())


def quotient_images(Vl, Tl):
    """Images of the rows of Vl in F_ell^dim / span(Tl), unprojected (the
    pivot columns of Tl's RREF dropped). Mirrors quotient_projection."""
    R_, dim = Tl.shape
    piv = []
    for r in range(R_):
        c = next(c for c in range(dim) if Tl[r, c] != 0
                 and all(Tl[s, c] == 0 for s in range(R_) if s != r))
        piv.append(c)
    Vq = Vl.copy() % ELL
    for r in range(R_):
        c = (Vq[:, piv[r]] * INV[Tl[r, piv[r]]]) % ELL
        Vq = (Vq - np.outer(c, Tl[r])) % ELL
    assert np.all(Vq[:, piv] == 0)
    keep = [c for c in range(dim) if c not in piv]
    return np.ascontiguousarray(Vq[:, keep])


# ----------------------------------------------------------- partition ------

def load_partition(path):
    with open(path) as f:
        part = json.load(f)
    body = {k: v for k, v in part.items() if k != "sha256"}
    if sha256_json(body) != part.get("sha256"):
        raise ValueError(f"{path}: sha256 does not match its content")
    return part


def batch_geometry(part, index):
    if not 0 <= index < len(part["batches"]):
        raise IndexError(f"batch index {index} outside 0..{len(part['batches']) - 1}")
    b = part["batches"][index]
    assert b["index"] == index
    return b
