"""Symmetry of the board's own witnesses, term set by term set.

For a decomposition |M>^m = sum_i c_i |s_i> the relevant group is the
symmetry group of the target,

    G = U^m semidirect S_m,

where U is the single-qudit Clifford stabilizer of |M> (order 2 for
qubit_H, 3 for qubit_T and T3, 4 for H3, 5 for T5, 6 for N, 24 for S) and
S_m permutes the copies. For the cat family |cat_m> is not a tensor power,
so U is taken diagonally: the single-qubit Cliffords u with u^(x m)|cat_m>
proportional to |cat_m>. The antiunitary symmetries are deliberately left
out, so every order below is a unitary one.

The script reports, for each stored term set T = {s_1, ..., s_R}:

  pure      the copy-permutation set-stabilizer {pi : P_pi T = T}, with its
            cycle types and its orbits on the copies;
  local     the local kernel K = {u in U^m : (x u_j) T = T};
  twisted   the image in S_m of the full set-stabilizer
            H = {(u, pi) : (x u_j) P_pi T = T}, that is, the permutations
            that fix T once a local Clifford twist is allowed.

|H| = |K| x |twisted image| is asserted on every run.

The search is a meet in the middle on one term: every element of H carries
s_1 to some s_j, so the projective digests of (x u) s_j over u in U^m and
j <= R are tabulated once, and each pi costs one lookup of P_pi s_1. Every
hit is then verified on the whole term set.

Sources: `bounds/*-upper-*.json` with witness terms, the stored minimal
lists under `data/`, and `research/qutrit_m4_rank5/N_m4_rank7_witness.json`
(the Lean rank-7 witness for |N>^4, whose bound file carries no witness
block).

Usage:
    witness_symmetry.py [--only ORBIT] [--m M] [--minimal] [--bounds-only]
                        [--index i,j,k] [--max-local N] [--max-perms N]
                        [--control]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, alpha, term_vector  # noqa: E402

from stabrank_verify import FAMILY, ORBIT_P, target_vector  # noqa: E402
from rank_exclusion import clifford_group  # noqa: E402

DEFAULT_MAX_LOCAL = 20_000
DEFAULT_MAX_PERMS = 50_000
MAX_BATCH_VECTORS = 5_000_000


class Capped(Exception):
    """Raised when an enumeration would exceed its cap, so that the row is
    reported as not computed rather than as a wrong subgroup."""


# ------------------------------------------------------------- targets ----

def target_numeric(orbit, m):
    if orbit in FAMILY:
        v = np.array([complex(x) for x in target_vector(orbit, m)]).ravel()
        return v / np.linalg.norm(v)
    a = alpha(orbit)
    a = a / np.linalg.norm(a)
    v = a
    for _ in range(m - 1):
        v = np.kron(v, a)
    return v


_LOCAL_CACHE = {}


def local_factors(orbit, m):
    """The single-qudit unitaries whose m-fold products fix the target, and
    whether only the diagonal of them does."""
    if (orbit, m) in _LOCAL_CACHE:
        return _LOCAL_CACHE[(orbit, m)]
    p = ORBIT_P[orbit]
    G = clifford_group(p)
    if orbit in FAMILY:
        psi = target_numeric(orbit, m)
        keep = []
        for U in G:
            g = U
            for _ in range(m - 1):
                g = np.kron(g, U)
            if abs(abs(np.vdot(psi, g @ psi)) - 1) < 1e-9:
                keep.append(U)
        out = (keep, True)
    else:
        a = alpha(orbit)
        a = a / np.linalg.norm(a)
        out = ([U for U in G if abs(abs(np.vdot(a, U @ a)) - 1) < 1e-9], False)
    _LOCAL_CACHE[(orbit, m)] = out
    return out


def _proportional(A, B):
    k = np.unravel_index(int(np.argmax(np.abs(B))), B.shape)
    if abs(B[k]) < 1e-12 or abs(A[k]) < 1e-12:
        return False
    return np.linalg.norm(A / A[k] - B / B[k]) < 1e-9


def inverse_table(factors, p):
    """inv[k] = l with factors[l] factors[k] proportional to the identity."""
    I = np.eye(p, dtype=complex)
    inv = {}
    for k, U in enumerate(factors):
        for l, V in enumerate(factors):
            if _proportional(V @ U, I):
                inv[k] = l
                break
        else:
            raise AssertionError("the local set is not closed under inverses")
    return inv


def identity_index(factors, p):
    I = np.eye(p, dtype=complex)
    out = [k for k, U in enumerate(factors) if _proportional(U, I)]
    if len(out) != 1:
        raise AssertionError("the local set does not contain a unique identity")
    return out[0]


# --------------------------------------------------------------- keys ----

def digest(v):
    """Projective digest: phase fixed by an entry of maximal modulus, then
    rounded and hashed."""
    A = np.abs(v)
    i = int(np.argmax(A > 0.5 * A.max()))
    w = np.round(v / v[i], 6) + 0.0  # the +0.0 folds -0.0 onto 0.0
    return hashlib.blake2b(w.astype(np.complex128).tobytes(), digest_size=16).digest()


def apply_local(v, us, p, m):
    """(u_1 (x) ... (x) u_m) v, by m mode products."""
    T = v.reshape((p,) * m)
    for j in range(m):
        T = np.moveaxis(np.tensordot(us[j], np.moveaxis(T, j, 0), axes=(1, 0)), 0, j)
    return T.ravel()


def local_images(terms, factors, codes, p, m, product):
    """Yield (code, [(x u_j) t for t in terms]) for every code in `codes`.

    With `product` false the codes are the diagonal ones and there are few of
    them. With `product` true they are all of U^m, and the images are built
    copy by copy so that prefixes are shared: one mode product per copy per
    prefix rather than m per code. The batch is sized to keep the working
    array small.
    """
    R = len(terms)
    if not product:
        for code in codes:
            us = [factors[c] for c in code]
            yield code, [apply_local(t, us, p, m) for t in terms]
        return
    n = len(factors)
    tail = 0
    while tail < m and R * n ** (tail + 1) * p ** m <= MAX_BATCH_VECTORS:
        tail += 1
    head = m - tail
    stack = np.stack([t.reshape((p,) * m) for t in terms])
    F = np.stack(factors)
    for prefix in itertools.product(range(n), repeat=head):
        cur = stack
        for j, c in enumerate(prefix):
            cur = np.moveaxis(np.tensordot(factors[c], np.moveaxis(cur, 1 + j, 0),
                                           axes=(1, 0)), 0, 1 + j)
        suffixes = [()]
        for j in range(head, m):
            Y = np.tensordot(F, np.moveaxis(cur, 1 + j, 0), axes=(2, 0))
            cur = np.moveaxis(Y, 1, 2 + j).reshape(-1, *(p,) * m)
            suffixes = [s + (k,) for k in range(n) for s in suffixes]
        flat = cur.reshape(-1, p ** m)
        for i, suf in enumerate(suffixes):
            yield prefix + suf, list(flat[i * R:(i + 1) * R])


# ------------------------------------------------------------ the group ----

def set_stabilizer(terms, orbit, m, max_local=DEFAULT_MAX_LOCAL,
                   max_perms=DEFAULT_MAX_PERMS):
    p = ORBIT_P[orbit]
    R = len(terms)
    keys = {digest(t) for t in terms}
    if len(keys) != R:
        raise AssertionError("two terms of a witness coincide up to phase")
    factors, diagonal = local_factors(orbit, m)
    nloc = len(factors)
    inv = inverse_table(factors, p)
    ident = identity_index(factors, p)
    local_capped = False
    product = not diagonal and nloc ** m <= max_local
    if product:
        codes = itertools.product(range(nloc), repeat=m)
    else:
        local_capped = not diagonal and nloc ** m > nloc
        codes = [(k,) * m for k in range(nloc)]

    table = {}
    ncodes = 0
    for code, vecs in local_images(terms, factors, codes, p, m, product):
        ncodes += 1
        for v in vecs:
            table.setdefault(digest(v), []).append(code)

    base = np.arange(p ** m).reshape((p,) * m)
    if math.factorial(m) > max_perms:
        raise Capped("m! = %d exceeds the permutation cap %d" % (math.factorial(m), max_perms))
    source = itertools.permutations(range(m))

    pure, twisted, elements = [], {}, 0
    t0 = terms[0]
    for pi in source:
        idx = np.transpose(base, pi).ravel()
        hits = table.get(digest(t0[idx]))
        if not hits:
            continue
        moved = [t[idx] for t in terms]
        for code in dict.fromkeys(hits):
            us = [factors[inv[c]] for c in code]
            if {digest(apply_local(t, us, p, m)) for t in moved} == keys:
                elements += 1
                twisted.setdefault(pi, []).append(code)
                if all(c == ident for c in code):
                    pure.append(pi)
    image = sorted(twisted)
    kernel = len(twisted.get(tuple(range(m)), []))
    if kernel and elements != kernel * len(image):
        raise AssertionError("the twisted stabilizer is not a union of cosets of K")
    return {"R": R, "m": m, "p": p, "nloc": nloc, "local_capped": local_capped,
            "pure": sorted(set(pure)), "image": image,
            "kernel": kernel, "elements": elements, "codes": ncodes}


# ---------------------------------------------------------- description ----

def cycle_type(pi):
    m = len(pi)
    seen = [False] * m
    out = []
    for i in range(m):
        if seen[i]:
            continue
        l, j = 0, i
        while not seen[j]:
            seen[j] = True
            j = pi[j]
            l += 1
        out.append(l)
    return tuple(sorted(out, reverse=True))


def _order(pi, m):
    k, q, idt = 1, pi, tuple(range(m))
    while q != idt:
        q = tuple(pi[q[i]] for i in range(m))
        k += 1
    return k


def name_group(perms, m):
    n = len(perms)
    if n == 1:
        return "trivial"
    if n == math.factorial(m):
        return "S_%d" % m
    abelian = all(tuple(a[b[i]] for i in range(m)) == tuple(b[a[i]] for i in range(m))
                  for a in perms for b in perms)
    orders = sorted({_order(pi, m) for pi in perms})
    if max(orders) == n:
        return "C_%d" % n
    if abelian:
        return "abelian %d, element orders %s" % (n, orders)
    return "nonabelian %d, element orders %s" % (n, orders)


def describe(perms, m):
    if not perms:
        return "none"
    types = {}
    for pi in perms:
        types[cycle_type(pi)] = types.get(cycle_type(pi), 0) + 1
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pi in perms:
        for i in range(m):
            a, b = find(i), find(pi[i])
            if a != b:
                parent[a] = b
    sizes = {}
    for i in range(m):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return "%d (%s) types %s orbits %s" % (
        len(perms), name_group(perms, m),
        ",".join("%s^%d" % (".".join(map(str, t)), c) for t, c in sorted(types.items())),
        "+".join(map(str, sorted(sizes.values(), reverse=True))))


# -------------------------------------------------------------- sources ----

def decode_code_terms(rec):
    """The amplitude-code format of research/qutrit_m4_rank5: 0 for zero,
    1..p for the p-th roots of unity, first nonzero entry set to 1."""
    p = ORBIT_P[rec["orbit"]]
    w = np.exp(2j * np.pi / p)
    out = []
    for row in rec["terms"]:
        v = np.array([0j if c == 0 else w ** (c - 1) for c in row])
        out.append(v / np.linalg.norm(v))
    return out


def sources(only=None, mval=None, bounds_only=False):
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "bounds", "*-upper-*.json"))):
        sub = json.load(open(path))
        if "witness" not in sub or not sub["witness"].get("terms"):
            continue
        orbit, m = sub["orbit"], int(sub["m"])
        if m < 2:
            continue
        out.append((os.path.basename(path)[:-5], orbit, m, len(sub["witness"]["terms"]),
                    [term_vector(t, ORBIT_P[orbit], m) for t in sub["witness"]["terms"]]))
    extra = os.path.join(ROOT, "research", "qutrit_m4_rank5", "N_m4_rank7_witness.json")
    if os.path.exists(extra):
        rec = json.load(open(extra))
        out.append(("N-m4-rank7 (research)", rec["orbit"], rec["m"], rec["rank"],
                    decode_code_terms(rec)))
    if not bounds_only:
        for path in sorted(glob.glob(os.path.join(DATA, "*.json"))):
            rec = json.load(open(path))
            orbit, m = rec["orbit"], rec["m"]
            if m < 2:
                continue
            p = ORBIT_P[orbit]
            for i, dec in enumerate(rec["decompositions"]):
                out.append(("%s-m%d-rank%d [%d/%d]" % (orbit, m, rec["rank"], i + 1, rec["count"]),
                            orbit, m, rec["rank"], [term_vector(t, p, m) for t in dec]))
    if only:
        out = [r for r in out if r[1] == only]
    if mval:
        out = [r for r in out if r[2] == mval]
    return out


def ledger_cells():
    rows = json.load(open(os.path.join(ROOT, "docs", "ledger.json")))["rows"]
    lo, hi = {}, {}
    for r in rows:
        if not r.get("ok"):
            continue
        k = (r["orbit"], r["m"])
        if r["direction"] == "lower":
            lo[k] = max(lo.get(k, 0), r["rank"])
        else:
            hi[k] = min(hi.get(k, 10 ** 9), r["rank"])
    return lo, hi


def check_target(terms, orbit, m):
    psi = target_numeric(orbit, m)
    U = np.column_stack(terms)
    d, *_ = np.linalg.lstsq(U, psi, rcond=None)
    return float(np.linalg.norm(U @ d - psi)), int(np.linalg.matrix_rank(U, tol=1e-8))


# ----------------------------------------------------------------- main ----

def cost(row, a):
    """Rough work estimate, so the cheap witnesses print first."""
    _, orbit, m, _, terms = row
    factors, diagonal = local_factors(orbit, m)
    n = len(factors)
    codes = n if (diagonal or n ** m > a.max_local) else n ** m
    if math.factorial(m) > a.max_perms:
        return float("inf")
    return codes * len(terms) * m * (ORBIT_P[orbit] ** m) + math.factorial(m) * (ORBIT_P[orbit] ** m)


def run(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--m", type=int)
    ap.add_argument("--minimal", action="store_true",
                    help="only witnesses at a cell the ledger knows exactly")
    ap.add_argument("--bounds-only", action="store_true")
    ap.add_argument("--index", default=None,
                    help="comma separated 1-based positions inside a stored list")
    ap.add_argument("--max-local", type=int, default=DEFAULT_MAX_LOCAL)
    ap.add_argument("--max-perms", type=int, default=DEFAULT_MAX_PERMS)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv[1:])
    if a.control:
        return control()
    lo, hi = ledger_cells()
    rows = sources(a.only, a.m, a.bounds_only)
    if a.minimal:
        rows = [r for r in rows if lo.get((r[1], r[2])) == hi.get((r[1], r[2])) == r[3]]
    if a.index:
        want = {int(x) for x in a.index.split(",")}
        rows = [r for r in rows if "[" in r[0] and int(r[0].split("[")[1].split("/")[0]) in want]
    print("%-26s %2s %3s %4s %5s  %-38s %s" %
          ("witness", "m", "R", "min", "|K|", "pure S_m stabilizer", "twisted image in S_m"),
          flush=True)
    rows.sort(key=lambda r: cost(r, a))
    for label, orbit, m, rank, terms in rows:
        t_start = time.time()
        res, rk = check_target(terms, orbit, m)
        if res > 1e-7 or rk != len(terms):
            print("%-26s residual %.2e, numeric rank %d: skipped" % (label, res, rk), flush=True)
            continue
        try:
            info = set_stabilizer(terms, orbit, m, a.max_local, a.max_perms)
        except Capped as exc:
            print("%-26s %2d %3d %4s  not computed: %s" % (label, m, len(terms), "", exc), flush=True)
            continue
        minimal = "yes" if lo.get((orbit, m)) == hi.get((orbit, m)) == rank else "no"
        flags = []
        if info["local_capped"]:
            flags.append("local diagonal only (%d of %d^%d)" % (info["codes"], info["nloc"], m))
        print("%-26s %2d %3d %4s %5d  %-38s %s%s" %
              (label, m, len(terms), minimal, info["kernel"],
               describe(info["pure"], m), describe(info["image"], m),
               ("  [" + "; ".join(flags) + "]") if flags else ""), flush=True)
        if time.time() - t_start > 30:
            print("    (%.0f s)" % (time.time() - t_start), flush=True)
    return 0


def control():
    ok = True

    def show(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print("  %-56s %-4s (got %s, want %s)" % (name, "ok" if good else "FAIL", got, want))

    # the batched image builder against the direct mode products
    rec = json.load(open(os.path.join(DATA, "H3_m2_rank3.json")))
    terms = [term_vector(t, 3, 2) for t in rec["decompositions"][0]]
    factors, _ = local_factors("H3", 2)
    direct, batched = {}, {}
    for code in itertools.product(range(len(factors)), repeat=2):
        direct[code] = [digest(apply_local(t, [factors[c] for c in code], 3, 2)) for t in terms]
    for code, vecs in local_images(terms, factors, None, 3, 2, True):
        batched[code] = [digest(v) for v in vecs]
    show("local_images agrees with the direct mode products",
         batched == direct and len(batched) == len(factors) ** 2, True)

    sub = json.load(open(os.path.join(ROOT, "bounds", "cat-m6-upper-3.json")))
    info = set_stabilizer([term_vector(t, 2, 6) for t in sub["witness"]["terms"]], "cat", 6)
    show("cat m=6 rank 3: pure stabilizer is all of S_6", len(info["pure"]), 720)

    sub = json.load(open(os.path.join(ROOT, "bounds", "cat-m7-upper-6.json")))
    info = set_stabilizer([term_vector(t, 2, 7) for t in sub["witness"]["terms"]], "cat", 7)
    show("cat m=7 rank 6: the 7-cycle is not in the twisted image",
         tuple(list(range(1, 7)) + [0]) in info["image"], False)

    for orbit in ("N", "H3"):
        rec = json.load(open(os.path.join(DATA, "%s_m3_rank4.json" % orbit)))
        best = max(len(set_stabilizer([term_vector(t, 3, 3) for t in dec], orbit, 3)["image"])
                   for dec in rec["decompositions"])
        show("%s m=3 rank 4: a stored decomposition reaches S_3" % orbit, best, 6)

    sub = json.load(open(os.path.join(ROOT, "bounds", "qubit_H-m2-upper-2.json")))
    info = set_stabilizer([term_vector(t, 2, 2) for t in sub["witness"]["terms"]], "qubit_H", 2)
    show("qubit_H m=2 rank 2: twisted image is S_2", len(info["image"]), 2)

    sub = json.load(open(os.path.join(ROOT, "bounds", "qubit_H-m4-upper-4.json")))
    info = set_stabilizer([term_vector(t, 2, 4) for t in sub["witness"]["terms"]], "qubit_H", 4)
    show("qubit_H m=4 rank 4: |H| equals |K| times |image|",
         info["elements"], info["kernel"] * len(info["image"]))

    T = [term_vector(t, 2, 4) for t in sub["witness"]["terms"]]
    shuffled = [T[2], T[0], T[3], T[1]]
    show("the answer does not depend on the order of the terms",
         set_stabilizer(shuffled, "qubit_H", 4)["elements"], info["elements"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv))
