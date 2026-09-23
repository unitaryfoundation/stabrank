"""Measurements for the rank-5 exclusion of |H>^5 by a two-qubit base slice
(n1 = 2, n2 = 3), reusing the H^6 machinery and census.

  witness [--base K] [--list]   the rank-6 witness of |H>^5 sliced along every
                                qubit pair: all-visible bases, flat types, and
                                the matcher on one base
  sampleA [--count N] [--reference]
                                stage A cost at x0 in {00, 01} on N census covers
  sampleBC [--count N] [--cap S]
                                stage B and C cost on covers of degenerate_covers_v2
  beta [--count N] [--plant K]  the invisible-line prototype on N full 4-covers
                                (and K planted instances)
"""
import argparse
import itertools
import json
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "research", "constructions"))
from slice_cover import (CoverEnumerator, Family, SliceMatcher, TermOptions, _reduce,  # noqa: E402
                         apply_pauli, exact_codes, pauli_reps, valid_term_codes, x0_reps)
from rank_exclusion import psi_for  # noqa: E402
import common as cc  # noqa: E402  (research/constructions/common.py)

N1, N2, M = 2, 3, 5
COS, SIN = np.cos(np.pi / 8), np.sin(np.pi / 8)


def alpha(x, n1=N1):
    w = bin(x).count("1")
    return COS ** (n1 - w) * SIN ** w


def genuine(h, rank):
    return h["rank"] == rank and h["exact"] and h["independent"] and h["nonzero"] and h["residual"] < 1e-8


class Deadline(Exception):
    pass


def _alarm(signum, frame):
    raise Deadline()


# ------------------------------------------------------------- witness ----

def slice_terms(terms, S, x0, m, n1, lookup):
    perm = list(S) + [q for q in range(m) if q not in S]
    out = []
    for v in terms:
        s = v.reshape([2] * m).transpose(perm).reshape(1 << n1, -1)[x0]
        if np.linalg.norm(s) < 1e-9:
            return None
        codes, _ = exact_codes(s)
        out.append(lookup[codes.tobytes()])
    return tuple(out)


def flat_type(v, S, m, n1):
    """Points of F_2^n1 at which the term is nonzero, along the qubits S."""
    perm = list(S) + [q for q in range(m) if q not in S]
    sl = v.reshape([2] * m).transpose(perm).reshape(1 << n1, -1)
    return tuple(int(x) for x in range(1 << n1) if np.linalg.norm(sl[x]) > 1e-9)


def witness(args):
    E = CoverEnumerator(N2)
    lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
    w = json.load(open(os.path.join(ROOT, "bounds", "qubit_H-m5-upper-6.json")))
    terms = [cc.term_vector(t, 2, M) for t in w["witness"]["terms"]]
    psi = psi_for("qubit_H", M)
    c, *_ = np.linalg.lstsq(np.column_stack(terms), psi, rcond=None)
    assert np.linalg.norm(np.column_stack(terms) @ c - psi) < 1e-9
    print("witness coefficients", np.round(c, 4))
    bases = {}
    for S in itertools.combinations(range(M), N1):
        types = [flat_type(v, S, M, N1) for v in terms]
        names = []
        for t in types:
            names.append({4: "plane", 2: "line" + str(t), 1: "point" + str(t)}[len(t)])
        print(f"pair {S}: " + ", ".join(names))
        for x0 in range(1 << N1):
            b = slice_terms(terms, S, x0, M, N1, lookup)
            if b is not None:
                bases.setdefault((b, x0), []).append(S)
    print(f"{len(bases)} distinct all-visible (base, x0) pairs")
    items = sorted(bases.items())
    for k, ((cover, x0), Ss) in enumerate(items):
        distinct = sorted(set(cover))
        fam = Family.from_cover(E, distinct)
        print(f"  base {k}: cover {cover} x0 {x0:02b} pairs {Ss} distinct {len(distinct)} "
              f"kappa {None if fam is None else fam.kappa}")
    if args.list:
        return 0
    Mt = SliceMatcher(E, N1, verbose=args.verbose)
    sel = range(len(items)) if args.base is None else [args.base]
    for k in sel:
        (cover, x0), Ss = items[k]
        t0 = time.time()
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(args.cap)
        try:
            hits, st = Mt.run(cover, x0)
        except Deadline:
            print(f"  base {k}: DEADLINE after {time.time() - t0:.0f}s")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  base {k}: ABORTED {type(exc).__name__}: {str(exc)[:120]} after {time.time() - t0:.0f}s")
            continue
        finally:
            signal.alarm(0)
        good = [h for h in hits if genuine(h, 6)]
        S = Ss[0]
        perm = list(S) + [q for q in range(M) if q not in S]
        wkey = sorted(exact_codes(v.reshape([2] * M).transpose(perm).reshape(-1))[0].tobytes() for v in terms)
        same = any(sorted(exact_codes(t)[0].tobytes() for t in h["terms"]) == wkey for h in good)
        print(f"  base {k}: {len(good)} genuine rank-6 decompositions, witness itself "
              f"{'recovered' if same else 'NOT among them'}, {time.time() - t0:.1f}s, "
              f"coord solutions {st['coord_solutions']}, joined {st['joined']}, kappa {st['kappa']}, "
              f"blocks {st['blocks']}, native {st['native']}")
    return 0


# ------------------------------------------------------------- stage A ----

def census_covers(E, count, pivot=0):
    i = int(E.reps[pivot])
    Qi, _ = _reduce(E.F1, E.Q1, E.Q1[i])
    members, partners = E.pivot_plan(i)
    mask = np.zeros(E.N, dtype=bool)
    mask[members] = True
    covers = []
    for j in partners:
        got, _ = E.pair_covers(5, i, int(j), Qi, mask)
        covers.extend(sorted(got))
        if len(covers) >= count:
            break
    return covers[:count]


def sampleA(args):
    E = CoverEnumerator(N2)
    Mt = SliceMatcher(E, N1, native=not args.reference)
    t0 = time.time()
    covers = census_covers(E, args.count)
    print(f"{len(covers)} covers from the first pivot in {time.time() - t0:.1f}s")
    x0s = [0b00, 0b01]
    times = {x: [] for x in x0s}
    hist = {}
    hits = 0
    for cover in covers:
        for x0 in x0s:
            t1 = time.time()
            h, st = Mt.run(cover, x0)
            times[x0].append(time.time() - t1)
            k = ",".join(str(b) for b in st["coord_solutions"])
            hist[k] = hist.get(k, 0) + 1
            hits += sum(genuine(x, 5) for x in h)
    for x0 in x0s:
        a = np.array(times[x0])
        print(f"x0 {x0:02b}: per (cover, x0) mean {a.mean() * 1e3:.2f} ms, median {np.median(a) * 1e3:.2f} ms, "
              f"max {a.max() * 1e3:.1f} ms ({'reference' if Mt.native is None else 'native'})")
    print("coordinate-slice solution histogram:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    print("genuine rank-5 hits:", hits)
    return 0


# --------------------------------------------------------- stages B, C ----

def sampleBC(args):
    E = CoverEnumerator(N2)
    Mt = SliceMatcher(E, N1, native=False)
    doc = json.load(open(os.path.join(ROOT, "research", "h6_rank5", "degenerate_covers_v2.json")))
    covers = [tuple(c) for c in doc["covers"]]
    by_pat = {}
    for cv in covers:
        pat = tuple(sorted((cv.count(u) for u in set(cv)), reverse=True))
        by_pat.setdefault(pat, []).append(cv)
    print({str(k): len(v) for k, v in by_pat.items()})
    signal.signal(signal.SIGALRM, _alarm)
    for pat, lst in sorted(by_pat.items(), key=lambda kv: -len(kv[1])):
        n = min(args.count, len(lst))
        pick = [lst[int(round(i * (len(lst) - 1) / max(n - 1, 1)))] for i in range(n)]
        times, capped, hits, reached = [], 0, 0, 0
        for cv in pick:
            for x0 in (0b00, 0b01):
                t1 = time.time()
                signal.alarm(args.cap)
                try:
                    h, st = Mt.run(cv, x0)
                    hits += sum(genuine(x, 5) for x in h)
                    reached += st.get("reconstructions", 0)
                except Deadline:
                    capped += 1
                finally:
                    signal.alarm(0)
                times.append(time.time() - t1)
        a = np.array(times)
        print(f"pattern {pat}: {n} covers x 2 base points: per run mean {a.mean():.2f} s, median "
              f"{np.median(a):.2f} s, max {a.max():.2f} s, capped {capped}, hits {hits}, "
              f"reconstructions {reached}")
    return 0


# ----------------------------------------------- the invisible-line case ----

class BetaMatcher:
    """Base point x0 with four visible terms (a full 4-cover, distinct and
    independent) and one invisible line term on the diagonal missing x0.
    The equation at x0 ^ 11 is exact over 32 options per term; at the two
    coordinate points the residual after the visible terms must be a
    nonzero multiple of a stabilizer state, the two residuals must be
    Pauli translates of each other with the same coefficient, and the
    visible terms' codes must compose to their 11-codes."""

    def __init__(self, E):
        self.E = E
        self.cache = {}
        self.rng = np.random.default_rng(5)
        # sorted dictionary codes for the residual lookup
        self.codes = E.codes.astype(np.int8)
        self.keys = np.ascontiguousarray(self.codes).view(np.dtype((np.void, self.codes.shape[1]))).ravel()
        order = np.argsort(self.keys)
        self.skeys = self.keys[order]
        self.sidx = order

    def options(self, idx):
        if idx not in self.cache:
            self.cache[idx] = TermOptions(self.E.C[:, idx], N2, self.E.F1, self.E.F2)
        return self.cache[idx]

    def stab_lookup(self, R):
        """For rows of R (n, 8): index of the stabilizer state proportional
        to the row, or -1."""
        n, dim = R.shape
        A = np.abs(R)
        mx = A.max(axis=1)
        ok = mx > 1e-7
        nz = A > 1e-7 * np.maximum(mx, 1e-300)[:, None]
        mn = np.where(nz, A, np.inf).min(axis=1)
        ok &= (mx - mn) < 1e-6 * mx
        cnt = nz.sum(axis=1)
        ok &= (cnt & (cnt - 1)) == 0
        first = np.argmax(nz, axis=1)
        ref = R[np.arange(n), first]
        W = R / np.where(np.abs(ref) > 0, ref, 1)[:, None]
        codes = np.zeros((n, dim), dtype=np.int8)
        for c, val in enumerate([1, 1j, -1, -1j], start=1):
            codes[np.abs(W - val) < 1e-6] = c
        ok &= (codes > 0).sum(axis=1) == cnt
        out = np.full(n, -1, dtype=np.int64)
        if not np.any(ok):
            return out
        k = np.ascontiguousarray(codes[ok]).view(np.dtype((np.void, dim))).ravel()
        pos = np.searchsorted(self.skeys, k)
        pos = np.minimum(pos, len(self.skeys) - 1)
        hit = self.skeys[pos] == k
        res = np.where(hit, self.sidx[pos], -1)
        out[np.flatnonzero(ok)] = res
        return out

    def run(self, cover, x0, rhs):
        """rhs: dict x -> 8-vector, the slices of the target along the pair.
        Returns the list of genuine 5-term decompositions found."""
        r = len(cover)
        opts = [self.options(u) for u in cover]
        U = np.column_stack([o.u for o in opts])
        c, *_ = np.linalg.lstsq(U, rhs[x0], rcond=None)
        assert np.linalg.norm(U @ c - rhs[x0]) < 1e-9 and np.all(np.abs(c) > 1e-9)
        xb, xa1, xa2 = x0 ^ 0b11, x0 ^ 0b01, x0 ^ 0b10
        V = [o.vecs[:32] for o in opts]                      # 32 x 8 each
        # exact slice at xb: meet in the middle on a random functional
        f = self.rng.standard_normal(8) + 1j * self.rng.standard_normal(8)
        L = (c[0] * V[0])[:, None, :] + (c[1] * V[1])[None, :, :]        # 32 x 32 x 8
        R = rhs[xb][None, None, :] - (c[2] * V[2])[:, None, :] - (c[3] * V[3])[None, :, :]
        kl = np.round((L @ f) * 1e5).ravel()
        kr = np.round((R @ f) * 1e5).ravel()
        kl_c = kl.real + 1j * kl.imag
        d = {}
        for n, key in enumerate(kl_c):
            d.setdefault(complex(key), []).append(n)
        sols_b = []
        for n2, key in enumerate(kr):
            for n1 in d.get(complex(key), ()):
                a0, a1 = divmod(n1, 32)
                a2, a3 = divmod(n2, 32)
                if np.linalg.norm(L[a0, a1] - R[a2, a3]) < 1e-8:
                    sols_b.append((a0, a1, a2, a3))
        stats = {"b_solutions": len(sols_b), "a1_candidates": 0, "a1_stab": 0, "pairs": 0, "hits": 0}
        hits = []
        for cb in sols_b:
            # options at xa1: absent (line term on {x0, xb}) or 32; enumerate all 33^4
            optl = [np.vstack([o.vecs[:32], np.zeros((1, 8))]) for o in opts]     # 33 x 8
            S01 = (c[0] * optl[0])[:, None, None, None, :] + (c[1] * optl[1])[None, :, None, None, :] \
                + (c[2] * optl[2])[None, None, :, None, :] + (c[3] * optl[3])[None, None, None, :, :]
            Rr = (rhs[xa1][None, None, None, None, :] - S01).reshape(-1, 8)
            stats["a1_candidates"] += Rr.shape[0]
            found = self.stab_lookup(Rr)
            cand = np.flatnonzero(found >= 0)
            stats["a1_stab"] += len(cand)
            for q in cand:
                a = np.unravel_index(q, (33, 33, 33, 33))
                res1 = Rr[q]
                # xa2 codes: a line term (absent at xa1) is absent at xa2; a plane
                # term's xa2 code is fixed by its xa1 and xb codes up to a sign
                choices = []
                okterm = True
                for i in range(r):
                    if a[i] == 32:
                        choices.append([32])
                        continue
                    ka, la = divmod(int(a[i]), 4)
                    kb, lb = divmod(int(cb[i]), 4)
                    kc = None
                    for k2 in range(8):
                        if int(opts[i].prod[ka, k2, 0]) == kb:
                            kc = k2
                            break
                    if kc is None:
                        okterm = False
                        break
                    # phase: lb = la + l2 + prod_phase + 2 sign
                    ph = int(opts[i].prod[ka, kc, 1])
                    l2s = [(lb - la - ph - 2 * s) % 4 for s in (0, 1)]
                    choices.append([4 * kc + l2 for l2 in l2s])
                if not okterm:
                    continue
                for a2 in itertools.product(*choices):
                    S10 = sum(c[i] * optl[i][a2[i]] for i in range(r))
                    res2 = rhs[xa2] - S10
                    if np.linalg.norm(res2) < 1e-9:
                        continue
                    stats["pairs"] += 1
                    # res2 must be i^l Q res1 for some Pauli (a, c) and phase l
                    matched = False
                    for pa in range(8):
                        for pc in range(8):
                            t = apply_pauli(res1, pa, pc, N2)
                            ratio = None
                            nz = np.flatnonzero(np.abs(t) > 1e-9)
                            ratio = res2[nz[0]] / t[nz[0]]
                            if abs(abs(ratio) - 1) < 1e-6 and np.linalg.norm(res2 - ratio * t) < 1e-8 \
                                    and min(abs(ratio - z) for z in (1, 1j, -1, -1j)) < 1e-6:
                                matched = True
                                break
                        if matched:
                            break
                    if not matched:
                        continue
                    # assemble the five terms and confirm
                    terms, coeffs = [], []
                    for i in range(r):
                        t = np.zeros((4, 8), dtype=complex)
                        t[x0] = opts[i].u
                        t[xa1] = optl[i][a[i]]
                        t[xa2] = optl[i][a2[i]]
                        t[xb] = opts[i].vecs[cb[i]]
                        codes = {0b01: None if a[i] == 32 else int(a[i]), 0b10: None if a2[i] == 32 else int(a2[i]),
                                 0b11: int(cb[i])}
                        terms.append(t.ravel())
                        coeffs.append(c[i])
                    v5 = np.zeros((4, 8), dtype=complex)
                    v5[xa1] = res1
                    v5[xa2] = res2
                    terms.append(v5.ravel())
                    coeffs.append(1.0)
                    A = np.column_stack(terms)
                    target = np.zeros((4, 8), dtype=complex)
                    for x in range(4):
                        target[x] = rhs[x]
                    target = target.ravel()
                    cc5, *_ = np.linalg.lstsq(A, target, rcond=None)
                    resid = np.linalg.norm(A @ cc5 - target)
                    rank = np.linalg.matrix_rank(A, tol=1e-8)
                    # every term a stabilizer state: check the phase-pattern form via exact_codes and support
                    good = resid < 1e-8 and rank == 5 and np.all(np.abs(cc5) > 1e-9)
                    stab = True
                    for t in terms:
                        try:
                            cd, _ = exact_codes(t)
                        except AssertionError:
                            stab = False
                            break
                        supp = np.count_nonzero(cd)
                        if supp & (supp - 1):
                            stab = False
                            break
                    if good and stab:
                        stats["hits"] += 1
                        hits.append((terms, cc5))
        return hits, stats


def random_stabilizer_term(rng, E, flat, n1=N1):
    """A random 5-qubit stabilizer state whose flat along the first two
    qubits is `flat` (a tuple of points), built by the structure lemma from
    a random 3-qubit base state and random Pauli classes and phases."""
    u = E.C[:, rng.integers(E.N)]
    reps, imgs = pauli_reps(u, N2)
    t = np.zeros((4, 8), dtype=complex)
    pts = sorted(flat)
    x0 = pts[0]
    t[x0] = u
    if len(pts) == 4:
        k1, k2 = rng.integers(8), rng.integers(8)
        l1, l2, s = rng.integers(4), rng.integers(4), rng.integers(2)
        (a1, c1), (a2, c2) = reps[k1], reps[k2]
        t[x0 ^ 0b01] = (1j) ** l1 * apply_pauli(u, a1, c1, N2)
        t[x0 ^ 0b10] = (1j) ** l2 * apply_pauli(u, a2, c2, N2)
        t[x0 ^ 0b11] = (-1) ** s * (1j) ** (l1 + l2) * apply_pauli(apply_pauli(u, a1, c1, N2), a2, c2, N2)
    elif len(pts) == 2:
        k, l = rng.integers(8), rng.integers(4)
        a, c = reps[k]
        t[pts[1]] = (1j) ** l * apply_pauli(u, a, c, N2)
    return t.ravel()


def beta(args):
    E = CoverEnumerator(N2)
    B = BetaMatcher(E)
    rng = np.random.default_rng(11)
    # planted instances: 3 planes, one line on {00, 11}, one line on {01, 10}
    for k in range(args.plant):
        while True:
            terms = [random_stabilizer_term(rng, E, (0, 1, 2, 3)) for _ in range(3)]
            terms.append(random_stabilizer_term(rng, E, (0b00, 0b11)))
            terms.append(random_stabilizer_term(rng, E, (0b01, 0b10)))
            coeffs = rng.integers(1, 4, size=5) * np.exp(2j * np.pi * rng.integers(8, size=5) / 8)
            if np.linalg.matrix_rank(np.column_stack(terms), tol=1e-8) == 5:
                break
        target = sum(cf * t for cf, t in zip(coeffs, terms)).reshape(4, 8)
        rhs = {x: target[x] for x in range(4)}
        x0 = 0b00
        lookup = {E.codes[i].tobytes(): i for i in range(E.N)}
        base = []
        for t in terms[:4]:
            cd, _ = exact_codes(t.reshape(4, 8)[x0])
            base.append(lookup[cd.tobytes()])
        if np.linalg.matrix_rank(np.column_stack([E.C[:, b] for b in base]), tol=1e-8) < 4:
            print(f"plant {k}: dependent base, skipped")
            continue
        t0 = time.time()
        hits, st = B.run(tuple(base), x0, rhs)
        # is the planted decomposition among the hits?
        wkey = sorted(exact_codes(t)[0].tobytes() for t in terms)
        same = any(sorted(exact_codes(t)[0].tobytes() for t in h[0]) == wkey for h in hits)
        print(f"plant {k}: base {base}, {len(hits)} hits, planted {'recovered' if same else 'NOT recovered'}, "
              f"{time.time() - t0:.2f}s, stats {st}")
    # real 4-covers of |H>^3 against |H>^5
    cache = os.path.join(SP, "covers4.json")
    if os.path.exists(cache):
        covers4 = [tuple(c) for c in json.load(open(cache))]
    else:
        t0 = time.time()
        covers4, _ = E.covers(4)
        os.makedirs(SP, exist_ok=True)
        json.dump([list(map(int, c)) for c in covers4], open(cache, "w"))
        print(f"{len(covers4)} full 4-covers of |H>^3 in {time.time() - t0:.0f}s")
    print(f"{len(covers4)} full 4-covers")
    psi3 = psi_for("qubit_H", N2)
    n = min(args.count, len(covers4))
    pick = [covers4[int(round(i * (len(covers4) - 1) / max(n - 1, 1)))] for i in range(n)]
    for x0 in (0b00, 0b01):
        rhs = {x: alpha(x) * psi3 for x in range(4)}
        times, tot = [], {}
        hits = 0
        for cv in pick:
            t0 = time.time()
            h, st = B.run(cv, x0, rhs)
            times.append(time.time() - t0)
            hits += len(h)
            for kk, v in st.items():
                tot[kk] = tot.get(kk, 0) + v
        a = np.array(times)
        print(f"x0 {x0:02b}: {n} covers, per cover mean {a.mean():.3f} s, median {np.median(a):.3f} s, "
              f"max {a.max():.3f} s; totals {tot}; genuine hits {hits}")
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("witness")
    p.add_argument("--base", type=int, default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--cap", type=int, default=500)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=witness)
    p = sub.add_parser("sampleA")
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--reference", action="store_true")
    p.set_defaults(fn=sampleA)
    p = sub.add_parser("sampleBC")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--cap", type=int, default=60)
    p.set_defaults(fn=sampleBC)
    p = sub.add_parser("beta")
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--plant", type=int, default=4)
    p.set_defaults(fn=beta)
    args = ap.parse_args(argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
