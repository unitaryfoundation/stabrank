"""Slice-and-lift as a constructor, with more terms than the minimal rank.

Setting. phi = sum_k alpha_k |k> is the single-qudit state, r = chi(phi^m) is
known exactly and the complete list of rank-r decompositions of phi^m up to
the unitary symmetry group is stored under data/. A rank-R decomposition of
phi^(m+1), sliced along the first qudit, has slices that are decompositions
of phi^m (or, for alpha_k = 0, null combinations). Each term is one of two
kinds (slice_lift.py, module docstring, case analysis of the stabilizer
group's image on the first qudit):

  full:  every slice nonzero, and slice k+1 = phase * Q * slice k for one
         Pauli Q per term (slice k+2 = phase * Q^-1 * slice k);
  local: the term is |k> (x) v, nonzero on one slice only.

With F full and L = R - F local terms, slice k carries F + L_k nonzero terms
and needs at least r of them when alpha_k != 0. A slice with exactly r
nonzero terms is a minimal decomposition, hence one of the stored ones up to
symmetry, and the symmetry I (x) U preserves slices, so it is enough to take
each stored decomposition as that slice. This script enumerates every
configuration in which some slice is minimal:

  R = r + 1:  one local term. Two slices are minimal. Take one as the base,
              match the other by Paulis and phases (residual zero), and the
              third slice's residual must be a stabilizer state (the local
              term).                                                  [A]
  R = r + 2:  two local terms on the same slice: two minimal slices matched,
              the third residual of stabilizer rank <= 2.             [B1]
              two local terms on different slices: one minimal slice, the
              other two residuals each a stabilizer state.            [B2]

Configurations with no minimal slice (all terms full, or R = r + 2 with one
local term) need the (r+1)- or (r+2)-term decompositions of phi^m and are
out of reach of the stored lists; see the note for what that leaves.

For qubits there are two slices: [A] is one local term on the non-base slice
(a full scan, since there is no third slice to match), and [B1] would need a
rank-2 residual test and is not run.

Every candidate lift is checked numerically against phi^(m+1) and written as
amplitude vectors for verify_challenge/to_witness.py.

Usage: relaxed_lift.py ORBIT M R [--cases A,B1,B2] [--only INDEX]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ORBIT_P, alpha, confirm_stabilizer, is_stabilizer_batch,  # noqa: E402
                    load_decompositions, phase_table, target)
from slice_lift import (_match_sums, _sum_table, _decode, pauli_images,  # noqa: E402
                        pauli_images_qubit)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TOL = 1e-7


def term_images(u, m, p):
    """Per Pauli class: (Q u, Q^-1 u) for qutrits, (Q u, Q u) for qubits
    (Q^-1 = Q up to phase there)."""
    if p == 3:
        return pauli_images(u, m)
    return [(v, v) for v in pauli_images_qubit(u, m)]


def options(dec_coeffs, imgs, which, phases):
    """Row tables, one per term: d_i * phase * image, indexed class*len(phases)+phase."""
    out = []
    for d, im in zip(dec_coeffs, imgs):
        rows = np.array([d * ph * pair[which] for pair in im for ph in phases])
        out.append(rows)
    return out


def match_all(opts, rhs, tol=TOL):
    """Every assignment (one row per term) whose rows sum to rhs."""
    r = len(opts)
    sizes = [len(o) for o in opts]
    half = max(1, r // 2)
    L = _sum_table(opts[:half])
    R = _sum_table(opts[half:]) if r > half else np.zeros((1, len(rhs)), dtype=complex)
    pairs, gap = _match_sums(L, R, rhs, tol)
    combos = []
    for l, rr in pairs:
        combos.append(tuple(_decode(l, sizes[:half]) + (_decode(rr, sizes[half:]) if r > half else [])))
    return combos, gap


def scan_stabilizer_residuals(opts, rhs, p, n):
    """Every assignment whose residual rhs - sum(rows) is a stabilizer state.

    Outer loop over the first two terms, vectorised over the rest.
    """
    r = len(opts)
    sizes = [len(o) for o in opts]
    outer_n = min(2, r)
    inner = _sum_table(opts[outer_n:]) if r > outer_n else np.zeros((1, len(rhs)), dtype=complex)
    found = []
    for outer in itertools.product(*[range(s) for s in sizes[:outer_n]]):
        base = rhs - sum(opts[i][outer[i]] for i in range(outer_n))
        res = base[None, :] - inner
        mask = is_stabilizer_batch(res, p)
        for j in np.flatnonzero(mask):
            t = confirm_stabilizer(res[j], p, n)
            if t is not None:
                combo = tuple(outer) + tuple(_decode(int(j), sizes[outer_n:]) if r > outer_n else ())
                found.append((combo, res[j]))
    return found


def assemble(p, m, base_slice, u, coeffs, combos_by_slice, locals_by_slice):
    """Lifted term vectors on m+1 qudits: full terms from the per-slice images,
    then local terms |k> (x) v."""
    dim = p ** m
    terms = []
    r = len(u)
    for i in range(r):
        blocks = [np.zeros(dim, dtype=complex) for _ in range(p)]
        blocks[base_slice] = u[i]
        for k, rows in combos_by_slice.items():
            blocks[k] = rows[i] / coeffs[i]
        terms.append(np.concatenate(blocks))
    for k, vs in locals_by_slice.items():
        for v in vs:
            blocks = [np.zeros(dim, dtype=complex) for _ in range(p)]
            blocks[k] = v
            terms.append(np.concatenate(blocks))
    return terms


def confirm(terms, psi_up):
    A = np.column_stack(terms)
    if np.linalg.matrix_rank(A, tol=1e-8) < A.shape[1]:
        return None
    c, *_ = np.linalg.lstsq(A, psi_up, rcond=None)
    if np.linalg.norm(A @ c - psi_up) > 1e-8:
        return None
    return c


def run_qutrit(orbit, m, R, decs, cases, log):
    a = alpha(orbit)
    psi = target(orbit, m)
    psi_up = np.kron(a, psi)
    r = len(decs[0][0])
    CUBE = phase_table(3)
    hits = []
    stats = []
    for di, (u, d) in enumerate(decs):
        imgs = [term_images(ui, m, 3) for ui in u]
        for b in range(3):
            if abs(a[b]) < 1e-12:
                continue
            f, g = (b + 1) % 3, (b + 2) % 3
            rho = {k: a[k] / a[b] for k in range(3)}
            opts = {f: options(d, imgs, 0, CUBE), g: options(d, imgs, 1, CUBE)}
            # matched slice t, residual slice k
            for t, k in ((f, g), (g, f)):
                combos, gap = match_all(opts[t], rho[t] * psi)
                rec = {"dec": di, "base": b, "matched": t, "free": k, "matches": len(combos), "gap": gap}
                if "A" in cases or "B1" in cases:
                    n_stab = 0
                    n_rank2_checked = 0
                    for combo in combos:
                        rows_t = [opts[t][i][c] for i, c in enumerate(combo)]
                        # residual at k over all phase assignments
                        cls = [c // 3 for c in combo]
                        which = 0 if k == f else 1
                        Qk = np.column_stack([d[i] * imgs[i][cls[i]][which] for i in range(r)])
                        PH = np.array(list(itertools.product(CUBE, repeat=r))).T
                        res = rho[k] * psi[:, None] - Qk @ PH            # (dim, 3^r)
                        if R == r + 1 and "A" in cases:
                            mask = is_stabilizer_batch(res.T, 3)
                            for j in np.flatnonzero(mask):
                                if confirm_stabilizer(res[:, j], 3, m) is None:
                                    continue
                                n_stab += 1
                                rows_k = [d[i] * PH[i, j] * imgs[i][cls[i]][which] for i in range(r)]
                                terms = assemble(3, m, b, u, d, {t: rows_t, k: rows_k}, {k: [res[:, j]]})
                                c = confirm(terms, psi_up)
                                if c is not None:
                                    hits.append({"case": "A", "dec": di, "base": b, "local_slice": k,
                                                 "terms": terms, "coeffs": c})
                        if R == r + 2 and "B1" in cases and not rank2_available(3, m):
                            rec["rank2_skipped"] = rec.get("rank2_skipped", 0) + res.shape[1]
                        if R == r + 2 and "B1" in cases and rank2_available(3, m):
                            n_rank2_checked += res.shape[1]
                            for j in range(res.shape[1]):
                                v = res[:, j]
                                pair = rank2_states(v, 3, m)
                                if pair is not None:
                                    rows_k = [d[i] * PH[i, j] * imgs[i][cls[i]][which] for i in range(r)]
                                    terms = assemble(3, m, b, u, d, {t: rows_t, k: rows_k}, {k: list(pair)})
                                    c = confirm(terms, psi_up)
                                    if c is not None:
                                        hits.append({"case": "B1", "dec": di, "base": b, "local_slice": k,
                                                     "terms": terms, "coeffs": c})
                    rec["stabilizer_residuals"] = n_stab
                    rec["rank2_residuals_checked"] = n_rank2_checked
                stats.append(rec)
                log(f"  dec {di} base {b}: matched slice {t} ({len(combos)} matches, gap {gap:.1e}), "
                    f"free slice {k}: {rec.get('stabilizer_residuals', '-')} stabilizer residuals"
                    + (f", {rec['rank2_residuals_checked']} residuals rank-2 tested"
                       if rec.get('rank2_residuals_checked') else "")
                    + (f", {rec['rank2_skipped']} residuals not rank-2 tested (dictionary too large)"
                       if rec.get('rank2_skipped') else ""))
            if R == r + 2 and "B2" in cases:
                t0 = time.time()
                surv = scan_stabilizer_residuals(opts[f], rho[f] * psi, 3, m)
                n_pairs = 0
                for combo, res_f in surv:
                    cls = [c // 3 for c in combo]
                    rows_f = [opts[f][i][c] for i, c in enumerate(combo)]
                    Qg = np.column_stack([d[i] * imgs[i][cls[i]][1] for i in range(r)])
                    PH = np.array(list(itertools.product(CUBE, repeat=r))).T
                    res = rho[g] * psi[:, None] - Qg @ PH
                    mask = is_stabilizer_batch(res.T, 3)
                    for j in np.flatnonzero(mask):
                        if confirm_stabilizer(res[:, j], 3, m) is None:
                            continue
                        n_pairs += 1
                        rows_g = [d[i] * PH[i, j] * imgs[i][cls[i]][1] for i in range(r)]
                        terms = assemble(3, m, b, u, d, {f: rows_f, g: rows_g}, {f: [res_f], g: [res[:, j]]})
                        c = confirm(terms, psi_up)
                        if c is not None:
                            hits.append({"case": "B2", "dec": di, "base": b, "terms": terms, "coeffs": c})
                rec = {"dec": di, "base": b, "case": "B2", "slice_f_stabilizer_residuals": len(surv),
                       "both_slices": n_pairs, "seconds": round(time.time() - t0, 1)}
                stats.append(rec)
                log(f"  dec {di} base {b}: B2 scan: {len(surv)} assignments with a stabilizer residual at "
                    f"slice {f}, {n_pairs} also at slice {g} [{time.time() - t0:.0f}s]")
    return hits, stats


def run_qubit(orbit, m, R, decs, cases, log):
    a = alpha(orbit)
    psi = target(orbit, m)
    psi_up = np.kron(a, psi)
    r = len(decs[0][0])
    FOURTH = phase_table(2)
    hits, stats = [], []
    for di, (u, d) in enumerate(decs):
        imgs = [term_images(ui, m, 2) for ui in u]
        for b in range(2):
            f = 1 - b
            rho = a[f] / a[b]
            opts = options(d, imgs, 0, FOURTH)
            t0 = time.time()
            combos, gap = match_all(opts, rho * psi)
            rec = {"dec": di, "base": b, "matches": len(combos), "gap": gap}
            if R == r + 1 and "A" in cases:
                surv = scan_stabilizer_residuals(opts, rho * psi, 2, m)
                rec["stabilizer_residuals"] = len(surv)
                for combo, res in surv:
                    rows_f = [opts[i][c] for i, c in enumerate(combo)]
                    terms = assemble(2, m, b, u, d, {f: rows_f}, {f: [res]})
                    c = confirm(terms, psi_up)
                    if c is not None:
                        hits.append({"case": "A", "dec": di, "base": b, "local_slice": f,
                                     "terms": terms, "coeffs": c})
            rec["seconds"] = round(time.time() - t0, 1)
            stats.append(rec)
            log(f"  dec {di} base {b}: {len(combos)} strict matches (gap {gap:.1e}); "
                f"{rec.get('stabilizer_residuals', '-')} stabilizer residuals [{rec['seconds']}s]")
    return hits, stats


_DICT = {}


def rank2_available(p, n):
    """Whether the n-qudit dictionary is small enough to hold for the rank-2
    residual test (3 qutrits: 30240 states; 4 qubits: 36720)."""
    return (p, n) in {(3, 1), (3, 2), (3, 3), (2, 1), (2, 2), (2, 3), (2, 4)}


def rank2_states(v, p, n):
    """Two dictionary states spanning v, or None. Uses the full dictionary on n
    qudits, so only for p^n small enough to hold it (3 qutrits, 4 qubits)."""
    from rank_exclusion import dictionary, rank2_search
    if (p, n) not in _DICT:
        _DICT[(p, n)] = dictionary(p, n)
    D = _DICT[(p, n)]
    res, _ = rank2_search(v, D)
    if res == "RANK1":
        return None
    if not res:
        return None
    i, j = res[0]
    return D[:, i], D[:, j]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("orbit")
    ap.add_argument("m", type=int, help="copies of the stored decompositions")
    ap.add_argument("R", type=int, help="target rank at m+1")
    ap.add_argument("--cases", default="A,B1,B2")
    ap.add_argument("--only", type=int, default=None, help="index of one stored decomposition")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    a = ap.parse_args(argv[1:])
    p = ORBIT_P[a.orbit]
    rank = {("N", 3): 4, ("H3", 3): 4, ("S", 3): 4, ("S", 4): 4, ("qubit_H", 4): 4, ("qubit_T", 4): 3,
            ("N", 2): 3, ("H3", 2): 3, ("S", 2): 2, ("T3", 2): 3, ("qubit_H", 3): 3, ("qubit_T", 3): 3}[(a.orbit, a.m)]
    decs, rec = load_decompositions(a.orbit, a.m, rank)
    idx = list(range(len(decs)))
    if a.only is not None:
        idx = [a.only]
    if a.first is not None or a.last is not None:
        idx = idx[a.first or 0:a.last]
    decs_sel = [decs[i] for i in idx]
    cases = set(a.cases.split(","))
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"{a.orbit} m={a.m} -> {a.m + 1}, target rank {a.R} from {len(decs_sel)} stored rank-{rank} "
        f"decompositions (indices {idx[0]}..{idx[-1]}), cases {sorted(cases)}")
    t0 = time.time()
    if p == 3:
        hits, stats = run_qutrit(a.orbit, a.m, a.R, decs_sel, cases, log)
    else:
        hits, stats = run_qubit(a.orbit, a.m, a.R, decs_sel, cases, log)
    for s in stats:
        if "dec" in s:
            s["dec"] = idx[s["dec"]]
    log(f"{len(hits)} lifts of rank {a.R} at m={a.m + 1} [{time.time() - t0:.0f}s]")
    tag = f"{a.orbit}_m{a.m + 1}_rank{a.R}" + (f"_{idx[0]}-{idx[-1]}" if (a.only is not None or a.first is not None or a.last is not None) else "")
    with open(os.path.join(RESULTS, f"lift_{tag}.json"), "w") as fh:
        json.dump({"orbit": a.orbit, "m": a.m + 1, "rank": a.R, "cases": sorted(cases),
                   "decompositions": idx, "stats": stats, "lifts": len(hits),
                   "log": lines}, fh, indent=1)
    for h, hit in enumerate(hits):
        vecs = [[[float(z.real), float(z.imag)] for z in t] for t in hit["terms"]]
        with open(os.path.join(RESULTS, f"lift_{tag}_hit{h}.json"), "w") as fh:
            json.dump(vecs, fh)
        log(f"  hit {h}: case {hit['case']} from decomposition {idx[hit['dec']]} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
