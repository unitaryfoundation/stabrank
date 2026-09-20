"""Coefficient classes of a product decomposition and exact merge tests.

|M>^(A+B) = sum_{i,j} a_i b_j s_i (x) t_j from the stored minimal
decompositions of |M>^A and |M>^B (research/constructions/data). Terms with
equal coefficient form a class; the class sum with its common coefficient is
the vector a merge would have to reproduce with fewer stabilizer states.

For every class this script reports, exactly:
  * whether the class sum is itself a stabilizer state (to_witness's
    flat-plus-quadratic-phase recovery on the vector);
  * the modulus pattern of the class sum and the rank-2 obstruction it
    implies (a sum of two stabilizer states has constant modulus on each
    difference of the two flats and at most g/2 + 1 further values on the
    intersection, g the phase-group order);
  * the stabilizer states in the span of the class terms other than the
    terms (mergelib.stabilizer_states_in_span, complete inside the span),
    and whether the class sum lies in the span of fewer of them than the
    class size.
It also lists the stabilizer states in the span of the whole product
decomposition and writes warm-start files for autoresearch/run.py: the full
product and the product with one term of the largest class removed.

Usage: class_sums.py ORBIT A B [--completion] [--max-null-dim 14]
       (qubit_T 4 4: nine terms on eight qubits; N 3 3, H3 3 3: sixteen
       terms on six qutrits)
--completion runs the one-state completion test (stabilizer states in
span(psi, kept) with two terms dropped) over the pairs inside the largest
class; the full pair list is out of the fifteen-minute budget at these sizes.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mergelib import (ORBIT_P, WARM, coefficient_classes, is_stabilizer, load_decompositions,  # noqa: E402
                      modulus_pattern, product_terms, rank2_excluded,
                      stabilizer_states_in_span, target, write_warm_file)

RANK = {("N", 3): 4, ("H3", 3): 4, ("S", 3): 4, ("S", 4): 4, ("T3", 2): 3,
        ("qubit_H", 4): 4, ("qubit_T", 4): 3, ("qubit_H", 3): 3, ("qubit_T", 3): 3}


def describe(v, p, n):
    t = is_stabilizer(v, p, n)
    pat = modulus_pattern(v)
    supp = int((np.abs(v) > 1e-7).sum())
    return t, pat, supp


def outside(found, basis):
    """States in `found` not parallel to any column of `basis`."""
    B = np.column_stack(basis)
    B = B / np.linalg.norm(B, axis=0)
    return [(u, t) for u, t in found if np.all(np.abs(np.abs(u.conj() @ B) - 1) > 1e-6)]


def fewer_span(v, states, size):
    """Does v lie in the span of some `size` - 1 of the given states?"""
    if len(states) < size - 1:
        return None
    for sub in itertools.combinations(range(len(states)), size - 1):
        A = np.column_stack([states[i] for i in sub])
        x, *_ = np.linalg.lstsq(A, v, rcond=None)
        if np.linalg.norm(A @ x - v) < 1e-8:
            return sub
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit")
    ap.add_argument("A", type=int)
    ap.add_argument("B", type=int)
    ap.add_argument("--completion", action="store_true")
    ap.add_argument("--max-null-dim", type=int, default=14)
    ap.add_argument("--skip-whole-span", action="store_true")
    ap.add_argument("--quick", action="store_true", help="class sums and rank tests only, no span scans or files")
    a = ap.parse_args(argv[1:])
    orbit, A, B = a.orbit, a.A, a.B
    p, m = ORBIT_P[orbit], A + B
    decsA, _ = load_decompositions(orbit, A, RANK[(orbit, A)])
    decsB, _ = load_decompositions(orbit, B, RANK[(orbit, B)])
    assert len(decsA) == 1 and len(decsB) == 1, "these cells have one minimal decomposition up to symmetry"
    terms, coeffs = product_terms(decsA[0], decsB[0])
    psi = target(orbit, m)
    V = np.column_stack(terms)
    assert np.linalg.norm(V @ coeffs - psi) < 1e-8
    dims = [int(is_stabilizer(t, p, m)["k"]) for t in terms]
    print(f"{orbit} m={m} = {A}+{B}: {len(terms)} product terms, flat dims {dims}")
    print("  factor coefficients A:", [f"{abs(c):.4f} e^(i {np.angle(c) / np.pi:+.4f} pi)" for c in decsA[0][1]])
    print("  factor coefficients B:", [f"{abs(c):.4f} e^(i {np.angle(c) / np.pi:+.4f} pi)" for c in decsB[0][1]])
    classes = coefficient_classes(coeffs)
    mods = coefficient_classes(np.abs(coeffs))
    print(f"  classes by equal coefficient: sizes {[len(c) for c in classes]}; by modulus alone: "
          f"{[len(c) for c in mods]}")
    t0 = time.time()
    for cl in classes:
        c0 = coeffs[cl[0]]
        v = V[:, cl].sum(axis=1)
        t, pat, supp = describe(v, p, m)
        exc = rank2_excluded(v, p) if len(cl) >= 3 else None
        print(f"  class {cl} (coefficient {abs(c0):.4f} e^(i {np.angle(c0) / np.pi:+.4f} pi), "
              f"member flat dims {[dims[i] for i in cl]}):")
        print(f"    class sum: {'STABILIZER' if t is not None else 'not a stabilizer state'}; support {supp}; "
              f"distinct moduli {len(pat)} (multiplicities {[q for _, q in pat]})")
        if len(cl) >= 3:
            print(f"    rank 2 {'EXCLUDED' if exc[0] else 'not excluded'}: {exc[1]}")
        if len(cl) >= 2 and not a.quick:
            found, diag = stabilizer_states_in_span(V[:, cl], p, m, max_null_dim=a.max_null_dim)
            extra = outside(found, [V[:, i] for i in cl])
            msg = (f"    stabilizer states in span(class): {len(found)} ({len(extra)} beyond the terms), "
                   f"{diag['flagged']} flagged flats of {diag['flats']}")
            if diag["skipped"]:
                msg += f", {len(diag['skipped'])} flats skipped (null dim > {a.max_null_dim})"
            if extra:
                sub = fewer_span(v, [u for u, _ in found], len(cl))
                msg += f"; class sum in the span of {len(cl) - 1} of them: {'YES ' + str(sub) if sub else 'no'}"
            print(msg + f" [{time.time() - t0:.0f}s]", flush=True)
    if a.quick:
        return 0
    if not a.skip_whole_span:
        found, diag = stabilizer_states_in_span(V, p, m, max_null_dim=a.max_null_dim)
        extra = outside(found, terms)
        print(f"  stabilizer states in span(all {len(terms)} terms): {len(found)} ({len(extra)} beyond the terms); "
              f"{diag['flagged']} flagged of {diag['flats']} flats, {diag['patterns']} phase patterns"
              f"{', skipped ' + str(diag['skipped']) if diag['skipped'] else ''} [{time.time() - t0:.0f}s]")
        if extra:
            states = [u for u, _ in found]
            sub = fewer_span(psi, states, len(terms))
            print(f"    target in the span of {len(terms) - 1} states of the span: {'YES ' + str(sub) if sub else 'no'}")
    # warm-start files
    big = classes[0]
    write_warm_file(os.path.join(WARM, f"{orbit}_m{m}_product{len(terms)}.json"), orbit, m, p, terms, coeffs,
                    note=f"{A}+{B} product of the stored minimal decompositions")
    for i in big:
        keep = [j for j in range(len(terms)) if j != i]
        write_warm_file(os.path.join(WARM, f"{orbit}_m{m}_drop{i}.json"), orbit, m, p,
                        [terms[j] for j in keep], coeffs[keep],
                        note=f"product with term {i} (largest coefficient class) removed")
    print(f"  warm files written for the product and for dropping each of terms {big}")
    if a.completion:
        hits = 0
        for drop in itertools.combinations(big, 2):
            kept = [terms[j] for j in range(len(terms)) if j not in drop]
            found, diag = stabilizer_states_in_span(np.column_stack([psi] + kept), p, m,
                                                    max_null_dim=a.max_null_dim)
            Qk, _ = np.linalg.qr(np.column_stack(kept))
            comp = [t for u, t in found if np.linalg.norm(u - Qk @ (Qk.conj().T @ u)) > 1e-6]
            if comp:
                hits += 1
                print(f"  COMPLETION dropping {drop}: {len(comp)} state(s) complete a rank-{len(terms) - 1} "
                      f"decomposition, flat dims {[t['k'] for t in comp]}")
                np.save(os.path.join(WARM, f"{orbit}_m{m}_completion_{drop[0]}_{drop[1]}.npy"),
                        np.column_stack(kept + [u for u, t in found if t in comp]))
            else:
                print(f"  dropping {drop}: {len(found)} stabilizer states in span(psi, kept), none outside "
                      f"span(kept){', skipped ' + str(len(diag['skipped'])) if diag['skipped'] else ''} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        print(f"  one-state completion over the {len(list(itertools.combinations(big, 2)))} pairs inside "
              f"the largest class: {hits} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
