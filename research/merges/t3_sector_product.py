"""The nine-term product decomposition of the Z-eigensectors of |T3>^6, its
class structure, warm-start files for the annealer, and the exact one-state
completion test with two terms dropped.

|T3>^2 = l_0 + l_1 + l_2 with l_j a line state supported on {x_1 + x_2 = j}
(the unique rank-3 decomposition, research/constructions/data/T3_m2_rank3),
so each l_j lies in one Z^(x2) eigensector. Hence |T3>^6 = (|T3>^2)^(x3)
splits its 27 product terms by sigma_1 + sigma_2 + sigma_3 mod 3, nine per
sector, and each nine-term sum is the sector projection c_s. In the
carry-state coordinates of autoresearch/run.py (`sector_target(6, s)`:
drop x_6 = s - x_1 - ... - x_5) every product term is a 3-flat state on
five qutrits. The coefficient of a term is c_{sigma_1} c_{sigma_2}
c_{sigma_3}, which depends on the integer weight |sigma|, so the terms fall
into weight classes (1, 7, 1 for s = 0; 3, 6 for s = 1; 6, 3 for s = 2).

Exact tests run here, all inside the 243-dimensional sector space:
  * stabilizer states in the span of the nine terms beyond the terms;
  * for every pair of dropped terms, stabilizer states in span(psi, seven
    kept terms): any state outside span(kept) completes a rank-8
    decomposition of the sector, i.e. chi(T3^6) <= 24;
  * the equal-coefficient class sums, tested for stabilizer-ness.

Usage: t3_sector_product.py [--sectors 0 1 2] [--no-completion]
Writes warm/T3sector<s>_m5_product9.json and warm/T3sector<s>_m5_drop<i>.json
(eight terms each) for autoresearch/run.py --warm-from.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mergelib import (WARM, coefficient_classes, is_stabilizer, kron_all, load_decompositions,  # noqa: E402
                    sector_target, stabilizer_states_in_span, write_warm_file)


def line_sectors(lines):
    """The Z^(x2) eigenvalue exponent of each 2-qutrit line state."""
    out = []
    for v in lines:
        supp = np.flatnonzero(np.abs(v) > 1e-9)
        sums = {(i // 3 + i % 3) % 3 for i in supp}
        assert len(sums) == 1, "a line of the T3^2 decomposition crosses sectors"
        out.append(sums.pop())
    return out


def restrict_to_carry(v6, s):
    """v5[x_1..x_5] = v6[x_1..x_5, s - sum x] (the sector code as five qutrits)."""
    idx5 = np.arange(243)
    digits = np.array([(idx5 // 3 ** (4 - j)) % 3 for j in range(5)])
    x6 = (s - digits.sum(axis=0)) % 3
    return v6[3 * idx5 + x6]


def sector_product(s, lines, coeffs, secs):
    """Nine 5-qutrit terms, their coefficients, and their sigma triples."""
    terms, cs, sig = [], [], []
    for tri in itertools.product(range(3), repeat=3):
        if sum(secs[t] for t in tri) % 3 != s:
            continue
        v6 = kron_all([lines[t] for t in tri])
        terms.append(restrict_to_carry(v6, s))
        cs.append(np.prod([coeffs[t] for t in tri]))
        sig.append(tuple(secs[t] for t in tri))
    return terms, np.array(cs), sig


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sectors", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--no-completion", action="store_true")
    a = ap.parse_args(argv[1:])
    decs, _ = load_decompositions("T3", 2, 3)
    lines, coeffs = decs[0]
    secs = line_sectors(lines)
    print(f"|T3>^2 lines lie in Z^(x2) sectors {secs}; coefficients "
          f"{[f'{abs(c):.4f} e^(i {np.angle(c) / np.pi:.4f} pi)' for c in coeffs]}")
    for s in a.sectors:
        t0 = time.time()
        terms, cs, sig = sector_product(s, lines, coeffs, secs)
        psi = sector_target(6, s)
        V = np.column_stack(terms)
        fit, *_ = np.linalg.lstsq(V, psi, rcond=None)
        resid = np.linalg.norm(V @ fit - psi)
        # the product coefficients up to one global scalar
        scale = fit / cs
        print(f"\nsector {s}: 9 product terms, lstsq residual to sector_target(6,{s}) {resid:.2e}, "
              f"rank of the term matrix {np.linalg.matrix_rank(V, tol=1e-8)}; "
              f"product coefficients agree with the fit up to a global scalar: "
              f"{np.allclose(scale, scale[0], atol=1e-9)}")
        assert resid < 1e-9, "the nine-term product does not reproduce the sector state"
        weights = [sum(t) for t in sig]
        for w in sorted(set(weights)):
            members = [i for i, x in enumerate(weights) if x == w]
            print(f"  weight {w}: {len(members)} terms {[sig[i] for i in members]}, "
                  f"coefficient {fit[members[0]]:.6f}, flat dims "
                  f"{sorted({int(is_stabilizer(terms[i], 3, 5)['k']) for i in members})}")
        classes = coefficient_classes(fit)
        print(f"  coefficient classes (equal complex value): sizes {[len(c) for c in classes]}")
        # class sums
        for cl in classes:
            v = V[:, cl].sum(axis=1)
            t = is_stabilizer(v, 3, 5)
            print(f"  class {cl}: equal-coefficient sum is "
                  f"{'a stabilizer state' if t is not None else 'not a stabilizer state'}; "
                  f"support {int((np.abs(v) > 1e-9).sum())}, distinct moduli "
                  f"{len(set(np.round(np.abs(v[np.abs(v) > 1e-9]), 7)))}")
        # warm-start files
        os.makedirs(WARM, exist_ok=True)
        write_warm_file(os.path.join(WARM, f"T3sector{s}_m5_product9.json"), f"T3sector{s}", 5, 3,
                        terms, fit, note=f"nine-term product decomposition of the Z-eigensector {s} "
                        f"of |T3>^6 in carry coordinates; sigma triples {sig}")
        for i in range(9):
            keep = [j for j in range(9) if j != i]
            write_warm_file(os.path.join(WARM, f"T3sector{s}_m5_drop{i}.json"), f"T3sector{s}", 5, 3,
                            [terms[j] for j in keep], fit[keep],
                            note=f"product decomposition with term {i} (sigma {sig[i]}) removed")
        # stabilizer states in the span of the nine terms
        found, diag = stabilizer_states_in_span(V, 3, 5)
        extra = [u for u, _ in found
                 if all(abs(abs(np.vdot(u, t / np.linalg.norm(t))) - 1) > 1e-6 for t in terms)]
        print(f"  stabilizer states in span(9 terms): {len(found)} ({len(extra)} beyond the terms); "
              f"{diag['flats']} flats scanned, {diag['flagged']} with a nonzero null space "
              f"[{time.time() - t0:.0f}s]")
        if a.no_completion:
            continue
        # one-state completion with two terms dropped
        hits = 0
        best = []
        for drop in itertools.combinations(range(9), 2):
            kept = [terms[j] for j in range(9) if j not in drop]
            W = np.column_stack([psi] + kept)
            found, diag = stabilizer_states_in_span(W, 3, 5)
            Qk, _ = np.linalg.qr(np.column_stack(kept))
            comp = []
            for u, term in found:
                r = u - Qk @ (Qk.conj().T @ u)
                if np.linalg.norm(r) > 1e-6:
                    comp.append(term)
            if comp:
                hits += 1
                print(f"  COMPLETION dropping {drop} (sigma {sig[drop[0]]}, {sig[drop[1]]}): "
                      f"{len(comp)} stabilizer state(s) in span(psi, kept) outside span(kept), "
                      f"e.g. flat dim {comp[0]['k']}")
                np.save(os.path.join(WARM, f"T3sector{s}_completion_{drop[0]}{drop[1]}.npy"),
                        np.column_stack(kept + [u for u, t in found if t in comp]))
            best.append((drop, len(found), diag["flagged"]))
        print(f"  one-state completion over the 36 dropped pairs: {hits} pairs complete; "
              f"stabilizer states in span(psi, kept) per pair: "
              f"{sorted({b for _, b, _ in best})} (7 = the kept terms only) [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
