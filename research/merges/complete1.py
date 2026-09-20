"""Exact one-state completion of a plateau or a stored partial decomposition,
without a dictionary.

autoresearch/kopt.py answers "can k of the plateau's terms be replaced so
that the rest plus k dictionary states span the target" by projecting the
dictionary; at five qutrits (5.4e9 states) or eight qubits there is no
dictionary to project. For k = 1 the question is whether span(psi, kept
terms) contains a stabilizer state outside span(kept), and
mergelib.stabilizer_states_in_span lists every stabilizer state in that
subspace flat by flat, exactly. This is that test for every choice of the
dropped term.

Usage: complete1.py ORBIT M plateau.npz        (run.py --save-plateaus output)
       complete1.py ORBIT M warm.json          (a bounds-style term file)
ORBIT may be T3sector<s> with M the carry-state qutrit count, as in run.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mergelib import ORBIT_P, sector_target, stabilizer_states_in_span, target  # noqa: E402


def load_terms(path, p, m):
    if path.endswith(".npz"):
        z = np.load(path)
        keys = sorted((k for k in z.files if k.startswith("arr_")), key=lambda s: int(s[4:]))
        return [np.asarray(z[k], dtype=complex) for k in keys], float(z["residual"]) if "residual" in z.files else None
    from stabrank_verify import stabilizer_vector
    sub = json.load(open(path))
    return [np.array([complex(x) for x in stabilizer_vector(t, p, m)]) for t in sub["witness"]["terms"]], None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit")
    ap.add_argument("m", type=int)
    ap.add_argument("path")
    ap.add_argument("--max-null-dim", type=int, default=14)
    a = ap.parse_args(argv[1:])
    if a.orbit.startswith("T3sector"):
        p, psi = 3, sector_target(a.m + 1, int(a.orbit[len("T3sector"):]))
    else:
        p, psi = ORBIT_P[a.orbit], target(a.orbit, a.m)
    psi = psi / np.linalg.norm(psi)
    terms, resid = load_terms(a.path, p, a.m)
    terms = [t / np.linalg.norm(t) for t in terms]
    r = len(terms)
    T = np.column_stack(terms)
    fit, *_ = np.linalg.lstsq(T, psi, rcond=None)
    print(f"{r} terms, residual {np.linalg.norm(T @ fit - psi):.4f}"
          f"{'' if resid is None else f' (saved {resid:.4f})'}, term matrix rank "
          f"{np.linalg.matrix_rank(T, tol=1e-8)}")
    t0 = time.time()
    total = 0
    for drop in range(r):
        kept = [terms[i] for i in range(r) if i != drop]
        found, diag = stabilizer_states_in_span(np.column_stack([psi] + kept), p, a.m,
                                                max_null_dim=a.max_null_dim)
        Qk, _ = np.linalg.qr(np.column_stack(kept))
        comp = [(u, t) for u, t in found if np.linalg.norm(u - Qk @ (Qk.conj().T @ u)) > 1e-6]
        total += len(comp)
        line = (f"  drop {drop}: {len(found)} stabilizer states in span(psi, kept), {len(comp)} outside span(kept)"
                f"{', skipped ' + str(len(diag['skipped'])) + ' flats' if diag['skipped'] else ''}"
                f" [{time.time() - t0:.0f}s]")
        if comp:
            line = "  COMPLETION " + line.strip() + f"; new term flat dims {[t['k'] for _, t in comp]}"
            out = os.path.splitext(a.path)[0] + f"_completed_drop{drop}.npz"
            np.savez(out, *(kept + [comp[0][0]]))
            line += f"; saved {out}"
        print(line, flush=True)
    print(f"{'no' if total == 0 else total} one-state completion(s) for any dropped term")
    return 0 if total else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
