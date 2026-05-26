"""Sweep mana + fidelity lower bounds across the four Clifford-inequivalent
qutrit magic state orbits (Strange, H3, Norrell, T3) for small m.

Outputs a JSON file `orbit_lower_bounds.json` aggregating all results, and
prints a per-orbit/per-m table to stdout.

Usage:
    python sweep_orbit_lower_bounds.py [--m_max=4]
"""

import argparse
import json
import time
import numpy as np

from stabrank.mana import compute_mana
from stabrank.stabrank_core import max_stabilizer_fidelity
from stabrank.target_functions import (
    qutrit_strange_state,
    qutrit_hadamard_eigenstate,
    qutrit_norrell_state,
    qutrit_complex_magic_state,
)


ORBITS = [
    ("Strange",      qutrit_strange_state),
    ("H3",           qutrit_hadamard_eigenstate),
    ("Norrell",      qutrit_norrell_state),
    ("T3",           qutrit_complex_magic_state),
]


def run_one(name: str, builder, m: int, do_fidelity: bool) -> dict:
    psi = builder(m)
    psi_normed = psi / np.linalg.norm(psi)
    out = {"orbit": name, "m": m, "dim": int(3 ** m)}

    t0 = time.time()
    try:
        mana = compute_mana(psi_normed, n=m, d=3)
        out["wigner_l1"] = float(mana["wigner_l1"])
        out["mana"] = float(mana["mana"])
        out["extent_lb_mana"] = float(mana["extent_lb"])
        out["t_mana_s"] = time.time() - t0
    except Exception as e:
        out["mana_error"] = str(e)

    if do_fidelity:
        t0 = time.time()
        try:
            fid = max_stabilizer_fidelity(psi_normed, n=m, d=3)
            out["f_max"] = float(fid["f_max"])
            out["extent_lb_fidelity"] = int(fid["extent_lb"])
            out["t_fidelity_s"] = time.time() - t0
        except Exception as e:
            out["fidelity_error"] = str(e)

    return out


def print_row(r: dict) -> None:
    fid = f"F_max={r.get('f_max', float('nan')):.6f}, xi_LB_fid={r.get('extent_lb_fidelity', '?')}"
    if "f_max" not in r:
        fid = "(fidelity skipped or failed)"
    print(
        f"  {r['orbit']:>8} m={r['m']} (dim={r['dim']:5d}): "
        f"||W||1={r.get('wigner_l1', float('nan')):.4f}, "
        f"xi_LB_mana={r.get('extent_lb_mana', '?'):.4f}, {fid}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_max", type=int, default=4,
                        help="Largest m to sweep (default 4).")
    parser.add_argument("--fid_m_max", type=int, default=None,
                        help="Largest m for which to run exhaustive fidelity "
                             "(default = m_max). Set lower if fidelity is too slow.")
    parser.add_argument("--out", type=str, default="orbit_lower_bounds.json")
    args = parser.parse_args()

    fid_m_max = args.fid_m_max if args.fid_m_max is not None else args.m_max

    print(f"=== Orbit lower-bound sweep, m=1..{args.m_max} ===")
    print(f"Mana: all m. Fidelity: m <= {fid_m_max}.")
    print()

    results = []
    for m in range(1, args.m_max + 1):
        do_fidelity = m <= fid_m_max
        print(f"--- m={m}, dim={3**m}, fidelity={'on' if do_fidelity else 'skipped'} ---")
        for name, builder in ORBITS:
            r = run_one(name, builder, m, do_fidelity)
            results.append(r)
            print_row(r)
        print()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
