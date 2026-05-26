"""Verify the qutrit injection-gadget no-go theorem (paper Theorem 4).

For each of |Strange>, |H_3>, |Norrell>, exhaustively search the
symplectic quotient Sp(4, F_3) of the two-qutrit Clifford group
(51,840 elements, per Lemma "gadget reduction to Sp(4, F_3)"),
checking all three post-selection branches per element via
`stabrank.qutrit_clifford.check_gadget`. Assert no element yields a
valid deterministic injection gadget; emit per-orbit JSON certificates.

The Heisenberg-Weyl factor of the full Cl(2, 3) group is irrelevant by
the gadget-reduction lemma, so the search over Sp(4, F_3) is
exhaustive for the gadget-existence question.

Usage:
    uv run python scripts/research/orbit_paper/verify_gadget_no_go.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from stabrank.qutrit_clifford import (
    build_clifford_lookup,
    check_gadget,
    enumerate_single_qutrit_cliffords,
    iter_two_qutrit_cliffords,
)
from stabrank.target_functions import (
    qutrit_hadamard_eigenstate,
    qutrit_norrell_state,
    qutrit_strange_state,
)


ORBITS = {
    "strange": qutrit_strange_state,
    "h3": qutrit_hadamard_eigenstate,
    "norrell": qutrit_norrell_state,
}

# certificates land under paper/certificates/ alongside the rank ones
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "paper" / "certificates"

# expected size of Sp(4, F_3)
SP4_F3_ORDER = 51_840


def _state_hash(state: np.ndarray) -> str:
    """Short SHA-256 of a state vector's bytes, for cross-run consistency."""
    return hashlib.sha256(state.tobytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing JSON certificates; only assert no-go.",
    )
    args = parser.parse_args()

    print("Qutrit injection-gadget no-go verifier")
    print("=" * 50)

    print("Enumerating single-qutrit Clifford group...")
    clifford_group = enumerate_single_qutrit_cliffords()
    cg_lookup = build_clifford_lookup(clifford_group)
    print(f"  {len(clifford_group)} single-qutrit Cliffords")

    targets = {name: state_fn(1) for name, state_fn in ORBITS.items()}
    hits: dict[str, list[dict]] = {name: [] for name in ORBITS}

    t0 = time.time()
    n_checked = 0
    for elem_idx, C_ent in enumerate(iter_two_qutrit_cliffords(report_every=0)):
        n_checked += 1
        for name, state in targets.items():
            result = check_gadget(C_ent, state, cg_lookup)
            if result is not None:
                hits[name].append(
                    {
                        "elem_index": elem_idx,
                        "success_prob": float(result["success_prob"]),
                    }
                )

    elapsed = time.time() - t0
    print(
        f"\nChecked {n_checked} Sp(4, F_3) elements x {len(ORBITS)} orbits "
        f"in {elapsed:.1f}s"
    )

    if n_checked != SP4_F3_ORDER:
        print(
            f"\nFAIL: expected to traverse {SP4_F3_ORDER} elements of "
            f"Sp(4, F_3); got {n_checked}."
        )
        return 1

    if not args.no_write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    failed = False
    for name in ORBITS:
        cert = {
            "orbit": name,
            "target_hash_sha256_16": _state_hash(targets[name]),
            "search_domain": "Sp(4, F_3)",
            "n_elements_checked": n_checked,
            "expected_domain_order": SP4_F3_ORDER,
            "n_hits": len(hits[name]),
            "hits": hits[name],
            "claim": (
                "no entangling two-qutrit Clifford yields a deterministic "
                "injection gadget for this orbit"
            ),
            "reference": "paper/main.tex Theorem 4 (sec:gadget-obstruction)",
        }
        status = "OK" if not hits[name] else "FAIL"
        line = f"  {name}: hits={len(hits[name])} [{status}]"
        if not args.no_write:
            cert_path = OUTPUT_DIR / f"gadget_no_go_{name}.json"
            cert_path.write_text(json.dumps(cert, indent=2) + "\n")
            line += f"  -> {cert_path.relative_to(Path.cwd())}"
        print(line)
        if hits[name]:
            failed = True

    if failed:
        print("\nFAIL: at least one orbit admits a deterministic injection gadget.")
        return 1

    print(
        "\nOK: no deterministic injection gadget exists for any of "
        f"{list(ORBITS.keys())} over the symplectic quotient Sp(4, F_3)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
