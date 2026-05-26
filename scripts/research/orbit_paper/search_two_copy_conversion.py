"""Search for a two-copy conversion protocol: |M>x|M> -> phase state.

For each of the three non-T3 qutrit magic states (Strange, H3, Norrell),
exhaustively search all two-qutrit Cliffords C in Sp(4, F_3) to find
protocols of the form:

    C (|M> tensor |M>)  ->  measure second qutrit  ->  phase state on first

This is the qutrit analogue of the Bravyi-Kitaev (2005) two-copy trick
for the qubit T-type magic state.

The search also classifies each hit as producing a Clifford or non-Clifford
diagonal gate, and decomposes the best protocol into elementary gates.

Usage:
    uv run python scripts/research/orbit_paper/search_two_copy_conversion.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np

from stabrank.qutrit_clifford import (
    H3,
    S3,
    SUM_GATE,
    _symplectic_generators_4,
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

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "paper" / "certificates"
SP4_F3_ORDER = 51_840
OMEGA = np.exp(2j * np.pi / 3)

GATE_NAMES = ["H1", "H2", "S1", "S2", "SUM"]

I3 = np.eye(3, dtype=complex)
GATE_UNITARIES = [
    np.kron(H3, I3),   # H1
    np.kron(I3, H3),   # H2
    np.kron(S3, I3),   # S1
    np.kron(I3, S3),   # S2
    SUM_GATE,           # SUM
]


class Hit(NamedTuple):
    elem_index: int
    measurement_outcome: int
    success_prob: float
    phase_angles_deg: tuple[float, float]
    is_clifford_gate: bool
    gate_word: list[int]
    state: np.ndarray


def is_phase_state(psi: np.ndarray, tol: float = 1e-6) -> bool:
    """Check whether a qutrit state has all amplitudes of equal modulus."""
    mods = np.abs(psi)
    if np.max(mods) < tol:
        return False
    mods_norm = mods / np.max(mods)
    return bool(np.allclose(mods_norm, 1.0, atol=tol))


def is_clifford_diagonal(alpha_deg: float, beta_deg: float, tol: float = 1.0) -> bool:
    """Check if diag(1, e^{i alpha}, e^{i beta}) is a Clifford gate.

    A diagonal qutrit Clifford has phases that are multiples of 120 degrees.
    """
    def is_mult_120(angle: float) -> bool:
        r = angle % 360.0
        return min(r, 360 - r) < tol or abs(r - 120) < tol or abs(r - 240) < tol

    return is_mult_120(alpha_deg) and is_mult_120(beta_deg)


def enumerate_with_words(
    report_every: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """BFS over Sp(4, F_3), tracking both unitary and generator word.

    Returns list of (F_symplectic, U_unitary, word) triples.
    The word is a list of generator indices [g_k, ..., g_1] such that
    U = G_k @ G_{k-1} @ ... @ G_1  (applied right to left on the state).
    The circuit order (state first encounters) is reversed: G_1, G_2, ..., G_k.
    """
    generators_symp = _symplectic_generators_4()
    generators_unitary = GATE_UNITARIES

    seen: set[bytes] = set()
    results: list[tuple[np.ndarray, np.ndarray, list[int]]] = []
    queue: deque[tuple[np.ndarray, np.ndarray, list[int]]] = deque()

    F_id = np.eye(4, dtype=np.int64)
    U_id = np.eye(9, dtype=complex)

    seen.add(F_id.tobytes())
    results.append((F_id, U_id, []))
    queue.append((F_id, U_id, []))

    while queue:
        F_curr, U_curr, word_curr = queue.popleft()
        for gi, (g_symp, g_unit) in enumerate(
            zip(generators_symp, generators_unitary)
        ):
            F_new = (g_symp @ F_curr) % 3
            key = F_new.tobytes()
            if key not in seen:
                seen.add(key)
                U_new = g_unit @ U_curr
                word_new = [gi] + word_curr
                results.append((F_new, U_new, word_new))
                queue.append((F_new, U_new, word_new))

        if report_every and len(results) % report_every == 0:
            print(f"  [BFS] {len(results)} elements, queue {len(queue)}")

    return results


def word_to_circuit(word: list[int]) -> list[str]:
    """Convert a generator word to a circuit description (state-first order).

    The word [g_k, ..., g_1] means U = G_k @ ... @ G_1, so the state
    first encounters G_1, then G_2, etc.
    """
    return [GATE_NAMES[g] for g in reversed(word)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write", action="store_true", help="Skip writing certificates."
    )
    args = parser.parse_args()

    print("Qutrit two-copy conversion search")
    print("=" * 60)
    print("Searching for: C(|M>x|M>) -> measure -> phase state")
    print()

    targets = {name: fn(1) for name, fn in ORBITS.items()}
    two_copy = {name: np.kron(s, s) for name, s in targets.items()}

    # BFS with word tracking
    print("  Enumerating Sp(4, F_3) with generator words...")
    elements = enumerate_with_words()
    print(f"  Found {len(elements)} elements")

    all_hits: dict[str, list[Hit]] = {name: [] for name in ORBITS}

    t0 = time.time()
    for elem_idx, (_, C_ent, word) in enumerate(elements):
        for name, mm in two_copy.items():
            psi_out = C_ent @ mm

            for k in range(3):
                psi_k = np.array(
                    [psi_out[3 * i + k] for i in range(3)], dtype=complex
                )
                norm = np.linalg.norm(psi_k)
                if norm < 1e-10:
                    continue

                psi_k_norm = psi_k / norm

                if is_phase_state(psi_k_norm):
                    phases = np.angle(psi_k_norm)
                    alpha = float(np.degrees(phases[1] - phases[0]))
                    beta = float(np.degrees(phases[2] - phases[0]))
                    prob = float(norm**2)
                    cliff = is_clifford_diagonal(alpha, beta)

                    all_hits[name].append(
                        Hit(
                            elem_index=elem_idx,
                            measurement_outcome=k,
                            success_prob=prob,
                            phase_angles_deg=(alpha, beta),
                            is_clifford_gate=cliff,
                            gate_word=word,
                            state=psi_k_norm,
                        )
                    )

    elapsed = time.time() - t0
    n_checked = len(elements)
    print(f"\nChecked {n_checked} elements x {len(ORBITS)} orbits x 3 outcomes "
          f"in {elapsed:.1f}s")
    assert n_checked == SP4_F3_ORDER, f"Expected {SP4_F3_ORDER}, got {n_checked}"

    # --- Analysis per orbit ---
    if not args.no_write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ORBITS:
        hits = all_hits[name]
        n_total = len(hits)
        non_cliff = [h for h in hits if not h.is_clifford_gate]
        n_nc = len(non_cliff)

        print(f"\n{'=' * 60}")
        print(f"  Orbit: {name}")
        print(f"  Total hits: {n_total}")
        print(f"  Non-Clifford hits: {n_nc}")

        if n_total == 0:
            print("  No conversion protocols found.")
            continue

        # Find best non-Clifford protocol (highest probability, shortest word)
        best = None
        if non_cliff:
            non_cliff.sort(key=lambda h: (-h.success_prob, len(h.gate_word)))
            best = non_cliff[0]
            print("  Best non-Clifford protocol:")
        else:
            hits.sort(key=lambda h: (-h.success_prob, len(h.gate_word)))
            best = hits[0]
            print("  Best protocol (Clifford-only):")

        circuit = word_to_circuit(best.gate_word)
        print(f"    Success probability: {best.success_prob:.4f}")
        print(f"    Measurement outcome: k={best.measurement_outcome}")
        print(f"    Phase angles: alpha={best.phase_angles_deg[0]:.1f} deg, "
              f"beta={best.phase_angles_deg[1]:.1f} deg")
        print(f"    Clifford gate: {best.is_clifford_gate}")
        print(f"    Gate sequence ({len(circuit)} gates): {' -> '.join(circuit)}")
        print(f"    Output state: {best.state}")

        # Distinct phase angle pairs (modulo Clifford equivalence)
        angle_set: set[tuple[float, float]] = set()
        for h in hits:
            a = round(h.phase_angles_deg[0], 1) % 360
            b = round(h.phase_angles_deg[1], 1) % 360
            angle_set.add((a, b))
        print(f"  Distinct phase-angle pairs: {len(angle_set)}")
        for a, b in sorted(angle_set)[:10]:
            cliff_str = "Cliff" if is_clifford_diagonal(a, b) else "NON-Cliff"
            print(f"    ({a:.1f} deg, {b:.1f} deg) [{cliff_str}]")

        # Distinct success probabilities
        prob_set = sorted(set(round(h.success_prob, 6) for h in hits), reverse=True)
        print(f"  Distinct success probabilities: {prob_set[:5]}")

        if not args.no_write:
            cert = {
                "orbit": name,
                "search_domain": "Sp(4, F_3)",
                "n_elements_checked": n_checked,
                "n_total_hits": n_total,
                "n_non_clifford_hits": n_nc,
                "best_protocol": {
                    "elem_index": best.elem_index,
                    "measurement_outcome": best.measurement_outcome,
                    "success_prob": best.success_prob,
                    "phase_angles_deg": list(best.phase_angles_deg),
                    "is_clifford_gate": best.is_clifford_gate,
                    "gate_sequence": circuit,
                    "state_real": best.state.real.tolist(),
                    "state_imag": best.state.imag.tolist(),
                },
                "distinct_phase_angles": sorted(
                    [list(ab) for ab in angle_set]
                ),
            }
            cert_path = OUTPUT_DIR / f"two_copy_conversion_{name}.json"
            cert_path.write_text(json.dumps(cert, indent=2) + "\n")
            print(f"    -> {cert_path.relative_to(Path.cwd())}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    for name in ORBITS:
        hits = all_hits[name]
        nc = sum(1 for h in hits if not h.is_clifford_gate)
        status = ("UNIVERSAL (non-Clifford gate)" if nc > 0
                  else "Clifford only" if hits else "NONE")
        print(f"  {name:8s}: {len(hits):6d} protocols, {nc:5d} non-Clifford -> {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
