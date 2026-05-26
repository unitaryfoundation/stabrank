"""Independent verifier for the two-copy conversion protocols.

Re-runs the explicit gate sequences in equation (eq:two-copy-protocols)
of paper/main.tex on the H_3 and Norrell magic states and checks that
the post-measurement state on the designated ancilla branch matches
the phase state claimed in equation (eq:two-copy-states):

  H_3:     k=1 branch, probability 3/8, phases (pi/2, pi/3)
  Norrell: k=0 branch, probability 1/4, phases (pi, pi)  (equivalently -1, -1)

This is independent of the search script that produced the JSON
certificates --- it executes the gate sequence directly from the paper
text and confirms numerical agreement.

Usage:
    uv run python scripts/research/orbit_paper/verify_two_copy_protocols.py
"""

from __future__ import annotations

import sys

import numpy as np

W = np.exp(2j * np.pi / 3)
H = np.array([[1, 1, 1], [1, W, W**2], [1, W**2, W]], dtype=np.complex128) / np.sqrt(3)
S = np.diag([1.0, 1.0, W]).astype(np.complex128)
I3 = np.eye(3, dtype=np.complex128)
SUM = np.zeros((9, 9), dtype=np.complex128)
for x in range(3):
    for y in range(3):
        SUM[3 * x + ((x + y) % 3), 3 * x + y] = 1.0


def apply_gate(label: str) -> np.ndarray:
    if label == "SUM":
        return SUM
    name, leg = label[0], int(label[1])
    g = {"H": H, "S": S}[name]
    return np.kron(g, I3) if leg == 1 else np.kron(I3, g)


def run_protocol(gate_seq: list[str], psi_pair: np.ndarray) -> np.ndarray:
    state = psi_pair.copy()
    for g in gate_seq:
        state = apply_gate(g) @ state
    return state


def verify(name: str, psi: np.ndarray, seq: list[str], branch_k: int,
           expected_prob: float, expected_phases_rad: tuple[float, float]
           ) -> bool:
    out = run_protocol(seq, np.kron(psi, psi))
    proj = out.reshape(3, 3)[:, branch_k]
    prob = float(np.linalg.norm(proj) ** 2)
    proj_norm = proj / np.linalg.norm(proj)
    canon = proj_norm / (proj_norm[0] / abs(proj_norm[0]))
    measured_phases = (float(np.angle(canon[1])), float(np.angle(canon[2])))

    def angle_eq(a: float, b: float, tol: float = 1e-10) -> bool:
        return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi) < tol

    prob_ok = abs(prob - expected_prob) < 1e-10
    phase_ok = (angle_eq(measured_phases[0], expected_phases_rad[0])
                and angle_eq(measured_phases[1], expected_phases_rad[1]))
    modulus_ok = all(abs(abs(c) - 1.0 / np.sqrt(3)) < 1e-10 for c in canon)

    status = "OK" if (prob_ok and phase_ok and modulus_ok) else "FAIL"
    print(f"  [{status}] {name}: prob={prob:.6f} (expected {expected_prob}); "
          f"phases (deg)=[{np.degrees(measured_phases[0]):.2f}, "
          f"{np.degrees(measured_phases[1]):.2f}] "
          f"(expected [{np.degrees(expected_phases_rad[0]):.2f}, "
          f"{np.degrees(expected_phases_rad[1]):.2f}])")
    return prob_ok and phase_ok and modulus_ok


def main() -> int:
    print("Verifying two-copy conversion protocols from paper §6.1:")

    H3 = np.array([1.0, (np.sqrt(3) - 1) / 2, (np.sqrt(3) - 1) / 2],
                  dtype=np.complex128)
    H3 = H3 / np.linalg.norm(H3)
    Norrell = np.array([1.0, 1.0, -2.0], dtype=np.complex128) / np.sqrt(6)

    seq_h3 = ["H1", "SUM", "H1", "SUM", "H1", "S2", "SUM",
              "H1", "H2", "H2"]
    seq_norrell = ["H1", "H2", "SUM", "H1", "H1", "H1", "H2",
                   "SUM", "H1"]

    all_ok = True
    all_ok &= verify("H_3", H3, seq_h3, branch_k=1,
                     expected_prob=3 / 8,
                     expected_phases_rad=(np.pi / 2, np.pi / 3))
    all_ok &= verify("Norrell", Norrell, seq_norrell, branch_k=0,
                     expected_prob=1 / 4,
                     expected_phases_rad=(np.pi, np.pi))

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
