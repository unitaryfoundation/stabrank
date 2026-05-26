"""Verify the JSON certificates under paper/certificates/.

Each pair_<orbit>_m<m>.json or triple_<orbit>_m<m>.json file claims:
  - the search ran to completion (n_pairs_processed == n_pairs_total)
  - no k-tuple of stabilizer states gives residual below `best_residual`
  - the witness pair/triple `best_pair`/`best_triple` achieves that residual
  - the certificate target and stabilizer dictionary match the saved hashes

The exhaustiveness claim (no k-tuple does better than `best_residual`) can
only be re-verified by re-running the underlying search (~minutes for k=2,
~days for k=3). What this script *can* check, in seconds, is that the
witness pair/triple actually does achieve the claimed residual when its
indices are decoded against the canonical stabilizer-state dictionary.

A failure here means the certificate has been tampered with or the
dictionary enumeration has changed in a non-backward-compatible way.

Usage:
    uv run python scripts/research/orbit_paper/verify_certificates.py
    uv run python scripts/research/orbit_paper/verify_certificates.py --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from stabrank.stabilizer_extent import enumerate_stabilizer_states
from stabrank.target_functions import (
    qubit_magic_phase_sum,
    qubit_t_type_magic_state,
    qutrit_complex_magic_state,
    qutrit_hadamard_eigenstate,
    qutrit_norrell_state,
    qutrit_strange_state,
)

try:
    from scripts.research.orbit_paper._certificate_metadata import (
        normalize_rows,
        validate_certificate_metadata,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _certificate_metadata import normalize_rows, validate_certificate_metadata

# The qudit enumerator in stabrank.stabilizer_extent uses d-th-root-of-unity
# phases and so misses the |+/-i> states for d=2. Use the proper qubit
# enumerator (Z/4 phase polynomial) when d == 2.
try:
    from scripts.research.qubit_stabilizer_enum import (
        enumerate_qubit_stabilizer_states,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from qubit_stabilizer_enum import enumerate_qubit_stabilizer_states

# Manifest of certificate files the paper's claims depend on. Each entry
# corresponds to a claim in the paper's state-of-the-art / lower-bound
# tables. Strict mode (`--strict`) fails if any REQUIRED file is missing
# or fails verification.
REQUIRED = (
    # Pair search: chi >= 3 lower bounds for every orbit at m = 1, 2, 3.
    "pair_strange_m1.json", "pair_strange_m2.json", "pair_strange_m3.json",
    "pair_h3_m1.json",      "pair_h3_m2.json",      "pair_h3_m3.json",
    "pair_norrell_m1.json", "pair_norrell_m2.json", "pair_norrell_m3.json",
    "pair_t3_m1.json",      "pair_t3_m2.json",      "pair_t3_m3.json",
    # Qubit T-type chi >= 3 lower bound at m = 3 (tightens the table
    # entry from chi <= 3 to chi = 3).
    "pair_qubit_t_m3.json",
    # Triple search: chi >= 4 lower bounds at m = 3 for the three
    # non-T_3 qutrit orbits.
    "triple_strange_m3.json",
    "triple_h3_m3.json",
    "triple_norrell_m3.json",
    # Qubit H-type chi >= 4 lower bound at m = 4, against the
    # standard phase-state representative (Clifford-equivalent to
    # cos(pi/8)|0> + sin(pi/8)|1>).
    "triple_qubit_h_m4.json",
)

# Certificates marked as in-progress in the paper (e.g., red
# \placeholdernote markers). Promote to REQUIRED in the same commit that
# adds the file.
PENDING = ()

ORBITS = {
    "strange": (qutrit_strange_state,       3),
    "h3":      (qutrit_hadamard_eigenstate, 3),
    "norrell": (qutrit_norrell_state,       3),
    "t3":      (qutrit_complex_magic_state, 3),
    "qubit_t": (qubit_t_type_magic_state,   2),
    # H-type qubit orbit, standard phase-state representative
    # (|0> + e^{i pi/4} |1>)/sqrt(2). Clifford-equivalent to the
    # cos(pi/8)|0> + sin(pi/8)|1> H-type form used in the paper.
    "qubit_h": (qubit_magic_phase_sum,      2),
}


def _enumerate_dictionary(n: int, d: int) -> np.ndarray:
    """Dispatch to the correct enumerator for the local dimension."""
    if d == 2:
        return enumerate_qubit_stabilizer_states(n).astype(np.complex128)
    return enumerate_stabilizer_states(n, d=d).astype(np.complex128)


def witness_residual(
    S: np.ndarray, psi: np.ndarray, indices: list[int]
) -> float:
    """Squared residual of psi when projected onto span(S[i] for i in indices),
    returned as residual = sqrt(max(0, 1 - ||P_span psi||^2)).
    """
    A = S[indices]                  # (k, D)
    G = A.conj() @ A.T              # (k, k) Gram matrix
    b = A.conj() @ psi              # (k,) inner products
    try:
        c = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(G, b, rcond=None)
    proj_norm_sq = float(np.real(c.conj() @ b))
    return float(np.sqrt(max(0.0, 1.0 - proj_norm_sq)))


def verify_certificate(path: Path, abs_tol: float = 1e-9) -> tuple[bool, str]:
    """Returns (ok, message). The message describes what was checked."""
    cert = json.loads(path.read_text())
    orbit = cert.get("orbit")
    m = cert.get("m")
    if orbit not in ORBITS or m is None:
        return False, "missing or unknown orbit/m field"

    # Decide pair vs triple from the field names.
    if "best_pair" in cert:
        kind = "pair"
        indices = cert["best_pair"]
        n_proc = cert.get("n_pairs_processed")
        n_tot = cert.get("n_pairs_total")
        witness_flag = "chi_le_2_witness"
        tuple_size = 2
    elif "best_triple" in cert:
        kind = "triple"
        indices = cert["best_triple"]
        n_proc = cert.get("n_triples_processed")
        n_tot = cert.get("n_triples_total")
        witness_flag = "chi_le_3_witness"
        tuple_size = 3
    else:
        return False, "neither best_pair nor best_triple present"

    # Completeness: search ran to total.
    if n_proc != n_tot:
        return False, (
            f"incomplete search: {kind}s processed {n_proc} of {n_tot}"
        )

    # Internal consistency: the chi_le_k_witness flag should match certificate
    # string and best_residual.
    is_witness = bool(cert.get(witness_flag, False))
    best_res_claimed = float(cert["best_residual"])
    if is_witness != (best_res_claimed < 1e-7):
        return False, (
            f"inconsistent: {witness_flag}={is_witness} vs best_residual="
            f"{best_res_claimed:.3e}"
        )

    # Recompute the witness residual and compare to the claimed value.
    target_fn, d = ORBITS[orbit]
    psi = target_fn(m).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    S = _enumerate_dictionary(m, d)
    S = normalize_rows(S)
    metadata_errors = validate_certificate_metadata(
        cert=cert,
        target=psi,
        stabilizer_dictionary=S,
        tuple_size=tuple_size,
    )
    if metadata_errors:
        return False, "metadata validation failed: " + "; ".join(metadata_errors)

    if max(indices) >= S.shape[0]:
        return False, (
            f"witness index {max(indices)} out of range for "
            f"|Stab_{m}^({d})| = {S.shape[0]}"
        )
    recomputed = witness_residual(S, psi, indices)

    # Below the 1e-7 witness threshold the claimed and recomputed values are
    # both "effectively zero" (the saved value is often clipped to exactly 0).
    # In that regime we only require both sides to be in the witness zone.
    witness_threshold = 1e-7
    in_witness_zone = (best_res_claimed < witness_threshold and
                       recomputed < witness_threshold)
    if in_witness_zone:
        pass  # both agree on "witness", precise values irrelevant below threshold
    elif abs(recomputed - best_res_claimed) > abs_tol:
        return False, (
            f"residual mismatch: claimed {best_res_claimed:.12e}, "
            f"recomputed {recomputed:.12e}, |diff|={abs(recomputed - best_res_claimed):.3e}"
        )

    return True, (
        f"{kind} {orbit} m={m}: {n_tot:,} {kind}s scanned; "
        f"witness {indices} gives residual {recomputed:.6e} "
        f"(claim {best_res_claimed:.6e}); certificate \"{cert['certificate']}\""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cert-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "paper" / "certificates",
        help="Directory containing pair_*.json / triple_*.json files.",
    )
    parser.add_argument(
        "--abs-tol", type=float, default=1e-9,
        help="Allowed absolute difference between claimed and recomputed "
             "residual.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 on any failure (default: report and exit 0).",
    )
    args = parser.parse_args()

    if not args.cert_dir.is_dir():
        print(f"ERROR: certificate directory {args.cert_dir} does not exist",
              file=sys.stderr)
        return 1

    paths = sorted(
        list(args.cert_dir.glob("pair_*.json"))
        + list(args.cert_dir.glob("triple_*.json"))
    )
    if not paths:
        print(f"WARNING: no certificate files found in {args.cert_dir}")
        return 0

    n_ok = 0
    n_fail = 0
    for p in paths:
        try:
            ok, msg = verify_certificate(p, abs_tol=args.abs_tol)
        except Exception as e:
            ok, msg = False, f"exception: {type(e).__name__}: {e}"
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {p.name}: {msg}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print()
    print(f"Verified {n_ok}/{n_ok + n_fail} certificates.")

    # Manifest enforcement. The REQUIRED tuple lists certificate files
    # whose existence is load-bearing for the paper's claims; PENDING
    # lists files marked as in-progress in the paper.
    present = {p.name for p in paths}
    missing_required = [name for name in REQUIRED if name not in present]
    missing_pending = [name for name in PENDING if name not in present]
    extra = sorted(present - set(REQUIRED) - set(PENDING))

    if missing_required:
        print()
        print(f"MISSING REQUIRED ({len(missing_required)}):")
        for name in missing_required:
            print(f"  - {name}")
    if missing_pending:
        print()
        print("PENDING (paper marks as in-progress; not yet enforced):")
        for name in missing_pending:
            print(f"  - {name}")
    if extra:
        print()
        print(f"NOTE: {len(extra)} cert file(s) present but not in manifest:")
        for name in extra:
            print(f"  - {name}")

    if args.strict and (n_fail > 0 or missing_required):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
