"""Shared metadata helpers for orbit-paper certificate JSON files."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


CERTIFICATE_SCHEMA_VERSION = 3
HASH_DECIMALS = 12


def normalize_rows(array: np.ndarray) -> np.ndarray:
    """Return a complex128 row-normalized copy of a stabilizer dictionary."""
    out = np.asarray(array, dtype=np.complex128)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / norms


def array_sha256(array: np.ndarray) -> str:
    """Hash a numerical array with explicit shape and canonical rounding.

    The certificate arrays are generated from roots of unity, square roots,
    and normalization steps. Raw complex128 bytes are too strict for CI:
    libm/platform differences can change the last few bits while leaving the
    represented mathematical certificate unchanged. Hash rounded real/imag
    components instead, and rely on the verifier's residual recomputation for
    the strict numerical check.
    """
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.complex128))
    real = np.round(arr.real, HASH_DECIMALS)
    imag = np.round(arr.imag, HASH_DECIMALS)
    real[real == 0.0] = 0.0
    imag[imag == 0.0] = 0.0
    payload = {
        "dtype": "complex128",
        "hash_decimals": HASH_DECIMALS,
        "shape": list(arr.shape),
    }
    hasher = hashlib.sha256()
    hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(np.ascontiguousarray(real).tobytes(order="C"))
    hasher.update(b"\0")
    hasher.update(np.ascontiguousarray(imag).tobytes(order="C"))
    return hasher.hexdigest()


def current_git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit if available."""
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def build_certificate_metadata(
    *,
    target: np.ndarray,
    stabilizer_dictionary: np.ndarray,
    tuple_size: int,
    script: str,
    parameters: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build reproducibility metadata for a pair/triple-search certificate."""
    return {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "target_sha256": array_sha256(target),
        "stabilizer_dictionary_sha256": array_sha256(stabilizer_dictionary),
        "stabilizer_dictionary_size": int(stabilizer_dictionary.shape[0]),
        "search": {
            "script": script,
            "tuple_size": int(tuple_size),
            "parameters": parameters,
            "git_commit": current_git_commit(repo_root),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
    }


def validate_certificate_metadata(
    *,
    cert: dict[str, Any],
    target: np.ndarray,
    stabilizer_dictionary: np.ndarray,
    tuple_size: int,
) -> list[str]:
    """Return a list of certificate metadata validation errors."""
    errors: list[str] = []

    schema = cert.get("certificate_schema_version")
    if schema != CERTIFICATE_SCHEMA_VERSION:
        errors.append(
            "certificate_schema_version "
            f"{schema!r} != {CERTIFICATE_SCHEMA_VERSION}"
        )

    target_hash = cert.get("target_sha256")
    expected_target_hash = array_sha256(target)
    if target_hash != expected_target_hash:
        errors.append("target_sha256 mismatch")

    dictionary_hash = cert.get("stabilizer_dictionary_sha256")
    expected_dictionary_hash = array_sha256(stabilizer_dictionary)
    if dictionary_hash != expected_dictionary_hash:
        errors.append("stabilizer_dictionary_sha256 mismatch")

    dictionary_size = cert.get("stabilizer_dictionary_size")
    expected_dictionary_size = int(stabilizer_dictionary.shape[0])
    if dictionary_size != expected_dictionary_size:
        errors.append(
            "stabilizer_dictionary_size "
            f"{dictionary_size!r} != {expected_dictionary_size}"
        )

    search = cert.get("search")
    if not isinstance(search, dict):
        errors.append("missing search metadata")
    elif search.get("tuple_size") != tuple_size:
        errors.append(
            f"search.tuple_size {search.get('tuple_size')!r} != {tuple_size}"
        )

    return errors
