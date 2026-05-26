<p align="center">
  <img src="docs/logo.svg" alt="stabrank" width="180">
</p>

<h1 align="center">stabrank</h1>

<p align="center">
  <em>Tools for stabilizer-rank bounds on quantum magic states.</em>
</p>

<p align="center">
  <a href="https://unitary.foundation"><img alt="Unitary Foundation" src="https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg"></a>
  <a href="https://github.com/unitaryfoundation/stabrank/actions/workflows/tests.yml"><img alt="CI" src="https://github.com/unitaryfoundation/stabrank/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/unitaryfoundation/stabrank/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

**stabrank** is an open-source Python package with a C++ core for computing,
certifying, and formally verifying stabilizer-rank decompositions of quantum
magic states.

Built and maintained by the [Unitary Foundation](https://unitary.foundation).

The library was developed to support the paper
*Stabilizer-rank bounds for qutrit magic-state orbits* and packages the
underlying search-and-verify toolchain as a reusable resource. It exposes
three independent search modes — a simulated-annealing decomposition
search for upper-bound witnesses, an exhaustive _k_-tuple enumeration over
the canonical-form stabilizer-state dictionary for lower-bound certificates,
and an enumeration over the symplectic quotient of the two-qutrit Clifford
group for injection-gadget existence — and pairs each with a stand-alone
verifier so that no result depends on trusting the search heuristic.

## What can it do?

- **Find** stabilizer decompositions of arbitrary qubit and qutrit target
  states via a multi-chain simulated-annealing engine, with Pauli, Clifford,
  and single-state perturbation moves and a C++ kernel.
- **Certify** small-_m_ tight values by exhaustive _k_-tuple enumeration
  over the qutrit stabilizer-state dictionary at _m_ = 1, 2, 3 (dictionary
  sizes 12, 414, 41,580); per-orbit JSON certificates record the
  witnessing tuple or the minimum-residual non-spanning tuple.
- **Verify** every closed-form decomposition independently. The seven
  qutrit identities are checked at three levels: numerical re-check at
  machine precision, exact rational re-check in
  [SymPy](https://www.sympy.org/) where available, and machine-checked
  [Lean 4](https://lean-lang.org/) +
  [mathlib](https://leanprover-community.github.io/) formalizations under
  `lean_proofs/`.
- **Search** the two-qutrit Clifford group $\mathrm{Cl}(2, 3)$ for injection
  gadgets via a reduction to the 51,840-element symplectic quotient
  $\mathrm{Sp}(4, \mathbb F_3)$.
- **Diagnose** with discrete-Wigner / mana routines and an exhaustive
  $F_{\max}$ solver tractable at $n \le 4$.

## State of the art

Best-known stabilizer-rank values and upper bounds across the qubit
and qutrit Clifford-inequivalent magic-state orbits, with citations and
provenance, are catalogued on the project page:

> **https://unitaryfoundation.github.io/stabrank/#soa**

## Installation

stabrank requires Python 3.12–3.14. Pre-built wheels are published on
PyPI for Linux, macOS, and Windows:

```bash
pip install stabrank
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install stabrank
```

To build from source (e.g.\ for development), clone the repo and
install in editable mode; the C++ core is compiled at install time via
[scikit-build-core](https://scikit-build-core.readthedocs.io/) and
[nanobind](https://nanobind.readthedocs.io/):

```bash
git clone https://github.com/unitaryfoundation/stabrank.git
cd stabrank
uv venv
uv pip install -e .
```

A working C++ toolchain is required (clang or g++); on macOS the Xcode
command-line tools are sufficient.

## Quick start

A two-term decomposition of $|\mathbb{S}\rangle^{\otimes 2}$ via the
simulated-annealing engine:

```python
import numpy as np
from stabrank.target_functions import qutrit_strange_state
from stabrank.stabrank_core import run_sa_pauli_expansion

target = qutrit_strange_state(2)              # 9-dim |S>^2
n, p, chi = 2, 3, 2

initial_basis = [
    np.ones(p**n, dtype=complex) / np.sqrt(p**n)
    for _ in range(chi)
]

_, basis, coeffs, residual, *_ = run_sa_pauli_expansion(
    target=target,
    n_orig=n,
    p_prime=p,
    k_subset_size=chi,
    initial_basis=initial_basis,
    num_chains=16,
    early_exit_threshold=1e-12,
)

print(f"|S>^2 ≈ sum of {chi} stabilizer states, residual = {residual:.2e}")
```

For more involved workflows — the Norrell $m = 4$ decomposition, the
exhaustive $k$-triple search at $m = 3$, and the gadget existence check
on $\mathrm{Sp}(4, \mathbb F_3)$ — see the public API surface in the
`stabrank` package and the test suite under `tests/`.

## Citation

If you use stabrank in your work, please cite the arXiv preprint:

```bibtex
@misc{labib2026stabilizer,
      title={Stabilizer-rank bounds for qutrit magic-state orbits},
      author={Farrokh Labib and Vincent Russo},
      year={2026},
      eprint={TBA},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/TBA},
}
```

## Funding

This work was supported by the U.S. Department of Energy, Office of
Science, Office of Advanced Scientific Computing Research, Accelerated
Research in Quantum Computing under Award Number DE-SC0025336.

This material is also based upon work supported by the U.S. Department
of Energy, Office of Science, National Quantum Information Science
Research Centers, Quantum Science Center.

## Development

The test suite runs under `pytest`. `pytest` is declared as an
optional `test` extra (it is not a runtime dependency of the
library), so run it via `uv` with the extra enabled:

```bash
uv run --extra test python -m pytest -q tests
```

## License

[Apache-2.0](LICENSE)
