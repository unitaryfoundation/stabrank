# Reproducing the board

Every command below is run from the repository root of a fresh clone. Each
step says what it needs beyond Python and `uv`: the `challenge` extra (sympy,
jsonschema, latex2mathml), the native extension (the C++ annealing and
pivot-search kernels, compiled on first `uv run`), or `elan` (the Lean
toolchain manager). Runtimes are quoted from the bound files' `notes` and
`compute` blocks, the CI workflow timeouts, `autoresearch/runs.jsonl`, or a
measurement in one session on an Apple silicon laptop, and each is labelled
with its source. Where no figure is recorded anywhere, the step says so.

## 1. Install

Requirements: Python 3.12 to 3.14, `uv`, and a C++20 toolchain (clang or g++;
on macOS the Xcode command-line tools) for the native extension
(`README.md`, "Installation"; `pyproject.toml`).

```
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/unitaryfoundation/stabrank.git
cd stabrank
uv run --extra challenge python -c "import stabrank.stabrank_core; print('ok')"
```

The first `uv run` creates `.venv`, installs the dependencies and compiles
the extension through scikit-build-core and nanobind (`pyproject.toml`,
`[build-system]`). Runtime: a few minutes, dominated by the compile; the
repository records no figure. Every later `uv run` reuses the build.

The `challenge` extra is needed by everything under `verify_challenge/`,
`site_challenge/` and `autoresearch/`; the `test` extra by `pytest`. Pass
both where a step needs both.

## 2. Validate and verify bounds

Schema validation of every submission (`challenge` extra):

```
uv run --extra challenge python verify_challenge/validate_bounds.py
uv run --extra challenge python verify_challenge/validate_bounds.py bounds/S-m2-upper-2.json
```

Runtime: not recorded; it is a JSON-schema pass over 54 files.

Verify one bound, or all of them (`challenge` extra; a lower bound's
certificate may also use the native extension, see section 6):

```
make verify BOUND=bounds/S-m4-upper-4.json
make verify
```

`make verify` runs `verify_challenge/stabrank_verify.py` on the files given
(`Makefile`). For an upper bound it rebuilds every term from its
`(k, x0, W, Q, l)` parametrisation and the target in exact arithmetic and
prints `PASS` with the tier and the implied gamma, or `FAIL`. For a lower
bound it runs the certificate script under its budget (section 3). Nothing
is written; receipts are written by the site build (section 5).

Runtime of an upper-bound check: not recorded in the repository. Runtime of
`make verify` over all 54 files: not recorded as a total; it is the sum of
the certificate runtimes in the table in section 3, and is dominated by
`cert_n_m4_lift.py` at 43 minutes on the 4-core CI runner
(`bounds/N-m4-lower-5.json`, compute block).

Recover exact coefficients for a proposed decomposition before submitting
(`challenge` extra):

```
make fit BOUND=bounds/yours.json
make fit BOUND=bounds/yours.json ARGS=--write
```

## 3. Run a certificate under its budget

A lower bound names a script and the string it must print
(`bounds/<slug>.json`, `certificate`). The verifier runs it with the
repository root as working directory under `budget_s` seconds of wall clock,
900 by default and at most 3600 when the submission declares more
(`verify_challenge/stabrank_verify.py`, `BUDGET_DEFAULT_S`, `BUDGET_CAP_S`).
Either run it through the verifier, which enforces the budget:

```
make verify BOUND=bounds/T3-m3-lower-7.json
```

or directly, which does not:

```
uv run --extra challenge python verify_challenge/cert_t3m3_rank7.py
```

The script must exit zero and its output must contain the declared string,
here `CERTIFIED chi(T3^3) >= 7`. Every certificate runs a positive control
first and exits non-zero without printing its claim if the control fails
(`CONTRIBUTING.md`, "Lower bounds").

Recorded runtimes, all from the bound files' `notes` or `compute` blocks
unless stated:

| certificate | bound | recorded runtime | hardware | source |
|---|---|---|---|---|
| `cert_t3_galois.py` | `T3-m2-lower-3` | not recorded (one symbolic rank of a 3-column matrix) | | `bounds/T3-m2-lower-3.json` |
| `cert_t3m1_rank2.py` | `T3-m1-lower-3` | not recorded (66 pairs) | | `bounds/T3-m1-lower-3.json` |
| `cert_m2_rank2.py` | `H3-m2-lower-3`, `N-m2-lower-3` | not recorded (64,620 pairs) | | `bounds/H3-m2-lower-3.json` |
| `cert_qubit_rank2.py` | `qubit_H-m3-lower-3`, `qubit_H-m4-lower-3`, `qubit_T-m4-lower-3` | not recorded | | |
| `cert_qubit_t_m3_rank2.py` | `qubit_T-m3-lower-3` | 0.001 CPU-h | Apple silicon laptop | compute block |
| `cert_s_m3_rank3.py`, `cert_n_m3_rank3.py`, `cert_h3_m3_rank3.py` | `S-m3-lower-4`, `N-m3-lower-4`, `H3-m3-lower-4` | 0.05 CPU-h, 0.05 wall-h each; "two minutes" for the Strange state | Apple silicon laptop, 8 worker processes | compute blocks; `CONTRIBUTING.md` |
| `cert_s_m4_from_m3.py`, `cert_s_m5_from_m3.py` | `S-m4-lower-4`, `S-m5-lower-4` | re-runs the m = 3 exclusion; 0.01 CPU-h recorded for m = 5 | Apple silicon laptop | compute block |
| `cert_qubit_h_m4_rank3.py` | `qubit_H-m4-lower-4` | 0.3 CPU-h, 0.08 wall-h | Apple silicon laptop, 8 workers | compute block |
| `cert_qubit_h_m6_from_m4.py` | `qubit_H-m6-lower-4` | 0.3 CPU-h, 0.08 wall-h | Apple silicon laptop, 8 workers | compute block |
| `cert_t3m3_rank6.py` | `T3-m3-lower-6` | "under three" minutes, down from 26 CPU-minutes | | `CONTRIBUTING.md`, script header |
| `cert_t3_m4_from_m3.py`, `cert_t3_m5_from_m3.py` | `T3-m4-lower-6`, `T3-m5-lower-6` | 0.2 and 0.15 CPU-h, 0.05 and 0.03 wall-h | Apple silicon laptop, 6 workers | compute blocks |
| `cert_h3_m4_rank2.py`, `cert_n_m4_rank2.py` | `H3-m4-lower-3`, `N-m4-lower-3` | "about two minutes", 600 MB | Apple silicon laptop, one core | notes, script headers |
| `cert_qubit_t_m5_from_m4.py` | `qubit_T-m5-lower-3` | 0.01 CPU-h | Apple silicon laptop | compute block |
| `cert_qubit_t_m5_lift.py` | `qubit_T-m5-lower-4` | "eight seconds" | | `CONTRIBUTING.md`, compute block 0.01 CPU-h |
| `cert_s_m5_lift.py` | `S-m5-lower-5` | "about two minutes on four cores" | Apple silicon laptop, low priority | notes |
| `cert_n_m4_lift.py` | `N-m4-lower-5` | 43 minutes on the 4-core CI runner with the numpy search; declared budget 3600 s | GitHub Actions `ubuntu-latest` | compute block, `certificate.budget_s` |
| `cert_t3m3_rank7.py` | `T3-m3-lower-7` | 99 s wall, 270 s CPU with 8 workers; about three minutes on the 4-core CI runner | Apple silicon laptop; CI | compute block |

Every certificate pins BLAS to one thread per worker
(`OMP_NUM_THREADS=1` at the top of the script) and picks its worker count
from `os.cpu_count()`, so wall-clock scales with cores up to the cap in the
script (4 for the lift certificates, 8 for `cert_t3m3_rank7.py`).

## 4. Tests

Python (`test` extra; the challenge tests skip themselves when sympy is
absent, so pass both extras to run them):

```
uv run --extra challenge --extra test python -m pytest -q tests
uv run --extra challenge --extra test python -m pytest -q tests/test_report.py tests/test_witness_format.py
```

Runtime: the two named files, 17 tests, took 27 s in one session on an Apple
silicon laptop under load. The whole suite is not timed in the repository;
CI allows 15 minutes per Python version (`.github/workflows/tests.yml`, job
`pytest`).

C++ (needs `cmake`, `ninja` and a C++20 compiler; no Python):

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Runtime: not recorded; CI allows 20 minutes (`tests.yml`, job `cpp-tests`).

## 5. Rebuild the site and the report page

Without running any certificate (`challenge` extra):

```
uv run --extra challenge python site_challenge/build.py --no-verify
```

This rebuilds every page under `docs/` from `bounds/`, the receipts in
`certs/` and the tiers in `docs/ledger.json`, then writes the report page
`docs/report/index.html` and `docs/evidence_index.json`
(`site_challenge/build.py`, `--no-verify` help; `site_challenge/report.py`,
`write_report`). A bound with neither a matching receipt nor a ledger row is
skipped with a message on stderr and shown as unverified in the evidence
index. The last line summarises the board, at commit `3570405`:
`docs/ written: 52 bounds, 12 lean-certified (16 claim a proof), 19 verified,
21 reproduced, 0 exponents beaten, 5 contributors`. Runtime: 9.5 s wall in
one session, after the venv existed.

With verification (`challenge` extra, native extension for the kernels the
certificates use):

```
make build
```

This runs the verifier on every bound whose receipt under `certs/` is
missing or does not match the file's content hash, writes the new receipts,
then builds the site (`Makefile`; `site_challenge/build.py`, `load_bounds`).
Runtime: the sum of the runtimes, from the section 3 table, of the
certificates whose receipts are stale or missing. A bound that names a Lean
module with an `ok` receipt takes the `lean` tier before any certificate is
considered (`verify_challenge/stabrank_verify.py`, `verify`), so a stale
receipt on such a bound costs nothing. At `3570405` the bounds that would
actually run are `S-m5-lower-5` (about two minutes), `T3-m3-lower-7` (about
three minutes) and `N-m4-lower-5` (43 minutes on four CI cores), so roughly
50 minutes on a four-core machine; the total is not recorded.

The report's cost table can also be printed on its own:

```
uv run --extra challenge python autoresearch/summary.py
uv run --extra challenge python autoresearch/summary.py --since 2026-09-15
```

Runtime: seconds (reads `autoresearch/runs.jsonl` and `docs/ledger.json`).

To view the built site as deployed:

```
make serve
```

Do not commit `docs/` from a submission branch (`CONTRIBUTING.md`,
"Submitting").

## 6. Build the Lean library and write receipts

Needs `elan`, which installs the pinned toolchain `leanprover/lean4:v4.29.1`
from `lean_proofs/lean-toolchain` on first use, and about 5 to 10 GB for the
mathlib binary cache (`lean_proofs/README.md`, "Build"). No Python is needed
for the build itself; `make lean-certify` needs the `challenge` extra only
because it runs under `uv run`.

```
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
cd lean_proofs
lake exe cache get
lake build
cd ..
```

`lake build` compiles the whole `LeanProofs` library. Recorded per-module
times, after the cache is present: `StrangeM2Pointwise` about 7 s,
`H3M2Pointwise` about 13 s, `NorrellM2Pointwise` about 10 s,
`NorrellM4Pointwise` about 2.5 minutes (`lean_proofs/README.md`). The time
for the whole library is not recorded; CI allows 30 minutes for the job
including the cache download (`.github/workflows/tests.yml`, job
`lean-build`).

Receipts, which the site trusts instead of rebuilding:

```
make lean-certify
```

This collects every module named in a `lean` block under `bounds/`, runs
`lake build MODULE` for each with a 5400 s timeout, and writes
`certs/lean-<module>.json` with `ok: true` only when the build exits zero
(`verify_challenge/lean_certify.py`). Sixteen bounds name a module at
`3570405`; the receipt log lines read "Build completed successfully (3301
jobs)" and similar, where the job count includes cached mathlib jobs. Check
that the named theorem is the bound as stated; the receipt proves the module
compiles, not what it says (`CONTRIBUTING.md`, "The Lean tier").

## 7. Re-run an annealing search with a given seed

`autoresearch/run.py` anneals one cell at a fixed rank and appends one line
per seed to `autoresearch/runs.jsonl` (`challenge` extra, native extension):

```
uv run --extra challenge python autoresearch/run.py T3 3 7 --seeds 1 --seed0 1 --chains 8 --iters 8000 --cooling 0.995
```

The settings above are the defaults (`autoresearch/run.py`, `main`) and are
those of the logged run on seed 1, whose line is

```
{"chains": 8, "cooling": 0.995, "cpu_s": 350.9, "exact": null, "hardware": "arm64 Darwin, 18 cores", "iters": 8000, "llm": null, "m": 3, "orbit": "T3", "rank": 7, "residual": 0.10363319733400939, "seed": 1, "solved": false, "wall_s": 119.8, "when": "2026-09-18T09:35:38+00:00"}
```

(`autoresearch/runs.jsonl`). The console line was `seed 1: residual
1.036e-01 in 120s wall, 351s CPU`. What should match on a re-run: `residual`
to rounding, `solved` and `exact`. The initial basis is drawn after
`np.random.seed(seed)` and chain `i` of the C++ engine is seeded with
`seed + i * 1000003`, so the result does not depend on thread scheduling
(`autoresearch/run.py`, `anneal_once`; `cpp/src/sa_engine.cpp`). Last-digit
differences between compilers or CPUs are possible and are not measured in
the repository. What will differ: `when`, `wall_s`, `cpu_s`, `hardware`.
Runtime: 119.8 s wall and 350.9 s CPU for this cell on an 18-core Apple
silicon machine with 8 chains; 6.4 s wall and 48.8 s CPU for
`qubit_H 4 4 --seeds 1` (`autoresearch/runs.jsonl`, seed 1 lines).

A re-run appends to the committed log. The log is meant to hold every run
ever made (`autoresearch/README.md`), so the new line is a legitimate record;
commit it only if you mean to add to the record.

The package's own search driver, without the log:

```
uv run python stabrank/examples/search_decomposition.py --target qutrit_strange --n 4 --k-start 4 --seed 1
```

It writes `solution_<target>_k<k>_n<n>_<timestamp>.npz`, which
`verify_challenge/to_witness.py` turns into a submission:

```
uv run --extra challenge python verify_challenge/to_witness.py S 4 solution.npz -o bounds/S-m4-upper-4.json
```

Runtimes for these two: not recorded in the repository, other than the
compute blocks of the witnesses they produced (0.01 to 0.02 CPU-h each for
the qubit witnesses, `bounds/qubit_H-m*-upper-*.json`).

## 8. Run the slice-and-lift pipeline for a cell

The pipeline is `verify_challenge/slice_lift.py` (`challenge` extra; the
rank-4 pivot-pair search dispatches to the compiled kernel
`stabrank_core.rank4_pivot_partners` when the native extension is present,
and `STABRANK_NO_NATIVE=1` forces the numpy reference).

Its controls, which every lift certificate also runs:

```
uv run --extra challenge python verify_challenge/slice_lift.py
```

Runtime: not recorded.

The three cells it has certified, as certificates (runtimes in section 3):

```
uv run --extra challenge python verify_challenge/cert_qubit_t_m5_lift.py
uv run --extra challenge python verify_challenge/cert_s_m5_lift.py
uv run --extra challenge python verify_challenge/cert_n_m4_lift.py
```

For a new cell where `chi(|M>^m) = r` is known exactly, `lift_chain`
enumerates the rank-`r` decompositions of `|M>^m` up to unitary symmetry and
lifts them copy by copy to `m_target`, returning the count at each m and the
smallest miss among rejected Pauli assignments:

```
uv run --extra challenge python -c "
import sys; sys.path.insert(0, 'verify_challenge')
from slice_lift import lift_chain
counts, gap = lift_chain('N', 3, 4, 4, workers=4)
print(counts, gap)"
```

A count of zero at `m_target` is the lower bound `chi(|M>^{m_target}) >= r + 1`
(`verify_challenge/slice_lift.py`, module docstring, "Corollary"). The
enumeration cost depends on the symmetry group of the target: the first
`N` m = 3 enumeration took 6.4 hours single-core with the numpy search and
the certificate takes 43 minutes on four CI cores with the compiled kernel
(`bounds/N-m4-lower-5.json`, compute block). Do not use the antiunitary
symmetry for the orbit reduction; it mixes slices (`slice_lift.py`,
"Symmetry").

## 9. Run the unattended loop on a manifest

`autoresearch/loop.py` runs `run.py` job by job from a manifest, under
`nice -n 19` with `OMP_NUM_THREADS=1` (`challenge` extra, native extension):

```
uv run --extra challenge python autoresearch/loop.py run autoresearch/manifests/example.json
uv run --extra challenge python autoresearch/loop.py run autoresearch/manifests/plateau_cells.json --max-hours 72
uv run --extra challenge python autoresearch/loop.py run autoresearch/manifests/plateau_cells.json --resume
uv run --extra challenge python autoresearch/loop.py replay autoresearch/state/plateau_cells.json
uv run --extra challenge python autoresearch/loop.py status --since 72
uv run --extra challenge python autoresearch/loop.py run autoresearch/manifests/example.json --dry-run
```

The example manifest is three cells at rank 1 or 2 with one chain and 20
iterations, so every job ends as a miss; its four runs in the committed log
span 2026-09-19T00:06:01 to 00:06:09 UTC (`autoresearch/runs.jsonl`, last
four lines), so the whole manifest is under ten seconds. `plateau_cells.json`
is the four open cells with 32 to 48 seeds each and a 7200 s cap per job; its
runtime is bounded by `--max-hours`. Each job writes one line to
`autoresearch/loop.log` and a checkpoint to `autoresearch/state/<name>.json`;
neither is committed. `run.py` still appends to `runs.jsonl` for every job
(`autoresearch/README.md`, "The unattended loop").

## 10. Regenerate the report page alone

The report is written by every site build (section 5); there is no separate
entry point. `docs/report/index.html` and `docs/evidence_index.json` are the
outputs, and the page's own "Reproducing this page" section lists the same
commands (`site_challenge/report.py`, `report_page`). The git history the
evidence index reads (adding commit, merge commit, pull request per bound)
needs a full clone, not a shallow one (`site_challenge/report.py`,
`git_history`).

## 11. Seed a clean ledger from the literature

`verify_challenge/seed_bounds.py` writes the literature values as `cited`
submissions into `bounds/` and skips any file already present, so on a
populated clone it writes nothing:

```
make seed
```

## What each step needs

| step | `challenge` extra | native extension | `elan` and mathlib cache |
|---|---|---|---|
| install (section 1) | | built here | |
| validate, verify an upper bound (2) | yes | | |
| run a certificate (3) | yes | the lift certificates use the compiled pivot-pair kernel when present and fall back to numpy; `cert_t3m3_rank7.py` uses numba, a package dependency | |
| Python tests (4) | for the challenge tests | for `test_bindings.py` | |
| site build, `--no-verify` (5) | yes | | |
| site build with verification (5) | yes | as for certificates | |
| Lean build and receipts (6) | for `make lean-certify` only | | yes |
| `run.py`, `loop.py`, `search_decomposition.py` (7, 9) | yes | yes | |
| slice-and-lift (8) | yes | optional; `STABRANK_NO_NATIVE=1` forces numpy | |
