# Contributing a bound

You bring a decomposition or a certificate; the pipeline decides whether it holds, and the board shows only what it could check.

A submission is one JSON file in `bounds/` claiming a bound on the exact stabilizer rank of a magic state. Write `chi(|M>^{ot m})` for the smallest number of stabilizer states whose complex span contains the m-fold tensor power of the orbit state `|M>`, and `gamma = log_p(r)/m` for the per-copy exponent a rank-`r` decomposition at `m` copies implies, where `p` is 2 for qubits and 3 for qutrits. Lower `gamma` is better, and `gamma` is what the board ranks.

Six orbits are open, with the published exponent each one is measured against:

| orbit | state | published `gamma` |
|---|---|---|
| `S` | Strange | `log_3(2)/2` = 0.3155 |
| `N` | Norrell | `log_3(4)/3` = 0.4206 |
| `H3` | qutrit Hadamard eigenvector | `log_3(4)/3` = 0.4206 |
| `T3` | qutrit T-type, the face centre | `1/2` = 0.5000 |
| `qubit_H` | qubit H-type, the edge centre | `log_2(3)/4` = 0.3962 |
| `qubit_T` | qubit Bravyi-Kitaev T-type | `log_2(3)/4` = 0.3962 |

The qubit exponent is the asymptotic value of the contracted cat-state family of Qassim, Pashayan, and Gosset (arXiv:2106.07740), not the exponent of any single cell. Kissinger, van de Wetering, and Vilmart (arXiv:2202.09202) restate that family in the ZX-calculus and add a partial decomposition of `|T>^5` into three terms that each keep one `|T>`, so `chi(T^t) <= 3 chi(T^(t-4))`; on the board that gives `chi(H^7) <= 9`, `chi(H^8) <= 12`, and the glued `|cat_10>` gives `chi(H^10) <= 18` (0.4170), all as verified witnesses. The finite-`m` values approach `log_2(3)/4` from above and none of them beats it.

## Five tiers, and only `cited` holds no record

| tier | meaning |
|---|---|
| `lean` | a Lean module builds and its theorem is the bound as stated |
| `verified` | the pipeline rebuilt the decomposition and confirmed the identity, or a lower-bound certificate is exact throughout, with no floating-point margin anywhere |
| `reproduced` | a certificate script ran under the budget and asserted the bound |
| `attested` | the argument is exact but rests on an offline enumeration no budget can re-run; the certificate hashed every stored batch output, re-decided the stored exceptions exactly and re-ran a declared subset of batches bit for bit |
| `cited` | attributed to the literature, not machine-checked here |

Seeded literature values populate the ledger so the picture is complete, and they can never crown a cell. The only way to take a record is to submit something the pipeline can check, which is the whole point of the distinction. When two bounds on a cell have the same rank, the higher tier holds it, so an `attested` bound gives way to a `reproduced` or `verified` one at the same rank.

## Upper bounds

An upper bound carries its decomposition, and each term is given by its stabilizer parametrisation rather than as a raw amplitude vector:

    sum_{y in F_p^k} w_p^{Q(y) + l.y} |x0 + W y>

Here `k` is the dimension of the support flat, `x0` is a coset representative of length `m`, `W` has `k` rows of length `m` spanning the flat, `Q` is a quadratic form read upper-triangular, `l` is a linear phase of length `k`, and `w_p` is a primitive p-th root of unity. For qubits the phase is `i^(l.y) (-1)^Q(y)` instead, with `l` read mod 4, since qubit stabilizer states carry fourth roots of unity and `w_2 = -1` alone would miss the Y eigenstates. A vector that is not a stabilizer state cannot be written in this form at all, and `W` is separately checked for injectivity, so the verifier never has to decide whether a supplied amplitude vector is stabilizer.

If you have the decomposition as amplitude vectors, from the annealer or by hand, `verify_challenge/to_witness.py` recovers the parametrisation of each term and the exact coefficients and writes the submission skeleton:

```
uv run --extra challenge python verify_challenge/to_witness.py S 4 solution.npz -o bounds/S-m4-upper-4.json
```

It refuses any vector that is not a stabilizer state, and it reads either an `.npz` written by `stabrank/examples/search_decomposition.py` or a JSON list of amplitude vectors.

Coefficients are exact sympy expressions as strings, never floats: `"3/4 + sqrt(3)*I/4"`, not `"0.75 + 0.433*I"`. The verifier rebuilds every term and the target and requires the identity to hold on the nose. A floating-point near-miss earns nothing, and the schema check rejects anything that looks like a decimal.

Solve for the coefficients rather than writing them by hand:

```
make fit BOUND=bounds/yours.json
```

This takes the terms you supplied, recovers the unique exact coefficients from the normal equations, and either prints them or tells you that no exact combination of those terms reproduces the target. Pass `ARGS=--write` to fill them into the file. Running it first turns a "close" decomposition into either a submission or a clear no, without burning a review.

## Lower bounds

A lower bound cannot be settled by a static witness, so it carries a certificate script that must exit zero having printed a claim string you declare in the submission:

```json
"certificate": {
  "script": "verify_challenge/cert_t3_galois.py",
  "expect": "CERTIFIED chi(T3^m) >= 3"
}
```

The script runs under a 900-second budget. That is a real constraint on what you can submit, not a formality: an exhaustion written as a Python loop over 30240 states will not fit, and the fix is usually to restate the search rather than to ask for more time. When the search is already restated as tightly as you can and still does not fit, the submission may declare its own budget, `"budget_s": 3600` at most, inside the `certificate` block. The declared value is shown on the board next to the tier and in the ledger, so the cost of the bound stays visible; it is not a way to skip the restating. `cert_n_m4_lift.py` is the first user: the Norrell state's symmetry group is small, so listing the rank-4 decompositions of |N>^3 takes tens of minutes on four cores even after the reduction that brought the strange state's enumeration from hours to two minutes. `verify_challenge/cert_t3m3_rank6.py` is worth reading as an example, because it documents both the reformulation that brought 26 CPU-minutes down to under three and the more aggressive version of the same idea that silently loses configurations.

A certificate that is exact throughout, so that no floating-point margin stands anywhere under the bound, may declare `"exact": true` and then earns `verified` rather than `reproduced`. The pipeline cannot check that declaration from outside, any more than it can check that the claim string follows from what the script computed, so both are what review of the script is for. The bar is the whole argument, not the last step: a search that harvests candidates numerically and then rejects each one exactly does not qualify, because the completeness of the candidate list still rests on the harvest's margin, and that margin is what a missed decomposition would hide behind. An argument like the Galois descent for T3, computed as an exact rank, does qualify. Say in `notes` which steps are exact and which rest on a margin, whether or not you claim the flag.

Two ways to reach a lower bound without a fresh exhaustion at the target size are in `verify_challenge/`. Projection monotonicity, chi(psi (x) phi) >= chi(psi) when phi has a nonzero computational amplitude, carries any lower bound up in m unchanged (`cert_s_m4_from_m3.py`). Slice-and-lift (`slice_lift.py`) raises it by one when chi(|M>^m) = r is known exactly: a rank-r decomposition of |M>^(m+1) sliced along one qudit gives, at every level, a rank-r decomposition of |M>^m whose terms are related slice to slice by a Pauli and a root of unity, so if no rank-r decomposition of |M>^m (listed up to the unitary symmetries by a pivot search) admits such Paulis, chi(|M>^(m+1)) >= r + 1. `cert_qubit_t_m5_lift.py` is the smallest case: the single rank-3 decomposition of |T>^4 up to symmetry does not extend, so chi(|T>^5) >= 4 in eight seconds. `cert_s_m5_lift.py` applies it twice, from the 15 rank-4 decompositions of |S>^3 through the 69 of |S>^4 to none of |S>^5, in about two minutes on four cores; the partner loop of its pivot-pair search is reduced by the pivot's stabilizer subgroup, which is what brought a 2.2-hour enumeration under the budget.

Write the script so it fails loudly. A certificate that prints its claim unconditionally is worse than no certificate, since it converts a bug into a board entry.

## Attested offline enumerations

Some exclusions are exact and machine-checked at every step and still cannot run under any budget the board allows. The rank-7 exclusion for T3 at `m=3` (`docs/notes/t3_rank7_exclusion.md`) is the case that forced the question: the arithmetic is exact throughout, but the enumeration behind it is about 90 CPU-hours, and the 3600-second cap is not a rounding error away from that. Claiming `verified` for it would misstate what the pipeline checked, since `verified` means the pipeline confirmed the argument, and here it confirms the decision and a sample of the enumeration. Claiming `reproduced` would be wrong for the same reason. The `attested` tier is for exactly this situation and for nothing else: an enumeration that fits the budget after restating belongs at `reproduced` or `verified`, and a numerical search that is merely long does not qualify, because the tier is about where the completeness argument lives, not about how slow the script is.

The enumeration runs offline, in batches, with a committed runner, and every batch writes a result file into the repository. The submission then declares, inside the `certificate` block,

```json
"attested": {
  "batches": "research/t3_rank7/results/manifest.json",
  "recomputed": 12,
  "compute_hours": 91,
  "hardware": "Apple M3 Max, 14 cores",
  "note": "completeness rests on the stored batch outputs; research/t3_rank7/batch.py regenerates any of them in the recorded time"
}
```

`batches` is a JSON manifest listing every batch: a list of `{"id", "params", "output", "sha256"}` entries, with `output` the stored result file relative to the repository root and `sha256` its digest. Before the script runs, the verifier requires the manifest to exist and every listed output to be present with the stated hash, and fails the bound otherwise. That check is cheap and it is the whole reason the outputs are committed: a result file that has changed since it was recorded is not the enumeration the bound rests on. `recomputed` is the number of batches the script re-runs from scratch, and `compute_hours` and `hardware` record what the offline run cost; `provenance.compute` is required alongside, so the ledger carries the cost. `exact: true` is an error next to `attested`, since the pipeline did not confirm the whole argument. `budget_s` may be declared as usual and will typically be the cap.

The certificate must do four things under the budget. Rebuild whatever the batches depend on (the dictionary, the symmetry data, the projection) so that the re-run is a re-run and not a replay. Check that the batch files cover the search space exactly once, with matching code and data hashes, and that no batch reports an unresolved case. Re-decide every stored exception exactly, by the same arithmetic the runner used. Choose `recomputed` batches deterministically from a seed, print that seed on a line `seed: <value>`, re-run them from scratch and compare their outputs bit for bit with the stored ones. Print the claim string only if all four pass. The verifier's detail then says what was re-run and what was only hashed, and the board pill shows the re-run fraction, so no reader mistakes the tier for a full reproduction.

Say in `notes` what the completeness rests on, in those words: that the enumeration was not re-run under the budget, that it is attested by the stored outputs and the committed runner, and how a reader would regenerate any batch or all of them. Record the offline cost in `provenance.compute` with the hardware, dates and the git hash of the runner.

`attested` holds records. The alternative would leave the board showing a weaker lower bound than the repository holds, on a cell where the bound is exact, every step of the argument is machine-checked, and any batch can be regenerated by anyone with the runner and the recorded CPU time. That is machine-checked evidence in a way a citation is not, and the tier sits below `reproduced` so that a certificate which re-runs the argument in full at the same rank displaces it. The trust it asks for is the same kind the other lower-bound tiers already ask for: the pipeline cannot audit that a script's claim follows from what it computed, or that an `exact` flag is honest, and both are settled at review. What the tier adds is a stored enumeration that review can spot-check and the verifier re-runs in part, chosen by a seed the script prints. What it does not add is a guarantee against a batch output that is wrong but consistently hashed; the deterministic subset re-run bounds how many such batches can hide, and the committed runner is how a reader closes the gap for good.

The tier is meant to be temporary for any given bound. A later full re-run, a smarter certificate that fits the budget, or a Lean proof should replace an `attested` bound at the higher tier, and the equal-rank rule above makes that replacement automatic once the new bound is on the board.

## The Lean tier

A bound whose statement is proved in `lean_proofs/` names the module and theorem:

```json
"lean": {"module": "LeanProofs.H3M3Pointwise", "theorem": "h3_m3_vector_decomposition"}
```

The site does not rebuild mathlib on every run. It trusts build receipts under `certs/`, written by `make lean-certify`, and a module whose build fails has no receipt and so cannot claim the tier. Check that the theorem really is the bound as stated, with no hypothesis that quietly weakens it; the receipt proves the module compiles, not that it says what you meant.

## Say how you got it

Every submission carries `provenance` and a free-text `notes` field, capped at 2000 characters, and both are rendered on the bound's own page.

```json
"provenance": {
  "author": "Your Name",
  "reference": "arXiv:2605.28586",
  "method": "one line on how the bound was obtained",
  "date": "2026-09-07",
  "github": ["yourhandle"]
}
```

`date` is when the bound first appeared: the arXiv v1 date for a literature value, otherwise the date of the work. It orders the recent-submissions list and the record-progress chart, so a placeholder there is not harmless. `github` binds the entry to its authors.

`provenance.compute` records what the search cost, and every field in it is optional:

```json
"compute": {
  "cpu_hours": 3.5,
  "gpu_hours": 0,
  "wall_clock_hours": 1.2,
  "runs": 40,
  "hardware": "Apple M3 Max, 14 cores",
  "llm": [{"model": "claude-fable-5-1", "input_tokens": 1200000,
           "output_tokens": 85000, "role": "drove the search"}],
  "cost_usd": 12.40
}
```

Count every run that led to the bound, including the ones that failed, and count what the search spent rather than what the verifier spends checking it. The site collects these into `docs/ledger.json`, one row per submission with its tier and date, which is the data behind a cost-per-discovery curve. That curve cannot be reconstructed after the fact from a board that only records results, so report the cost at the time even when it is small; a certificate that took ten CPU-minutes to find is a data point, and a blank is not.

Use `notes` for what the fields cannot hold: the search that found it, the margin by which the check passed, what you ruled out on the way, and any respect in which the result is weaker than it looks. A route you can show is barren is worth recording next to the bound it failed to improve. The board's value compounds through shared search experience, and the alternative is that every contributor rediscovers the same dead end at their own expense.

## Submitting

Validate and verify locally first. Both run in CI and there is no reason to learn the outcome from a red check:

```
make verify BOUND=bounds/yours.json
uv run --extra challenge python verify_challenge/validate_bounds.py bounds/yours.json
```

Then open a pull request adding your file under `bounds/`. Do not commit `docs/`; the site is rebuilt from `bounds/` and a hand-edited page will conflict. Open the pull request from your own account so the `github` handles on the submission match its author.

CI validates every submission against `schema/bound.schema.json` and runs the verifier on the files the pull request touched. The schema and the verifier are taken from the base branch rather than from the pull request, so a submission cannot redefine the checks that judge it. A lower bound is the exception in kind, since its certificate script is the submission: for a branch in this repository the script runs from the pull request, and for a fork it is reported missing and left for a maintainer to run.

If you need to change the verifier, the schema, or the site builder, do that in a pull request separate from any submission.

## The autoresearch loop

`autoresearch/run.py ORBIT M RANK` is one iteration of search, exact refit, verification and cost recording. Every annealing run it makes, successful or not, is appended to `autoresearch/runs.jsonl` with its configuration, seed, wall-clock, CPU time and residual; a run that reaches the rank is refit exactly and written as a submission whose `compute` block sums every logged run for that cell. `autoresearch/summary.py` prints CPU-hours per exact solution by cell, joined with the board's tiers. Use it rather than the bare annealer when the cost of the search is part of the result, which on this board it always is.

`autoresearch/loop.py run MANIFEST` runs many such iterations unattended: a JSON manifest lists cells, seed ranges, annealer settings, priorities and a wall-clock cap per job, and the loop runs one seed at a time under `nice -n 19` with a single thread, round-robin over the cells so none starves. It checkpoints after every job to `autoresearch/state/`, continues with `--resume`, re-runs a recorded sequence with `replay`, classifies every failure from the exit code and stderr into `autoresearch/loop.log`, retries once on infrastructure failures and never on a search miss, and `loop.py status --since 72` prints the uptime covered and the longest gap between jobs. `autoresearch/manifests/plateau_cells.json` is the current set of open cells; `example.json` finishes in seconds. The loop never writes `runs.jsonl` itself, so the run ledger keeps its one-line-per-annealing-run contract.

## The progress report

`site_challenge/report.py` writes `docs/report/index.html` and `docs/evidence_index.json` on every build. The page compares the best exponent on the board with the published one per orbit, lists the interval `lower <= chi <= upper` on every cell with the date each side last moved, draws cumulative CPU-hours against cumulative record-tier bounds from the run log and the compute blocks, and indexes the evidence behind every bound, with a tally by tier: the pull request and merge commit that brought it to main, its receipt under `certs/`, its Lean module, its `attested` block and the manifest's batch count when it has one, and its compute block. The JSON file holds the same data. To rebuild the page without running any certificate:

```
uv run --extra challenge python site_challenge/build.py --no-verify
```

This uses the receipts in `certs/` and `docs/ledger.json` as they are; a bound with neither is left off the board and shown as unverified in the evidence index. The default `make build` re-verifies every bound whose receipt is missing or stale.

## Contributing with an LLM

An agent can do the whole loop: pick a cell, search for a decomposition, verify it, and open the pull request. Paste the prompt below.

This applies when you have explicitly asked the agent to submit on your behalf. You are then the author of record, and the `github` handles on the submission are yours. Without that, have the agent leave the file in your working tree and open the pull request yourself.

```
You are contributing to the Stabilizer Rank Challenge, a leaderboard of exact
stabilizer-rank bounds on magic states. Goal: find a bound the pipeline can
verify, and submit it.

1. Clone https://github.com/unitaryfoundation/stabrank and read
   CONTRIBUTING.md, schema/bound.schema.json, and the orbit pages under
   https://unitaryfoundation.github.io/stabrank/orbits/. Each orbit page has a
   "what would move it" line naming the smallest rank at each m that would beat
   the published exponent. Pick one of those cells.

2. Search for a decomposition of |M>^{ot m} into that many stabilizer states.
   Parametrise each term as (k, x0, W, Q, l) rather than as an amplitude
   vector, so every candidate is a stabilizer state by construction, and
   enumerate over the flats and phase forms rather than over vectors.

3. Screen candidates numerically, then confirm in exact arithmetic. A
   least-squares residual near zero is not a bound: verification requires the
   identity to hold exactly over the relevant cyclotomic field, and decimal
   coefficients are rejected outright. Recover exact coefficients with
   verify_challenge/fit_coeffs.py, which returns them or tells you no exact
   combination of your terms works.

4. Before trusting a search that found nothing, run it against a cell whose
   answer is already on the board and confirm it recovers the known
   decomposition. Until it does, a null result is evidence about the search,
   not about the rank, and is not submittable as a lower bound.

5. Verify and validate locally:
     make verify BOUND=bounds/yours.json
     uv run --extra challenge python verify_challenge/validate_bounds.py bounds/yours.json
   Fill in provenance.method and notes: the search you ran, its depth, what
   collapsed, and how close the near-misses were. Do not commit docs/.

Report the orbit and m, the rank, the implied gamma against the published
exponent for that orbit, and whether the identity was confirmed exactly or only
numerically. If only numerically, say so plainly rather than submitting.
```

## What makes a submission interesting

A bound matters when it moves a cell. Each orbit page names the cheapest cell that would beat its published exponent, which is more actionable than a leaderboard position; as of this writing no published exponent has been beaten, and the headline counter says so.

Bounds that match rather than beat the literature are still worth submitting, because a `cited` value carries no machine-checked evidence and a `verified` or `lean` one at the same rank replaces it with something the pipeline stands behind. Tightening a lower bound is equally welcome and usually more tractable: the T3 cell at `m=3` sits at `7 <= chi <= 8` after the exact rank-6 exclusion, and rank 7 is the one value left to settle.

## Tips

- Cap is `m <= 10`, set by the verification budget rather than by the mathematics; the qubit cells at `m = 10` verify in seconds, and a qutrit witness of that size would not.
- `make build` rebuilds `docs/`, and `serve` hosts it at `http://localhost:8765/` so links and rendered math behave as deployed.
- Verification results are cached in `certs/` against the content hash of the whole submission file, so any edit re-verifies it, including one that only touches `notes`. Expect a rebuild to spend the budget again after a typo fix.
- The verifier escalates from symbolic to 60-digit numeric with a tolerance of `1e-45` when sympy cannot decide that a cyclotomic expression vanishes. If your coefficients defeat both, say so in `notes` rather than loosening the check.
