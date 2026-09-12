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

## Four tiers, and only the top three hold records

| tier | meaning |
|---|---|
| `lean` | a Lean module builds and its theorem is the bound as stated |
| `verified` | the pipeline rebuilt the decomposition and confirmed the identity |
| `reproduced` | a certificate script ran under the budget and asserted the bound |
| `cited` | attributed to the literature, not machine-checked here |

Seeded literature values populate the ledger so the picture is complete, and they can never crown a cell. The only way to take a record is to submit something the pipeline can check, which is the whole point of the distinction.

## Upper bounds

An upper bound carries its decomposition, and each term is given by its stabilizer parametrisation rather than as a raw amplitude vector:

    sum_{y in F_p^k} w_p^{Q(y) + l.y} |x0 + W y>

Here `k` is the dimension of the support flat, `x0` is a coset representative of length `m`, `W` has `k` rows of length `m` spanning the flat, `Q` is a quadratic form read upper-triangular, `l` is a linear phase of length `k`, and `w_p` is a primitive p-th root of unity. A vector that is not a stabilizer state cannot be written in this form at all, and `W` is separately checked for injectivity, so the verifier never has to decide whether a supplied amplitude vector is stabilizer.

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

The script runs under a 900-second budget. That is a real constraint on what you can submit, not a formality: an exhaustion written as a Python loop over 30240 states will not fit, and the fix is usually to restate the search rather than to ask for more time. `verify_challenge/cert_t3m3_rank6.py` is worth reading as an example, because it documents both the reformulation that brought 26 CPU-minutes down to under three and the more aggressive version of the same idea that silently loses configurations.

Write the script so it fails loudly. A certificate that prints its claim unconditionally is worse than no certificate, since it converts a bug into a board entry.

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

Bounds that match rather than beat the literature are still worth submitting, because a `cited` value carries no machine-checked evidence and a `verified` or `lean` one at the same rank replaces it with something the pipeline stands behind. Tightening a lower bound is equally welcome and usually more tractable: the T3 cell at `m=3` sits at `6 <= chi <= 8`, and both sides are open.

## Tips

- Cap is `m <= 8`, set by the verification budget rather than by the mathematics.
- `make build` rebuilds `docs/`, and `serve` hosts it at `http://localhost:8765/` so links and rendered math behave as deployed.
- Verification results are cached in `certs/` against the content hash of the whole submission file, so any edit re-verifies it, including one that only touches `notes`. Expect a rebuild to spend the budget again after a typo fix.
- The verifier escalates from symbolic to 60-digit numeric with a tolerance of `1e-45` when sympy cannot decide that a cyclotomic expression vanishes. If your coefficients defeat both, say so in `notes` rather than loosening the check.
