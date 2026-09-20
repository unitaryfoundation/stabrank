# M5 progress report, draft

Milestone M5 of the AutoQEC Genesis Phase I project (Task 3, stabilizer-rank
optimization) reads: "Operate the stabilizer-rank autoresearch loop on
magic-state decompositions; match or improve the current best exponent; issue
a progress report", with completion evidence "reproducible baseline, automated
search data, exponent comparison, and report" (roadmap deck, slide 6; proposal
`main.tex`, milestone table, row M5-7). This draft is written against the
repository at commit `3570405` (2026-09-18). Every number below is taken from
a file in the repository, named in the text or the table; nothing is
estimated. Where two files disagree, both values are given.

Two states of the repository are referred to throughout, because they differ:

| state | bounds | what it is |
|---|---|---|
| committed ledger | 52 | `docs/ledger.json` and the site under `docs/`, last rebuilt at commit `c4ee903` |
| submission files | 54 | `bounds/*.json` at `3570405`; `N-m4-lower-5` (PR #42) and `T3-m3-lower-7` (PR #41, #49) have no matching receipt or ledger row yet |

A `--no-verify` site build at `3570405` reports "52 bounds, 12 lean-certified
(16 claim a proof), 19 verified, 21 reproduced, 0 exponents beaten, 5
contributors" and skips the two unledgered bounds. A full `make build` would
verify them (the N certificate is 43 minutes on the CI runner) and would also
move four cells to the `lean` tier whose Lean modules were added after the
last rebuild (section 6).

## 1. What the board measures

For a single-qudit magic state `|M>` and `m` copies, `chi(|M>^{ot m})` is the
smallest number of stabilizer states whose complex span contains
`|M>^{ot m}`, and `gamma = log_p(r)/m` is the per-copy exponent implied by a
rank-`r` decomposition at `m` copies, with `p = 2` for qubits and `p = 3` for
qutrits (`CONTRIBUTING.md`). Lower `gamma` is better. The exponent is what
sets the cost of the stabilizer-decomposition simulators the proposal cites,
which is why Task 3 is stated in terms of it (`main.tex`, section "Task 3:
Stabilizer Rank Optimization").

Six Clifford-inequivalent orbits are open, each measured against a published
exponent (`CONTRIBUTING.md`; `site_challenge/build.py`, `BASELINE`;
`docs/refs.bib`):

| orbit | state | published gamma | reference |
|---|---|---|---|
| `S` | qutrit Strange | log_3(2)/2 = 0.3155 | arXiv:2605.28586 |
| `N` | qutrit Norrell | log_3(4)/3 = 0.4206 | arXiv:2605.28586 |
| `H3` | qutrit Hadamard eigenvector | log_3(4)/3 = 0.4206 | arXiv:2605.28586 |
| `T3` | qutrit T-type | 1/2 = 0.5000 | arXiv:2605.28586 |
| `qubit_H` | qubit H-type (edge) | log_2(3)/4 = 0.3962 | arXiv:2106.07740 |
| `qubit_T` | qubit Bravyi-Kitaev T-type (face) | log_2(3)/4 = 0.3962 | arXiv:2106.07740 |

A cell is one (orbit, m) pair and holds an interval `lower <= chi <= upper`.
Every entry is a JSON file under `bounds/` and earns one of four tiers
(`CONTRIBUTING.md`, `verify_challenge/stabrank_verify.py`):

| tier | what the pipeline checked |
|---|---|
| `lean` | a Lean 4 module builds (receipt under `certs/lean-*.json`) and its theorem is the bound as stated |
| `verified` | the decomposition was rebuilt in exact arithmetic and the identity holds on the nose, or a lower-bound certificate is exact throughout |
| `reproduced` | a certificate script ran under its budget (900 s default, 3600 s cap) and printed the declared claim |
| `cited` | attributed to the literature only; can never hold a record |

Upper bounds carry their decomposition term by term in the stabilizer
parametrisation `(k, x0, W, Q, l)`, so a term that is not a stabilizer state
cannot be written down, and coefficients are exact sympy expressions
(`CONTRIBUTING.md`, "Upper bounds"). Lower bounds carry a script that must
exit zero having printed a declared string (`CONTRIBUTING.md`, "Lower
bounds"). Every submission carries `provenance.compute` (CPU-hours,
wall-clock, runs, hardware, LLM usage), which the site aggregates into
`docs/ledger.json` (`schema/bound.schema.json`, `provenance.compute`).

This is the Task 3 substrate for three reasons the proposal and the goals
document name. The proposal says the LLM agent "iterates over candidate
decomposition strategies, evaluating each against the current best-known
exponent, and retains only improvements" and names `stabrank` as the baseline
codebase (`main.tex`, Task 3). The goals document asks for a discovery ledger
recording compute per run (G1) and for formal verification as the project's
signature, with the stabrank instruction to "displace at least one cited
literature value per orbit with a verified or lean entry, and close or narrow
the open T3 cell at m = 3" (G4) (`../genesis-grant/phase1_goals.html`). The
board's tiers, compute blocks, run log and report page are the artifacts
those goals name.

One ambiguity the goals document records is unresolved here: "the narrative
cites the qubit linear-code exponent (Qassim et al., about 0.40), the
stabrank board ranks orbit-based exponents, mostly qutrit. Which exponent
'matched or improved' (milestone M5 to M7) refers to has to be fixed before
the work starts" (`phase1_goals.html`, section 2). This report compares all
six orbits and treats both qubit orbits against log_2(3)/4, as
`site_challenge/report.py` (`PUBLISHED_REF`) does.

## 2. Reproducible baseline

Every published exponent in the table above is realised by a cell on the
board at a record tier, so no exponent rests on a citation
(`docs/evidence_index.json`, `exponents`; `docs/ledger.json`):

| orbit | published gamma | cell realising it | rank | tier | evidence |
|---|---|---|---|---|---|
| `S` | 0.3155 | m = 2 | 2 | lean | `LeanProofs.Stabilizer.IsStab`, `strange_m2_stabRank_le_two` |
| `N` | 0.4206 | m = 3 | 4 | lean | `LeanProofs.M3StabRank`, `norrell_m3_stabRank_le_four` |
| `H3` | 0.4206 | m = 3 | 4 | lean | `LeanProofs.M3StabRank`, `h3_m3_stabRank_le_four` |
| `T3` | 0.5000 | m = 2 | 3 | lean | `LeanProofs.T3M2StabRank`, `t3_m2_stabRank_le_three` |
| `qubit_H` | 0.3962 | not attained; best is m = 6, rank 6 | 6 | verified | `bounds/qubit_H-m6-upper-6.json`, `certs/qubit_H-m6-upper-6.json` |
| `qubit_T` | 0.3962 | m = 4 | 3 | lean | `LeanProofs.QubitTM4`, `stabilizer_rank_le_three` |

The 28 literature values the board was seeded with (`provenance.date` before
2026-09-01; `verify_challenge/seed_bounds.py`) are all machine-checked in
the committed ledger: 14 `verified`, 9 `lean`, 5 `reproduced`, 0 `cited`
(`docs/ledger.json`). The upper bounds that had been cited without a
witness were given one by the package's own annealer and exact refit
(`bounds/qubit_H-m2-upper-2.json`, `qubit_H-m3-upper-3`, `qubit_H-m4-upper-4`,
`qubit_H-m5-upper-6`, `qubit_H-m6-upper-7`, `T3-m1-upper-3`, `T3-m3-upper-8`,
`T3-m4-upper-9`, notes fields), and the five exhaustive lower bounds of
arXiv:2605.28586 were re-run from scratch as certificate scripts
(`bounds/S-m3-lower-4.json`, `N-m3-lower-4`, `H3-m3-lower-4`, `S-m4-lower-4`,
`qubit_H-m6-lower-4`, notes fields).

Tier totals in the committed ledger (`docs/ledger.json`):

| direction | lean | verified | reproduced | cited | total |
|---|---|---|---|---|---|
| upper | 9 | 19 | 0 | 0 | 28 |
| lower | 3 | 0 | 21 | 0 | 24 |
| all | 12 | 19 | 21 | 0 | 52 |

Reproducibility rests on four mechanisms, each with a file:

| mechanism | file | what it fixes |
|---|---|---|
| exact verification of every upper bound | `verify_challenge/stabrank_verify.py` | symbolic identity over the cyclotomic field, 60-digit fallback at tolerance 1e-45 |
| certificate scripts with budgets | `verify_challenge/cert_*.py`, `stabrank_verify.py` (`BUDGET_DEFAULT_S = 900`, `BUDGET_CAP_S = 3600`) | a lower bound is a re-runnable program, not a claim |
| receipts keyed to the content hash | `certs/<slug>.json`, `site_challenge/build.py` (`content_hash`) | any edit to a submission forces re-verification |
| CI with base-branch checks | `.github/workflows/submissions.yml`, `tests.yml` | a pull request cannot redefine the verifier or schema that judges it; the Lean library is rebuilt from the PR |

The exact commands, from a clean clone, are in `REPRODUCING.md`.

## 3. Exponent comparison

From `docs/evidence_index.json` (`exponents`), built from the committed
ledger. The gap is best minus published; a match at the same rank replaces a
cited value with a machine-checked one and does not move the exponent.

| orbit | published | reference | best on board | cell | tier | gap | verdict |
|---|---|---|---|---|---|---|---|
| `S` | 0.3155 | arXiv:2605.28586 | 0.3155 | chi <= 2 at m = 2 | lean | 0.0000 | matches |
| `N` | 0.4206 | arXiv:2605.28586 | 0.4206 | chi <= 4 at m = 3 | lean | 0.0000 | matches |
| `H3` | 0.4206 | arXiv:2605.28586 | 0.4206 | chi <= 4 at m = 3 | lean | 0.0000 | matches |
| `T3` | 0.5000 | arXiv:2605.28586 | 0.5000 | chi <= 3 at m = 2 | lean | 0.0000 | matches |
| `qubit_H` | 0.3962 | arXiv:2106.07740 | 0.4308 | chi <= 6 at m = 6 | verified | +0.0346 | trails |
| `qubit_T` | 0.3962 | arXiv:2106.07740 | 0.3962 | chi <= 3 at m = 4 | lean | 0.0000 | matches |

Five orbits match, one trails, none is beaten (`docs/report/index.html`,
"0 beaten, 5 matched, 1 trailing or open"; `docs/index.html` headline
counter).

### Why the gaps are what they are

The published exponents are attained at small m (m = 2 for `S` and `T3`,
m = 3 for `N` and `H3`, m = 4 for `qubit_T`), and the board's lower bounds now
show that most cells at or near those m cannot beat them. The cells that
could still beat an exponent within the cap m <= 8 are the ones the annealer
was run on, and at every one of them it stopped at a residual that is the
same across seeds or takes a few fixed values: an exact algebraic local
optimum of the one-term exchange landscape, not a near miss
(`autoresearch/kopt.py`, module docstring: "The annealer stops at
configurations whose residual is an exact algebraic number, the same one
across seeds: local optima of its exchange landscape"). The run log records
this directly (`autoresearch/runs.jsonl`, aggregated by
`autoresearch/summary.py`):

| cell (orbit, m, rank) | gamma if found | beats published | runs | CPU-h | distinct final residuals (rounded to 1e-4) |
|---|---|---|---|---|---|
| `T3` m = 3 r = 7 | 0.5904 | no (narrows the cell) | 15 | 1.090 | 0.0825, 0.0833, 0.0995, 0.1036, 0.1088, 0.1134, 0.1158, 0.1173 |
| `T3` m = 4 r = 8 | 0.4732 | yes | 8 | 1.407 | 0.3330, 0.3352, 0.3372, 0.3493, 0.3495, 0.3569, 0.3662 |
| `H3` m = 4 r = 6 | 0.4077 | yes | 3 | 0.402 | 0.1841 (all three seeds) |
| `H3` m = 4 r = 7 | 0.4428 | no | 4 | 0.577 | 0.1042 (all four seeds) |
| `qubit_H` m = 6 r = 5 | 0.3870 | yes | 4 | 0.466 | 0.1516 (all four seeds) |
| `qubit_T` m = 5 r = 5 | 0.4644 | no | 4 | 0.269 | 0.1212, 0.1282 |
| `qubit_T` m = 6 r = 5 | 0.3870 | yes | 4 | 0.465 | 0.2363 (all four seeds) |

The seeded literature values carry the same kind of plateau in their notes,
with the residual given in closed form (`verify_challenge/seed_bounds.py`;
`bounds/*.json`, `notes`): `S` m = 6 rank 7 at sqrt(8/27) = 0.5443, `N` m = 4
rank 6 at sqrt(211/4043) = 0.2284, `H3` m = 4 rank 6 at
sqrt(70 - 37 sqrt 3)/12 = 0.2027, `qubit_T` m = 6 rank 5 at
sqrt(5/6) sin(pi/12) = 0.2363. The `qubit_T` value agrees with the run log to
four digits. The `H3` m = 4 rank 6 value does not: the log's three seeds all
stop at 0.1841, below the 0.2027 the note gives as the plateau. One of the two
is wrong or they measure different things; this is flagged in section 7.

Per orbit:

`S`, `N`, `H3`, `T3`: the exponent is matched at the cell where the
literature attains it, in Lean. Beating it needs a rank below the product
bound at a larger m. The routes at m <= 5 that the board has closed by lower
bounds are listed in section 4; the ones still open are `S` m = 5 rank 5
(gamma 0.2930; the cell is 5 <= chi <= 8), `N` m = 4 ranks 5 and 6 (0.3662,
0.4077; the cell is 5 <= chi <= 7 in `bounds/`, 3 <= chi <= 7 in the
committed ledger), `H3` m = 4 ranks 3 to 6 (cell 3 <= chi <= 8), `T3` m = 4
ranks 6 to 8 (cell 6 <= chi <= 9) and `T3` m = 5 ranks 6 to 15 (cell
6 <= chi <= 18). The annealer plateaus at `H3` m = 4 rank 6 and `T3` m = 4
rank 8 are in the table above. The `T3` m = 5 upper bound of 18 came from a
new construction rather than from the annealer at full size (the Z^5
eigensector decomposition, each sector a 4-qutrit carry state annealed to rank
6; rank 5 in every sector would give chi <= 15 and gamma 0.4930, the first
value below 1/2 for this orbit; `bounds/T3-m5-upper-18.json`, notes).

`qubit_T`: matched at m = 4 rank 3 (Lean, `LeanProofs.QubitTM4`). The m = 5
cell is closed as a route: chi(|T>^5) >= 4 by slice-and-lift
(`bounds/qubit_T-m5-lower-4.json`) and rank 4 at m = 5 gives 0.4000 > 0.3962.
Beating the exponent needs rank 5 at m = 6 (0.3870), where the annealer stops
at 0.2363 on every seed, or a cell at m = 7 or 8, which the board does not yet
hold.

`qubit_H`: trails by 0.0346. The exponent log_2(3)/4 is not attained at any
finite m on the board: chi(|H>^4) = 4 is settled in both directions
(`bounds/qubit_H-m4-lower-4.json`, `qubit_H-m4-upper-4.json`), so rank 3 at
m = 4 is impossible for this orbit, unlike `qubit_T`. The best cell is the
rank-6 cat-state decomposition of Qassim, Pashayan, and Gosset at m = 6
(0.4308), which the board holds as a verified witness built from their three
terms (`bounds/qubit_H-m6-upper-6.json`, notes). The annealer did not find it
unaided: two runs from rank 6 stopped at residual 0.101
(`bounds/qubit_H-m6-upper-7.json`, notes), and four runs at rank 5 stop at
0.1516 (`runs.jsonl`). Matching the exponent within the cap would need rank 9
at m = 8 (2^{8 x 0.3962} = 9); the board has no `qubit_H` cell above m = 6.

## 4. New results beyond the literature

The literature the board was seeded with carries lower bounds at five cells
only (`verify_challenge/seed_bounds.py`, `LIT`: `S` m = 3 and m = 4, `N`
m = 3, `H3` m = 3, `qubit_H` m = 6) and no lower bound at all for `T3` or
`qubit_T`. Everything in the following table is new to the board, dated by
`provenance.date` and located by the pull request that brought it to main
(`docs/evidence_index.json`, `bounds`). CPU-hours are the submission's
`provenance.compute.cpu_hours`; a blank means the file carries no compute
block.

### Lower bounds

| date | cell | bound | method | tier | PR | CPU-h | file |
|---|---|---|---|---|---|---|---|
| 2026-09-07 | `T3` every m | chi >= 3 | Galois descent over Z[w3], exact; also in Lean for every m | lean | #16, #17 | | `bounds/T3-m2-lower-3.json`, `LeanProofs.T3GaloisM` |
| 2026-09-07 | `T3` m = 3 | chi >= 6 | Galois descent to 45 candidate lines, then exhaustion | reproduced | #16, #17 | | `bounds/T3-m3-lower-6.json` |
| 2026-09-12 | `H3`, `N` m = 2 | chi >= 3 | exhaustion over 64,620 pairs | reproduced | #18 | | `bounds/H3-m2-lower-3.json`, `N-m2-lower-3.json` |
| 2026-09-12 | `qubit_H` m = 3, 4; `qubit_T` m = 4 | chi >= 3 | rank-2 exhaustion via the quotient by the target | reproduced | #18 | | `bounds/qubit_H-m3-lower-3.json`, `qubit_H-m4-lower-3.json`, `qubit_T-m4-lower-3.json` |
| 2026-09-15 | `S` m = 2 | chi >= 2 | affine-support obstruction, in Lean | lean | #19 | | `bounds/S-m2-lower-2.json`, `LeanProofs.Stabilizer.IsStab` |
| 2026-09-16 | `T3` m = 1 | chi >= 3 | 66-pair exhaustion; Galois argument in Lean | lean | #19 | | `bounds/T3-m1-lower-3.json`, `LeanProofs.T3M1StabRank` |
| 2026-09-16 | `qubit_H` m = 4 | chi >= 4 | rank-3 exhaustion over 36,720 states; settles chi(|H>^4) = 4 | reproduced | #24 | 0.3 | `bounds/qubit_H-m4-lower-4.json` |
| 2026-09-16 | `qubit_T` m = 3 | chi >= 3 | rank-2 exhaustion; settles chi(|T>^3) = 3 | reproduced | #24 | 0.001 | `bounds/qubit_T-m3-lower-3.json` |
| 2026-09-16 | `T3` m = 4 | chi >= 6 | projection monotonicity from m = 3 | reproduced | #24 | 0.2 | `bounds/T3-m4-lower-6.json` |
| 2026-09-18 | `H3`, `N` m = 4 | chi >= 3 | rank-2 exclusion over all 7,439,040 four-qutrit states held as phase codes | reproduced | #38 | 0.05 each | `bounds/H3-m4-lower-3.json`, `N-m4-lower-3.json` |
| 2026-09-18 | `S` m = 5 | chi >= 4 | projection monotonicity from m = 3 | reproduced | #37 | 0.01 | `bounds/S-m5-lower-4.json` |
| 2026-09-18 | `qubit_T` m = 5 | chi >= 3 | projection monotonicity from m = 4 | reproduced | #37 | 0.01 | `bounds/qubit_T-m5-lower-3.json` |
| 2026-09-18 | `T3` m = 5 | chi >= 6 | projection monotonicity applied twice | reproduced | #30 | 0.15 | `bounds/T3-m5-lower-6.json` |
| 2026-09-18 | `qubit_T` m = 5 | chi >= 4 | slice-and-lift from the single rank-3 decomposition of |T>^4 | reproduced | #39 | 0.01 | `bounds/qubit_T-m5-lower-4.json` |
| 2026-09-18 | `S` m = 5 | chi >= 5 | slice-and-lift twice, via 27 rank-4 decompositions of |S>^4 | reproduced | #40 | 2.7 | `bounds/S-m5-lower-5.json` |
| 2026-09-18 | `N` m = 4 | chi >= 5 | slice-and-lift from the single rank-4 decomposition of |N>^3; declared budget 3600 s | not yet in ledger | #42 | 7.6 | `bounds/N-m4-lower-5.json` |
| 2026-09-16 (file), merged 2026-09-18 | `T3` m = 3 | chi >= 7 | Galois descent to Q(w3), exhaustive coplanarity scan of 259,423 six-state classes, ranks exact mod 2^31 - 1; declared `exact: true` | not yet in ledger (would be verified) | #41, #49 | 0.15 | `bounds/T3-m3-lower-7.json` |

The two methods behind the largest moves are documented in
`verify_challenge/`: projection monotonicity, chi(psi (x) phi) >= chi(psi)
when phi has a nonzero computational amplitude (`cert_s_m4_from_m3.py`), and
slice-and-lift, which raises a lower bound by one when chi(|M>^m) = r is known
exactly and no rank-r decomposition of |M>^m, listed up to unitary symmetry,
extends to |M>^{m+1} by Paulis and roots of unity (`slice_lift.py`, module
docstring; `CONTRIBUTING.md`, "Lower bounds").

Cells settled in both directions by these results, with the tier of each
side (`docs/evidence_index.json`, `cells`, `settled: true`):

| cell | chi | lower tier | upper tier |
|---|---|---|---|
| `S` m = 2 | 2 | lean | lean |
| `S` m = 3 | 4 | reproduced | lean |
| `S` m = 4 | 4 | reproduced | verified |
| `N` m = 2 | 3 | reproduced | lean |
| `N` m = 3 | 4 | reproduced | lean |
| `H3` m = 2 | 3 | reproduced | lean |
| `H3` m = 3 | 4 | reproduced | lean |
| `T3` m = 1 | 3 | lean | verified |
| `T3` m = 2 | 3 | lean | lean |
| `qubit_H` m = 3 | 3 | reproduced | verified |
| `qubit_H` m = 4 | 4 | reproduced | verified |
| `qubit_T` m = 3 | 3 | reproduced | verified |
| `qubit_T` m = 4 | 3 | reproduced | lean |

Thirteen of 27 cells are settled (`docs/report/index.html`, "27 cells, 13
settled"). The goals document's G4 target for the board, "close or narrow the
open T3 cell at m = 3 (currently 6 to 8)", is met in full: the cell is
closed at chi(T3^3) = 8 (`bounds/T3-m3-lower-8.json`, attested tier, PR #62,
2026-09-20: an exact three-pivot scan of every rank-7 configuration after
Galois descent, 1,213,458,815 candidate class sets over 459 stored batches,
80.1 CPU-hours), against the verified eight-term witness.

### Upper bounds

| date | cell | bound | gamma | method | tier | PR | file |
|---|---|---|---|---|---|---|---|
| 2026-09-16 | `qubit_T` m = 2 | chi <= 2 | 0.5000 | annealer, exact refit; settles chi(|T>^2) = 2 | verified | #23 | `bounds/qubit_T-m2-upper-2.json` |
| 2026-09-16 | `qubit_T` m = 3 | chi <= 3 | 0.5283 | annealer, exact refit | verified | #23 | `bounds/qubit_T-m3-upper-3.json` |
| 2026-09-17 | `T3` m = 5 | chi <= 18 | 0.5262 | Z^5 eigensector (qutrit cat-state) decomposition, exact coefficients over Q(w9) | verified | #28 | `bounds/T3-m5-upper-18.json` |
| 2026-09-18 | `S` m = 5 | chi <= 8 | 0.3786 | product of board witnesses | verified | #37 | `bounds/S-m5-upper-8.json` |
| 2026-09-18 | `qubit_T` m = 5 | chi <= 6 | 0.5170 | product of board witnesses | verified | #37 | `bounds/qubit_T-m5-upper-6.json` |

None of these moves an exponent; they fill cells so that the interval on
every cell up to m = 5 has both sides.

### Lean tier growth

The `README.md` describes the Lean development as "seven qutrit identities"
checked pointwise. At `3570405` the library has 29 source files, 24 of them
imported by the library root (`lean_proofs/LeanProofs.lean`,
`lean_proofs/LeanProofs/`), 16 bounds name a
module and theorem, and 12 hold the `lean` tier in the committed ledger
(section 6). The development moved from pointwise identities between
hand-written vectors to statements about `Stabilizer.stabRank` against a
concrete predicate `IsStab` (`lean_proofs/LeanProofs/Stabilizer/IsStab.lean`,
header comment), which is the difference between "these vectors satisfy this
identity" and "the stabilizer rank is at most k". Lower bounds in Lean are
new: `strange_m2_stabRank_gt_one` (affine support) and `t3M_stabRank_gt_two`
for every m (Galois descent), giving `strange_m2_stabRank_eq_two` and
`t3_m2_stabRank_eq_three` as single theorems (`lean_proofs/README.md`;
`bounds/S-m2-lower-2.json`, `T3-m2-lower-3.json`, notes). Dates from the git
history: `IsStab` and the T3 Galois bound on 2026-09-16 (commits `6035640`,
`0749519`), `T3` m = 2 as one theorem on 2026-09-17 (`02333ce`), the m = 2, 3,
4 upper bounds against `stabRank` and the tensor-product lemma on 2026-09-18
(`28a4a7e`, `2483c15`, `0f03d79`, `9a5de54`).

## 5. Automated search data

### What is recorded

`autoresearch/run.py ORBIT M RANK` is one iteration of search, exact refit,
verification and cost recording. Every annealing run, successful or not,
appends one JSON line to `autoresearch/runs.jsonl` with the fields `when`,
`orbit`, `m`, `rank`, `seed`, `chains`, `iters`, `cooling`, `wall_s`,
`cpu_s`, `residual`, `solved`, `exact`, `hardware`, `llm`, `warm_from`, and
`refit` when the exact refit fails (`autoresearch/run.py`, `main`). A run
that reaches the rank is refit exactly by `verify_challenge/to_witness.py`
and written as a submission whose compute block sums every logged run for
that cell (`autoresearch/README.md`). The log is committed and never edited
(`autoresearch/README.md`, "The log is committed").

`autoresearch/loop.py run MANIFEST` runs many iterations unattended from a
JSON manifest (cells, seed ranges, annealer settings, priorities, wall-clock
cap per job), one job at a time under `nice -n 19` with `OMP_NUM_THREADS=1`,
round-robin over cells. It checkpoints to `autoresearch/state/<manifest>.json`
after every job, resumes, replays a recorded sequence, and writes one line per
job to `autoresearch/loop.log` with start and end time, wall clock, CPU time
from `getrusage`, exit code, the last eight lines of stderr and an outcome
class (`discovery`, `miss`, `near_miss`, `verifier_refusal`, `timeout`, `oom`,
`exception:<Type>`, `infrastructure:<Type>`, `killed:<SIG>`). Infrastructure
failures are retried once; a search miss never is. `loop.py status --since H`
prints hours covered, the fraction of that time a job was running and the
longest gap between jobs, which is the evidence for the 72-hour
autonomous-loop metric (`autoresearch/README.md`, "The unattended loop").
No `loop.log` or `state/` directory is committed, and `loop.py status` reports
"no jobs logged"; the four runs in `runs.jsonl` dated 2026-09-19 UTC with
`chains: 1, iters: 20` are the example manifest's smoke test
(`autoresearch/manifests/example.json`). The 72-hour run has not been made.

### What the log holds

`autoresearch/runs.jsonl` at `3570405` (aggregated by
`autoresearch/summary.py`; the same numbers appear in
`docs/evidence_index.json`, `cost_by_cell`):

| quantity | value |
|---|---|
| runs | 49 |
| cells | 13 |
| dates | 2026-09-18 and 2026-09-19 (UTC) |
| CPU-hours | 4.690 |
| wall-clock hours | 1.191 |
| runs that reached the rank | 3 |
| exact refits | 2 |
| runs with an `llm` field set | 0 |
| CPU-hours in runs that did not reach the rank | 4.675 (46 runs) |

The three runs that reached the rank re-found bounds already on the board:
`qubit_H` m = 4 rank 4 on seed 1 (48.8 CPU-s, exact refit), `qubit_T` m = 6
rank 6 from a warm start at the existing witness (0.0 CPU-s recorded), and the
`T3` sector state at m = 2 rank 3 (3.7 CPU-s; sector runs are logged, not
submitted). No new bound came from the annealer through `run.py`. Per cell:

| cell | runs | CPU-h | exact | best residual | on the board |
|---|---|---|---|---|---|
| `H3` m = 4 r = 6 | 3 | 0.402 | 0 | 0.1841 | no |
| `H3` m = 4 r = 7 | 4 | 0.577 | 0 | 0.1042 | no |
| `T3` m = 2 r = 1 | 2 | 0.000 | 0 | 0.7018 | no (smoke test) |
| `T3` m = 3 r = 7 | 15 | 1.090 | 0 | 0.0825 | no |
| `T3` m = 4 r = 8 | 8 | 1.407 | 0 | 0.3330 | no |
| `T3sector0` m = 2 r = 2 | 1 | 0.000 | 0 | 0.4714 | no (smoke test) |
| `T3sector0` m = 2 r = 3 | 1 | 0.001 | 0 | 0.0000 | sector, not submitted |
| `qubit_H` m = 2 r = 1 | 1 | 0.000 | 0 | 0.5210 | no (smoke test) |
| `qubit_H` m = 4 r = 4 | 1 | 0.014 | 1 | 0.0000 | verified |
| `qubit_H` m = 6 r = 5 | 4 | 0.466 | 0 | 0.1516 | no |
| `qubit_T` m = 5 r = 5 | 4 | 0.269 | 0 | 0.1212 | no |
| `qubit_T` m = 6 r = 5 | 4 | 0.465 | 0 | 0.2363 | no |
| `qubit_T` m = 6 r = 6 | 1 | 0.000 | 1 | 0.0000 | verified |

### Cost per discovery

The report page joins the run log with the compute blocks on the bounds and
dates every record-tier bound by its provenance date
(`site_challenge/report.py`, `cost_curve`; `docs/evidence_index.json`,
`cost_curve`). Literature values dated before 2026-09-01 are folded into a
seeded first row.

| date | bounds | CPU-h that day | median declared CPU-h | cumulative bounds | cumulative CPU-h |
|---|---|---|---|---|---|
| 2026-09-01 and earlier (seeded) | 28 | 0.60 | 0.050 | 28 | 0.60 |
| 2026-09-07 | 2 | 0.00 | | 30 | 0.60 |
| 2026-09-12 | 5 | 0.00 | | 35 | 0.60 |
| 2026-09-15 | 1 | 0.00 | | 36 | 0.60 |
| 2026-09-16 | 6 | 0.52 | 0.010 | 42 | 1.12 |
| 2026-09-17 | 1 | 0.60 | 0.600 | 43 | 1.72 |
| 2026-09-18 | 9 | 7.67 | 0.010 | 52 | 9.39 |
| 2026-09-19 | 0 | 0.00 | | 52 | 9.39 |

Of the 12.453 CPU-hours declared across the 26 submission files that carry a
compute block (`bounds/*.json`, `provenance.compute`), 7.6 are the
`N-m4-lower-5` enumeration and 0.15 the `T3-m3-lower-7` scan, neither yet in
the ledger; with them the cumulative figure is 17.1 CPU-hours for 54 bounds.
Bounds without a compute block (the seeded Lean and product witnesses, the PR
#16 to #19 certificates) count as discoveries at no recorded cost, which the
report page states (`docs/report/index.html`, "Cost per discovery").

Two readings of these numbers, both from the files above. First, since
2026-09-07 the board gained 24 record-tier bounds for 8.8 logged CPU-hours,
and every one of them came from an exclusion, a lift, an exact algebraic
argument or a product of existing witnesses; none came from the annealer
finding a new decomposition. Second, of the 4.69 CPU-hours in the run log,
99.7 percent went into 46 runs that stopped on a plateau. The plateaus are
reproducible across seeds and, where a closed form is recorded, exact
algebraic numbers (section 3), which is why `autoresearch/kopt.py` exists: it
tests exhaustively whether any k terms of a plateau configuration can be
replaced to reach the target, and for the `T3` m = 3 rank-7 plateaus it finds
no two-term completion (`bounds/T3-m3-lower-7.json`, notes: "an exact
swap-neighbourhood search around the eight-term witness (three or more shared
states) finds no seven-state span containing V_3"). Read together, the run log
says that further random-seed annealing at these cells is not where the next
bound is, and that the cost-per-discovery curve for this board is a curve of
certificate costs, not of annealing costs.

LLM attribution: one bound records model use, `T3-m5-upper-18`
(`claude-fable-5-1`, "designed the sector decomposition and drove the search";
`bounds/T3-m5-upper-18.json`, `provenance.compute.llm`). No line in
`runs.jsonl` has `llm` set, so token counts are not recorded anywhere yet.

## 6. Formal verification status

Toolchain `leanprover/lean4:v4.29.1` with mathlib `v4.29.1`
(`lean_proofs/lean-toolchain`, `lakefile.toml`). Receipts under
`certs/lean-<module>.json` are written only by `verify_challenge/lean_certify.py`
after a real `lake build` of the module exits zero; the site trusts the
receipt rather than rebuilding (`CONTRIBUTING.md`, "The Lean tier"). CI
rebuilds the whole library from every pull request with a 30-minute timeout
(`.github/workflows/tests.yml`, job `lean-build`). Nineteen receipts are
committed, all `ok: true` (`certs/lean-*.json`).

Bounds that name a Lean theorem (`bounds/*.json`, `lean`; tier from
`docs/ledger.json`):

| bound | module | theorem | statement kind | tier in ledger |
|---|---|---|---|---|
| `S-m1-upper-2` | `M1StabRank` | `strange_m1_stabRank_le_two` | stabRank | verified (lean pending rebuild) |
| `S-m2-lower-2` | `Stabilizer.IsStab` | `strange_m2_stabRank_gt_one` | stabRank, lower | lean |
| `S-m2-upper-2` | `Stabilizer.IsStab` | `strange_m2_stabRank_le_two` | stabRank | lean |
| `S-m3-upper-4` | `M3StabRank` | `strange_m3_stabRank_le_four` | stabRank | lean |
| `S-m4-upper-4` | `TensorStabRank` | `strange_m4_stabRank_le_four` | stabRank, tensor lemma | verified (lean pending rebuild) |
| `S-m6-upper-8` | `TensorStabRank` | `strange_m6_stabRank_le_eight` | stabRank, tensor lemma | verified (lean pending rebuild) |
| `N-m2-upper-3` | `M2StabRank` | `norrell_m2_stabRank_le_three` | stabRank | lean |
| `N-m3-upper-4` | `M3StabRank` | `norrell_m3_stabRank_le_four` | stabRank | lean |
| `N-m4-upper-7` | `M4StabRank` | `norrell_m4_stabRank_le_seven` | stabRank | lean |
| `H3-m2-upper-3` | `M2StabRank` | `h3_m2_stabRank_le_three` | stabRank | lean |
| `H3-m3-upper-4` | `M3StabRank` | `h3_m3_stabRank_le_four` | stabRank | lean |
| `T3-m1-lower-3` | `T3M1StabRank` | `t3_m1_stabRank_gt_two` | stabRank, lower | lean |
| `T3-m1-upper-3` | `M1StabRank` | `t3_m1_stabRank_le_three` | stabRank | verified (lean pending rebuild) |
| `T3-m2-lower-3` | `T3GaloisM` | `t3_m2_stabRank_gt_two` | stabRank, lower, every m | lean |
| `T3-m2-upper-3` | `T3M2StabRank` | `t3_m2_stabRank_le_three` | stabRank | lean |
| `qubit_T-m4-upper-3` | `QubitTM4` | `stabilizer_rank_le_three` | pointwise identity | lean |

The four "pending rebuild" rows have `ok: true` receipts for their modules
(`certs/lean-LeanProofs-M1StabRank.json`,
`lean-LeanProofs-TensorStabRank.json`) but were given their Lean claim after
the last site build, so the committed ledger and their `certs/<slug>.json`
receipts (now stale against the content hash) still say `verified`. A
`make build` moves them to `lean`, for 16 lean-tier bounds.

`QubitTM4.stabilizer_rank_le_three` is a different kind of statement from the
other fifteen. It proves that three explicitly written amplitude functions
`sigma1Amp`, `sigma2Amp`, `sigma3Amp` on `Fin 4 -> Fin 2` combine to the
target at every point (`lean_proofs/LeanProofs/QubitTM4.lean`, "Layer 5:
Assembly"); Lean does not check that the three functions are stabilizer
states, and the file does not use `Stabilizer.stabRank`. The board's
`verified` witness for the same cell supplies that check in exact arithmetic
(`bounds/qubit_T-m4-upper-3.json`, notes). The notes on `N-m4-upper-7` say
"With this every lean-tier upper bound on the board is a theorem about
stabRank"; with `QubitTM4` on the board that sentence is not accurate.

What a generic-qudit extension would add. The rank definition is fixed to
qutrits: `QutritVec n := Fin (3 ^ n) -> C` and `stabVecN` takes its phase
mod 3 (`lean_proofs/LeanProofs/Stabilizer/Rank.lean`, `IsStab.lean`). A
predicate parametrised by the local dimension `p`, with the qubit phase group
being the fourth roots of unity as in the verifier's witness format
(`verify_challenge/stabrank_verify.py`, module docstring), would let
`qubit_T-m4-upper-3` be restated against `stabRank`, would let the tensor
lemma of `TensorStabRank` carry the qubit product witnesses
(`qubit_T-m5-upper-6`, `qubit_T-m6-upper-6`), and would put the ten
`verified` qubit upper bounds within reach of the `lean` tier by the same
route the qutrit ones took. It would not by itself add any lower bound; Lean
lower bounds beyond `S` m = 2 and the `T3` Galois family need a formalised
exhaustive search or a new argument (`bounds/S-m2-lower-2.json`, notes:
"Lower bounds are universally quantified, which is why the rest of the Lean
development proves only upper bounds").

## 7. Risks and open items

Open exponent targets. The cells the manifest attacks
(`autoresearch/manifests/plateau_cells.json`) and the bound notes name two
that would move an exponent at a cost the board can pay:

| target | what it would give | evidence so far | source |
|---|---|---|---|
| `T3` five-copy Z-eigensector at rank 5 (81-dimensional carry state; `T3sector0` m = 4 rank 5 in `run.py`'s convention) | chi(|T3>^5) <= 15, gamma 0.4930, first value below 1/2; 0.4912 at m = 8 by contraction | rank 6 found in every sector; rank 5 failed in five runs at residuals 0.455 to 0.494 (`bounds/T3-m5-upper-18.json`, notes); "twelve seeds in 2026-09 stopped on five exact plateau values between 0.448 and 0.464" (`plateau_cells.json`, note). Neither batch is in `runs.jsonl` | `bounds/T3-m5-upper-18.json`, `autoresearch/manifests/plateau_cells.json` |
| `qubit_H` m = 6 rank 5 | gamma 0.3870, below log_2(3)/4 | four seeds, all at 0.1516 (`runs.jsonl`); cell is 4 <= chi <= 6 | `autoresearch/manifests/plateau_cells.json`, `bounds/qubit_H-m6-lower-4.json` |

Both notes recommend a lower-bound exclusion inside the relevant space
rather than more annealing (`bounds/T3-m5-upper-18.json`: "a rank-4 exclusion
inside the 81-dimensional code is the natural companion").

`T3` m = 3 is closed: chi = 8 (`bounds/T3-m3-lower-8.json`). The rank-7
exclusion cost 80.1 CPU-hours, between the 70 CPU-hours the rank-7 bound
file estimated and the 500 to 770 the project progress note of 18 September
quoted; the difference against the latter is the orbit-block pivot order of
`docs/notes/t3_rank7_exclusion.md`, a factor 4.4, and the kernel's 21 ns per
step. The bound sits on the attested tier because no certificate budget
re-runs the enumeration; the certificate re-runs two of the 459 batches and
hashes the rest (58 minutes on the CI runner).

Ledger and receipt staleness. The committed `docs/ledger.json` and site
predate PRs #41, #42, #48 and #49: two bounds are missing and four tiers are
low (sections 2 and 6). Eight receipts under `certs/` no longer match their
submission's content hash (`docs/evidence_index.json`, `receipt_matches:
false`: `S-m1-upper-2`, `S-m4-upper-4`, `S-m5-lower-5`, `S-m6-upper-8`,
`T3-m1-lower-3`, `T3-m1-upper-3`, `T3-m2-lower-3`, `T3-m3-lower-7`), in each
case because the file was edited (a `notes`, `lean`, `certificate` or
`compute` field) after its receipt was written. The next `make build`
re-verifies those files. The six that name a Lean module take the `lean`
tier from their module receipt without running anything
(`verify_challenge/stabrank_verify.py`, `verify`); the other two re-run
their certificates, about two and three minutes, and the missing
`N-m4-lower-5` receipt costs `cert_n_m4_lift.py`, 43 minutes on the 4-core
CI runner under the declared 3600 s budget (`bounds/N-m4-lower-5.json`,
compute block and `certificate.budget_s`).

Margin-based certificates. Of the 24 lower bounds in the ledger, 21 are
`reproduced`, meaning the exclusion rests on a floating-point parallelism
margin that the script reports and refuses to claim below 0.01
(`verify_challenge/rank_exclusion.py`, module docstring; bound notes). Only
the `T3` Galois bounds and `T3-m3-lower-7` are exact throughout. The
`reproduced` tier is honest about this, but a reviewer should know that
"settled" cells outside `S` m = 2 and `T3` m = 1, 2 have one exact side and
one margin side.

Plateau value disagreement. `H3` m = 4 rank 6: the seeded note gives the
plateau as sqrt(70 - 37 sqrt 3)/12 = 0.2027 (`bounds/H3-m4-upper-8.json`,
notes; `stabrank/examples/README.md`, "plateau certificates"), while the three
logged runs all stop at 0.1841 (`autoresearch/runs.jsonl`). If the residual
convention is the same in both places, the run log has found a better rank-6
approximation than the one the examples README describes as conjecturally
optimal.

Orbit-page target lines. Each orbit page carries a "What would move it"
sentence that `site_challenge/build.py` (`next_target`) fills with the
lowest-gamma cell in m <= 8, which is "chi <= 2 at m=2" for `T3`, `N` and
`H3` and "chi <= 2 at m=4" for both qubit orbits (`docs/orbits/*.html`). For
`T3` that cell is excluded by the board's own Lean theorem
`t3M_stabRank_gt_two`, and for `N` and `H3` by the m = 2 certificates. The
rendered sentence also reads "needs needs". `CONTRIBUTING.md` describes the
line as naming "the smallest rank at each m that would beat the published
exponent", which is not what the function computes.

Unlogged searches. The `T3` sector annealing behind `T3-m5-upper-18` (8 runs,
0.6 CPU-h) and the annealing that produced the seeded qubit witnesses predate
`run.py` and are recorded only in compute blocks, not in `runs.jsonl`. The
manifest's twelve-seed sector batch is recorded only in a note.

No 72-hour run. `loop.py` exists and is tested (`tests/test_autoresearch_loop.py`)
but has not been run for the metric; `autoresearch/loop.log` does not exist.

Timing reference. Every local timing in the bound notes was taken on a
laptop under load from other jobs; the notes say so where it matters
(`bounds/T3-m5-upper-18.json`, `N-m4-lower-5.json`). CI timings are the
reference where recorded.

Exponent target ambiguity. The proposal's "current best exponent" is not
fixed to a state (`phase1_goals.html`, section 2). On the board's reading, the
five matched orbits and the one trailing orbit are the comparison; on the
narrower qubit linear-code reading, the milestone would compare against
log_2(3)/4 for `qubit_H` alone, where the board trails by 0.0346 and the
literature's own value is asymptotic.

Documentation drift. `CONTRIBUTING.md` still says "the T3 cell at m=3 sits at
6 <= chi <= 8"; `README.md` says "seven qutrit identities" for the Lean
development; `lean_proofs/README.md` refers to `paper/main.tex`, which is not
in this repository.

## 8. Evidence map

| M5 evidence item | where it is |
|---|---|
| reproducible baseline: every published exponent as a record-tier cell | `docs/ledger.json`; `docs/report/index.html`, "Exponent comparison"; `bounds/S-m2-upper-2.json`, `N-m3-upper-4.json`, `H3-m3-upper-4.json`, `T3-m2-upper-3.json`, `qubit_T-m4-upper-3.json`, `qubit_H-m6-upper-6.json` |
| reproducible baseline: commands from a clean clone | `REPRODUCING.md` |
| reproducible baseline: verification receipts | `certs/<slug>.json` (43 files), `certs/lean-*.json` (19 files) |
| automated search data: run log | `autoresearch/runs.jsonl` (49 lines), `autoresearch/summary.py` |
| automated search data: loop, manifests, failure classes | `autoresearch/loop.py`, `autoresearch/manifests/plateau_cells.json`, `autoresearch/manifests/example.json`, `autoresearch/README.md` |
| automated search data: plateau completion test | `autoresearch/kopt.py` |
| exponent comparison | `docs/evidence_index.json`, `exponents`; `docs/report/index.html`; `site_challenge/report.py`, `exponent_comparison` |
| cell intervals with dates | `docs/evidence_index.json`, `cells`; `docs/report/index.html`, "Cell intervals" |
| cost per discovery | `docs/evidence_index.json`, `cost_curve`, `cost_by_cell`; `docs/report/index.html`, "Cost per discovery"; `bounds/*.json`, `provenance.compute` |
| provenance per bound: PR, merge commit, receipt, Lean module, compute | `docs/evidence_index.json`, `bounds`; `docs/report/index.html`, "Evidence index" |
| formal verification | `lean_proofs/`, `certs/lean-*.json`, `verify_challenge/lean_certify.py`, `.github/workflows/tests.yml` (`lean-build`) |
| new lower bounds | `bounds/*-lower-*.json` dated 2026-09-07 to 2026-09-18, `verify_challenge/cert_*.py`, `verify_challenge/slice_lift.py`, `rank_exclusion.py`, `rank_exclusion_codes.py` |
| report | this file; `docs/report/index.html` (regenerated by every site build) |
| roadmap items W18 (reproduce the baseline), W19 (rank harness with metadata and provenance), W28 to W30 (reruns, bounded exponent comparison, reviewed report) | sections 2, 5, 3 and this document respectively |
