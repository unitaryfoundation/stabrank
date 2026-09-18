# Lean 4 formalization of m ∈ {2, 3, 4} appendix identities

Scope: machine-verify the closed-form χ ≤ k stabilizer decompositions of
`paper/main.tex` Appendix A at m ∈ {2, 3, 4}, using Lean 4 and mathlib4.

These files are an audit artifact for the appendix identities, not a
replacement for the mathematical exposition in the paper. The paper-level
mapping is:

| Paper location | Lean files |
| --- | --- |
| Appendix A.1, Strange m=2,3 | `StrangeM2Pointwise.lean`, `StrangeM3.lean`, `StrangeM3Pointwise.lean` |
| Appendix A.2, H_3 m=2,3 | `H3M2Pointwise.lean`, `H3M3.lean`, `H3M3Pointwise.lean` |
| Appendix A.3, Norrell m=2,3,4 | `NorrellM2Pointwise.lean`, `NorrellM3.lean`, `NorrellM3Pointwise.lean`, `NorrellM4Pointwise.lean` |

## What's here

- `LeanProofs/Basic.lean` — universal foundation. Defines the cube root
  of unity and its algebraic relations, used by all downstream files.
- `LeanProofs/H3Shared.lean` — H_3-orbit constants and helper lemmas
  (`cH3C`, `NH3C`, `h3Amp1`, `sqrt3_sq_cH`, `NH3C_sq`, `NH3C_ne_zero`,
  `sqrt3_ne_zero`). Shared by `H3M2Pointwise.lean` and
  `H3M3Pointwise.lean` so neither has to depend on the other.
- `LeanProofs/NorrellShared.lean` — Norrell-orbit helpers (`norrellAmp1`,
  `isTwo`, `sqrt2_sq_cN`, `sqrt3_sq_cN`). Shared by all three Norrell
  pointwise files (m=2, m=3, m=4) so they form a flat DAG rooted at
  `NorrellShared` rather than a chain m=2 → m=3 → m=4.

  Basic.lean theorems: `omega3_pow_three` (ω³ = 1), `omega3_re` (Re(ω)
  = -1/2), `omega3_im` (Im(ω) = √3/2), `omega3_sq_add_omega3_add_one`
  (ω² + ω + 1 = 0), `omega3_add_omega3_sq` (ω + ω² = -1),
  `omega3_sq_eq` (ω² = -1 - ω), `omega3_diff_eq_I_sqrt3` (ω - ω² = i·√3).
  All proofs complete (no `sorry`).

- `LeanProofs/StrangeM2Pointwise.lean` — paper App A.1.1 vector-level lift.
  Proves the m=2 χ ≤ 2 pointwise identity
  `[|S⟩^⊗2]_y = -(i√3/2)·ω·([S_1]_y - [S_2]_y)` over all 9 y ∈ F_3².
  Five trivial cases close by `simp`; the four non-trivial cases all
  reduce (after `field_simp`) to one `linear_combination` (or its
  negation) over `{hI, h2, h3, hΩcyc, h2ω}`. Coefficients derived
  analytically; build ≈ 7 s.

- `LeanProofs/H3M2Pointwise.lean` — paper App A.2.1 vector-level lift.
  Proves the H_3 m=2 χ ≤ 3 pointwise identity
  `[|H_3⟩^⊗2]_y = α_1·[S_1]_y + α_2·[S_2]_y + α_3·[S_3]_y` over all
  9 y ∈ F_3², with α_j = (c√3/N²)(1 − cω^j) etc. The 9 cases dispatch
  via 3 `linear_combination` patterns (one per (a, b) ∈ {0,1}² class,
  with two pairs sharing). Coefficients derived analytically from the
  factored residuals; build ≈ 13 s.

- `LeanProofs/NorrellM2Pointwise.lean` — paper App A.3.1 vector-level lift.
  Proves the Norrell m=2 χ ≤ 3 pointwise identity
  `[|N⟩^⊗2]_y = α·[S_1]_y + [S_2]_y + ᾱ·[S_3]_y`
  (α = −ω/2, ᾱ = −ω²/2; |S_2⟩ = |2,2⟩) over all 9 y ∈ F_3². The cases
  dispatch via 4 `linear_combination` patterns keyed by
  (Q_1(y), [y = (2,2)]) over `{h2, h3, hΩsum, hΩ3}`; build ≈ 10 s.

- `LeanProofs/StrangeM3.lean` — paper App A.1.2.
  Proves the two scalar identities the appendix proof collapses to:
  - `strange_m3_alpha_identity`: α(ω² - 1) = -3√2/4
  - `strange_m3_beta_identity`:  β · i√3   =  3√2/4

  Both proofs are complete (no `sorry`). The α identity goes via the
  Cartesian form 2ω = -1 + i√3 plus the ring relations i² = -1 and
  (√3)² = 3, packaged as a single `linear_combination`. The β identity
  is a direct algebraic simplification using i² = -1 and √6·√3 = 3√2.

- `LeanProofs/StrangeM3Pointwise.lean` — paper App A.1.2 vector-level lift.
  Lifts the scalar identities in `StrangeM3.lean` to the full pointwise
  amplitude equality

      [|S⟩^⊗3]_y  =  α · ([S_1]_y - [S_3]_y)  +  β · ([S_2]_y - [S_4]_y)

  over all 27 indices y ∈ F_3³. Defines amplitude functions for |S⟩^⊗3
  and the four 3-qutrit stabilizer states (using their canonical-form
  quadratic forms Q_1, Q_3, Q_2, Q_4 mod 3), then proves the identity
  pointwise via `fin_cases y_0 <;> fin_cases y_1 <;> fin_cases y_2`.
  Of the 27 cases, 19 are trivially zero on both sides (closed by `simp`);
  the 8 non-trivial cases close via `field_simp` + a `linear_combination`
  over `strange_m3_alpha_identity`, `strange_m3_beta_identity`, the
  sqrt-square identity, and the helper `ω - ω² = i·√3`.

- `LeanProofs/H3M3Pointwise.lean` — paper App A.2.2 vector-level lift.
  Defines the 1-qutrit H_3 amplitude h3Amp1, the 3-qutrit h3Amp3 tensor,
  the four stabilizer-state amplitudes (S_1, S_2 on full F_3³; S_3 = |0,0,+⟩;
  S_4 = |+,+,0⟩), the coefficients (c, N, α_1, α_2, α_34) with
  c = (√3-1)/2, N² = 3-√3, α_1, α_2 = (3c/4N)(1±i), α_34 = 3/(4N).
  Proves the **full pointwise identity** over all 27 indices y ∈ F_3³
  via `fin_cases × 3` + `field_simp` + a 6-way `first | A | … | F`
  `linear_combination` dispatch — one pattern per (n,z) class.
  Coefficients derived by symbolic Gröbner-basis ideal-membership against
  {h3, hN, hΩcyc, hI, h2ω}; the `h2ω : 2ω + 1 = i√3` helper is itself
  a `linear_combination omega3_diff_eq_I_sqrt3 + omega3_sq_eq`.
  Bumps maxHeartbeats to 3.2M because the LC coefficients are large
  polynomial expressions (each LC contributes ~12 monomials, 27 × 6
  candidate attempts).

- `LeanProofs/H3M3.lean` — paper App A.2.2.
  Proves the H_3 m=3 scalar identity:
  - `h3_m3_identity`: c · (1 + c) = 1/2 with c = (√3 - 1)/2

  A one-liner: `linear_combination (1/4) * (√3 · √3 = 3)`.

- `LeanProofs/NorrellM4Pointwise.lean` — paper App A.3.3 vector-level lift.
  Lifts the m=4 χ ≤ 7 decomposition to a full pointwise amplitude equality

      [|N⟩^⊗4]_y  =  α_0·[S_0]_y + α_1·[S_1]_y + … + α_6·[S_6]_y

  over all 81 indices y ∈ F_3⁴, with 7 stabilizer states and coefficients
  α_0 = α_4 = σω, α_1 = α_2 = α_3 = α_5 = σξ, α_6 = ω²/4 where
  σ = √3/4 and ξ = e^{iπ/6} = (√3 + i)/2. The Q_j polynomials are entered
  in their indicator-reduced forms (paper eq. norrell-m4-Qreductions).
  Proof: fin_cases × 4 → 81 cases collapse to 9 distinct cleared-form
  residuals; each closes via one of 9 `linear_combination` patterns
  dispatched by `first`. Coefficients derived by symbolic Gröbner-basis
  ideal-membership over {h2, h3, hΩcyc, hΩ3, hI, h2ω}. Bumps
  maxHeartbeats to 12.8M; build time ≈ 2.5 minutes.

- `LeanProofs/NorrellM3Pointwise.lean` — paper App A.3.2 vector-level lift.
  Lifts the scalar B(n₂) collapse in `NorrellM3.lean` to the full pointwise
  amplitude equality

      [|N⟩^⊗3]_y  =  α_12 · ([S_1]_y + [S_2]_y + [S_4]_y) + α_3 · [S_3]_y

  over all 27 indices y ∈ F_3³. Definitions use √2·√3 instead of √6
  throughout to keep all algebra in (√2, √3, ω). The proof dispatches all
  27 cases via `fin_cases × 3 ; simp ; field_simp ; first | A | B | C | D`,
  one `linear_combination` per n₂ ∈ {0, 1, 2, 3} class (the LHS depends
  only on n₂ via norrellAmp1's values, and the S_j sum collapses via the
  Basic ω-relations). Bumps `maxHeartbeats` to 800K (27 × 4 attempts in
  one `first` block exceeds default 200K).

- `LeanProofs/NorrellM3.lean` — paper App A.3.2.
  Verifies all 8 cases of the Norrell m=3 sum
  `B(a₀, a₁, a₂) = a₀·ω^(a₁+2a₂) + a₁·ω^(2a₀+a₂) + a₂·ω^(a₀+2a₁)`
  for (a₀, a₁, a₂) ∈ {0,1}³, confirming that B depends only on
  n₂ = a₀ + a₁ + a₂ and takes the values (0, 1, -1, 3) for n₂ = 0,1,2,3.
  The kernel relations are ω + ω² = -1 and ω³ = 1 (from Basic).

- `LeanProofs/Stabilizer/IsStab.lean` — the concrete stabilizer predicate
  and the first bounds stated against `Stabilizer.stabRank`. `stabVecN n k
  x0 W Q l` is the standard parametrisation on `n` qutrits with a
  quadratic-plus-linear phase mod 3; `IsStab v` says `v` is a nonzero
  multiple of one whose support parametrisation is injective, so the
  predicate is at most the true set of stabilizer states and a bound
  against it is a bound on the stabilizer rank. Proves
  `strange_m2_stabRank_le_two` (from `StrangeM2Pointwise`, via
  `stabRank_le_of_decomp`), `strange_m2_stabRank_gt_one` (from
  `StrangeM2Lower`, whose affine-support hypothesis is derived from the
  definition, with the computational basis discharging nonemptiness) and
  `strange_m2_stabRank_eq_two`. The earlier `StabDef.stabVec` accepts an
  arbitrary phase function and must not be used with
  `stabRank_le_of_decomp`, since under it any vector with entries that are
  powers of `ω₃` would count as stabilizer.

- `LeanProofs/T3GaloisDescent.lean` — discharges the Galois-closure
  hypothesis of `T3Galois.three_le_of_span_galois_orbit`. `mem_span_descend`
  descends coefficients from `ℂ` to a subfield by a dual-functional
  argument; `galEmb e` is the `ℚ`-algebra embedding `ℚ(ω₉) → ℂ` with
  `ω₉ ↦ ω₉^e` from the power basis and the ninth cyclotomic polynomial;
  for `e ∈ {1, 4, 7}` it fixes `ℤ[ω₃]` and permutes the conjugates.
  `t3_three_le`: any family of vectors over `ℤ[ω₃]` whose span contains
  `|T3⟩` has at least three members. No Galois group is used.
- `LeanProofs/T3M1StabRank.lean` — `t3_m1_stabRank_gt_two`:
  `stabRank IsStab |T3⟩ > 2`, the Galois lower bound stated against the
  concrete stabilizer predicate. `stabVecN_mem_Zomega3` shows every
  `stabVecN` is over `ℤ[ω₃]`, so every `IsStab` vector is a multiple of one,
  and `t3_three_le` applies to any set of at most two of them.

- `LeanProofs/T3GaloisM.lean` — the Galois lower bound at every number of
  copies: `t3M_stabRank_gt_two m : stabRank IsStab |T3⟩^⊗m > 2` for `m ≥ 1`
  (the conjugates `tConjM m a`, entries `ω₉^(e_a · digit sum)`, are
  independent since restriction to `firstDigit` indices gives the one-qutrit
  Vandermonde; the closure of the span is the descent of `T3GaloisDescent`
  on `Fin (3^m)`), `stabRank_smul` (rescaling the target does not change the
  rank) and the normalised form `t3TargetM_stabRank_gt_two`;
  `t3_m2_stabRank_gt_two` is the `m = 2` instance the board cites.

- `LeanProofs/T3M2StabRank.lean` — `t3_m2_stabRank_eq_three`:
  `stabRank IsStab |T3⟩^⊗2 = 3`. The three carry blocks of `T3M2Pointwise`
  are shown to be `IsStab` (the line `{(t, σ - t)}` with phases `t²`,
  `2t² + t`, `0`), the carry identity gives `≤ 3` through
  `stabRank_le_of_decomp`, and `T3GaloisM` gives `> 2`.

- `LeanProofs/M2StabRank.lean` — the Norrell and H₃ two-copy upper bounds
  against `stabRank`: `norrell_m2_stabRank_le_three` and
  `h3_m2_stabRank_le_three`, with each term of the pointwise files shown to
  be `IsStab` (full-support phases via `stabVecN_id`, the point `|2,2⟩` via
  `stabVecN_zero`, the lines `|0,+⟩`, `|+,0⟩` via `stabVecN_k1`).

- `LeanProofs/M3StabRank.lean` — the three-copy upper bounds against
  `stabRank`: `strange_m3_stabRank_le_four`, `norrell_m3_stabRank_le_four`,
  `h3_m3_stabRank_le_four`. Plane terms (`k = 2`) are evaluated by
  `stabVecN_k2` (nine-term sum) and decided at all 27 points; full-support
  terms via `stabVecN_id` with a mod-3 phase agreement lemma; the line
  `|0,0,+⟩` via `stabVecN_k1`. `stabRank_le_four_of` is the four-term
  analogue of the m=2 lemma.

## Build

Requires Lean 4 + mathlib4 (cached). From the `lean_proofs/` directory:

```bash
lake exe cache get  # pulls mathlib4 binary cache
lake build          # compiles the full LeanProofs library
```

The toolchain is pinned via `lean-toolchain`. First-time setup pulls
~5-10 GB of mathlib cache.
