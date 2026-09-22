# Lean 4 formalization of m ∈ {2, 3, 4} appendix identities

Scope: machine-verify the closed-form χ ≤ k stabilizer decompositions of
Appendix A of Labib and Russo, arXiv:2605.28586, at m ∈ {2, 3, 4}, using Lean 4 and mathlib4.

These files are an audit artifact for the appendix identities, not a
replacement for the mathematical exposition in that paper. The paper-level
mapping is:

| Paper location | Lean files |
| --- | --- |
| Appendix A.1, Strange m=2,3 | `StrangeM2Pointwise.lean`, `StrangeM3.lean`, `StrangeM3Pointwise.lean` |
| Appendix A.2, H_3 m=2,3 | `H3M2Pointwise.lean`, `H3M3.lean`, `H3M3Pointwise.lean` |
| Appendix A.3, Norrell m=2,3,4 | `NorrellM2Pointwise.lean`, `NorrellM3.lean`, `NorrellM3Pointwise.lean`, `NorrellM4Pointwise.lean` |
| Qubit H-type m=2,3,4 and T-type m=2,3,4 (bound files) | `QubitShared.lean`, `QubitHStabRank.lean`, `QubitTStabRank.lean`, `QubitTM4StabRank.lean` |
| Ququint T5 m=1 (both directions) and m=2 upper (bound files) | `Ququint.lean`, `T5Minors.lean`, `T5M1StabRank.lean`, `T5M2StabRank.lean` |
| Lower bounds S m=1, H m=2, T m=2, N m=2, H_3 m=2 (bound files) | `Stabilizer/RankOne.lean`, `StrangeM1Lower.lean`, `QubitM2Lower.lean`, `M2Lower.lean` |
| H_3 m=4, T_3 m=3,4, qubit H-type m=5,6,7,8, qubit T-type m=5,6 (bound-file witnesses, reflection route) | `Stabilizer/Reflect.lean`, `ReflectBases.lean`, `ReflectQubit.lean`, `ReflectQutrit.lean`, `H3M4StabRank.lean`, `T3M3StabRank.lean`, `T3M4StabRank.lean`, `QubitHM5StabRank.lean`, `QubitHM6StabRank.lean`, `QubitHM7StabRank.lean`, `QubitHM8StabRank.lean`, `QubitTM5StabRank.lean`, `QubitTM6StabRank.lean` |
| H_3 m=4, T_3 m=3,4, qubit H-type m=5,6,7,8,10, qubit T-type m=5,6 (bound-file witnesses, reflection route) | `Stabilizer/Reflect.lean`, `Stabilizer/Chunks.lean`, `ReflectBases.lean`, `ReflectQubit.lean`, `ReflectQutrit.lean`, `H3M4StabRank.lean`, `T3M3StabRank.lean`, `T3M4StabRank.lean`, `QubitHM5StabRank.lean`, `QubitHM6StabRank.lean`, `QubitHM7StabRank.lean`, `QubitHM8StabRank.lean`, `QubitHM10Data.lean`, `QubitHM10Key0.lean` to `QubitHM10Key3.lean`, `QubitHM10StabRank.lean`, `QubitTM5StabRank.lean`, `QubitTM6StabRank.lean` |

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

- `LeanProofs/M4StabRank.lean` — `norrell_m4_stabRank_le_seven`: the
  seven-term Norrell decomposition at four copies against `stabRank`. Adds
  `stabVecN_k3` (27-term expansion), general-`n` injectivity lemmas for
  planes and 3-flats, and a seven-term span lemma. With this every lean-tier
  upper bound on the board is a statement about `stabRank`.

- `LeanProofs/TensorStabRank.lean` — tensor products against `stabRank`.
  `tensor ψ φ` is the product state on `n + m` qutrits (a digit string is
  split into its first `n` and last `m` digits). `IsStab.tensor`: the tensor
  product of two `IsStab` vectors is `IsStab`, with `Fin.append` on `x₀` and
  `l`, block-diagonal `W` and `Q`, and the sum over `F_3^(k+k')` factored
  through `Fin.appendEquiv`. `stabRank_tensor_le`:
  `stabRank IsStab (ψ ⊗ φ) ≤ stabRank IsStab ψ * stabRank IsStab φ`, from
  minimal decompositions of each factor and their pairwise products.
  `strange_m4_stabRank_le_four` and `strange_m6_stabRank_le_eight` are the
  tensor square and cube of `strange_m2_stabRank_le_two`;
  `strangeVec4_apply`, `strangeVec6_apply` identify the states as the
  four- and six-fold products of the one-qutrit Strange amplitude.

- `LeanProofs/M1StabRank.lean` — one-copy upper bounds. `stabRank_le_pow`:
  the computational basis bounds `stabRank IsStab ψ ≤ 3^n` for every `ψ`.
  `strange_m1_stabRank_le_two` (`|S⟩ = (|1⟩ - |2⟩)/√2`),
  `t3_m1_stabRank_le_three`, and with `T3M1StabRank` the exact value
  `t3_m1_stabRank_eq_three`.

- `LeanProofs/Stabilizer/Rank.lean` — `DecompCards`, `stabRank` and the
  reduction lemmas, stated for an abstract predicate on `ι → ℂ` for any index
  type `ι`, so the same definitions serve `QutritVec n = Fin (3 ^ n) → ℂ` and
  `Fin (p ^ n) → ℂ`.

- `LeanProofs/Stabilizer/IsStabP.lean` — the stabilizer predicate for qudits
  of any prime dimension, `IsStabP (p : ℕ) [Fact p.Prime] (v : Fin (p ^ n) → ℂ)`,
  and `stabRankP p v := stabRank (IsStabP p) v`. It follows the verifier's
  conventions (`verify_challenge/stabrank_verify.py`): digits in `ZMod p`
  (`digitsP`), affine flats `y ↦ x₀ + Wᵀ y` over `ZMod p` (`affinePtP`), and
  the phase `ζ_D ^ (Q(y) · (D / p) + l·y)` (`stabVecP`, `quadPhaseP`) with
  `D = stabPeriod p` (`p` for odd `p`, `4` for `p = 2`), `ζ_D = exp(2πi/D)`
  (`zeta p`) and `l ∈ (ZMod D)^k`. For odd `p` this is `ω_p^(Q(y) + l·y)`; for
  `p = 2` it is `i^(l·y) (-1)^(Q(y))`. `ZMod 3` is `Fin 3` by definition and
  its ring structure and `ZMod.val` unfold to those of `Fin 3`, so the bridge
  `isStab_iff_isStabP : IsStab v ↔ IsStabP 3 v` and
  `stabRank_eq_stabRankP : stabRank IsStab v = stabRankP 3 v` are proved by
  `rfl` plus `zeta_three : zeta 3 = ω₃` and the factor `D / p = 1`; the qutrit
  files are unchanged and every `stabRank IsStab` theorem in them is a
  `stabRankP 3` theorem through `stabRank_eq_stabRankP`. Also here: the
  generic forms of the qutrit tools (`stabVecP_id`, `stabVecP_zero`,
  `isStabP_single`, `decompCardsP_nonempty`, `stabRankP_le_pow`),
  `affinePtP_injective_of_pivots` (injectivity from pivot columns of `W`),
  `stabTerm` (a `stabVecP` on computational-basis indices, `IsStabP` by
  `isStabP_stabTerm`) and `stabRankP_le_of_terms` (a pointwise identity
  `ψ = Σ αⱼ • σⱼ` gives `stabRankP p ψ ≤ r`), and the explicit sums over
  `F_2^k` for `k ≤ 3` (`stabVecP_two_k1`, `_k2`, `_k3`).

- `LeanProofs/Stabilizer/TensorP.lean` — `TensorStabRank.lean` for
  `IsStabP p`: `tensorP`, `IsStabP.tensor`, and
  `stabRankP_tensor_le : stabRankP p (ψ ⊗ φ) ≤ stabRankP p ψ * stabRankP p φ`.

- `LeanProofs/Stabilizer/PhaseP.lean` — the phase exponent of a `stabVecP`
  under an affine substitution of its parameters, for any prime `p`. The
  computation is done in `ZMod D` (`D = stabPeriod p`) through two maps
  `ZMod p → ZMod D`: `qcast p u = (D / p) · u.val`, which is additive because
  `(D / p) · n` mod `D` depends only on `n` mod `p` (`natCast_div_mul_mod`),
  and `liftD p u = u.val`, which equals `qcast` for odd `p` and for `p = 2`
  satisfies `liftD (u + v) = liftD u + liftD v + 2 · liftD u · liftD v` in
  `ZMod 4`. `phaseZ` is `quadPhaseP` read in `ZMod D` (`natCast_quadPhaseP`),
  `IsPhaseExp p g` says `g` is a constant plus some `phaseZ p Q' l'`, and the
  class is closed under sums (`IsPhaseExp.sum`), `qcast` of a product of two
  affine functions (`IsPhaseExp.qcast_mul_affine_affine`) and `l · liftD` of
  an affine function (`IsPhaseExp.mul_liftD_affine`, by induction on the set
  of summands in the `p = 2` case). `zeta_pow_quadPhaseP_comp`: if every
  coordinate `y_i(z)` is affine in `z`, then
  `ζ ^ quadPhaseP Q l (y(z)) = ζ ^ C * ζ ^ quadPhaseP Q' l' z` for some
  `C`, `Q'`, `l'`.

- `LeanProofs/Stabilizer/SliceP.lean` — slices and the projection bound.
  `sliceP p v a` fixes the last digit of `v` to `a`.
  `IsStabP.slice : IsStabP p v → sliceP p v a = 0 ∨ IsStabP p (sliceP p v a)`:
  the hyperplane `x₀_n + Σ_j y_j W_j n = a` is either independent of `y`
  (`stabVecP_snoc_of_last_zero`: the slice is zero or the same state with
  the last column dropped) or solved for a pivot coordinate `y_{j₀}` and
  parametrised by the remaining `k - 1` (`subP`, `sliceX0`, `sliceW`,
  `stabVecP_snoc_pivot`), with the phase handled by `PhaseP.lean`.
  Slicing is linear (`sliceLinP`), so
  `stabRankP_sliceP_le : stabRankP p (sliceP p v a) ≤ stabRankP p v`
  (the zero slices of a minimal decomposition are dropped). For products
  `sliceP_tensorP : sliceP (ψ ⊗ φ) a = ψ ⊗ sliceP φ a`, and slicing `m`
  times along the digits of a point where `φ` is nonzero gives the
  **projection bound**
  `stabRankP_le_tensorP : φ ≠ 0 → stabRankP p ψ ≤ stabRankP p (tensorP p ψ φ)`
  (with `stabRankP_smul` for the final scalar). `powVecP p f m` is the
  `m`-fold tensor power of a one-qudit amplitude `f : ZMod p → ℂ` on digit
  strings (`tensorP_powVecP : ψ^a ⊗ ψ^b = ψ^(a+b)`), and against it
  `stabRankP_powVecP_le_succ : f ≠ 0 → χ(ψ^m) ≤ χ(ψ^(m+1))`,
  `stabRankP_powVecP_mono : f ≠ 0 → a ≤ b → χ(ψ^a) ≤ χ(ψ^b)`, and
  `stabRankP_powVecP_add_le : χ(ψ^(a+b)) ≤ χ(ψ^a) · χ(ψ^b)` (from
  `stabRankP_tensor_le`).

  Board status. The Lean lower bounds are `T3` at `m = 1, 2` (`≥ 3`, and
  `t3M_stabRank_gt_two` already gives `≥ 3` at every `m`) and `S` at `m = 2`
  (`≥ 2`); every cell one or more copies up already holds a stronger
  computational bound (`T3`: `≥ 8` at `m = 3, 4, 5`; `S`: `≥ 4` at `m = 3, 4`,
  `≥ 5` at `m = 5, 6`), so the projection lemma changes no tier on the board.
  The bound files whose notes cite projection monotonicity
  (`S-m4/m5/m6`, `T3-m4/m5`, `qubit_H-m5/m6`, `qubit_T-m5/m6` lower) project
  from computational exclusions, so the projection step is now formal but
  their sources are not, and they keep their tier.

- `LeanProofs/QubitShared.lean` — the qubit magic states on digit strings,
  `hVec m` (`|H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩`) and `tVec m`
  (`|T⟩ = cos β|0⟩ + e^{iπ/4} sin β|1⟩`, `β = arccos(1/√3)/2`), with the
  trigonometric relations the cells use: the squares and the product of the
  cosine and sine (`cH_sq`, `sH_sq`, `sH_mul_cH`, `cT_sq`, `sT_sq`,
  `sT_mul_cT`), the linear relations `cH_eq : cos(π/8) = (1 + √2) sin(π/8)`
  and `cT_eq : cos β = sin β · √2(√3 + 1)/2` that eliminate the cosine, and
  `exp_pi_div_four_mul_I`.

- `LeanProofs/QubitHStabRank.lean` — `qubit_h_m2_stabRankP_le_two`,
  `qubit_h_m3_stabRankP_le_three` (the witnesses of the bound files as
  `stabTerm`s, the identity decided at every digit string) and
  `qubit_h_m4_stabRankP_le_four` as the tensor square of m=2 through
  `stabRankP_tensor_le` and `tensorP_hVec`.

- `LeanProofs/QubitTStabRank.lean` — `qubit_t_m2_stabRankP_le_two`,
  `qubit_t_m3_stabRankP_le_three`, same method.

- `LeanProofs/QubitTM4StabRank.lean` — `qubit_t_m4_stabRankP_le_three`: the
  three-term decomposition of `QubitTM4.lean` against `stabRankP 2`, from the
  witness of `bounds/qubit_T-m4-upper-3.json`. `QubitTM4.lean` and its
  `stabilizer_rank_le_three` are kept as they were.

- `LeanProofs/Ququint.lean`: the ququint foundation. `omega5 = exp(2πi/5)`
  with `omega5_pow_five`, the cyclotomic relation `omega5_geom_sum`
  (`1 + ω + ω² + ω³ + ω⁴ = 0`) and its rewriting form `omega5_pow_four`,
  the exponent reductions `omega5_pow_reduce` (`ω^n = ω^(n-5)` for `n ≥ 5`,
  the non-looping form) and `zeta_five_pow` (`ζ₅^n = ω₅^((n : ZMod 5).val)`),
  and the linear independence of `1, ω, ω², ω³` over `ℚ`
  (`omega5_linearIndependent`, from `linearIndependent_pow` and the fifth
  cyclotomic polynomial being the minimal polynomial). From it the
  equal-coefficient zero test of `cert_t5_m1_rank2.py`:
  `omega5_int_comb_eq_zero_iff`, an integer combination of `1, …, ω⁴` is zero
  iff its five coefficients agree, and the `ZMod 5 → ℤ` form
  `omega5_sum_val_eq_zero_iff`. Also the T5 state `t5Amp x = ω^(x³)/√5`,
  `t5Vec m = |T5⟩^⊗m` on digit strings, `t5Amp_mul`, the `ZMod 5` case split
  `zmod5_cases`, the `val` lemmas for numerals in `ZMod 5` and
  `ZMod (stabPeriod 5)`, and `stabVecP_five_k1` (a line on ququints as a
  five-term sum).

- `LeanProofs/T5Minors.lean`: the rank-2 exclusion of `|T5⟩` over the 30
  single-ququint stabilizer states, as the certificate argues it. `Shape` is
  the exponent data of a state up to a scalar (a point `pt x₀`, or the
  full-support `full q l` with phase `ω^(q y² + l y)`), `Shape.vec` its
  complex vector, `t5v = √5 |T5⟩`. A `3 × 3` minor of `[s | s' | T5]` is a
  signed sum of six monomials `ω^e`, accumulated computably as five integer
  coefficients (`minorTerms`, `coeffOf`); `minorNonzero` is the
  equal-coefficient test, and `det_minorMat`, `evalL_eq`,
  `det_ne_zero_of_minorNonzero` identify it with the complex determinant, so
  `minorNonzero = true` proves the minor nonzero. `pairs_ok` is the search
  (every pair of distinct shapes has a nonzero minor among the ten row
  triples), closed by `decide +kernel` in a few seconds.
  `t5v_not_mem_span_smul`: no two scalar multiples of shape vectors span
  `t5v`, with a repeated shape padded by a different one.

- `LeanProofs/T5M1StabRank.lean`: `χ(|T5⟩) = 3` against `stabRankP 5`.
  `t5_m1_stabRankP_le_three`: the witness of `bounds/T5-m1-upper-3.json`
  (`|0⟩`, `|1⟩`, `Σ_y ω^(4y² + 4y)|y⟩`) as `stabTerm`s with coefficients
  `(2 + ω + ω² + ω³)/√5`, `(ω - ω²)/√5`, `-(1 + ω + ω² + ω³)/√5`, the file's
  nested radicals evaluated exactly in `ℚ(ω₅)`; the identity is decided at
  the five digits by `simp`, then `ring_nf`, exponent reduction, `ω⁴`
  rewriting and `ring`. `isStabP_one_ququint`: every `IsStabP 5` vector on
  one ququint is a scalar multiple of a `Shape` vector (injectivity of the
  flat bounds the number of generators by one; with one generator `w ≠ 0`
  and the substitution `y = (x - x₀)/w` turns `ω^(Q y² + l y)` into
  `ω^(q' x² + l' x)` up to a constant). `t5_m1_stabRankP_gt_two`: with
  `reindex5` carrying spans from `Fin (5 ^ 1) → ℂ` to `ZMod 5 → ℂ`, a set of
  at most two `IsStabP 5` vectors is two scalar multiples of shapes, and
  `t5v_not_mem_span_smul` applies. `t5_m1_stabRankP_eq_three` combines them.

- `LeanProofs/T5M2StabRank.lean`: `t5_m2_stabRankP_le_eight`: the eight
  terms of `bounds/T5-m2-upper-8.json` (three points, four lines, one
  full-support state) with coefficients `β_j / 5`, `β_j ∈ ℤ[ω₅]` (the file's
  `c_j` times `√5^(k_j - 2)`), decided at the 25 digit strings by the same
  tactic sequence as the one-copy identity.
- `LeanProofs/Stabilizer/RankOne.lean`: `stabRank ψ > 1` from
  non-membership. `stabRank_gt_one_of_not_stab`: for a predicate closed under
  nonzero rescaling (`IsStabP.smul`, `IsStab.smul`), a nonzero `ψ` outside it
  has rank at least two, since a decomposition of size at most one is empty
  or a single rescaled state. `IsStabP.pow_period_eq`: every nonzero entry of
  an `IsStabP p` vector is `c ζ_D^e`, so its `D`-th power is `c^D` at every
  point of the support; this is the one-modulus argument of
  `cert_rank1_moduli.py` without norms. Also the generic support lemmas
  `stabVecP_ne_zero_imp` and `stabVecP_apply_affinePtP`.

- `LeanProofs/StrangeM1Lower.lean`: `strange_m1_stabRank_gt_one`:
  `stabRank IsStab |S⟩ > 1`, from `IsStab.affine_support` at `a = d = 1`,
  `b = 2` (the support `{1, 2}` would have to contain `0`), and
  `strange_m1_stabRank_eq_two` with `M1StabRank`. Build ≈ 3 s.

- `LeanProofs/QubitM2Lower.lean`: `qubit_h_m2_stabRankP_gt_one` and
  `qubit_t_m2_stabRankP_gt_one`: `stabRankP 2 |H⟩^⊗2 > 1` and
  `stabRankP 2 |T⟩^⊗2 > 1`. The amplitudes at `00` and `01` are
  `cos²(π/8)`, `cos(π/8) sin(π/8)` and `cos² β`, `e^{iπ/4} cos β sin β`;
  equal fourth powers would force `√2 = -4/3` and `√3 = -2` respectively,
  through `linear_combination` with cofactors from polynomial division
  against the relations of `QubitShared`. With the upper bounds,
  `qubit_h_m2_stabRankP_eq_two` and `qubit_t_m2_stabRankP_eq_two`.
  Build ≈ 4 s.

- `LeanProofs/M2Lower.lean`: `norrell_m2_stabRank_gt_two` and
  `h3_m2_stabRank_gt_two`: `stabRank IsStab |N⟩^⊗2 > 2` and
  `stabRank IsStab |H₃⟩^⊗2 > 2`, with no enumeration of the 360 two-qutrit
  stabilizer states. `IsStab.norm_eq`: the nonzero entries of an `IsStab`
  vector share one modulus. `IsStab.full_or_small`: on two qutrits an
  `IsStab` vector has full support with entries `c ω^e` (`k = 2`, injective
  hence surjective) or a support of at most three points (`k ≤ 1`).
  `stabRank_gt_two_of_three_moduli`: a vector with modulus `r₁` on four
  points, `r₂` on four points and `r₃` at one point, `0 < r₁ < r₂ < r₃`,
  `r₁ + r₂ < r₃`, has rank at least three. A small-support term leaves a
  point of each four-point class to the other term alone, which then carries
  two moduli; two full-support terms put `ψ(x) ω^(2e₁(x))` on the triangle
  `a' + b' ω^j`, and with distinct moduli at the probe points `g` takes all
  three values, so `Σ ω^(gᵢ) uᵢ = 0` and the triangle inequality gives
  `r₃ ≤ r₁ + r₂` (`pompeiu`, 27 cases by `interval_cases`). Both targets are
  `h ⊗ h` with `h` of two moduli in ratio `2` (Norrell) or `√3 + 1` (H₃), so
  the classes are the digit strings with zero, one, or two occurrences of the
  distinguished digit. `norrell_m2_stabRank_eq_three` and
  `h3_m2_stabRank_eq_three` with `M2StabRank`. Build ≈ 5 s.
- `LeanProofs/QubitProjection.lean` — `SliceP.lean` at the qubit states:
  `hVec m = powVecP 2 hAmp1 m` and `tVec m = powVecP 2 tAmp1 m` by `rfl`,
  `hAmp1_ne_zero`, `tAmp1_ne_zero` (from `cH_sq`, `cT_sq`), and
  `qubit_h_stabRankP_le_succ`, `qubit_h_stabRankP_mono`,
  `qubit_h_stabRankP_add_le` with the `qubit_t_` versions: a Lean lower bound
  on `|H⟩^⊗m` or `|T⟩^⊗m` at one `m` becomes one at every larger `m` by
  `qubit_h_stabRankP_mono`.
- `LeanProofs/Stabilizer/Reflect.lean` — the reflection route for witness
  checks. A bound-file witness is a list of `stabTerm`s with coefficients in a
  number field `ℚ(ζ, √2, √3, …)`; instead of a `linear_combination` per digit
  string, the identity is decided by the kernel on integer coordinate vectors.
  `ev B v = Σ vᵢ Bᵢ` evaluates an integer vector against a ℤ-basis `B` of the
  ring; `mulM M` is multiplication by an integer matrix and `Represents B M θ`
  says `M` is the matrix of `θ`, so `ev B (mulM M v) = θ * ev B v` and
  iterates give powers. `ysol x0 piv x` recovers the flat coordinates of a
  digit string from the pivot columns of `W`, and `stabVecP_eq_ite_of_pivots`
  collapses the sum over `F_p^k` in `stabVecP` to a single `if`, so the
  integer side `termZ` of a term costs `O(kn)` per digit string. `ev_termZ`
  identifies `ev B (termZ M C … x)` with `ev B C * stabVecP … x`. `count0`
  and `prod_two_valued` write a product of a two-valued amplitude over the
  digits as `a^(zeros) b^(n - zeros)`.

- `LeanProofs/ReflectBases.lean` — generated by
  `tools/gen_witness_lean.py --bases`. The bases `B4 = (1, √2, i, √2 i)` of
  `ℤ[√2, i]`, `B8` of `ℤ[√2, √3, i]`, `B3 = (1, ω₃, √3, √3 ω₃)` of
  `ℤ[ω₃, √3]` and `B9 = (ω₉^t, √3 ω₉^t)_{t<6}` of `ℤ[ω₉, √3]`, the matrices of
  `i`, `ω₃`, `ω₉`, `ω₉³` and of the factors of the target amplitudes
  (`1+√2`, `2-√2`, `√2√3+√2`, `√2+i√2`, `3-√3`, `3+√3`, `√3`), and the
  `Represents` proofs: one column identity per basis element, closed by
  `linear_combination` with cofactors found by polynomial division in the
  generator. `MI4_represents`, `MI8_represents`, `Mw3_represents`,
  `Mw9c_represents` are the forms with `zeta p` that `ev_termZ` takes.

- `LeanProofs/ReflectQubit.lean`, `LeanProofs/ReflectQutrit.lean` — the
  targets as `scalar * ev B (integer vector)`. At a digit string with `a` zero
  digits `hVec m` is `sin(π/8)^(m%2)/4^(m/2)` times `(1+√2)^a (2-√2)^(m/2)`
  (`hVec_eq_ev`), `tVec m` is `sin β^(m%2)/(2^m 6^(m/2))` times
  `(√2√3+√2)^a (√2+i√2)^(m-a) (3-√3)^(m/2)` (`tVec_eq_ev`), `h3Vec m` (the
  product of `h3Amp1` of `H3Shared`) is `N^(m%2)/6^m` times
  `(3+√3)^a √3^(m-a) (3-√3)^(m/2)` (`h3Vec_eq_ev`), and `t3TargetM m` of
  `T3GaloisM` is `(1/√3)^m` times `ω₉^(digit sum)` (`t3TargetM_eq_ev`). The
  integer vectors are iterates of the matrices of `ReflectBases`; the
  algebra with variable exponents is in `h_alg`, `t_alg`, `h3_alg`.

- `LeanProofs/H3M4StabRank.lean`, `T3M3StabRank.lean`, `T3M4StabRank.lean`,
  `QubitHM5StabRank.lean`, `QubitHM6StabRank.lean`, `QubitHM7StabRank.lean`,
  `QubitHM8StabRank.lean`, `QubitTM5StabRank.lean`, `QubitTM6StabRank.lean`
  — generated from the bound files by `tools/gen_witness_lean.py
  bounds/<cell>.json`; do not edit by hand. Each states the terms of the
  witness as `stabTerm`s with their pivot columns (`IsStabP` by
  `isStabP_stabTerm` and `affinePtP_injective_of_pivots`), one integer vector
  `Cj` per coefficient (the file's coefficient divided by the `√p^k`
  normalisation of its term and by the scalar in front of the target, times
  a common denominator `Den`), the identity `Den • tgt = Σ termZ` decided by
  `decide +kernel` at every index (`key`), and the assembly `target_eq` that
  feeds `stabRankP_le_of_terms`. The qutrit cells also restate the bound
  against `IsStab` through `stabRank_eq_stabRankP`. Build times on a laptop:
  6 s to 16 s for `p^n ≤ 128` indices, 42 s at 256. The generator checks the
  integer identity itself before writing, and the coefficient expressions of
  the bound files (nested radicals such as `√(1/2 - √2/4) = sin(π/8)`) are
  recognised in the ring by dividing the radicand by `sin²` and taking the
  square root in the quadratic or biquadratic field.

- `LeanProofs/Stabilizer/Chunks.lean`, `QubitHM10Data.lean`,
  `QubitHM10Key0.lean` to `QubitHM10Key3.lean`, `QubitHM10StabRank.lean` —
  the `m = 10` H-type cell (`qubit_h_m10_stabRankP_le_eighteen`, from the
  eighteen terms of `bounds/qubit_H-m10-upper-18.json`). One kernel check over
  the `1024` indices runs for longer than a build step is allowed here, so
  `gen_witness_lean.py --chunks 4` writes the data (terms, coefficient
  vectors, `rhsZ`) to `QubitHM10Data`, one slice theorem `key<q>` over the
  indices `q * 256 + i` to each `QubitHM10Key<q>` (130 s to 164 s each), and
  the assembly to `QubitHM10StabRank`, where `forall_fin_of_chunks` of
  `Chunks.lean` (`chunkIdx q i = ⟨q * N + i, _⟩`, and every index is
  `chunkIdx (idx / N) (idx % N)`) recovers the statement over `Fin (2 ^ 10)`.
  `cidx` fixes the implicit `c` and `N` of `chunkIdx`; without them
  unification reads `2 ^ 10` as `2 * 2 ^ 9`.

## Pitfalls

Things that cost time in this development and are not obvious from the
error messages.

- `StabDef.stabVec` accepts an arbitrary phase function and must not be used
  with `stabRank_le_of_decomp`: under it any vector with entries that are
  powers of `ω₃` would count as stabilizer. Use `IsStab` or `IsStabP`.
- `stabVecN`, `stabVecP` and `stabTerm` carry no `1/√(p^k)` normalisation.
  A coefficient taken from a bound file has to be divided by `√(p^k)` for the
  term with `k` generators.
- `fin_cases` on a variable of type `ZMod 2` substitutes `⟨0, _⟩ : Fin 2`,
  not the numeral `(0 : ZMod 2)`, so lemmas about `ZMod.val` of numerals
  never fire afterwards. Split with `rcases zmod2_cases a with rfl | rfl`
  instead; it substitutes the numerals.
- `decide` on `Finset.univ = {…}` for `Fin k → ZMod 2` times out in the
  elaborator's `whnf` even for `k = 1`; `decide +kernel` closes it at once.
  The same goals over `Fin k → Fin 3` were fine with plain `decide`.
- Do not put `stabPeriod_two : stabPeriod 2 = 4` in a `simp` set that also
  has to evaluate `ZMod.val` on `ZMod (stabPeriod 2)`. Rewriting the numeral
  inside the type leaves `ZMod.val 0` terms that neither `ZMod.val_zero` nor
  the local `zmod4_val_*` lemmas match. Use `stabPeriod_two_div :
  stabPeriod 2 / 2 = 2` for the exponent factor and leave the type alone.
- Reducing powers of `i`: `simp` applies `pow_one` before a lemma about
  `zeta 2 ^ n`, so `zeta_two_pow` alone leaves a bare `zeta 2`; add
  `zeta_two`. Exponents above `3` then survive as `I ^ 5`, and a simp lemma
  `I ^ n = I ^ (n % 4)` loops. `I_pow_reduce (h : 4 ≤ n) : I ^ n = I ^ (n - 4)`
  terminates because `simp` decides the side condition on numerals.
- `Real.cos_pi_div_eight` and `Real.sin_pi_div_eight` are simp lemmas, so a
  bare `simp` turns `cos(π/8)` into `√(2 + √2) / 2` and the trigonometric
  relations no longer apply. Wrap the atoms in definitions (`cH`, `sH`, `cT`,
  `sT`) and never unfold them in the case analysis.
- `∏ i : Fin 3, f i` is not expanded by `simp` on its own; add
  `Fin.prod_univ_three` (and `Fin.prod_univ_four`). `Fin.prod_univ_two` is
  a simp lemma, which is why the `m = 2` cells worked without it.
- `linear_combination` cofactors for the qubit identities are found by
  polynomial division in the atoms `√2, √3, i, sin`, after the cosine has
  been eliminated through `cH_eq` / `cT_eq` (put the equation in the simp
  set). Without that elimination the ideal `⟨cos², sin², cos·sin, √2², …⟩`
  is not a Gröbner basis and the cofactors are much larger. The generators
  `sin² - β, √2² - 2, √3² - 3, i² + 1` have pairwise coprime leading terms,
  so division gives remainder `0` exactly when the identity holds.
- `Finset.univ.image σ` in a theorem statement needs `DecidableEq` on the
  vector type, which `classical` inside the proof does not provide. State
  span membership with `Set.range σ` and convert with `Finset.coe_image`,
  `Finset.coe_univ`, `Set.image_univ` in the proof.
- `rw` closes goals only with reducible `rfl`. After rewriting the qutrit
  data into the `ZMod 3` form (`stabVecP_three`) the two `if`s differ only
  in their `Decidable` instances and need an explicit `rfl`.
- Identities in `ℚ(ω₅)`: `ring` alone cannot use `ω⁵ = 1` or `Φ₅(ω) = 0`.
  The sequence that works is `ring_nf` (a sum of monomials `ω^n` with numeral
  exponents), then `simp only [omega5_pow_reduce, omega5_pow_four,
  Nat.reduceSub, Nat.reduceLeDiff, pow_zero]` (every exponent below 4; the
  two `Nat` simprocs must be listed, since `simp only` does not evaluate
  `6 - 5` or decide `5 ≤ 6` on its own), then `ring`. Both sides are then
  polynomials of degree at most 3 in `ω`, and linear independence of
  `1, ω, ω², ω³` says equal values have equal reduced forms. `ω^n = ω^(n % 5)`
  as a simp lemma loops on `ω^4`.
- `ring_nf` can close a goal outright, and then the next tactic in a `;`
  chain fails with "no goals". Run the stages as `all_goals (try ring_nf)`,
  `all_goals (try simp only [...])`, `all_goals ring`.
- `rw [mul_assoc]` on a goal whose exponent is a `Nat` product rewrites the
  exponent first. Use `conv_rhs => rw [mul_assoc, ...]` to aim it.
- `decide +kernel` handles the full `T5Minors.pairs_ok` search (870 ordered
  shape pairs, up to ten `3 × 3` minors each, `ZMod 5` arithmetic on
  `Option` values) in a few seconds when the data are `List`s and `Shape` is
  an inductive with `deriving DecidableEq`.
- `first | ring | linear_combination …` never reaches the second alternative:
  `ring` that fails to close the goal falls back to `ring_nf`, reports a
  suggestion and succeeds. Use `ring1` in a `first` block.
- `ring` cannot prove `(x * (y + z) / 6) ^ a = x ^ a * (y + z) ^ a / 6 ^ a`
  for a variable `a`: it distributes the base into a sum first and then
  cannot split the power. Rewrite with `mul_pow`, `div_pow` (and
  `pow_add` for `x ^ (a + b)`) before calling `ring`; afterwards `(y + z) ^ a`
  is an atom on both sides.
- `simp` does not rewrite `h3Amp1 (v : ZMod 3)` with equations stated on
  `Fin 3` numerals: `ZMod 3` and `Fin 3` are defeq but not reducibly so, and
  the numerals carry different instances. State the value lemmas on `ZMod 3`
  and prove them by `rfl`, and case-split with a `decide +revert` lemma
  (`zmod3_cases`) rather than `fin_cases`.
- `decide +kernel` on `∀ idx : Fin (2 ^ 8), …` over integer vectors runs in
  seconds; at `2 ^ 10` the elaborator's `whnf` hits the default
  `maxHeartbeats` before the kernel is reached. Set `maxHeartbeats 0` on the
  theorem.

## Build

Requires Lean 4 + mathlib4 (cached). From the `lean_proofs/` directory:

```bash
lake exe cache get  # pulls mathlib4 binary cache
lake build          # compiles the full LeanProofs library
```

The toolchain is pinned via `lean-toolchain`. First-time setup pulls
~5-10 GB of mathlib cache.
