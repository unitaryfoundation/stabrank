/-
The T-type qubit upper bounds against `stabRankP 2`: `χ(|T⟩^⊗2) ≤ 2` and
`χ(|T⟩^⊗3) ≤ 3`.

The witnesses are those of `bounds/qubit_T-m2-upper-2.json` and
`bounds/qubit_T-m3-upper-3.json`, entered term by term as `stabTerm 2 n k
x0 W Q l` with `l` read in `ZMod 4`; `stabTerm` carries no `1/√(2^k)`
normalisation, so each coefficient is the file's divided by `√(2^k)`, and the
nested radicals of the file are written in the atoms `√2`, `√3`, `i`,
`cos β`, `sin β` (for instance `√(√3/6 + 1/2) = cos β` and
`√(1/2 - 5√3/18) = √3 (cos β - √2 sin β)/3`). Every term is `IsStabP 2` by
`isStabP_stabTerm` once its parametrisation is injective, and the pointwise
identity is decided at every digit string: the sums over `F_2^k` are expanded,
the `if`s and the exponents are computed, `e^{iπ/4}` becomes `√2(1 + i)/2`,
`cos β` is rewritten as `sin β · √2(√3 + 1)/2`, and what remains is a
polynomial identity in `√2`, `√3`, `i`, `sin β` modulo `√2² = 2`, `√3² = 3`,
`i² = -1` and `sin β² = 1/2 - √3/6`, closed by `linear_combination` with
cofactors found by polynomial division.
-/
import LeanProofs.QubitShared

namespace StabRank

open Stabilizer

/-! ### m = 2 -/

noncomputable def tT2_0 : Fin (2 ^ 2) → ℂ :=
  stabTerm 2 2 1 ![0, 0] ![![1, 1]] ![![0]] ![1]
noncomputable def tT2_1 : Fin (2 ^ 2) → ℂ :=
  stabTerm 2 2 2 0 (idWP 2 2) ![![0, 1], ![0, 0]] ![0, 0]

theorem tT2_0_isStab : IsStabP 2 tT2_0 :=
  isStabP_stabTerm 2 2 1 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0] (by decide +kernel))
theorem tT2_1_isStab : IsStabP 2 tT2_1 :=
  isStabP_stabTerm 2 2 2 _ _ _ _ (affinePtP_id_injective 2 2)

noncomputable def tCoef2 : Fin 2 → ℂ :=
  ![-(Real.sqrt 3 : ℂ)*Complex.I/6 + 1/2, (Real.sqrt 3 : ℂ)*Complex.I/6 + (Real.sqrt 3 : ℂ)/6]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
set_option linter.style.longLine false in
set_option maxHeartbeats 1600000 in
theorem tVec2_eq : tVec 2 = ∑ j : Fin 2, tCoef2 j • ![tT2_0, tT2_1] j := by
  have hs := sT_sq
  have h2 := sqrt2_sq_c
  have h3 := sqrt3_sq_c
  have hI := Complex.I_sq
  have hlin := cT_eq
  funext idx
  simp only [Fin.sum_univ_two, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, tVec, tCoef2,
    tT2_0, tT2_1, stabTerm, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons, Matrix.cons_val_three]
  generalize hx : digitsP 2 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] :=
    ⟨x 0, x 1, fin2_eta_z x⟩
  simp only [stabVecP_id, stabVecP_two_k1]
  rcases zmod2_cases a with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;>
    simp (config := { decide := true }) [tAmp1, exp_pi_div_four_mul_I, quadPhaseP, Fin.sum_univ_two, Fin.sum_univ_one,
      Fin.prod_univ_two, zmod2_val_zero, zmod2_val_one, zmod4_val_zero, zmod4_val_one, zmod4_val_two,
      zmod4_val_three, stabPeriod_two_div, zeta_two_pow, zeta_two, Complex.I_sq, I_pow_three,
      I_pow_reduce, hlin]
  all_goals first
    | linear_combination ((Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^2/4 + (sT : ℂ)^2*Complex.I/4 + (sT : ℂ)^2/4) * h2 + (-Complex.I/12 - 1/12) * h3 + ((Real.sqrt 3 : ℂ)*Complex.I/2 + (Real.sqrt 3 : ℂ)/2 + Complex.I/2 + 1/2) * hs
    | linear_combination ((Real.sqrt 3 : ℂ)^2*(sT : ℂ)^2/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^2/2 + (sT : ℂ)^2/4) * h2 + ((sT : ℂ)^2/2 - 1/6) * h3 + ((Real.sqrt 3 : ℂ) + 2) * hs
    | linear_combination ((sT : ℂ)^2*Complex.I^2/4 + (sT : ℂ)^2*Complex.I/2 + (sT : ℂ)^2/4) * h2 + ((Real.sqrt 3 : ℂ)/6 + (sT : ℂ)^2/2) * hI + (Complex.I) * hs

/-- **χ(|T⟩^⊗2) ≤ 2 against `IsStabP 2`.** -/
theorem qubit_t_m2_stabRankP_le_two : stabRankP 2 (tVec 2) ≤ 2 :=
  stabRankP_le_of_terms 2 ![tT2_0, tT2_1] tCoef2
    (fun j => by fin_cases j; exacts [tT2_0_isStab, tT2_1_isStab]) tVec2_eq

/-! ### m = 3 -/

noncomputable def tT3_0 : Fin (2 ^ 3) → ℂ :=
  stabTerm 2 3 3 0 (idWP 2 3) ![![0, 1, 1], ![0, 0, 1], ![0, 0, 0]] ![3, 3, 0]
noncomputable def tT3_1 : Fin (2 ^ 3) → ℂ :=
  stabTerm 2 3 2 ![0, 0, 0] ![![1, 0, 0], ![0, 1, 0]] ![![0, 0], ![0, 0]] ![1, 1]
noncomputable def tT3_2 : Fin (2 ^ 3) → ℂ :=
  stabTerm 2 3 2 ![0, 0, 0] ![![1, 1, 0], ![0, 0, 1]] ![![0, 1], ![0, 0]] ![3, 0]

theorem tT3_0_isStab : IsStabP 2 tT3_0 :=
  isStabP_stabTerm 2 3 3 _ _ _ _ (affinePtP_id_injective 2 3)
theorem tT3_1_isStab : IsStabP 2 tT3_1 :=
  isStabP_stabTerm 2 3 2 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0, 1] (by decide +kernel))
theorem tT3_2_isStab : IsStabP 2 tT3_2 :=
  isStabP_stabTerm 2 3 2 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0, 2] (by decide +kernel))

noncomputable def tCoef3 : Fin 3 → ℂ :=
  ![(Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/6, -(Real.sqrt 3 : ℂ)*(cT : ℂ)*Complex.I/6 + (cT : ℂ)/2, -(Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/6 + (Real.sqrt 3 : ℂ)*(cT : ℂ)*Complex.I/6 + (Real.sqrt 3 : ℂ)*(cT : ℂ)/6]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
set_option linter.style.longLine false in
set_option maxHeartbeats 1600000 in
theorem tVec3_eq : tVec 3 = ∑ j : Fin 3, tCoef3 j • ![tT3_0, tT3_1, tT3_2] j := by
  have hs := sT_sq
  have h2 := sqrt2_sq_c
  have h3 := sqrt3_sq_c
  have hI := Complex.I_sq
  have hlin := cT_eq
  funext idx
  simp only [Fin.sum_univ_three, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, tVec, tCoef3,
    tT3_0, tT3_1, tT3_2, stabTerm, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons, Matrix.cons_val_three]
  generalize hx : digitsP 2 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] :=
    ⟨x 0, x 1, x 2, fin3_eta_z x⟩
  simp only [stabVecP_id, stabVecP_two_k2]
  rcases zmod2_cases a with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;> rcases zmod2_cases c with rfl | rfl <;>
    simp (config := { decide := true }) [tAmp1, exp_pi_div_four_mul_I, quadPhaseP, Fin.sum_univ_three, Fin.sum_univ_two,
      Fin.prod_univ_three, zmod2_val_zero, zmod2_val_one, zmod4_val_zero, zmod4_val_one, zmod4_val_two,
      zmod4_val_three, stabPeriod_two_div, zeta_two_pow, zeta_two, Complex.I_sq, I_pow_three,
      I_pow_reduce, hlin]
  all_goals first
    | linear_combination ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^3*Complex.I/8 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^3/8 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/8) * h2 + ((Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I^2/12 - (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/12 - (Real.sqrt 2 : ℂ)*(sT : ℂ)/12) * h3 + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/12 + (Real.sqrt 2 : ℂ)*(sT : ℂ)/4) * hI + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)*Complex.I/2 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/2 + (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I + (Real.sqrt 2 : ℂ)*(sT : ℂ)) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3*Complex.I^2/8 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I^2/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/8) * h2 + (-(Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/12) * h3 + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/4) * hI + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)*Complex.I/2 + (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/2) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)^3*(sT : ℂ)^3/8 + 3*(Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^3/8 + 3*(Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/8) * h2 + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/4 + 3*(Real.sqrt 2 : ℂ)*(sT : ℂ)^3/4 - (Real.sqrt 2 : ℂ)*(sT : ℂ)/3) * h3 + (3*(Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/2 + 5*(Real.sqrt 2 : ℂ)*(sT : ℂ)/2) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^3*Complex.I/8 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^3/8 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/8) * h2 + ((Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/4 - (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/6 - (Real.sqrt 2 : ℂ)*(sT : ℂ)/6) * h3 + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)*Complex.I/2 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/2 + (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I + (Real.sqrt 2 : ℂ)*(sT : ℂ)) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3*Complex.I^2/8 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I^2/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/8) * h2 + ((Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I^2/12 - (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/12) * h3 + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)^3/4 + (Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/12 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/4 + (Real.sqrt 2 : ℂ)*(sT : ℂ)/4) * hI + ((Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)*Complex.I/2 + (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/2) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I^3/8 + 3*(Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I^2/8 + 3*(Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/8 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3/8) * h2 + (-(Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I^2/12 - (Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/12) * h3 + (-(Real.sqrt 2 : ℂ)*(Real.sqrt 3 : ℂ)*(sT : ℂ)/12 + (Real.sqrt 2 : ℂ)*(sT : ℂ)^3*Complex.I/4 + 3*(Real.sqrt 2 : ℂ)*(sT : ℂ)^3/4 - (Real.sqrt 2 : ℂ)*(sT : ℂ)/4) * hI + ((Real.sqrt 2 : ℂ)*(sT : ℂ)*Complex.I/2 - (Real.sqrt 2 : ℂ)*(sT : ℂ)/2) * hs

/-- **χ(|T⟩^⊗3) ≤ 3 against `IsStabP 2`.** -/
theorem qubit_t_m3_stabRankP_le_three : stabRankP 2 (tVec 3) ≤ 3 :=
  stabRankP_le_of_terms 2 ![tT3_0, tT3_1, tT3_2] tCoef3
    (fun j => by fin_cases j; exacts [tT3_0_isStab, tT3_1_isStab, tT3_2_isStab]) tVec3_eq

end StabRank
