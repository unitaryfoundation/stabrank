/-
`χ(|T⟩^⊗4) ≤ 3` against `stabRankP 2`: the three-term decomposition of
`QubitTM4.lean` (Proposition C.4 of Labib and Russo), restated against the
concrete qudit stabilizer predicate. `QubitTM4.stabilizer_rank_le_three` is
kept as it was.

The witnesses are those of `bounds/qubit_T-m4-upper-3.json`, entered term by term as `stabTerm 2 n k
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

/-! ### m = 4 -/

noncomputable def tT4_0 : Fin (2 ^ 4) → ℂ :=
  stabTerm 2 4 3 ![0, 0, 0, 0] ![![1, 0, 1, 0], ![0, 1, 0, 0], ![0, 0, 0, 1]] ![![0, 0, 0], ![0, 0, 1], ![0, 0, 0]] ![1, 0, 0]
noncomputable def tT4_1 : Fin (2 ^ 4) → ℂ :=
  stabTerm 2 4 4 0 (idWP 2 4) ![![0, 0, 1, 0], ![0, 0, 0, 1], ![0, 0, 0, 0], ![0, 0, 0, 0]] ![0, 1, 0, 1]
noncomputable def tT4_2 : Fin (2 ^ 4) → ℂ :=
  stabTerm 2 4 3 ![0, 0, 0, 0] ![![1, 0, 0, 0], ![0, 1, 0, 1], ![0, 0, 1, 0]] ![![0, 0, 1], ![0, 0, 0], ![0, 0, 0]] ![1, 1, 1]

theorem tT4_0_isStab : IsStabP 2 tT4_0 :=
  isStabP_stabTerm 2 4 3 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0, 1, 3] (by decide +kernel))
theorem tT4_1_isStab : IsStabP 2 tT4_1 :=
  isStabP_stabTerm 2 4 4 _ _ _ _ (affinePtP_id_injective 2 4)
theorem tT4_2_isStab : IsStabP 2 tT4_2 :=
  isStabP_stabTerm 2 4 3 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0, 1, 2] (by decide +kernel))

noncomputable def tCoef4 : Fin 3 → ℂ :=
  ![(Real.sqrt 3 : ℂ)*Complex.I/12 + (Real.sqrt 3 : ℂ)/12 - Complex.I/12 + 1/12, 1/6, -(Real.sqrt 3 : ℂ)*Complex.I/12 + (Real.sqrt 3 : ℂ)/12 + Complex.I/12 + 1/12]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
set_option linter.style.longLine false in
set_option maxHeartbeats 1600000 in
theorem tVec4_eq : tVec 4 = ∑ j : Fin 3, tCoef4 j • ![tT4_0, tT4_1, tT4_2] j := by
  have hs := sT_sq
  have h2 := sqrt2_sq_c
  have h3 := sqrt3_sq_c
  have hI := Complex.I_sq
  have hlin := cT_eq
  funext idx
  simp only [Fin.sum_univ_three, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, tVec, tCoef4,
    tT4_0, tT4_1, tT4_2, stabTerm, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons, Matrix.cons_val_three]
  generalize hx : digitsP 2 4 idx = x
  obtain ⟨a, b, c, d, rfl⟩ : ∃ a b c d, x = ![a, b, c, d] :=
    ⟨x 0, x 1, x 2, x 3, fin4_eta_z x⟩
  simp only [stabVecP_id, stabVecP_two_k3]
  rcases zmod2_cases a with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;> rcases zmod2_cases c with rfl | rfl <;> rcases zmod2_cases d with rfl | rfl <;>
    simp (config := { decide := true }) [tAmp1, exp_pi_div_four_mul_I, quadPhaseP, Fin.sum_univ_four, Fin.sum_univ_three,
      Fin.prod_univ_four, zmod2_val_zero, zmod2_val_one, zmod4_val_zero, zmod4_val_one, zmod4_val_two,
      zmod4_val_three, stabPeriod_two_div, zeta_two_pow, zeta_two, Complex.I_sq, I_pow_three,
      I_pow_reduce, hlin]
  all_goals first
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/8 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/8 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/8 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/8 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/2 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 + (sT : ℂ)^4*Complex.I^2/8 + (sT : ℂ)^4*Complex.I/4 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)*Complex.I/36 + (sT : ℂ)^4*Complex.I^2/4 + (sT : ℂ)^4*Complex.I/2 + (sT : ℂ)^4/4 - Complex.I/9) * h3 + ((Real.sqrt 3 : ℂ)*(sT : ℂ)^4/2 + (sT : ℂ)^4) * hI + (-(Real.sqrt 3 : ℂ)^2*Complex.I/6 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I + (Real.sqrt 3 : ℂ)*Complex.I/6 + 2*(sT : ℂ)^2*Complex.I + Complex.I) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4/8 + 3*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + 3*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/8 + (sT : ℂ)^4*Complex.I/8 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 + (Real.sqrt 3 : ℂ)*Complex.I/24 + (Real.sqrt 3 : ℂ)/24 + 3*(sT : ℂ)^4*Complex.I/4 + 3*(sT : ℂ)^4/4 - 13*Complex.I/72 - 13/72) * h3 + (-(Real.sqrt 3 : ℂ)^2*Complex.I/4 - (Real.sqrt 3 : ℂ)^2/4 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I/2 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^2/2 + (Real.sqrt 3 : ℂ)*Complex.I/3 + (Real.sqrt 3 : ℂ)/3 + 5*(sT : ℂ)^2*Complex.I/2 + 5*(sT : ℂ)^2/2 + 5*Complex.I/4 + 5/4) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4/8 + 3*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + 3*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/8 + (sT : ℂ)^4*Complex.I/8 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 + (Real.sqrt 3 : ℂ)*Complex.I/24 + (Real.sqrt 3 : ℂ)/24 + 3*(sT : ℂ)^4*Complex.I/4 + 3*(sT : ℂ)^4/4 - 13*Complex.I/72 - 13/72) * h3 + ((Real.sqrt 3 : ℂ)/12 - 1/12) * hI + (-(Real.sqrt 3 : ℂ)^2*Complex.I/4 - (Real.sqrt 3 : ℂ)^2/4 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I/2 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^2/2 + (Real.sqrt 3 : ℂ)*Complex.I/3 + (Real.sqrt 3 : ℂ)/3 + 5*(sT : ℂ)^2*Complex.I/2 + 5*(sT : ℂ)^2/2 + 5*Complex.I/4 + 5/4) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^3/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^3/16 + 3*(Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + 3*(Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^3/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/8 + (sT : ℂ)^4*Complex.I^3/8 + 3*(sT : ℂ)^4*Complex.I^2/8 + 3*(sT : ℂ)^4*Complex.I/8 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)*Complex.I/72 - (Real.sqrt 3 : ℂ)/72 - 5*Complex.I/72 + 5/72) * h3 + ((Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/4 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 + (sT : ℂ)^4*Complex.I/4 + 3*(sT : ℂ)^4/4) * hI + (-(Real.sqrt 3 : ℂ)^2*Complex.I/12 + (Real.sqrt 3 : ℂ)^2/12 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I/2 - (Real.sqrt 3 : ℂ)*(sT : ℂ)^2/2 + (Real.sqrt 3 : ℂ)*Complex.I/6 - (Real.sqrt 3 : ℂ)/6 + (sT : ℂ)^2*Complex.I/2 - (sT : ℂ)^2/2 + Complex.I/4 - 1/4) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^3/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/16 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^3/16 + 3*(Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + 3*(Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^3/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/8 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/8 + (sT : ℂ)^4*Complex.I^3/8 + 3*(sT : ℂ)^4*Complex.I^2/8 + 3*(sT : ℂ)^4*Complex.I/8 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)*Complex.I/72 - (Real.sqrt 3 : ℂ)/72 - 5*Complex.I/72 + 5/72) * h3 + ((Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/4 + 3*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 - (Real.sqrt 3 : ℂ)/12 + (sT : ℂ)^4*Complex.I/4 + 3*(sT : ℂ)^4/4 + 1/12) * hI + (-(Real.sqrt 3 : ℂ)^2*Complex.I/12 + (Real.sqrt 3 : ℂ)^2/12 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I/2 - (Real.sqrt 3 : ℂ)*(sT : ℂ)^2/2 + (Real.sqrt 3 : ℂ)*Complex.I/6 - (Real.sqrt 3 : ℂ)/6 + (sT : ℂ)^2*Complex.I/2 - (sT : ℂ)^2/2 + Complex.I/4 - 1/4) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4/4 + 3*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/8 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)^4*(sT : ℂ)^4/8 + (Real.sqrt 3 : ℂ)^3*(sT : ℂ)^4/2 + 3*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/2 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4 + (Real.sqrt 3 : ℂ)/9 + 9*(sT : ℂ)^4/4 - 17/36) * h3 + (-2*(Real.sqrt 3 : ℂ)^2/3 + 4*(Real.sqrt 3 : ℂ)*(sT : ℂ)^2 + 5*(Real.sqrt 3 : ℂ)/6 + 7*(sT : ℂ)^2 + 7/2) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/8 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/8 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/8 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/8 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/2 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 + (sT : ℂ)^4*Complex.I^2/8 + (sT : ℂ)^4*Complex.I/4 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)*Complex.I/36 + (sT : ℂ)^4*Complex.I^2/4 + (sT : ℂ)^4*Complex.I/2 + (sT : ℂ)^4/4 - Complex.I/9) * h3 + ((Real.sqrt 3 : ℂ)*(sT : ℂ)^4/2 + (Real.sqrt 3 : ℂ)/12 + (sT : ℂ)^4 - 1/12) * hI + (-(Real.sqrt 3 : ℂ)^2*Complex.I/6 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I + (Real.sqrt 3 : ℂ)*Complex.I/6 + 2*(sT : ℂ)^2*Complex.I + Complex.I) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/8 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)*(sT : ℂ)^4/8 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/8 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/8 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 3 : ℂ)^2*(sT : ℂ)^4/8 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I^2/4 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4*Complex.I/2 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^4/4 + (sT : ℂ)^4*Complex.I^2/8 + (sT : ℂ)^4*Complex.I/4 + (sT : ℂ)^4/8) * h2 + ((Real.sqrt 3 : ℂ)*Complex.I/36 + (sT : ℂ)^4*Complex.I^2/4 + (sT : ℂ)^4*Complex.I/2 + (sT : ℂ)^4/4 - Complex.I/9) * h3 + ((Real.sqrt 3 : ℂ)*(sT : ℂ)^4/2 - (Real.sqrt 3 : ℂ)/12 + (sT : ℂ)^4 + 1/12) * hI + (-(Real.sqrt 3 : ℂ)^2*Complex.I/6 + (Real.sqrt 3 : ℂ)*(sT : ℂ)^2*Complex.I + (Real.sqrt 3 : ℂ)*Complex.I/6 + 2*(sT : ℂ)^2*Complex.I + Complex.I) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^4/16 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^3/4 + 3*(Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I^2/8 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4*Complex.I/4 + (Real.sqrt 2 : ℂ)^2*(sT : ℂ)^4/16 + (sT : ℂ)^4*Complex.I^4/8 + (sT : ℂ)^4*Complex.I^3/2 + 3*(sT : ℂ)^4*Complex.I^2/4 + (sT : ℂ)^4*Complex.I/2 + (sT : ℂ)^4/8) * h2 + (-1/36) * h3 + ((Real.sqrt 3 : ℂ)/6 + (sT : ℂ)^4*Complex.I^2/4 + (sT : ℂ)^4*Complex.I + 5*(sT : ℂ)^4/4 - 1/6) * hI + ((Real.sqrt 3 : ℂ)/6 - (sT : ℂ)^2 - 1/2) * hs

/-- **χ(|T⟩^⊗4) ≤ 3 against `IsStabP 2`.** -/
theorem qubit_t_m4_stabRankP_le_three : stabRankP 2 (tVec 4) ≤ 3 :=
  stabRankP_le_of_terms 2 ![tT4_0, tT4_1, tT4_2] tCoef4
    (fun j => by fin_cases j; exacts [tT4_0_isStab, tT4_1_isStab, tT4_2_isStab]) tVec4_eq

end StabRank
