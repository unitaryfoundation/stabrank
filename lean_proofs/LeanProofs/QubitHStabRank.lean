/-
The H-type qubit upper bounds against `stabRankP 2`: `χ(|H⟩^⊗2) ≤ 2`,
`χ(|H⟩^⊗3) ≤ 3`, and `χ(|H⟩^⊗4) ≤ 4` as the tensor square of the first.

The witnesses are those of `bounds/qubit_H-m2-upper-2.json` and
`bounds/qubit_H-m3-upper-3.json`, entered term by term as `stabTerm 2 n k x0
W Q l`; `stabTerm` carries no `1/√(2^k)` normalisation, so each coefficient
is the file's divided by `√(2^k)`. Every term is `IsStabP 2` by
`isStabP_stabTerm` once its parametrisation is injective (pivot columns of
`W`), and the pointwise identity is decided at every digit string: the sums
over `F_2^k` are expanded (`stabVecP_id`, `stabVecP_two_k1`, `stabVecP_zero`),
the `if`s and the exponents are computed, `cos(π/8)` is rewritten as
`(1 + √2) sin(π/8)`, and what remains is a polynomial identity in `√2` and
`sin(π/8)` modulo `√2² = 2` and `sin(π/8)² = 1/2 - √2/4`, closed by
`linear_combination` with cofactors found by polynomial division.
-/
import LeanProofs.QubitShared

namespace StabRank

open Stabilizer

/-! ### m = 2 -/

/-- Full support, `Q(y) = y₀y₁`: `|00⟩ + |01⟩ + |10⟩ - |11⟩`. -/
noncomputable def hT2_0 : Fin (2 ^ 2) → ℂ :=
  stabTerm 2 2 2 0 (idWP 2 2) ![![0, 1], ![0, 0]] ![0, 0]
/-- The line `{(t, t)}`: `|00⟩ + |11⟩`. -/
noncomputable def hT2_1 : Fin (2 ^ 2) → ℂ :=
  stabTerm 2 2 1 ![0, 0] ![![1, 1]] ![![0]] ![0]

theorem hT2_0_isStab : IsStabP 2 hT2_0 :=
  isStabP_stabTerm 2 2 2 _ _ _ _ (affinePtP_id_injective 2 2)
theorem hT2_1_isStab : IsStabP 2 hT2_1 :=
  isStabP_stabTerm 2 2 1 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0] (by decide +kernel))

/-- Coefficients: `√2/2` and `√2/2` in the bound file, divided by `2` and `√2`. -/
noncomputable def hCoef2 : Fin 2 → ℂ := ![(Real.sqrt 2 : ℂ) / 4, 1 / 2]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
set_option linter.style.longLine false in
theorem hVec2_eq : hVec 2 = ∑ j : Fin 2, hCoef2 j • ![hT2_0, hT2_1] j := by
  have hs := sH_sq
  have h2 := sqrt2_sq_c
  have hlin := cH_eq
  funext idx
  simp only [Fin.sum_univ_two, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, hVec, hCoef2,
    hT2_0, hT2_1, stabTerm, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  generalize hx : digitsP 2 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta_z x⟩
  simp only [stabVecP_id, stabVecP_two_k1]
  rcases zmod2_cases a with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;>
    simp (config := { decide := true }) [hAmp1, quadPhaseP, Fin.sum_univ_two, Fin.sum_univ_one,
      zmod2_val_zero, zmod2_val_one, zmod4_val_zero, zmod4_val_one, zmod4_val_two,
      zmod4_val_three, stabPeriod_two_div, zeta_two_pow, zeta_two, Complex.I_sq, I_pow_three,
      I_pow_reduce, hlin]
  all_goals first
    | linear_combination (-1/4) * h2 + ((Real.sqrt 2 : ℂ) + 1) * hs
    | linear_combination ((sH : ℂ)^2 - 1/2) * h2 + (2*(Real.sqrt 2 : ℂ) + 3) * hs
    | linear_combination (1) * hs

/-- **χ(|H⟩^⊗2) ≤ 2 against `IsStabP 2`.** -/
theorem qubit_h_m2_stabRankP_le_two : stabRankP 2 (hVec 2) ≤ 2 :=
  stabRankP_le_of_terms 2 ![hT2_0, hT2_1] hCoef2
    (fun j => by fin_cases j; exacts [hT2_0_isStab, hT2_1_isStab]) hVec2_eq

/-! ### m = 3 -/

/-- The plane `x₀ + x₁ + x₂ = 1`, trivial phase. -/
noncomputable def hT3_0 : Fin (2 ^ 3) → ℂ :=
  stabTerm 2 3 2 ![0, 0, 1] ![![1, 0, 1], ![0, 1, 1]] ![![0, 0], ![0, 0]] ![0, 0]
/-- Full support, phase `(-1)^(y₀y₁ + y₀y₂ + y₁y₂ + y₀ + y₁ + y₂)`. -/
noncomputable def hT3_1 : Fin (2 ^ 3) → ℂ :=
  stabTerm 2 3 3 0 (idWP 2 3) ![![0, 1, 1], ![0, 0, 1], ![0, 0, 0]] ![2, 2, 2]
/-- The point `|000⟩`. -/
noncomputable def hT3_2 : Fin (2 ^ 3) → ℂ :=
  stabTerm 2 3 0 ![0, 0, 0] (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i)

theorem hT3_0_isStab : IsStabP 2 hT3_0 :=
  isStabP_stabTerm 2 3 2 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0, 1] (by decide +kernel))
theorem hT3_1_isStab : IsStabP 2 hT3_1 :=
  isStabP_stabTerm 2 3 3 _ _ _ _ (affinePtP_id_injective 2 3)
theorem hT3_2_isStab : IsStabP 2 hT3_2 :=
  isStabP_stabTerm 2 3 0 _ _ _ _ (Function.injective_of_subsingleton _)

/-- Coefficients: `sin(π/8)`, `-sin(π/8)`, `cos(π/8)` in the bound file, divided
    by `2`, `2√2`, `1`. -/
noncomputable def hCoef3 : Fin 3 → ℂ :=
  ![(sH : ℂ) / 2, -(Real.sqrt 2 : ℂ) * (sH : ℂ) / 4, (cH : ℂ)]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
set_option linter.style.longLine false in
theorem hVec3_eq : hVec 3 = ∑ j : Fin 3, hCoef3 j • ![hT3_0, hT3_1, hT3_2] j := by
  have hs := sH_sq
  have h2 := sqrt2_sq_c
  have hlin := cH_eq
  funext idx
  simp only [Fin.sum_univ_three, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, hVec, hCoef3,
    hT3_0, hT3_1, hT3_2, stabTerm, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons]
  generalize hx : digitsP 2 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta_z x⟩
  simp only [stabVecP_id, stabVecP_two_k2, stabVecP_zero]
  rcases zmod2_cases a with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;>
    rcases zmod2_cases c with rfl | rfl <;>
    simp (config := { decide := true }) [hAmp1, quadPhaseP, Fin.sum_univ_three, Fin.sum_univ_two,
      Fin.prod_univ_three, zmod2_val_zero, zmod2_val_one, zmod4_val_zero, zmod4_val_one, zmod4_val_two,
      zmod4_val_three, stabPeriod_two_div, zeta_two_pow, zeta_two, Complex.I_sq, I_pow_three,
      I_pow_reduce, hlin]
  all_goals first
    | linear_combination ((sH : ℂ)^3 - (sH : ℂ)/2) * h2 + (2*(Real.sqrt 2 : ℂ)*(sH : ℂ) + 3*(sH : ℂ)) * hs
    | linear_combination (-(sH : ℂ)/4) * h2 + ((Real.sqrt 2 : ℂ)*(sH : ℂ) + (sH : ℂ)) * hs
    | linear_combination ((Real.sqrt 2 : ℂ)*(sH : ℂ)^3 + 3*(sH : ℂ)^3 - 5*(sH : ℂ)/4) * h2 + (5*(Real.sqrt 2 : ℂ)*(sH : ℂ) + 7*(sH : ℂ)) * hs
    | linear_combination ((sH : ℂ)) * hs

/-- **χ(|H⟩^⊗3) ≤ 3 against `IsStabP 2`.** -/
theorem qubit_h_m3_stabRankP_le_three : stabRankP 2 (hVec 3) ≤ 3 :=
  stabRankP_le_of_terms 2 ![hT3_0, hT3_1, hT3_2] hCoef3
    (fun j => by fin_cases j; exacts [hT3_0_isStab, hT3_1_isStab, hT3_2_isStab]) hVec3_eq

/-! ### m = 4, as the tensor square of m = 2 -/

/-- `|H⟩^⊗n ⊗ |H⟩^⊗m = |H⟩^⊗(n+m)`. -/
theorem tensorP_hVec (n m : ℕ) : tensorP 2 (hVec n) (hVec m) = hVec (n + m) := by
  funext idx
  simp only [tensorP, hVec, Equiv.apply_symm_apply, Fin.prod_univ_add]
  rfl

/-- **χ(|H⟩^⊗4) ≤ 4 against `IsStabP 2`**, from `χ(|H⟩^⊗2) ≤ 2` and
    multiplicativity. -/
theorem qubit_h_m4_stabRankP_le_four : stabRankP 2 (hVec 4) ≤ 4 := by
  have h := le_trans (stabRankP_tensor_le (p := 2) (hVec 2) (hVec 2))
    (Nat.mul_le_mul qubit_h_m2_stabRankP_le_two qubit_h_m2_stabRankP_le_two)
  rw [tensorP_hVec] at h
  exact h

end StabRank
