/-
`χ(|T5⟩^⊗2) ≤ 8` against `stabRankP 5`: the eight-term decomposition of
`bounds/T5-m2-upper-8.json`; and `χ(|T5⟩^⊗2) ≤ 5`, the five `Z ⊗ Z`
eigensectors of `bounds/T5-m2-upper-5.json` (second section).

The eight terms (three points, four lines, one full-support state) are
entered as `stabTerm`s. `stabTerm` carries no `1/√(5^k)` normalisation and
`|T5⟩^⊗2` carries `1/5`, so the coefficient on the unnormalised term `j` is
`β_j / 5` with `β_j = c_j √5^(k_j - 2)` for the file's `c_j`; evaluated
exactly in `ℚ(ω₅)`, every `β_j` is an integer combination of `1, ω, ω², ω³`
(`t5Coef2`). The identity is decided at each of the 25 digit strings the way
the one-copy identity is: the sums over `F_5` are expanded, the `if`s and
exponents are computed, the exponents are reduced mod 5, `ω⁴` is rewritten
into the lower powers, and `ring` closes the reduced form.
-/
import LeanProofs.T5M1StabRank

namespace StabRank

open Stabilizer

/-- The eight terms of `bounds/T5-m2-upper-8.json`. -/
noncomputable def t5T2 : Fin 8 → (Fin (5 ^ 2) → ℂ)
  | 0 => stabTerm 5 2 0 ![4, 3] (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i)
      (fun i => Fin.elim0 i)
  | 1 => stabTerm 5 2 1 ![0, 3] ![![1, 0]] ![![3]] ![3]
  | 2 => stabTerm 5 2 1 ![0, 4] ![![1, 0]] ![![0]] ![1]
  | 3 => stabTerm 5 2 0 ![2, 4] (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i)
      (fun i => Fin.elim0 i)
  | 4 => stabTerm 5 2 0 ![3, 1] (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i)
      (fun i => Fin.elim0 i)
  | 5 => stabTerm 5 2 1 ![3, 0] ![![0, 1]] ![![0]] ![4]
  | 6 => stabTerm 5 2 2 0 (idWP 5 2) ![![3, 0], ![0, 3]] ![3, 3]
  | 7 => stabTerm 5 2 1 ![4, 0] ![![0, 1]] ![![3]] ![3]

theorem t5T2_isStab (j : Fin 8) : IsStabP 5 (t5T2 j) := by
  fin_cases j
  · exact isStabP_stabTerm 5 2 0 _ _ _ _ (Function.injective_of_subsingleton _)
  · exact isStabP_stabTerm 5 2 1 _ _ _ _
      (affinePtP_injective_of_pivots _ _ ![0] (by decide +kernel))
  · exact isStabP_stabTerm 5 2 1 _ _ _ _
      (affinePtP_injective_of_pivots _ _ ![0] (by decide +kernel))
  · exact isStabP_stabTerm 5 2 0 _ _ _ _ (Function.injective_of_subsingleton _)
  · exact isStabP_stabTerm 5 2 0 _ _ _ _ (Function.injective_of_subsingleton _)
  · exact isStabP_stabTerm 5 2 1 _ _ _ _
      (affinePtP_injective_of_pivots _ _ ![1] (by decide +kernel))
  · exact isStabP_stabTerm 5 2 2 _ _ _ _ (affinePtP_id_injective 5 2)
  · exact isStabP_stabTerm 5 2 1 _ _ _ _
      (affinePtP_injective_of_pivots _ _ ![1] (by decide +kernel))

/-- The coefficients on the unnormalised terms, `β_j / 5` with
    `β_j ∈ ℤ[ω₅]`. -/
noncomputable def t5Coef2 : Fin 8 → ℂ :=
  ![(-1 + 2 * omega5 - omega5 ^ 2) / 5,
    (-omega5 + omega5 ^ 2) / 5,
    (-2 - omega5 - omega5 ^ 2 - omega5 ^ 3) / 5,
    (-omega5 + 2 * omega5 ^ 2 - omega5 ^ 3) / 5,
    (1 - omega5 - omega5 ^ 2 + omega5 ^ 3) / 5,
    (-omega5 + omega5 ^ 2) / 5,
    1 / 5,
    (-2 - omega5 - omega5 ^ 2 - omega5 ^ 3) / 5]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
set_option maxHeartbeats 1600000 in
-- 25 digit strings, each with four five-term sums to expand and decide
theorem t5Vec2_eq : t5Vec 2 = ∑ j : Fin 8, t5Coef2 j • t5T2 j := by
  funext idx
  simp only [Fin.sum_univ_eight, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, t5Vec, t5Coef2,
    t5T2, stabTerm, Matrix.cons_val, Fin.prod_univ_two]
  rw [t5Amp_mul]
  generalize hx : digitsP 5 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta_z5 x⟩
  simp only [stabVecP_zero, stabVecP_five_k1, stabVecP_id]
  rcases zmod5_cases a with rfl | rfl | rfl | rfl | rfl <;>
    rcases zmod5_cases b with rfl | rfl | rfl | rfl | rfl <;>
    simp (config := { decide := true }) [cubeExp_zero, cubeExp_one, cubeExp_two, cubeExp_three,
      cubeExp_four, quadPhaseP, affinePtP, Fin.sum_univ_one, Fin.sum_univ_two, zmod5_val_zero,
      zmod5_val_one, zmod5_val_two, zmod5_val_three, zmod5_val_four, zmodP5_val_zero,
      zmodP5_val_one, zmodP5_val_two, zmodP5_val_three, zmodP5_val_four, stabPeriod_five_div,
      zeta_five, omega5_pow_reduce]
  all_goals (try ring_nf)
  all_goals (try simp only [omega5_pow_reduce, omega5_pow_four, Nat.reduceSub, Nat.reduceLeDiff,
    pow_zero])
  all_goals ring

/-- **χ(|T5⟩^⊗2) ≤ 8 against `IsStabP 5`.** -/
theorem t5_m2_stabRankP_le_eight : stabRankP 5 (t5Vec 2) ≤ 8 :=
  stabRankP_le_of_terms 5 t5T2 t5Coef2 t5T2_isStab t5Vec2_eq

/-! ### χ(|T5⟩^⊗2) ≤ 5: the `Z ⊗ Z` eigensectors

On the line `x + y = c` the cubic `x³ + y³` is quadratic in the line
parameter, `x³ + (c - x)³ = c³ + 3c x² - 3c² x`, so the projection of
`|T5⟩^⊗2` onto the eigenspace `Z ⊗ Z = ω^c` is `ω^(c³)/√5` times the
stabilizer state `5^(-1/2) Σ_x ω^(3c x² - 3c² x) |x, c - x⟩`. The five
sectors are the five terms of `bounds/T5-m2-upper-5.json`, found by the
census of `research/t5q_m2_rank5`; with the rank-4 exclusion of
`bounds/T5-m2-lower-5.json` this settles `χ(|T5⟩^⊗2) = 5`. -/

/-- The five terms of `bounds/T5-m2-upper-5.json`: the line `x + y = c`
    parametrized by `x`, with phase `ω^(3c x² - 3c² x)`, for `c = 0, …, 4`. -/
noncomputable def t5T2s : Fin 5 → (Fin (5 ^ 2) → ℂ)
  | 0 => stabTerm 5 2 1 ![0, 0] ![![1, 4]] ![![0]] ![0]
  | 1 => stabTerm 5 2 1 ![0, 1] ![![1, 4]] ![![3]] ![2]
  | 2 => stabTerm 5 2 1 ![0, 2] ![![1, 4]] ![![1]] ![3]
  | 3 => stabTerm 5 2 1 ![0, 3] ![![1, 4]] ![![4]] ![3]
  | 4 => stabTerm 5 2 1 ![0, 4] ![![1, 4]] ![![2]] ![2]

theorem t5T2s_isStab (j : Fin 5) : IsStabP 5 (t5T2s j) := by
  fin_cases j <;>
    exact isStabP_stabTerm 5 2 1 _ _ _ _
      (affinePtP_injective_of_pivots _ _ ![0] (by decide +kernel))

/-- The coefficients on the unnormalized terms, `ω^(c³) / 5`: the file's
    `ω^(c³)/√5` on the `1/√5`-normalised lines, against the `1/5` of
    `|T5⟩^⊗2`. -/
noncomputable def t5Coef2s : Fin 5 → ℂ :=
  ![1 / 5, omega5 / 5, omega5 ^ 3 / 5, omega5 ^ 2 / 5, omega5 ^ 4 / 5]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
set_option maxHeartbeats 1600000 in
-- 25 digit strings, each with five five-term sums to expand and decide
theorem t5Vec2_eq_sectors : t5Vec 2 = ∑ j : Fin 5, t5Coef2s j • t5T2s j := by
  funext idx
  simp only [Fin.sum_univ_five, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, t5Vec, t5Coef2s,
    t5T2s, stabTerm, Matrix.cons_val, Fin.prod_univ_two]
  rw [t5Amp_mul]
  generalize hx : digitsP 5 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta_z5 x⟩
  simp only [stabVecP_five_k1]
  rcases zmod5_cases a with rfl | rfl | rfl | rfl | rfl <;>
    rcases zmod5_cases b with rfl | rfl | rfl | rfl | rfl <;>
    simp (config := { decide := true }) [cubeExp_zero, cubeExp_one, cubeExp_two, cubeExp_three,
      cubeExp_four, quadPhaseP, affinePtP, Fin.sum_univ_one, Fin.sum_univ_two, zmod5_val_zero,
      zmod5_val_one, zmod5_val_two, zmod5_val_three, zmod5_val_four, zmodP5_val_zero,
      zmodP5_val_one, zmodP5_val_two, zmodP5_val_three, zmodP5_val_four, stabPeriod_five_div,
      zeta_five, omega5_pow_reduce]
  all_goals (try ring_nf)
  all_goals (try simp only [omega5_pow_reduce, omega5_pow_four, Nat.reduceSub, Nat.reduceLeDiff,
    pow_zero])
  all_goals ring

/-- **χ(|T5⟩^⊗2) ≤ 5 against `IsStabP 5`.** -/
theorem t5_m2_stabRankP_le_five : stabRankP 5 (t5Vec 2) ≤ 5 :=
  stabRankP_le_of_terms 5 t5T2s t5Coef2s t5T2s_isStab t5Vec2_eq_sectors

end StabRank
