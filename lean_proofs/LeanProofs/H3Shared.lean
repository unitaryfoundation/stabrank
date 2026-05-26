/-
Shared utilities for the H_3-orbit pointwise proofs (paper App A.2).

Contains the H_3-specific constants and algebraic helper lemmas used
by both `H3M2Pointwise.lean` and `H3M3Pointwise.lean`. Extracted so
that the m=2 file does not have to depend on the heavier m=3 file
purely to reuse `cH3C`, `NH3C`, and their basic properties.
-/
import LeanProofs.Basic

namespace StabRank

open Complex Real

/-- c = (√3 - 1)/2. -/
noncomputable def cH3C : ℂ := ((Real.sqrt 3 : ℂ) - 1) / 2

/-- N = √(3 - √3). -/
noncomputable def NH3C : ℂ := ((Real.sqrt (3 - Real.sqrt 3) : ℝ) : ℂ)

/-- 1-qutrit H_3 amplitude: H_3 = (N/2)(|0⟩ + |+⟩). -/
noncomputable def h3Amp1 : Fin 3 → ℂ
  | 0 => NH3C / 2 * (1 + 1 / (Real.sqrt 3 : ℂ))
  | 1 => NH3C / (2 * (Real.sqrt 3 : ℂ))
  | 2 => NH3C / (2 * (Real.sqrt 3 : ℂ))

theorem sqrt3_sq_cH : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 3 := by
  exact_mod_cast Real.mul_self_sqrt (by norm_num : (3:ℝ) ≥ 0)

theorem sqrt3_lt_3 : Real.sqrt 3 < 3 := by
  have h : Real.sqrt 3 < Real.sqrt 9 :=
    Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  have h9 : Real.sqrt 9 = 3 := by
    rw [show (9 : ℝ) = 3^2 from by norm_num, Real.sqrt_sq (by norm_num : (3:ℝ) ≥ 0)]
  linarith

/-- N² = 3 - √3. -/
theorem NH3C_sq : NH3C * NH3C = 3 - (Real.sqrt 3 : ℂ) := by
  unfold NH3C
  have h₁ : (3 : ℝ) - Real.sqrt 3 ≥ 0 := by linarith [sqrt3_lt_3]
  exact_mod_cast Real.mul_self_sqrt h₁

theorem NH3C_ne_zero : NH3C ≠ 0 := by
  intro hz
  have hsq : NH3C * NH3C = 0 := by rw [hz]; ring
  rw [NH3C_sq] at hsq
  -- 3 - √3 ≠ 0 because √3 < 3
  have : (3 : ℂ) - (Real.sqrt 3 : ℂ) = ((3 - Real.sqrt 3 : ℝ) : ℂ) := by push_cast; ring
  rw [this] at hsq
  have hr : (3 - Real.sqrt 3 : ℝ) = 0 := by exact_mod_cast hsq
  linarith [sqrt3_lt_3]

theorem sqrt3_ne_zero : (Real.sqrt 3 : ℂ) ≠ 0 := by
  intro hz
  have hp : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 0 := by rw [hz]; ring
  rw [sqrt3_sq_cH] at hp; norm_num at hp

end StabRank
