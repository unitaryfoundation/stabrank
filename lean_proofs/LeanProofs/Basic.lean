/-
Foundation: cube root of unity and basic identities.
-/
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

namespace StabRank

open Complex Real

/-- Primitive 3rd root of unity. -/
noncomputable def omega3 : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

/-- `ω³ = 1`. -/
theorem omega3_pow_three : omega3 ^ 3 = 1 := by
  unfold omega3
  rw [← Complex.exp_nat_mul]
  have : (3 : ℕ) * (2 * (Real.pi : ℂ) * Complex.I / 3) = 2 * Real.pi * Complex.I := by
    push_cast; ring
  rw [this, Complex.exp_two_pi_mul_I]

/-- Helper: `cos(2π/3) = -1/2`. -/
theorem cos_two_pi_div_three : Real.cos (2 * Real.pi / 3) = -(1/2) := by
  have h : 2 * Real.pi / 3 = Real.pi - Real.pi / 3 := by ring
  rw [h, Real.cos_pi_sub, Real.cos_pi_div_three]

/-- Helper: `sin(2π/3) = √3/2`. -/
theorem sin_two_pi_div_three : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  have h : 2 * Real.pi / 3 = Real.pi - Real.pi / 3 := by ring
  rw [h, Real.sin_pi_sub, Real.sin_pi_div_three]

/-- Explicit form: real and imaginary parts of `ω`. -/
theorem omega3_re : omega3.re = -(1/2) := by
  unfold omega3
  rw [show 2 * (Real.pi : ℂ) * Complex.I / 3 = (2 * Real.pi / 3 : ℝ) * Complex.I by push_cast; ring,
      Complex.exp_mul_I, Complex.add_re, Complex.cos_ofReal_re,
      Complex.mul_I_re, Complex.sin_ofReal_im]
  simp [cos_two_pi_div_three]

theorem omega3_im : omega3.im = Real.sqrt 3 / 2 := by
  unfold omega3
  rw [show 2 * (Real.pi : ℂ) * Complex.I / 3 = (2 * Real.pi / 3 : ℝ) * Complex.I by push_cast; ring,
      Complex.exp_mul_I, Complex.add_im, Complex.cos_ofReal_im,
      Complex.mul_I_im, Complex.sin_ofReal_re]
  simp [sin_two_pi_div_three]

/-- The defining cyclotomic relation: `ω² + ω + 1 = 0`. -/
theorem omega3_sq_add_omega3_add_one : omega3 ^ 2 + omega3 + 1 = 0 := by
  have hr := omega3_re
  have hi := omega3_im
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num : (3:ℝ) ≥ 0)
  apply Complex.ext <;>
    simp only [sq, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im,
      Complex.one_re, Complex.one_im, Complex.zero_re, Complex.zero_im, hr, hi]
  · nlinarith [h3]
  · ring

/-- `ω + ω² = -1`. -/
theorem omega3_add_omega3_sq : omega3 + omega3 ^ 2 = -1 := by
  linear_combination omega3_sq_add_omega3_add_one

/-- `ω² = -1 - ω`. -/
theorem omega3_sq_eq : omega3 ^ 2 = -1 - omega3 := by
  linear_combination omega3_sq_add_omega3_add_one

/-- `ω - ω² = i·√3` — both have Re = 0 and Im = √3.  Used wherever
    a `linear_combination` block mixes ω-arithmetic with `i`. -/
theorem omega3_diff_eq_I_sqrt3 :
    omega3 - omega3 ^ 2 = Complex.I * (Real.sqrt 3 : ℂ) := by
  have hsub : omega3 - omega3 ^ 2 = 2 * omega3 + 1 := by rw [omega3_sq_eq]; ring
  rw [hsub]
  apply Complex.ext
  · simp [Complex.add_re, Complex.mul_re, Complex.one_re, Complex.I_re, Complex.I_im,
          Complex.ofReal_re, Complex.ofReal_im, omega3_re, omega3_im]
  · simp [Complex.add_im, Complex.mul_im, Complex.one_im, Complex.I_re, Complex.I_im,
          Complex.ofReal_re, Complex.ofReal_im, omega3_re, omega3_im]
    ring

end StabRank
