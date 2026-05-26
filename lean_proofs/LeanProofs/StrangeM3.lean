/-
Strange m=3 χ≤4 algebraic identities (paper App A.1.2).
-/
import LeanProofs.Basic

namespace StabRank

open Complex Real

/-- α coefficient: α = (3√2 - i√6)/8 = (√6/4)·e^(-iπ/6). -/
noncomputable def alphaStrange : ℂ :=
  (3 * (Real.sqrt 2 : ℂ) - Complex.I * (Real.sqrt 6 : ℂ)) / 8

/-- β coefficient: β = -i√6/4. -/
noncomputable def betaStrange : ℂ := -Complex.I * (Real.sqrt 6 : ℂ) / 4

/-- √6 = √2 · √3 as a complex equality. -/
private theorem sqrt6_cx : (Real.sqrt 6 : ℂ) = (Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) := by
  have h : Real.sqrt 6 = Real.sqrt 2 * Real.sqrt 3 := by
    rw [show (6 : ℝ) = 2 * 3 by norm_num]
    exact Real.sqrt_mul (by norm_num : (2:ℝ) ≥ 0) 3
  rw [show (Real.sqrt 6 : ℂ) = ((Real.sqrt 6 : ℝ) : ℂ) from rfl, h]
  push_cast; ring

private theorem sqrt3_sq_cx : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 3 := by
  have h : Real.sqrt 3 * Real.sqrt 3 = 3 := Real.mul_self_sqrt (by norm_num : (3:ℝ) ≥ 0)
  exact_mod_cast h

private theorem sqrt2_sq_cx : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
  have h : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)
  exact_mod_cast h

/-- Cartesian form: `2·ω = -1 + I·√3`. Avoids the division-by-2 form. -/
private theorem two_mul_omega3 :
    (2 : ℂ) * omega3 = -1 + Complex.I * (Real.sqrt 3 : ℂ) := by
  apply Complex.ext
  · -- Real part
    simp [Complex.mul_re, Complex.add_re, Complex.neg_re,
          Complex.one_re, Complex.I_re, Complex.I_im,
          Complex.ofReal_re, Complex.ofReal_im, omega3_re, omega3_im]
  · -- Imaginary part
    simp [Complex.mul_im, Complex.add_im, Complex.neg_im,
          Complex.one_im, Complex.I_re, Complex.I_im,
          Complex.ofReal_re, Complex.ofReal_im, omega3_re, omega3_im]
    ring

/-- Key scalar identity 2 (simpler one): `β · i√3 = 3/(2√2) = 3√2/4`. -/
theorem strange_m3_beta_identity :
    betaStrange * (Complex.I * (Real.sqrt 3 : ℂ)) = 3 * (Real.sqrt 2 : ℂ) / 4 := by
  unfold betaStrange
  rw [sqrt6_cx]
  have hI : Complex.I * Complex.I = -1 := Complex.I_mul_I
  have h33 : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 3 := sqrt3_sq_cx
  have step :
      (-Complex.I * ((Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ)) / 4) *
        (Complex.I * (Real.sqrt 3 : ℂ))
      = (-(Complex.I * Complex.I)) * (Real.sqrt 2 : ℂ) *
        ((Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ)) / 4 := by ring
  rw [step, hI, h33]
  ring

/-- Key scalar identity 1: `α · (ω² - 1) = -3/(2√2) = -3√2/4`.

    Proof: Substitute ω² = -1 - ω (Basic), √6 = √2·√3, and 2ω = -1 + I·√3.
    The resulting polynomial identity in √2, √3, I closes via the ring
    relations I² = -1 and √3² = 3, packaged as a `linear_combination`. -/
theorem strange_m3_alpha_identity :
    alphaStrange * (omega3 ^ 2 - 1) = -(3 * (Real.sqrt 2 : ℂ) / 4) := by
  unfold alphaStrange
  have h3 : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 3 := sqrt3_sq_cx
  have h6 : (Real.sqrt 6 : ℂ) = (Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) := sqrt6_cx
  have h2ω : (2 : ℂ) * omega3 = -1 + Complex.I * (Real.sqrt 3 : ℂ) := two_mul_omega3
  have hω2 : omega3 ^ 2 = -1 - omega3 := omega3_sq_eq
  rw [hω2, h6]
  linear_combination
    ((Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) / 16) * Complex.I_mul_I +
    (-(Real.sqrt 2 : ℂ) / 16) * h3 +
    ((Real.sqrt 2 : ℂ) * (-3 + Complex.I * (Real.sqrt 3 : ℂ)) / 16) * h2ω

end StabRank
