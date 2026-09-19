/-
The two qubit magic states and the trigonometric facts their decompositions
use.

`|H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩` is the H-type (edge) state and
`|T⟩ = cos β|0⟩ + e^{iπ/4} sin β|1⟩` with `cos 2β = 1/√3` the Bravyi–Kitaev
T-type (face) state, in the conventions of `verify_challenge/stabrank_verify.py`.
`hVec m`, `tVec m` are their `m`-fold tensor powers on digit strings.

The pointwise identities in the qubit cells are polynomial identities in the
atoms `√2`, `√3`, `i`, `cos`, `sin`, modulo the relations collected here:
`cH² = 1/2 + √2/4`, `sH² = 1/2 - √2/4`, `sH cH = √2/4`, and for the T state
`cT² = 1/2 + √3/6`, `sT² = 1/2 - √3/6`, `sT cT = √2√3/6`. Each pair of
squares with the product gives a linear relation between the cosine and the
sine (`cH_eq`, `cT_eq`), which the cells use to eliminate the cosine before
`linear_combination`.
-/
import LeanProofs.Stabilizer.TensorP

namespace StabRank

open Stabilizer

theorem sqrt2_sq_c : (Real.sqrt 2 : ℂ) ^ 2 = 2 := by
  rw [← Complex.ofReal_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  norm_num

/-- `i^n = i^(n-4)` for `n ≥ 4`; with `n` a numeral, `simp` discharges the side
    condition and reduces the exponent, so it does not loop the way
    `i^n = i^(n % 4)` would. -/
theorem I_pow_reduce (n : ℕ) (h : 4 ≤ n) : Complex.I ^ n = Complex.I ^ (n - 4) := by
  conv_lhs => rw [← Nat.sub_add_cancel h, pow_add, Complex.I_pow_four, mul_one]

theorem sqrt3_sq_c : (Real.sqrt 3 : ℂ) ^ 2 = 3 := by
  rw [← Complex.ofReal_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]
  norm_num

/-! ### The H-type state -/

/-- `cos(π/8)`. -/
noncomputable def cH : ℝ := Real.cos (Real.pi / 8)
/-- `sin(π/8)`. -/
noncomputable def sH : ℝ := Real.sin (Real.pi / 8)

/-- The one-qubit H-type amplitude. -/
noncomputable def hAmp1 (a : ZMod 2) : ℂ := if a = 0 then (cH : ℂ) else (sH : ℂ)

/-- `|H⟩^⊗m` on digit strings. -/
noncomputable def hVec (m : ℕ) : Fin (2 ^ m) → ℂ :=
  fun idx => ∏ i, hAmp1 (digitsP 2 m idx i)

theorem cH_sq : (cH : ℂ) ^ 2 = 1 / 2 + (Real.sqrt 2 : ℂ) / 4 := by
  unfold cH
  have h := Real.cos_sq (Real.pi / 8)
  rw [show 2 * (Real.pi / 8) = Real.pi / 4 by ring, Real.cos_pi_div_four] at h
  have h' : Real.cos (Real.pi / 8) ^ 2 = 1 / 2 + Real.sqrt 2 / 4 := by rw [h]; ring
  rw [← Complex.ofReal_pow, h']
  push_cast
  ring

theorem sH_sq : (sH : ℂ) ^ 2 = 1 / 2 - (Real.sqrt 2 : ℂ) / 4 := by
  unfold sH
  have h := Real.sin_sq (Real.pi / 8)
  rw [Real.cos_sq, show 2 * (Real.pi / 8) = Real.pi / 4 by ring, Real.cos_pi_div_four] at h
  have h' : Real.sin (Real.pi / 8) ^ 2 = 1 / 2 - Real.sqrt 2 / 4 := by rw [h]; ring
  rw [← Complex.ofReal_pow, h']
  push_cast
  ring

theorem sH_mul_cH : (sH : ℂ) * (cH : ℂ) = (Real.sqrt 2 : ℂ) / 4 := by
  unfold sH cH
  have h := Real.sin_two_mul (Real.pi / 8)
  rw [show 2 * (Real.pi / 8) = Real.pi / 4 by ring, Real.sin_pi_div_four] at h
  have h' : Real.sin (Real.pi / 8) * Real.cos (Real.pi / 8) = Real.sqrt 2 / 4 := by linarith
  rw [← Complex.ofReal_mul, h']
  push_cast
  ring

/-- `cos(π/8) = (1 + √2) sin(π/8)`, i.e. `tan(π/8) = √2 - 1`. -/
theorem cH_eq : (cH : ℂ) = (sH : ℂ) * ((Real.sqrt 2 : ℂ) + 1) := by
  have hc := cH_sq
  have hs := sH_sq
  have hcs := sH_mul_cH
  linear_combination (2 * (sH : ℂ)) * hc + (-2 * (cH : ℂ)) * hs
    + (-2 * (cH : ℂ) + 2 * (sH : ℂ)) * hcs

/-! ### The T-type state -/

/-- `β = arccos(1/√3) / 2`. -/
noncomputable def tBeta : ℝ := Real.arccos (1 / Real.sqrt 3) / 2
/-- `cos β`. -/
noncomputable def cT : ℝ := Real.cos tBeta
/-- `sin β`. -/
noncomputable def sT : ℝ := Real.sin tBeta

/-- The one-qubit T-type amplitude. -/
noncomputable def tAmp1 (a : ZMod 2) : ℂ :=
  if a = 0 then (cT : ℂ) else Complex.exp (Real.pi / 4 * Complex.I) * (sT : ℂ)

/-- `|T⟩^⊗m` on digit strings. -/
noncomputable def tVec (m : ℕ) : Fin (2 ^ m) → ℂ :=
  fun idx => ∏ i, tAmp1 (digitsP 2 m idx i)

theorem exp_pi_div_four_mul_I :
    Complex.exp (Real.pi / 4 * Complex.I) = (Real.sqrt 2 : ℂ) / 2 * (1 + Complex.I) := by
  have h : (Real.pi : ℂ) / 4 = ((Real.pi / 4 : ℝ) : ℂ) := by push_cast; ring
  rw [h, Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_pi_div_four,
    Real.sin_pi_div_four]
  push_cast
  ring

theorem sqrt3_pos : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)

theorem one_div_sqrt3 : 1 / Real.sqrt 3 = Real.sqrt 3 / 3 := by
  rw [div_eq_div_iff sqrt3_pos.ne' (by norm_num), one_mul, Real.mul_self_sqrt (by norm_num)]

theorem cos_two_tBeta : Real.cos (2 * tBeta) = 1 / Real.sqrt 3 := by
  unfold tBeta
  rw [show 2 * (Real.arccos (1 / Real.sqrt 3) / 2) = Real.arccos (1 / Real.sqrt 3) by ring]
  have h1 : (1 : ℝ) ≤ Real.sqrt 3 := by
    rw [show (1 : ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have h0 : (0 : ℝ) ≤ 1 / Real.sqrt 3 := by positivity
  exact Real.cos_arccos (by linarith) ((div_le_one sqrt3_pos).mpr h1)

theorem sin_two_tBeta : Real.sin (2 * tBeta) = Real.sqrt 2 * Real.sqrt 3 / 3 := by
  unfold tBeta
  rw [show 2 * (Real.arccos (1 / Real.sqrt 3) / 2) = Real.arccos (1 / Real.sqrt 3) by ring,
    Real.sin_arccos, one_div_sqrt3, div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]
  rw [show (1 : ℝ) - 3 / 3 ^ 2 = 2 / 3 by norm_num, Real.sqrt_div (by norm_num),
    div_eq_div_iff sqrt3_pos.ne' (by norm_num), mul_assoc, Real.mul_self_sqrt (by norm_num)]

theorem cT_sq : (cT : ℂ) ^ 2 = 1 / 2 + (Real.sqrt 3 : ℂ) / 6 := by
  unfold cT
  have h := Real.cos_sq tBeta
  rw [cos_two_tBeta, one_div_sqrt3] at h
  have h' : Real.cos tBeta ^ 2 = 1 / 2 + Real.sqrt 3 / 6 := by rw [h]; ring
  rw [← Complex.ofReal_pow, h']
  push_cast
  ring

theorem sT_sq : (sT : ℂ) ^ 2 = 1 / 2 - (Real.sqrt 3 : ℂ) / 6 := by
  unfold sT
  have h := Real.sin_sq tBeta
  rw [Real.cos_sq, cos_two_tBeta, one_div_sqrt3] at h
  have h' : Real.sin tBeta ^ 2 = 1 / 2 - Real.sqrt 3 / 6 := by rw [h]; ring
  rw [← Complex.ofReal_pow, h']
  push_cast
  ring

theorem sT_mul_cT : (sT : ℂ) * (cT : ℂ) = (Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) / 6 := by
  unfold sT cT
  have h := Real.sin_two_mul tBeta
  rw [sin_two_tBeta] at h
  have h' : Real.sin tBeta * Real.cos tBeta = Real.sqrt 2 * Real.sqrt 3 / 6 := by linarith
  rw [← Complex.ofReal_mul, h']
  push_cast
  ring

/-- `cos β = sin β · √2 (√3 + 1) / 2`. -/
theorem cT_eq : (cT : ℂ) = (sT : ℂ) * ((Real.sqrt 2 : ℂ) * ((Real.sqrt 3 : ℂ) + 1) / 2) := by
  have hc := cT_sq
  have hs := sT_sq
  have hcs := sT_mul_cT
  have h2 := sqrt2_sq_c
  linear_combination ((Real.sqrt 2 : ℂ) * (sT : ℂ)) * hc + (-2 * (cT : ℂ)) * hs
    + (-(Real.sqrt 2 : ℂ) * (cT : ℂ) + 2 * (sT : ℂ)) * hcs
    + (-(Real.sqrt 3 : ℂ) * (cT : ℂ) / 6) * h2

end StabRank
