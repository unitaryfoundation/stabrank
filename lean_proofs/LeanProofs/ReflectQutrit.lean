/-
The qutrit targets of the reflection proofs: `|H₃⟩^⊗m` and `|T₃⟩^⊗m` as
`ev` of an integer vector times a scalar.

`h3Vec m` is the `m`-fold product of the one-qutrit `H₃` amplitude `h3Amp1` of
`H3Shared` on digit strings. Its value at a string with `a` zero digits is
`(N(3+√3)/6)^a (N√3/6)^(m-a)` with `N = √(3-√3)`, which is
`N^(m%2)/6^m` times `(3+√3)^a √3^(m-a) (3-√3)^(m/2)`, an element of
`ℤ[ω₃, √3]` computed by `tgtH3` as iterates of the matrices of `ReflectBases`
(`h3Vec_eq_ev`).

For `T₃` the target is `t3TargetM m` of `T3GaloisM`, the state the lower
bounds of that file are about; its amplitude at a string with digit sum `d`
is `(1/√3)^m ω₉^d`, and `tgtT3` is `ω₉^d` in `ℤ[ω₉, √3]`
(`t3TargetM_eq_ev`).

The generated cell modules (`H3M4StabRank`, `T3M3StabRank`, `T3M4StabRank`)
import this file.
-/
import LeanProofs.ReflectBases
import LeanProofs.H3Shared
import LeanProofs.T3GaloisM

namespace StabRank

open Stabilizer

/-! ### `H₃` -/

/-- `|H₃⟩^⊗m` on digit strings, from the one-qutrit amplitude of `H3Shared`. -/
noncomputable def h3Vec (m : ℕ) : Fin (3 ^ m) → ℂ :=
  fun idx => ∏ i, h3Amp1 (digitsP 3 m idx i)

theorem zmod3_cases (a : ZMod 3) : a = 0 ∨ a = 1 ∨ a = 2 := by decide +revert

theorem h3Amp1_zero : h3Amp1 (0 : ZMod 3) = NH3C / 2 * (1 + 1 / (Real.sqrt 3 : ℂ)) := rfl
theorem h3Amp1_one : h3Amp1 (1 : ZMod 3) = NH3C / (2 * (Real.sqrt 3 : ℂ)) := rfl
theorem h3Amp1_two : h3Amp1 (2 : ZMod 3) = NH3C / (2 * (Real.sqrt 3 : ℂ)) := rfl

/-- The one-qutrit amplitude by whether the digit vanishes: `N(3+√3)/6` at `0`
    and `N√3/6` at `1`, `2`. -/
theorem h3Amp1_eq (v : ZMod 3) :
    h3Amp1 v = if v = 0 then NH3C * (3 + (Real.sqrt 3 : ℂ)) / 6
      else NH3C * (Real.sqrt 3 : ℂ) / 6 := by
  have h3 := sqrt3_sq_cH
  have hne := sqrt3_ne_zero
  rcases zmod3_cases v with rfl | rfl | rfl
  · rw [h3Amp1_zero, if_pos rfl]
    field_simp
    linear_combination (-2 * NH3C) * h3
  · rw [h3Amp1_one, if_neg (by decide)]
    field_simp
    linear_combination (-2 * NH3C) * h3
  · rw [h3Amp1_two, if_neg (by decide)]
    field_simp
    linear_combination (-2 * NH3C) * h3

theorem h3Vec_eq_pow (m : ℕ) (idx : Fin (3 ^ m)) :
    h3Vec m idx = (NH3C * (3 + (Real.sqrt 3 : ℂ)) / 6) ^ count0 (digitsP 3 m idx)
      * (NH3C * (Real.sqrt 3 : ℂ) / 6) ^ (m - count0 (digitsP 3 m idx)) := by
  unfold h3Vec
  simp only [h3Amp1_eq]
  exact prod_two_valued (digitsP 3 m idx) _ _

/-- `(3+√3)^a √3^(m-a) (3-√3)^(m/2)` as an integer vector in `ℤ[ω₃, √3]`. -/
def tgtH3 (m : ℕ) (x : Fin m → ZMod 3) : Fin 4 → ℤ :=
  (mulM M3ps3)^[count0 x] ((mulM Ms3)^[m - count0 x] ((mulM M3ms3)^[m / 2] (unitZ 1)))

theorem h3_alg (a b e h m : ℕ) (hab : a + b = m) (heh : e + 2 * h = m) :
    (NH3C * (3 + (Real.sqrt 3 : ℂ)) / 6) ^ a * (NH3C * (Real.sqrt 3 : ℂ) / 6) ^ b
      = NH3C ^ e / 6 ^ m
        * ((3 + (Real.sqrt 3 : ℂ)) ^ a * ((Real.sqrt 3 : ℂ) ^ b
          * ((3 - (Real.sqrt 3 : ℂ)) ^ h * ((1 : ℤ) : ℂ)))) := by
  subst hab
  have hN : NH3C ^ (a + b) = NH3C ^ e * (3 - (Real.sqrt 3 : ℂ)) ^ h := by
    rw [← heh, pow_add, pow_mul, sq, NH3C_sq]
  rw [div_pow, div_pow, mul_pow, mul_pow]
  calc NH3C ^ a * (3 + (Real.sqrt 3 : ℂ)) ^ a / 6 ^ a * (NH3C ^ b * (Real.sqrt 3 : ℂ) ^ b / 6 ^ b)
      = NH3C ^ (a + b) * ((3 + (Real.sqrt 3 : ℂ)) ^ a * (Real.sqrt 3 : ℂ) ^ b) / 6 ^ (a + b) := by
        rw [pow_add, pow_add]
        ring
    _ = _ := by
        rw [hN]
        push_cast
        ring

/-- **`|H₃⟩^⊗m` reflected**: the amplitude is `N^(m%2)/6^m` times the
    evaluation of `tgtH3`. -/
theorem h3Vec_eq_ev (m : ℕ) (idx : Fin (3 ^ m)) :
    h3Vec m idx = NH3C ^ (m % 2) / 6 ^ m * ev B3 (tgtH3 m (digitsP 3 m idx)) := by
  rw [h3Vec_eq_pow]
  unfold tgtH3
  rw [ev_mulM_iterate B3 M3ps3_represents', ev_mulM_iterate B3 Ms3_represents',
    ev_mulM_iterate B3 M3ms3_represents', ev_unitZ B3 (by norm_num) B3_zero]
  have hle := count0_le (digitsP 3 m idx)
  exact h3_alg _ _ _ _ m (Nat.add_sub_cancel' hle) (Nat.mod_add_div m 2)

/-! ### `T₃` -/

/-- `ω₉^(digit sum)` as an integer vector in `ℤ[ω₉, √3]`. -/
def tgtT3 (m : ℕ) (idx : Fin (3 ^ m)) : Fin 12 → ℤ :=
  (mulM Mw9)^[digitSum m idx] (unitZ 1)

theorem t3TargetM_apply (m : ℕ) (idx : Fin (3 ^ m)) :
    t3TargetM m idx = ((1 / Real.sqrt 3 : ℝ) : ℂ) ^ m * omega9 ^ digitSum m idx := by
  simp [t3TargetM, tConjM, galoisExp]

/-- **`|T₃⟩^⊗m` reflected**: the amplitude is `(1/√3)^m` times the evaluation
    of `tgtT3`. -/
theorem t3TargetM_eq_ev (m : ℕ) (idx : Fin (3 ^ m)) :
    t3TargetM m idx = ((1 / Real.sqrt 3 : ℝ) : ℂ) ^ m * ev B9 (tgtT3 m idx) := by
  rw [t3TargetM_apply]
  unfold tgtT3
  rw [ev_mulM_iterate B9 Mw9_represents', ev_unitZ B9 (by norm_num) B9_zero]
  push_cast
  ring

end StabRank
