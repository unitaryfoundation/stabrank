/-
The qubit targets of the reflection proofs: `|H⟩^⊗m` and `|T⟩^⊗m` as `ev` of
an integer vector times a scalar.

At a digit string with `a` zero digits, `hVec m` is `cos(π/8)^a sin(π/8)^(m-a)`
and `tVec m` is `cos β^a (e^{iπ/4} sin β)^(m-a)`. With `cos(π/8) = (1+√2)
sin(π/8)`, `sin²(π/8) = (2-√2)/4`, `cos β = sin β · √2(√3+1)/2`,
`e^{iπ/4} = √2(1+i)/2` and `sin² β = (3-√3)/6` (all from `QubitShared`) these
are

  `sin(π/8)^(m%2) / 4^(m/2) · (1+√2)^a (2-√2)^(m/2)`               (`tgtH`)
  `sin β^(m%2) / (2^m 6^(m/2)) · (√2√3+√2)^a (√2+i√2)^(m-a) (3-√3)^(m/2)`  (`tgtT`)

with the second factor an element of `ℤ[√2, i]`, respectively `ℤ[√2, √3, i]`,
computed as iterates of the matrices of `ReflectBases` (`hVec_eq_ev`,
`tVec_eq_ev`). The generated cell modules (`QubitHM*StabRank`,
`QubitTM*StabRank`) import this file.
-/
import LeanProofs.ReflectBases

namespace StabRank

open Stabilizer

/-! ### The H-type state -/

theorem hVec_eq_pow (m : ℕ) (idx : Fin (2 ^ m)) :
    hVec m idx = (cH : ℂ) ^ count0 (digitsP 2 m idx) * (sH : ℂ) ^ (m - count0 (digitsP 2 m idx)) := by
  unfold hVec hAmp1
  exact prod_two_valued (digitsP 2 m idx) _ _

/-- `(1+√2)^a (2-√2)^(m/2)` as an integer vector in `ℤ[√2, i]`. -/
def tgtH (m : ℕ) (x : Fin m → ZMod 2) : Fin 4 → ℤ :=
  (mulM M1ps2)^[count0 x] ((mulM M2ms2)^[m / 2] (unitZ 1))

theorem h_alg (a b e h m : ℕ) (hab : a + b = m) (heh : e + 2 * h = m) :
    (cH : ℂ) ^ a * (sH : ℂ) ^ b
      = (sH : ℂ) ^ e / 4 ^ h * ((1 + (Real.sqrt 2 : ℂ)) ^ a
        * ((2 - (Real.sqrt 2 : ℂ)) ^ h * ((1 : ℤ) : ℂ))) := by
  subst hab
  have hs : (sH : ℂ) ^ (a + b) = (sH : ℂ) ^ e * ((2 - (Real.sqrt 2 : ℂ)) / 4) ^ h := by
    rw [← heh, pow_add, pow_mul, sH_sq,
      show (1 / 2 - (Real.sqrt 2 : ℂ) / 4) = (2 - Real.sqrt 2) / 4 by ring]
  rw [cH_eq, mul_pow, show ((Real.sqrt 2 : ℂ) + 1) = 1 + Real.sqrt 2 by ring]
  calc (sH : ℂ) ^ a * (1 + (Real.sqrt 2 : ℂ)) ^ a * (sH : ℂ) ^ b
      = (sH : ℂ) ^ (a + b) * (1 + (Real.sqrt 2 : ℂ)) ^ a := by
        rw [pow_add]
        ring
    _ = _ := by
        rw [hs, div_pow]
        push_cast
        ring

/-- **`|H⟩^⊗m` reflected**: the amplitude is `sin(π/8)^(m%2) / 4^(m/2)` times
    the evaluation of `tgtH`. -/
theorem hVec_eq_ev (m : ℕ) (idx : Fin (2 ^ m)) :
    hVec m idx = (sH : ℂ) ^ (m % 2) / 4 ^ (m / 2) * ev B4 (tgtH m (digitsP 2 m idx)) := by
  rw [hVec_eq_pow]
  unfold tgtH
  rw [ev_mulM_iterate B4 M1ps2_represents', ev_mulM_iterate B4 M2ms2_represents',
    ev_unitZ B4 (by norm_num) B4_zero]
  have hle := count0_le (digitsP 2 m idx)
  exact h_alg _ _ _ _ m (Nat.add_sub_cancel' hle) (Nat.mod_add_div m 2)

/-! ### The T-type state -/

theorem tVec_eq_pow (m : ℕ) (idx : Fin (2 ^ m)) :
    tVec m idx = (cT : ℂ) ^ count0 (digitsP 2 m idx)
      * (Complex.exp (Real.pi / 4 * Complex.I) * (sT : ℂ)) ^ (m - count0 (digitsP 2 m idx)) := by
  unfold tVec tAmp1
  exact prod_two_valued (digitsP 2 m idx) _ _

/-- `(√2√3+√2)^a (√2+i√2)^(m-a) (3-√3)^(m/2)` as an integer vector in
    `ℤ[√2, √3, i]`. -/
def tgtT (m : ℕ) (x : Fin m → ZMod 2) : Fin 8 → ℤ :=
  (mulM Ma8)^[count0 x] ((mulM Mb8)^[m - count0 x] ((mulM Mc8)^[m / 2] (unitZ 1)))

theorem t_alg (a b e h m : ℕ) (hab : a + b = m) (heh : e + 2 * h = m) :
    (cT : ℂ) ^ a * (Complex.exp (Real.pi / 4 * Complex.I) * (sT : ℂ)) ^ b
      = (sT : ℂ) ^ e / (2 ^ m * 6 ^ h)
        * (((Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) + (Real.sqrt 2 : ℂ)) ^ a
          * (((Real.sqrt 2 : ℂ) + Complex.I * (Real.sqrt 2 : ℂ)) ^ b
            * ((3 - (Real.sqrt 3 : ℂ)) ^ h * ((1 : ℤ) : ℂ)))) := by
  subst hab
  have hs : (sT : ℂ) ^ (a + b) = (sT : ℂ) ^ e * ((3 - (Real.sqrt 3 : ℂ)) / 6) ^ h := by
    rw [← heh, pow_add, pow_mul, sT_sq,
      show (1 / 2 - (Real.sqrt 3 : ℂ) / 6) = (3 - Real.sqrt 3) / 6 by ring]
  rw [cT_eq, exp_pi_div_four_mul_I, mul_pow, mul_pow,
    show ((Real.sqrt 2 : ℂ) * ((Real.sqrt 3 : ℂ) + 1) / 2)
      = (Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 2) / 2 by ring,
    show ((Real.sqrt 2 : ℂ) / 2 * (1 + Complex.I))
      = (Real.sqrt 2 + Complex.I * Real.sqrt 2) / 2 by ring,
    div_pow, div_pow]
  calc (sT : ℂ) ^ a * (((Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) + (Real.sqrt 2 : ℂ)) ^ a / 2 ^ a)
        * (((Real.sqrt 2 : ℂ) + Complex.I * (Real.sqrt 2 : ℂ)) ^ b / 2 ^ b * (sT : ℂ) ^ b)
      = (sT : ℂ) ^ (a + b) * (((Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) + (Real.sqrt 2 : ℂ)) ^ a
          * ((Real.sqrt 2 : ℂ) + Complex.I * (Real.sqrt 2 : ℂ)) ^ b) / 2 ^ (a + b) := by
        rw [pow_add, pow_add]
        ring
    _ = _ := by
        rw [hs, div_pow]
        push_cast
        ring

/-- **`|T⟩^⊗m` reflected**: the amplitude is `sin β^(m%2) / (2^m 6^(m/2))`
    times the evaluation of `tgtT`. -/
theorem tVec_eq_ev (m : ℕ) (idx : Fin (2 ^ m)) :
    tVec m idx = (sT : ℂ) ^ (m % 2) / (2 ^ m * 6 ^ (m / 2)) * ev B8 (tgtT m (digitsP 2 m idx)) := by
  rw [tVec_eq_pow]
  unfold tgtT
  rw [ev_mulM_iterate B8 Ma8_represents', ev_mulM_iterate B8 Mb8_represents',
    ev_mulM_iterate B8 Mc8_represents', ev_unitZ B8 (by norm_num) B8_zero]
  have hle := count0_le (digitsP 2 m idx)
  exact t_alg _ _ _ _ m (Nat.add_sub_cancel' hle) (Nat.mod_add_div m 2)

end StabRank
