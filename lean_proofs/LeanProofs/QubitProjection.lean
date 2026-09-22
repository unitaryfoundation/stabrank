/-
The projection and submultiplicativity bounds of `Stabilizer/SliceP.lean`
instantiated at the qubit magic states `hVec m` and `tVec m` of
`QubitShared.lean`.

`hVec m` and `tVec m` are the tensor powers `powVecP 2 hAmp1 m` and
`powVecP 2 tAmp1 m` by definition, and the one-qubit amplitudes are nonzero
(their `|0⟩` components are `cos(π/8)` and `cos β`, both positive), so
`stabRankP 2 (hVec m) ≤ stabRankP 2 (hVec (m + 1))` and the same for `tVec`.
Combined with a lower bound at some `m`, these give the lower bound at every
larger `m` as a Lean theorem; the board's qubit lower bounds are currently
computational (`cert_qubit_rank2.py`, `cert_qubit_h_m4_rank3.py`, ...), so
nothing on the board changes tier through this file yet.
-/
import LeanProofs.Stabilizer.SliceP
import LeanProofs.QubitShared

namespace StabRank

open Stabilizer

theorem hVec_eq_powVecP (m : ℕ) : hVec m = powVecP 2 hAmp1 m := rfl

theorem tVec_eq_powVecP (m : ℕ) : tVec m = powVecP 2 tAmp1 m := rfl

theorem cH_ne_zero : (cH : ℂ) ≠ 0 := by
  intro h
  have h2 := cH_sq
  rw [h, zero_pow two_ne_zero] at h2
  have h3 : ((1 / 2 + Real.sqrt 2 / 4 : ℝ) : ℂ) = 0 := by
    push_cast
    exact h2.symm
  have h4 := Complex.ofReal_eq_zero.mp h3
  have := Real.sqrt_nonneg 2
  linarith

theorem cT_ne_zero : (cT : ℂ) ≠ 0 := by
  intro h
  have h2 := cT_sq
  rw [h, zero_pow two_ne_zero] at h2
  have h3 : ((1 / 2 + Real.sqrt 3 / 6 : ℝ) : ℂ) = 0 := by
    push_cast
    exact h2.symm
  have h4 := Complex.ofReal_eq_zero.mp h3
  have := sqrt3_pos
  linarith

theorem hAmp1_ne_zero : hAmp1 ≠ 0 := by
  intro h
  have := congrFun h 0
  simp only [hAmp1, if_true, Pi.zero_apply] at this
  exact cH_ne_zero this

theorem tAmp1_ne_zero : tAmp1 ≠ 0 := by
  intro h
  have := congrFun h 0
  simp only [tAmp1, if_true, Pi.zero_apply] at this
  exact cT_ne_zero this

/-- `χ(|H⟩^⊗m) ≤ χ(|H⟩^⊗(m+1))` against `stabRankP 2`. -/
theorem qubit_h_stabRankP_le_succ (m : ℕ) :
    stabRankP 2 (hVec m) ≤ stabRankP 2 (hVec (m + 1)) := by
  rw [hVec_eq_powVecP, hVec_eq_powVecP]
  exact stabRankP_powVecP_le_succ hAmp1 hAmp1_ne_zero m

/-- `χ(|H⟩^⊗a) ≤ χ(|H⟩^⊗b)` for `a ≤ b`. -/
theorem qubit_h_stabRankP_mono {a b : ℕ} (h : a ≤ b) :
    stabRankP 2 (hVec a) ≤ stabRankP 2 (hVec b) := by
  rw [hVec_eq_powVecP, hVec_eq_powVecP]
  exact stabRankP_powVecP_mono hAmp1 hAmp1_ne_zero h

/-- `χ(|H⟩^⊗(a+b)) ≤ χ(|H⟩^⊗a) χ(|H⟩^⊗b)`. -/
theorem qubit_h_stabRankP_add_le (a b : ℕ) :
    stabRankP 2 (hVec (a + b)) ≤ stabRankP 2 (hVec a) * stabRankP 2 (hVec b) := by
  rw [hVec_eq_powVecP, hVec_eq_powVecP, hVec_eq_powVecP]
  exact stabRankP_powVecP_add_le hAmp1 a b

/-- `χ(|T⟩^⊗m) ≤ χ(|T⟩^⊗(m+1))` against `stabRankP 2`. -/
theorem qubit_t_stabRankP_le_succ (m : ℕ) :
    stabRankP 2 (tVec m) ≤ stabRankP 2 (tVec (m + 1)) := by
  rw [tVec_eq_powVecP, tVec_eq_powVecP]
  exact stabRankP_powVecP_le_succ tAmp1 tAmp1_ne_zero m

/-- `χ(|T⟩^⊗a) ≤ χ(|T⟩^⊗b)` for `a ≤ b`. -/
theorem qubit_t_stabRankP_mono {a b : ℕ} (h : a ≤ b) :
    stabRankP 2 (tVec a) ≤ stabRankP 2 (tVec b) := by
  rw [tVec_eq_powVecP, tVec_eq_powVecP]
  exact stabRankP_powVecP_mono tAmp1 tAmp1_ne_zero h

/-- `χ(|T⟩^⊗(a+b)) ≤ χ(|T⟩^⊗a) χ(|T⟩^⊗b)`. -/
theorem qubit_t_stabRankP_add_le (a b : ℕ) :
    stabRankP 2 (tVec (a + b)) ≤ stabRankP 2 (tVec a) * stabRankP 2 (tVec b) := by
  rw [tVec_eq_powVecP, tVec_eq_powVecP, tVec_eq_powVecP]
  exact stabRankP_powVecP_add_le tAmp1 a b

end StabRank
