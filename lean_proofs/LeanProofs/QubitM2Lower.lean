/-
χ(|H⟩^⊗2) ≥ 2 and χ(|T⟩^⊗2) ≥ 2 against `stabRankP 2`: neither two-qubit
state is a stabilizer state.

The argument is that of `verify_challenge/cert_rank1_moduli.py`, in a form
that needs no moduli: every nonzero entry of an `IsStabP 2` vector is `c i^e`,
so its fourth power is `c^4` at every point of the support
(`IsStabP.pow_period_eq`). The amplitudes of `|H⟩^⊗2` at `00` and `01` are
`cos²(π/8)` and `cos(π/8) sin(π/8)`; equal fourth powers would force
`√2 = -4/3`. For `|T⟩^⊗2` they are `cos² β` and `e^{iπ/4} cos β sin β`;
equal fourth powers would force `√3 = -2`. The cofactors of the
`linear_combination`s were found by polynomial division in the atoms
`√2, √3, i, cos, sin` against the relations of `QubitShared`. With the upper
bounds of `QubitHStabRank` and `QubitTStabRank` both cells are settled at `2`.
-/
import LeanProofs.QubitHStabRank
import LeanProofs.QubitTStabRank
import LeanProofs.Stabilizer.RankOne

namespace StabRank

open Stabilizer

/-! ### The trigonometric atoms do not vanish -/

theorem cH_ne_zero : (cH : ℂ) ≠ 0 := by
  intro h
  have hc := cH_sq
  have h2 := sqrt2_sq_c
  rw [h] at hc
  have hs : (Real.sqrt 2 : ℂ) = -2 := by linear_combination (-4) * hc
  rw [hs] at h2
  norm_num at h2

theorem sH_ne_zero : (sH : ℂ) ≠ 0 := by
  intro h
  have hs := sH_sq
  have h2 := sqrt2_sq_c
  rw [h] at hs
  have hs' : (Real.sqrt 2 : ℂ) = 2 := by linear_combination 4 * hs
  rw [hs'] at h2
  norm_num at h2

theorem cT_ne_zero : (cT : ℂ) ≠ 0 := by
  intro h
  have hc := cT_sq
  have h3 := sqrt3_sq_c
  rw [h] at hc
  have ht : (Real.sqrt 3 : ℂ) = -3 := by linear_combination (-6) * hc
  rw [ht] at h3
  norm_num at h3

theorem sT_ne_zero : (sT : ℂ) ≠ 0 := by
  intro h
  have hs := sT_sq
  have h3 := sqrt3_sq_c
  rw [h] at hs
  have ht : (Real.sqrt 3 : ℂ) = 3 := by linear_combination 6 * hs
  rw [ht] at h3
  norm_num at h3

/-! ### The probe indices `00` and `01` -/

def idx00 : Fin (2 ^ 2) := (digitsP 2 2).symm ![0, 0]
def idx01 : Fin (2 ^ 2) := (digitsP 2 2).symm ![0, 1]

/-! ### `|H⟩^⊗2` -/

set_option linter.flexible false in
theorem hVec2_idx00 : hVec 2 idx00 = (cH : ℂ) * cH := by
  simp (config := { decide := true }) [hVec, hAmp1, idx00, Fin.prod_univ_two]

set_option linter.flexible false in
theorem hVec2_idx01 : hVec 2 idx01 = (cH : ℂ) * sH := by
  simp (config := { decide := true }) [hVec, hAmp1, idx01, Fin.prod_univ_two]

/-- `|H⟩^⊗2` is not a stabilizer state: `cos⁸(π/8) = cos⁴(π/8) sin⁴(π/8)`
    would force `√2 = -4/3`. -/
theorem hVec2_not_isStabP : ¬ IsStabP 2 (hVec 2) := by
  intro hstab
  have hx : hVec 2 idx00 ≠ 0 := by
    rw [hVec2_idx00]
    exact mul_ne_zero cH_ne_zero cH_ne_zero
  have hy : hVec 2 idx01 ≠ 0 := by
    rw [hVec2_idx01]
    exact mul_ne_zero cH_ne_zero sH_ne_zero
  have h := hstab.pow_period_eq hx hy
  rw [hVec2_idx00, hVec2_idx01, stabPeriod_two] at h
  have hc := cH_sq
  have hs := sH_sq
  have h2 := sqrt2_sq_c
  have h' : ((cH : ℂ) ^ 2) ^ 4 = ((cH : ℂ) ^ 2) ^ 2 * ((sH : ℂ) ^ 2) ^ 2 := by
    linear_combination h
  rw [hc, hs] at h'
  have hs2 : (Real.sqrt 2 : ℂ) = -4 / 3 := by
    linear_combination (16 / 3) * h' - ((Real.sqrt 2 : ℂ) + 4) / 6 * h2
  rw [hs2] at h2
  norm_num at h2

theorem hVec2_ne_zero : hVec 2 ≠ 0 := by
  intro h
  have h0 := hVec2_idx00
  rw [h, Pi.zero_apply] at h0
  exact mul_ne_zero cH_ne_zero cH_ne_zero h0.symm

/-- **χ(|H⟩^⊗2) ≥ 2 against `IsStabP 2`.** -/
theorem qubit_h_m2_stabRankP_gt_one : stabRankP 2 (hVec 2) > 1 :=
  stabRank_gt_one_of_not_stab _ _ (decompCardsP_nonempty 2 2 _)
    (fun _ _ hσ ha => hσ.smul ha) hVec2_ne_zero hVec2_not_isStabP

/-- **χ(|H⟩^⊗2) = 2**, both directions machine-checked against `IsStabP 2`. -/
theorem qubit_h_m2_stabRankP_eq_two : stabRankP 2 (hVec 2) = 2 :=
  le_antisymm qubit_h_m2_stabRankP_le_two qubit_h_m2_stabRankP_gt_one

/-! ### `|T⟩^⊗2` -/

set_option linter.flexible false in
theorem tVec2_idx00 : tVec 2 idx00 = (cT : ℂ) * cT := by
  simp (config := { decide := true }) [tVec, tAmp1, idx00, Fin.prod_univ_two]

set_option linter.flexible false in
theorem tVec2_idx01 :
    tVec 2 idx01 = (cT : ℂ) * (Complex.exp (Real.pi / 4 * Complex.I) * sT) := by
  simp (config := { decide := true }) [tVec, tAmp1, idx01, Fin.prod_univ_two]

/-- `|T⟩^⊗2` is not a stabilizer state: `cos⁸ β = e^{iπ} cos⁴ β sin⁴ β` would
    force `√3 = -2`. -/
theorem tVec2_not_isStabP : ¬ IsStabP 2 (tVec 2) := by
  intro hstab
  have he : Complex.exp (Real.pi / 4 * Complex.I) ≠ 0 := Complex.exp_ne_zero _
  have hx : tVec 2 idx00 ≠ 0 := by
    rw [tVec2_idx00]
    exact mul_ne_zero cT_ne_zero cT_ne_zero
  have hy : tVec 2 idx01 ≠ 0 := by
    rw [tVec2_idx01]
    exact mul_ne_zero cT_ne_zero (mul_ne_zero he sT_ne_zero)
  have h := hstab.pow_period_eq hx hy
  rw [tVec2_idx00, tVec2_idx01, stabPeriod_two, exp_pi_div_four_mul_I] at h
  have hc := cT_sq
  have hs := sT_sq
  have h2 := sqrt2_sq_c
  have h3 := sqrt3_sq_c
  have hI := Complex.I_sq
  have h' : ((cT : ℂ) ^ 2) ^ 4 = -(((cT : ℂ) ^ 2) ^ 2 * ((sT : ℂ) ^ 2) ^ 2) := by
    linear_combination h
      + ((cT : ℂ) ^ 4 * (sT : ℂ) ^ 4 * (Complex.I + 1) ^ 4
          * ((Real.sqrt 2 : ℂ) ^ 2 + 2) / 16) * h2
      + ((cT : ℂ) ^ 4 * (sT : ℂ) ^ 4 * (Complex.I ^ 2 + 4 * Complex.I + 5) / 4) * hI
  rw [hc, hs] at h'
  have ht : (Real.sqrt 3 : ℂ) = -2 := by
    linear_combination 9 * h'
      - (((Real.sqrt 3 : ℂ) ^ 2 + 6 * (Real.sqrt 3 : ℂ) + 21) / 72) * h3
  rw [ht] at h3
  norm_num at h3

theorem tVec2_ne_zero : tVec 2 ≠ 0 := by
  intro h
  have h0 := tVec2_idx00
  rw [h, Pi.zero_apply] at h0
  exact mul_ne_zero cT_ne_zero cT_ne_zero h0.symm

/-- **χ(|T⟩^⊗2) ≥ 2 against `IsStabP 2`.** -/
theorem qubit_t_m2_stabRankP_gt_one : stabRankP 2 (tVec 2) > 1 :=
  stabRank_gt_one_of_not_stab _ _ (decompCardsP_nonempty 2 2 _)
    (fun _ _ hσ ha => hσ.smul ha) tVec2_ne_zero tVec2_not_isStabP

/-- **χ(|T⟩^⊗2) = 2**, both directions machine-checked against `IsStabP 2`. -/
theorem qubit_t_m2_stabRankP_eq_two : stabRankP 2 (tVec 2) = 2 :=
  le_antisymm qubit_t_m2_stabRankP_le_two qubit_t_m2_stabRankP_gt_one

end StabRank
