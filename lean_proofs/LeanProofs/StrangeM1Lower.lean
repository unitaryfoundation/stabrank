/-
χ(|S⟩) ≥ 2 against `stabRank`: the one-qutrit Strange state is not a
stabilizer state.

`|S⟩ = (|1⟩ - |2⟩)/√2` is supported on `{1, 2}`, and the support of a
stabilizer state is an affine flat, closed under `a - b + d`. With
`a = d = 1` and `b = 2` that puts `0` in the support, where `|S⟩` vanishes
(`IsStab.affine_support`, exactly as in the two-copy proof of
`Stabilizer/IsStab.lean`). This is the argument of
`verify_challenge/cert_rank1_moduli.py` for `S` at `m = 1`: a support of size
two is not the size of a flat of `F_3`. With
`M1StabRank.strange_m1_stabRank_le_two` the cell is settled at
`stabRank IsStab |S⟩ = 2`.
-/
import LeanProofs.M1StabRank
import LeanProofs.Stabilizer.RankOne

namespace StabRank

open Stabilizer

/-- `|S⟩` on the digit string `f`, as the one-qutrit amplitude at `f 0`. -/
theorem strangeVec1_digits (f : Fin 1 → Fin 3) :
    strangeVec1 ((digits 1).symm f) = strangeAmp1' (f 0) := by
  simp [strangeVec1, idx1]

theorem strangeAmp1'_one_ne_zero : strangeAmp1' 1 ≠ 0 := by
  simp [strangeAmp1']

theorem strangeAmp1'_two_ne_zero : strangeAmp1' 2 ≠ 0 := by
  simp [strangeAmp1']

/-- `|S⟩` is not a stabilizer state: its support `{1, 2}` is not closed under
    `a - b + d`. -/
theorem strangeVec1_not_isStab : ¬ IsStab strangeVec1 := by
  intro hσ
  have h1 : strangeVec1 ((digits 1).symm fun _ => 1) ≠ 0 := by
    rw [strangeVec1_digits]
    exact strangeAmp1'_one_ne_zero
  have h2 : strangeVec1 ((digits 1).symm fun _ => 2) ≠ 0 := by
    rw [strangeVec1_digits]
    exact strangeAmp1'_two_ne_zero
  have hkey := hσ.affine_support _ _ _ h1 h2 h1
  rw [strangeVec1_digits] at hkey
  simp only [Pi.add_apply, Pi.sub_apply] at hkey
  have h0 : (1 : Fin 3) - 2 + 1 = 0 := by decide
  rw [h0] at hkey
  exact hkey (by simp [strangeAmp1'])

theorem strangeVec1_ne_zero : strangeVec1 ≠ 0 := by
  intro h
  have h1 := strangeVec1_digits (fun _ => 1)
  rw [h] at h1
  exact strangeAmp1'_one_ne_zero (by simpa using h1.symm)

/-- **χ(|S⟩) ≥ 2 against the concrete stabilizer predicate.** -/
theorem strange_m1_stabRank_gt_one :
    Stabilizer.stabRank (IsStab (n := 1)) strangeVec1 > 1 :=
  stabRank_gt_one_of_not_stab _ _ (decompCards_nonempty 1 _)
    (fun _ _ hσ ha => hσ.smul ha) strangeVec1_ne_zero strangeVec1_not_isStab

/-- **χ(|S⟩) = 2**, both directions machine-checked against `IsStab`. -/
theorem strange_m1_stabRank_eq_two :
    Stabilizer.stabRank (IsStab (n := 1)) strangeVec1 = 2 :=
  le_antisymm strange_m1_stabRank_le_two strange_m1_stabRank_gt_one

end StabRank
