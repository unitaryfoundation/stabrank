/-
The one-copy upper bounds as theorems about `Stabilizer.stabRank`.

`|S⟩ = (|1⟩ - |2⟩)/√2` is a combination of two computational-basis states,
and every one-qutrit vector is a combination of the three, so
`stabRank IsStab |S⟩ ≤ 2` and `stabRank IsStab |T3⟩ ≤ 3`. The second is the
instance `n = 1` of the general `stabRank_le_pow`, which reads the
computational basis (`isStab_single`) as a decomposition of size `3^n`.
Together with `T3M1StabRank.t3_m1_stabRank_gt_two` this settles
`stabRank IsStab |T3⟩ = 3`.
-/
import LeanProofs.T3M1StabRank

namespace StabRank

open Stabilizer

/-- The computational basis on `n` qutrits is a stabilizer decomposition of
    every vector, so `stabRank IsStab ψ ≤ 3^n`. -/
theorem stabRank_le_pow (n : ℕ) (ψ : QutritVec n) :
    Stabilizer.stabRank (IsStab (n := n)) ψ ≤ 3 ^ n := by
  classical
  refine le_trans (Stabilizer.stabRank_le_of_decomp
    (S := (Finset.univ : Finset (Fin (3 ^ n))).image
      (fun idx => (Pi.single idx (1 : ℂ) : QutritVec n))) ?_ ?_) ?_
  · intro σ hσ
    obtain ⟨idx, _, rfl⟩ := Finset.mem_image.mp hσ
    exact isStab_single n idx
  · have hψ : ψ = ∑ idx : Fin (3 ^ n), ψ idx • (Pi.single idx (1 : ℂ) : QutritVec n) := by
      funext j
      simp [Finset.sum_apply, Pi.single_apply]
    rw [hψ]
    refine Submodule.sum_mem _ (fun idx _ => Submodule.smul_mem _ _ (Submodule.subset_span ?_))
    simp only [Finset.coe_image, Finset.coe_univ, Set.image_univ, Set.mem_range]
    exact ⟨idx, rfl⟩
  · exact le_trans Finset.card_image_le (by simp)

/-! ### One qutrit -/

/-- The computational-basis state `|a⟩` on one qutrit. -/
noncomputable def basis1 (a : Fin 3) : QutritVec 1 := Pi.single (idx1.symm a) (1 : ℂ)

theorem basis1_isStab (a : Fin 3) : IsStab (basis1 a) := isStab_single 1 _

/-- `|S⟩ = (|1⟩ - |2⟩)/√2` as a one-qutrit vector. -/
noncomputable def strangeVec1 : QutritVec 1 := fun idx => strangeAmp1' (idx1 idx)

theorem strangeVec1_eq :
    strangeVec1 = (1 / (Real.sqrt 2 : ℂ)) • basis1 1 + (-(1 / (Real.sqrt 2 : ℂ))) • basis1 2 := by
  funext idx
  obtain ⟨a, rfl⟩ := idx1.symm.surjective idx
  simp only [strangeVec1, basis1, Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.single_apply,
    Equiv.apply_symm_apply, idx1.symm.injective.eq_iff]
  fin_cases a <;> simp [strangeAmp1']
  ring

/-- **χ(|S⟩) ≤ 2 against the concrete stabilizer predicate.** -/
theorem strange_m1_stabRank_le_two :
    Stabilizer.stabRank (IsStab (n := 1)) strangeVec1 ≤ 2 := by
  classical
  refine le_trans
    (Stabilizer.stabRank_le_of_decomp (S := {basis1 1, basis1 2}) ?_ ?_) Finset.card_le_two
  · intro σ hσ
    rcases Finset.mem_insert.mp hσ with rfl | hσ
    · exact basis1_isStab 1
    · rw [Finset.mem_singleton.mp hσ]
      exact basis1_isStab 2
  · rw [Finset.coe_pair]
    exact Submodule.mem_span_pair.mpr ⟨_, _, strangeVec1_eq.symm⟩

/-- **χ(|T3⟩) ≤ 3 against the concrete stabilizer predicate**: the
    computational basis. -/
theorem t3_m1_stabRank_le_three :
    Stabilizer.stabRank (IsStab (n := 1)) t3Vec1 ≤ 3 := by
  simpa using stabRank_le_pow 1 t3Vec1

/-- **χ(|T3⟩) = 3**, both directions machine-checked against `IsStab`. -/
theorem t3_m1_stabRank_eq_three :
    Stabilizer.stabRank (IsStab (n := 1)) t3Vec1 = 3 :=
  le_antisymm t3_m1_stabRank_le_three t3_m1_stabRank_gt_two

end StabRank
