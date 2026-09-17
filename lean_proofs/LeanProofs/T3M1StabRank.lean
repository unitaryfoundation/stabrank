/-
χ(|T3⟩) ≥ 3 as a theorem about `Stabilizer.stabRank`.

`T3GaloisDescent.t3_three_le` says that any family of vectors over `ℤ[ω₃]`
whose `ℂ`-span contains the T3 amplitude vector has at least three members.
This file connects it to the concrete stabilizer predicate of
`Stabilizer.IsStab`: every `stabVecN` has entries in `ℤ[ω₃]`, so every
`IsStab` vector is a nonzero multiple of one, and a set of at most two
`IsStab` vectors cannot span `|T3⟩`. With the computational basis providing
some decomposition, the reduction lemma of `Stabilizer.Rank` gives
`stabRank IsStab |T3⟩ > 2`.

This is the Galois argument of `verify_challenge/cert_t3_galois.py` at
`m = 1`, machine-checked end to end: the only inputs are the definitions of
the stabilizer predicate and of `|T3⟩`.
-/
import LeanProofs.Stabilizer.IsStab
import LeanProofs.T3GaloisDescent

namespace StabRank

open Stabilizer

/-- Every entry of a `stabVecN` is an integer combination of `1` and `ω₃`. -/
theorem stabVecN_mem_Zomega3 (n k : ℕ) (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3) :
    OverZomega3 (stabVecN n k x0 W Q l) := by
  classical
  intro x
  unfold stabVecN
  induction (Finset.univ : Finset (Fin k → Fin 3)) using Finset.induction with
  | empty => exact ⟨0, 0, by simp⟩
  | insert y s hy ih =>
    obtain ⟨a, b, hab⟩ := ih
    by_cases h : x = affinePt x0 W y
    · obtain ⟨c, d, hcd⟩ := omega3_pow_mem (quadPhase Q l y)
      refine ⟨a + c, b + d, ?_⟩
      rw [Finset.sum_insert hy, if_pos h, hab, hcd]
      push_cast
      ring
    · refine ⟨a, b, ?_⟩
      rw [Finset.sum_insert hy, if_neg h, hab]
      ring

/-- An `IsStab` vector is a scalar multiple of a vector over `ℤ[ω₃]`. -/
theorem IsStab.exists_overZomega3 {n : ℕ} {v : QutritVec n} (hv : IsStab v) :
    ∃ u : QutritVec n, OverZomega3 u ∧ ∃ c : ℂ, v = c • u := by
  obtain ⟨c, k, x0, W, Q, l, _, _, rfl⟩ := hv
  refine ⟨fun idx => stabVecN n k x0 W Q l (digits n idx), fun idx => ?_, c, rfl⟩
  exact stabVecN_mem_Zomega3 n k x0 W Q l (digits n idx)

/-- `t3_three_le` for an arbitrary finite index type. -/
theorem t3_three_le_card {κ : Type*} [Fintype κ] (s : κ → (Fin 3 → ℂ))
    (hs : ∀ j, OverZomega3 (s j))
    (h0 : tConj 0 ∈ Submodule.span ℂ (Set.range s)) : 3 ≤ Fintype.card κ := by
  have hr : Set.range (s ∘ (Fintype.equivFin κ).symm) = Set.range s :=
    (Fintype.equivFin κ).symm.surjective.range_comp s
  refine t3_three_le (s ∘ (Fintype.equivFin κ).symm) (fun j => hs _) ?_
  rw [hr]
  exact h0

/-- The single-qutrit index, `Fin (3 ^ 1) ≃ Fin 3`. -/
def idx1 : Fin (3 ^ 1) ≃ Fin 3 := (digits 1).trans (Equiv.funUnique (Fin 1) (Fin 3))

/-- The T3 amplitude vector `(1, ω₉, ω₉²)` as a one-qutrit vector. -/
noncomputable def t3Vec1 : QutritVec 1 := fun idx => tConj 0 (idx1 idx)

/-- Reindexing along `idx1`, as a linear equivalence. -/
noncomputable def reindex1 : QutritVec 1 ≃ₗ[ℂ] (Fin 3 → ℂ) :=
  LinearEquiv.funCongrLeft ℂ ℂ idx1.symm

theorem reindex1_apply (v : QutritVec 1) (i : Fin 3) : reindex1 v i = v (idx1.symm i) := rfl

theorem reindex1_t3Vec1 : reindex1 t3Vec1 = tConj 0 := by
  funext i
  simp [reindex1_apply, t3Vec1]

theorem reindex1_overZomega3 {u : QutritVec 1} (hu : OverZomega3 u) :
    OverZomega3 (reindex1 u) := fun i => hu (idx1.symm i)

/-- **χ(|T3⟩) ≥ 3 against the concrete stabilizer predicate**: no two
    stabilizer states span the T3 state. -/
theorem t3_m1_stabRank_gt_two :
    Stabilizer.stabRank (IsStab (n := 1)) t3Vec1 > 2 := by
  classical
  refine Stabilizer.stabRank_gt_of_no_decomp_le _ _ 2 (decompCards_nonempty 1 _) ?_
  intro S hcard hstab hmem
  choose u hu c hc using fun σ : S => IsStab.exists_overZomega3 (hstab σ σ.2)
  let s : S → (Fin 3 → ℂ) := fun σ => reindex1 (u σ)
  have hs : ∀ σ, OverZomega3 (s σ) := fun σ => reindex1_overZomega3 (hu σ)
  -- the span of S is carried into the span of the s σ by reindexing
  have hle : Submodule.span ℂ (S : Set (QutritVec 1)) ≤
      (Submodule.span ℂ (Set.range s)).comap reindex1.toLinearMap := by
    rw [Submodule.span_le]
    intro σ hσ
    have hσ' : σ = c ⟨σ, hσ⟩ • u ⟨σ, hσ⟩ := hc ⟨σ, hσ⟩
    change reindex1 σ ∈ Submodule.span ℂ (Set.range s)
    rw [hσ', map_smul]
    exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨⟨σ, hσ⟩, rfl⟩)
  have h0 : tConj 0 ∈ Submodule.span ℂ (Set.range s) := by
    rw [← reindex1_t3Vec1]
    exact hle hmem
  have h3 := t3_three_le_card s hs h0
  rw [Fintype.card_coe] at h3
  omega

end StabRank
