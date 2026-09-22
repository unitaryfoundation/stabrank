/-
The covering lemma behind the symmetry reduction of the pivot sieves.

`verify_challenge/rank_exclusion.py` scans one pivot per orbit of a group of
symmetries of the target: a symmetry `g` carries a decomposition
`ψ = Σ cⱼ sⱼ` to `ψ = Σ c'ⱼ g sⱼ`, so if some decomposition contains `s`
and `g s` is a scalar times the orbit representative `s'`, a decomposition
containing `s'` exists and is found when `s'` is the pivot. This file is the
pure lemma, for an abstract predicate `P` closed under nonzero rescaling and
a linear automorphism `g` that preserves `P` and fixes the target up to a
nonzero scalar: `exists_decomp_mem_of_symm`. The size bound is kept, so the
lemma composes with `stabRank_gt_of_no_decomp_le` of `Rank.lean`.

What a cell has to supply, per generator `g` of its group, is the hypothesis
`hg : ∀ σ, P σ → P (g σ)` together with the permutation of the dictionary.
For the monomial subgroup (copy permutations, the local `X` and `Z`-type
Cliffords, and any diagonal Clifford with phases in `ζ_D`), `g` permutes
digit strings and multiplies by a phase, so `g (stabVecP x0 W Q l)` is
again a `stabVecP` with `x0`, `W` relabelled and the phase reparametrised
through `PhaseP.lean`; that is the same shape of argument as `SliceP.lean`
and gives `hg` once and for all, with the dictionary permutation then a
finite check on exponent tables. The non-monomial generators (the qutrit
Fourier transform, the qubit Hadamard) need a Gauss sum per dictionary
element and are not covered by a generic `hg`; with dictionary completeness
(`QutritDict2.lean` at two qutrits) they can instead be verified element by
element, `g tⱼ = cⱼ • t_{π j}` for every listed table, which is a finite
exact computation in `ℤ[ζ]`. The antiunitary symmetries (a Clifford
composed with complex conjugation) are not linear and need the conjugate
form of this lemma, stated the same way with `g` semilinear over
`starRingEnd ℂ`; it is not here.
-/
import Mathlib.LinearAlgebra.Span.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Complex.Basic
import Mathlib.Tactic

namespace StabRank

variable {ι : Type*}

/-- **The covering lemma.** A linear automorphism `g` that preserves the
    predicate `P` and fixes `ψ` up to a nonzero scalar carries a decomposition
    of `ψ` of size at most `r` containing `s` to one of size at most `r`
    containing `s'`, whenever `g s = a • s'` with `a ≠ 0`. -/
theorem exists_decomp_mem_of_symm (P : (ι → ℂ) → Prop)
    (hP : ∀ σ (a : ℂ), P σ → a ≠ 0 → P (a • σ))
    (g : (ι → ℂ) ≃ₗ[ℂ] (ι → ℂ)) (hg : ∀ σ, P σ → P (g σ))
    (ψ : ι → ℂ) {c : ℂ} (hc : c ≠ 0) (hψ : g ψ = c • ψ)
    {r : ℕ} {S : Finset (ι → ℂ)} (hS : S.card ≤ r) (hSP : ∀ σ ∈ S, P σ)
    (hspan : ψ ∈ Submodule.span ℂ (S : Set (ι → ℂ)))
    {s : ι → ℂ} (hs : s ∈ S) {s' : ι → ℂ} {a : ℂ} (ha : a ≠ 0)
    (hgs : g s = a • s') :
    ∃ S' : Finset (ι → ℂ), S'.card ≤ r ∧ (∀ σ ∈ S', P σ) ∧
      ψ ∈ Submodule.span ℂ (S' : Set (ι → ℂ)) ∧ s' ∈ S' := by
  classical
  refine ⟨insert s' ((S.erase s).image g), ?_, ?_, ?_, Finset.mem_insert_self _ _⟩
  · calc (insert s' ((S.erase s).image g)).card
        ≤ ((S.erase s).image g).card + 1 := Finset.card_insert_le _ _
      _ ≤ (S.erase s).card + 1 := by
          have := Finset.card_image_le (s := S.erase s) (f := g)
          omega
      _ = S.card := Finset.card_erase_add_one hs
      _ ≤ r := hS
  · intro σ hσ
    rw [Finset.mem_insert, Finset.mem_image] at hσ
    rcases hσ with hσ | ⟨τ, hτ, rfl⟩
    · have h1 : s' = a⁻¹ • g s := by
        rw [hgs, smul_smul, inv_mul_cancel₀ ha, one_smul]
      rw [hσ, h1]
      exact hP _ _ (hg s (hSP s hs)) (inv_ne_zero ha)
    · exact hg τ (hSP τ (Finset.mem_of_mem_erase hτ))
  · have h1 : g ψ ∈ Submodule.span ℂ
        ((g : (ι → ℂ) →ₗ[ℂ] (ι → ℂ)) '' (S : Set (ι → ℂ))) :=
      Submodule.apply_mem_span_image_of_mem_span _ hspan
    have h2 : (g : (ι → ℂ) →ₗ[ℂ] (ι → ℂ)) '' (S : Set (ι → ℂ))
        ⊆ (Submodule.span ℂ ((insert s' ((S.erase s).image g) : Finset _) : Set (ι → ℂ)) :
          Set (ι → ℂ)) := by
      rintro τ ⟨σ, hσ, rfl⟩
      by_cases hσs : σ = s
      · subst hσs
        rw [SetLike.mem_coe, LinearEquiv.coe_coe, hgs]
        exact Submodule.smul_mem _ _ (Submodule.subset_span (by simp))
      · apply Submodule.subset_span
        rw [Finset.coe_insert, Set.mem_insert_iff, Finset.coe_image]
        exact Or.inr ⟨σ, by rw [Finset.mem_coe, Finset.mem_erase]; exact ⟨hσs, hσ⟩, rfl⟩
    have h3 := Submodule.span_mono h2 h1
    rw [Submodule.span_eq, hψ] at h3
    exact (Submodule.smul_mem_iff _ hc).mp h3

end StabRank
