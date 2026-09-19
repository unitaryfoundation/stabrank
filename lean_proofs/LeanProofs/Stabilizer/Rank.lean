/-
Stabilizer rank: Phase 1 scaffolding.

We define the n-qutrit ambient vector space `QutritVec n := Fin (3^n) → ℂ`
and the stabilizer rank `stabRank` relative to an abstract predicate
`IsStab : (ι → ℂ) → Prop` on vectors indexed by any type `ι`. The
predicate is left abstract here; the concrete qutrit definition is
`StabRank.IsStab` in `Stabilizer/IsStab.lean` and the qudit-generic one
is `StabRank.IsStabP` in `Stabilizer/IsStabP.lean`. Everything below is
stated for `ι → ℂ`, so it applies verbatim to `QutritVec n = Fin (3 ^ n) → ℂ`
and to `Fin (p ^ n) → ℂ`.

The key result of this file is the **reduction lemma**:

  stabRank IsStab ψ > k  ←  ∀ S : Finset (QutritVec n), S.card ≤ k →
                             (∀ σ ∈ S, IsStab σ) → ψ ∉ Submodule.span ℂ S
  (given DecompCards nonempty)

This is the structural piece that converts an exhaustive per-triple
non-decomposition check (the right-hand side at k = 3) into a stabilizer
rank lower bound stabRank ψ ≥ 4 (the left-hand side at k = 3). The
nonempty-decompositions hypothesis is discharged in a later phase by
exhibiting a stabilizer basis (e.g., the computational basis on n
qutrits, which consists entirely of stabilizer states).
-/
import Mathlib.LinearAlgebra.Span.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Lattice
import Mathlib.Tactic

namespace StabRank
namespace Stabilizer

/-- The ambient amplitude space for `n` qutrits. -/
abbrev QutritVec (n : ℕ) : Type := Fin (3 ^ n) → ℂ

variable {ι : Type*}

/-- The set of natural numbers `k` for which `ψ` admits a `k`-element
    decomposition into states satisfying `IsStab`. -/
def DecompCards (IsStab : (ι → ℂ) → Prop) (ψ : ι → ℂ) : Set ℕ :=
  { k | ∃ S : Finset (ι → ℂ),
          S.card = k ∧ (∀ σ ∈ S, IsStab σ) ∧
          ψ ∈ Submodule.span ℂ (↑S : Set (ι → ℂ)) }

/-- **Stabilizer rank**: the smallest `k` for which `ψ` is in the span
    of `k` distinct states satisfying `IsStab`. Returns `0` when the
    set of decomposition cardinalities is empty (i.e. no finite
    stabilizer decomposition exists at all); in our setting this
    fallback is later eliminated by exhibiting a stabilizer basis. -/
noncomputable def stabRank (IsStab : (ι → ℂ) → Prop) (ψ : ι → ℂ) : ℕ :=
  sInf (DecompCards IsStab ψ)

/-- A witnessing Finset shows membership in `DecompCards`. -/
lemma DecompCards.of_witness {IsStab : (ι → ℂ) → Prop} {ψ : ι → ℂ}
    {S : Finset (ι → ℂ)} (hstab : ∀ σ ∈ S, IsStab σ)
    (hspan : ψ ∈ Submodule.span ℂ (↑S : Set (ι → ℂ))) :
    S.card ∈ DecompCards IsStab ψ :=
  ⟨S, rfl, hstab, hspan⟩

/-- Decomposition existence ⇒ stabilizer rank bounded above. -/
lemma stabRank_le_of_decomp {IsStab : (ι → ℂ) → Prop} {ψ : ι → ℂ}
    {S : Finset (ι → ℂ)} (hstab : ∀ σ ∈ S, IsStab σ)
    (hspan : ψ ∈ Submodule.span ℂ (↑S : Set (ι → ℂ))) :
    stabRank IsStab ψ ≤ S.card :=
  Nat.sInf_le (DecompCards.of_witness hstab hspan)

/-- **Reduction lemma (forward direction)**: ruling out every
    `k`-element stabilizer decomposition forces the stabilizer rank to
    exceed `k`, provided some decomposition exists.

    Concretely: for `k = 3`, the right-hand side is exactly the property
    that the exhaustive triple-search certificate verifies, and the
    conclusion is the rank claim `stabRank ψ ≥ 4`.

    The `hNonempty` hypothesis says `ψ` admits *some* stabilizer
    decomposition; in the qutrit setting it is discharged by the
    computational basis (a stabilizer basis on `n` qutrits). -/
theorem stabRank_gt_of_no_decomp_le
    (IsStab : (ι → ℂ) → Prop) (ψ : ι → ℂ) (k : ℕ)
    (hNonempty : (DecompCards IsStab ψ).Nonempty)
    (h : ∀ S : Finset (ι → ℂ), S.card ≤ k →
          (∀ σ ∈ S, IsStab σ) →
          ψ ∉ Submodule.span ℂ (↑S : Set (ι → ℂ))) :
    stabRank IsStab ψ > k := by
  -- Nat.sInf is attained at some witness when the set is nonempty.
  have hmem : sInf (DecompCards IsStab ψ) ∈ DecompCards IsStab ψ :=
    Nat.sInf_mem hNonempty
  obtain ⟨S, hcard, hstab, hspan⟩ := hmem
  by_contra hcontra
  have hle : S.card ≤ k := hcard ▸ not_lt.mp hcontra
  exact h S hle hstab hspan

/-- **Reduction lemma (backward direction)**: if the stabilizer rank
    exceeds `k`, then no decomposition with `≤ k` stabilizer states
    spans `ψ`. -/
theorem no_decomp_le_of_stabRank_gt
    (IsStab : (ι → ℂ) → Prop) (ψ : ι → ℂ) (k : ℕ)
    (hrank : stabRank IsStab ψ > k) :
    ∀ S : Finset (ι → ℂ), S.card ≤ k →
      (∀ σ ∈ S, IsStab σ) →
      ψ ∉ Submodule.span ℂ (↑S : Set (ι → ℂ)) := by
  intro S hcard hstab hspan
  have hmem : S.card ∈ DecompCards IsStab ψ :=
    DecompCards.of_witness hstab hspan
  have : stabRank IsStab ψ ≤ S.card := Nat.sInf_le hmem
  omega

end Stabilizer
end StabRank
