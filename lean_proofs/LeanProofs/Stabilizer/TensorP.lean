/-
Tensor products against the qudit-generic predicate `IsStabP p`, and the
multiplicativity of the upper bound: `stabRankP p (ψ ⊗ φ) ≤ stabRankP p ψ *
stabRankP p φ`.

This is `TensorStabRank.lean` with `Fin 3` replaced by `ZMod p` and
`stabVecN` by `stabVecP`; the proofs are the same. `tensorP ψ φ` is the
product state on `n + m` qudits, read off a digit string by splitting it into
its first `n` and last `m` digits. The tensor product of two `stabVecP` is
again a `stabVecP`: the flats concatenate (`Fin.append` on `x₀`,
block-diagonal `W`), the exponents add (block-diagonal `Q`, `Fin.append` on
`l`), and the sum over `(ZMod p)^(k+k')` factors as a product of the two sums.
A decomposition of `ψ` into `S` and of `φ` into `T` gives one of `ψ ⊗ φ` into
the pairwise products, of which there are at most `|S| · |T|`.
-/
import LeanProofs.Stabilizer.IsStabP
import Mathlib.Logic.Equiv.Fin.Basic

namespace StabRank

open Stabilizer

variable {p : ℕ}

/-! ### The tensor product on digit-indexed vectors -/

/-- The first `n` digits of a string on `n + m` qudits. -/
def splitLP {n m : ℕ} (x : Fin (n + m) → ZMod p) : Fin n → ZMod p :=
  fun i => x (Fin.castAdd m i)

/-- The last `m` digits of a string on `n + m` qudits. -/
def splitRP {n m : ℕ} (x : Fin (n + m) → ZMod p) : Fin m → ZMod p :=
  fun j => x (Fin.natAdd n j)

theorem append_splitLP_splitRP {n m : ℕ} (x : Fin (n + m) → ZMod p) :
    Fin.append (splitLP x) (splitRP x) = x :=
  Fin.append_castAdd_natAdd

/-- `ψ ⊗ φ`: the amplitude at a digit string is the product of the amplitudes
    at its two halves. -/
noncomputable def tensorP (p : ℕ) [NeZero p] {n m : ℕ} (ψ : Fin (p ^ n) → ℂ)
    (φ : Fin (p ^ m) → ℂ) : Fin (p ^ (n + m)) → ℂ :=
  fun idx =>
    ψ ((digitsP p n).symm (splitLP (digitsP p (n + m) idx)))
      * φ ((digitsP p m).symm (splitRP (digitsP p (n + m) idx)))

theorem appendP_eq_append_iff {n m : ℕ} {a b : Fin n → ZMod p} {c d : Fin m → ZMod p} :
    Fin.append a c = Fin.append b d ↔ a = b ∧ c = d := by
  constructor
  · intro h
    have h' : ((a, c) : (Fin n → ZMod p) × (Fin m → ZMod p)) = (b, d) :=
      (Fin.appendEquiv n m).injective h
    exact Prod.mk.inj h'
  · rintro ⟨rfl, rfl⟩
    rfl

/-! ### Block data for the tensor product of two `stabVecP` -/

/-- Block-diagonal `W`. -/
def blockWP {n m k k' : ℕ} (W : Fin k → Fin n → ZMod p) (W' : Fin k' → Fin m → ZMod p) :
    Fin (k + k') → Fin (n + m) → ZMod p :=
  Fin.append (fun j => Fin.append (W j) 0) (fun j => Fin.append 0 (W' j))

/-- Block-diagonal `Q`. -/
def blockQP {k k' : ℕ} (Q : Fin k → Fin k → ZMod p) (Q' : Fin k' → Fin k' → ZMod p) :
    Fin (k + k') → Fin (k + k') → ZMod p :=
  Fin.append (fun i => Fin.append (Q i) 0) (fun i => Fin.append 0 (Q' i))

theorem affinePtP_append {n m k k' : ℕ} (x0 : Fin n → ZMod p) (W : Fin k → Fin n → ZMod p)
    (x0' : Fin m → ZMod p) (W' : Fin k' → Fin m → ZMod p)
    (y : Fin k → ZMod p) (y' : Fin k' → ZMod p) :
    affinePtP (Fin.append x0 x0') (blockWP W W') (Fin.append y y')
      = Fin.append (affinePtP x0 W y) (affinePtP x0' W' y') := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · simp [affinePtP, blockWP, Fin.sum_univ_add]
  · simp [affinePtP, blockWP, Fin.sum_univ_add]

theorem quadPhaseP_append {k k' : ℕ} (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (Q' : Fin k' → Fin k' → ZMod p)
    (l' : Fin k' → ZMod (stabPeriod p)) (y : Fin k → ZMod p) (y' : Fin k' → ZMod p) :
    quadPhaseP (blockQP Q Q') (Fin.append l l') (Fin.append y y')
      = quadPhaseP Q l y + quadPhaseP Q' l' y' := by
  simp only [quadPhaseP, blockQP, Fin.sum_univ_add, Fin.append_left, Fin.append_right,
    Pi.zero_apply, ZMod.val_zero, zero_mul, Finset.sum_const_zero, add_zero, zero_add]
  ring

theorem affinePtP_append_injective {n m k k' : ℕ} {x0 : Fin n → ZMod p}
    {W : Fin k → Fin n → ZMod p} {x0' : Fin m → ZMod p} {W' : Fin k' → Fin m → ZMod p}
    (h : Function.Injective (affinePtP x0 W)) (h' : Function.Injective (affinePtP x0' W')) :
    Function.Injective (affinePtP (Fin.append x0 x0') (blockWP W W')) := by
  intro z z' hz
  rw [← append_splitLP_splitRP z, ← append_splitLP_splitRP z', affinePtP_append,
    affinePtP_append, appendP_eq_append_iff] at hz
  rw [← append_splitLP_splitRP z, ← append_splitLP_splitRP z', h hz.1, h' hz.2]

/-- The sum over `(ZMod p)^(k+k')` factors: a tensor product of two `stabVecP`
    is the `stabVecP` of the block data. -/
theorem stabVecP_append [NeZero p] {n m k k' : ℕ} (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p) (l : Fin k → ZMod (stabPeriod p))
    (x0' : Fin m → ZMod p) (W' : Fin k' → Fin m → ZMod p) (Q' : Fin k' → Fin k' → ZMod p)
    (l' : Fin k' → ZMod (stabPeriod p)) (x : Fin n → ZMod p) (x' : Fin m → ZMod p) :
    stabVecP p (n + m) (k + k') (Fin.append x0 x0') (blockWP W W') (blockQP Q Q')
        (Fin.append l l') (Fin.append x x')
      = stabVecP p n k x0 W Q l x * stabVecP p m k' x0' W' Q' l' x' := by
  unfold stabVecP
  rw [Fintype.sum_mul_sum, ← Equiv.sum_comp (Fin.appendEquiv k k'), Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun y' _ => ?_
  rw [show (Fin.appendEquiv k k') (y, y') = Fin.append y y' from rfl]
  simp only [affinePtP_append, quadPhaseP_append, appendP_eq_append_iff, pow_add,
    ite_zero_mul_ite_zero]

/-- **The tensor product of two stabilizer states is a stabilizer state.** -/
theorem IsStabP.tensor [Fact p.Prime] {n m : ℕ} {ψ : Fin (p ^ n) → ℂ} {φ : Fin (p ^ m) → ℂ}
    (hψ : IsStabP p ψ) (hφ : IsStabP p φ) : IsStabP p (tensorP p ψ φ) := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hψ
  obtain ⟨c', k', x0', W', Q', l', hc', hinj', rfl⟩ := hφ
  refine ⟨c * c', k + k', Fin.append x0 x0', blockWP W W', blockQP Q Q', Fin.append l l',
    mul_ne_zero hc hc', affinePtP_append_injective hinj hinj', ?_⟩
  funext idx
  simp only [tensorP, Equiv.apply_symm_apply]
  conv_rhs => rw [← append_splitLP_splitRP (digitsP p (n + m) idx)]
  rw [stabVecP_append]
  ring

/-! ### Bilinearity, and spans of tensor products -/

theorem tensorP_sum_left [NeZero p] {n m : ℕ} {ι : Type*} (s : Finset ι)
    (f : ι → (Fin (p ^ n) → ℂ)) (φ : Fin (p ^ m) → ℂ) :
    tensorP p (∑ i ∈ s, f i) φ = ∑ i ∈ s, tensorP p (f i) φ := by
  funext idx
  simp [tensorP, Finset.sum_apply, Finset.sum_mul]

theorem tensorP_sum_right [NeZero p] {n m : ℕ} {ι : Type*} (ψ : Fin (p ^ n) → ℂ)
    (s : Finset ι) (g : ι → (Fin (p ^ m) → ℂ)) :
    tensorP p ψ (∑ i ∈ s, g i) = ∑ i ∈ s, tensorP p ψ (g i) := by
  funext idx
  simp [tensorP, Finset.sum_apply, Finset.mul_sum]

theorem tensorP_smul_left [NeZero p] {n m : ℕ} (a : ℂ) (ψ : Fin (p ^ n) → ℂ)
    (φ : Fin (p ^ m) → ℂ) : tensorP p (a • ψ) φ = a • tensorP p ψ φ := by
  funext idx
  simp [tensorP, mul_assoc]

theorem tensorP_smul_right [NeZero p] {n m : ℕ} (a : ℂ) (ψ : Fin (p ^ n) → ℂ)
    (φ : Fin (p ^ m) → ℂ) : tensorP p ψ (a • φ) = a • tensorP p ψ φ := by
  funext idx
  simp only [tensorP, Pi.smul_apply, smul_eq_mul]
  ring

/-- The pairwise tensor products of two finite sets of vectors. -/
noncomputable def tensorSetP (p : ℕ) [NeZero p] {n m : ℕ} (S : Finset (Fin (p ^ n) → ℂ))
    (T : Finset (Fin (p ^ m) → ℂ)) : Finset (Fin (p ^ (n + m)) → ℂ) := by
  classical
  exact (S ×ˢ T).image fun q => tensorP p q.1 q.2

theorem tensorP_mem_span [NeZero p] {n m : ℕ} {ψ : Fin (p ^ n) → ℂ} {φ : Fin (p ^ m) → ℂ}
    {S : Finset (Fin (p ^ n) → ℂ)} {T : Finset (Fin (p ^ m) → ℂ)}
    (hψ : ψ ∈ Submodule.span ℂ (S : Set (Fin (p ^ n) → ℂ)))
    (hφ : φ ∈ Submodule.span ℂ (T : Set (Fin (p ^ m) → ℂ))) :
    tensorP p ψ φ ∈ Submodule.span ℂ (tensorSetP p S T : Set (Fin (p ^ (n + m)) → ℂ)) := by
  classical
  obtain ⟨f, -, rfl⟩ := Submodule.mem_span_finset.mp hψ
  obtain ⟨g, -, rfl⟩ := Submodule.mem_span_finset.mp hφ
  rw [tensorP_sum_left]
  refine Submodule.sum_mem _ fun s hs => ?_
  rw [tensorP_smul_left, tensorP_sum_right]
  refine Submodule.smul_mem _ _ (Submodule.sum_mem _ fun t ht => ?_)
  rw [tensorP_smul_right]
  refine Submodule.smul_mem _ _ (Submodule.subset_span ?_)
  simp only [tensorSetP, Finset.coe_image, Set.mem_image, Finset.mem_coe, Finset.mem_product]
  exact ⟨(s, t), ⟨hs, ht⟩, rfl⟩

theorem tensorSetP_card_le [NeZero p] {n m : ℕ} (S : Finset (Fin (p ^ n) → ℂ))
    (T : Finset (Fin (p ^ m) → ℂ)) : (tensorSetP p S T).card ≤ S.card * T.card := by
  classical
  simp only [tensorSetP]
  exact le_trans Finset.card_image_le (le_of_eq (Finset.card_product _ _))

theorem tensorSetP_isStabP [Fact p.Prime] {n m : ℕ} {S : Finset (Fin (p ^ n) → ℂ)}
    {T : Finset (Fin (p ^ m) → ℂ)} (hS : ∀ σ ∈ S, IsStabP p σ) (hT : ∀ τ ∈ T, IsStabP p τ) :
    ∀ σ ∈ tensorSetP p S T, IsStabP p σ := by
  classical
  intro σ hσ
  simp only [tensorSetP, Finset.mem_image, Finset.mem_product] at hσ
  obtain ⟨⟨s, t⟩, ⟨hs, ht⟩, rfl⟩ := hσ
  exact (hS s hs).tensor (hT t ht)

/-- **Multiplicativity of the upper bound**: `χ(ψ ⊗ φ) ≤ χ(ψ) χ(φ)` against
    `IsStabP p`. -/
theorem stabRankP_tensor_le [Fact p.Prime] {n m : ℕ} (ψ : Fin (p ^ n) → ℂ)
    (φ : Fin (p ^ m) → ℂ) :
    stabRankP p (tensorP p ψ φ) ≤ stabRankP p ψ * stabRankP p φ := by
  obtain ⟨S, hS, hSstab, hSspan⟩ := Nat.sInf_mem (decompCardsP_nonempty p n ψ)
  obtain ⟨T, hT, hTstab, hTspan⟩ := Nat.sInf_mem (decompCardsP_nonempty p m φ)
  unfold stabRankP Stabilizer.stabRank
  rw [← hS, ← hT]
  exact le_trans (Stabilizer.stabRank_le_of_decomp (tensorSetP_isStabP hSstab hTstab)
    (tensorP_mem_span hSspan hTspan)) (tensorSetP_card_le S T)

end StabRank
