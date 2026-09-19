/-
Tensor products of stabilizer states, and the multiplicativity of the
stabilizer-rank upper bound: `stabRank (ψ ⊗ φ) ≤ stabRank ψ * stabRank φ`.

`tensor ψ φ` is the product state on `n + m` qutrits, read off a digit string
by splitting it into its first `n` and last `m` digits. The tensor product of
two `stabVecN` vectors is again a `stabVecN`: the flats concatenate
(`Fin.append` on `x₀`, block-diagonal `W`), the phase polynomials add
(block-diagonal `Q`, `Fin.append` on `l`), and the sum over `F_3^(k+k')`
factors as a product of the two sums. Injectivity of the parametrisation is
inherited componentwise. This gives `IsStab.tensor`.

For the rank bound, a decomposition of `ψ` into `S` and of `φ` into `T` gives
a decomposition of `ψ ⊗ φ` into the pairwise tensor products, of which there
are at most `|S| · |T|`; taking `S`, `T` of minimal size (they exist because
the computational basis is a stabilizer basis) gives `stabRank_tensor_le`.

The file ends with the two consequences the board records: `|S⟩^⊗4` as the
tensor square of `|S⟩^⊗2` has rank at most `2 · 2 = 4`, and `|S⟩^⊗6` as the
tensor cube has rank at most `8`.
-/
import LeanProofs.Stabilizer.IsStab
import Mathlib.Logic.Equiv.Fin.Basic

namespace StabRank

open Stabilizer

/-! ### The tensor product on digit-indexed vectors -/

/-- The first `n` digits of a string on `n + m` qutrits. -/
def splitL {n m : ℕ} (x : Fin (n + m) → Fin 3) : Fin n → Fin 3 :=
  fun i => x (Fin.castAdd m i)

/-- The last `m` digits of a string on `n + m` qutrits. -/
def splitR {n m : ℕ} (x : Fin (n + m) → Fin 3) : Fin m → Fin 3 :=
  fun j => x (Fin.natAdd n j)

theorem append_splitL_splitR {n m : ℕ} (x : Fin (n + m) → Fin 3) :
    Fin.append (splitL x) (splitR x) = x :=
  Fin.append_castAdd_natAdd

/-- `ψ ⊗ φ`: the amplitude at a digit string is the product of the amplitudes
    at its two halves. -/
noncomputable def tensor {n m : ℕ} (ψ : QutritVec n) (φ : QutritVec m) : QutritVec (n + m) :=
  fun idx =>
    ψ ((digits n).symm (splitL (digits (n + m) idx)))
      * φ ((digits m).symm (splitR (digits (n + m) idx)))

theorem append_eq_append_iff {n m : ℕ} {a b : Fin n → Fin 3} {c d : Fin m → Fin 3} :
    Fin.append a c = Fin.append b d ↔ a = b ∧ c = d := by
  constructor
  · intro h
    have h' : ((a, c) : (Fin n → Fin 3) × (Fin m → Fin 3)) = (b, d) :=
      (Fin.appendEquiv n m).injective h
    exact Prod.mk.inj h'
  · rintro ⟨rfl, rfl⟩
    rfl

/-! ### Block data for the tensor product of two `stabVecN` -/

/-- Block-diagonal `W`: the first `k` rows act on the first `n` qutrits, the
    last `k'` rows on the last `m`. -/
def blockW {n m k k' : ℕ} (W : Fin k → Fin n → Fin 3) (W' : Fin k' → Fin m → Fin 3) :
    Fin (k + k') → Fin (n + m) → Fin 3 :=
  Fin.append (fun j => Fin.append (W j) 0) (fun j => Fin.append 0 (W' j))

/-- Block-diagonal `Q`. -/
def blockQ {k k' : ℕ} (Q : Fin k → Fin k → Fin 3) (Q' : Fin k' → Fin k' → Fin 3) :
    Fin (k + k') → Fin (k + k') → Fin 3 :=
  Fin.append (fun i => Fin.append (Q i) 0) (fun i => Fin.append 0 (Q' i))

theorem affinePt_append {n m k k' : ℕ} (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (x0' : Fin m → Fin 3) (W' : Fin k' → Fin m → Fin 3)
    (y : Fin k → Fin 3) (y' : Fin k' → Fin 3) :
    affinePt (Fin.append x0 x0') (blockW W W') (Fin.append y y')
      = Fin.append (affinePt x0 W y) (affinePt x0' W' y') := by
  funext i
  refine Fin.addCases (fun i => ?_) (fun i => ?_) i
  · simp [affinePt, blockW, Fin.sum_univ_add]
  · simp [affinePt, blockW, Fin.sum_univ_add]

theorem quadPhase_append {k k' : ℕ} (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3)
    (Q' : Fin k' → Fin k' → Fin 3) (l' : Fin k' → Fin 3)
    (y : Fin k → Fin 3) (y' : Fin k' → Fin 3) :
    quadPhase (blockQ Q Q') (Fin.append l l') (Fin.append y y')
      = quadPhase Q l y + quadPhase Q' l' y' := by
  simp only [quadPhase, blockQ, Fin.sum_univ_add, Fin.append_left, Fin.append_right,
    Pi.zero_apply, Fin.val_zero, zero_mul, Finset.sum_const_zero, add_zero, zero_add]
  ring

theorem affinePt_append_injective {n m k k' : ℕ} {x0 : Fin n → Fin 3}
    {W : Fin k → Fin n → Fin 3} {x0' : Fin m → Fin 3} {W' : Fin k' → Fin m → Fin 3}
    (h : Function.Injective (affinePt x0 W)) (h' : Function.Injective (affinePt x0' W')) :
    Function.Injective (affinePt (Fin.append x0 x0') (blockW W W')) := by
  intro z z' hz
  rw [← append_splitL_splitR z, ← append_splitL_splitR z', affinePt_append, affinePt_append,
    append_eq_append_iff] at hz
  rw [← append_splitL_splitR z, ← append_splitL_splitR z', h hz.1, h' hz.2]

/-- The sum over `F_3^(k+k')` factors: a tensor product of two `stabVecN` is
    the `stabVecN` of the block data. -/
theorem stabVecN_append {n m k k' : ℕ} (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3)
    (x0' : Fin m → Fin 3) (W' : Fin k' → Fin m → Fin 3)
    (Q' : Fin k' → Fin k' → Fin 3) (l' : Fin k' → Fin 3)
    (x : Fin n → Fin 3) (x' : Fin m → Fin 3) :
    stabVecN (n + m) (k + k') (Fin.append x0 x0') (blockW W W') (blockQ Q Q')
        (Fin.append l l') (Fin.append x x')
      = stabVecN n k x0 W Q l x * stabVecN m k' x0' W' Q' l' x' := by
  unfold stabVecN
  rw [Fintype.sum_mul_sum, ← Equiv.sum_comp (Fin.appendEquiv k k'), Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun y' _ => ?_
  rw [show (Fin.appendEquiv k k') (y, y') = Fin.append y y' from rfl]
  simp only [affinePt_append, quadPhase_append, append_eq_append_iff, pow_add,
    ite_zero_mul_ite_zero]

/-- **The tensor product of two stabilizer states is a stabilizer state.** -/
theorem IsStab.tensor {n m : ℕ} {ψ : QutritVec n} {φ : QutritVec m}
    (hψ : IsStab ψ) (hφ : IsStab φ) : IsStab (tensor ψ φ) := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hψ
  obtain ⟨c', k', x0', W', Q', l', hc', hinj', rfl⟩ := hφ
  refine ⟨c * c', k + k', Fin.append x0 x0', blockW W W', blockQ Q Q', Fin.append l l',
    mul_ne_zero hc hc', affinePt_append_injective hinj hinj', ?_⟩
  funext idx
  simp only [StabRank.tensor, Equiv.apply_symm_apply]
  conv_rhs => rw [← append_splitL_splitR (digits (n + m) idx)]
  rw [stabVecN_append]
  ring

/-! ### Bilinearity, and spans of tensor products -/

theorem tensor_sum_left {n m : ℕ} {ι : Type*} (s : Finset ι) (f : ι → QutritVec n)
    (φ : QutritVec m) : tensor (∑ i ∈ s, f i) φ = ∑ i ∈ s, tensor (f i) φ := by
  funext idx
  simp [tensor, Finset.sum_apply, Finset.sum_mul]

theorem tensor_sum_right {n m : ℕ} {ι : Type*} (ψ : QutritVec n) (s : Finset ι)
    (g : ι → QutritVec m) : tensor ψ (∑ i ∈ s, g i) = ∑ i ∈ s, tensor ψ (g i) := by
  funext idx
  simp [tensor, Finset.sum_apply, Finset.mul_sum]

theorem tensor_smul_left {n m : ℕ} (a : ℂ) (ψ : QutritVec n) (φ : QutritVec m) :
    tensor (a • ψ) φ = a • tensor ψ φ := by
  funext idx
  simp [tensor, mul_assoc]

theorem tensor_smul_right {n m : ℕ} (a : ℂ) (ψ : QutritVec n) (φ : QutritVec m) :
    tensor ψ (a • φ) = a • tensor ψ φ := by
  funext idx
  simp only [tensor, Pi.smul_apply, smul_eq_mul]
  ring

/-- The pairwise tensor products of two finite sets of vectors. -/
noncomputable def tensorSet {n m : ℕ} (S : Finset (QutritVec n)) (T : Finset (QutritVec m)) :
    Finset (QutritVec (n + m)) := by
  classical
  exact (S ×ˢ T).image fun p => tensor p.1 p.2

theorem tensor_mem_span {n m : ℕ} {ψ : QutritVec n} {φ : QutritVec m}
    {S : Finset (QutritVec n)} {T : Finset (QutritVec m)}
    (hψ : ψ ∈ Submodule.span ℂ (S : Set (QutritVec n)))
    (hφ : φ ∈ Submodule.span ℂ (T : Set (QutritVec m))) :
    tensor ψ φ ∈ Submodule.span ℂ (tensorSet S T : Set (QutritVec (n + m))) := by
  classical
  obtain ⟨f, -, rfl⟩ := Submodule.mem_span_finset.mp hψ
  obtain ⟨g, -, rfl⟩ := Submodule.mem_span_finset.mp hφ
  rw [tensor_sum_left]
  refine Submodule.sum_mem _ fun s hs => ?_
  rw [tensor_smul_left, tensor_sum_right]
  refine Submodule.smul_mem _ _ (Submodule.sum_mem _ fun t ht => ?_)
  rw [tensor_smul_right]
  refine Submodule.smul_mem _ _ (Submodule.subset_span ?_)
  simp only [tensorSet, Finset.coe_image, Set.mem_image, Finset.mem_coe, Finset.mem_product]
  exact ⟨(s, t), ⟨hs, ht⟩, rfl⟩

theorem tensorSet_card_le {n m : ℕ} (S : Finset (QutritVec n)) (T : Finset (QutritVec m)) :
    (tensorSet S T).card ≤ S.card * T.card := by
  classical
  simp only [tensorSet]
  exact le_trans Finset.card_image_le (le_of_eq (Finset.card_product _ _))

theorem tensorSet_isStab {n m : ℕ} {S : Finset (QutritVec n)} {T : Finset (QutritVec m)}
    (hS : ∀ σ ∈ S, IsStab σ) (hT : ∀ τ ∈ T, IsStab τ) :
    ∀ σ ∈ tensorSet S T, IsStab σ := by
  classical
  intro σ hσ
  simp only [tensorSet, Finset.mem_image, Finset.mem_product] at hσ
  obtain ⟨⟨s, t⟩, ⟨hs, ht⟩, rfl⟩ := hσ
  exact (hS s hs).tensor (hT t ht)

/-- **Multiplicativity of the upper bound**: `χ(ψ ⊗ φ) ≤ χ(ψ) χ(φ)` against the
    concrete stabilizer predicate. -/
theorem stabRank_tensor_le {n m : ℕ} (ψ : QutritVec n) (φ : QutritVec m) :
    Stabilizer.stabRank (IsStab (n := n + m)) (tensor ψ φ)
      ≤ Stabilizer.stabRank (IsStab (n := n)) ψ * Stabilizer.stabRank (IsStab (n := m)) φ := by
  obtain ⟨S, hS, hSstab, hSspan⟩ := Nat.sInf_mem (decompCards_nonempty n ψ)
  obtain ⟨T, hT, hTstab, hTspan⟩ := Nat.sInf_mem (decompCards_nonempty m φ)
  unfold Stabilizer.stabRank
  rw [← hS, ← hT]
  exact le_trans (Stabilizer.stabRank_le_of_decomp (tensorSet_isStab hSstab hTstab)
    (tensor_mem_span hSspan hTspan)) (tensorSet_card_le S T)

/-! ### Strange m=4 and m=6 -/

/-- `|S⟩^⊗4` as the tensor square of `|S⟩^⊗2`. -/
noncomputable def strangeVec4 : QutritVec 4 := tensor strangeVec2 strangeVec2

/-- `|S⟩^⊗6` as the tensor cube of `|S⟩^⊗2`. -/
noncomputable def strangeVec6 : QutritVec 6 := tensor strangeVec4 strangeVec2

/-- `strangeVec4` is the four-fold product of the one-qutrit Strange
    amplitude: this is the state the board's `S`, `m = 4` cell refers to. -/
theorem strangeVec4_apply (idx : Fin (3 ^ 4)) :
    strangeVec4 idx = ∏ i : Fin 4, strangeAmp1' (digits 4 idx i) := by
  simp only [strangeVec4, tensor, strangeVec2, ofPair, Equiv.apply_symm_apply,
    finTwoArrowEquiv_pair, strangeAmp2, splitL, splitR, Fin.prod_univ_four]
  change strangeAmp1' (digits 4 idx 0) * strangeAmp1' (digits 4 idx 1)
    * (strangeAmp1' (digits 4 idx 2) * strangeAmp1' (digits 4 idx 3)) = _
  ring

/-- `strangeVec6` is the six-fold product of the one-qutrit Strange amplitude. -/
theorem strangeVec6_apply (idx : Fin (3 ^ 6)) :
    strangeVec6 idx = ∏ i : Fin 6, strangeAmp1' (digits 6 idx i) := by
  simp only [strangeVec6, tensor, strangeVec4_apply, strangeVec2, ofPair,
    Equiv.apply_symm_apply, finTwoArrowEquiv_pair, strangeAmp2, splitL, splitR,
    Fin.prod_univ_four, Fin.prod_univ_six]
  change strangeAmp1' (digits 6 idx 0) * strangeAmp1' (digits 6 idx 1)
    * strangeAmp1' (digits 6 idx 2) * strangeAmp1' (digits 6 idx 3)
    * (strangeAmp1' (digits 6 idx 4) * strangeAmp1' (digits 6 idx 5)) = _
  ring

/-- **χ(|S⟩^⊗4) ≤ 4 against the concrete stabilizer predicate.** -/
theorem strange_m4_stabRank_le_four :
    Stabilizer.stabRank (IsStab (n := 4)) strangeVec4 ≤ 4 :=
  le_trans (stabRank_tensor_le strangeVec2 strangeVec2)
    (Nat.mul_le_mul strange_m2_stabRank_le_two strange_m2_stabRank_le_two)

/-- **χ(|S⟩^⊗6) ≤ 8 against the concrete stabilizer predicate.** -/
theorem strange_m6_stabRank_le_eight :
    Stabilizer.stabRank (IsStab (n := 6)) strangeVec6 ≤ 8 :=
  le_trans (stabRank_tensor_le strangeVec4 strangeVec2)
    (Nat.mul_le_mul strange_m4_stabRank_le_four strange_m2_stabRank_le_two)

end StabRank
