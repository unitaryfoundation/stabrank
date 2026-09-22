/-
Slices of stabilizer states and the projection bound
`stabRankP p ψ ≤ stabRankP p (ψ ⊗ φ)` for `φ ≠ 0`, for any prime `p`.

`sliceP p v a` fixes the last digit of an `(n + 1)`-qudit vector `v` to `a`,
that is `x ↦ v (x, a)` on `n` qudits; it is `(I ⊗ ⟨a|) v` up to the digit
ordering. The structural fact is `IsStabP.slice`: a slice of a stabilizer
state is zero or a stabilizer state (with `IsStabP` absorbing the scalar).

Proof. Write the state as `stabVecP` with support `y ↦ x₀ + Wᵀ y` over
`(ZMod p)^k` and phase `ζ ^ quadPhaseP Q l y`. The slice keeps the `y` with
`x₀_n + Σ_j y_j W_j n = a`, one affine equation in `y`.

- If every `W_j n` vanishes, the equation is `x₀_n = a`: the slice is zero
  when it fails, and when it holds the slice is the same `stabVecP` with the
  last column of `x₀`, `W` dropped (`stabVecP_snoc_of_last_zero`).
- Otherwise pick `j₀` with `w = W_{j₀} n ≠ 0`, solve for
  `y_{j₀} = w⁻¹ (a - x₀_n) - Σ_t w⁻¹ W_{t} n · z_t` and parametrise the
  hyperplane by the remaining coordinates `z ∈ (ZMod p)^(k-1)` (`subP`).
  The support of the slice is `z ↦ x₀' + W'ᵀ z` with `x₀'`, `W'` read off
  `Fin.init` of the substituted parametrisation (`sliceX0`, `sliceW`), it is
  injective because the original one was, and the phase is
  `ζ ^ C · ζ ^ quadPhaseP Q' l' z` by `zeta_pow_quadPhaseP_comp` of
  `PhaseP.lean`, since every coordinate of `subP z` is affine in `z`.

Slicing is linear, so a decomposition of `v` into `r` stabilizer states
slices to a decomposition of `sliceP v a` into at most `r` stabilizer states
once the zero slices are dropped: `stabRankP_sliceP_le`. For a product,
`sliceP (ψ ⊗ φ) a = ψ ⊗ sliceP φ a`, and after `m` slices along the digits
of a string where `φ` is nonzero the right factor is a nonzero scalar, which
does not change the rank: `stabRankP_le_tensorP`. Together with
`stabRankP_tensor_le` of `TensorP.lean` this gives, for the tensor powers
`powVecP` of a one-qudit amplitude, monotonicity in the number of copies
and submultiplicativity in the exponent.
-/
import LeanProofs.Stabilizer.PhaseP
import LeanProofs.Stabilizer.TensorP

namespace StabRank

open Stabilizer

variable {p : ℕ}

/-! ### The slice -/

/-- Fix the last digit to `a`. -/
noncomputable def sliceP (p : ℕ) [NeZero p] {n : ℕ} (v : Fin (p ^ (n + 1)) → ℂ)
    (a : ZMod p) : Fin (p ^ n) → ℂ :=
  fun idx => v ((digitsP p (n + 1)).symm (Fin.snoc (digitsP p n idx) a))

theorem affinePtP_last {n k : ℕ} (x0 : Fin (n + 1) → ZMod p)
    (W : Fin k → Fin (n + 1) → ZMod p) (y : Fin k → ZMod p) :
    affinePtP x0 W y (Fin.last n) = x0 (Fin.last n) + ∑ j, y j * W j (Fin.last n) := rfl

theorem init_affinePtP {n k : ℕ} (x0 : Fin (n + 1) → ZMod p)
    (W : Fin k → Fin (n + 1) → ZMod p) (y : Fin k → ZMod p) :
    Fin.init (affinePtP x0 W y) = affinePtP (Fin.init x0) (fun j => Fin.init (W j)) y := rfl

/-! ### Case 1: the last column of `W` vanishes -/

section LastZero

variable {n k : ℕ} (x0 : Fin (n + 1) → ZMod p) (W : Fin k → Fin (n + 1) → ZMod p)

theorem affinePtP_eq_snoc (hW : ∀ j, W j (Fin.last n) = 0) (y : Fin k → ZMod p) :
    affinePtP x0 W y
      = Fin.snoc (affinePtP (Fin.init x0) (fun j => Fin.init (W j)) y) (x0 (Fin.last n)) := by
  rw [← init_affinePtP]
  have hlast : affinePtP x0 W y (Fin.last n) = x0 (Fin.last n) := by
    rw [affinePtP_last]
    simp [hW]
  rw [← hlast, Fin.snoc_init_self]

theorem affinePtP_init_injective (hW : ∀ j, W j (Fin.last n) = 0)
    (hinj : Function.Injective (affinePtP x0 W)) :
    Function.Injective (affinePtP (Fin.init x0) (fun j => Fin.init (W j))) := by
  intro y y' h
  apply hinj
  rw [affinePtP_eq_snoc x0 W hW, affinePtP_eq_snoc x0 W hW, h]

theorem stabVecP_snoc_of_last_zero [NeZero p] (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (hW : ∀ j, W j (Fin.last n) = 0) (a : ZMod p)
    (x : Fin n → ZMod p) :
    stabVecP p (n + 1) k x0 W Q l (Fin.snoc x a)
      = if x0 (Fin.last n) = a
        then stabVecP p n k (Fin.init x0) (fun j => Fin.init (W j)) Q l x else 0 := by
  unfold stabVecP
  split_ifs with ha
  · refine Finset.sum_congr rfl fun y _ => ?_
    simp only [affinePtP_eq_snoc x0 W hW, Fin.snoc_inj, ha, and_true]
  · refine Finset.sum_eq_zero fun y _ => ?_
    have ha' : a ≠ x0 (Fin.last n) := fun h => ha h.symm
    simp only [affinePtP_eq_snoc x0 W hW, Fin.snoc_inj, ha', and_false, if_false]

end LastZero

/-! ### Case 2: a pivot `j₀` with `W j₀ n ≠ 0` -/

section Pivot

variable [Fact p.Prime] {n k' : ℕ} (x0 : Fin (n + 1) → ZMod p)
  (W : Fin (k' + 1) → Fin (n + 1) → ZMod p) (j0 : Fin (k' + 1)) (a : ZMod p)

/-- The constant term of the eliminated parameter `y_{j₀}`. -/
def pivotConst : ZMod p := (W j0 (Fin.last n))⁻¹ * (a - x0 (Fin.last n))

/-- The coefficients of the remaining parameters in `y_{j₀}`. -/
def pivotCoef : Fin k' → ZMod p :=
  fun t => -((W j0 (Fin.last n))⁻¹ * W (j0.succAbove t) (Fin.last n))

/-- The parametrisation of the hyperplane: `y_{j₀}` is affine in the others. -/
def subP (z : Fin k' → ZMod p) : Fin (k' + 1) → ZMod p :=
  Fin.insertNth j0 (affineP (pivotConst x0 W j0 a) (pivotCoef W j0) z) z

/-- The base point of the sliced support. -/
def sliceX0 : Fin n → ZMod p :=
  fun i => x0 (Fin.castSucc i) + pivotConst x0 W j0 a * W j0 (Fin.castSucc i)

/-- The generators of the sliced support. -/
def sliceW : Fin k' → Fin n → ZMod p :=
  fun t i => W (j0.succAbove t) (Fin.castSucc i) + pivotCoef W j0 t * W j0 (Fin.castSucc i)

/-- The constant terms of the coordinates of `subP`. -/
def sliceAlpha : Fin (k' + 1) → ZMod p :=
  Fin.insertNth j0 (pivotConst x0 W j0 a) (fun _ => 0)

/-- The linear parts of the coordinates of `subP`. -/
def sliceN : Fin (k' + 1) → Fin k' → ZMod p :=
  Fin.insertNth j0 (pivotCoef W j0) (fun t0 => Pi.single t0 1)

theorem subP_apply_same (z : Fin k' → ZMod p) :
    subP x0 W j0 a z j0 = affineP (pivotConst x0 W j0 a) (pivotCoef W j0) z :=
  Fin.insertNth_apply_same _ _ _

theorem subP_apply_succAbove (z : Fin k' → ZMod p) (t : Fin k') :
    subP x0 W j0 a z (j0.succAbove t) = z t :=
  Fin.insertNth_apply_succAbove _ _ _ _

/-- Every coordinate of `subP z` is an affine function of `z`. -/
theorem subP_eq_affine (z : Fin k' → ZMod p) :
    subP x0 W j0 a z = fun i => affineP (sliceAlpha x0 W j0 a i) (sliceN W j0 i) z := by
  funext i
  rcases eq_or_ne i j0 with rfl | hne
  · simp [subP, sliceAlpha, sliceN, Fin.insertNth_apply_same]
  · obtain ⟨t, rfl⟩ := Fin.exists_succAbove_eq hne
    simp [subP, sliceAlpha, sliceN, Fin.insertNth_apply_succAbove, affineP, Pi.single_apply,
      ite_mul]

theorem affinePtP_insertNth_last (u : ZMod p) (z : Fin k' → ZMod p) :
    affinePtP x0 W (Fin.insertNth j0 u z) (Fin.last n)
      = x0 (Fin.last n) + u * W j0 (Fin.last n)
        + ∑ t, z t * W (j0.succAbove t) (Fin.last n) := by
  rw [affinePtP_last, Fin.sum_univ_succAbove _ j0, Fin.insertNth_apply_same, add_assoc]
  congr 2
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Fin.insertNth_apply_succAbove]

/-- The last coordinate hits `a` exactly when `u` is the eliminated value. -/
theorem insertNth_last_eq_iff (hw : W j0 (Fin.last n) ≠ 0) (u : ZMod p) (z : Fin k' → ZMod p) :
    affinePtP x0 W (Fin.insertNth j0 u z) (Fin.last n) = a
      ↔ u = affineP (pivotConst x0 W j0 a) (pivotCoef W j0) z := by
  rw [affinePtP_insertNth_last]
  unfold affineP pivotConst pivotCoef
  have hS : ∑ t, -((W j0 (Fin.last n))⁻¹ * W (j0.succAbove t) (Fin.last n)) * z t
      = -((W j0 (Fin.last n))⁻¹ * ∑ t, z t * W (j0.succAbove t) (Fin.last n)) := by
    rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun t _ => ?_
    ring
  rw [hS]
  constructor
  · intro h
    linear_combination (W j0 (Fin.last n))⁻¹ * h - u * inv_mul_cancel₀ hw
  · intro h
    linear_combination (W j0 (Fin.last n)) * h
      + (a - x0 (Fin.last n) - ∑ t, z t * W (j0.succAbove t) (Fin.last n)) * mul_inv_cancel₀ hw

/-- The first `n` coordinates of the substituted parametrisation. -/
theorem init_affinePtP_subP (z : Fin k' → ZMod p) :
    Fin.init (affinePtP x0 W (subP x0 W j0 a z))
      = affinePtP (sliceX0 x0 W j0 a) (sliceW W j0) z := by
  funext i
  simp only [Fin.init, affinePtP, sliceX0, sliceW]
  rw [Fin.sum_univ_succAbove _ j0, subP_apply_same]
  simp only [subP_apply_succAbove, affineP, add_mul, Finset.sum_mul, mul_add,
    Finset.sum_add_distrib]
  have h1 : ∀ t, pivotCoef W j0 t * z t * W j0 (Fin.castSucc i)
      = z t * (pivotCoef W j0 t * W j0 (Fin.castSucc i)) := fun t => by ring
  simp only [h1]
  ring

theorem affinePtP_subP_last (hw : W j0 (Fin.last n) ≠ 0) (z : Fin k' → ZMod p) :
    affinePtP x0 W (subP x0 W j0 a z) (Fin.last n) = a :=
  (insertNth_last_eq_iff x0 W j0 a hw _ z).mpr rfl

theorem sliceW_injective (hw : W j0 (Fin.last n) ≠ 0)
    (hinj : Function.Injective (affinePtP x0 W)) :
    Function.Injective (affinePtP (sliceX0 x0 W j0 a) (sliceW W j0)) := by
  intro z z' h
  have h2 : affinePtP x0 W (subP x0 W j0 a z) = affinePtP x0 W (subP x0 W j0 a z') := by
    rw [← Fin.snoc_init_self (affinePtP x0 W (subP x0 W j0 a z)),
      ← Fin.snoc_init_self (affinePtP x0 W (subP x0 W j0 a z')),
      init_affinePtP_subP, init_affinePtP_subP, affinePtP_subP_last x0 W j0 a hw,
      affinePtP_subP_last x0 W j0 a hw, h]
  have h3 := hinj h2
  funext t
  rw [← subP_apply_succAbove x0 W j0 a z t, ← subP_apply_succAbove x0 W j0 a z' t, h3]

/-- The slice at a pivot, as a `stabVecP` on `n` qudits times a phase. -/
theorem stabVecP_snoc_pivot (Q : Fin (k' + 1) → Fin (k' + 1) → ZMod p)
    (l : Fin (k' + 1) → ZMod (stabPeriod p)) (hw : W j0 (Fin.last n) ≠ 0) {C : ℕ}
    {Q' : Fin k' → Fin k' → ZMod p} {l' : Fin k' → ZMod (stabPeriod p)}
    (hphase : ∀ z, zeta p ^ quadPhaseP Q l
        (fun i => affineP (sliceAlpha x0 W j0 a i) (sliceN W j0 i) z)
      = zeta p ^ C * zeta p ^ quadPhaseP Q' l' z)
    (x : Fin n → ZMod p) :
    stabVecP p (n + 1) (k' + 1) x0 W Q l (Fin.snoc x a)
      = zeta p ^ C * stabVecP p n k' (sliceX0 x0 W j0 a) (sliceW W j0) Q' l' x := by
  unfold stabVecP
  rw [← Equiv.sum_comp (Fin.insertNthEquiv (fun _ => ZMod p) j0), Fintype.sum_prod_type,
    Finset.sum_comm, Finset.mul_sum]
  refine Finset.sum_congr rfl fun z _ => ?_
  simp only [Fin.insertNthEquiv, Equiv.coe_fn_mk]
  have hcond : ∀ u, (Fin.snoc x a = affinePtP x0 W (Fin.insertNth j0 u z))
      ↔ (u = affineP (pivotConst x0 W j0 a) (pivotCoef W j0) z
          ∧ x = Fin.init (affinePtP x0 W (Fin.insertNth j0 u z))) := by
    intro u
    conv_lhs => rw [← Fin.snoc_init_self (affinePtP x0 W (Fin.insertNth j0 u z))]
    rw [Fin.snoc_inj, eq_comm (a := a), insertNth_last_eq_iff x0 W j0 a hw u z, and_comm]
  simp_rw [hcond, ite_and, Finset.sum_ite_eq', Finset.mem_univ, if_true]
  change (if x = Fin.init (affinePtP x0 W (subP x0 W j0 a z))
    then zeta p ^ quadPhaseP Q l (subP x0 W j0 a z) else 0) = _
  rw [init_affinePtP_subP, subP_eq_affine, hphase, mul_ite, mul_zero]

end Pivot

/-! ### The slice of a stabilizer state -/

/-- A slice of a `stabVecP` with injective parametrisation is zero or a nonzero
    multiple of a `stabVecP` with injective parametrisation. -/
theorem stabVecP_snoc_cases [Fact p.Prime] {n k : ℕ} (x0 : Fin (n + 1) → ZMod p)
    (W : Fin k → Fin (n + 1) → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (hinj : Function.Injective (affinePtP x0 W)) (a : ZMod p) :
    (∀ x, stabVecP p (n + 1) k x0 W Q l (Fin.snoc x a) = 0) ∨
    ∃ (c : ℂ) (k' : ℕ) (x0' : Fin n → ZMod p) (W' : Fin k' → Fin n → ZMod p)
      (Q' : Fin k' → Fin k' → ZMod p) (l' : Fin k' → ZMod (stabPeriod p)),
      c ≠ 0 ∧ Function.Injective (affinePtP x0' W') ∧
      ∀ x, stabVecP p (n + 1) k x0 W Q l (Fin.snoc x a) = c * stabVecP p n k' x0' W' Q' l' x := by
  by_cases hW : ∀ j, W j (Fin.last n) = 0
  · by_cases ha : x0 (Fin.last n) = a
    · right
      exact ⟨1, k, Fin.init x0, fun j => Fin.init (W j), Q, l, one_ne_zero,
        affinePtP_init_injective x0 W hW hinj,
        fun x => by rw [stabVecP_snoc_of_last_zero x0 W Q l hW, if_pos ha, one_mul]⟩
    · left
      intro x
      rw [stabVecP_snoc_of_last_zero x0 W Q l hW, if_neg ha]
  · obtain ⟨j0, hw⟩ := not_forall.mp hW
    cases k with
    | zero => exact j0.elim0
    | succ k' =>
      right
      obtain ⟨C, Q', l', hphase⟩ :=
        zeta_pow_quadPhaseP_comp Q l (sliceAlpha x0 W j0 a) (sliceN W j0)
      exact ⟨zeta p ^ C, k', sliceX0 x0 W j0 a, sliceW W j0, Q', l',
        pow_ne_zero _ (zeta_ne_zero p), sliceW_injective x0 W j0 a hw hinj,
        fun x => stabVecP_snoc_pivot x0 W j0 a Q l hw hphase x⟩

/-- **A slice of a stabilizer state is zero or a stabilizer state.** -/
theorem IsStabP.slice [Fact p.Prime] {n : ℕ} {v : Fin (p ^ (n + 1)) → ℂ} (hv : IsStabP p v)
    (a : ZMod p) : sliceP p v a = 0 ∨ IsStabP p (sliceP p v a) := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hv
  have hs : sliceP p (fun idx => c * stabVecP p (n + 1) k x0 W Q l (digitsP p (n + 1) idx)) a
      = fun idx => c * stabVecP p (n + 1) k x0 W Q l (Fin.snoc (digitsP p n idx) a) := by
    funext idx
    simp [sliceP]
  rw [hs]
  rcases stabVecP_snoc_cases x0 W Q l hinj a with h | ⟨c', k', x0', W', Q', l', hc', hinj', h⟩
  · left
    funext idx
    simp [h]
  · right
    exact ⟨c * c', k', x0', W', Q', l', mul_ne_zero hc hc', hinj',
      by funext idx; rw [h]; ring⟩

/-! ### Slicing does not increase the rank -/

/-- Slicing as a linear map. -/
noncomputable def sliceLinP (p : ℕ) [NeZero p] {n : ℕ} (a : ZMod p) :
    (Fin (p ^ (n + 1)) → ℂ) →ₗ[ℂ] (Fin (p ^ n) → ℂ) where
  toFun v := sliceP p v a
  map_add' u v := by
    funext idx
    simp [sliceP]
  map_smul' c v := by
    funext idx
    simp [sliceP]

theorem stabRankP_sliceP_le [Fact p.Prime] {n : ℕ} (v : Fin (p ^ (n + 1)) → ℂ) (a : ZMod p) :
    stabRankP p (sliceP p v a) ≤ stabRankP p v := by
  classical
  obtain ⟨S, hS, hSstab, hSspan⟩ := Nat.sInf_mem (decompCardsP_nonempty p (n + 1) v)
  unfold stabRankP Stabilizer.stabRank
  rw [← hS]
  refine le_trans (Stabilizer.stabRank_le_of_decomp
    (S := (S.image fun σ => sliceP p σ a).erase 0) ?_ ?_)
    (le_trans Finset.card_erase_le Finset.card_image_le)
  · intro τ hτ
    rw [Finset.mem_erase, Finset.mem_image] at hτ
    obtain ⟨hne, σ, hσ, rfl⟩ := hτ
    rcases (hSstab σ hσ).slice a with h | h
    · exact absurd h hne
    · exact h
  · have h1 : sliceP p v a ∈ Submodule.span ℂ ((fun σ => sliceP p σ a) '' (S : Set _)) :=
      Submodule.apply_mem_span_image_of_mem_span (sliceLinP p a) hSspan
    have h2 : (fun σ => sliceP p σ a) '' (S : Set _)
        ⊆ insert (0 : Fin (p ^ n) → ℂ)
          (((S.image fun σ => sliceP p σ a).erase 0 : Finset _) : Set _) := by
      rintro τ ⟨σ, hσ, rfl⟩
      by_cases h0 : sliceP p σ a = 0
      · exact Or.inl h0
      · right
        rw [Finset.mem_coe, Finset.mem_erase, Finset.mem_image]
        exact ⟨h0, σ, hσ, rfl⟩
    have h3 := Submodule.span_mono h2 h1
    rwa [Submodule.span_insert_zero] at h3

/-! ### The projection bound -/

theorem stabRankP_smul [Fact p.Prime] {n : ℕ} (ψ : Fin (p ^ n) → ℂ) {c : ℂ}
    (hc : c ≠ 0) : stabRankP p (c • ψ) = stabRankP p ψ := by
  unfold stabRankP Stabilizer.stabRank Stabilizer.DecompCards
  congr 1
  ext k
  simp only [Set.mem_setOf_eq]
  constructor <;> rintro ⟨S, hk, hS, hmem⟩ <;> refine ⟨S, hk, hS, ?_⟩
  · exact (Submodule.smul_mem_iff _ hc).mp hmem
  · exact (Submodule.smul_mem_iff _ hc).mpr hmem

theorem splitLP_snoc {n m : ℕ} (x : Fin (n + m) → ZMod p) (a : ZMod p) :
    splitLP (n := n) (m := m + 1) (Fin.snoc x a : Fin (n + m + 1) → ZMod p) = splitLP x := by
  funext i
  simp only [splitLP]
  rw [show Fin.castAdd (m + 1) i = Fin.castSucc (Fin.castAdd m i) from Fin.ext rfl,
    Fin.snoc_castSucc]

theorem splitRP_snoc {n m : ℕ} (x : Fin (n + m) → ZMod p) (a : ZMod p) :
    splitRP (n := n) (m := m + 1) (Fin.snoc x a : Fin (n + m + 1) → ZMod p)
      = Fin.snoc (splitRP x) a := by
  funext j
  refine Fin.lastCases ?_ (fun j => ?_) j
  · simp only [splitRP, Fin.snoc_last]
    rw [show Fin.natAdd n (Fin.last m) = Fin.last (n + m) from Fin.ext rfl, Fin.snoc_last]
  · simp only [splitRP, Fin.snoc_castSucc]
    rw [show Fin.natAdd n (Fin.castSucc j) = Fin.castSucc (Fin.natAdd n j) from Fin.ext rfl,
      Fin.snoc_castSucc]

/-- A slice of a product is the product with the sliced right factor. -/
theorem sliceP_tensorP [NeZero p] {n m : ℕ} (ψ : Fin (p ^ n) → ℂ)
    (φ : Fin (p ^ (m + 1)) → ℂ) (a : ZMod p) :
    sliceP p (n := n + m) (tensorP p (n := n) (m := m + 1) ψ φ) a
      = tensorP p ψ (sliceP p φ a) := by
  have key : ∀ idx : Fin (p ^ (n + m + 1)), tensorP p (n := n) (m := m + 1) ψ φ idx
      = ψ ((digitsP p n).symm
          (splitLP (n := n) (m := m + 1) (digitsP p (n + m + 1) idx)))
        * φ ((digitsP p (m + 1)).symm
          (splitRP (n := n) (m := m + 1) (digitsP p (n + m + 1) idx))) :=
    fun idx => rfl
  funext idx
  rw [show sliceP p (n := n + m) (tensorP p (n := n) (m := m + 1) ψ φ) a idx
      = tensorP p (n := n) (m := m + 1) ψ φ
          ((digitsP p (n + m + 1)).symm (Fin.snoc (digitsP p (n + m) idx) a)) from rfl,
    key, Equiv.apply_symm_apply, splitLP_snoc, splitRP_snoc]
  simp only [tensorP, sliceP, Equiv.apply_symm_apply]

/-- With no right factor, `tensorP` is a scalar multiple. -/
theorem tensorP_zero_right [NeZero p] {n : ℕ} (ψ : Fin (p ^ n) → ℂ)
    (φ : Fin (p ^ 0) → ℂ) : tensorP p ψ φ = φ ((digitsP p 0).symm Fin.elim0) • ψ := by
  funext idx
  simp only [tensorP, Pi.smul_apply, smul_eq_mul]
  have hL : splitLP (n := n) (m := 0) (digitsP p (n + 0) idx) = digitsP p n idx := by
    funext i
    rfl
  have hR : splitRP (n := n) (m := 0) (digitsP p (n + 0) idx) = Fin.elim0 := by
    funext i
    exact i.elim0
  rw [hL, hR, Equiv.symm_apply_apply, mul_comm]

theorem eq_of_fin_pow_zero [NeZero p] (i j : Fin (p ^ 0)) : i = j :=
  (digitsP p 0).injective (funext fun t => Fin.elim0 t)

/-- **The projection bound**: `χ(ψ) ≤ χ(ψ ⊗ φ)` whenever `φ ≠ 0`. -/
theorem stabRankP_le_tensorP [Fact p.Prime] {n m : ℕ} (ψ : Fin (p ^ n) → ℂ)
    (φ : Fin (p ^ m) → ℂ) (hφ : φ ≠ 0) :
    stabRankP p ψ ≤ stabRankP p (tensorP p ψ φ) := by
  induction m with
  | zero =>
    rw [tensorP_zero_right]
    have h0 : φ ((digitsP p 0).symm Fin.elim0) ≠ 0 := by
      intro h
      apply hφ
      funext i
      rw [eq_of_fin_pow_zero i ((digitsP p 0).symm Fin.elim0)]
      exact h
    rw [stabRankP_smul _ h0]
  | succ m ih =>
    obtain ⟨idx, hidx⟩ : ∃ idx, φ idx ≠ 0 := Function.ne_iff.mp hφ
    have hφ' : sliceP p φ (digitsP p (m + 1) idx (Fin.last m)) ≠ 0 := by
      intro h
      have := congrFun h ((digitsP p m).symm (Fin.init (digitsP p (m + 1) idx)))
      simp only [sliceP, Equiv.apply_symm_apply, Fin.snoc_init_self, Equiv.symm_apply_apply,
        Pi.zero_apply] at this
      exact hidx this
    have := ih _ hφ'
    rw [← sliceP_tensorP] at this
    exact le_trans this (stabRankP_sliceP_le (n := n + m) _ _)

/-! ### Tensor powers of a one-qudit amplitude -/

/-- The `m`-fold tensor power of a one-qudit amplitude function, on digit strings.
    `hVec` and `tVec` of `QubitShared.lean` are `powVecP 2 hAmp1` and `powVecP 2 tAmp1`. -/
noncomputable def powVecP (p : ℕ) [NeZero p] (f : ZMod p → ℂ) (m : ℕ) :
    Fin (p ^ m) → ℂ :=
  fun idx => ∏ i, f (digitsP p m idx i)

theorem tensorP_powVecP [NeZero p] (f : ZMod p → ℂ) (a b : ℕ) :
    tensorP p (powVecP p f a) (powVecP p f b) = powVecP p f (a + b) := by
  funext idx
  simp only [tensorP, powVecP, Equiv.apply_symm_apply, Fin.prod_univ_add]
  rfl

theorem powVecP_one_ne_zero [NeZero p] (f : ZMod p → ℂ) (hf : f ≠ 0) :
    powVecP p f 1 ≠ 0 := by
  obtain ⟨u, hu⟩ : ∃ u, f u ≠ 0 := Function.ne_iff.mp hf
  intro h
  have := congrFun h ((digitsP p 1).symm fun _ => u)
  simp only [powVecP, Equiv.apply_symm_apply, Fin.prod_univ_one, Pi.zero_apply] at this
  exact hu this

/-- **Submultiplicativity in the exponent**: `χ(ψ^(a+b)) ≤ χ(ψ^a) χ(ψ^b)`. -/
theorem stabRankP_powVecP_add_le [Fact p.Prime] (f : ZMod p → ℂ) (a b : ℕ) :
    stabRankP p (powVecP p f (a + b))
      ≤ stabRankP p (powVecP p f a) * stabRankP p (powVecP p f b) := by
  rw [← tensorP_powVecP]
  exact stabRankP_tensor_le _ _

/-- **Projection, one copy**: `χ(ψ^m) ≤ χ(ψ^(m+1))` for a nonzero one-qudit `ψ`. -/
theorem stabRankP_powVecP_le_succ [Fact p.Prime] (f : ZMod p → ℂ) (hf : f ≠ 0) (m : ℕ) :
    stabRankP p (powVecP p f m) ≤ stabRankP p (powVecP p f (m + 1)) := by
  rw [← tensorP_powVecP]
  exact stabRankP_le_tensorP _ _ (powVecP_one_ne_zero f hf)

/-- **Monotonicity in the number of copies**: `χ(ψ^a) ≤ χ(ψ^b)` for `a ≤ b`. -/
theorem stabRankP_powVecP_mono [Fact p.Prime] (f : ZMod p → ℂ) (hf : f ≠ 0) {a b : ℕ}
    (h : a ≤ b) : stabRankP p (powVecP p f a) ≤ stabRankP p (powVecP p f b) := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact le_trans ih (stabRankP_powVecP_le_succ f hf _)

end StabRank
