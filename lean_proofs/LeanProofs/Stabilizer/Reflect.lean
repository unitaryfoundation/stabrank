/-
Reflection of a stabilizer decomposition to an identity of integer vectors.

A witness on the board is a list of stabilizer terms with coefficients in a
number field `K = ℚ(ζ, θ₁, …)`. Checking `ψ = Σ αⱼ σⱼ` amplitude by amplitude
with `simp` and `linear_combination` costs a case per digit string and a
cofactor per case. Here the identity is instead decided by the kernel on
integer coordinate vectors, and a handful of generic lemmas carry the result
to `ℂ`:

* `ev B v = Σᵢ vᵢ Bᵢ` evaluates an integer vector against a basis `B` of the
  ℤ-span in which everything lives; `ev` is additive and commutes with
  `if … then v else 0`.
* `mulM M` is multiplication by an integer matrix; `Represents B M θ` says
  `M` is the matrix of multiplication by `θ` on `B`, and then
  `ev B (mulM M v) = θ * ev B v`. Powers of `θ` are iterates of `mulM M`.
* `ysol x0 piv x` recovers the flat coordinates of a digit string from the
  pivot columns of `W`, and `stabVecP_eq_ite_of_pivots` turns the sum over
  `F_p^k` in `stabVecP` into a single `if`, so the ℤ-side `termZ` costs
  `O(k n)` per digit string instead of `O(p^k n)`.
* `termZ M C … x` is the integer vector `C · ζ^(phase)` when `x` is on the
  flat and `0` otherwise, and `ev_termZ` identifies its evaluation with
  `ev B C * stabVecP … x`.

A cell then supplies its basis, the matrix of `ζ`, one integer vector `Cⱼ`
per term (the coefficient with the `1/√(p^k)` normalisation and a common
scalar folded in, times a common denominator), a computable integer vector
for the target amplitude, and one `decide +kernel` over the `p^n` indices.
The generated modules in `LeanProofs/*StabRank.lean` produced by
`tools/gen_witness_lean.py` use exactly these pieces.
-/
import LeanProofs.Stabilizer.IsStabP

namespace StabRank

open Stabilizer

/-! ### Integer vectors against a basis -/

section Ev

variable {r : ℕ} (B : Fin r → ℂ)

/-- Evaluate an integer coordinate vector against the basis `B`. -/
noncomputable def ev (v : Fin r → ℤ) : ℂ := ∑ i, (v i : ℂ) * B i

theorem ev_zero : ev B 0 = 0 := by simp [ev]

theorem ev_add (v w : Fin r → ℤ) : ev B (v + w) = ev B v + ev B w := by
  simp [ev, add_mul, Finset.sum_add_distrib]

theorem ev_zsmul (c : ℤ) (v : Fin r → ℤ) : ev B (c • v) = (c : ℂ) * ev B v := by
  simp [ev, Finset.mul_sum, mul_assoc]

theorem ev_ite (P : Prop) [Decidable P] (v : Fin r → ℤ) :
    ev B (if P then v else 0) = if P then ev B v else 0 := by
  split_ifs <;> simp [ev_zero]

/-- The vector with `c` in coordinate `0` and `0` elsewhere. -/
def unitZ (c : ℤ) : Fin r → ℤ := fun i => if (i : ℕ) = 0 then c else 0

theorem ev_unitZ (hr : 0 < r) (hB0 : B ⟨0, hr⟩ = 1) (c : ℤ) : ev B (unitZ c) = c := by
  unfold ev unitZ
  rw [Finset.sum_eq_single ⟨0, hr⟩]
  · simp [hB0]
  · intro i _ hi
    have : (i : ℕ) ≠ 0 := fun h => hi (Fin.ext h)
    simp [this]
  · simp

/-- Multiplication by an integer matrix, `(mulM M v) i = Σⱼ M i j * v j`. -/
def mulM (M : Fin r → Fin r → ℤ) (v : Fin r → ℤ) : Fin r → ℤ := fun i => ∑ j, M i j * v j

/-- `M` is the matrix of multiplication by `θ` on the basis `B`: column `j`
    holds the coordinates of `θ * B j`. -/
def Represents (M : Fin r → Fin r → ℤ) (θ : ℂ) : Prop :=
  ∀ j, θ * B j = ∑ i, (M i j : ℂ) * B i

theorem ev_mulM {M : Fin r → Fin r → ℤ} {θ : ℂ} (h : Represents B M θ) (v : Fin r → ℤ) :
    ev B (mulM M v) = θ * ev B v := by
  unfold ev mulM
  calc ∑ i, ((∑ j, M i j * v j : ℤ) : ℂ) * B i
      = ∑ i, ∑ j, (v j : ℂ) * ((M i j : ℂ) * B i) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        push_cast
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun j _ => ?_
        ring
    _ = ∑ j, (v j : ℂ) * ∑ i, (M i j : ℂ) * B i := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [Finset.mul_sum]
    _ = ∑ j, (v j : ℂ) * (θ * B j) := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [h j]
    _ = θ * ∑ j, (v j : ℂ) * B j := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun j _ => ?_
        ring

theorem ev_mulM_iterate {M : Fin r → Fin r → ℤ} {θ : ℂ} (h : Represents B M θ) (t : ℕ)
    (v : Fin r → ℤ) : ev B ((mulM M)^[t] v) = θ ^ t * ev B v := by
  induction t with
  | zero => simp
  | succ t ih =>
    rw [Function.iterate_succ_apply', ev_mulM B h, ih, pow_succ]
    ring

end Ev

/-! ### The flat coordinates from pivot columns -/

/-- Recover the flat coordinates of `x` from the pivot columns of `W`. -/
def ysol {p n k : ℕ} (x0 : Fin n → ZMod p) (piv : Fin k → Fin n) (x : Fin n → ZMod p) :
    Fin k → ZMod p :=
  fun j => x (piv j) - x0 (piv j)

theorem ysol_affinePtP {p n k : ℕ} (x0 : Fin n → ZMod p) (W : Fin k → Fin n → ZMod p)
    (piv : Fin k → Fin n) (h : ∀ j j', W j (piv j') = if j = j' then 1 else 0)
    (y : Fin k → ZMod p) : ysol x0 piv (affinePtP x0 W y) = y := by
  funext j
  simp [ysol, affinePtP, h, mul_ite, Finset.sum_ite_eq']

/-- With pivot columns, `stabVecP` at `x` is a single `if`: `x` is on the flat
    exactly when it is the image of its recovered coordinates. -/
theorem stabVecP_eq_ite_of_pivots (p : ℕ) [NeZero p] {n k : ℕ} (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p) (l : Fin k → ZMod (stabPeriod p))
    (piv : Fin k → Fin n) (h : ∀ j j', W j (piv j') = if j = j' then 1 else 0)
    (x : Fin n → ZMod p) :
    stabVecP p n k x0 W Q l x
      = if x = affinePtP x0 W (ysol x0 piv x)
          then zeta p ^ quadPhaseP Q l (ysol x0 piv x) else 0 := by
  unfold stabVecP
  split_ifs with hx
  · rw [Finset.sum_eq_single (ysol x0 piv x)]
    · rw [if_pos hx]
    · intro y _ hy
      rw [if_neg]
      intro hxy
      apply hy
      rw [hxy, ysol_affinePtP x0 W piv h]
    · simp
  · refine Finset.sum_eq_zero fun y _ => ?_
    rw [if_neg]
    intro hxy
    apply hx
    rw [hxy, ysol_affinePtP x0 W piv h]

/-! ### The integer side of one term -/

/-- The integer vector `C · ζ^(phase)` on the flat and `0` off it, with the
    phase reduced modulo the period so the iterate count stays small. -/
def termZ {r p n k : ℕ} (M : Fin r → Fin r → ℤ) (C : Fin r → ℤ) (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p) (l : Fin k → ZMod (stabPeriod p))
    (piv : Fin k → Fin n) (x : Fin n → ZMod p) : Fin r → ℤ :=
  if x = affinePtP x0 W (ysol x0 piv x)
    then (mulM M)^[quadPhaseP Q l (ysol x0 piv x) % stabPeriod p] C else 0

theorem ev_termZ {r : ℕ} (B : Fin r → ℂ) {p : ℕ} [NeZero p] {n k : ℕ}
    {M : Fin r → Fin r → ℤ} (hM : Represents B M (zeta p))
    {W : Fin k → Fin n → ZMod p} {piv : Fin k → Fin n}
    (hpiv : ∀ j j', W j (piv j') = if j = j' then 1 else 0)
    (C : Fin r → ℤ) (x0 : Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (x : Fin n → ZMod p) :
    ev B (termZ M C x0 W Q l piv x) = ev B C * stabVecP p n k x0 W Q l x := by
  rw [stabVecP_eq_ite_of_pivots p x0 W Q l piv hpiv, termZ]
  split_ifs
  · rw [ev_mulM_iterate B hM, ← zeta_pow_mod]
    ring
  · simp [ev_zero]

/-- The number of zero digits of a string. -/
def count0 {p n : ℕ} (x : Fin n → ZMod p) : ℕ := (Finset.univ.filter fun i => x i = 0).card

theorem count0_le {p n : ℕ} (x : Fin n → ZMod p) : count0 x ≤ n := by
  unfold count0
  exact le_trans (Finset.card_filter_le _ _) (by simp)

theorem count0_add_count_ne {p n : ℕ} (x : Fin n → ZMod p) :
    count0 x + (Finset.univ.filter fun i => ¬ x i = 0).card = n := by
  unfold count0
  rw [Finset.card_filter_add_card_filter_not]
  simp

/-- A product of a two-valued amplitude over the digits, by the number of
    zero digits. -/
theorem prod_two_valued {p n : ℕ} (x : Fin n → ZMod p) (a b : ℂ) :
    (∏ i, if x i = 0 then a else b) = a ^ count0 x * b ^ (n - count0 x) := by
  rw [Finset.prod_ite]
  simp only [Finset.prod_const]
  congr 2
  have := count0_add_count_ne x
  unfold count0 at this ⊢
  omega

end StabRank
