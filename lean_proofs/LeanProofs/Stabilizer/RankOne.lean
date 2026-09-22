/-
`stabRank ψ > 1` from non-membership: a state that is not itself a stabilizer
state, up to a nonzero scalar, has stabilizer rank at least two.

`stabRank_gt_one_of_not_stab` is the reduction for an abstract predicate
closed under nonzero rescaling: a decomposition of size at most one is either
empty, which forces `ψ = 0`, or a single `σ` with `ψ = a • σ`, which makes
`ψ` itself satisfy the predicate. `IsStabP.smul` and `IsStab.smul` are the
closure properties of the two concrete predicates.

The tool that excludes a stabilizer state is `IsStabP.pow_period_eq`: on its
support a `stabVecP` is a power of `ζ_D`, so every nonzero entry of an
`IsStabP p` vector is `c ζ_D^e` and its `D`-th power is `c^D`, the same at
every point of the support. Two nonzero amplitudes whose `D`-th powers differ
therefore exclude the state, with no norms involved. This is the
"one modulus on the support" argument of
`verify_challenge/cert_rank1_moduli.py` in algebraic form.
-/
import LeanProofs.Stabilizer.IsStabP

namespace StabRank

open Stabilizer

/-! ### The reduction -/

/-- A decomposition of size at most one is empty or a single rescaled state,
    so a nonzero `ψ` outside a predicate closed under nonzero rescaling has
    rank at least two against it. -/
theorem stabRank_gt_one_of_not_stab {ι : Type*} (P : (ι → ℂ) → Prop) (ψ : ι → ℂ)
    (hne : (DecompCards P ψ).Nonempty)
    (hsmul : ∀ (σ : ι → ℂ) (a : ℂ), P σ → a ≠ 0 → P (a • σ))
    (hψ : ψ ≠ 0) (hP : ¬ P ψ) : Stabilizer.stabRank P ψ > 1 := by
  classical
  refine Stabilizer.stabRank_gt_of_no_decomp_le _ _ 1 hne ?_
  intro S hcard hstab hmem
  rcases S.eq_empty_or_nonempty with rfl | hS
  · rw [Finset.coe_empty, Submodule.span_empty, Submodule.mem_bot] at hmem
    exact hψ hmem
  · obtain ⟨σ, rfl⟩ := Finset.card_eq_one.mp (le_antisymm hcard hS.card_pos)
    rw [Finset.coe_singleton] at hmem
    obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hmem
    have ha : a ≠ 0 := by
      rintro rfl
      exact hψ (zero_smul ℂ σ)
    exact hP (hsmul σ a (hstab σ (Finset.mem_singleton_self σ)) ha)

/-! ### Closure under nonzero rescaling -/

theorem IsStabP.smul {p : ℕ} [Fact p.Prime] {n : ℕ} {σ : Fin (p ^ n) → ℂ}
    (hσ : IsStabP p σ) {a : ℂ} (ha : a ≠ 0) : IsStabP p (a • σ) := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hσ
  refine ⟨a * c, k, x0, W, Q, l, mul_ne_zero ha hc, hinj, ?_⟩
  funext idx
  simp only [Pi.smul_apply, smul_eq_mul, mul_assoc]

theorem IsStab.smul {n : ℕ} {σ : QutritVec n} (hσ : IsStab σ) {a : ℂ} (ha : a ≠ 0) :
    IsStab (a • σ) := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hσ
  refine ⟨a * c, k, x0, W, Q, l, mul_ne_zero ha hc, hinj, ?_⟩
  funext idx
  simp only [Pi.smul_apply, smul_eq_mul, mul_assoc]

/-! ### The support of a `stabVecP` -/

/-- A nonzero amplitude sits on the flat. -/
theorem stabVecP_ne_zero_imp {p n k : ℕ} [NeZero p] {x0 : Fin n → ZMod p}
    {W : Fin k → Fin n → ZMod p} {Q : Fin k → Fin k → ZMod p}
    {l : Fin k → ZMod (stabPeriod p)} {x : Fin n → ZMod p}
    (h : stabVecP p n k x0 W Q l x ≠ 0) : ∃ y, x = affinePtP x0 W y := by
  by_contra hne
  simp only [not_exists] at hne
  apply h
  unfold stabVecP
  exact Finset.sum_eq_zero (fun y _ => if_neg (hne y))

/-- On the flat, with an injective parametrisation, the amplitude is the pure
    phase `ζ_D^(Q(y)·(D/p) + l·y)`. -/
theorem stabVecP_apply_affinePtP {p n k : ℕ} [NeZero p] (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (hinj : Function.Injective (affinePtP x0 W))
    (y : Fin k → ZMod p) :
    stabVecP p n k x0 W Q l (affinePtP x0 W y) = zeta p ^ quadPhaseP Q l y := by
  unfold stabVecP
  rw [Finset.sum_eq_single y]
  · simp
  · intro y' _ hy'
    rw [if_neg]
    intro h
    exact hy' (hinj h).symm
  · simp

/-- **One modulus on the support, algebraically.** Every nonzero entry of an
    `IsStabP p` vector has the same `D`-th power, `D = stabPeriod p`. -/
theorem IsStabP.pow_period_eq {p : ℕ} [Fact p.Prime] {n : ℕ} {v : Fin (p ^ n) → ℂ}
    (hv : IsStabP p v) {x y : Fin (p ^ n)} (hx : v x ≠ 0) (hy : v y ≠ 0) :
    v x ^ stabPeriod p = v y ^ stabPeriod p := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hv
  have key : ∀ z : Fin (p ^ n),
      (fun idx => c * stabVecP p n k x0 W Q l (digitsP p n idx)) z ≠ 0 →
      (fun idx => c * stabVecP p n k x0 W Q l (digitsP p n idx)) z ^ stabPeriod p
        = c ^ stabPeriod p := by
    intro z hz
    have h' : stabVecP p n k x0 W Q l (digitsP p n z) ≠ 0 := by
      intro h0
      apply hz
      change c * stabVecP p n k x0 W Q l (digitsP p n z) = 0
      rw [h0, mul_zero]
    obtain ⟨w, hw⟩ := stabVecP_ne_zero_imp h'
    change (c * stabVecP p n k x0 W Q l (digitsP p n z)) ^ stabPeriod p = c ^ stabPeriod p
    rw [hw, stabVecP_apply_affinePtP x0 W Q l hinj, mul_pow, ← pow_mul,
      mul_comm (quadPhaseP Q l w), pow_mul, zeta_pow_period, one_pow, mul_one]
  exact (key x hx).trans (key y hy).symm

end StabRank
