/-
A Lean definition of a qutrit stabilizer state, and a proof that the T3 carry
blocks are ones.

Every Pointwise file in this development proves an identity between a target and
some explicitly written vectors, and then says in a comment that those vectors
are stabilizer states because of the form they were written in. Lean never
checks that part, so the `lean` tier rests on a definition-shaped assertion. The
same gap blocks the Galois lower bound, which needs stabilizer states to be
defined over Z[ω₃] and so needs them to be defined at all.

`stabVec` is the standard parametrisation on two qutrits,

  Σ_{y ∈ F_3^k} ω₃^(Q(y) + l·y) |x₀ + W y⟩,

and `v0T3 = stabVec ...` below discharges, for the carry blocks of the T3 m=2
decomposition, the claim their file only asserted.

`stabVec_mem_Zomega3` is the other half: every entry of a stabilizer state is an
integer combination of 1 and ω₃, since each term is a power of ω₃ and ω₃² =
-1-ω₃. That is the property the Galois descent runs on, and with it the
remaining gap in T3Galois is the descent step itself rather than the structure
of stabilizer states.
-/
import LeanProofs.T3M2Pointwise

namespace StabRank

open Complex

/-- Entries lie in `Z[ω₃]`: an integer combination of `1` and `ω₃`. -/
def OverZomega3 {ι : Type*} (v : ι → ℂ) : Prop :=
  ∀ i, ∃ a b : ℤ, v i = (a : ℂ) + (b : ℂ) * omega3

/-- `ω₃² = -1 - ω₃`, so every power of `ω₃` is an integer combination of `1`
    and `ω₃`. -/
theorem omega3_sq_eq_neg : omega3 ^ 2 = -1 - omega3 := by
  have h := omega3_sq_add_omega3_add_one
  linear_combination h

/-- Every power of `ω₃` lies in `Z[ω₃]`. -/
theorem omega3_pow_mem (j : ℕ) : ∃ a b : ℤ, omega3 ^ j = (a : ℂ) + (b : ℂ) * omega3 := by
  induction j with
  | zero => exact ⟨1, 0, by norm_num⟩
  | succ n ih =>
    obtain ⟨a, b, hab⟩ := ih
    -- ω^(n+1) = ω·ω^n = aω + bω² = -b + (a-b)ω
    refine ⟨-b, a - b, ?_⟩
    rw [pow_succ, hab]
    have h2 := omega3_sq_eq_neg
    push_cast
    linear_combination (b : ℂ) * h2

/-- A stabilizer state on two qutrits, in the standard parametrisation:
    supported on the flat `x₀ + span(W)`, with a quadratic phase in `ω₃`. -/
noncomputable def stabVec (k : ℕ) (x0 : Fin 2 → Fin 3) (W : Fin k → Fin 2 → Fin 3)
    (phase : (Fin k → Fin 3) → ℕ) : Fin 3 → Fin 3 → ℂ :=
  fun x1 x2 => ∑ y : Fin k → Fin 3,
    if x1 = x0 0 + ∑ j, y j * W j 0 ∧ x2 = x0 1 + ∑ j, y j * W j 1
      then omega3 ^ phase y else 0

/-- **Every stabilizer state is defined over `Z[ω₃]`.**

    This is the structural input the Galois descent needs: the span of a
    decomposition into stabilizer states is a span of vectors over `Q(ω₃)`, so
    it is fixed by `Gal(Q(ω₉)/Q(ω₃))`. -/
theorem stabVec_mem_Zomega3 (k : ℕ) (x0 : Fin 2 → Fin 3) (W : Fin k → Fin 2 → Fin 3)
    (phase : (Fin k → Fin 3) → ℕ) (x1 x2 : Fin 3) :
    ∃ a b : ℤ, stabVec k x0 W phase x1 x2 = (a : ℂ) + (b : ℂ) * omega3 := by
  classical
  unfold stabVec
  induction (Finset.univ : Finset (Fin k → Fin 3)) using Finset.induction with
  | empty => exact ⟨0, 0, by simp⟩
  | insert y s hy ih =>
    obtain ⟨a, b, hab⟩ := ih
    by_cases h : x1 = x0 0 + ∑ j, y j * W j 0 ∧ x2 = x0 1 + ∑ j, y j * W j 1
    · obtain ⟨c, d, hcd⟩ := omega3_pow_mem (phase y)
      refine ⟨a + c, b + d, ?_⟩
      rw [Finset.sum_insert hy, if_pos h, hab, hcd]
      push_cast; ring
    · refine ⟨a, b, ?_⟩
      rw [Finset.sum_insert hy, if_neg h, hab]
      ring

/-- The σ = 0 carry block really is a stabilizer state: `k = 1`, `x₀ = (0,0)`,
    `W = (1,2)`, phase `Q(t) = t²`. This is the claim T3M2Pointwise asserted in
    a comment and did not check. -/
theorem v0T3_is_stabilizer :
    v0T3 = stabVec 1 ![0, 0] ![![1, 2]] (fun y => ((y 0 : Fin 3).val) ^ 2) := by
  funext x1 x2
  have hsum : ∀ (f : (Fin 1 → Fin 3) → ℂ),
      (∑ y : Fin 1 → Fin 3, f y) = f ![0] + f ![1] + f ![2] := by
    intro f
    rw [show (Finset.univ : Finset (Fin 1 → Fin 3)) = {![0], ![1], ![2]} by decide]
    simp [Finset.sum_insert, Finset.mem_insert]
    ring
  fin_cases x1 <;> fin_cases x2 <;>
    simp (config := { decide := true }) [v0T3, stabVec, hsum]
  -- the surviving case is y = 2, where the phase is 2^2 = 4 and w3^3 = 1
  all_goals rw [show (4 : ℕ) = 3 + 1 from rfl, pow_add, pow_one, omega3_pow_three, one_mul]

end StabRank
