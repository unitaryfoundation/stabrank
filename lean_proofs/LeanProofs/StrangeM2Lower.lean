/-
Strange m=2 χ ≥ 2, the first lower bound in this development.

Lower bounds are universally quantified (no r stabilizer states reproduce the
target), which is why the rest of this development proves only upper bounds.
At r = 1 the quantifier is over a single state, and one structural fact closes
it without any enumeration: the support of a stabilizer state is an affine
subspace of F_3^n, hence closed under `a - b + c`.

|S⟩ = (|1⟩ - |2⟩)/√2 is supported on {1,2}, so |S⟩^⊗2 is supported on the four
points {1,2}². That set is not closed under `a - b + c`: taking a = d = (1,1)
and b = (2,2) lands on (0,0), where the amplitude vanishes. So |S⟩^⊗2 is not a
scalar multiple of any stabilizer state, giving χ(|S⟩^⊗2) ≥ 2. With the
matching χ ≤ 2 of StrangeM2Pointwise this settles χ(|S⟩^⊗2) = 2, both
directions machine-checked.

The affine-support property enters as the hypothesis `affine_support` rather
than being derived, exactly as the Pointwise files carry stabilizer-ness in the
shape of their definitions rather than in a Lean statement. It is the standard
structure theorem: a stabilizer state is uniform on a coset of a subgroup of
F_3^n up to phase. Anything satisfying it is covered, so the theorem is strictly
more general than the stabilizer case.
-/
import LeanProofs.StrangeM2Pointwise

namespace StabRank

open Complex Real

private theorem sqrt2_ne_zero_L : (Real.sqrt 2 : ℂ) ≠ 0 := by
  intro hz
  have hp : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 0 := by rw [hz]; ring
  have h2 : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
    exact_mod_cast Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)
  rw [h2] at hp; norm_num at hp

/-- `|S⟩^⊗2` vanishes at the origin: `[S]_0 = 0`. -/
theorem strangeAmp2_zero : strangeAmp2 (0, 0) = 0 := by
  simp [strangeAmp2, strangeAmp1']

/-- `|S⟩^⊗2` is nonzero at `(1,1)`, with amplitude `1/2`. -/
theorem strangeAmp2_one_one : strangeAmp2 (1, 1) ≠ 0 := by
  have h2 : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
    exact_mod_cast Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)
  simp only [strangeAmp2, strangeAmp1']
  field_simp
  rw [sq, h2]
  norm_num

/-- `|S⟩^⊗2` is nonzero at `(2,2)`, with amplitude `1/2`. -/
theorem strangeAmp2_two_two : strangeAmp2 (2, 2) ≠ 0 := by
  have h2 : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
    exact_mod_cast Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)
  simp only [strangeAmp2, strangeAmp1']
  field_simp
  rw [sq, h2]
  norm_num

/-- χ(|S⟩^⊗2) ≥ 2.

    No scalar multiple of a state with affine support equals `|S⟩^⊗2`. The
    hypothesis `affine_support` says the support of `s` is closed under
    `a - b + d`, which every stabilizer state satisfies. -/
theorem strange_m2_chi_ge_two
    (c : ℂ) (s : Fin 3 × Fin 3 → ℂ)
    (affine_support : ∀ a b d : Fin 3 × Fin 3,
        s a ≠ 0 → s b ≠ 0 → s d ≠ 0 → s (a - b + d) ≠ 0) :
    strangeAmp2 ≠ fun y => c * s y := by
  intro heq
  -- the target is nonzero at (1,1) and (2,2), so `c` and both values are nonzero
  have h11 : c * s (1, 1) ≠ 0 := by
    rw [← congrFun heq (1, 1)]; exact strangeAmp2_one_one
  have h22 : c * s (2, 2) ≠ 0 := by
    rw [← congrFun heq (2, 2)]; exact strangeAmp2_two_two
  have hs11 : s (1, 1) ≠ 0 := fun h => h11 (by rw [h]; ring)
  have hs22 : s (2, 2) ≠ 0 := fun h => h22 (by rw [h]; ring)
  -- affine closure at a = d = (1,1), b = (2,2) forces the origin into the support
  have hkey : s ((1, 1) - (2, 2) + (1, 1)) ≠ 0 :=
    affine_support (1, 1) (2, 2) (1, 1) hs11 hs22 hs11
  have hzero : ((1, 1) : Fin 3 × Fin 3) - (2, 2) + (1, 1) = (0, 0) := by decide
  rw [hzero] at hkey
  -- but the target vanishes at the origin, and `c ≠ 0`
  have hc : c ≠ 0 := fun h => h11 (by rw [h]; ring)
  have : c * s (0, 0) = 0 := by
    rw [← congrFun heq (0, 0)]; exact strangeAmp2_zero
  exact hkey (by
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h hc
    · exact h)

end StabRank
