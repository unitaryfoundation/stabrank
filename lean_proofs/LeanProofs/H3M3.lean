/-
H_3 m=3 χ≤4 algebraic identity (paper App A.2.2).

The m=3 chi≤4 decomposition for the H_3 magic state collapses to a single
real scalar identity in `c = (√3 - 1)/2`:

  c · (1 + c) = 1/2.

This file machine-verifies that identity. The reduction from the full
27-amplitude pointwise equality system to this single scalar is in the
paper; here we only certify the algebraic kernel.
-/
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

namespace StabRank

open Real

/-- The H_3 m=3 coefficient: `c = (√3 - 1) / 2`. -/
noncomputable def cH3 : ℝ := (Real.sqrt 3 - 1) / 2

/-- H_3 m=3 scalar identity: `c · (1 + c) = 1/2`.

    Algebra: `c·(1+c) = ((√3-1)/2)·((√3+1)/2) = (√3·√3 - 1)/4 = (3-1)/4 = 1/2`. -/
theorem h3_m3_identity : cH3 * (1 + cH3) = 1/2 := by
  unfold cH3
  have h3 : Real.sqrt 3 * Real.sqrt 3 = 3 :=
    Real.mul_self_sqrt (by norm_num : (3:ℝ) ≥ 0)
  linear_combination (1/4 : ℝ) * h3

end StabRank
