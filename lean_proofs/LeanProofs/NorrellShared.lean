/-
Shared utilities for the Norrell-orbit pointwise proofs (paper App A.3).

Contains the single-qutrit Norrell amplitude and the `[y = 2]`
indicator used by all three Norrell pointwise files
(`NorrellM2Pointwise.lean`, `NorrellM3Pointwise.lean`,
`NorrellM4Pointwise.lean`), along with the elementary
sqrt-squaring lemmas they all rely on. Extracted so that the m=2 and
m=4 files do not have to depend on the m=3 file purely to reuse these.
-/
import LeanProofs.Basic

namespace StabRank

open Complex Real

/-- 1-qutrit Norrell amplitude: |N⟩ = (|0⟩ + |1⟩ - 2|2⟩)/√6.
    Stored as √2·√3/6 etc. to keep all algebra in (√2, √3). -/
noncomputable def norrellAmp1 : Fin 3 → ℂ
  | 0 =>  (Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) / 6
  | 1 =>  (Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ) / 6
  | 2 => -((Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ)) / 3

/-- Indicator a_i := [y_i = 2]. -/
def isTwo : Fin 3 → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1

theorem sqrt2_sq_cN : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
  exact_mod_cast Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)

theorem sqrt3_sq_cN : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 3 := by
  exact_mod_cast Real.mul_self_sqrt (by norm_num : (3:ℝ) ≥ 0)

end StabRank
