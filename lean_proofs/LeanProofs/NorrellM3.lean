/-
Norrell m=3 χ≤4 algebraic identity (paper App A.3.2).

The m=3 chi≤4 decomposition for the Norrell magic state collapses to
verifying a sum over 0/1-valued indicators (a₀, a₁, a₂) ∈ {0,1}³ of
the cube root of unity:

  B(a₀, a₁, a₂) := a₀·ω^(a₁ + 2a₂) + a₁·ω^(2a₀ + a₂) + a₂·ω^(a₀ + 2a₁)

depends only on n₂ := a₀ + a₁ + a₂ and takes values (0, 1, -1, 3) for
n₂ = 0, 1, 2, 3. All 8 cases are verified below; the kernel relations
are ω + ω² = -1 (for the n₂ = 2 cases) and ω³ = 1 (for the n₂ = 3
case), both already proved in `LeanProofs.Basic`.
-/
import LeanProofs.Basic

namespace StabRank

open Complex Real

/-- The Norrell m=3 sum:
    `B(a₀, a₁, a₂) = a₀·ω^(a₁+2a₂) + a₁·ω^(2a₀+a₂) + a₂·ω^(a₀+2a₁)`. -/
noncomputable def normalB (a₀ a₁ a₂ : ℕ) : ℂ :=
  (a₀ : ℂ) * omega3 ^ (a₁ + 2 * a₂) +
  (a₁ : ℂ) * omega3 ^ (2 * a₀ + a₂) +
  (a₂ : ℂ) * omega3 ^ (a₀ + 2 * a₁)

-- n₂ = 0: the single case (0, 0, 0) yields 0.
theorem norrell_m3_B_000 : normalB 0 0 0 = 0 := by simp [normalB]

-- n₂ = 1: each of the three cases yields 1 (a single ω⁰ = 1 contribution).
theorem norrell_m3_B_100 : normalB 1 0 0 = 1 := by simp [normalB]
theorem norrell_m3_B_010 : normalB 0 1 0 = 1 := by simp [normalB]
theorem norrell_m3_B_001 : normalB 0 0 1 = 1 := by simp [normalB]

-- n₂ = 2: each of the three cases yields ω + ω² = -1.
theorem norrell_m3_B_110 : normalB 1 1 0 = -1 := by
  simp [normalB]
  linear_combination omega3_add_omega3_sq

theorem norrell_m3_B_101 : normalB 1 0 1 = -1 := by
  simp [normalB]
  linear_combination omega3_add_omega3_sq

theorem norrell_m3_B_011 : normalB 0 1 1 = -1 := by
  simp [normalB]
  linear_combination omega3_add_omega3_sq

-- n₂ = 3: the single case (1, 1, 1) yields 3·ω³ = 3.
theorem norrell_m3_B_111 : normalB 1 1 1 = 3 := by
  simp [normalB]
  linear_combination 3 * omega3_pow_three

end StabRank
