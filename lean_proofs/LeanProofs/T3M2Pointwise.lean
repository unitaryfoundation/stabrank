/-
T3 m=2 χ ≤ 3 pointwise vector identity (carry construction, Kocia-Sarovar
arXiv:2003.01130).

  [|T3⟩^⊗2]_y = (v_0(y) + ω₉·v_1(y) + ω₉²·v_2(y)) / 3   for every y ∈ F_3².

|T3⟩ = (|0⟩ + ω₉|1⟩ + ω₉²|2⟩)/√3, so the m=2 amplitude at (x₁,x₂) is
ω₉^(x₁+x₂)/3.  Writing x₁+x₂ = σ + 3c with σ = (x₁+x₂) mod 3 and c the carry
splits that as ω₉^σ · ω₃^c, which is exactly the carry decomposition: v_σ
collects the three points of the line x₁+x₂ ≡ σ (mod 3), carrying phase ω₃^c.

Each v_σ is a stabilizer state.  Its support is the affine line
{(t, σ-t) : t ∈ F_3}, so k = 1 with x₀ = (0,σ) and W = (1,2), and on that
line the carry is c(t) = t² mod 3, a quadratic phase.  In the parametrisation
  Σ_y ω₃^(Q(y)+l·y) |x₀ + W y⟩
this is Q(t) = t², l = 0.  As in the other Pointwise files, Lean checks the
identity; that the v_σ are stabilizer states is carried by the shape of the
definition above, not by a separate Lean statement.

This file establishes only the χ ≤ 3 upper bound.  The matching χ ≥ 3 follows
from the Galois argument (dim V_m = 3 at every m), which is not formalized.
-/
import LeanProofs.Basic

namespace StabRank

open Complex Real

/-- Primitive 9th root of unity. -/
noncomputable def omega9 : ℂ := Complex.exp (2 * Real.pi * Complex.I / 9)

/-- `ω₉³ = ω₃`: cubing a primitive 9th root gives a primitive 3rd root. -/
theorem omega9_cube : omega9 ^ 3 = omega3 := by
  unfold omega9 omega3
  rw [← Complex.exp_nat_mul]
  congr 1
  push_cast
  ring

/-- 1-qutrit T3 amplitude: `ω₉^x / √3`. -/
noncomputable def t3Amp1 : Fin 3 → ℂ
  | 0 => 1 / (Real.sqrt 3 : ℂ)
  | 1 => omega9 / (Real.sqrt 3 : ℂ)
  | 2 => omega9 ^ 2 / (Real.sqrt 3 : ℂ)

/-- 2-qutrit T3⊗² amplitude. -/
noncomputable def t3Amp2 (y : Fin 3 × Fin 3) : ℂ :=
  t3Amp1 y.1 * t3Amp1 y.2

/-- Carry block σ = 0: supported on x₁+x₂ ≡ 0, phases `1, ω₃, ω₃`. -/
noncomputable def v0T3 : Fin 3 → Fin 3 → ℂ
  | ⟨0, _⟩, ⟨0, _⟩ => 1
  | ⟨1, _⟩, ⟨2, _⟩ => omega3
  | ⟨2, _⟩, ⟨1, _⟩ => omega3
  | ⟨_, _⟩, ⟨_, _⟩ => 0

/-- Carry block σ = 1: supported on x₁+x₂ ≡ 1, phases `1, 1, ω₃`. -/
noncomputable def v1T3 : Fin 3 → Fin 3 → ℂ
  | ⟨0, _⟩, ⟨1, _⟩ => 1
  | ⟨1, _⟩, ⟨0, _⟩ => 1
  | ⟨2, _⟩, ⟨2, _⟩ => omega3
  | ⟨_, _⟩, ⟨_, _⟩ => 0

/-- Carry block σ = 2: supported on x₁+x₂ ≡ 2, all phases `1`. -/
noncomputable def v2T3 : Fin 3 → Fin 3 → ℂ
  | ⟨0, _⟩, ⟨2, _⟩ => 1
  | ⟨1, _⟩, ⟨1, _⟩ => 1
  | ⟨2, _⟩, ⟨0, _⟩ => 1
  | ⟨_, _⟩, ⟨_, _⟩ => 0

private theorem sqrt3_sq_T2 : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 3 := by
  exact_mod_cast Real.mul_self_sqrt (by norm_num : (3:ℝ) ≥ 0)

/-- The two `1/√3` normalisations combine into a single `1/3`, once. Clearing
    the radical here keeps the nine case computations purely in `ω₉`. -/
private theorem mul_div_sqrt3 (a b : ℂ) :
    (a / (Real.sqrt 3 : ℂ)) * (b / (Real.sqrt 3 : ℂ)) = a * b / 3 := by
  rw [div_mul_div_comm, sqrt3_sq_T2]

set_option linter.flexible false in
/-- Pointwise carry identity:
    `[|T3⟩^⊗2]_y = (v_0(y) + ω₉·v_1(y) + ω₉²·v_2(y))/3` at every y ∈ F_3².

    Each case is `ω₉^(x₁+x₂) = ω₉^σ · ω₃^c`. The six with carry `c = 0` are
    `ring` identities; the two with `x₁+x₂ = 3` need `ω₉³ = ω₃` once, and the
    single case `x₁+x₂ = 4` needs it scaled by `ω₉`. -/
theorem t3_m2_decomposition (y : Fin 3 × Fin 3) :
    t3Amp2 y = (v0T3 y.1 y.2 + omega9 * v1T3 y.1 y.2
                  + omega9 ^ 2 * v2T3 y.1 y.2) / 3 := by
  have hc := omega9_cube
  obtain ⟨y_0, y_1⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;>
    simp only [t3Amp2, t3Amp1, v0T3, v1T3, v2T3] <;>
    rw [mul_div_sqrt3]
  all_goals first | ring1 | linear_combination hc / 3 | linear_combination omega9 * hc / 3

/-- Vector form: amplitude functions agree as `Fin 3 × Fin 3 → ℂ`. -/
theorem t3_m2_vector_decomposition :
    t3Amp2 = fun y => (v0T3 y.1 y.2 + omega9 * v1T3 y.1 y.2
                        + omega9 ^ 2 * v2T3 y.1 y.2) / 3 :=
  funext t3_m2_decomposition

end StabRank
