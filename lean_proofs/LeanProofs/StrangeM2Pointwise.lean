/-
Strange m=2 χ = 2 pointwise vector identity (paper App A.1.1).

  [|S⟩^⊗2]_y = -(i√3/2)·ω · ([S_1]_y - [S_2]_y)   for every y ∈ F_3².

Two 2-qutrit stabilizer states S_1, S_2 with k = 2, x_0 = 0, W = I_2,
and quadratic phase polynomials
  Q_1(y) = y_0² + y_0·y_1 + y_1²    (mod 3)
  Q_2(y) = y_0² + 2·y_0·y_1 + y_1²  (mod 3)

This file establishes only the χ ≤ 2 upper bound (the explicit
decomposition).  The matching χ ≥ 2 (from support cardinality
|supp(|S⟩^⊗2)| = 4 ∉ {1, 3, 9}) is by hand in the paper.
-/
import LeanProofs.Basic

namespace StabRank

open Complex Real

/-- 1-qutrit Strange amplitude. -/
noncomputable def strangeAmp1' : Fin 3 → ℂ
  | 0 => 0
  | 1 =>  1 / (Real.sqrt 2 : ℂ)
  | 2 => -1 / (Real.sqrt 2 : ℂ)

/-- 2-qutrit Strange⊗² amplitude. -/
noncomputable def strangeAmp2 (y : Fin 3 × Fin 3) : ℂ :=
  strangeAmp1' y.1 * strangeAmp1' y.2

/-- Q_1(y_0, y_1) = y_0² + y_0·y_1 + y_1² mod 3, tabulated. -/
def q1Str2 : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, ⟨0, _⟩ => 0
  | ⟨0, _⟩, ⟨1, _⟩ => 1
  | ⟨0, _⟩, ⟨2, _⟩ => 1  -- 0 + 0 + 1 (y_1²=4≡1) = 1
  | ⟨1, _⟩, ⟨0, _⟩ => 1
  | ⟨1, _⟩, ⟨1, _⟩ => 0  -- 1+1+1 = 3 ≡ 0
  | ⟨1, _⟩, ⟨2, _⟩ => 1  -- 1+2+1 = 4 ≡ 1
  | ⟨2, _⟩, ⟨0, _⟩ => 1
  | ⟨2, _⟩, ⟨1, _⟩ => 1  -- 1+2+1 ≡ 1
  | ⟨2, _⟩, ⟨2, _⟩ => 0  -- 1+4+1 ≡ 0
  | ⟨_+3, h⟩, _    => absurd h (by omega)
  | _, ⟨_+3, h⟩    => absurd h (by omega)

/-- Q_2(y_0, y_1) = y_0² + 2·y_0·y_1 + y_1² mod 3, tabulated. -/
def q2Str2 : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, ⟨0, _⟩ => 0
  | ⟨0, _⟩, ⟨1, _⟩ => 1
  | ⟨0, _⟩, ⟨2, _⟩ => 1
  | ⟨1, _⟩, ⟨0, _⟩ => 1
  | ⟨1, _⟩, ⟨1, _⟩ => 1  -- 1+2+1 ≡ 1
  | ⟨1, _⟩, ⟨2, _⟩ => 0  -- 1+4+1 = 6 ≡ 0
  | ⟨2, _⟩, ⟨0, _⟩ => 1
  | ⟨2, _⟩, ⟨1, _⟩ => 0  -- 1+4+1 ≡ 0
  | ⟨2, _⟩, ⟨2, _⟩ => 1  -- 1+8+1 = 10 ≡ 1
  | ⟨_+3, h⟩, _    => absurd h (by omega)
  | _, ⟨_+3, h⟩    => absurd h (by omega)

noncomputable def s1Str2 (y : Fin 3 × Fin 3) : ℂ := omega3 ^ q1Str2 y.1 y.2 / 3
noncomputable def s2Str2 (y : Fin 3 × Fin 3) : ℂ := omega3 ^ q2Str2 y.1 y.2 / 3

/-- The coefficient: -(i√3/2)·ω. -/
noncomputable def alphaStr2 : ℂ := -(Complex.I * (Real.sqrt 3 : ℂ) / 2) * omega3

private theorem sqrt2_sq_S2 : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
  exact_mod_cast Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)

private theorem sqrt3_sq_S2 : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 3 := by
  exact_mod_cast Real.mul_self_sqrt (by norm_num : (3:ℝ) ≥ 0)

private theorem sqrt2_ne_zero_S2 : (Real.sqrt 2 : ℂ) ≠ 0 := by
  intro hz
  have hp : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 0 := by rw [hz]; ring
  rw [sqrt2_sq_S2] at hp; norm_num at hp

-- The flexible-tactic and long-line linters fire on the `fin_cases ; simp`
-- + `linear_combination` shape this proof uses, same as the m=3/m=4 lifts.
set_option linter.flexible false in
set_option linter.style.longLine false in
/-- Pointwise vector identity:
    `|S⟩^⊗2 = -(i√3/2)·ω·(|S_1⟩ - |S_2⟩)` at every y ∈ F_3².

    The 5 cases with some `y_i = 0` collapse to `0 = 0` (closed by `simp`).
    Each of the 4 non-trivial cases (y_0, y_1 ∈ {1,2}) clears via
    `field_simp` to a polynomial residual proportional to
      `6 + s²·i·t·(ω - ω²) = 0`   (where s = √2, t = √3),
    closed by one `linear_combination` (or its negation, by case sign). -/
theorem strange_m2_decomposition (y : Fin 3 × Fin 3) :
    strangeAmp2 y = alphaStr2 * (s1Str2 y - s2Str2 y) := by
  have h2 := sqrt2_sq_S2
  have h3 := sqrt3_sq_S2
  have hI : Complex.I * Complex.I = -1 := Complex.I_mul_I
  have hΩcyc := omega3_sq_add_omega3_add_one
  have h2ω : 2 * omega3 + 1 = Complex.I * (Real.sqrt 3 : ℂ) := by
    linear_combination omega3_diff_eq_I_sqrt3 + omega3_sq_eq
  have hne2 := sqrt2_ne_zero_S2
  obtain ⟨y_0, y_1⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;>
    simp [strangeAmp2, strangeAmp1', s1Str2, s2Str2, q1Str2, q2Str2, alphaStr2]
  all_goals
    (field_simp;
     first
     | linear_combination
         ((Real.sqrt 2 : ℂ)^2 * (Real.sqrt 3 : ℂ)^2) * hI
         + (-(Real.sqrt 3 : ℂ)^2) * h2
         + (-2 : ℂ) * h3
         + (-(Real.sqrt 2 : ℂ)^2 * (Real.sqrt 3 : ℂ) * Complex.I) * hΩcyc
         + ((Real.sqrt 2 : ℂ)^2 * (Real.sqrt 3 : ℂ) * Complex.I) * h2ω
     | linear_combination
         (-((Real.sqrt 2 : ℂ)^2 * (Real.sqrt 3 : ℂ)^2)) * hI
         + ((Real.sqrt 3 : ℂ)^2) * h2
         + (2 : ℂ) * h3
         + ((Real.sqrt 2 : ℂ)^2 * (Real.sqrt 3 : ℂ) * Complex.I) * hΩcyc
         + (-((Real.sqrt 2 : ℂ)^2 * (Real.sqrt 3 : ℂ) * Complex.I)) * h2ω)

/-- Vector form of `strange_m2_decomposition`: amplitude functions
    are equal as `Fin 3 × Fin 3 → ℂ`. -/
theorem strange_m2_vector_decomposition :
    strangeAmp2 = fun y => alphaStr2 * (s1Str2 y - s2Str2 y) :=
  funext strange_m2_decomposition

end StabRank
