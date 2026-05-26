/-
H_3 m=2 χ ≤ 3 pointwise vector identity (paper App A.2.1).

  [|H_3⟩^⊗2]_y  =  α_1·[S_1]_y + α_2·[S_2]_y + α_3·[S_3]_y

for every y ∈ F_3², with
  c  = (√3 - 1)/2,           N  = √(3 - √3),
  α_1 = (c√3/N²)(1 - c·ω),    α_2 = (c√3/N²)(1 - c·ω²),
  α_3 = 3 c² / N².

|S_1⟩ = |0, +⟩, |S_2⟩ = |+, 0⟩, and |S_3⟩ has canonical form
  k = 2, x_0 = 0, W = I_2, Q_3(y) = 2 y_0² + y_1² (mod 3).

Since y_i² ≡ [y_i ≠ 0] (mod 3), every right-hand component depends on
y only through the pair (a, b) = ([y_0 ≠ 0], [y_1 ≠ 0]) ∈ {0,1}², and
the 9 pointwise identities collapse to 4 class identities.
-/
import LeanProofs.H3Shared

namespace StabRank

open Complex Real

/-- 2-qutrit H_3⊗² amplitude built from `h3Amp1`. -/
noncomputable def h3Amp2 (y : Fin 3 × Fin 3) : ℂ :=
  h3Amp1 y.1 * h3Amp1 y.2

/-- Q_3(y) = 2 y_0² + y_1² (mod 3), tabulated. -/
def q3H2 : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, ⟨0, _⟩ => 0  -- 0 + 0
  | ⟨0, _⟩, ⟨1, _⟩ => 1  -- 0 + 1
  | ⟨0, _⟩, ⟨2, _⟩ => 1  -- 0 + 4≡1
  | ⟨1, _⟩, ⟨0, _⟩ => 2  -- 2 + 0
  | ⟨1, _⟩, ⟨1, _⟩ => 0  -- 2 + 1 = 3 ≡ 0
  | ⟨1, _⟩, ⟨2, _⟩ => 0  -- 2 + 1 ≡ 0
  | ⟨2, _⟩, ⟨0, _⟩ => 2  -- 2·4≡2 + 0
  | ⟨2, _⟩, ⟨1, _⟩ => 0  -- 2 + 1 ≡ 0
  | ⟨2, _⟩, ⟨2, _⟩ => 0  -- 2 + 1 ≡ 0
  | ⟨_+3, h⟩, _ => absurd h (by omega)
  | _, ⟨_+3, h⟩ => absurd h (by omega)

/-- |S_1⟩ = |0, +⟩: amplitude 1/√3 at y_0 = 0, else 0. -/
noncomputable def s1H2Amp (y : Fin 3 × Fin 3) : ℂ :=
  if y.1 = 0 then 1 / (Real.sqrt 3 : ℂ) else 0

/-- |S_2⟩ = |+, 0⟩: amplitude 1/√3 at y_1 = 0, else 0. -/
noncomputable def s2H2Amp (y : Fin 3 × Fin 3) : ℂ :=
  if y.2 = 0 then 1 / (Real.sqrt 3 : ℂ) else 0

/-- |S_3⟩: ω^{Q_3(y)} / 3. -/
noncomputable def s3H2Amp (y : Fin 3 × Fin 3) : ℂ :=
  omega3 ^ (q3H2 y.1 y.2) / 3

noncomputable def alphaH2_1 : ℂ :=
  cH3C * (Real.sqrt 3 : ℂ) / (NH3C * NH3C) * (1 - cH3C * omega3)
noncomputable def alphaH2_2 : ℂ :=
  cH3C * (Real.sqrt 3 : ℂ) / (NH3C * NH3C) * (1 - cH3C * omega3 ^ 2)
noncomputable def alphaH2_3 : ℂ := 3 * cH3C ^ 2 / (NH3C * NH3C)

set_option linter.flexible false in
set_option linter.style.longLine false in
set_option maxHeartbeats 800000 in -- 9 cases × 3 LC patterns over (N, √3, ω)
/-- Pointwise vector identity:
    `|H_3⟩^⊗2 = α_1·|S_1⟩ + α_2·|S_2⟩ + α_3·|S_3⟩` holds at every y ∈ F_3².

    All 9 cases close via `fin_cases × 2` + `field_simp` + one of 4
    `linear_combination` patterns (one per (a, b) class).  The
    `linear_combination` coefficient expressions were derived by symbolic
    Gröbner-basis ideal-membership against the relations
    `{h3, hN, hΩcyc}`. -/
theorem h3_m2_decomposition (y : Fin 3 × Fin 3) :
    h3Amp2 y = alphaH2_1 * s1H2Amp y + alphaH2_2 * s2H2Amp y + alphaH2_3 * s3H2Amp y := by
  have h3 := sqrt3_sq_cH
  have hN := NH3C_sq
  have hΩcyc := omega3_sq_add_omega3_add_one
  have hne3 := sqrt3_ne_zero
  have hneN := NH3C_ne_zero
  obtain ⟨y_0, y_1⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;>
    simp [h3Amp2, h3Amp1, s1H2Amp, s2H2Amp, s3H2Amp, q3H2,
          alphaH2_1, alphaH2_2, alphaH2_3, cH3C]
  all_goals
    (field_simp;
     first
     -- Class (0, 0): residual = (t+1)·(N⁴·(t+1) - 2t²(t-1)) + t²(t-1)²·(ω²+ω+1).
     | linear_combination
         (-2 * NH3C ^ 2 * ((Real.sqrt 3 : ℂ) + 1)
            - 3 * ((Real.sqrt 3 : ℂ) ^ 2 - 1)) * h3
         + (4 * (Real.sqrt 3 : ℂ) ^ 2 + 4 * (Real.sqrt 3 : ℂ)
            + ((Real.sqrt 3 : ℂ) + 1) ^ 2 * (NH3C ^ 2 - 3 + (Real.sqrt 3 : ℂ))) * hN
         + ((Real.sqrt 3 : ℂ) ^ 2 * ((Real.sqrt 3 : ℂ) - 1) ^ 2) * hΩcyc
     -- Class (0, 1): residual = N⁴·(t+1) - 2t²(t-1), no ω.
     | linear_combination
         (-(2 * NH3C ^ 2 + 3 * ((Real.sqrt 3 : ℂ) - 1))) * h3
         + (4 * (Real.sqrt 3 : ℂ) + ((Real.sqrt 3 : ℂ) + 1) * (NH3C ^ 2 - 3 + (Real.sqrt 3 : ℂ))) * hN
     -- Class (1, 1): residual = N⁴ - t²(t-1)², no ω.
     | linear_combination
         (-(Real.sqrt 3 : ℂ) ^ 2 + 2 * (Real.sqrt 3 : ℂ) - 3) * h3
         + (NH3C ^ 2 + 3 - (Real.sqrt 3 : ℂ)) * hN)

/-- Vector form of `h3_m2_decomposition`. -/
theorem h3_m2_vector_decomposition :
    h3Amp2 = fun y =>
      alphaH2_1 * s1H2Amp y + alphaH2_2 * s2H2Amp y + alphaH2_3 * s3H2Amp y :=
  funext h3_m2_decomposition

end StabRank
