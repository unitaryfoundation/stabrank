/-
H_3 m=3 χ ≤ 4 pointwise vector identity (paper App A.2.2 full lift).

  [|H_3⟩^⊗3]_y  =  α_1·[S_1]_y + α_2·[S_2]_y + α_3·[S_3]_y + α_4·[S_4]_y

for every y ∈ F_3³, with
  α_1 = (3c/4N)(1 + i),  α_2 = (3c/4N)(1 - i),  α_3 = α_4 = 3/(4N),
  c = (√3 - 1)/2,  N = √(3 - √3).

S_1, S_2 have full F_3³ support with phase polynomials
  Q_1(y) = 2y_0² + 2y_1² + y_2², Q_2(y) = y_0² + y_1² + 2y_2² (mod 3).
S_3 = |0,0,+⟩, S_4 = |+,+,0⟩.

The identity collapses to 6 (n,z)-class identities (n = #{y_0,y_1 ≠ 0},
z = [y_2 ≠ 0]), all reducing to (3+√3)·N² = (3+√3)(3-√3) = 6.
-/
import LeanProofs.H3Shared

namespace StabRank

open Complex Real

/-- Indicator z(y) := [y ≠ 0]. -/
def zNz : Fin 3 → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1

def q1H3 (y_0 y_1 y_2 : Fin 3) : ℕ :=
  (2 * zNz y_0 + 2 * zNz y_1 + zNz y_2) % 3

def q2H3 (y_0 y_1 y_2 : Fin 3) : ℕ :=
  (zNz y_0 + zNz y_1 + 2 * zNz y_2) % 3

noncomputable def h3Amp3 (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  h3Amp1 y.1 * h3Amp1 y.2.1 * h3Amp1 y.2.2

noncomputable def sH1Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  omega3 ^ (q1H3 y.1 y.2.1 y.2.2) / (3 * (Real.sqrt 3 : ℂ))

noncomputable def sH2Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  omega3 ^ (q2H3 y.1 y.2.1 y.2.2) / (3 * (Real.sqrt 3 : ℂ))

noncomputable def sH3Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.1 = 0 ∧ y.2.1 = 0 then 1 / (Real.sqrt 3 : ℂ) else 0

noncomputable def sH4Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.2 = 0 then (1 : ℂ) / 3 else 0

noncomputable def alphaH1 : ℂ := 3 * cH3C / (4 * NH3C) * (1 + Complex.I)
noncomputable def alphaH2 : ℂ := 3 * cH3C / (4 * NH3C) * (1 - Complex.I)
noncomputable def alphaH34 : ℂ := 3 / (4 * NH3C)

-- Same justification as in NorrellM3Pointwise.lean / StrangeM3Pointwise.lean.
set_option linter.flexible false in
set_option linter.style.longLine false in -- LC coefficients are long polynomial expressions
set_option maxHeartbeats 3200000 in -- 27 cases × 6 LC patterns over (N, √3, ω, I)
/-- Pointwise vector identity:
    `|H_3⟩^⊗3 = α_1·|S_1⟩ + α_2·|S_2⟩ + α_3·|S_3⟩ + α_4·|S_4⟩` holds at every y ∈ F_3³.
    All 27 cases close via fin_cases × 3 + field_simp + one of 6
    `linear_combination` patterns (one per (n,z)-class).  The coefficient
    expressions were derived by symbolic Gröbner-basis ideal-membership
    against the relations {h3, hN, hΩcyc, hI, h2ω}. -/
theorem h3_m3_decomposition (y : Fin 3 × Fin 3 × Fin 3) :
    h3Amp3 y = alphaH1 * sH1Amp y + alphaH2 * sH2Amp y
             + alphaH34 * sH3Amp y + alphaH34 * sH4Amp y := by
  have h3 := sqrt3_sq_cH
  have hN := NH3C_sq
  have hΩcyc := omega3_sq_add_omega3_add_one
  have hI : Complex.I * Complex.I = -1 := Complex.I_mul_I
  -- `2ω + 1 = i√3` follows from ω - ω² = i√3 and ω² = -1 - ω.
  have h2ω : 2 * omega3 + 1 = Complex.I * (Real.sqrt 3 : ℂ) := by
    linear_combination omega3_diff_eq_I_sqrt3 + omega3_sq_eq
  have hne3 := sqrt3_ne_zero
  have hneN := NH3C_ne_zero
  obtain ⟨y_0, y_1, y_2⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;> fin_cases y_2 <;>
    simp [h3Amp3, h3Amp1, sH1Amp, sH2Amp, sH3Amp, sH4Amp, zNz, q1H3, q2H3,
          alphaH1, alphaH2, alphaH34, cH3C]
  all_goals
    (field_simp;
     first
     -- Form 1: (n,z) = (0,0).  1 case.
     | linear_combination
         (4*NH3C^4*(Real.sqrt 3 : ℂ) + 12*NH3C^4 - 24*NH3C^2 - 16*(Real.sqrt 3 : ℂ) - 48) * h3
         + (24*NH3C^2*(Real.sqrt 3 : ℂ) + 40*NH3C^2 + 32*(Real.sqrt 3 : ℂ) + 48) * hN
     -- Form 2: (n,z) = (1,1).  8 cases.
     | linear_combination
         (-4*NH3C^2 - 8*(Real.sqrt 3 : ℂ)) * h3
         + (4*NH3C^2*(Real.sqrt 3 : ℂ) + 4*NH3C^2 + 8*(Real.sqrt 3 : ℂ)) * hN
     -- Form 3.
     | linear_combination
         (-12*Complex.I^2 + 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3^2 - 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3 - 4*Complex.I*omega3^2 + 4*Complex.I*omega3 - 4*NH3C^2 - 4*(Real.sqrt 3 : ℂ)*omega3^2 - 4*(Real.sqrt 3 : ℂ)*omega3 - 8*(Real.sqrt 3 : ℂ) + 4*omega3^2 + 4*omega3 - 8) * h3
         + (4*NH3C^2*(Real.sqrt 3 : ℂ) + 4*NH3C^2 + 8*(Real.sqrt 3 : ℂ)) * hN
         + (12*Complex.I*(Real.sqrt 3 : ℂ) - 12*Complex.I - 12*(Real.sqrt 3 : ℂ) + 12) * hΩcyc
         + (12*(Real.sqrt 3 : ℂ) - 36) * hI
         + (-12*Complex.I*(Real.sqrt 3 : ℂ) + 12*Complex.I) * h2ω
     -- Form 4.
     | linear_combination
         (12*Complex.I^2 - 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3^2 + 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3 + 4*Complex.I*omega3^2 - 4*Complex.I*omega3 + 4*NH3C^4 - 8*NH3C^2 - 4*(Real.sqrt 3 : ℂ)*omega3^2 - 4*(Real.sqrt 3 : ℂ)*omega3 - 8*(Real.sqrt 3 : ℂ) + 4*omega3^2 + 4*omega3 - 8) * h3
         + (8*NH3C^2*(Real.sqrt 3 : ℂ) + 16*NH3C^2 + 8*(Real.sqrt 3 : ℂ) + 24) * hN
         + (-12*Complex.I*(Real.sqrt 3 : ℂ) + 12*Complex.I - 12*(Real.sqrt 3 : ℂ) + 12) * hΩcyc
         + (-12*(Real.sqrt 3 : ℂ) + 36) * hI
         + (12*Complex.I*(Real.sqrt 3 : ℂ) - 12*Complex.I) * h2ω
     -- Form 5.
     | linear_combination
         (-12*Complex.I^2 + 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3^2 - 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3 - 4*Complex.I*omega3^2 + 4*Complex.I*omega3 + 4*NH3C^4 - 8*NH3C^2 - 4*(Real.sqrt 3 : ℂ)*omega3^2 - 4*(Real.sqrt 3 : ℂ)*omega3 + 4*omega3^2 + 4*omega3 - 32) * h3
         + (8*NH3C^2*(Real.sqrt 3 : ℂ) + 16*NH3C^2 + 8*(Real.sqrt 3 : ℂ) + 24) * hN
         + (12*Complex.I*(Real.sqrt 3 : ℂ) - 12*Complex.I - 12*(Real.sqrt 3 : ℂ) + 12) * hΩcyc
         + (12*(Real.sqrt 3 : ℂ) - 36) * hI
         + (-12*Complex.I*(Real.sqrt 3 : ℂ) + 12*Complex.I) * h2ω
     -- Form 6.
     | linear_combination
         (12*Complex.I^2 - 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3^2 + 4*Complex.I*(Real.sqrt 3 : ℂ)*omega3 + 4*Complex.I*omega3^2 - 4*Complex.I*omega3 - 4*(Real.sqrt 3 : ℂ)*omega3^2 - 4*(Real.sqrt 3 : ℂ)*omega3 + 4*omega3^2 + 4*omega3 + 4) * h3
         + (4*NH3C^2 - 4*(Real.sqrt 3 : ℂ) + 12) * hN
         + (-12*Complex.I*(Real.sqrt 3 : ℂ) + 12*Complex.I - 12*(Real.sqrt 3 : ℂ) + 12) * hΩcyc
         + (-12*(Real.sqrt 3 : ℂ) + 36) * hI
         + (12*Complex.I*(Real.sqrt 3 : ℂ) - 12*Complex.I) * h2ω)

/-- Vector form of `h3_m3_decomposition`. -/
theorem h3_m3_vector_decomposition :
    h3Amp3 = fun y =>
      alphaH1 * sH1Amp y + alphaH2 * sH2Amp y
        + alphaH34 * sH3Amp y + alphaH34 * sH4Amp y :=
  funext h3_m3_decomposition

end StabRank
