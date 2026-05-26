/-
Norrell m=2 χ ≤ 3 pointwise vector identity (paper App A.3.1).

  [|N⟩^⊗2]_y  =  α·[S_1]_y + [S_2]_y + ᾱ·[S_3]_y

for every y ∈ F_3², with α = -ω/2 and ᾱ = -ω²/2.

  |S_1⟩: k=2, x_0=0, W=I_2, Q_1(y) = y_0 + y_1 + y_0·y_1  (mod 3)
  |S_2⟩ = |2, 2⟩  (the basis state)
  |S_3⟩: k=2, x_0=0, W=I_2, Q_3(y) = 2·Q_1(y) (= conjugate of S_1)

The 9 cases collapse to four (Q_1, special) classes:
  Q_1 = 0:                  y ∈ {(0,0), (1,1)}
  Q_1 = 1:                  y ∈ {(0,1), (1,0)}
  Q_1 = 2 (no S_2):         y ∈ {(0,2), (1,2), (2,0), (2,1)}
  Q_1 = 2 at y = (2,2):     {(2,2)}      (S_2 contributes)
-/
import LeanProofs.NorrellShared

namespace StabRank

open Complex Real

/-- 2-qutrit Norrell⊗² amplitude built from `norrellAmp1`. -/
noncomputable def norrellAmp2 (y : Fin 3 × Fin 3) : ℂ :=
  norrellAmp1 y.1 * norrellAmp1 y.2

/-- Q_1(y_0, y_1) = y_0 + y_1 + y_0·y_1  (mod 3), tabulated.

    Equivalently Q_1(y) = (1+y_0)(1+y_1) − 1 mod 3. -/
def q1Nor2 : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, ⟨0, _⟩ => 0
  | ⟨0, _⟩, ⟨1, _⟩ => 1
  | ⟨0, _⟩, ⟨2, _⟩ => 2
  | ⟨1, _⟩, ⟨0, _⟩ => 1
  | ⟨1, _⟩, ⟨1, _⟩ => 0  -- 1 + 1 + 1 = 3 ≡ 0
  | ⟨1, _⟩, ⟨2, _⟩ => 2  -- 1 + 2 + 2 = 5 ≡ 2
  | ⟨2, _⟩, ⟨0, _⟩ => 2
  | ⟨2, _⟩, ⟨1, _⟩ => 2  -- 2 + 1 + 2 = 5 ≡ 2
  | ⟨2, _⟩, ⟨2, _⟩ => 2  -- 2 + 2 + 4 = 8 ≡ 2
  | ⟨_+3, h⟩, _ => absurd h (by omega)
  | _, ⟨_+3, h⟩ => absurd h (by omega)

/-- Q_3 = 2·Q_1 mod 3, tabulated. -/
def q3Nor2 : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, ⟨0, _⟩ => 0
  | ⟨0, _⟩, ⟨1, _⟩ => 2
  | ⟨0, _⟩, ⟨2, _⟩ => 1
  | ⟨1, _⟩, ⟨0, _⟩ => 2
  | ⟨1, _⟩, ⟨1, _⟩ => 0
  | ⟨1, _⟩, ⟨2, _⟩ => 1
  | ⟨2, _⟩, ⟨0, _⟩ => 1
  | ⟨2, _⟩, ⟨1, _⟩ => 1
  | ⟨2, _⟩, ⟨2, _⟩ => 1
  | ⟨_+3, h⟩, _ => absurd h (by omega)
  | _, ⟨_+3, h⟩ => absurd h (by omega)

/-- |S_1⟩_y = ω^{Q_1(y)} / 3 on full F_3². -/
noncomputable def sN1AmpM2 (y : Fin 3 × Fin 3) : ℂ := omega3 ^ q1Nor2 y.1 y.2 / 3

/-- |S_2⟩ = |2, 2⟩: amplitude 1 at (2,2), else 0. -/
noncomputable def sN2AmpM2 (y : Fin 3 × Fin 3) : ℂ :=
  if y.1 = 2 ∧ y.2 = 2 then 1 else 0

/-- |S_3⟩_y = ω^{2·Q_1(y)} / 3 (the complex conjugate of |S_1⟩). -/
noncomputable def sN3AmpM2 (y : Fin 3 × Fin 3) : ℂ := omega3 ^ q3Nor2 y.1 y.2 / 3

/-- α = -ω / 2. -/
noncomputable def alphaNor2 : ℂ := -omega3 / 2

/-- ᾱ = -ω² / 2 (complex conjugate of α). -/
noncomputable def alphaNor2bar : ℂ := -omega3 ^ 2 / 2

set_option linter.flexible false in
set_option linter.style.longLine false in
set_option maxHeartbeats 800000 in -- 9 cases × 4 LC patterns over (√2, √3, ω)
/-- Pointwise vector identity:
    `|N⟩^⊗2 = α·|S_1⟩ + |S_2⟩ + ᾱ·|S_3⟩` holds at every y ∈ F_3².

    The 9 cases close via `fin_cases × 2` + `field_simp` + one of 4
    `linear_combination` patterns indexed by (Q_1, [y = (2,2)]).
    Coefficients derived analytically (the cleared residual factors
    cleanly into (√2² − 2), (√3² − 3), (ω + ω² + 1), and (ω³ − 1)). -/
theorem norrell_m2_decomposition (y : Fin 3 × Fin 3) :
    norrellAmp2 y =
      alphaNor2 * sN1AmpM2 y + sN2AmpM2 y + alphaNor2bar * sN3AmpM2 y := by
  have h2 := sqrt2_sq_cN
  have h3 := sqrt3_sq_cN
  have hΩ3 := omega3_pow_three
  have hΩsum := omega3_add_omega3_sq
  have hne2 : (Real.sqrt 2 : ℂ) ≠ 0 := by
    intro hz
    have hp : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 0 := by rw [hz]; ring
    rw [h2] at hp; norm_num at hp
  have hne3 : (Real.sqrt 3 : ℂ) ≠ 0 := by
    intro hz
    have hp : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 0 := by rw [hz]; ring
    rw [h3] at hp; norm_num at hp
  obtain ⟨y_0, y_1⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;>
    simp [norrellAmp2, norrellAmp1, sN1AmpM2, sN2AmpM2, sN3AmpM2,
          q1Nor2, q3Nor2, alphaNor2, alphaNor2bar]
  all_goals
    (field_simp;
     first
     -- Q_1 = 0: goal.lhs - goal.rhs = 6·s²t² + 36·(ω + ω²).
     | linear_combination (6 * (Real.sqrt 3 : ℂ) ^ 2) * h2 + (12 : ℂ) * h3
         + (36 : ℂ) * hΩsum
     -- Q_1 = 1: as above, with extra 36·ω·(ω³ − 1) from the ω⁴ term.
     | linear_combination (6 * (Real.sqrt 3 : ℂ) ^ 2) * h2 + (12 : ℂ) * h3
         + (36 : ℂ) * hΩsum + (36 * omega3) * hΩ3
     -- Q_1 = 2, y ≠ (2,2): goal.lhs - goal.rhs = -2·s²t² + 12·ω³.
     | linear_combination (-2 * (Real.sqrt 3 : ℂ) ^ 2) * h2 + (-4 : ℂ) * h3
         + (12 : ℂ) * hΩ3
     -- Q_1 = 2 at y = (2,2): goal.lhs - goal.rhs = 2·s²t² - 18 + 6·ω³.
     | linear_combination (2 * (Real.sqrt 3 : ℂ) ^ 2) * h2 + (4 : ℂ) * h3
         + (6 : ℂ) * hΩ3)

/-- Vector form of `norrell_m2_decomposition`. -/
theorem norrell_m2_vector_decomposition :
    norrellAmp2 = fun y =>
      alphaNor2 * sN1AmpM2 y + sN2AmpM2 y + alphaNor2bar * sN3AmpM2 y :=
  funext norrell_m2_decomposition

end StabRank
