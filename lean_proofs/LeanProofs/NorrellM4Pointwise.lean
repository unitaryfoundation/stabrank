/-
Norrell m=4 χ ≤ 7 pointwise vector identity (paper App A.3.3).

  [|N⟩^⊗4]_y = α_0·[S_0]_y + α_1·[S_1]_y + α_2·[S_2]_y + α_3·[S_3]_y
             + α_4·[S_4]_y + α_5·[S_5]_y + α_6·[S_6]_y

for every y ∈ F_3⁴, with seven 4-qutrit stabilizer states and coefficients

  α_0 = α_4 = σω,  α_1 = α_2 = α_3 = α_5 = σξ,  α_6 = ω²/4

where σ = √3/4 and ξ = e^{iπ/6} = (√3 + i)/2 is a primitive 12th root of unity.

The Q_j polynomials reduce on each support to indicator-only forms
(using `y + 2y² ≡ [y=2]` and `2y + y² ≡ 2[y=2]` mod 3) — we use the
reduced forms directly here.  Setting a_i := [y_i = 2] ∈ {0,1}:

  supp(S_0) = {y_1 = 2}:           Q_0 = 2a_0 + a_2 + 2a_3
  supp(S_1) = F_3⁴:                Q_1 = 2a_0 + 2a_1 + a_2 + a_3
  supp(S_2) = {y_0 = y_3 = 2}:     Q_2 = a_1 + a_2
  supp(S_3) = {y_2 = y_3 = 2}:     Q_3 = 2a_0 + a_1
  supp(S_4) = {y_0 = 2}:           Q_4 = 2a_1 + 2a_2 + a_3
  supp(S_5) = {y_1 = y_2 = 2}:     Q_5 = a_0 + 2a_3
  supp(S_6) = F_3⁴:                Q_6 = 0  (so S_6 = |+⟩^⊗4)

Identity collapses to 5 n₂-classes (n₂ = a_0+a_1+a_2+a_3) with the
value [N⟩^⊗4]_y = (-2)^{n₂}/36.
-/
import LeanProofs.NorrellShared

namespace StabRank

open Complex Real

/-- 4-qutrit Norrell⊗⁴ amplitude. -/
noncomputable def norrellAmp4 (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  norrellAmp1 y.1 * norrellAmp1 y.2.1 * norrellAmp1 y.2.2.1 * norrellAmp1 y.2.2.2

/-- ξ = e^{iπ/6} = (√3 + i)/2 = cos(π/6) + i·sin(π/6).
    Satisfies ξ² - (√3/1)·ξ + 1 = 0 (or 2ξ = √3 + i directly). -/
noncomputable def xiN : ℂ := ((Real.sqrt 3 : ℂ) + Complex.I) / 2

/-- σ = √3/4. -/
noncomputable def sigmaN : ℂ := (Real.sqrt 3 : ℂ) / 4

/-- Q-polynomial reductions (indicator form, mod 3). -/
def qN0 (y_0 y_2 y_3 : Fin 3) : ℕ :=
  (2 * isTwo y_0 + isTwo y_2 + 2 * isTwo y_3) % 3

def qN1 (y_0 y_1 y_2 y_3 : Fin 3) : ℕ :=
  (2 * isTwo y_0 + 2 * isTwo y_1 + isTwo y_2 + isTwo y_3) % 3

def qN2 (y_1 y_2 : Fin 3) : ℕ :=
  (isTwo y_1 + isTwo y_2) % 3

def qN3 (y_0 y_1 : Fin 3) : ℕ :=
  (2 * isTwo y_0 + isTwo y_1) % 3

def qN4 (y_1 y_2 y_3 : Fin 3) : ℕ :=
  (2 * isTwo y_1 + 2 * isTwo y_2 + isTwo y_3) % 3

def qN5 (y_0 y_3 : Fin 3) : ℕ :=
  (isTwo y_0 + 2 * isTwo y_3) % 3

/-- Stabilizer state amplitudes.  Off-support = 0; on-support = ω^Q / √(3^k). -/
noncomputable def sN0' (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.1 = 2 then omega3 ^ qN0 y.1 y.2.2.1 y.2.2.2 / (3 * (Real.sqrt 3 : ℂ)) else 0

noncomputable def sN1' (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  omega3 ^ qN1 y.1 y.2.1 y.2.2.1 y.2.2.2 / 9

noncomputable def sN2' (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.1 = 2 ∧ y.2.2.2 = 2 then omega3 ^ qN2 y.2.1 y.2.2.1 / 3 else 0

noncomputable def sN3' (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.2.1 = 2 ∧ y.2.2.2 = 2 then omega3 ^ qN3 y.1 y.2.1 / 3 else 0

noncomputable def sN4' (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.1 = 2 then omega3 ^ qN4 y.2.1 y.2.2.1 y.2.2.2 / (3 * (Real.sqrt 3 : ℂ)) else 0

noncomputable def sN5' (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.1 = 2 ∧ y.2.2.1 = 2 then omega3 ^ qN5 y.1 y.2.2.2 / 3 else 0

noncomputable def sN6' (_y : Fin 3 × Fin 3 × Fin 3 × Fin 3) : ℂ :=
  (1 : ℂ) / 9

/-- α coefficients. -/
noncomputable def alphaN0 : ℂ := sigmaN * omega3              -- σω
noncomputable def alphaN1 : ℂ := sigmaN * xiN                 -- σξ
noncomputable def alphaN4 : ℂ := sigmaN * omega3              -- σω
noncomputable def alphaN6 : ℂ := omega3 ^ 2 / 4               -- ω²/4

set_option linter.flexible false in
set_option linter.style.longLine false in
set_option maxHeartbeats 12800000 in -- 81 cases × 9 LC patterns over (√2, √3, ω, I)
/-- Pointwise vector identity:
    `|N⟩^⊗4 = α_0·|S_0⟩ + α_1·|S_1⟩ + … + α_6·|S_6⟩` holds at every y ∈ F_3⁴.
    All 81 cases close via fin_cases × 4 + field_simp + one of 9 `linear_combination`
    patterns dispatched by `first`.  Coefficients derived by Gröbner-basis
    ideal-membership against {h2, h3, hΩcyc, hΩ3, hI, h2ω}. -/
theorem norrell_m4_decomposition (y : Fin 3 × Fin 3 × Fin 3 × Fin 3) :
    norrellAmp4 y = alphaN0 * sN0' y + alphaN1 * sN1' y + alphaN1 * sN2' y
                + alphaN1 * sN3' y + alphaN4 * sN4' y + alphaN1 * sN5' y
                + alphaN6 * sN6' y := by
  have h2 := sqrt2_sq_cN
  have h3 := sqrt3_sq_cN
  have hΩ3 := omega3_pow_three
  have hΩcyc := omega3_sq_add_omega3_add_one
  have hI : Complex.I * Complex.I = -1 := Complex.I_mul_I
  -- 2ω + 1 = i√3 (proved via direct Re/Im computation).
  have h2ω : 2 * omega3 + 1 = Complex.I * (Real.sqrt 3 : ℂ) := by
    apply Complex.ext
    · simp [Complex.add_re, Complex.mul_re, Complex.one_re, Complex.I_re,
            Complex.I_im, Complex.ofReal_re, Complex.ofReal_im, omega3_re, omega3_im]
    · simp [Complex.add_im, Complex.mul_im, Complex.one_im, Complex.I_re,
            Complex.I_im, Complex.ofReal_re, Complex.ofReal_im, omega3_re, omega3_im]
      ring
  have hne3 : (Real.sqrt 3 : ℂ) ≠ 0 := by
    intro hz
    have hp : (Real.sqrt 3 : ℂ) * (Real.sqrt 3 : ℂ) = 0 := by rw [hz]; ring
    rw [h3] at hp; norm_num at hp
  obtain ⟨y_0, y_1, y_2, y_3⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;> fin_cases y_2 <;> fin_cases y_3 <;>
    simp [norrellAmp4, norrellAmp1, sN0', sN1', sN2', sN3', sN4', sN5', sN6',
          isTwo, qN0, qN1, qN2, qN3, qN4, qN5, alphaN0, alphaN1, alphaN4,
          alphaN6, sigmaN, xiN]
  all_goals
    (field_simp;
     first
     | linear_combination (-72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 - 144*(Real.sqrt 3 : ℂ)^4) * h2 + (-162*Complex.I^2 - 288*(Real.sqrt 3 : ℂ)^2 - 162*omega3^2 - 486*omega3 - 864) * h3 + (-162*Complex.I*(Real.sqrt 3 : ℂ) - 1782) * hΩcyc + (-486) * hI + (-162*Complex.I*(Real.sqrt 3 : ℂ) - 324) * h2ω
     | linear_combination (-72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 - 144*(Real.sqrt 3 : ℂ)^4) * h2 + (-81*Complex.I^2 - 288*(Real.sqrt 3 : ℂ)^2 - 486*omega3^2 - 648*omega3 - 864) * h3 + (-486*Complex.I*(Real.sqrt 3 : ℂ) - 1782) * hΩcyc + (-243) * hI + (-81*Complex.I*(Real.sqrt 3 : ℂ) - 567) * h2ω
     | linear_combination (-72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 - 144*(Real.sqrt 3 : ℂ)^4) * h2 + (-324*Complex.I^2 - 288*(Real.sqrt 3 : ℂ)^2 - 648*omega3 - 864) * h3 + (-1296) * hΩcyc + (-972) * hI + (-324*Complex.I*(Real.sqrt 3 : ℂ) - 324) * h2ω
     | linear_combination (-72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 - 144*(Real.sqrt 3 : ℂ)^4) * h2 + (324*Complex.I^2 - 288*(Real.sqrt 3 : ℂ)^2 - 648*omega3^2 - 864) * h3 + (-648*Complex.I*(Real.sqrt 3 : ℂ) - 3240) * hΩcyc + (972) * hI + (324*Complex.I*(Real.sqrt 3 : ℂ) - 324) * h2ω
     | linear_combination (72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 + 144*(Real.sqrt 3 : ℂ)^4) * h2 + (288*(Real.sqrt 3 : ℂ)^2 - 432) * h3 + (-2592) * hΩcyc + (1296) * h2ω
     | linear_combination (72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 + 144*(Real.sqrt 3 : ℂ)^4) * h2 + (288*(Real.sqrt 3 : ℂ)^2 + 540) * h3 + (-1944*omega3 + 1296) * hΩcyc + (324) * h2ω
     | linear_combination (72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 + 144*(Real.sqrt 3 : ℂ)^4) * h2 + (162*Complex.I^2 + 288*(Real.sqrt 3 : ℂ)^2 - 324*omega3^2 - 108) * h3 + (-324*Complex.I*(Real.sqrt 3 : ℂ) - 1620) * hΩcyc + (486) * hI + (162*Complex.I*(Real.sqrt 3 : ℂ) + 810) * h2ω
     | linear_combination (72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 + 144*(Real.sqrt 3 : ℂ)^4) * h2 + (-162*Complex.I^2 + 288*(Real.sqrt 3 : ℂ)^2 - 324*omega3 + 864) * h3 + (-3888*omega3 + 3240) * hΩcyc + (-486) * hI + (-162*Complex.I*(Real.sqrt 3 : ℂ) - 162) * h2ω
     | linear_combination (72*(Real.sqrt 2 : ℂ)^2*(Real.sqrt 3 : ℂ)^4 + 144*(Real.sqrt 3 : ℂ)^4) * h2 + (243*Complex.I^2/2 + 288*(Real.sqrt 3 : ℂ)^2 - 243*omega3^2 + 297) * h3 + (-243*Complex.I*(Real.sqrt 3 : ℂ) - 972*omega3 + 81) * hΩcyc + (729/2) * hI + (243*Complex.I*(Real.sqrt 3 : ℂ)/2 + 891/2) * h2ω)

/-- Vector form of `norrell_m4_decomposition`. -/
theorem norrell_m4_vector_decomposition :
    norrellAmp4 = fun y =>
      alphaN0 * sN0' y + alphaN1 * sN1' y + alphaN1 * sN2' y
        + alphaN1 * sN3' y + alphaN4 * sN4' y + alphaN1 * sN5' y
        + alphaN6 * sN6' y :=
  funext norrell_m4_decomposition

end StabRank
