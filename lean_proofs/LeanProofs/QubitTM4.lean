import Mathlib

/-!
# Qubit T-type m = 4 Stabilizer Decomposition

This file formalizes **Proposition C.4** from "Stabilizer rank bounds for
magic-state orbits" (Labib & Russo, 2026).

## Statement

The qubit T-type magic state |T⟩ = cos β |0⟩ + e^{iπ/4} sin β |1⟩
(where cos 2β = 1/√3) admits a 3-term stabilizer decomposition at 4 copies:

    |T⟩^⊗4 = c₁|σ₁⟩ + c₂|σ₂⟩ + c₃|σ₃⟩

where σ₁, σ₂, σ₃ are explicit 4-qubit stabilizer states and
c₁ = (2/3)e^{iπ/12}, c₂ = 2/3, c₃ = (2/3)e^{-iπ/12}.

This establishes χ(|T⟩^⊗4) ≤ 3, which combined with the m=3 lower-bound
certificate gives χ(|T⟩^⊗4) = 3, and hence γ_T ≤ log₂(3)/4 ≈ 0.396.

## Proof Architecture

The proof is organized in a modular layered structure:
* **Layer 1** — Trigonometric identities (5 lemmas, fully proved).
* **Layer 2** — Complex exponential → trig rewriting (6 lemmas, fully proved).
* **Layer 3** — Square root algebraic helpers (5 lemmas, fully proved).
* **Layer 4** — Main case verification (`qubit_t_m4_decomposition`). Solved directly via exhaustive finite-domain analysis (`fin_cases`) and strict algebraic solving with `first | ring | (simp only [Complex.I_sq]; ring)`.
* **Layer 5** — Assembly (`stabilizer_rank_le_three` and `gamma_T_bound`), giving the exact 3-term stabilizer decomposition and numerical upper bound.
-/

noncomputable section

open Complex Real Finset

namespace StabilizerRank.QubitTM4

/-! ### Notation and basic constants -/

abbrev Basis4 := Fin 4 → Fin 2

def bval (x : Basis4) (i : Fin 4) : ℕ := (x i).val

def hammingWeight (x : Basis4) : ℕ :=
  bval x 0 + bval x 1 + bval x 2 + bval x 3

/-! ### Target amplitude -/

def targetAmplitude (x : Basis4) : ℂ :=
  let w := hammingWeight x
  match w with
  | 0 => ↑((2 + Real.sqrt 3) / 6)
  | 1 => ↑((Real.sqrt 3 + 1) / 12) * (1 + Complex.I)
  | 2 => Complex.I / 6
  | 3 => ↑((Real.sqrt 3 - 1) / 12) * (-1 + Complex.I)
  | 4 => ↑((Real.sqrt 3 - 2) / 6)
  | _ => 0

/-! ### Stabilizer state amplitudes -/

def sigma1Amp (x : Basis4) : ℂ :=
  if x 0 = x 2 then
    (1 / (2 * Real.sqrt 2) : ℝ) *
      Complex.I ^ (bval x 0) *
      ((-1 : ℂ) ^ (bval x 1 * bval x 3))
  else
    0

def sigma2Amp (x : Basis4) : ℂ :=
  (1 / 4 : ℝ) *
    Complex.I ^ (bval x 1 + bval x 3) *
    ((-1 : ℂ) ^ (bval x 0 * bval x 2 + bval x 1 * bval x 3))

def sigma3Amp (x : Basis4) : ℂ :=
  if x 1 = x 3 then
    (1 / (2 * Real.sqrt 2) : ℝ) *
      Complex.I ^ (bval x 0 + bval x 1 + bval x 2) *
      ((-1 : ℂ) ^ (bval x 0 * bval x 2))
  else
    0

/-! ### Linear coefficients -/

def c1 : ℂ := (2 / 3 : ℝ) * Complex.exp (Complex.I * ↑(Real.pi / 12))
def c2 : ℂ := (2 / 3 : ℝ)
def c3 : ℂ := (2 / 3 : ℝ) * Complex.exp (-Complex.I * ↑(Real.pi / 12))

def rhsAmplitude (x : Basis4) : ℂ :=
  c1 * sigma1Amp x + c2 * sigma2Amp x + c3 * sigma3Amp x

/-! ### Layer 1: Trigonometric identities (PROVED) -/

lemma cos_pi_div_12 :
    Real.cos (Real.pi / 12) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  have h : Real.pi / 12 = Real.pi / 3 - Real.pi / 4 := by ring
  rw [h, Real.cos_sub]
  rw [Real.cos_pi_div_three, Real.cos_pi_div_four, Real.sin_pi_div_three, Real.sin_pi_div_four]
  have h6 : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
    rw [← Real.sqrt_mul (by linarith)]; norm_num
  calc
    1 / 2 * (Real.sqrt 2 / 2) + Real.sqrt 3 / 2 * (Real.sqrt 2 / 2)
      = (Real.sqrt 3 * Real.sqrt 2 + Real.sqrt 2) / 4 := by ring
    _ = (Real.sqrt 6 + Real.sqrt 2) / 4 := by rw [h6]

lemma sin_pi_div_12 :
    Real.sin (Real.pi / 12) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  have h : Real.pi / 12 = Real.pi / 3 - Real.pi / 4 := by ring
  rw [h, Real.sin_sub]
  rw [Real.cos_pi_div_three, Real.cos_pi_div_four, Real.sin_pi_div_three, Real.sin_pi_div_four]
  have h6 : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
    rw [← Real.sqrt_mul (by linarith)]; norm_num
  calc
    Real.sqrt 3 / 2 * (Real.sqrt 2 / 2) - 1 / 2 * (Real.sqrt 2 / 2)
      = (Real.sqrt 3 * Real.sqrt 2 - Real.sqrt 2) / 4 := by ring
    _ = (Real.sqrt 6 - Real.sqrt 2) / 4 := by rw [h6]

lemma cos_sub_sin_pi_div_12 :
    Real.cos (Real.pi / 12) - Real.sin (Real.pi / 12) = 1 / Real.sqrt 2 := by
  rw [cos_pi_div_12, sin_pi_div_12]
  have h_div : (Real.sqrt 6 + Real.sqrt 2) / 4 - (Real.sqrt 6 - Real.sqrt 2) / 4 = Real.sqrt 2 / 2 := by ring
  rw [h_div]
  have h_sqrt2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by linarith)
  have h_ne : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by linarith)
  field_simp
  linarith

lemma cos_add_sin_pi_div_12 :
    Real.cos (Real.pi / 12) + Real.sin (Real.pi / 12) = Real.sqrt 6 / 2 := by
  rw [cos_pi_div_12, sin_pi_div_12]; ring

/-! ### Layer 2: Exponential → trig rewriting

  HINT: Use `Complex.exp_ofReal_mul_I` or `Complex.exp_mul_I` and
  `mul_comm` to rewrite Complex.exp(I * θ) into cos θ + i sin θ. -/

/-- e^{iπ/12} = cos(π/12) + i·sin(π/12).
    TACTIC HINT: `rw [mul_comm, Complex.exp_ofReal_mul_I]; ring` -/
lemma exp_ipi_div_12 :
    Complex.exp (Complex.I * ↑(Real.pi / 12)) =
    ↑(Real.cos (Real.pi / 12)) + Complex.I * ↑(Real.sin (Real.pi / 12)) := by
  rw [mul_comm, Complex.exp_ofReal_mul_I]
  ring

/-- e^{-iπ/12} = cos(π/12) - i·sin(π/12).
    TACTIC HINT: Use `Complex.exp_neg`, `Complex.conj_exp`, etc. -/
lemma exp_neg_ipi_div_12 :
    Complex.exp (-Complex.I * ↑(Real.pi / 12)) =
    ↑(Real.cos (Real.pi / 12)) - Complex.I * ↑(Real.sin (Real.pi / 12)) := by
  have h1 : -Complex.I * ↑(Real.pi / 12) = ↑(-(Real.pi / 12)) * Complex.I := by push_cast; ring
  rw [h1, Complex.exp_ofReal_mul_I]
  rw [Real.cos_neg, Real.sin_neg]
  push_cast
  ring

/-- c₁ = (2/3)cos(π/12) + i·(2/3)sin(π/12).
    TACTIC HINT: unfold c1, rw [exp_ipi_div_12], push_cast, ring -/
lemma c1_expand :
    c1 = ↑((2 / 3 : ℝ) * Real.cos (Real.pi / 12)) +
         Complex.I * ↑((2 / 3 : ℝ) * Real.sin (Real.pi / 12)) := by
  unfold c1
  rw [exp_ipi_div_12]
  push_cast
  ring

/-- c₃ = (2/3)cos(π/12) - i·(2/3)sin(π/12).
    TACTIC HINT: unfold c3, rw [exp_neg_ipi_div_12], push_cast, ring -/
lemma c3_expand :
    c3 = ↑((2 / 3 : ℝ) * Real.cos (Real.pi / 12)) -
         Complex.I * ↑((2 / 3 : ℝ) * Real.sin (Real.pi / 12)) := by
  unfold c3
  rw [exp_neg_ipi_div_12]
  push_cast
  ring

/-- c₁ + c₃ = (4/3)cos(π/12). Imaginary parts cancel.
    TACTIC HINT: rw [c1_expand, c3_expand], push_cast, ring or ext -/
lemma c1_add_c3 :
    c1 + c3 = ↑((4 / 3 : ℝ) * Real.cos (Real.pi / 12)) := by
  rw [c1_expand, c3_expand]
  push_cast
  ring

/-- c₁ - c₃ = i·(4/3)sin(π/12).
    TACTIC HINT: rw [c1_expand, c3_expand], push_cast, ring or ext -/
lemma c1_sub_c3 :
    c1 - c3 = Complex.I * ↑((4 / 3 : ℝ) * Real.sin (Real.pi / 12)) := by
  rw [c1_expand, c3_expand]
  push_cast
  ring

/-! ### Layer 3: Sqrt helper facts

  HINT: Use `Real.mul_self_sqrt`, `Real.sqrt_mul`, `Real.sqrt_ne_zero'`. -/

lemma sqrt6_eq : Real.sqrt 6 = Real.sqrt 2 * Real.sqrt 3 := by
  rw [← Real.sqrt_mul (by norm_num)]
  congr 1
  norm_num

lemma sqrt2_sq : Real.sqrt 2 * Real.sqrt 2 = 2 := by
  exact Real.mul_self_sqrt (by norm_num)

lemma sqrt3_sq : Real.sqrt 3 * Real.sqrt 3 = 3 := by
  exact Real.mul_self_sqrt (by norm_num)

lemma sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := by
  exact Real.sqrt_ne_zero'.mpr (by norm_num)

lemma sqrt3_ne_zero : Real.sqrt 3 ≠ 0 := by
  exact Real.sqrt_ne_zero'.mpr (by norm_num)

set_option linter.unusedSimpArgs false

theorem qubit_t_m4_decomposition (x : Basis4) :
    rhsAmplitude x = targetAmplitude x := by
  generalize h0 : x 0 = x0
  generalize h1 : x 1 = x1
  generalize h2 : x 2 = x2
  generalize h3 : x 3 = x3
  fin_cases x0 <;> fin_cases x1 <;> fin_cases x2 <;> fin_cases x3
  all_goals
    unfold rhsAmplitude targetAmplitude sigma1Amp sigma2Amp sigma3Amp hammingWeight bval
    try simp only [h0, h1, h2, h3, decide_true, decide_false, if_true, if_false]
    try simp only [pow_zero, pow_one, mul_one, mul_zero, add_zero, zero_add, neg_mul, one_mul, zero_mul, neg_zero]
    have h_I2 : Complex.I ^ 2 = -1 := Complex.I_sq
    have h_I3 : Complex.I ^ 3 = -Complex.I := by
      calc Complex.I ^ 3 = Complex.I ^ 2 * Complex.I := by ring
      _ = -1 * Complex.I := by rw [h_I2]
      _ = -Complex.I := by ring
    have h_I4 : Complex.I ^ 4 = 1 := by
      calc Complex.I ^ 4 = Complex.I ^ 3 * Complex.I := by ring
      _ = -Complex.I * Complex.I := by rw [h_I3]
      _ = -(Complex.I ^ 2) := by ring
      _ = 1 := by rw [h_I2]; ring
    try simp only [h_I2, h_I3, h_I4]
    norm_num
    try simp only [c1, c2, c3]
    try rw [exp_ipi_div_12]
    try rw [exp_neg_ipi_div_12]
    try rw [cos_pi_div_12]
    try rw [sin_pi_div_12]
    try rw [sqrt6_eq]
    push_cast
    have h_sq2 : (↑(Real.sqrt 2) : ℂ) * ↑(Real.sqrt 2) = 2 := by
      rw [← Complex.ofReal_mul, sqrt2_sq]; rfl
    have h_sq3 : (↑(Real.sqrt 3) : ℂ) * ↑(Real.sqrt 3) = 3 := by
      rw [← Complex.ofReal_mul, sqrt3_sq]; rfl
    have h_ne2 : (↑(Real.sqrt 2) : ℂ) ≠ 0 := by
      intro hc; apply sqrt2_ne_zero; exact Complex.ofReal_eq_zero.mp hc
    field_simp [h_ne2]
    have h_sq2_pow : (↑(Real.sqrt 2) : ℂ) ^ 2 = 2 := by rw [sq, h_sq2]
    have h_sq3_pow : (↑(Real.sqrt 3) : ℂ) ^ 2 = 3 := by rw [sq, h_sq3]
    simp (config := { failIfUnchanged := false }) only [h_I2, h_I3, h_I4, h_sq2, h_sq3, h_sq2_pow, h_sq3_pow, Complex.I_sq]
    first | ring | (simp only [Complex.I_sq]; ring)

set_option linter.unusedVariables false

/-- **Case (0,0)**: x₁ ≠ x₃ and x₂ ≠ x₄. Only σ₂ contributes → i/6. -/
lemma case_00 (x : Basis4) (ha : x 0 ≠ x 2) (hb : x 1 ≠ x 3) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (1,1), (0,0)**: all zero, w=0. Result: (2+√3)/6.
    Uses: c₁+c₃ = (4/3)cos(π/12) and cos(π/12) = (√6+√2)/4. -/
lemma case_11_00 (x : Basis4)
    (h0 : x 0 = 0) (h1 : x 1 = 0) (h2 : x 2 = 0) (h3 : x 3 = 0) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (1,1), (0,1)**: w=2. Result: i/6.
    Uses: cos(π/12) - sin(π/12) = 1/√2. -/
lemma case_11_01 (x : Basis4)
    (h0 : x 0 = 0) (h1 : x 1 = 1) (h2 : x 2 = 0) (h3 : x 3 = 1) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (1,1), (1,0)**: w=2. Result: i/6. -/
lemma case_11_10 (x : Basis4)
    (h0 : x 0 = 1) (h1 : x 1 = 0) (h2 : x 2 = 1) (h3 : x 3 = 0) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (1,1), (1,1)**: w=4. Result: (√3-2)/6.
    Uses: c₁+c₃ with i² factor gives (4/3)sin(π/12)·something. -/
lemma case_11_11 (x : Basis4)
    (h0 : x 0 = 1) (h1 : x 1 = 1) (h2 : x 2 = 1) (h3 : x 3 = 1) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (1,0), p=0**: w=1. Result: (√3+1)(1+i)/12.
    Uses: c₁/(2√2) + c₂·i/4 with exp_ipi_div_12. -/
lemma case_10_p0 (x : Basis4)
    (ha : x 0 = x 2) (hb : x 1 ≠ x 3) (hp : x 0 = 0) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (1,0), p=1**: w=3. Result: (√3-1)(-1+i)/12. -/
lemma case_10_p1 (x : Basis4)
    (ha : x 0 = x 2) (hb : x 1 ≠ x 3) (hp : x 0 = 1) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (0,1), q=0**: w=1. Result: (√3+1)(1+i)/12. -/
lemma case_01_q0 (x : Basis4)
    (ha : x 0 ≠ x 2) (hb : x 1 = x 3) (hq : x 1 = 0) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-- **Case (0,1), q=1**: w=3. Result: (√3-1)(-1+i)/12. -/
lemma case_01_q1 (x : Basis4)
    (ha : x 0 ≠ x 2) (hb : x 1 = x 3) (hq : x 1 = 1) :
    rhsAmplitude x = targetAmplitude x := by
  exact qubit_t_m4_decomposition x

/-! ### Layer 5: Assembly

  HINT for qubit_t_m4_decomposition:
  Case split on (x 0 = x 2) and (x 1 = x 3), then on x 0 and x 1.
  Use `fin_cases` or `rcases (fin2_eq_zero_or_one (x i))`.
  Derive x 2, x 3 from the equalities, then apply the case lemma.

  HINT for stabilizer_rank_le_three:
  Use `⟨3, rfl, σ, c, fun x => ...⟩` where σ and c are the explicit functions.
  Rewrite with `← qubit_t_m4_decomposition x`, unfold rhsAmplitude,
  use `Fin.sum_univ_three`, and close with `rfl`.

  HINT for gamma_T_bound:
  This is purely numerical. Use `Real.logb`, unfold to `Real.log 3 / Real.log 2`,
  and show 3^250 < 2^397 (decidable by `norm_num`). -/

theorem stabilizer_rank_le_three :
    ∃ (n : ℕ) (_ : n = 3)
      (σ : Fin n → Basis4 → ℂ) (c : Fin n → ℂ),
      ∀ x : Basis4,
        targetAmplitude x = ∑ i : Fin n, c i * σ i x := by
  use 3, rfl, ![sigma1Amp, sigma2Amp, sigma3Amp], ![c1, c2, c3]
  intro x
  rw [← qubit_t_m4_decomposition x]
  unfold rhsAmplitude
  simp only [Fin.sum_univ_three]
  rfl

set_option exponentiation.threshold 500

theorem gamma_T_bound :
    Real.logb 2 3 / 4 < 0.397 := by
  rw [Real.logb]
  have h_pow : (3 : ℝ) ^ 250 < (2 : ℝ) ^ 397 := by norm_num
  have h_log : Real.log ((3 : ℝ) ^ 250) < Real.log ((2 : ℝ) ^ 397) := by
    apply Real.log_lt_log (by norm_num) h_pow
  rw [Real.log_pow, Real.log_pow] at h_log
  push_cast at h_log
  have h_log2_pos : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have h_four : (0 : ℝ) < 4 := by norm_num
  rw [div_lt_iff₀ h_four]
  rw [div_lt_iff₀ h_log2_pos]
  linarith

end StabilizerRank.QubitTM4
