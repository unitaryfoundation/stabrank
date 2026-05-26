/-
Strange m=3 χ ≤ 4 pointwise vector identity (paper App A.1.2 full lift).

We lift the scalar algebraic identities `strange_m3_alpha_identity` and
`strange_m3_beta_identity` (in `LeanProofs.StrangeM3`) to the full
27-amplitude pointwise equality

  [|S⟩^⊗3]_y  =  α · ([S_1]_y - [S_3]_y)  +  β · ([S_2]_y - [S_4]_y)

for every y = (y_0, y_1, y_2) ∈ F_3³.

The four 3-qutrit stabilizer states |S_1⟩, |S_3⟩ (supported on the plane
y_1 = 2) and |S_2⟩, |S_4⟩ (supported on y_1 = 1) all share the matrix
W = (e_0; e_2) of canonical-form data; their quadratic forms over F_3
are:

  Q_1(w_0, w_1) = w_0² +   w_0 w_1
  Q_3(w_0, w_1) = w_0² + 2 w_0 w_1
  Q_2(w_0, w_1) = 2 w_0² +   w_0 w_1 +   w_1²
  Q_4(w_0, w_1) = 2 w_0² + 2 w_0 w_1 +   w_1²

with w = (y_0, y_2) on the respective plane. Amplitudes are ω^{Q_j}/3
on support (the factor 1/√(3^k) for k=2), zero off support.
-/
import LeanProofs.StrangeM3

namespace StabRank

open Complex Real

/-- 1-qutrit Strange amplitude: |S⟩ = (|1⟩ - |2⟩)/√2. -/
noncomputable def strangeAmp1 : Fin 3 → ℂ
  | 0 => 0
  | 1 =>  1 / (Real.sqrt 2 : ℂ)
  | 2 => -1 / (Real.sqrt 2 : ℂ)

/-- 3-qutrit Strange⊗³ amplitude. -/
noncomputable def strangeAmp3 (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  strangeAmp1 y.1 * strangeAmp1 y.2.1 * strangeAmp1 y.2.2

/-- Exponent Q_1(y_0, y_2) = y_0² + y_0·y_2 (mod 3), lifted to ℕ. -/
def q1Nat : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, _      => 0
  | ⟨1, _⟩, ⟨0, _⟩ => 1
  | ⟨1, _⟩, ⟨1, _⟩ => 2
  | ⟨1, _⟩, ⟨2, _⟩ => 0  -- 1 + 2 = 3 ≡ 0
  | ⟨2, _⟩, ⟨0, _⟩ => 1  -- 4 + 0 ≡ 1
  | ⟨2, _⟩, ⟨1, _⟩ => 0  -- 4 + 2 ≡ 0
  | ⟨2, _⟩, ⟨2, _⟩ => 2  -- 4 + 4 ≡ 2
  | ⟨_+3, h⟩, _    => absurd h (by omega)
  | _, ⟨_+3, h⟩    => absurd h (by omega)

/-- Exponent Q_3(y_0, y_2) = y_0² + 2·y_0·y_2 (mod 3). -/
def q3Nat : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, _      => 0
  | ⟨1, _⟩, ⟨0, _⟩ => 1
  | ⟨1, _⟩, ⟨1, _⟩ => 0  -- 1 + 2 ≡ 0
  | ⟨1, _⟩, ⟨2, _⟩ => 2  -- 1 + 4 ≡ 2
  | ⟨2, _⟩, ⟨0, _⟩ => 1
  | ⟨2, _⟩, ⟨1, _⟩ => 2  -- 4 + 4 ≡ 2
  | ⟨2, _⟩, ⟨2, _⟩ => 0  -- 4 + 8 ≡ 0
  | ⟨_+3, h⟩, _    => absurd h (by omega)
  | _, ⟨_+3, h⟩    => absurd h (by omega)

/-- Exponent Q_2(y_0, y_2) = 2·y_0² + y_0·y_2 + y_2² (mod 3). -/
def q2Nat : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, ⟨0, _⟩ => 0
  | ⟨0, _⟩, ⟨1, _⟩ => 1
  | ⟨0, _⟩, ⟨2, _⟩ => 1  -- 0 + 0 + 4 ≡ 1
  | ⟨1, _⟩, ⟨0, _⟩ => 2
  | ⟨1, _⟩, ⟨1, _⟩ => 1  -- 2 + 1 + 1 ≡ 1
  | ⟨1, _⟩, ⟨2, _⟩ => 2  -- 2 + 2 + 4 ≡ 2
  | ⟨2, _⟩, ⟨0, _⟩ => 2  -- 8 + 0 + 0 ≡ 2
  | ⟨2, _⟩, ⟨1, _⟩ => 2  -- 8 + 2 + 1 ≡ 2
  | ⟨2, _⟩, ⟨2, _⟩ => 1  -- 8 + 4 + 4 ≡ 1
  | ⟨_+3, h⟩, _    => absurd h (by omega)
  | _, ⟨_+3, h⟩    => absurd h (by omega)

/-- Exponent Q_4(y_0, y_2) = 2·y_0² + 2·y_0·y_2 + y_2² (mod 3). -/
def q4Nat : Fin 3 → Fin 3 → ℕ
  | ⟨0, _⟩, ⟨0, _⟩ => 0
  | ⟨0, _⟩, ⟨1, _⟩ => 1
  | ⟨0, _⟩, ⟨2, _⟩ => 1
  | ⟨1, _⟩, ⟨0, _⟩ => 2
  | ⟨1, _⟩, ⟨1, _⟩ => 2  -- 2 + 2 + 1 ≡ 2
  | ⟨1, _⟩, ⟨2, _⟩ => 1  -- 2 + 4 + 4 ≡ 1
  | ⟨2, _⟩, ⟨0, _⟩ => 2
  | ⟨2, _⟩, ⟨1, _⟩ => 1  -- 8 + 4 + 1 ≡ 1
  | ⟨2, _⟩, ⟨2, _⟩ => 2  -- 8 + 8 + 4 ≡ 2
  | ⟨_+3, h⟩, _    => absurd h (by omega)
  | _, ⟨_+3, h⟩    => absurd h (by omega)

/-- Amplitude of |S_1⟩ on F_3³. Support on the plane y_1 = 2. -/
noncomputable def s1Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.1 = 2 then omega3 ^ q1Nat y.1 y.2.2 / 3 else 0

noncomputable def s3Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.1 = 2 then omega3 ^ q3Nat y.1 y.2.2 / 3 else 0

noncomputable def s2Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.1 = 1 then omega3 ^ q2Nat y.1 y.2.2 / 3 else 0

noncomputable def s4Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.1 = 1 then omega3 ^ q4Nat y.1 y.2.2 / 3 else 0

/-- `(√2)² = 2` as a complex equality, used by the main pointwise theorem. -/
private theorem sqrt2_sq_cx_local : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 2 := by
  have h : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)
  exact_mod_cast h

-- The proof below uses `simp [..]` and `field_simp` — both flagged by Lean's
-- `flexible` linter as "open-ended" tactics that may break under future
-- mathlib changes.  The proof is sound; we silence the linter on this one
-- declaration via `set_option ... in`.
set_option linter.flexible false in
/-- Pointwise vector identity: `|S⟩^⊗3 = α·(|S_1⟩ - |S_3⟩) + β·(|S_2⟩ - |S_4⟩)`
    holds at every y ∈ F_3³. -/
theorem strange_m3_decomposition (y : Fin 3 × Fin 3 × Fin 3) :
    strangeAmp3 y = alphaStrange * (s1Amp y - s3Amp y)
                  + betaStrange  * (s2Amp y - s4Amp y) := by
  have hα := strange_m3_alpha_identity
  have hβ := strange_m3_beta_identity
  have h2 := sqrt2_sq_cx_local
  have hω := omega3_diff_eq_I_sqrt3
  have hne : (Real.sqrt 2 : ℂ) ≠ 0 := by
    intro hzero
    have hp : (Real.sqrt 2 : ℂ) * (Real.sqrt 2 : ℂ) = 0 := by rw [hzero]; ring
    rw [h2] at hp; norm_num at hp
  obtain ⟨y_0, y_1, y_2⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;> fin_cases y_2 <;>
    simp [strangeAmp3, strangeAmp1, s1Amp, s2Amp, s3Amp, s4Amp,
          q1Nat, q2Nat, q3Nat, q4Nat]
  -- 8 non-trivial cases remain.  Each case clears via field_simp and
  -- closes with one of four linear-combination patterns.
  all_goals
    (field_simp;
     first
     | linear_combination (-((Real.sqrt 2 : ℂ) ^ 3)) * hα
         + (3 * ((Real.sqrt 2 : ℂ) ^ 2 + 2) / 4) * h2
     | linear_combination ((Real.sqrt 2 : ℂ) ^ 3) * hα
         + (-3 * ((Real.sqrt 2 : ℂ) ^ 2 + 2) / 4) * h2
     | linear_combination (-((Real.sqrt 2 : ℂ) ^ 3)) * hβ
         + (-3 * ((Real.sqrt 2 : ℂ) ^ 2 + 2) / 4) * h2
         + (-betaStrange * (Real.sqrt 2 : ℂ) ^ 3) * hω
     | linear_combination ((Real.sqrt 2 : ℂ) ^ 3) * hβ
         + (3 * ((Real.sqrt 2 : ℂ) ^ 2 + 2) / 4) * h2
         + (betaStrange * (Real.sqrt 2 : ℂ) ^ 3) * hω)

/-- Vector form of `strange_m3_decomposition`. -/
theorem strange_m3_vector_decomposition :
    strangeAmp3 = fun y =>
      alphaStrange * (s1Amp y - s3Amp y) + betaStrange * (s2Amp y - s4Amp y) :=
  funext strange_m3_decomposition

end StabRank
