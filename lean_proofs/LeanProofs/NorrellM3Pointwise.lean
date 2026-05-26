/-
Norrell m=3 χ ≤ 4 pointwise vector identity (paper App A.3.2 full lift).

  [|N⟩^⊗3]_y  =  α_12 · ([S_1]_y + [S_2]_y + [S_4]_y) + α_3 · [S_3]_y

for every y ∈ F_3³, with α_12 = -√6/4 = -(√2·√3)/4 and α_3 = √2/4. The
four 3-qutrit stabilizer states have supports

  supp(S_1) = {y : y_0 = 2},  supp(S_2) = {y : y_1 = 2},
  supp(S_3) = F_3³ (the all-plus state),
  supp(S_4) = {y : y_2 = 2},

and on support, [S_j]_y = ω^{Q_j(y)} / 3 with the indicator-only
quadratic forms

  Q_1(y) = a_1 + 2 a_2,  Q_2(y) = 2 a_0 + a_2,  Q_4(y) = a_0 + 2 a_1
  where a_i := [y_i = 2] ∈ {0,1}.   [S_3]_y = 1/(3√3).

Implementation note: definitions use √2·√3 instead of √6 throughout, so
`linear_combination` only ever needs `√2² = 2`, `√3² = 3`, `ω+ω² = -1`,
and `ω³ = 1` — no separate √6 relation, which previously inflated the
linear-combination search space and timed out kernel reduction.
-/
import LeanProofs.NorrellShared

namespace StabRank

open Complex Real

/-- 3-qutrit Norrell⊗³ amplitude. -/
noncomputable def norrellAmp3 (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  norrellAmp1 y.1 * norrellAmp1 y.2.1 * norrellAmp1 y.2.2

/-- S_1 amplitude: supported on y_0 = 2, value ω^{a_1 + 2·a_2}/3. -/
noncomputable def sN1Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.1 = 2 then omega3 ^ (isTwo y.2.1 + 2 * isTwo y.2.2) / 3 else 0

/-- S_2 amplitude: supported on y_1 = 2, value ω^{2·a_0 + a_2}/3. -/
noncomputable def sN2Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.1 = 2 then omega3 ^ (2 * isTwo y.1 + isTwo y.2.2) / 3 else 0

/-- S_3 amplitude: |+⟩^⊗³, value 1/(3√3) everywhere. -/
noncomputable def sN3Amp (_y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  1 / (3 * (Real.sqrt 3 : ℂ))

/-- S_4 amplitude: supported on y_2 = 2, value ω^{a_0 + 2·a_1}/3. -/
noncomputable def sN4Amp (y : Fin 3 × Fin 3 × Fin 3) : ℂ :=
  if y.2.2 = 2 then omega3 ^ (isTwo y.1 + 2 * isTwo y.2.1) / 3 else 0

/-- α_1 = α_2 = α_4 = -√6/4, written as -(√2·√3)/4 to stay in (√2, √3). -/
noncomputable def alphaN12 : ℂ := -((Real.sqrt 2 : ℂ) * (Real.sqrt 3 : ℂ)) / 4

/-- α_3 = √2/4. -/
noncomputable def alphaN3 : ℂ := (Real.sqrt 2 : ℂ) / 4

-- The proof below uses `simp` and `field_simp` flexibly; same justification
-- as in `StrangeM3Pointwise.lean`.
set_option linter.flexible false in
set_option maxHeartbeats 800000 in -- 27 cases × 4 linear_combination patterns over (√2, √3, ω) exceed default 200K.
/-- Pointwise vector identity:
    `|N⟩^⊗3 = α_12·(|S_1⟩ + |S_2⟩ + |S_4⟩) + α_3·|S_3⟩` holds at every y ∈ F_3³. -/
theorem norrell_m3_decomposition (y : Fin 3 × Fin 3 × Fin 3) :
    norrellAmp3 y = alphaN12 * sN1Amp y + alphaN12 * sN2Amp y
                 + alphaN3 * sN3Amp y + alphaN12 * sN4Amp y := by
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
  obtain ⟨y_0, y_1, y_2⟩ := y
  fin_cases y_0 <;> fin_cases y_1 <;> fin_cases y_2 <;>
    simp [norrellAmp3, norrellAmp1, sN1Amp, sN2Amp, sN3Amp, sN4Amp, isTwo,
          alphaN12, alphaN3]
  -- 27 non-trivial goals.  Each one is a polynomial identity in √2, √3,
  -- ω modulo h2 (√2² = 2), h3 (√3² = 3), hΩsum (ω + ω² = -1),
  -- hΩ3 (ω³ = 1).  We dispatch via `first` over 4 patterns × ± signs.
  all_goals
    (field_simp;
     first
     -- n_2 = 0:  -216 + √2²·√3⁴·12 = 0
     | linear_combination (12 * (Real.sqrt 3 : ℂ) ^ 4) * h2
         + (24 * (Real.sqrt 3 : ℂ) ^ 2 + 72) * h3
     -- n_2 = 1:  -36 - √2²·√3⁴·4 + √3²·36 = 0
     | linear_combination (-4 * (Real.sqrt 3 : ℂ) ^ 4) * h2
         + (12 - 8 * (Real.sqrt 3 : ℂ) ^ 2) * h3
     -- n_2 = 2:  -18 + √2²·√3⁴·4 + √3²·(ω+ω²)·18 = 0
     | linear_combination (4 * (Real.sqrt 3 : ℂ) ^ 4) * h2
         + (6 + 8 * (Real.sqrt 3 : ℂ) ^ 2) * h3
         + (18 * (Real.sqrt 3 : ℂ) ^ 2) * hΩsum
     -- n_2 = 3:  -9 - √2²·√3⁴·4 + √3²·ω³·27 = 0
     | linear_combination (-4 * (Real.sqrt 3 : ℂ) ^ 4) * h2
         + (3 - 8 * (Real.sqrt 3 : ℂ) ^ 2) * h3
         + (27 * (Real.sqrt 3 : ℂ) ^ 2) * hΩ3)

/-- Vector form of `norrell_m3_decomposition`. -/
theorem norrell_m3_vector_decomposition :
    norrellAmp3 = fun y =>
      alphaN12 * sN1Amp y + alphaN12 * sN2Amp y
        + alphaN3 * sN3Amp y + alphaN12 * sN4Amp y :=
  funext norrell_m3_decomposition

end StabRank
