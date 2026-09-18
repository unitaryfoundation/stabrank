/-
The three-copy upper bounds (Strange, Norrell, H₃) as theorems about
`Stabilizer.stabRank`.

`StrangeM3Pointwise`, `NorrellM3Pointwise` and `H3M3Pointwise` prove the
rank-4 identities pointwise on `F_3³`. Here every term is shown to be `IsStab`
in the concrete sense, with the flats and phases read off from the tabulated
amplitudes:

  Strange: two planes `y₁ = 2` and `y₁ = 1`, coordinates `(y₀, y₂)`, phases
    `w₀² + w₀w₂`, `w₀² + 2w₀w₂` on the first and `2w₀² + w₀w₂ + w₂²`,
    `2w₀² + 2w₀w₂ + w₂²` on the second;
  Norrell: the planes `y₀ = 2`, `y₁ = 2`, `y₂ = 2` with phases built from the
    indicator `[y = 2] = 2y² + y`, and the uniform state `|+⟩^⊗3`;
  H₃: two full-support states with phases `2y₀² + 2y₁² + y₂²` and
    `y₀² + y₁² + 2y₂²` (the indicator `[y ≠ 0]` is `y²` mod 3), the line
    `|0,0,+⟩` and the plane `|+,+,0⟩`.

`stabRank_le_of_decomp` then gives `stabRank IsStab ψ ≤ 4` in each case.
-/
import LeanProofs.M2StabRank
import LeanProofs.StrangeM3Pointwise
import LeanProofs.NorrellM3Pointwise
import LeanProofs.H3M3Pointwise

namespace StabRank

open Stabilizer

/-! ### Generic lemmas: reindexing, planes, four-term spans -/

/-- A `Fin 3 × Fin 3 × Fin 3`-indexed amplitude function as a 3-qutrit vector. -/
noncomputable def ofTriple (f : Fin 3 × Fin 3 × Fin 3 → ℂ) : QutritVec 3 :=
  fun idx => f (digits 3 idx 0, digits 3 idx 1, digits 3 idx 2)

theorem fin3_eta (x : Fin 3 → Fin 3) : x = ![x 0, x 1, x 2] := by
  funext i
  fin_cases i <;> rfl

/-- The nine strings of `Fin 2 → Fin 3`. -/
theorem sum_fin2_fin3 (f : (Fin 2 → Fin 3) → ℂ) :
    ∑ y : Fin 2 → Fin 3, f y
      = f ![0, 0] + f ![0, 1] + f ![0, 2] + f ![1, 0] + f ![1, 1] + f ![1, 2]
        + f ![2, 0] + f ![2, 1] + f ![2, 2] := by
  rw [show (Finset.univ : Finset (Fin 2 → Fin 3))
      = {![0, 0], ![0, 1], ![0, 2], ![1, 0], ![1, 1], ![1, 2], ![2, 0], ![2, 1], ![2, 2]}
      by decide]
  simp [Finset.sum_insert, Finset.mem_insert]
  ring

/-- `stabVecN` with `k = 2`: nine terms. -/
theorem stabVecN_k2 (n : ℕ) (x0 : Fin n → Fin 3) (W : Fin 2 → Fin n → Fin 3)
    (Q : Fin 2 → Fin 2 → Fin 3) (l : Fin 2 → Fin 3) (x : Fin n → Fin 3) :
    stabVecN n 2 x0 W Q l x
      = (if x = affinePt x0 W ![0, 0] then omega3 ^ quadPhase Q l ![0, 0] else 0)
      + (if x = affinePt x0 W ![0, 1] then omega3 ^ quadPhase Q l ![0, 1] else 0)
      + (if x = affinePt x0 W ![0, 2] then omega3 ^ quadPhase Q l ![0, 2] else 0)
      + (if x = affinePt x0 W ![1, 0] then omega3 ^ quadPhase Q l ![1, 0] else 0)
      + (if x = affinePt x0 W ![1, 1] then omega3 ^ quadPhase Q l ![1, 1] else 0)
      + (if x = affinePt x0 W ![1, 2] then omega3 ^ quadPhase Q l ![1, 2] else 0)
      + (if x = affinePt x0 W ![2, 0] then omega3 ^ quadPhase Q l ![2, 0] else 0)
      + (if x = affinePt x0 W ![2, 1] then omega3 ^ quadPhase Q l ![2, 1] else 0)
      + (if x = affinePt x0 W ![2, 2] then omega3 ^ quadPhase Q l ![2, 2] else 0) := by
  unfold stabVecN
  rw [sum_fin2_fin3]

/-- Injectivity of a plane parametrisation whose two rows of `W` carry a
    `1` in distinct coordinates `i₀ ≠ i₁` and the other row a `0` there. -/
theorem affinePt_plane_injective {x0 : Fin 3 → Fin 3} {W : Fin 2 → Fin 3 → Fin 3}
    (i0 i1 : Fin 3) (h00 : W 0 i0 = 1) (h10 : W 1 i0 = 0) (h01 : W 0 i1 = 0)
    (h11 : W 1 i1 = 1) : Function.Injective (affinePt x0 W) := by
  intro y y' h
  have e0 := congrFun h i0
  have e1 := congrFun h i1
  simp only [affinePt, Fin.sum_univ_two, h00, h10, h01, h11, mul_one, mul_zero, add_zero,
    zero_add] at e0 e1
  funext j
  fin_cases j
  · exact add_left_cancel e0
  · exact add_left_cancel e1

theorem mem_span_four {a b c d v : QutritVec 3} (α β γ δ : ℂ)
    (h : v = α • a + β • b + γ • c + δ • d) :
    v ∈ Submodule.span ℂ (({a, b, c, d} : Finset (QutritVec 3)) : Set (QutritVec 3)) := by
  classical
  rw [h]
  have ha : a ∈ Submodule.span ℂ (({a, b, c, d} : Finset (QutritVec 3)) : Set _) :=
    Submodule.subset_span (by simp)
  have hb : b ∈ Submodule.span ℂ (({a, b, c, d} : Finset (QutritVec 3)) : Set _) :=
    Submodule.subset_span (by simp)
  have hc : c ∈ Submodule.span ℂ (({a, b, c, d} : Finset (QutritVec 3)) : Set _) :=
    Submodule.subset_span (by simp)
  have hd : d ∈ Submodule.span ℂ (({a, b, c, d} : Finset (QutritVec 3)) : Set _) :=
    Submodule.subset_span (by simp)
  exact Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _
    (Submodule.smul_mem _ _ ha) (Submodule.smul_mem _ _ hb)) (Submodule.smul_mem _ _ hc))
    (Submodule.smul_mem _ _ hd)

theorem stabRank_le_four_of {a b c d v : QutritVec 3} (ha : IsStab a) (hb : IsStab b)
    (hc : IsStab c) (hd : IsStab d) (α β γ δ : ℂ) (h : v = α • a + β • b + γ • c + δ • d) :
    Stabilizer.stabRank (IsStab (n := 3)) v ≤ 4 := by
  classical
  have hcard : ({a, b, c, d} : Finset (QutritVec 3)).card ≤ 4 :=
    le_trans (Finset.card_insert_le _ _) (Nat.succ_le_succ Finset.card_le_three)
  refine le_trans
    (Stabilizer.stabRank_le_of_decomp (S := {a, b, c, d}) ?_ (mem_span_four α β γ δ h)) hcard
  intro σ hσ
  simp only [Finset.mem_insert, Finset.mem_singleton] at hσ
  rcases hσ with rfl | rfl | rfl | rfl
  · exact ha
  · exact hb
  · exact hc
  · exact hd

private theorem inv3_ne : (1 / 3 : ℂ) ≠ 0 := by norm_num

private theorem inv_sqrt3_ne : (1 / (Real.sqrt 3 : ℂ)) ≠ 0 := by
  have : (Real.sqrt 3 : ℂ) ≠ 0 := by exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num)
  exact one_div_ne_zero this

private theorem inv_3sqrt3_ne : (1 / (3 * (Real.sqrt 3 : ℂ))) ≠ 0 := by
  have : (Real.sqrt 3 : ℂ) ≠ 0 := by exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num)
  exact one_div_ne_zero (mul_ne_zero (by norm_num) this)

/-! ### Strange m=3 -/

noncomputable def strangeVec3 : QutritVec 3 := ofTriple strangeAmp3
noncomputable def s1Vec3 : QutritVec 3 := ofTriple s1Amp
noncomputable def s2Vec3 : QutritVec 3 := ofTriple s2Amp
noncomputable def s3Vec3 : QutritVec 3 := ofTriple s3Amp
noncomputable def s4Vec3 : QutritVec 3 := ofTriple s4Amp

/-- Rows of `W` for the planes `y₁ = const`: coordinates `(y₀, y₂)`. -/
def wPlane1 : Fin 2 → Fin 3 → Fin 3 := ![![1, 0, 0], ![0, 0, 1]]

theorem wPlane1_injective (x0 : Fin 3 → Fin 3) : Function.Injective (affinePt x0 wPlane1) :=
  affinePt_plane_injective 0 2 rfl rfl rfl rfl

/-- Rows of `W` for the planes `y₀ = const`: coordinates `(y₁, y₂)`. -/
def wPlane0 : Fin 2 → Fin 3 → Fin 3 := ![![0, 1, 0], ![0, 0, 1]]
/-- Rows of `W` for the planes `y₂ = const`: coordinates `(y₀, y₁)`. -/
def wPlane2 : Fin 2 → Fin 3 → Fin 3 := ![![1, 0, 0], ![0, 1, 0]]

theorem wPlane0_injective (x0 : Fin 3 → Fin 3) : Function.Injective (affinePt x0 wPlane0) :=
  affinePt_plane_injective 1 2 rfl rfl rfl rfl

theorem wPlane2_injective (x0 : Fin 3 → Fin 3) : Function.Injective (affinePt x0 wPlane2) :=
  affinePt_plane_injective 0 1 rfl rfl rfl rfl

set_option linter.flexible false in
theorem s1Vec3_isStab : IsStab s1Vec3 := by
  refine ⟨1 / 3, 2, ![0, 2, 0], wPlane1, ![![1, 1], ![0, 0]], 0, inv3_ne,
    wPlane1_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : s1Vec3 idx = s1Amp (a, b, c) := by
    simp only [s1Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [s1Amp, q1Nat, quadPhase, omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
theorem s3Vec3_isStab : IsStab s3Vec3 := by
  refine ⟨1 / 3, 2, ![0, 2, 0], wPlane1, ![![1, 2], ![0, 0]], 0, inv3_ne,
    wPlane1_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : s3Vec3 idx = s3Amp (a, b, c) := by
    simp only [s3Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [s3Amp, q3Nat, quadPhase, omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
theorem s2Vec3_isStab : IsStab s2Vec3 := by
  refine ⟨1 / 3, 2, ![0, 1, 0], wPlane1, ![![2, 1], ![0, 1]], 0, inv3_ne,
    wPlane1_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : s2Vec3 idx = s2Amp (a, b, c) := by
    simp only [s2Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [s2Amp, q2Nat, quadPhase, omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
theorem s4Vec3_isStab : IsStab s4Vec3 := by
  refine ⟨1 / 3, 2, ![0, 1, 0], wPlane1, ![![2, 2], ![0, 1]], 0, inv3_ne,
    wPlane1_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : s4Vec3 idx = s4Amp (a, b, c) := by
    simp only [s4Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [s4Amp, q4Nat, quadPhase, omega3_pow_mod]
  all_goals ring

theorem strangeVec3_eq :
    strangeVec3 = alphaStrange • s1Vec3 + (-alphaStrange) • s3Vec3
      + betaStrange • s2Vec3 + (-betaStrange) • s4Vec3 := by
  funext idx
  simp only [strangeVec3, s1Vec3, s2Vec3, s3Vec3, s4Vec3, ofTriple, Pi.add_apply,
    Pi.smul_apply, smul_eq_mul, strange_m3_decomposition]
  ring

/-- **χ(|S⟩^⊗3) ≤ 4 against the concrete stabilizer predicate.** -/
theorem strange_m3_stabRank_le_four :
    Stabilizer.stabRank (IsStab (n := 3)) strangeVec3 ≤ 4 :=
  stabRank_le_four_of s1Vec3_isStab s3Vec3_isStab s2Vec3_isStab s4Vec3_isStab _ _ _ _
    strangeVec3_eq

/-! ### Norrell m=3 -/

noncomputable def norrellVec3 : QutritVec 3 := ofTriple norrellAmp3
noncomputable def sN1Vec3 : QutritVec 3 := ofTriple sN1Amp
noncomputable def sN2Vec3 : QutritVec 3 := ofTriple sN2Amp
noncomputable def sN3Vec3 : QutritVec 3 := ofTriple sN3Amp
noncomputable def sN4Vec3 : QutritVec 3 := ofTriple sN4Amp

set_option linter.flexible false in
theorem sN1Vec3_isStab : IsStab sN1Vec3 := by
  refine ⟨1 / 3, 2, ![2, 0, 0], wPlane0, ![![2, 0], ![0, 1]], ![1, 2], inv3_ne,
    wPlane0_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : sN1Vec3 idx = sN1Amp (a, b, c) := by
    simp only [sN1Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [sN1Amp, isTwo, quadPhase, omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
theorem sN2Vec3_isStab : IsStab sN2Vec3 := by
  refine ⟨1 / 3, 2, ![0, 2, 0], wPlane1, ![![1, 0], ![0, 2]], ![2, 1], inv3_ne,
    wPlane1_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : sN2Vec3 idx = sN2Amp (a, b, c) := by
    simp only [sN2Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [sN2Amp, isTwo, quadPhase, omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
theorem sN3Vec3_isStab : IsStab sN3Vec3 := by
  refine ⟨1 / (3 * (Real.sqrt 3 : ℂ)), 3, 0, idW 3, 0, 0, inv_3sqrt3_ne,
    affinePt_id_injective 3, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  simp [sN3Vec3, ofTriple, stabVecN_id, sN3Amp, quadPhase]

set_option linter.flexible false in
theorem sN4Vec3_isStab : IsStab sN4Vec3 := by
  refine ⟨1 / 3, 2, ![0, 0, 2], wPlane2, ![![2, 0], ![0, 1]], ![1, 2], inv3_ne,
    wPlane2_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : sN4Vec3 idx = sN4Amp (a, b, c) := by
    simp only [sN4Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [sN4Amp, isTwo, quadPhase, omega3_pow_mod]
  all_goals ring

theorem norrellVec3_eq :
    norrellVec3 = alphaN12 • sN1Vec3 + alphaN12 • sN2Vec3 + alphaN3 • sN3Vec3
      + alphaN12 • sN4Vec3 := by
  funext idx
  simp only [norrellVec3, sN1Vec3, sN2Vec3, sN3Vec3, sN4Vec3, ofTriple, Pi.add_apply,
    Pi.smul_apply, smul_eq_mul, norrell_m3_decomposition]

/-- **χ(|N⟩^⊗3) ≤ 4 against the concrete stabilizer predicate.** -/
theorem norrell_m3_stabRank_le_four :
    Stabilizer.stabRank (IsStab (n := 3)) norrellVec3 ≤ 4 :=
  stabRank_le_four_of sN1Vec3_isStab sN2Vec3_isStab sN3Vec3_isStab sN4Vec3_isStab _ _ _ _
    norrellVec3_eq

/-! ### H₃ m=3 -/

noncomputable def h3Vec3 : QutritVec 3 := ofTriple h3Amp3
noncomputable def sH1Vec3 : QutritVec 3 := ofTriple sH1Amp
noncomputable def sH2Vec3 : QutritVec 3 := ofTriple sH2Amp
noncomputable def sH3Vec3 : QutritVec 3 := ofTriple sH3Amp
noncomputable def sH4Vec3 : QutritVec 3 := ofTriple sH4Amp

/-- `2y₀² + 2y₁² + y₂²`. -/
def qH1Mat3 : Fin 3 → Fin 3 → Fin 3 := ![![2, 0, 0], ![0, 2, 0], ![0, 0, 1]]
/-- `y₀² + y₁² + 2y₂²`. -/
def qH2Mat3 : Fin 3 → Fin 3 → Fin 3 := ![![1, 0, 0], ![0, 1, 0], ![0, 0, 2]]

theorem qH1_phase_mod3 (a b c : Fin 3) :
    q1H3 a b c % 3 = quadPhase qH1Mat3 0 ![a, b, c] % 3 := by
  fin_cases a <;> fin_cases b <;> fin_cases c <;> decide

theorem qH2_phase_mod3 (a b c : Fin 3) :
    q2H3 a b c % 3 = quadPhase qH2Mat3 0 ![a, b, c] % 3 := by
  fin_cases a <;> fin_cases b <;> fin_cases c <;> decide

theorem qH1_phase_mod3' (x : Fin 3 → Fin 3) :
    q1H3 (x 0) (x 1) (x 2) % 3 = quadPhase qH1Mat3 0 x % 3 := by
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  simpa using qH1_phase_mod3 a b c

theorem qH2_phase_mod3' (x : Fin 3 → Fin 3) :
    q2H3 (x 0) (x 1) (x 2) % 3 = quadPhase qH2Mat3 0 x % 3 := by
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  simpa using qH2_phase_mod3 a b c

theorem sH1Vec3_isStab : IsStab sH1Vec3 := by
  refine ⟨1 / (3 * (Real.sqrt 3 : ℂ)), 3, 0, idW 3, qH1Mat3, 0, inv_3sqrt3_ne,
    affinePt_id_injective 3, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  simp only [sH1Vec3, ofTriple, hx, stabVecN_id, sH1Amp]
  rw [omega3_pow_eq_of_mod (qH1_phase_mod3' x)]
  ring

theorem sH2Vec3_isStab : IsStab sH2Vec3 := by
  refine ⟨1 / (3 * (Real.sqrt 3 : ℂ)), 3, 0, idW 3, qH2Mat3, 0, inv_3sqrt3_ne,
    affinePt_id_injective 3, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  simp only [sH2Vec3, ofTriple, hx, stabVecN_id, sH2Amp]
  rw [omega3_pow_eq_of_mod (qH2_phase_mod3' x)]
  ring

set_option linter.flexible false in
/-- The line `{(0, 0, t)}` is injectively parametrised. -/
theorem affinePt_third_injective :
    Function.Injective (affinePt (n := 3) 0 ![![0, 0, 1]]) := by
  intro y y' h
  have h2 := congrFun h 2
  simp [affinePt] at h2
  funext j
  rw [Subsingleton.elim j 0]
  exact h2

set_option linter.flexible false in
theorem sH3Vec3_isStab : IsStab sH3Vec3 := by
  refine ⟨1 / (Real.sqrt 3 : ℂ), 1, 0, ![![0, 0, 1]], ![![0]], 0, inv_sqrt3_ne,
    affinePt_third_injective, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : sH3Vec3 idx = sH3Amp (a, b, c) := by
    simp only [sH3Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k1]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [sH3Amp, quadPhase]

set_option linter.flexible false in
theorem sH4Vec3_isStab : IsStab sH4Vec3 := by
  refine ⟨1 / 3, 2, 0, wPlane2, 0, 0, inv3_ne, wPlane2_injective _, ?_⟩
  funext idx
  generalize hx : digits 3 idx = x
  obtain ⟨a, b, c, rfl⟩ : ∃ a b c, x = ![a, b, c] := ⟨x 0, x 1, x 2, fin3_eta x⟩
  have hab : sH4Vec3 idx = sH4Amp (a, b, c) := by
    simp only [sH4Vec3, ofTriple, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c
  all_goals simp (config := { decide := true }) [sH4Amp, quadPhase]

theorem h3Vec3_eq :
    h3Vec3 = alphaH1 • sH1Vec3 + alphaH2 • sH2Vec3 + alphaH34 • sH3Vec3
      + alphaH34 • sH4Vec3 := by
  funext idx
  simp only [h3Vec3, sH1Vec3, sH2Vec3, sH3Vec3, sH4Vec3, ofTriple, Pi.add_apply,
    Pi.smul_apply, smul_eq_mul, h3_m3_decomposition]

/-- **χ(|H₃⟩^⊗3) ≤ 4 against the concrete stabilizer predicate.** -/
theorem h3_m3_stabRank_le_four :
    Stabilizer.stabRank (IsStab (n := 3)) h3Vec3 ≤ 4 :=
  stabRank_le_four_of sH1Vec3_isStab sH2Vec3_isStab sH3Vec3_isStab sH4Vec3_isStab _ _ _ _
    h3Vec3_eq

end StabRank
