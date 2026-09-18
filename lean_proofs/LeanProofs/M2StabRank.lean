/-
The Norrell and H₃ two-copy upper bounds as theorems about `Stabilizer.stabRank`.

`NorrellM2Pointwise` and `H3M2Pointwise` prove the rank-3 identities
pointwise and carry stabilizer-ness in the shape of their definitions. Here
each term is shown to be `IsStab` in the concrete sense:

  Norrell: `S₁` full support with phase `y₀ + y₁ + y₀y₁`, `S₂ = |2,2⟩`
  (`k = 0`), `S₃` full support with phase `2(y₀ + y₁ + y₀y₁)`;
  H₃: `S₁ = |0,+⟩` and `S₂ = |+,0⟩` (lines with trivial phase), `S₃` full
  support with phase `2y₀² + y₁²`.

`stabRank_le_of_decomp` then turns each pointwise identity into
`stabRank IsStab ψ ≤ 3`.
-/
import LeanProofs.T3M2StabRank
import LeanProofs.NorrellM2Pointwise
import LeanProofs.H3M2Pointwise

namespace StabRank

open Stabilizer

/-! ### Generic evaluation lemmas -/

/-- `stabVecN` with `k = 1` on any number of qutrits: three terms. -/
theorem stabVecN_k1 (n : ℕ) (x0 : Fin n → Fin 3) (W : Fin 1 → Fin n → Fin 3)
    (Q : Fin 1 → Fin 1 → Fin 3) (l : Fin 1 → Fin 3) (x : Fin n → Fin 3) :
    stabVecN n 1 x0 W Q l x
      = (if x = affinePt x0 W ![0] then omega3 ^ quadPhase Q l ![0] else 0)
      + (if x = affinePt x0 W ![1] then omega3 ^ quadPhase Q l ![1] else 0)
      + (if x = affinePt x0 W ![2] then omega3 ^ quadPhase Q l ![2] else 0) := by
  unfold stabVecN
  rw [sum_fin1_fin3]

/-- Membership of a vector in the span of three named vectors. -/
theorem mem_span_three {a b c v : QutritVec 2} (α β γ : ℂ)
    (h : v = α • a + β • b + γ • c) :
    v ∈ Submodule.span ℂ (({a, b, c} : Finset (QutritVec 2)) : Set (QutritVec 2)) := by
  classical
  rw [h]
  have ha : a ∈ Submodule.span ℂ (({a, b, c} : Finset (QutritVec 2)) : Set _) :=
    Submodule.subset_span (by simp)
  have hb : b ∈ Submodule.span ℂ (({a, b, c} : Finset (QutritVec 2)) : Set _) :=
    Submodule.subset_span (by simp)
  have hc : c ∈ Submodule.span ℂ (({a, b, c} : Finset (QutritVec 2)) : Set _) :=
    Submodule.subset_span (by simp)
  exact Submodule.add_mem _ (Submodule.add_mem _ (Submodule.smul_mem _ _ ha)
    (Submodule.smul_mem _ _ hb)) (Submodule.smul_mem _ _ hc)

/-- `stabRank ≤ 3` from three `IsStab` vectors and a decomposition. -/
theorem stabRank_le_three_of {a b c v : QutritVec 2} (ha : IsStab a) (hb : IsStab b)
    (hc : IsStab c) (α β γ : ℂ) (h : v = α • a + β • b + γ • c) :
    Stabilizer.stabRank (IsStab (n := 2)) v ≤ 3 := by
  classical
  refine le_trans (Stabilizer.stabRank_le_of_decomp (S := {a, b, c}) ?_ (mem_span_three α β γ h))
    Finset.card_le_three
  intro σ hσ
  simp only [Finset.mem_insert, Finset.mem_singleton] at hσ
  rcases hσ with rfl | rfl | rfl
  · exact ha
  · exact hb
  · exact hc

/-! ### Norrell m=2 -/

noncomputable def norrellVec2 : QutritVec 2 := ofPair norrellAmp2
noncomputable def sN1Vec : QutritVec 2 := ofPair sN1AmpM2
noncomputable def sN2Vec : QutritVec 2 := ofPair sN2AmpM2
noncomputable def sN3Vec : QutritVec 2 := ofPair sN3AmpM2

/-- `Q₁(y) = y₀ + y₁ + y₀y₁`: cross term `1`, linear part `(1, 1)`. -/
def qN1Mat : Fin 2 → Fin 2 → Fin 3 := ![![0, 1], ![0, 0]]
/-- `Q₃ = 2 Q₁`. -/
def qN3Mat : Fin 2 → Fin 2 → Fin 3 := ![![0, 2], ![0, 0]]

theorem qN1_phase_mod (a b : Fin 3) :
    q1Nor2 a b % 3 = quadPhase qN1Mat ![1, 1] ![a, b] % 3 := by
  fin_cases a <;> fin_cases b <;> decide

theorem qN3_phase_mod (a b : Fin 3) :
    q3Nor2 a b % 3 = quadPhase qN3Mat ![2, 2] ![a, b] % 3 := by
  fin_cases a <;> fin_cases b <;> decide

theorem qN1_phase_mod' (x : Fin 2 → Fin 3) :
    q1Nor2 (x 0) (x 1) % 3 = quadPhase qN1Mat ![1, 1] x % 3 := by
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  simpa using qN1_phase_mod a b

theorem qN3_phase_mod' (x : Fin 2 → Fin 3) :
    q3Nor2 (x 0) (x 1) % 3 = quadPhase qN3Mat ![2, 2] x % 3 := by
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  simpa using qN3_phase_mod a b

theorem sN1Vec_isStab : IsStab sN1Vec := by
  refine ⟨1 / 3, 2, 0, idW 2, qN1Mat, ![1, 1], by norm_num, affinePt_id_injective 2, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  simp only [sN1Vec, ofPair, hx, stabVecN_id, sN1AmpM2, finTwoArrowEquiv_pair]
  rw [omega3_pow_eq_of_mod (qN1_phase_mod' x)]
  ring

theorem sN3Vec_isStab : IsStab sN3Vec := by
  refine ⟨1 / 3, 2, 0, idW 2, qN3Mat, ![2, 2], by norm_num, affinePt_id_injective 2, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  simp only [sN3Vec, ofPair, hx, stabVecN_id, sN3AmpM2, finTwoArrowEquiv_pair]
  rw [omega3_pow_eq_of_mod (qN3_phase_mod' x)]
  ring

set_option linter.flexible false in
theorem sN2Vec_isStab : IsStab sN2Vec := by
  refine ⟨1, 0, ![2, 2], fun j => Fin.elim0 j, fun i _ => Fin.elim0 i, fun i => Fin.elim0 i,
    one_ne_zero, Function.injective_of_subsingleton _, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  have hab : sN2Vec idx = sN2AmpM2 (a, b) := by
    simp only [sN2Vec, ofPair, hx]
    rfl
  rw [hab, stabVecN_zero, one_mul]
  fin_cases a <;> fin_cases b <;> simp (config := { decide := true }) [sN2AmpM2]

theorem norrellVec2_eq :
    norrellVec2 = alphaNor2 • sN1Vec + (1 : ℂ) • sN2Vec + alphaNor2bar • sN3Vec := by
  funext idx
  simp only [norrellVec2, sN1Vec, sN2Vec, sN3Vec, ofPair, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul, norrell_m2_decomposition]
  ring

/-- **χ(|N⟩^⊗2) ≤ 3 against the concrete stabilizer predicate.** -/
theorem norrell_m2_stabRank_le_three :
    Stabilizer.stabRank (IsStab (n := 2)) norrellVec2 ≤ 3 :=
  stabRank_le_three_of sN1Vec_isStab sN2Vec_isStab sN3Vec_isStab _ _ _ norrellVec2_eq

/-! ### H₃ m=2 -/

noncomputable def h3Vec2 : QutritVec 2 := ofPair h3Amp2
noncomputable def sH1Vec : QutritVec 2 := ofPair s1H2Amp
noncomputable def sH2Vec : QutritVec 2 := ofPair s2H2Amp
noncomputable def sH3Vec : QutritVec 2 := ofPair s3H2Amp

/-- `Q₃(y) = 2y₀² + y₁²`. -/
def qH3Mat : Fin 2 → Fin 2 → Fin 3 := ![![2, 0], ![0, 1]]

theorem qH3_phase_mod (a b : Fin 3) :
    q3H2 a b % 3 = quadPhase qH3Mat 0 ![a, b] % 3 := by
  fin_cases a <;> fin_cases b <;> decide

theorem qH3_phase_mod' (x : Fin 2 → Fin 3) :
    q3H2 (x 0) (x 1) % 3 = quadPhase qH3Mat 0 x % 3 := by
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  simpa using qH3_phase_mod a b

theorem sH3Vec_isStab : IsStab sH3Vec := by
  refine ⟨1 / 3, 2, 0, idW 2, qH3Mat, 0, by norm_num, affinePt_id_injective 2, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  simp only [sH3Vec, ofPair, hx, stabVecN_id, s3H2Amp, finTwoArrowEquiv_pair]
  rw [omega3_pow_eq_of_mod (qH3_phase_mod' x)]
  ring

set_option linter.flexible false in
/-- The line `{(0, t)}` is injectively parametrised. -/
theorem affinePt_second_injective :
    Function.Injective (affinePt (n := 2) ![0, 0] ![![0, 1]]) := by
  intro y y' h
  have h1 := congrFun h 1
  simp [affinePt] at h1
  funext j
  rw [Subsingleton.elim j 0]
  exact h1

set_option linter.flexible false in
/-- The line `{(t, 0)}` is injectively parametrised. -/
theorem affinePt_first_injective :
    Function.Injective (affinePt (n := 2) ![0, 0] ![![1, 0]]) := by
  intro y y' h
  have h0 := congrFun h 0
  simp [affinePt] at h0
  funext j
  rw [Subsingleton.elim j 0]
  exact h0

private theorem sqrt3_ne_zero_M2 : (1 / (Real.sqrt 3 : ℂ)) ≠ 0 := by
  have : (Real.sqrt 3 : ℂ) ≠ 0 := by exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num)
  exact one_div_ne_zero this

set_option linter.flexible false in
theorem sH1Vec_isStab : IsStab sH1Vec := by
  refine ⟨1 / (Real.sqrt 3 : ℂ), 1, ![0, 0], ![![0, 1]], ![![0]], 0, sqrt3_ne_zero_M2,
    affinePt_second_injective, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  have hab : sH1Vec idx = s1H2Amp (a, b) := by
    simp only [sH1Vec, ofPair, hx]
    rfl
  rw [hab, stabVecN_k1]
  fin_cases a <;> fin_cases b <;>
    simp (config := { decide := true }) [s1H2Amp, quadPhase]

set_option linter.flexible false in
theorem sH2Vec_isStab : IsStab sH2Vec := by
  refine ⟨1 / (Real.sqrt 3 : ℂ), 1, ![0, 0], ![![1, 0]], ![![0]], 0, sqrt3_ne_zero_M2,
    affinePt_first_injective, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  have hab : sH2Vec idx = s2H2Amp (a, b) := by
    simp only [sH2Vec, ofPair, hx]
    rfl
  rw [hab, stabVecN_k1]
  fin_cases a <;> fin_cases b <;>
    simp (config := { decide := true }) [s2H2Amp, quadPhase]

theorem h3Vec2_eq :
    h3Vec2 = alphaH2_1 • sH1Vec + alphaH2_2 • sH2Vec + alphaH2_3 • sH3Vec := by
  funext idx
  simp only [h3Vec2, sH1Vec, sH2Vec, sH3Vec, ofPair, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul, h3_m2_decomposition]

/-- **χ(|H₃⟩^⊗2) ≤ 3 against the concrete stabilizer predicate.** -/
theorem h3_m2_stabRank_le_three :
    Stabilizer.stabRank (IsStab (n := 2)) h3Vec2 ≤ 3 :=
  stabRank_le_three_of sH1Vec_isStab sH2Vec_isStab sH3Vec_isStab _ _ _ h3Vec2_eq

end StabRank
