/-
χ(|T3⟩^⊗2) = 3 as a theorem about `Stabilizer.stabRank`.

`T3M2Pointwise` proves the carry identity `|T3⟩^⊗2 = (v₀ + ω₉ v₁ + ω₉² v₂)/3`
and asserts in a comment that the three carry blocks are stabilizer states.
Here that assertion is discharged against `IsStab`: each `v_σ` is
`stabVecN 2 1 ![0, σ] ![![1, 2]] Q l` for the line `{(t, σ - t)}` with the
phase `Q t² + l t` read off from the block (`t²` for σ = 0, `2t² + t` for
σ = 1, `0` for σ = 2). With `stabRank_le_of_decomp` that gives the upper
bound, `T3GaloisM.t3_m2_stabRank_gt_two` gives the lower bound, and the two
combine into `t3_m2_stabRank_eq_three`.
-/
import LeanProofs.T3GaloisM

namespace StabRank

open Stabilizer

/-- A sum over `Fin 1 → Fin 3` is a sum over the three constant strings. -/
theorem sum_fin1_fin3 (f : (Fin 1 → Fin 3) → ℂ) :
    ∑ y : Fin 1 → Fin 3, f y = f ![0] + f ![1] + f ![2] := by
  rw [show (Finset.univ : Finset (Fin 1 → Fin 3)) = {![0], ![1], ![2]} by decide]
  simp [Finset.sum_insert, Finset.mem_insert]
  ring

/-- `ω₃^a = ω₃^(a % 3)`; used once per case, never as a simp lemma. -/
theorem omega3_pow_mod (a : ℕ) : omega3 ^ a = omega3 ^ (a % 3) :=
  pow_eq_pow_mod a omega3_pow_three

/-- The carry blocks as two-qutrit vectors. -/
noncomputable def v0Vec : QutritVec 2 := fun idx => v0T3 (digits 2 idx 0) (digits 2 idx 1)
noncomputable def v1Vec : QutritVec 2 := fun idx => v1T3 (digits 2 idx 0) (digits 2 idx 1)
noncomputable def v2Vec : QutritVec 2 := fun idx => v2T3 (digits 2 idx 0) (digits 2 idx 1)

set_option linter.flexible false in
/-- The line `{(t, σ - t)}` is injectively parametrised by `t`. -/
theorem affinePt_line_injective (σ : Fin 3) :
    Function.Injective (affinePt (n := 2) ![0, σ] ![![1, 2]]) := by
  intro y y' h
  have h0 := congrFun h 0
  simp [affinePt] at h0
  funext j
  rw [Subsingleton.elim j 0]
  exact h0

/-- `stabVecN` on the line, evaluated at a digit string `![a, b]`. -/
theorem stabVecN_line (σ : Fin 3) (Q : Fin 1 → Fin 1 → Fin 3) (l : Fin 1 → Fin 3)
    (a b : Fin 3) :
    stabVecN 2 1 ![0, σ] ![![1, 2]] Q l ![a, b]
      = (if ![a, b] = affinePt ![0, σ] ![![1, 2]] ![0]
          then omega3 ^ quadPhase Q l ![0] else 0)
      + (if ![a, b] = affinePt ![0, σ] ![![1, 2]] ![1]
          then omega3 ^ quadPhase Q l ![1] else 0)
      + (if ![a, b] = affinePt ![0, σ] ![![1, 2]] ![2]
          then omega3 ^ quadPhase Q l ![2] else 0) := by
  unfold stabVecN
  rw [sum_fin1_fin3]

set_option linter.flexible false in
theorem v0Vec_isStab : IsStab v0Vec := by
  refine ⟨1, 1, ![0, 0], ![![1, 2]], ![![1]], 0, one_ne_zero, affinePt_line_injective 0, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  have hab : v0Vec idx = v0T3 a b := by
    simp only [v0Vec, hx]
    rfl
  rw [hab, stabVecN_line, one_mul]
  fin_cases a <;> fin_cases b <;>
    simp (config := { decide := true }) [v0T3, quadPhase, omega3_pow_mod]

set_option linter.flexible false in
theorem v1Vec_isStab : IsStab v1Vec := by
  refine ⟨1, 1, ![0, 1], ![![1, 2]], ![![2]], ![1], one_ne_zero, affinePt_line_injective 1, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  have hab : v1Vec idx = v1T3 a b := by
    simp only [v1Vec, hx]
    rfl
  rw [hab, stabVecN_line, one_mul]
  fin_cases a <;> fin_cases b <;>
    simp (config := { decide := true }) [v1T3, quadPhase, omega3_pow_mod]

set_option linter.flexible false in
theorem v2Vec_isStab : IsStab v2Vec := by
  refine ⟨1, 1, ![0, 2], ![![1, 2]], ![![0]], 0, one_ne_zero, affinePt_line_injective 2, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  have hab : v2Vec idx = v2T3 a b := by
    simp only [v2Vec, hx]
    rfl
  rw [hab, stabVecN_line, one_mul]
  fin_cases a <;> fin_cases b <;>
    simp (config := { decide := true }) [v2T3, quadPhase]

/-- `|T3⟩^⊗2` of `T3M2Pointwise`, reindexed. -/
noncomputable def t3Vec2 : QutritVec 2 := ofPair t3Amp2

/-- The carry identity, as vectors. -/
theorem t3Vec2_eq :
    t3Vec2 = (1 / 3 : ℂ) • v0Vec + (omega9 / 3) • v1Vec + (omega9 ^ 2 / 3) • v2Vec := by
  funext idx
  simp only [t3Vec2, ofPair, v0Vec, v1Vec, v2Vec, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
    finTwoArrowEquiv_pair, t3_m2_decomposition]
  ring

/-- **χ(|T3⟩^⊗2) ≤ 3 against the concrete stabilizer predicate.** -/
theorem t3_m2_stabRank_le_three :
    Stabilizer.stabRank (IsStab (n := 2)) t3Vec2 ≤ 3 := by
  classical
  refine le_trans
    (Stabilizer.stabRank_le_of_decomp (S := {v0Vec, v1Vec, v2Vec}) ?_ ?_)
    Finset.card_le_three
  · intro σ hσ
    simp only [Finset.mem_insert, Finset.mem_singleton] at hσ
    rcases hσ with rfl | rfl | rfl
    · exact v0Vec_isStab
    · exact v1Vec_isStab
    · exact v2Vec_isStab
  · rw [t3Vec2_eq]
    have h0 : v0Vec ∈ Submodule.span ℂ (({v0Vec, v1Vec, v2Vec} : Finset (QutritVec 2)) : Set _) :=
      Submodule.subset_span (by simp)
    have h1 : v1Vec ∈ Submodule.span ℂ (({v0Vec, v1Vec, v2Vec} : Finset (QutritVec 2)) : Set _) :=
      Submodule.subset_span (by simp)
    have h2 : v2Vec ∈ Submodule.span ℂ (({v0Vec, v1Vec, v2Vec} : Finset (QutritVec 2)) : Set _) :=
      Submodule.subset_span (by simp)
    exact Submodule.add_mem _ (Submodule.add_mem _ (Submodule.smul_mem _ _ h0)
      (Submodule.smul_mem _ _ h1)) (Submodule.smul_mem _ _ h2)

/-- The single-copy amplitude is `ω₉^a / √3`. -/
theorem t3Amp1_eq (a : Fin 3) : t3Amp1 a = omega9 ^ (a : ℕ) / (Real.sqrt 3 : ℂ) := by
  fin_cases a <;> simp [t3Amp1]

/-- `t3Vec2` is the normalised target of `T3GaloisM`. -/
theorem t3Vec2_eq_target : t3Vec2 = t3TargetM 2 := by
  funext idx
  simp only [t3Vec2, ofPair, finTwoArrowEquiv_pair, t3Amp2, t3TargetM, tConjM, Pi.smul_apply,
    smul_eq_mul, t3Amp1_eq, digitSum, galoisExp, Fin.sum_univ_two]
  push_cast
  simp only [one_mul]
  rw [pow_add]
  ring

/-- **χ(|T3⟩^⊗2) = 3**, both directions machine-checked against `IsStab`. -/
theorem t3_m2_stabRank_eq_three :
    Stabilizer.stabRank (IsStab (n := 2)) (t3TargetM 2) = 3 := by
  rw [← t3Vec2_eq_target]
  refine le_antisymm t3_m2_stabRank_le_three ?_
  have := t3_m2_stabRank_gt_two
  rw [← t3Vec2_eq_target] at this
  omega

end StabRank
