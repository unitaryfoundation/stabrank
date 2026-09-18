/-
The Norrell four-copy upper bound as a theorem about `Stabilizer.stabRank`.

`NorrellM4Pointwise` proves `|N⟩^⊗4 = Σ α_j |S_j⟩` with seven terms. Here each
term is shown to be `IsStab` in the concrete sense. The phases are sums of
indicators `[y_i = 2]`, and `[y = 2] = 2y² + y` mod 3, so each is a diagonal
quadratic form plus a linear form on the flat's coordinates:

  S₀: flat `y₁ = 2`, coordinates `(y₀, y₂, y₃)`, phase `2[y₀=2] + [y₂=2] + 2[y₃=2]`
  S₁: full support, phase `2[y₀=2] + 2[y₁=2] + [y₂=2] + [y₃=2]`
  S₂: flat `y₀ = y₃ = 2`, coordinates `(y₁, y₂)`, phase `[y₁=2] + [y₂=2]`
  S₃: flat `y₂ = y₃ = 2`, coordinates `(y₀, y₁)`, phase `2[y₀=2] + [y₁=2]`
  S₄: flat `y₀ = 2`, coordinates `(y₁, y₂, y₃)`, phase `2[y₁=2] + 2[y₂=2] + [y₃=2]`
  S₅: flat `y₁ = y₂ = 2`, coordinates `(y₀, y₃)`, phase `[y₀=2] + 2[y₃=2]`
  S₆: full support, phase `0`.

With this every upper bound the board holds at the lean tier is a statement
about `stabRank` with the concrete predicate.
-/
import LeanProofs.M3StabRank
import LeanProofs.NorrellM4Pointwise

namespace StabRank

open Stabilizer

/-! ### Generic lemmas -/

noncomputable def ofQuad (f : Fin 3 × Fin 3 × Fin 3 × Fin 3 → ℂ) : QutritVec 4 :=
  fun idx => f (digits 4 idx 0, digits 4 idx 1, digits 4 idx 2, digits 4 idx 3)

theorem fin4_eta (x : Fin 4 → Fin 3) : x = ![x 0, x 1, x 2, x 3] := by
  funext i
  fin_cases i <;> rfl

/-- The 27 strings of `Fin 3 → Fin 3`. -/
theorem sum_fin3_fin3 (f : (Fin 3 → Fin 3) → ℂ) :
    ∑ y : Fin 3 → Fin 3, f y
      = f ![0, 0, 0] + f ![0, 0, 1] + f ![0, 0, 2] + f ![0, 1, 0] + f ![0, 1, 1] + f ![0, 1, 2]
        + f ![0, 2, 0] + f ![0, 2, 1] + f ![0, 2, 2]
        + f ![1, 0, 0] + f ![1, 0, 1] + f ![1, 0, 2] + f ![1, 1, 0] + f ![1, 1, 1] + f ![1, 1, 2]
        + f ![1, 2, 0] + f ![1, 2, 1] + f ![1, 2, 2]
        + f ![2, 0, 0] + f ![2, 0, 1] + f ![2, 0, 2] + f ![2, 1, 0] + f ![2, 1, 1] + f ![2, 1, 2]
        + f ![2, 2, 0] + f ![2, 2, 1] + f ![2, 2, 2] := by
  rw [show (Finset.univ : Finset (Fin 3 → Fin 3))
      = {![0, 0, 0], ![0, 0, 1], ![0, 0, 2], ![0, 1, 0], ![0, 1, 1], ![0, 1, 2],
         ![0, 2, 0], ![0, 2, 1], ![0, 2, 2],
         ![1, 0, 0], ![1, 0, 1], ![1, 0, 2], ![1, 1, 0], ![1, 1, 1], ![1, 1, 2],
         ![1, 2, 0], ![1, 2, 1], ![1, 2, 2],
         ![2, 0, 0], ![2, 0, 1], ![2, 0, 2], ![2, 1, 0], ![2, 1, 1], ![2, 1, 2],
         ![2, 2, 0], ![2, 2, 1], ![2, 2, 2]} by decide]
  simp [Finset.sum_insert, Finset.mem_insert]
  ring

/-- `stabVecN` with `k = 3`, as an explicit 27-term sum. -/
theorem stabVecN_k3 (n : ℕ) (x0 : Fin n → Fin 3) (W : Fin 3 → Fin n → Fin 3)
    (Q : Fin 3 → Fin 3 → Fin 3) (l : Fin 3 → Fin 3) (x : Fin n → Fin 3) :
    stabVecN n 3 x0 W Q l x
      = ∑ y ∈ ({![0, 0, 0], ![0, 0, 1], ![0, 0, 2], ![0, 1, 0], ![0, 1, 1], ![0, 1, 2],
         ![0, 2, 0], ![0, 2, 1], ![0, 2, 2],
         ![1, 0, 0], ![1, 0, 1], ![1, 0, 2], ![1, 1, 0], ![1, 1, 1], ![1, 1, 2],
         ![1, 2, 0], ![1, 2, 1], ![1, 2, 2],
         ![2, 0, 0], ![2, 0, 1], ![2, 0, 2], ![2, 1, 0], ![2, 1, 1], ![2, 1, 2],
         ![2, 2, 0], ![2, 2, 1], ![2, 2, 2]} : Finset (Fin 3 → Fin 3)),
          if x = affinePt x0 W y then omega3 ^ quadPhase Q l y else 0 := by
  unfold stabVecN
  congr 1
  decide

/-- A plane in `F_3^n` with rows of `W` standard basis vectors at `i₀ ≠ i₁`. -/
theorem affinePt_plane_injective' {n : ℕ} {x0 : Fin n → Fin 3} {W : Fin 2 → Fin n → Fin 3}
    (i0 i1 : Fin n) (h00 : W 0 i0 = 1) (h10 : W 1 i0 = 0) (h01 : W 0 i1 = 0)
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

/-- A 3-flat in `F_3^n` with rows of `W` standard basis vectors at distinct
    coordinates `i₀, i₁, i₂`. -/
theorem affinePt_cube_injective {n : ℕ} {x0 : Fin n → Fin 3} {W : Fin 3 → Fin n → Fin 3}
    (i0 i1 i2 : Fin n)
    (h00 : W 0 i0 = 1) (h10 : W 1 i0 = 0) (h20 : W 2 i0 = 0)
    (h01 : W 0 i1 = 0) (h11 : W 1 i1 = 1) (h21 : W 2 i1 = 0)
    (h02 : W 0 i2 = 0) (h12 : W 1 i2 = 0) (h22 : W 2 i2 = 1) :
    Function.Injective (affinePt x0 W) := by
  intro y y' h
  have e0 := congrFun h i0
  have e1 := congrFun h i1
  have e2 := congrFun h i2
  simp only [affinePt, Fin.sum_univ_three, h00, h10, h20, h01, h11, h21, h02, h12, h22,
    mul_one, mul_zero, add_zero, zero_add] at e0 e1 e2
  funext j
  fin_cases j
  · exact add_left_cancel e0
  · exact add_left_cancel e1
  · exact add_left_cancel e2

theorem mem_span_seven {a b c d e f g v : QutritVec 4} (α β γ δ ε ζ η : ℂ)
    (h : v = α • a + β • b + γ • c + δ • d + ε • e + ζ • f + η • g) :
    v ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set (QutritVec 4)) := by
  classical
  rw [h]
  have ha : a ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set _) :=
    Submodule.subset_span (by simp)
  have hb : b ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set _) :=
    Submodule.subset_span (by simp)
  have hc : c ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set _) :=
    Submodule.subset_span (by simp)
  have hd : d ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set _) :=
    Submodule.subset_span (by simp)
  have he : e ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set _) :=
    Submodule.subset_span (by simp)
  have hf : f ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set _) :=
    Submodule.subset_span (by simp)
  have hg : g ∈ Submodule.span ℂ (({a, b, c, d, e, f, g} : Finset (QutritVec 4)) : Set _) :=
    Submodule.subset_span (by simp)
  exact Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _ (Submodule.add_mem _
    (Submodule.add_mem _ (Submodule.add_mem _ (Submodule.smul_mem _ _ ha)
      (Submodule.smul_mem _ _ hb)) (Submodule.smul_mem _ _ hc)) (Submodule.smul_mem _ _ hd))
    (Submodule.smul_mem _ _ he)) (Submodule.smul_mem _ _ hf)) (Submodule.smul_mem _ _ hg)

theorem stabRank_le_seven_of {a b c d e f g v : QutritVec 4}
    (ha : IsStab a) (hb : IsStab b) (hc : IsStab c) (hd : IsStab d) (he : IsStab e)
    (hf : IsStab f) (hg : IsStab g) (α β γ δ ε ζ η : ℂ)
    (h : v = α • a + β • b + γ • c + δ • d + ε • e + ζ • f + η • g) :
    Stabilizer.stabRank (IsStab (n := 4)) v ≤ 7 := by
  classical
  have hcard : ({a, b, c, d, e, f, g} : Finset (QutritVec 4)).card ≤ 7 :=
    le_trans (Finset.card_insert_le _ _) (Nat.succ_le_succ
      (le_trans (Finset.card_insert_le _ _) (Nat.succ_le_succ
        (le_trans (Finset.card_insert_le _ _) (Nat.succ_le_succ Finset.card_le_four)))))
  refine le_trans (Stabilizer.stabRank_le_of_decomp (S := {a, b, c, d, e, f, g}) ?_
    (mem_span_seven α β γ δ ε ζ η h)) hcard
  intro σ hσ
  simp only [Finset.mem_insert, Finset.mem_singleton] at hσ
  rcases hσ with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ha
  · exact hb
  · exact hc
  · exact hd
  · exact he
  · exact hf
  · exact hg

private theorem inv9_ne : (1 / 9 : ℂ) ≠ 0 := by norm_num

private theorem inv3_ne4 : (1 / 3 : ℂ) ≠ 0 := by norm_num

private theorem inv_3sqrt3_ne4 : (1 / (3 * (Real.sqrt 3 : ℂ))) ≠ 0 := by
  have : (Real.sqrt 3 : ℂ) ≠ 0 := by exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num)
  exact one_div_ne_zero (mul_ne_zero (by norm_num) this)

/-! ### The seven terms -/

noncomputable def norrellVec4 : QutritVec 4 := ofQuad norrellAmp4
noncomputable def sN0Vec4 : QutritVec 4 := ofQuad sN0'
noncomputable def sN1Vec4 : QutritVec 4 := ofQuad sN1'
noncomputable def sN2Vec4 : QutritVec 4 := ofQuad sN2'
noncomputable def sN3Vec4 : QutritVec 4 := ofQuad sN3'
noncomputable def sN4Vec4 : QutritVec 4 := ofQuad sN4'
noncomputable def sN5Vec4 : QutritVec 4 := ofQuad sN5'
noncomputable def sN6Vec4 : QutritVec 4 := ofQuad sN6'

/-- Rows for the flat `y₁ = c`: coordinates `(y₀, y₂, y₃)`. -/
def w4_023 : Fin 3 → Fin 4 → Fin 3 := ![![1, 0, 0, 0], ![0, 0, 1, 0], ![0, 0, 0, 1]]
/-- Rows for the flat `y₀ = c`: coordinates `(y₁, y₂, y₃)`. -/
def w4_123 : Fin 3 → Fin 4 → Fin 3 := ![![0, 1, 0, 0], ![0, 0, 1, 0], ![0, 0, 0, 1]]
/-- Rows for `y₀, y₃` fixed: coordinates `(y₁, y₂)`. -/
def w4_12 : Fin 2 → Fin 4 → Fin 3 := ![![0, 1, 0, 0], ![0, 0, 1, 0]]
/-- Rows for `y₂, y₃` fixed: coordinates `(y₀, y₁)`. -/
def w4_01 : Fin 2 → Fin 4 → Fin 3 := ![![1, 0, 0, 0], ![0, 1, 0, 0]]
/-- Rows for `y₁, y₂` fixed: coordinates `(y₀, y₃)`. -/
def w4_03 : Fin 2 → Fin 4 → Fin 3 := ![![1, 0, 0, 0], ![0, 0, 0, 1]]

set_option linter.flexible false in
set_option maxHeartbeats 3200000 in -- 81 points, 27-term sums
theorem sN0Vec4_isStab : IsStab sN0Vec4 := by
  refine ⟨1 / (3 * (Real.sqrt 3 : ℂ)), 3, ![0, 2, 0, 0], w4_023,
    ![![1, 0, 0], ![0, 2, 0], ![0, 0, 1]], ![2, 1, 2], inv_3sqrt3_ne4,
    affinePt_cube_injective 0 2 3 rfl rfl rfl rfl rfl rfl rfl rfl rfl, ?_⟩
  funext idx
  generalize hx : digits 4 idx = x
  obtain ⟨a, b, c, d, rfl⟩ : ∃ a b c d, x = ![a, b, c, d] := ⟨x 0, x 1, x 2, x 3, fin4_eta x⟩
  have hab : sN0Vec4 idx = sN0' (a, b, c, d) := by
    simp only [sN0Vec4, ofQuad, hx]
    rfl
  rw [hab, stabVecN_k3]
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d
  all_goals simp (config := { decide := true }) [sN0', qN0, isTwo, quadPhase, Fin.sum_univ_three,
    omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
set_option maxHeartbeats 3200000 in -- 81 points, 27-term sums
theorem sN4Vec4_isStab : IsStab sN4Vec4 := by
  refine ⟨1 / (3 * (Real.sqrt 3 : ℂ)), 3, ![2, 0, 0, 0], w4_123,
    ![![1, 0, 0], ![0, 1, 0], ![0, 0, 2]], ![2, 2, 1], inv_3sqrt3_ne4,
    affinePt_cube_injective 1 2 3 rfl rfl rfl rfl rfl rfl rfl rfl rfl, ?_⟩
  funext idx
  generalize hx : digits 4 idx = x
  obtain ⟨a, b, c, d, rfl⟩ : ∃ a b c d, x = ![a, b, c, d] := ⟨x 0, x 1, x 2, x 3, fin4_eta x⟩
  have hab : sN4Vec4 idx = sN4' (a, b, c, d) := by
    simp only [sN4Vec4, ofQuad, hx]
    rfl
  rw [hab, stabVecN_k3]
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d
  all_goals simp (config := { decide := true }) [sN4', qN4, isTwo, quadPhase, Fin.sum_univ_three,
    omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
set_option maxHeartbeats 1600000 in -- 81 points, nine-term sums
theorem sN2Vec4_isStab : IsStab sN2Vec4 := by
  refine ⟨1 / 3, 2, ![2, 0, 0, 2], w4_12, ![![2, 0], ![0, 2]], ![1, 1], inv3_ne4,
    affinePt_plane_injective' 1 2 rfl rfl rfl rfl, ?_⟩
  funext idx
  generalize hx : digits 4 idx = x
  obtain ⟨a, b, c, d, rfl⟩ : ∃ a b c d, x = ![a, b, c, d] := ⟨x 0, x 1, x 2, x 3, fin4_eta x⟩
  have hab : sN2Vec4 idx = sN2' (a, b, c, d) := by
    simp only [sN2Vec4, ofQuad, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d
  all_goals simp (config := { decide := true }) [sN2', qN2, isTwo, quadPhase, omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
set_option maxHeartbeats 1600000 in -- 81 points, nine-term sums
theorem sN3Vec4_isStab : IsStab sN3Vec4 := by
  refine ⟨1 / 3, 2, ![0, 0, 2, 2], w4_01, ![![1, 0], ![0, 2]], ![2, 1], inv3_ne4,
    affinePt_plane_injective' 0 1 rfl rfl rfl rfl, ?_⟩
  funext idx
  generalize hx : digits 4 idx = x
  obtain ⟨a, b, c, d, rfl⟩ : ∃ a b c d, x = ![a, b, c, d] := ⟨x 0, x 1, x 2, x 3, fin4_eta x⟩
  have hab : sN3Vec4 idx = sN3' (a, b, c, d) := by
    simp only [sN3Vec4, ofQuad, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d
  all_goals simp (config := { decide := true }) [sN3', qN3, isTwo, quadPhase, omega3_pow_mod]
  all_goals ring

set_option linter.flexible false in
set_option maxHeartbeats 1600000 in -- 81 points, nine-term sums
theorem sN5Vec4_isStab : IsStab sN5Vec4 := by
  refine ⟨1 / 3, 2, ![0, 2, 2, 0], w4_03, ![![2, 0], ![0, 1]], ![1, 2], inv3_ne4,
    affinePt_plane_injective' 0 3 rfl rfl rfl rfl, ?_⟩
  funext idx
  generalize hx : digits 4 idx = x
  obtain ⟨a, b, c, d, rfl⟩ : ∃ a b c d, x = ![a, b, c, d] := ⟨x 0, x 1, x 2, x 3, fin4_eta x⟩
  have hab : sN5Vec4 idx = sN5' (a, b, c, d) := by
    simp only [sN5Vec4, ofQuad, hx]
    rfl
  rw [hab, stabVecN_k2]
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d
  all_goals simp (config := { decide := true }) [sN5', qN5, isTwo, quadPhase, omega3_pow_mod]
  all_goals ring

/-- `2[y₀=2] + 2[y₁=2] + [y₂=2] + [y₃=2]` as a diagonal form plus linear part. -/
def qN1Mat4 : Fin 4 → Fin 4 → Fin 3 :=
  ![![1, 0, 0, 0], ![0, 1, 0, 0], ![0, 0, 2, 0], ![0, 0, 0, 2]]

theorem qN1_phase_mod4 (a b c d : Fin 3) :
    qN1 a b c d % 3 = quadPhase qN1Mat4 ![2, 2, 1, 1] ![a, b, c, d] % 3 := by
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d <;> decide

theorem qN1_phase_mod4' (x : Fin 4 → Fin 3) :
    qN1 (x 0) (x 1) (x 2) (x 3) % 3 = quadPhase qN1Mat4 ![2, 2, 1, 1] x % 3 := by
  obtain ⟨a, b, c, d, rfl⟩ : ∃ a b c d, x = ![a, b, c, d] := ⟨x 0, x 1, x 2, x 3, fin4_eta x⟩
  simpa using qN1_phase_mod4 a b c d

theorem sN1Vec4_isStab : IsStab sN1Vec4 := by
  refine ⟨1 / 9, 4, 0, idW 4, qN1Mat4, ![2, 2, 1, 1], inv9_ne, affinePt_id_injective 4, ?_⟩
  funext idx
  generalize hx : digits 4 idx = x
  simp only [sN1Vec4, ofQuad, hx, stabVecN_id, sN1']
  rw [omega3_pow_eq_of_mod (qN1_phase_mod4' x)]
  ring

set_option linter.flexible false in
theorem sN6Vec4_isStab : IsStab sN6Vec4 := by
  refine ⟨1 / 9, 4, 0, idW 4, 0, 0, inv9_ne, affinePt_id_injective 4, ?_⟩
  funext idx
  generalize hx : digits 4 idx = x
  simp [sN6Vec4, ofQuad, stabVecN_id, sN6', quadPhase]

theorem norrellVec4_eq :
    norrellVec4 = alphaN0 • sN0Vec4 + alphaN1 • sN1Vec4 + alphaN1 • sN2Vec4
      + alphaN1 • sN3Vec4 + alphaN4 • sN4Vec4 + alphaN1 • sN5Vec4 + alphaN6 • sN6Vec4 := by
  funext idx
  simp only [norrellVec4, sN0Vec4, sN1Vec4, sN2Vec4, sN3Vec4, sN4Vec4, sN5Vec4, sN6Vec4,
    ofQuad, Pi.add_apply, Pi.smul_apply, smul_eq_mul, norrell_m4_decomposition]

/-- **χ(|N⟩^⊗4) ≤ 7 against the concrete stabilizer predicate.** -/
theorem norrell_m4_stabRank_le_seven :
    Stabilizer.stabRank (IsStab (n := 4)) norrellVec4 ≤ 7 :=
  stabRank_le_seven_of sN0Vec4_isStab sN1Vec4_isStab sN2Vec4_isStab sN3Vec4_isStab
    sN4Vec4_isStab sN5Vec4_isStab sN6Vec4_isStab _ _ _ _ _ _ _ norrellVec4_eq

end StabRank
