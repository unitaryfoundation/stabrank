/-
`χ(|T5⟩) = 3` against `stabRankP 5`.

Upper bound: the witness of `bounds/T5-m1-upper-3.json`, the points `|0⟩`,
`|1⟩` and the full-support state `Σ_y ω₅^(4y² + 4y)|y⟩`, entered as
`stabTerm`s. With the `1/√5` normalisations pulled out, the coefficients are
`(2 + ω + ω² + ω³)/√5`, `(ω - ω²)/√5` and `-(1 + ω + ω² + ω³)/√5`
(the file's nested radicals, evaluated exactly in `ℚ(ω₅)`), and the identity
is decided at each of the five digits: exponents are reduced mod 5, `ω⁴` is
rewritten into the lower powers, and `ring` closes the reduced form.

Lower bound: every `IsStabP 5` vector on one ququint is a scalar multiple of
one of the 30 shapes of `T5Minors` (`isStabP_one_ququint`): the flat
parametrisation is injective, so it has at most one generator; with none it
is a point, and with one, `x = x₀ + w y` with `w ≠ 0`, the phase
`ω^(Q y² + l y)` is `ω^(q' x² + l' x)` up to a constant after the substitution
`y = (x - x₀)/w`. Then `t5v_not_mem_span_smul` says no two of them span
`√5 |T5⟩`, and the reduction lemma of `Stabilizer/Rank.lean` gives
`stabRankP 5 |T5⟩ > 2`. This is `verify_challenge/cert_t5_m1_rank2.py`,
including its enumeration of the stabilizer states, checked end to end from
the definition of `IsStabP`.
-/
import LeanProofs.T5Minors

namespace StabRank

open Stabilizer T5

/-- The single-ququint index, `Fin (5 ^ 1) ≃ ZMod 5`. -/
def idx5 : Fin (5 ^ 1) ≃ ZMod 5 := (digitsP 5 1).trans (Equiv.funUnique (Fin 1) (ZMod 5))

theorem idx5_apply (idx : Fin (5 ^ 1)) : idx5 idx = digitsP 5 1 idx 0 := rfl

/-! ### χ(|T5⟩) ≤ 3 -/

noncomputable def t5T1_0 : Fin (5 ^ 1) → ℂ :=
  stabTerm 5 1 0 ![0] (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i)
noncomputable def t5T1_1 : Fin (5 ^ 1) → ℂ :=
  stabTerm 5 1 0 ![1] (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i)
noncomputable def t5T1_2 : Fin (5 ^ 1) → ℂ :=
  stabTerm 5 1 1 ![0] ![![1]] ![![4]] ![4]

theorem t5T1_0_isStab : IsStabP 5 t5T1_0 :=
  isStabP_stabTerm 5 1 0 _ _ _ _ (Function.injective_of_subsingleton _)
theorem t5T1_1_isStab : IsStabP 5 t5T1_1 :=
  isStabP_stabTerm 5 1 0 _ _ _ _ (Function.injective_of_subsingleton _)
theorem t5T1_2_isStab : IsStabP 5 t5T1_2 :=
  isStabP_stabTerm 5 1 1 _ _ _ _ (affinePtP_injective_of_pivots _ _ ![0] (by decide +kernel))

/-- The coefficients of `bounds/T5-m1-upper-3.json` on the unnormalised
    terms, as elements of `ℤ[ω₅]/√5`. -/
noncomputable def t5Coef1 : Fin 3 → ℂ :=
  ![(2 + omega5 + omega5 ^ 2 + omega5 ^ 3) / (Real.sqrt 5 : ℂ),
    (omega5 - omega5 ^ 2) / (Real.sqrt 5 : ℂ),
    (-1 - omega5 - omega5 ^ 2 - omega5 ^ 3) / (Real.sqrt 5 : ℂ)]

set_option linter.flexible false in
set_option linter.unusedSimpArgs false in
theorem t5Vec1_eq : t5Vec 1 = ∑ j : Fin 3, t5Coef1 j • ![t5T1_0, t5T1_1, t5T1_2] j := by
  funext idx
  simp only [Fin.sum_univ_three, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, t5Vec, t5Coef1,
    t5T1_0, t5T1_1, t5T1_2, stabTerm, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons, Fin.prod_univ_one]
  generalize hx : digitsP 5 1 idx = x
  obtain ⟨a, rfl⟩ : ∃ a, x = ![a] := ⟨x 0, fin1_eta_z5 x⟩
  simp only [stabVecP_zero, stabVecP_five_k1]
  rcases zmod5_cases a with rfl | rfl | rfl | rfl | rfl <;>
    simp (config := { decide := true }) [t5Amp, cubeExp_zero, cubeExp_one, cubeExp_two,
      cubeExp_three, cubeExp_four, quadPhaseP, affinePtP, Fin.sum_univ_one, zmod5_val_zero,
      zmod5_val_one, zmod5_val_two, zmod5_val_three, zmod5_val_four, zmodP5_val_four,
      stabPeriod_five_div, zeta_five, omega5_pow_reduce]
  all_goals (try ring_nf)
  all_goals (try simp only [omega5_pow_reduce, omega5_pow_four, Nat.reduceSub, Nat.reduceLeDiff,
    pow_zero])
  all_goals ring

/-- **χ(|T5⟩) ≤ 3 against `IsStabP 5`.** -/
theorem t5_m1_stabRankP_le_three : stabRankP 5 (t5Vec 1) ≤ 3 :=
  stabRankP_le_of_terms 5 ![t5T1_0, t5T1_1, t5T1_2] t5Coef1
    (fun j => by fin_cases j; exacts [t5T1_0_isStab, t5T1_1_isStab, t5T1_2_isStab]) t5Vec1_eq

/-! ### Every single-ququint stabilizer state is a shape -/

/-- An `IsStabP 5` vector on one ququint is a scalar multiple of a shape
    vector. -/
theorem isStabP_one_ququint {v : Fin (5 ^ 1) → ℂ} (hv : IsStabP 5 v) :
    ∃ (c : ℂ) (s : Shape), v = fun idx => c * s.vec (idx5 idx) := by
  obtain ⟨c, k, x0, W, Q, l, -, hinj, rfl⟩ := hv
  have hcard := Fintype.card_le_of_injective _ hinj
  simp only [Fintype.card_fun, ZMod.card, Fintype.card_fin] at hcard
  have hk : k ≤ 1 := (Nat.pow_le_pow_iff_right (by norm_num)).mp hcard
  interval_cases k
  · -- a point
    refine ⟨c, Shape.pt (x0 0), ?_⟩
    have hW : W = fun j => Fin.elim0 j := funext fun j => Fin.elim0 j
    have hQ : Q = fun i _ => Fin.elim0 i := funext fun i => Fin.elim0 i
    have hl : l = fun i => Fin.elim0 i := funext fun i => Fin.elim0 i
    subst hW hQ hl
    funext idx
    rw [stabVecP_zero]
    simp only [Shape.vec, Shape.exp, idx5_apply, fin1_ext_iff]
    split_ifs <;> simp [termVal]
  · -- a line: full support
    set w := W 0 0 with hw
    have hw0 : w ≠ 0 := by
      intro h0
      have h := hinj (a₁ := fun _ => 0) (a₂ := fun _ => 1) ?_
      · exact absurd (congrFun h 0) (by decide)
      · funext i
        rw [Fin.fin_one_eq_zero i]
        simp [affinePtP, ← hw, h0]
    have hval : ∀ x : Fin 1 → ZMod 5, stabVecP 5 1 1 x0 W Q l x
        = zeta 5 ^ quadPhaseP Q l (fun _ => (x 0 - x0 0) * w⁻¹) := by
      intro x
      unfold stabVecP
      rw [Finset.sum_eq_single (fun _ => (x 0 - x0 0) * w⁻¹)]
      · rw [if_pos]
        funext i
        rw [Fin.fin_one_eq_zero i]
        simp only [affinePtP, Fin.sum_univ_one, ← hw]
        field_simp
        ring
      · intro y _ hy
        rw [if_neg]
        intro hxy
        apply hy
        have h := congrFun hxy 0
        simp only [affinePtP, Fin.sum_univ_one, ← hw] at h
        funext i
        rw [Fin.fin_one_eq_zero i, h]
        field_simp
        ring
      · simp
    set L : ZMod 5 := ((l 0).val : ZMod 5) with hL
    refine ⟨c * omega5 ^ (Q 0 0 * x0 0 ^ 2 * w⁻¹ ^ 2 - L * x0 0 * w⁻¹).val,
      Shape.full (Q 0 0 * w⁻¹ ^ 2) (L * w⁻¹ - 2 * Q 0 0 * x0 0 * w⁻¹ ^ 2), ?_⟩
    funext idx
    rw [hval, zeta_five_pow]
    simp only [Shape.vec, Shape.exp, termVal, idx5_apply, quadPhaseP, Fin.sum_univ_one,
      stabPeriod_five_div, one_mul]
    conv_rhs => rw [mul_assoc, ← omega5_pow_val_add]
    congr 3
    push_cast [ZMod.natCast_zmod_val, ← hL]
    field_simp
    ring

/-! ### χ(|T5⟩) ≥ 3 -/

/-- Reindexing along `idx5`, as a linear equivalence. -/
noncomputable def reindex5 : (Fin (5 ^ 1) → ℂ) ≃ₗ[ℂ] (ZMod 5 → ℂ) :=
  LinearEquiv.funCongrLeft ℂ ℂ idx5.symm

theorem reindex5_apply (v : Fin (5 ^ 1) → ℂ) (x : ZMod 5) : reindex5 v x = v (idx5.symm x) := rfl

theorem reindex5_shape (c : ℂ) (s : Shape) :
    reindex5 (fun idx => c * s.vec (idx5 idx)) = c • s.vec := by
  funext x
  change c * s.vec (idx5 (idx5.symm x)) = c * s.vec x
  rw [Equiv.apply_symm_apply]

theorem reindex5_t5Vec : reindex5 (t5Vec 1) = (Real.sqrt 5 : ℂ)⁻¹ • t5v := by
  funext x
  simp only [reindex5_apply, t5Vec, Fin.prod_univ_one, t5Amp, Pi.smul_apply, smul_eq_mul, t5v,
    termVal, t5exp]
  rw [← idx5_apply, Equiv.apply_symm_apply]
  unfold cubeExp
  ring

/-- **χ(|T5⟩) ≥ 3 against `IsStabP 5`**: no two stabilizer states span the
    T5 state. -/
theorem t5_m1_stabRankP_gt_two : stabRankP 5 (t5Vec 1) > 2 := by
  classical
  refine Stabilizer.stabRank_gt_of_no_decomp_le _ _ 2 (decompCardsP_nonempty 5 1 _) ?_
  intro S hcard hstab hmem
  -- push the span through the reindexing
  have hmem' : reindex5 (t5Vec 1) ∈ Submodule.span ℂ ((reindex5 : (Fin (5 ^ 1) → ℂ) →ₗ[ℂ]
      (ZMod 5 → ℂ)) '' (S : Set (Fin (5 ^ 1) → ℂ))) := by
    rw [← Submodule.map_span]
    exact Submodule.mem_map_of_mem hmem
  rw [reindex5_t5Vec, Submodule.smul_mem_iff _ (inv_ne_zero sqrt5_ne_zero_c)] at hmem'
  have h012 : S.card = 0 ∨ S.card = 1 ∨ S.card = 2 := by omega
  rcases h012 with h0 | h1 | h2
  · rw [Finset.card_eq_zero] at h0
    subst h0
    simp only [Finset.coe_empty, Set.image_empty, Submodule.span_empty, Submodule.mem_bot] at hmem'
    exact absurd (congrFun hmem' 0) (by simp [t5v, termVal, t5exp])
  · obtain ⟨u, rfl⟩ := Finset.card_eq_one.mp h1
    obtain ⟨c, s, rfl⟩ := isStabP_one_ququint (hstab u (Finset.mem_singleton_self u))
    rw [Finset.coe_singleton, Set.image_singleton, LinearEquiv.coe_coe, reindex5_shape,
      ← Set.pair_eq_singleton] at hmem'
    exact t5v_not_mem_span_smul c c s s hmem'
  · obtain ⟨u, v, -, rfl⟩ := Finset.card_eq_two.mp h2
    obtain ⟨c, s, rfl⟩ := isStabP_one_ququint (hstab u (by simp))
    obtain ⟨c', s', rfl⟩ := isStabP_one_ququint (hstab v (by simp))
    rw [Finset.coe_pair, Set.image_pair, LinearEquiv.coe_coe, reindex5_shape,
      reindex5_shape] at hmem'
    exact t5v_not_mem_span_smul c c' s s' hmem'

/-- **χ(|T5⟩) = 3**, both directions machine-checked against `IsStabP 5`. -/
theorem t5_m1_stabRankP_eq_three : stabRankP 5 (t5Vec 1) = 3 :=
  le_antisymm t5_m1_stabRankP_le_three t5_m1_stabRankP_gt_two

end StabRank
