/-
The descent step of the T3 Galois lower bound.

`T3Galois.three_le_of_span_galois_orbit` proves `3 ≤ r` from the hypothesis
that the span of `r` vectors contains all three Galois conjugates of the T3
amplitude vector. This file discharges that hypothesis for vectors over
`ℤ[ω₃]`, which by `StabDef.stabVec_mem_Zomega3` includes every qutrit
stabilizer state up to a positive real normalisation.

The argument. Let `K₉ = ℚ(ω₉) ⊂ ℂ`. If `t₁ := tConj 0` lies in the `ℂ`-span
of vectors `s_j` with entries in `K₉`, then the coefficients can be taken in
`K₉` (`mem_span_descend`: otherwise a `K₉`-linear functional kills the
`K₉`-span and not `t₁`, and the same row vector gives a `ℂ`-linear functional
doing the same over `ℂ`). For `e ∈ {1, 4, 7}` there is a `ℚ`-algebra
embedding `σ_e : K₉ → ℂ` with `σ_e(ω₉) = ω₉^e` (`galEmb`, from the power
basis of `ℚ(ω₉)` and the fact that `ω₉^e` is a root of the ninth cyclotomic
polynomial, the minimal polynomial of `ω₉`). Since `e ≡ 1 (mod 3)`, `σ_e`
fixes `ω₃ = ω₉³` and hence every entry of every `s_j`, while it carries
`t₁` to `tConj a`. Applying `σ_e` to the coefficients of the `K₉`-linear
decomposition of `t₁` therefore gives a `ℂ`-linear decomposition of `tConj a`
over the same `s_j`, so `tConj a` is in the span.

No Galois group or `IsCyclotomicExtension` instance is used; an embedding
into `ℂ` is all the argument needs.
-/
import LeanProofs.T3Galois
import LeanProofs.StabDef
import Mathlib.FieldTheory.IntermediateField.Adjoin.Basic
import Mathlib.RingTheory.Polynomial.Cyclotomic.Roots
import Mathlib.LinearAlgebra.Dual.Lemmas

namespace StabRank

open Complex

/-! ### `ℚ(ω₉)` as a subfield of `ℂ` -/

open IntermediateField in
/-- `ℚ(ω₉)` as an intermediate field of `ℂ` over `ℚ`. -/
noncomputable abbrev K9 : IntermediateField ℚ ℂ := ℚ⟮omega9⟯

theorem omega9_mem_K9 : omega9 ∈ K9 :=
  IntermediateField.mem_adjoin_simple_self ℚ omega9

theorem omega3_mem_K9 : omega3 ∈ K9 := by
  rw [← omega9_cube]
  exact pow_mem omega9_mem_K9 3

theorem tConj_mem_K9 (a i : Fin 3) : tConj a i ∈ K9 :=
  pow_mem omega9_mem_K9 _

theorem overZomega3_mem_K9 {ι : Type*} {v : ι → ℂ} (hv : OverZomega3 v) (i : ι) :
    v i ∈ K9 := by
  obtain ⟨a, b, hab⟩ := hv i
  rw [hab]
  exact add_mem (intCast_mem K9 a) (mul_mem (intCast_mem K9 b) omega3_mem_K9)

/-! ### Descent of coefficients to a subfield -/

set_option linter.unusedFintypeInType false in
/-- If a vector with entries in `K` lies in the `ℂ`-span of vectors with
    entries in `K`, its coefficients can be taken in `K`. -/
theorem mem_span_descend {ι κ : Type*} [Fintype ι] [Fintype κ]
    (K : IntermediateField ℚ ℂ)
    (s : κ → (ι → ℂ)) (hs : ∀ j i, s j i ∈ K)
    (v : ι → ℂ) (hv : ∀ i, v i ∈ K)
    (h : v ∈ Submodule.span ℂ (Set.range s)) :
    ∃ c : κ → K, ∑ j, ((c j : ℂ)) • s j = v := by
  classical
  let s' : κ → (ι → K) := fun j i => ⟨s j i, hs j i⟩
  let v' : ι → K := fun i => ⟨v i, hv i⟩
  by_cases hv' : v' ∈ Submodule.span K (Set.range s')
  · obtain ⟨c, hc⟩ := (Submodule.mem_span_range_iff_exists_fun _).mp hv'
    refine ⟨c, funext fun i => ?_⟩
    have := congrArg (fun w : ι → K => ((w i : K) : ℂ)) hc
    simpa [s', v', Finset.sum_apply, Pi.smul_apply, smul_eq_mul] using this
  · exfalso
    obtain ⟨f, hfv, hfs⟩ :=
      Submodule.exists_dual_map_eq_bot_of_notMem hv' inferInstance
    let w : ι → K := fun i => f (fun j => if i = j then 1 else 0)
    have hf : ∀ x : ι → K, f x = ∑ i, x i * w i := fun x => by
      rw [LinearMap.pi_apply_eq_sum_univ]
      simp [w, smul_eq_mul]
    let F : (ι → ℂ) →ₗ[ℂ] ℂ := ∑ i, ((w i : ℂ)) • LinearMap.proj i
    have hF : ∀ x : ι → ℂ, F x = ∑ i, x i * (w i : ℂ) := fun x => by
      simp [F, LinearMap.sum_apply, LinearMap.smul_apply, LinearMap.proj_apply, mul_comm]
    have hFs : ∀ j, F (s j) = 0 := fun j => by
      have hmem : f (s' j) ∈ (Submodule.span K (Set.range s')).map f :=
        Submodule.mem_map_of_mem (Submodule.subset_span ⟨j, rfl⟩)
      rw [hfs, Submodule.mem_bot, hf] at hmem
      rw [hF]
      have := congrArg (fun z : K => (z : ℂ)) hmem
      push_cast at this
      simpa [s'] using this
    have hFv : F v ≠ 0 := by
      rw [hF]
      intro h0
      apply hfv
      rw [hf]
      apply Subtype.ext
      push_cast
      simpa [v'] using h0
    have hle : Submodule.span ℂ (Set.range s) ≤ LinearMap.ker F := by
      rw [Submodule.span_le]
      rintro _ ⟨j, rfl⟩
      exact hFs j
    exact hFv (hle h)

/-! ### The embeddings `ℚ(ω₉) → ℂ` with `ω₉ ↦ ω₉^e` -/

theorem omega9_pow_nine : omega9 ^ 9 = 1 := omega9_primitive.pow_eq_one

theorem omega9_isIntegral : IsIntegral ℚ omega9 :=
  ⟨Polynomial.X ^ 9 - Polynomial.C 1, Polynomial.monic_X_pow_sub_C 1 (by norm_num),
    by simp [Polynomial.eval₂_sub, Polynomial.eval₂_X_pow, omega9_pow_nine]⟩

/-- `ω₉^e` is a root of the minimal polynomial of the generator of `ℚ(ω₉)`,
    which is the ninth cyclotomic polynomial. -/
theorem omega9_pow_root (e : ℕ) (he : Nat.Coprime e 9) :
    Polynomial.aeval (omega9 ^ e)
      (minpoly ℚ (IntermediateField.adjoin.powerBasis omega9_isIntegral).gen) = 0 := by
  rw [IntermediateField.adjoin.powerBasis_gen]
  erw [IntermediateField.minpoly_gen]
  rw [← Polynomial.cyclotomic_eq_minpoly_rat omega9_primitive (by norm_num),
    Polynomial.aeval_def, Polynomial.eval₂_eq_eval_map, Polynomial.map_cyclotomic]
  exact (omega9_primitive.pow_of_coprime e he).isRoot_cyclotomic (by norm_num)

/-- The `ℚ`-algebra embedding of `ℚ(ω₉)` into `ℂ` sending `ω₉` to `ω₉^e`,
    for `e` coprime to 9: `ω₉^e` is a primitive ninth root of unity, hence a
    root of the ninth cyclotomic polynomial, which is the minimal polynomial
    of `ω₉` over `ℚ`. -/
noncomputable def galEmb (e : ℕ) (he : Nat.Coprime e 9) : K9 →ₐ[ℚ] ℂ :=
  (IntermediateField.adjoin.powerBasis omega9_isIntegral).lift (omega9 ^ e) (omega9_pow_root e he)

theorem galEmb_omega9 (e : ℕ) (he : Nat.Coprime e 9) :
    galEmb e he ⟨omega9, omega9_mem_K9⟩ = omega9 ^ e := by
  have h := PowerBasis.lift_gen (IntermediateField.adjoin.powerBasis omega9_isIntegral)
    (omega9 ^ e) (omega9_pow_root e he)
  simpa [galEmb, IntermediateField.adjoin.powerBasis_gen,
    IntermediateField.AdjoinSimple.gen] using h

theorem galEmb_omega3 (e : ℕ) (he : Nat.Coprime e 9) (h3 : e % 3 = 1) :
    galEmb e he ⟨omega3, omega3_mem_K9⟩ = omega3 := by
  have hpow : (⟨omega3, omega3_mem_K9⟩ : K9) = ⟨omega9, omega9_mem_K9⟩ ^ 3 :=
    Subtype.ext (by simp [omega9_cube])
  rw [hpow, map_pow, galEmb_omega9, ← pow_mul, mul_comm, pow_mul, omega9_cube,
    pow_eq_pow_mod e omega3_pow_three, h3, pow_one]

/-- `σ_e` fixes `ℤ[ω₃]` pointwise when `e ≡ 1 (mod 3)`. -/
theorem galEmb_fix_Zomega3 (e : ℕ) (he : Nat.Coprime e 9) (h3 : e % 3 = 1) (a b : ℤ)
    (hmem : (a : ℂ) + (b : ℂ) * omega3 ∈ K9) :
    galEmb e he ⟨(a : ℂ) + (b : ℂ) * omega3, hmem⟩ = (a : ℂ) + (b : ℂ) * omega3 := by
  have : (⟨(a : ℂ) + (b : ℂ) * omega3, hmem⟩ : K9)
      = (a : K9) + (b : K9) * ⟨omega3, omega3_mem_K9⟩ := Subtype.ext (by push_cast; rfl)
  rw [this, map_add, map_mul, map_intCast, map_intCast, galEmb_omega3 e he h3]

/-- `σ_e` carries `ω₉^k` to `ω₉^(e k)`. -/
theorem galEmb_omega9_pow (e : ℕ) (he : Nat.Coprime e 9) (k : ℕ) (hmem : omega9 ^ k ∈ K9) :
    galEmb e he ⟨omega9 ^ k, hmem⟩ = omega9 ^ (e * k) := by
  have : (⟨omega9 ^ k, hmem⟩ : K9) = ⟨omega9, omega9_mem_K9⟩ ^ k := Subtype.ext (by simp)
  rw [this, map_pow, galEmb_omega9, ← pow_mul]

/-! ### Assembly -/

/-- **Galois closure of the span.** If `t₁ = tConj 0` lies in the `ℂ`-span of
    vectors over `ℤ[ω₃]`, so does every Galois conjugate `tConj a`. -/
theorem tConj_mem_span_of_overZomega3 {r : ℕ} (s : Fin r → (Fin 3 → ℂ))
    (hs : ∀ j, OverZomega3 (s j))
    (h0 : tConj 0 ∈ Submodule.span ℂ (Set.range s)) (a : Fin 3) :
    tConj a ∈ Submodule.span ℂ (Set.range s) := by
  obtain ⟨c, hc⟩ := mem_span_descend K9 s (fun j i => overZomega3_mem_K9 (hs j) i)
    (tConj 0) (fun i => tConj_mem_K9 0 i) h0
  have he : Nat.Coprime (galoisExp a) 9 := by fin_cases a <;> decide
  have h3 : galoisExp a % 3 = 1 := by fin_cases a <;> decide
  set σ := galEmb (galoisExp a) he with hσ
  have key : tConj a = ∑ j, σ (c j) • s j := by
    funext i
    -- coordinate i of hc, as an identity in K9
    have hi : (∑ j, c j * ⟨s j i, overZomega3_mem_K9 (hs j) i⟩ : K9)
        = ⟨tConj 0 i, tConj_mem_K9 0 i⟩ := by
      apply Subtype.ext
      have := congrFun hc i
      simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul] at this
      push_cast
      exact this
    have hσi := congrArg σ hi
    rw [map_sum] at hσi
    simp only [map_mul] at hσi
    have hfix : ∀ j, σ ⟨s j i, overZomega3_mem_K9 (hs j) i⟩ = s j i := fun j => by
      obtain ⟨p, q, hpq⟩ := hs j i
      have hmem' : (p : ℂ) + (q : ℂ) * omega3 ∈ K9 := hpq ▸ overZomega3_mem_K9 (hs j) i
      have : (⟨s j i, overZomega3_mem_K9 (hs j) i⟩ : K9) = ⟨(p : ℂ) + (q : ℂ) * omega3, hmem'⟩ :=
        Subtype.ext hpq
      rw [this, hσ, galEmb_fix_Zomega3 _ he h3, ← hpq]
    have htgt : σ ⟨tConj 0 i, tConj_mem_K9 0 i⟩ = tConj a i := by
      have hk : (⟨tConj 0 i, tConj_mem_K9 0 i⟩ : K9)
          = ⟨omega9 ^ (galoisExp 0 * (i : ℕ)), pow_mem omega9_mem_K9 _⟩ := rfl
      rw [hk, hσ, galEmb_omega9_pow]
      simp only [tConj, galoisExp]
      simp
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    rw [← htgt, ← hσi]
    exact Finset.sum_congr rfl fun j _ => by rw [hfix j]
  rw [key]
  exact Submodule.sum_mem _ fun j _ =>
    Submodule.smul_mem _ _ (Submodule.subset_span ⟨j, rfl⟩)

/-- **χ(|T3⟩) ≥ 3 for vectors over `ℤ[ω₃]`**: any family whose `ℂ`-span
    contains the T3 amplitude vector has at least three members. Every
    qutrit stabilizer state is a positive real multiple of such a vector, and
    the span is unchanged by rescaling its generators. -/
theorem t3_three_le {r : ℕ} (s : Fin r → (Fin 3 → ℂ))
    (hs : ∀ j, OverZomega3 (s j))
    (h0 : tConj 0 ∈ Submodule.span ℂ (Set.range s)) : 3 ≤ r :=
  three_le_of_span_galois_orbit s (tConj_mem_span_of_overZomega3 s hs h0)

end StabRank
