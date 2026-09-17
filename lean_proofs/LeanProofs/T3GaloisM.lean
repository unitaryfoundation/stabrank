/-
χ(|T3⟩^⊗m) ≥ 3 for every m ≥ 1, as a theorem about `Stabilizer.stabRank`.

The Galois argument does not care how many copies there are: the Galois
conjugates of `|T3⟩^⊗m` are the vectors with entries `ω₉^(e · digit sum)` for
`e ∈ {1, 4, 7}`, they are linearly independent because restricting to the
indices whose digits vanish beyond the first recovers the one-qutrit
Vandermonde, and a `ℂ`-span of vectors over `ℤ[ω₃]` containing one conjugate
contains all three, by the same descent and embedding as `T3GaloisDescent`.
So no two stabilizer states span `|T3⟩^⊗m`, at any `m`.

This is the `m`-copy statement certified numerically by
`verify_challenge/cert_t3_galois.py`. It is far weaker than the best known
lower bounds at `m = 3` and `m = 4` (six, by exhaustion after the same Galois
reduction), but at `m = 2` it is tight, and with `T3M2Pointwise` it settles
`χ(|T3⟩^⊗2) = 3` with both directions in Lean.
-/
import LeanProofs.T3M1StabRank

namespace StabRank

open Stabilizer

/-! ### The conjugates of `|T3⟩^⊗m` -/

/-- Digit sum of a computational-basis index on `m` qutrits. -/
def digitSum (m : ℕ) (idx : Fin (3 ^ m)) : ℕ := ∑ i, ((digits m idx) i).val

/-- The Galois conjugates of the unnormalised `|T3⟩^⊗m`: entries
    `ω₉^(e_a · digit sum)`, so `tConjM m 0` is `√3^m |T3⟩^⊗m`. -/
noncomputable def tConjM (m : ℕ) (a : Fin 3) : QutritVec m :=
  fun idx => omega9 ^ (galoisExp a * digitSum m idx)

theorem tConjM_mem_K9 (m : ℕ) (a : Fin 3) (idx : Fin (3 ^ m)) : tConjM m a idx ∈ K9 :=
  pow_mem omega9_mem_K9 _

/-- The index whose first digit is `i` and whose other digits vanish. -/
def firstDigit (m : ℕ) [NeZero m] (i : Fin 3) : Fin (3 ^ m) :=
  (digits m).symm (fun j => if j = 0 then i else 0)

theorem digitSum_firstDigit (m : ℕ) [NeZero m] (i : Fin 3) :
    digitSum m (firstDigit m i) = i.val := by
  unfold digitSum firstDigit
  rw [Equiv.apply_symm_apply]
  simp [apply_ite Fin.val, Finset.sum_ite_eq']

/-- Restriction of an `m`-qutrit vector to the indices `firstDigit m i`. -/
noncomputable def restrictFirst (m : ℕ) [NeZero m] : QutritVec m →ₗ[ℂ] (Fin 3 → ℂ) where
  toFun v := fun i => v (firstDigit m i)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

theorem restrictFirst_tConjM (m : ℕ) [NeZero m] (a : Fin 3) :
    restrictFirst m (tConjM m a) = tConj a := by
  funext i
  simp [restrictFirst, tConjM, tConj, digitSum_firstDigit]

/-- The three conjugates are independent: their restriction is the
    one-qutrit Vandermonde family. -/
theorem tConjM_linearIndependent (m : ℕ) [NeZero m] : LinearIndependent ℂ (tConjM m) :=
  LinearIndependent.of_comp (restrictFirst m) (by
    have h : (restrictFirst m) ∘ (tConjM m) = tConj := funext (restrictFirst_tConjM m)
    rw [h]
    exact tConj_linearIndependent)

/-- Any family whose span contains three independent vectors has at least
    three members. -/
theorem three_le_of_span_of_linearIndependent {m r : ℕ} (t : Fin 3 → QutritVec m)
    (ht : LinearIndependent ℂ t) (s : Fin r → QutritVec m)
    (hspan : ∀ a : Fin 3, t a ∈ Submodule.span ℂ (Set.range s)) : 3 ≤ r := by
  classical
  have hle : Submodule.span ℂ (Set.range t) ≤ Submodule.span ℂ (Set.range s) := by
    rw [Submodule.span_le]
    rintro _ ⟨a, rfl⟩
    exact hspan a
  have h3 : Module.finrank ℂ (Submodule.span ℂ (Set.range t)) = 3 := by
    rw [finrank_span_eq_card ht]
    simp
  have hmono : Module.finrank ℂ (Submodule.span ℂ (Set.range t))
      ≤ Module.finrank ℂ (Submodule.span ℂ (Set.range s)) :=
    Submodule.finrank_mono hle
  have hcard : (Set.range s).toFinset.card ≤ r := by
    rw [Set.toFinset_range]
    simpa using Finset.card_image_le (s := (Finset.univ : Finset (Fin r))) (f := s)
  have hr : Module.finrank ℂ (Submodule.span ℂ (Set.range s)) ≤ r :=
    le_trans (finrank_span_le_card (Set.range s)) hcard
  omega

/-! ### Galois closure of the span, `m` copies -/

theorem tConjM_mem_span_of_overZomega3 {m r : ℕ} (s : Fin r → QutritVec m)
    (hs : ∀ j, OverZomega3 (s j))
    (h0 : tConjM m 0 ∈ Submodule.span ℂ (Set.range s)) (a : Fin 3) :
    tConjM m a ∈ Submodule.span ℂ (Set.range s) := by
  obtain ⟨c, hc⟩ := mem_span_descend K9 s (fun j i => overZomega3_mem_K9 (hs j) i)
    (tConjM m 0) (fun i => tConjM_mem_K9 m 0 i) h0
  have he : Nat.Coprime (galoisExp a) 9 := by fin_cases a <;> decide
  have h3 : galoisExp a % 3 = 1 := by fin_cases a <;> decide
  set σ := galEmb (galoisExp a) he with hσ
  have key : tConjM m a = ∑ j, σ (c j) • s j := by
    funext i
    have hi : (∑ j, c j * ⟨s j i, overZomega3_mem_K9 (hs j) i⟩ : K9)
        = ⟨tConjM m 0 i, tConjM_mem_K9 m 0 i⟩ := by
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
    have htgt : σ ⟨tConjM m 0 i, tConjM_mem_K9 m 0 i⟩ = tConjM m a i := by
      have hk : (⟨tConjM m 0 i, tConjM_mem_K9 m 0 i⟩ : K9)
          = ⟨omega9 ^ (galoisExp 0 * digitSum m i), pow_mem omega9_mem_K9 _⟩ := rfl
      rw [hk, hσ, galEmb_omega9_pow]
      simp only [tConjM, galoisExp]
      simp
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    rw [← htgt, ← hσi]
    exact Finset.sum_congr rfl fun j _ => by rw [hfix j]
  rw [key]
  exact Submodule.sum_mem _ fun j _ =>
    Submodule.smul_mem _ _ (Submodule.subset_span ⟨j, rfl⟩)

/-- Any family of vectors over `ℤ[ω₃]` whose span contains `|T3⟩^⊗m` has at
    least three members. -/
theorem t3M_three_le_card {m : ℕ} [NeZero m] {κ : Type*} [Fintype κ] (s : κ → QutritVec m)
    (hs : ∀ j, OverZomega3 (s j))
    (h0 : tConjM m 0 ∈ Submodule.span ℂ (Set.range s)) : 3 ≤ Fintype.card κ := by
  have hr : Set.range (s ∘ (Fintype.equivFin κ).symm) = Set.range s :=
    (Fintype.equivFin κ).symm.surjective.range_comp s
  refine three_le_of_span_of_linearIndependent (tConjM m) (tConjM_linearIndependent m)
    (s ∘ (Fintype.equivFin κ).symm) (fun a => ?_)
  refine tConjM_mem_span_of_overZomega3 (s ∘ (Fintype.equivFin κ).symm) (fun j => hs _) ?_ a
  rw [hr]
  exact h0

/-! ### The bound against `stabRank` -/

/-- **χ(|T3⟩^⊗m) ≥ 3 for every m ≥ 1**, against the concrete stabilizer
    predicate. -/
theorem t3M_stabRank_gt_two (m : ℕ) [NeZero m] :
    Stabilizer.stabRank (IsStab (n := m)) (tConjM m 0) > 2 := by
  classical
  refine Stabilizer.stabRank_gt_of_no_decomp_le _ _ 2 (decompCards_nonempty m _) ?_
  intro S hcard hstab hmem
  choose u hu c hc using fun σ : S => IsStab.exists_overZomega3 (hstab σ σ.2)
  have hle : Submodule.span ℂ (S : Set (QutritVec m)) ≤ Submodule.span ℂ (Set.range u) := by
    rw [Submodule.span_le]
    intro σ hσ
    have hσ' : σ = c ⟨σ, hσ⟩ • u ⟨σ, hσ⟩ := hc ⟨σ, hσ⟩
    rw [hσ']
    exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨⟨σ, hσ⟩, rfl⟩)
  have h3 := t3M_three_le_card u hu (hle hmem)
  rw [Fintype.card_coe] at h3
  omega

/-- `stabRank` is unchanged by a nonzero rescaling of the target. -/
theorem stabRank_smul {n : ℕ} (P : QutritVec n → Prop) (ψ : QutritVec n) {c : ℂ} (hc : c ≠ 0) :
    Stabilizer.stabRank P (c • ψ) = Stabilizer.stabRank P ψ := by
  unfold Stabilizer.stabRank Stabilizer.DecompCards
  congr 1
  ext k
  simp only [Set.mem_setOf_eq]
  constructor <;> rintro ⟨S, hk, hS, hmem⟩ <;> refine ⟨S, hk, hS, ?_⟩
  · exact (Submodule.smul_mem_iff _ hc).mp hmem
  · exact (Submodule.smul_mem_iff _ hc).mpr hmem

/-- The normalised state `|T3⟩^⊗m = 3^(-m/2) · tConjM m 0`. -/
noncomputable def t3TargetM (m : ℕ) : QutritVec m :=
  ((1 / Real.sqrt 3 : ℝ) : ℂ) ^ m • tConjM m 0

/-- **χ(|T3⟩^⊗m) ≥ 3** for the normalised state. -/
theorem t3TargetM_stabRank_gt_two (m : ℕ) [NeZero m] :
    Stabilizer.stabRank (IsStab (n := m)) (t3TargetM m) > 2 := by
  unfold t3TargetM
  rw [stabRank_smul _ _ (pow_ne_zero _ (by
    have : (Real.sqrt 3 : ℝ) ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
    exact_mod_cast one_div_ne_zero this))]
  exact t3M_stabRank_gt_two m

/-- **χ(|T3⟩^⊗2) ≥ 3**: with `T3M2Pointwise` this settles the two-copy cell. -/
theorem t3_m2_stabRank_gt_two :
    Stabilizer.stabRank (IsStab (n := 2)) (t3TargetM 2) > 2 :=
  t3TargetM_stabRank_gt_two 2

end StabRank
