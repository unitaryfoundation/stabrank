/-
A concrete stabilizer predicate for `Stabilizer.stabRank`, and the first
bound stated against it: `stabRank IsStab |S⟩^⊗2 ≤ 2`.

Every Pointwise file proves an identity between a target and vectors written
in a shape that a reader recognises as a stabilizer state. Lean checks the
identity and nothing about the shape, and `Stabilizer.Rank` defines
`stabRank` against an abstract predicate that nothing instantiates. This file
closes that gap for qutrits.

`stabVecN n k x0 W Q l` is the standard parametrisation on `n` qutrits,

  Σ_{y ∈ F_3^k} ω₃^(Q(y) + l·y) |x₀ + Wᵀ y⟩,

with the phase restricted to a quadratic-plus-linear form mod 3, which for an
odd prime is exactly the stabilizer phase set, and `IsStab v` says `v` is a
nonzero multiple of one whose support parametrisation `y ↦ x₀ + Wᵀ y` is
injective. Both restrictions matter for an upper bound: `StabDef.stabVec`
accepts an arbitrary phase function, under which every vector with entries
that are powers of `ω₃` would count as a stabilizer state and a bound
`stabRank ≤ k` would certify nothing. The predicate here is at most the true
set of qutrit stabilizer states, so a bound against it is a bound on the
stabilizer rank.

`strange_m2_stabRank_le_two` then combines `StrangeM2Pointwise` with
`Stabilizer.stabRank_le_of_decomp`: the two terms are shown to be `IsStab`
(full support, `W = I`, the tabulated phases agree mod 3 with the quadratic
forms `y₀² + y₀y₁ + y₁²` and `y₀² + 2y₀y₁ + y₁²`), and the pointwise identity
puts the target in their span.
-/
import LeanProofs.Stabilizer.Rank
import LeanProofs.StrangeM2Pointwise
import LeanProofs.StrangeM2Lower
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Logic.Equiv.Fin.Basic

namespace StabRank

open Stabilizer

/-- `ω₃^a = ω₃^b` whenever `a ≡ b (mod 3)`. Not a simp lemma: it would loop
    with `Nat.reduceMod`. -/
theorem omega3_pow_eq_of_mod {a b : ℕ} (h : a % 3 = b % 3) : omega3 ^ a = omega3 ^ b := by
  rw [pow_eq_pow_mod a omega3_pow_three, pow_eq_pow_mod b omega3_pow_three, h]

/-- Digit string of a computational-basis index. -/
def digits (n : ℕ) : Fin (3 ^ n) ≃ (Fin n → Fin 3) := finFunctionFinEquiv.symm

/-- A `Fin 3 × Fin 3`-indexed amplitude function, as a 2-qutrit vector. -/
noncomputable def ofPair (f : Fin 3 × Fin 3 → ℂ) : QutritVec 2 :=
  fun idx => f (finTwoArrowEquiv (Fin 3) (digits 2 idx))

/-- Quadratic-plus-linear phase `Q(y) + l·y`, as a natural-number exponent of
    `ω₃`; reduction mod 3 is absorbed by `ω₃^3 = 1`. -/
def quadPhase {k : ℕ} (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3)
    (y : Fin k → Fin 3) : ℕ :=
  ∑ i, ∑ j, (Q i j).val * (y i).val * (y j).val + ∑ i, (l i).val * (y i).val

/-- The affine parametrisation of the support, `y ↦ x₀ + Wᵀ y` over `F₃`. -/
def affinePt {n k : ℕ} (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (y : Fin k → Fin 3) : Fin n → Fin 3 :=
  fun i => x0 i + ∑ j, y j * W j i

/-- An `n`-qutrit stabilizer state in the standard parametrisation, as a
    function on digit strings. -/
noncomputable def stabVecN (n k : ℕ) (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3) : (Fin n → Fin 3) → ℂ :=
  fun x => ∑ y : Fin k → Fin 3,
    if x = affinePt x0 W y then omega3 ^ quadPhase Q l y else 0

/-- **The stabilizer predicate.** `v` is a nonzero scalar multiple of some
    `stabVecN` whose support parametrisation is injective. -/
def IsStab {n : ℕ} (v : QutritVec n) : Prop :=
  ∃ (c : ℂ) (k : ℕ) (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3),
    c ≠ 0 ∧ Function.Injective (affinePt x0 W) ∧
    v = fun idx => c * stabVecN n k x0 W Q l (digits n idx)

/-! ### Full support: `k = n`, `x₀ = 0`, `W = I` -/

/-- The identity matrix over `F₃`, as rows of `W`. -/
def idW (n : ℕ) : Fin n → Fin n → Fin 3 := fun j i => if j = i then 1 else 0

theorem affinePt_id (n : ℕ) (y : Fin n → Fin 3) : affinePt 0 (idW n) y = y := by
  funext i
  simp [affinePt, idW, mul_ite, Finset.sum_ite_eq']

theorem affinePt_id_injective (n : ℕ) : Function.Injective (affinePt 0 (idW n)) := by
  intro y y' h
  simpa [affinePt_id] using h

/-- With full support exactly one `y` survives, and the state is the pure
    phase `ω₃^(Q(x) + l·x)`. -/
theorem stabVecN_id (n : ℕ) (Q : Fin n → Fin n → Fin 3) (l : Fin n → Fin 3)
    (x : Fin n → Fin 3) :
    stabVecN n n 0 (idW n) Q l x = omega3 ^ quadPhase Q l x := by
  unfold stabVecN
  simp only [affinePt_id]
  rw [Finset.sum_eq_single x]
  · simp
  · intro y _ hyx
    rw [if_neg (Ne.symm hyx)]
  · simp

/-! ### Strange m=2 -/

/-- `Q₁(y) = y₀² + y₀y₁ + y₁²`. -/
def q1Mat : Fin 2 → Fin 2 → Fin 3 := ![![1, 1], ![0, 1]]
/-- `Q₂(y) = y₀² + 2y₀y₁ + y₁²`. -/
def q2Mat : Fin 2 → Fin 2 → Fin 3 := ![![1, 2], ![0, 1]]

/-- The tabulated phase of `StrangeM2Pointwise` agrees mod 3 with `Q₁`. -/
theorem q1_phase_mod (a b : Fin 3) :
    q1Str2 a b % 3 = quadPhase q1Mat 0 ![a, b] % 3 := by
  fin_cases a <;> fin_cases b <;> decide

/-- The tabulated phase of `StrangeM2Pointwise` agrees mod 3 with `Q₂`. -/
theorem q2_phase_mod (a b : Fin 3) :
    q2Str2 a b % 3 = quadPhase q2Mat 0 ![a, b] % 3 := by
  fin_cases a <;> fin_cases b <;> decide

/-- The pointwise vectors of `StrangeM2Pointwise`, reindexed to `QutritVec 2`. -/
noncomputable def strangeVec2 : QutritVec 2 := ofPair strangeAmp2
noncomputable def s1Vec2 : QutritVec 2 := ofPair s1Str2
noncomputable def s2Vec2 : QutritVec 2 := ofPair s2Str2

/-- A digit string on two qutrits is the pair of its digits. -/
theorem fin2_eta (x : Fin 2 → Fin 3) : x = ![x 0, x 1] := by
  funext i
  fin_cases i <;> rfl

theorem finTwoArrowEquiv_pair (x : Fin 2 → Fin 3) :
    finTwoArrowEquiv (Fin 3) x = (x 0, x 1) := rfl

theorem q1_phase_mod' (x : Fin 2 → Fin 3) :
    q1Str2 (x 0) (x 1) % 3 = quadPhase q1Mat 0 x % 3 := by
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  simpa using q1_phase_mod a b

theorem q2_phase_mod' (x : Fin 2 → Fin 3) :
    q2Str2 (x 0) (x 1) % 3 = quadPhase q2Mat 0 x % 3 := by
  obtain ⟨a, b, rfl⟩ : ∃ a b, x = ![a, b] := ⟨x 0, x 1, fin2_eta x⟩
  simpa using q2_phase_mod a b

theorem s1Vec2_isStab : IsStab s1Vec2 := by
  refine ⟨1 / 3, 2, 0, idW 2, q1Mat, 0, by norm_num, affinePt_id_injective 2, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  simp only [s1Vec2, ofPair, hx, stabVecN_id, s1Str2, finTwoArrowEquiv_pair]
  rw [omega3_pow_eq_of_mod (q1_phase_mod' x)]
  ring

theorem s2Vec2_isStab : IsStab s2Vec2 := by
  refine ⟨1 / 3, 2, 0, idW 2, q2Mat, 0, by norm_num, affinePt_id_injective 2, ?_⟩
  funext idx
  generalize hx : digits 2 idx = x
  simp only [s2Vec2, ofPair, hx, stabVecN_id, s2Str2, finTwoArrowEquiv_pair]
  rw [omega3_pow_eq_of_mod (q2_phase_mod' x)]
  ring

/-- **χ(|S⟩^⊗2) ≤ 2, against the concrete stabilizer predicate.** -/
theorem strange_m2_stabRank_le_two :
    Stabilizer.stabRank (IsStab (n := 2)) strangeVec2 ≤ 2 := by
  classical
  refine le_trans
    (Stabilizer.stabRank_le_of_decomp (S := {s1Vec2, s2Vec2}) ?_ ?_)
    Finset.card_le_two
  · intro σ hσ
    rcases Finset.mem_insert.mp hσ with rfl | hσ
    · exact s1Vec2_isStab
    · rw [Finset.mem_singleton.mp hσ]
      exact s2Vec2_isStab
  · rw [Finset.coe_pair]
    refine Submodule.mem_span_pair.mpr ⟨alphaStr2, -alphaStr2, ?_⟩
    funext idx
    simp only [strangeVec2, s1Vec2, s2Vec2, ofPair, Pi.add_apply, Pi.smul_apply,
      smul_eq_mul, strange_m2_decomposition]
    ring

/-! ### Structure of `stabVecN`: support is the image of `affinePt` -/

theorem omega3_ne_zero : omega3 ≠ 0 := by
  intro h
  have h3 := omega3_pow_three
  rw [h] at h3
  norm_num at h3

/-- A nonzero amplitude sits on the flat. -/
theorem stabVecN_ne_zero_imp {n k : ℕ} {x0 : Fin n → Fin 3} {W : Fin k → Fin n → Fin 3}
    {Q : Fin k → Fin k → Fin 3} {l : Fin k → Fin 3} {x : Fin n → Fin 3}
    (h : stabVecN n k x0 W Q l x ≠ 0) : ∃ y, x = affinePt x0 W y := by
  by_contra hne
  simp only [not_exists] at hne
  apply h
  unfold stabVecN
  exact Finset.sum_eq_zero (fun y _ => if_neg (hne y))

/-- On the flat, with an injective parametrisation, the amplitude is a pure
    phase and in particular nonzero. -/
theorem stabVecN_apply_affinePt {n k : ℕ} (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3)
    (hinj : Function.Injective (affinePt x0 W)) (y : Fin k → Fin 3) :
    stabVecN n k x0 W Q l (affinePt x0 W y) = omega3 ^ quadPhase Q l y := by
  unfold stabVecN
  rw [Finset.sum_eq_single y]
  · simp
  · intro y' _ hy'
    rw [if_neg]
    intro h
    exact hy' (hinj h).symm
  · simp

/-- The flat is closed under `a - b + d`. -/
theorem affinePt_sub_add {n k : ℕ} (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (a b d : Fin k → Fin 3) :
    affinePt x0 W a - affinePt x0 W b + affinePt x0 W d = affinePt x0 W (a - b + d) := by
  funext i
  simp only [affinePt, Pi.sub_apply, Pi.add_apply, sub_mul, add_mul,
    Finset.sum_add_distrib, Finset.sum_sub_distrib]
  abel

/-- **Affine support.** If `v` is `IsStab` and nonzero at the digit strings
    `a`, `b`, `d`, it is nonzero at `a - b + d`. -/
theorem IsStab.affine_support {n : ℕ} {v : QutritVec n} (hv : IsStab v)
    (a b d : Fin n → Fin 3)
    (ha : v ((digits n).symm a) ≠ 0) (hb : v ((digits n).symm b) ≠ 0)
    (hd : v ((digits n).symm d) ≠ 0) :
    v ((digits n).symm (a - b + d)) ≠ 0 := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hv
  simp only [Equiv.apply_symm_apply, ne_eq, mul_eq_zero, hc, false_or] at ha hb hd ⊢
  obtain ⟨ya, rfl⟩ := stabVecN_ne_zero_imp ha
  obtain ⟨yb, rfl⟩ := stabVecN_ne_zero_imp hb
  obtain ⟨yd, rfl⟩ := stabVecN_ne_zero_imp hd
  rw [affinePt_sub_add, stabVecN_apply_affinePt x0 W Q l hinj]
  exact pow_ne_zero _ omega3_ne_zero

/-! ### The computational basis is a stabilizer basis -/

/-- `|x₀⟩` as `stabVecN` with `k = 0`: the flat is the single point `x₀`. -/
theorem stabVecN_zero (n : ℕ) (x0 : Fin n → Fin 3) (x : Fin n → Fin 3) :
    stabVecN n 0 x0 (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i) x
      = if x = x0 then 1 else 0 := by
  have hpt : ∀ y : Fin 0 → Fin 3, affinePt x0 (fun j => Fin.elim0 j) y = x0 := by
    intro y
    funext i
    simp [affinePt]
  have hph : ∀ y : Fin 0 → Fin 3,
      quadPhase (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i) y = 0 := by
    intro y
    simp [quadPhase]
  unfold stabVecN
  rw [Fintype.sum_unique]
  simp only [hpt, hph, pow_zero]

theorem isStab_single (n : ℕ) (idx : Fin (3 ^ n)) : IsStab (Pi.single idx (1 : ℂ)) := by
  refine ⟨1, 0, digits n idx, fun j => Fin.elim0 j, fun i _ => Fin.elim0 i, fun i => Fin.elim0 i,
    one_ne_zero, Function.injective_of_subsingleton _, ?_⟩
  funext i
  rw [stabVecN_zero, one_mul, Pi.single_apply]
  by_cases h : i = idx
  · subst h
    simp
  · rw [if_neg h, if_neg]
    intro h'
    exact h ((digits n).injective h')

/-- Every vector is a combination of computational-basis states, so
    `DecompCards IsStab ψ` is nonempty for every `ψ`. -/
theorem decompCards_nonempty (n : ℕ) (ψ : QutritVec n) :
    (DecompCards (IsStab (n := n)) ψ).Nonempty := by
  classical
  refine ⟨_, (Finset.univ : Finset (Fin (3 ^ n))).image
      (fun idx => (Pi.single idx (1 : ℂ) : QutritVec n)), rfl, ?_, ?_⟩
  · intro σ hσ
    obtain ⟨idx, _, rfl⟩ := Finset.mem_image.mp hσ
    exact isStab_single n idx
  · have hψ : ψ = ∑ idx : Fin (3 ^ n), ψ idx • (Pi.single idx (1 : ℂ) : QutritVec n) := by
      funext j
      simp [Finset.sum_apply, Pi.single_apply]
    rw [hψ]
    refine Submodule.sum_mem _ (fun idx _ => Submodule.smul_mem _ _ (Submodule.subset_span ?_))
    simp only [Finset.coe_image, Finset.coe_univ, Set.image_univ, Set.mem_range]
    exact ⟨idx, rfl⟩

/-! ### Strange m=2: the lower bound against `IsStab`, and the exact value -/

/-- The pair index of a digit string, inverse to `ofPair`'s reindexing. -/
noncomputable def pairIdx (y : Fin 3 × Fin 3) : Fin (3 ^ 2) :=
  (digits 2).symm ((finTwoArrowEquiv (Fin 3)).symm y)

theorem ofPair_pairIdx (f : Fin 3 × Fin 3 → ℂ) (y : Fin 3 × Fin 3) :
    ofPair f (pairIdx y) = f y := by
  simp [ofPair, pairIdx]

theorem pair_sub_add (a b d : Fin 3 × Fin 3) :
    (finTwoArrowEquiv (Fin 3)).symm (a - b + d)
      = (finTwoArrowEquiv (Fin 3)).symm a - (finTwoArrowEquiv (Fin 3)).symm b
        + (finTwoArrowEquiv (Fin 3)).symm d := by
  funext i
  fin_cases i <;> rfl

/-- No single stabilizer state is proportional to `|S⟩^⊗2`. -/
theorem strangeVec2_not_mem_span_singleton (σ : QutritVec 2) (hσ : IsStab σ) :
    strangeVec2 ∉ Submodule.span ℂ ({σ} : Set (QutritVec 2)) := by
  intro hmem
  obtain ⟨c, hc⟩ := Submodule.mem_span_singleton.mp hmem
  -- transport to the pair-indexed picture of StrangeM2Lower
  set s : Fin 3 × Fin 3 → ℂ := fun y => σ (pairIdx y) with hs
  have hid : strangeAmp2 = fun y => c * s y := by
    funext y
    have := congrFun hc (pairIdx y)
    simp only [Pi.smul_apply, smul_eq_mul, strangeVec2, ofPair_pairIdx] at this
    exact this.symm
  refine strange_m2_chi_ge_two c s ?_ hid
  intro a b d ha hb hd
  simp only [hs, pairIdx] at ha hb hd ⊢
  rw [pair_sub_add]
  exact hσ.affine_support _ _ _ ha hb hd

/-- **χ(|S⟩^⊗2) ≥ 2, against the concrete stabilizer predicate.** -/
theorem strange_m2_stabRank_gt_one :
    Stabilizer.stabRank (IsStab (n := 2)) strangeVec2 > 1 := by
  classical
  refine Stabilizer.stabRank_gt_of_no_decomp_le _ _ 1 (decompCards_nonempty 2 _) ?_
  intro S hcard hstab hmem
  rcases S.eq_empty_or_nonempty with rfl | hne
  · rw [Finset.coe_empty, Submodule.span_empty, Submodule.mem_bot] at hmem
    have h11 := strangeAmp2_one_one
    rw [← ofPair_pairIdx strangeAmp2 (1, 1)] at h11
    exact h11 (by rw [show ofPair strangeAmp2 = strangeVec2 from rfl, hmem]; rfl)
  · obtain ⟨σ, rfl⟩ := Finset.card_eq_one.mp (le_antisymm hcard hne.card_pos)
    rw [Finset.coe_singleton] at hmem
    exact strangeVec2_not_mem_span_singleton σ (hstab σ (Finset.mem_singleton_self σ)) hmem

/-- **χ(|S⟩^⊗2) = 2**, both directions machine-checked against `IsStab`. -/
theorem strange_m2_stabRank_eq_two :
    Stabilizer.stabRank (IsStab (n := 2)) strangeVec2 = 2 :=
  le_antisymm strange_m2_stabRank_le_two strange_m2_stabRank_gt_one

end StabRank
