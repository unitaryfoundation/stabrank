/-
χ(|N⟩^⊗2) ≥ 3 and χ(|H₃⟩^⊗2) ≥ 3 against `stabRank`: no two two-qutrit
stabilizer states span either target.

`verify_challenge/cert_m2_rank2.py` certifies both by a numerical search over
the 64620 pairs of two-qutrit stabilizer states. The proof here needs no
enumeration. It rests on two facts about an `IsStab` vector `σ` on two
qutrits: its nonzero entries all have the same modulus (`IsStab.norm_eq`),
and either it has full support with entries `c ω^e`, or its support has at
most three points (`IsStab.full_or_small`: the flat parametrisation is
injective, so `k ≤ 2`, and `k = 2` forces it to be surjective).

Both targets are products `h ⊗ h` of a one-qutrit vector `h` with two
distinct moduli, so their nine amplitudes fall into three modulus classes:
`r₁` on four points, `r₂` on four points, `r₃` at one point, with
`r₁ < r₂ < r₃` and `r₁ + r₂ < r₃`. Suppose `ψ = a σ + b τ`.

* If `τ` has at most three support points, each four-point class has a point
  outside its support; there `ψ = a σ`, and `σ` has one modulus on its
  support, so `r₁ = r₂`. Symmetrically if `σ` is small.
* If both have full support, `ψ(x) ω^(2e₁(x)) = a' + b' ω^(g(x))` with
  `g(x) ∈ {0, 1, 2}`. At the three probe points the moduli are `r₁, r₂, r₃`,
  so the three values are distinct and `g` takes all three values; then
  `Σᵢ ω^(g(xᵢ)) ψ(xᵢ) ω^(2e₁(xᵢ)) = 0`, because `Σⱼ ω^j = Σⱼ ω^(2j) = 0`,
  and the triangle inequality gives `r₃ ≤ r₁ + r₂`. This is Pompeiu's
  inequality for the equilateral triangle `a' + b' ω^j`.

`stabRank_gt_two_of_three_moduli` packages the argument for any two-qutrit
vector with such classes; `norrell_m2_stabRank_gt_two` and
`h3_m2_stabRank_gt_two` instantiate it, and with `M2StabRank` the two cells
are settled at `3` with both directions machine-checked.
-/
import LeanProofs.M2StabRank

namespace StabRank

open Stabilizer

/-! ### Moduli of stabilizer amplitudes -/

theorem norm_omega3 : ‖omega3‖ = 1 := by
  have h : ‖omega3‖ ^ 3 = 1 := by rw [← norm_pow, omega3_pow_three, norm_one]
  exact (pow_eq_one_iff_of_nonneg (norm_nonneg _) (by norm_num)).mp h

/-- Every nonzero entry of an `IsStab` vector has the same modulus. -/
theorem IsStab.norm_eq {n : ℕ} {σ : QutritVec n} (hσ : IsStab σ) {x y : Fin (3 ^ n)}
    (hx : σ x ≠ 0) (hy : σ y ≠ 0) : ‖σ x‖ = ‖σ y‖ := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hσ
  have key : ∀ z : Fin (3 ^ n),
      (fun idx => c * stabVecN n k x0 W Q l (digits n idx)) z ≠ 0 →
      ‖(fun idx => c * stabVecN n k x0 W Q l (digits n idx)) z‖ = ‖c‖ := by
    intro z hz
    have h' : stabVecN n k x0 W Q l (digits n z) ≠ 0 := by
      intro h0
      apply hz
      change c * stabVecN n k x0 W Q l (digits n z) = 0
      rw [h0, mul_zero]
    obtain ⟨w, hw⟩ := stabVecN_ne_zero_imp h'
    change ‖c * stabVecN n k x0 W Q l (digits n z)‖ = ‖c‖
    rw [hw, stabVecN_apply_affinePt x0 W Q l hinj, norm_mul, norm_pow, norm_omega3, one_pow,
      mul_one]
  exact (key x hx).trans (key y hy).symm

/-! ### Support of a two-qutrit stabilizer state -/

/-- A two-qutrit `IsStab` vector has full support with entries `c ω^e`, or
    its support has at most three points. -/
theorem IsStab.full_or_small {σ : QutritVec 2} (hσ : IsStab σ) :
    (∃ c : ℂ, c ≠ 0 ∧ ∀ x, ∃ e : ℕ, σ x = c * omega3 ^ e) ∨
    (∀ T : Finset (Fin (3 ^ 2)), 3 < T.card → ∃ x ∈ T, σ x = 0) := by
  classical
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hσ
  rcases Nat.lt_or_ge k 2 with hk | hk
  · right
    intro T hT
    by_contra hne
    simp only [not_exists, not_and] at hne
    have hsub : T.image (digits 2) ⊆
        (Finset.univ : Finset (Fin k → Fin 3)).image (affinePt x0 W) := by
      intro d hd
      obtain ⟨x, hx, rfl⟩ := Finset.mem_image.mp hd
      have h' : stabVecN 2 k x0 W Q l (digits 2 x) ≠ 0 := by
        intro h0
        apply hne x hx
        rw [h0, mul_zero]
      obtain ⟨y, hy⟩ := stabVecN_ne_zero_imp h'
      exact Finset.mem_image.mpr ⟨y, Finset.mem_univ _, hy.symm⟩
    have h1 : (T.image (digits 2)).card = T.card :=
      Finset.card_image_of_injective T (digits 2).injective
    have h2 := Finset.card_le_card hsub
    have h3 : ((Finset.univ : Finset (Fin k → Fin 3)).image (affinePt x0 W)).card ≤ 3 ^ k :=
      le_trans Finset.card_image_le (by simp)
    have h4 : 3 ^ k ≤ 3 := by
      interval_cases k <;> norm_num
    omega
  · left
    have hcard := Fintype.card_le_of_injective _ hinj
    simp only [Fintype.card_fun, Fintype.card_fin] at hcard
    have hk2 : k = 2 := le_antisymm ((Nat.pow_le_pow_iff_right (by norm_num)).mp hcard) hk
    subst hk2
    refine ⟨c, hc, fun x => ?_⟩
    obtain ⟨y, hy⟩ := Finite.injective_iff_surjective.mp hinj (digits 2 x)
    refine ⟨quadPhase Q l y, ?_⟩
    change c * stabVecN 2 2 x0 W Q l (digits 2 x) = c * omega3 ^ quadPhase Q l y
    rw [← hy, stabVecN_apply_affinePt x0 W Q l hinj]

/-! ### Pompeiu's inequality for the triangle `a + b ω^j` -/

theorem norm_le_of_eq_neg_add (u v w : ℂ) (p q : ℕ)
    (h : u = -(omega3 ^ p * v + omega3 ^ q * w)) : ‖u‖ ≤ ‖v‖ + ‖w‖ := by
  rw [h, norm_neg]
  refine le_trans (norm_add_le _ _) ?_
  simp only [norm_mul, norm_pow, norm_omega3, one_pow, one_mul, le_refl]

/-- Three values `a + b ω^gᵢ` with strictly increasing moduli have distinct
    `gᵢ`, hence `{g₁, g₂, g₃} = {0, 1, 2}`, and then `Σᵢ ω^(gᵢ) uᵢ = 0`
    bounds the largest modulus by the sum of the other two. -/
theorem pompeiu (a b u1 u2 u3 : ℂ) (g1 g2 g3 : ℕ) (hg1 : g1 < 3) (hg2 : g2 < 3)
    (hg3 : g3 < 3) (h1 : u1 = a + b * omega3 ^ g1) (h2 : u2 = a + b * omega3 ^ g2)
    (h3 : u3 = a + b * omega3 ^ g3) (h12 : ‖u1‖ < ‖u2‖) (h23 : ‖u2‖ < ‖u3‖) :
    ‖u3‖ ≤ ‖u1‖ + ‖u2‖ := by
  have hΩcyc := omega3_sq_add_omega3_add_one
  have hΩ3 := omega3_pow_three
  interval_cases g1 <;> interval_cases g2 <;> interval_cases g3
  all_goals first
    | exact absurd (congrArg norm (h1.trans h2.symm)) h12.ne
    | exact absurd (congrArg norm (h2.trans h3.symm)) h23.ne
    | exact absurd (congrArg norm (h1.trans h3.symm)) (h12.trans h23).ne
    | exact norm_le_of_eq_neg_add u3 u1 u2 1 2
        (by rw [h1, h2, h3]; linear_combination (a + b) * hΩcyc + b * hΩ3)
    | exact norm_le_of_eq_neg_add u3 u1 u2 1 2
        (by rw [h1, h2, h3]; linear_combination (a + b) * hΩcyc + b * omega3 * hΩ3)
    | exact norm_le_of_eq_neg_add u3 u1 u2 2 1
        (by rw [h1, h2, h3]; linear_combination (a + b) * hΩcyc + b * hΩ3)
    | exact norm_le_of_eq_neg_add u3 u1 u2 2 1
        (by rw [h1, h2, h3]; linear_combination (a + b) * hΩcyc + b * omega3 * hΩ3)

/-! ### The rank-2 exclusion from three modulus classes -/

/-- A term with at most three support points leaves a point of each four-point
    class to the other term alone, which then carries two moduli. -/
theorem no_pair_of_small {ψ σ τ : QutritVec 2} (hσ : IsStab σ)
    (hτ : ∀ T : Finset (Fin (3 ^ 2)), 3 < T.card → ∃ x ∈ T, τ x = 0)
    (a b : ℂ) (hψ : ∀ x, ψ x = a * σ x + b * τ x)
    (C1 C2 : Finset (Fin (3 ^ 2))) (hC1 : 3 < C1.card) (hC2 : 3 < C2.card)
    (r1 r2 : ℝ) (h1 : ∀ x ∈ C1, ‖ψ x‖ = r1) (h2 : ∀ x ∈ C2, ‖ψ x‖ = r2)
    (hr0 : 0 < r1) (hr12 : r1 < r2) : False := by
  obtain ⟨x, hx, hτx⟩ := hτ C1 hC1
  obtain ⟨y, hy, hτy⟩ := hτ C2 hC2
  have hx' : ψ x = a * σ x := by rw [hψ, hτx, mul_zero, add_zero]
  have hy' : ψ y = a * σ y := by rw [hψ, hτy, mul_zero, add_zero]
  have hσx : σ x ≠ 0 := by
    intro h0
    have e := h1 x hx
    rw [hx', h0, mul_zero, norm_zero] at e
    linarith
  have hσy : σ y ≠ 0 := by
    intro h0
    have e := h2 y hy
    rw [hy', h0, mul_zero, norm_zero] at e
    linarith
  have hn := hσ.norm_eq hσx hσy
  have e1 := h1 x hx
  have e2 := h2 y hy
  rw [hx', norm_mul] at e1
  rw [hy', norm_mul, ← hn] at e2
  linarith

/-- Two full-support terms put the probe values on an equilateral triangle
    `a' + b' ω^j`, and Pompeiu's inequality contradicts `r₁ + r₂ < r₃`. -/
theorem no_pair_of_full {ψ σ τ : QutritVec 2}
    (hσ : ∃ c : ℂ, c ≠ 0 ∧ ∀ x, ∃ e : ℕ, σ x = c * omega3 ^ e)
    (hτ : ∃ c : ℂ, c ≠ 0 ∧ ∀ x, ∃ e : ℕ, τ x = c * omega3 ^ e)
    (a b : ℂ) (hψ : ∀ x, ψ x = a * σ x + b * τ x)
    (x1 x2 x3 : Fin (3 ^ 2)) (r1 r2 r3 : ℝ) (h1 : ‖ψ x1‖ = r1) (h2 : ‖ψ x2‖ = r2)
    (h3 : ‖ψ x3‖ = r3) (hr12 : r1 < r2) (hr23 : r2 < r3) (hsum : r1 + r2 < r3) : False := by
  obtain ⟨c1, -, hσ⟩ := hσ
  obtain ⟨c2, -, hτ⟩ := hτ
  choose e1 he1 using hσ
  choose e2 he2 using hτ
  have hu : ∀ x, ψ x * omega3 ^ (2 * e1 x)
      = a * c1 + b * c2 * omega3 ^ ((e2 x + 2 * e1 x) % 3) := by
    intro x
    have h3e : omega3 ^ (3 * e1 x) = 1 := by rw [pow_mul, omega3_pow_three, one_pow]
    have hmod : omega3 ^ (e2 x + 2 * e1 x) = omega3 ^ ((e2 x + 2 * e1 x) % 3) :=
      pow_eq_pow_mod _ omega3_pow_three
    rw [hψ x, he1 x, he2 x]
    linear_combination (a * c1) * h3e + (b * c2) * hmod
  have hn : ∀ x, ‖ψ x * omega3 ^ (2 * e1 x)‖ = ‖ψ x‖ := by
    intro x
    rw [norm_mul, norm_pow, norm_omega3, one_pow, mul_one]
  have key := pompeiu (a * c1) (b * c2) _ _ _ _ _ _
    (Nat.mod_lt _ (by norm_num : (0 : ℕ) < 3)) (Nat.mod_lt _ (by norm_num : (0 : ℕ) < 3))
    (Nat.mod_lt _ (by norm_num : (0 : ℕ) < 3)) (hu x1) (hu x2) (hu x3)
    (by rw [hn, hn, h1, h2]; exact hr12) (by rw [hn, hn, h2, h3]; exact hr23)
  rw [hn, hn, hn, h1, h2, h3] at key
  linarith

/-- **Rank-2 exclusion from three modulus classes.** A two-qutrit vector with
    modulus `r₁` on four points, `r₂` on four points and `r₃` at a point, where
    `0 < r₁ < r₂ < r₃` and `r₁ + r₂ < r₃`, has stabilizer rank at least three. -/
theorem stabRank_gt_two_of_three_moduli (ψ : QutritVec 2) (C1 C2 : Finset (Fin (3 ^ 2)))
    (x3 : Fin (3 ^ 2)) (hC1 : 3 < C1.card) (hC2 : 3 < C2.card) (r1 r2 r3 : ℝ)
    (h1 : ∀ x ∈ C1, ‖ψ x‖ = r1) (h2 : ∀ x ∈ C2, ‖ψ x‖ = r2) (h3 : ‖ψ x3‖ = r3)
    (hr0 : 0 < r1) (hr12 : r1 < r2) (hr23 : r2 < r3) (hsum : r1 + r2 < r3) :
    Stabilizer.stabRank (IsStab (n := 2)) ψ > 2 := by
  classical
  have no_pair : ∀ σ τ : QutritVec 2, IsStab σ → IsStab τ → ∀ a b : ℂ,
      (∀ x, ψ x = a * σ x + b * τ x) → False := by
    intro σ τ hσ hτ a b hψ
    rcases hτ.full_or_small with hτf | hτs
    · rcases hσ.full_or_small with hσf | hσs
      · obtain ⟨x1, hx1⟩ : C1.Nonempty := Finset.card_pos.mp (by omega)
        obtain ⟨x2, hx2⟩ : C2.Nonempty := Finset.card_pos.mp (by omega)
        exact no_pair_of_full hσf hτf a b hψ x1 x2 x3 r1 r2 r3 (h1 x1 hx1) (h2 x2 hx2) h3
          hr12 hr23 hsum
      · exact no_pair_of_small hτ hσs b a (fun x => by rw [hψ x, add_comm]) C1 C2 hC1 hC2
          r1 r2 h1 h2 hr0 hr12
    · exact no_pair_of_small hσ hτs a b hψ C1 C2 hC1 hC2 r1 r2 h1 h2 hr0 hr12
  have hψ3 : ψ x3 ≠ 0 := by
    intro h0
    rw [h0, norm_zero] at h3
    linarith
  refine Stabilizer.stabRank_gt_of_no_decomp_le _ _ 2 (decompCards_nonempty 2 _) ?_
  intro S hcard hstab hmem
  obtain h0 | hone | htwo : S.card = 0 ∨ S.card = 1 ∨ S.card = 2 := by omega
  · rw [Finset.card_eq_zero] at h0
    subst h0
    rw [Finset.coe_empty, Submodule.span_empty, Submodule.mem_bot] at hmem
    exact hψ3 (by rw [hmem]; rfl)
  · obtain ⟨σ, rfl⟩ := Finset.card_eq_one.mp hone
    rw [Finset.coe_singleton] at hmem
    obtain ⟨a, ha⟩ := Submodule.mem_span_singleton.mp hmem
    exact no_pair σ σ (hstab σ (by simp)) (hstab σ (by simp)) a 0
      (fun x => by rw [← ha]; simp)
  · obtain ⟨σ, τ, -, rfl⟩ := Finset.card_eq_two.mp htwo
    rw [Finset.coe_pair] at hmem
    obtain ⟨a, b, hab⟩ := Submodule.mem_span_pair.mp hmem
    exact no_pair σ τ (hstab σ (by simp)) (hstab τ (by simp)) a b
      (fun x => by rw [← hab]; simp)

/-! ### Norrell m=2 -/

theorem norrellVec2_apply (x : Fin (3 ^ 2)) :
    norrellVec2 x = norrellAmp1 (digits 2 x 0) * norrellAmp1 (digits 2 x 1) := rfl

/-- Digit strings with no digit `2`: four points of modulus `‖N₀‖²`. -/
def norClass1 : Finset (Fin (3 ^ 2)) :=
  Finset.univ.filter fun idx => digits 2 idx 0 ≠ 2 ∧ digits 2 idx 1 ≠ 2

/-- Digit strings with exactly one digit `2`: four points of modulus
    `‖N₀‖ ‖N₂‖`. -/
def norClass2 : Finset (Fin (3 ^ 2)) :=
  Finset.univ.filter fun idx =>
    (digits 2 idx 0 = 2 ∧ digits 2 idx 1 ≠ 2) ∨ (digits 2 idx 0 ≠ 2 ∧ digits 2 idx 1 = 2)

theorem norClass1_card : 3 < norClass1.card := by decide
theorem norClass2_card : 3 < norClass2.card := by decide

set_option linter.flexible false in
theorem norrellVec2_class1 (x : Fin (3 ^ 2)) (hx : x ∈ norClass1) :
    ‖norrellVec2 x‖ = ‖norrellAmp1 0‖ * ‖norrellAmp1 0‖ := by
  simp only [norClass1, Finset.mem_filter, Finset.mem_univ, true_and] at hx
  rw [norrellVec2_apply, norm_mul]
  revert hx
  generalize digits 2 x 0 = d0
  generalize digits 2 x 1 = d1
  fin_cases d0 <;> fin_cases d1 <;> simp (config := { decide := true }) [norrellAmp1]

set_option linter.flexible false in
theorem norrellVec2_class2 (x : Fin (3 ^ 2)) (hx : x ∈ norClass2) :
    ‖norrellVec2 x‖ = ‖norrellAmp1 0‖ * ‖norrellAmp1 2‖ := by
  simp only [norClass2, Finset.mem_filter, Finset.mem_univ, true_and] at hx
  rw [norrellVec2_apply, norm_mul]
  revert hx
  generalize digits 2 x 0 = d0
  generalize digits 2 x 1 = d1
  fin_cases d0 <;> fin_cases d1 <;>
    simp (config := { decide := true }) [norrellAmp1, mul_comm]

theorem norrellAmp1_zero_ne_zero : norrellAmp1 0 ≠ 0 := by
  have h2 : (Real.sqrt 2 : ℂ) ≠ 0 := by exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num)
  have h3 : (Real.sqrt 3 : ℂ) ≠ 0 := by exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num)
  simp [norrellAmp1, h2, h3]

/-- `N₂ = -2 N₀`. -/
theorem norm_norrellAmp1_two : ‖norrellAmp1 2‖ = 2 * ‖norrellAmp1 0‖ := by
  have h : norrellAmp1 2 = -2 * norrellAmp1 0 := by
    simp only [norrellAmp1]
    ring
  rw [h, norm_mul, norm_neg, Complex.norm_two]

/-- **χ(|N⟩^⊗2) ≥ 3 against the concrete stabilizer predicate.** -/
theorem norrell_m2_stabRank_gt_two :
    Stabilizer.stabRank (IsStab (n := 2)) norrellVec2 > 2 := by
  have hv : 0 < ‖norrellAmp1 0‖ := norm_pos_iff.mpr norrellAmp1_zero_ne_zero
  have hw := norm_norrellAmp1_two
  refine stabRank_gt_two_of_three_moduli norrellVec2 norClass1 norClass2 (pairIdx (2, 2))
    norClass1_card norClass2_card (‖norrellAmp1 0‖ * ‖norrellAmp1 0‖)
    (‖norrellAmp1 0‖ * ‖norrellAmp1 2‖) (‖norrellAmp1 2‖ * ‖norrellAmp1 2‖)
    norrellVec2_class1 norrellVec2_class2 ?_ ?_ ?_ ?_ ?_
  · simp only [norrellVec2, ofPair_pairIdx, norrellAmp2, norm_mul]
  all_goals
    try rw [hw]
    nlinarith [hv, mul_pos hv hv]

/-- **χ(|N⟩^⊗2) = 3**, both directions machine-checked against `IsStab`. -/
theorem norrell_m2_stabRank_eq_three :
    Stabilizer.stabRank (IsStab (n := 2)) norrellVec2 = 3 :=
  le_antisymm norrell_m2_stabRank_le_three norrell_m2_stabRank_gt_two

/-! ### H₃ m=2 -/

theorem h3Vec2_apply (x : Fin (3 ^ 2)) :
    h3Vec2 x = h3Amp1 (digits 2 x 0) * h3Amp1 (digits 2 x 1) := rfl

/-- Digit strings with no digit `0`: four points of modulus `‖H₁‖²`. -/
def h3Class1 : Finset (Fin (3 ^ 2)) :=
  Finset.univ.filter fun idx => digits 2 idx 0 ≠ 0 ∧ digits 2 idx 1 ≠ 0

/-- Digit strings with exactly one digit `0`: four points of modulus
    `‖H₀‖ ‖H₁‖`. -/
def h3Class2 : Finset (Fin (3 ^ 2)) :=
  Finset.univ.filter fun idx =>
    (digits 2 idx 0 = 0 ∧ digits 2 idx 1 ≠ 0) ∨ (digits 2 idx 0 ≠ 0 ∧ digits 2 idx 1 = 0)

theorem h3Class1_card : 3 < h3Class1.card := by decide
theorem h3Class2_card : 3 < h3Class2.card := by decide

set_option linter.flexible false in
theorem h3Vec2_class1 (x : Fin (3 ^ 2)) (hx : x ∈ h3Class1) :
    ‖h3Vec2 x‖ = ‖h3Amp1 1‖ * ‖h3Amp1 1‖ := by
  simp only [h3Class1, Finset.mem_filter, Finset.mem_univ, true_and] at hx
  rw [h3Vec2_apply, norm_mul]
  revert hx
  generalize digits 2 x 0 = d0
  generalize digits 2 x 1 = d1
  fin_cases d0 <;> fin_cases d1 <;> simp (config := { decide := true }) [h3Amp1]

set_option linter.flexible false in
theorem h3Vec2_class2 (x : Fin (3 ^ 2)) (hx : x ∈ h3Class2) :
    ‖h3Vec2 x‖ = ‖h3Amp1 0‖ * ‖h3Amp1 1‖ := by
  simp only [h3Class2, Finset.mem_filter, Finset.mem_univ, true_and] at hx
  rw [h3Vec2_apply, norm_mul]
  revert hx
  generalize digits 2 x 0 = d0
  generalize digits 2 x 1 = d1
  fin_cases d0 <;> fin_cases d1 <;>
    simp (config := { decide := true }) [h3Amp1, mul_comm]

theorem h3Amp1_one_ne_zero : h3Amp1 1 ≠ 0 := by
  simp [h3Amp1, NH3C_ne_zero]

/-- `H₀ = (√3 + 1) H₁`. -/
theorem h3Amp1_zero_eq : h3Amp1 0 = ((Real.sqrt 3 : ℂ) + 1) * h3Amp1 1 := by
  have h3 := sqrt3_ne_zero
  simp only [h3Amp1]
  field_simp

theorem norm_h3Amp1_zero : ‖h3Amp1 0‖ = (Real.sqrt 3 + 1) * ‖h3Amp1 1‖ := by
  rw [h3Amp1_zero_eq, norm_mul]
  congr 1
  rw [show ((Real.sqrt 3 : ℂ) + 1) = ((Real.sqrt 3 + 1 : ℝ) : ℂ) by push_cast; ring,
    Complex.norm_real, Real.norm_of_nonneg (by positivity)]

/-- **χ(|H₃⟩^⊗2) ≥ 3 against the concrete stabilizer predicate.** -/
theorem h3_m2_stabRank_gt_two :
    Stabilizer.stabRank (IsStab (n := 2)) h3Vec2 > 2 := by
  have hu : 0 < ‖h3Amp1 1‖ := norm_pos_iff.mpr h3Amp1_one_ne_zero
  have hw := norm_h3Amp1_zero
  have hρ : (2 : ℝ) < Real.sqrt 3 + 1 := by
    have : (1 : ℝ) < Real.sqrt 3 := (Real.lt_sqrt (by norm_num)).mpr (by norm_num)
    linarith
  have huu := mul_pos hu hu
  have hmix := mul_pos huu (sub_pos.mpr hρ)
  refine stabRank_gt_two_of_three_moduli h3Vec2 h3Class1 h3Class2 (pairIdx (0, 0))
    h3Class1_card h3Class2_card (‖h3Amp1 1‖ * ‖h3Amp1 1‖) (‖h3Amp1 0‖ * ‖h3Amp1 1‖)
    (‖h3Amp1 0‖ * ‖h3Amp1 0‖) h3Vec2_class1 h3Vec2_class2 ?_ ?_ ?_ ?_ ?_
  · simp only [h3Vec2, ofPair_pairIdx, h3Amp2, norm_mul]
  all_goals
    try rw [hw]
    nlinarith [hu, huu, hρ, hmix]

/-- **χ(|H₃⟩^⊗2) = 3**, both directions machine-checked against `IsStab`. -/
theorem h3_m2_stabRank_eq_three :
    Stabilizer.stabRank (IsStab (n := 2)) h3Vec2 = 3 :=
  le_antisymm h3_m2_stabRank_le_three h3_m2_stabRank_gt_two

end StabRank
