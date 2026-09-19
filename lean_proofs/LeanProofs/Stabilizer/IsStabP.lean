/-
The stabilizer predicate for qudits of any prime dimension `p`, and the
stabilizer rank against it.

`IsStab` in `Stabilizer/IsStab.lean` is the qutrit predicate: digits in
`Fin 3`, flats over `F_3`, phase `ω₃^(Q(y) + l·y)`. This file states the same
parametrisation for a prime `p`, following the conventions of the verifier
(`verify_challenge/stabrank_verify.py`): a stabilizer state on `n` qudits is
uniform on an affine flat `x = x₀ + Wᵀ y` over `ZMod p` and carries the phase

  ζ_D ^ (Q(y) · (D / p) + l·y),        D = p for odd p,  D = 4 for p = 2,

with `ζ_D = exp(2πi/D)`, `Q` a `k × k` matrix over `ZMod p` read as the
quadratic form `Σ Q_ij y_i y_j`, and `l ∈ (ZMod D)^k`. For odd `p` the factor
`D / p` is `1` and the phase is `ω_p^(Q(y) + l·y)`; for `p = 2` it is `2` and
the phase is `i^(l·y) (-1)^(Q(y))`, the fourth roots of unity that the qubit
stabilizer group needs. Every state of this form is a stabilizer state, so
`stabRankP p` is at most the true stabilizer rank and an upper bound against
it is an upper bound on the stabilizer rank.

`ZMod 3` is `Fin 3` by definition, its ring structure unfolds to the one on
`Fin 3`, and `ZMod.val` unfolds to `Fin.val`, so at `p = 3` every piece of
the generic parametrisation is definitionally the qutrit one except for the
factor `D / p = 1` in the exponent and `ζ_3 = ω₃`; `isStab_iff_isStabP` and
`stabRank_eq_stabRankP` record that the two predicates and ranks agree, and
the qutrit files are left as they are.

The rest of the file is the generic form of the tools the qutrit files use:
full-support states (`stabVecP_id`), points (`stabVecP_zero`), the
computational basis as a stabilizer basis (`isStabP_single`,
`decompCardsP_nonempty`, `stabRankP_le_pow`), the explicit sums for
`k ≤ 3` over `ZMod 2`, and `stabRankP_le_of_terms`, which turns a pointwise
identity `ψ = Σ αⱼ • σⱼ` into `stabRankP p ψ ≤ (number of terms)`.
-/
import LeanProofs.Stabilizer.IsStab
import Mathlib.Data.ZMod.Basic

namespace StabRank

open Stabilizer

/-! ### The period of the phase group and its root of unity -/

/-- The order of the phase group: `p` for odd `p`, `4` for `p = 2`. -/
def stabPeriod : ℕ → ℕ
  | 2 => 4
  | p => p

theorem stabPeriod_two : stabPeriod 2 = 4 := rfl
theorem stabPeriod_three : stabPeriod 3 = 3 := rfl

theorem stabPeriod_ne_zero (p : ℕ) [NeZero p] : stabPeriod p ≠ 0 := by
  unfold stabPeriod
  split
  · norm_num
  · exact NeZero.ne p

theorem stabPeriod_pos (p : ℕ) [NeZero p] : 0 < stabPeriod p :=
  Nat.pos_of_ne_zero (stabPeriod_ne_zero p)

instance neZero_of_fact_prime (p : ℕ) [hp : Fact p.Prime] : NeZero p := ⟨hp.out.ne_zero⟩

/-- `ζ_D = exp(2πi/D)` with `D = stabPeriod p`. -/
noncomputable def zeta (p : ℕ) : ℂ :=
  Complex.exp (2 * Real.pi * Complex.I / (stabPeriod p : ℂ))

theorem zeta_pow_period (p : ℕ) [NeZero p] : zeta p ^ stabPeriod p = 1 := by
  unfold zeta
  rw [← Complex.exp_nat_mul]
  have hD : (stabPeriod p : ℂ) ≠ 0 := by exact_mod_cast stabPeriod_ne_zero p
  rw [mul_div_cancel₀ _ hD, Complex.exp_two_pi_mul_I]

/-- `ζ^a = ζ^b` whenever `a ≡ b (mod D)`. Not a simp lemma. -/
theorem zeta_pow_eq_of_mod (p : ℕ) [NeZero p] {a b : ℕ}
    (h : a % stabPeriod p = b % stabPeriod p) : zeta p ^ a = zeta p ^ b := by
  rw [pow_eq_pow_mod a (zeta_pow_period p), pow_eq_pow_mod b (zeta_pow_period p), h]

theorem zeta_pow_mod (p : ℕ) [NeZero p] (a : ℕ) : zeta p ^ a = zeta p ^ (a % stabPeriod p) :=
  pow_eq_pow_mod a (zeta_pow_period p)

theorem zeta_ne_zero (p : ℕ) : zeta p ≠ 0 := Complex.exp_ne_zero _

theorem zeta_three : zeta 3 = omega3 := by
  simp [zeta, omega3, stabPeriod]

theorem zeta_two : zeta 2 = Complex.I := by
  have h : 2 * (Real.pi : ℂ) * Complex.I / ((stabPeriod 2 : ℕ) : ℂ)
      = (Real.pi : ℂ) / 2 * Complex.I := by
    rw [stabPeriod_two]
    push_cast
    ring
  rw [zeta, h, Complex.exp_pi_div_two_mul_I]

/-- `i^n` by the residue of `n` mod 4. With `n` a numeral, `simp` reduces the
    right-hand side to one of `1`, `i`, `-1`, `-i`. -/
theorem I_pow_mod (n : ℕ) : Complex.I ^ n = Complex.I ^ (n % 4) :=
  pow_eq_pow_mod n Complex.I_pow_four

theorem I_pow_three : Complex.I ^ 3 = -Complex.I := by
  rw [pow_succ, Complex.I_sq]
  ring

/-- `ζ_4^n` for `p = 2`, reduced to a power of `i` with exponent below `4`. -/
theorem zeta_two_pow (n : ℕ) : zeta 2 ^ n = Complex.I ^ (n % 4) := by
  rw [zeta_two, I_pow_mod]

/-! ### The parametrisation -/

/-- Digit string of a computational-basis index, base `p`. -/
def digitsP (p n : ℕ) [NeZero p] : Fin (p ^ n) ≃ (Fin n → ZMod p) :=
  finFunctionFinEquiv.symm.trans (Equiv.piCongrRight fun _ => (ZMod.finEquiv p).toEquiv)

/-- The affine parametrisation of the support, `y ↦ x₀ + Wᵀ y` over `ZMod p`. -/
def affinePtP {p n k : ℕ} (x0 : Fin n → ZMod p) (W : Fin k → Fin n → ZMod p)
    (y : Fin k → ZMod p) : Fin n → ZMod p :=
  fun i => x0 i + ∑ j, y j * W j i

/-- The exponent `Q(y) · (D / p) + l·y` of `ζ_D`, as a natural number;
    reduction mod `D` is absorbed by `ζ_D^D = 1`. -/
def quadPhaseP {p k : ℕ} (Q : Fin k → Fin k → ZMod p) (l : Fin k → ZMod (stabPeriod p))
    (y : Fin k → ZMod p) : ℕ :=
  (stabPeriod p / p) * ∑ i, ∑ j, (Q i j).val * (y i).val * (y j).val
    + ∑ i, (l i).val * (y i).val

/-- An `n`-qudit stabilizer state in the standard parametrisation, as a
    function on digit strings. -/
noncomputable def stabVecP (p : ℕ) [NeZero p] (n k : ℕ) (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) : (Fin n → ZMod p) → ℂ :=
  fun x => ∑ y : Fin k → ZMod p,
    if x = affinePtP x0 W y then zeta p ^ quadPhaseP Q l y else 0

/-- **The stabilizer predicate for `p`-dimensional qudits.** `v` is a nonzero
    scalar multiple of some `stabVecP` whose support parametrisation is
    injective. -/
def IsStabP (p : ℕ) [Fact p.Prime] {n : ℕ} (v : Fin (p ^ n) → ℂ) : Prop :=
  ∃ (c : ℂ) (k : ℕ) (x0 : Fin n → ZMod p) (W : Fin k → Fin n → ZMod p)
    (Q : Fin k → Fin k → ZMod p) (l : Fin k → ZMod (stabPeriod p)),
    c ≠ 0 ∧ Function.Injective (affinePtP x0 W) ∧
    v = fun idx => c * stabVecP p n k x0 W Q l (digitsP p n idx)

/-- **Stabilizer rank against `IsStabP p`.** -/
noncomputable def stabRankP (p : ℕ) [Fact p.Prime] {n : ℕ} (v : Fin (p ^ n) → ℂ) : ℕ :=
  Stabilizer.stabRank (IsStabP p) v

theorem stabRankP_def (p : ℕ) [Fact p.Prime] {n : ℕ} (v : Fin (p ^ n) → ℂ) :
    stabRankP p v = Stabilizer.stabRank (IsStabP p) v := rfl

/-- A `stabVecP` read on computational-basis indices; `isStabP_stabTerm` is
    the predicate for it, so a decomposition into `stabTerm`s needs no
    evaluation of the sum to be a stabilizer decomposition. -/
noncomputable def stabTerm (p : ℕ) [NeZero p] (n k : ℕ) (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) : Fin (p ^ n) → ℂ :=
  fun idx => stabVecP p n k x0 W Q l (digitsP p n idx)

theorem isStabP_stabTerm (p : ℕ) [Fact p.Prime] (n k : ℕ) (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (hinj : Function.Injective (affinePtP x0 W)) :
    IsStabP p (stabTerm p n k x0 W Q l) :=
  ⟨1, k, x0, W, Q, l, one_ne_zero, hinj, by funext idx; simp [stabTerm]⟩

/-! ### `p = 3` is the qutrit predicate -/

theorem digitsP_three {n : ℕ} (idx : Fin (3 ^ n)) : digitsP 3 n idx = digits n idx := rfl

theorem affinePtP_three {n k : ℕ} (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (y : Fin k → Fin 3) : affinePtP (p := 3) x0 W y = affinePt x0 W y := rfl

theorem quadPhaseP_three {k : ℕ} (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3)
    (y : Fin k → Fin 3) : quadPhaseP (p := 3) Q l y = quadPhase Q l y := by
  change 1 * _ + _ = _
  rw [one_mul]
  rfl

theorem stabVecP_three (n k : ℕ) (x0 : Fin n → Fin 3) (W : Fin k → Fin n → Fin 3)
    (Q : Fin k → Fin k → Fin 3) (l : Fin k → Fin 3) (x : Fin n → Fin 3) :
    stabVecP 3 n k x0 W Q l x = stabVecN n k x0 W Q l x := by
  unfold stabVecP stabVecN
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [affinePtP_three, quadPhaseP_three, zeta_three]
  rfl

/-- The qutrit predicate is the generic one at `p = 3`. -/
theorem isStab_iff_isStabP {n : ℕ} (v : QutritVec n) : IsStab v ↔ IsStabP 3 v := by
  constructor
  · rintro ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩
    exact ⟨c, k, x0, W, Q, l, hc, hinj, by funext idx; rw [stabVecP_three]; rfl⟩
  · rintro ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩
    exact ⟨c, k, x0, W, Q, l, hc, hinj, by funext idx; rw [stabVecP_three]; rfl⟩

theorem isStab_eq_isStabP (n : ℕ) : (IsStab (n := n)) = IsStabP 3 := by
  funext v
  exact propext (isStab_iff_isStabP v)

/-- The qutrit rank is the generic one at `p = 3`. Every `stabRank IsStab`
    theorem of the qutrit files is a `stabRankP 3` theorem through this. -/
theorem stabRank_eq_stabRankP {n : ℕ} (v : QutritVec n) :
    Stabilizer.stabRank (IsStab (n := n)) v = stabRankP 3 v := by
  rw [stabRankP_def, isStab_eq_isStabP]

/-! ### Full support: `k = n`, `x₀ = 0`, `W = I` -/

/-- The identity matrix over `ZMod p`, as rows of `W`. -/
def idWP (p n : ℕ) : Fin n → Fin n → ZMod p := fun j i => if j = i then 1 else 0

theorem affinePtP_id (p n : ℕ) (y : Fin n → ZMod p) : affinePtP 0 (idWP p n) y = y := by
  funext i
  simp [affinePtP, idWP, mul_ite, Finset.sum_ite_eq']

theorem affinePtP_id_injective (p n : ℕ) : Function.Injective (affinePtP 0 (idWP p n)) := by
  intro y y' h
  simpa [affinePtP_id] using h

/-- With full support exactly one `y` survives, and the state is the pure
    phase `ζ^(Q(x)·(D/p) + l·x)`. -/
theorem stabVecP_id (p : ℕ) [NeZero p] (n : ℕ) (Q : Fin n → Fin n → ZMod p)
    (l : Fin n → ZMod (stabPeriod p)) (x : Fin n → ZMod p) :
    stabVecP p n n 0 (idWP p n) Q l x = zeta p ^ quadPhaseP Q l x := by
  unfold stabVecP
  simp only [affinePtP_id]
  rw [Finset.sum_eq_single x]
  · simp
  · intro y _ hyx
    rw [if_neg (Ne.symm hyx)]
  · simp

/-- Injectivity of a flat parametrisation from pivot columns: if row `j` of
    `W` has a `1` in column `piv j` and every other row has a `0` there, the
    parametrisation `y ↦ x₀ + Wᵀ y` is injective. -/
theorem affinePtP_injective_of_pivots {p n k : ℕ} (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (piv : Fin k → Fin n)
    (h : ∀ j j', W j (piv j') = if j = j' then 1 else 0) :
    Function.Injective (affinePtP x0 W) := by
  intro y y' hy
  funext j
  have e := congrFun hy (piv j)
  simp only [affinePtP, h, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ,
    if_true] at e
  exact add_left_cancel e

/-! ### Points: `k = 0` -/

/-- `|x₀⟩` as `stabVecP` with `k = 0`: the flat is the single point `x₀`. -/
theorem stabVecP_zero (p : ℕ) [NeZero p] (n : ℕ) (x0 : Fin n → ZMod p) (x : Fin n → ZMod p) :
    stabVecP p n 0 x0 (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i) x
      = if x = x0 then 1 else 0 := by
  have hpt : ∀ y : Fin 0 → ZMod p, affinePtP x0 (fun j => Fin.elim0 j) y = x0 := by
    intro y
    funext i
    simp [affinePtP]
  have hph : ∀ y : Fin 0 → ZMod p,
      quadPhaseP (p := p) (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i) y = 0 := by
    intro y
    simp [quadPhaseP]
  unfold stabVecP
  rw [Fintype.sum_unique]
  simp only [hpt, hph, pow_zero]

theorem isStabP_single (p : ℕ) [Fact p.Prime] (n : ℕ) (idx : Fin (p ^ n)) :
    IsStabP p (Pi.single idx (1 : ℂ)) := by
  refine ⟨1, 0, digitsP p n idx, fun j => Fin.elim0 j, fun i _ => Fin.elim0 i,
    fun i => Fin.elim0 i, one_ne_zero, Function.injective_of_subsingleton _, ?_⟩
  funext i
  rw [stabVecP_zero, one_mul, Pi.single_apply]
  by_cases h : i = idx
  · subst h
    simp
  · rw [if_neg h, if_neg]
    intro h'
    exact h ((digitsP p n).injective h')

/-- Every vector is a combination of computational-basis states, so
    `DecompCards (IsStabP p) ψ` is nonempty for every `ψ`. -/
theorem decompCardsP_nonempty (p : ℕ) [Fact p.Prime] (n : ℕ) (ψ : Fin (p ^ n) → ℂ) :
    (DecompCards (IsStabP p (n := n)) ψ).Nonempty := by
  classical
  refine ⟨_, (Finset.univ : Finset (Fin (p ^ n))).image
      (fun idx => (Pi.single idx (1 : ℂ) : Fin (p ^ n) → ℂ)), rfl, ?_, ?_⟩
  · intro σ hσ
    obtain ⟨idx, _, rfl⟩ := Finset.mem_image.mp hσ
    exact isStabP_single p n idx
  · have hψ : ψ = ∑ idx : Fin (p ^ n), ψ idx • (Pi.single idx (1 : ℂ) : Fin (p ^ n) → ℂ) := by
      funext j
      simp [Finset.sum_apply, Pi.single_apply]
    rw [hψ]
    refine Submodule.sum_mem _ (fun idx _ => Submodule.smul_mem _ _ (Submodule.subset_span ?_))
    simp only [Finset.coe_image, Finset.coe_univ, Set.image_univ, Set.mem_range]
    exact ⟨idx, rfl⟩

/-- The computational basis bounds `stabRankP p ψ ≤ p ^ n` for every `ψ`. -/
theorem stabRankP_le_pow (p : ℕ) [Fact p.Prime] (n : ℕ) (ψ : Fin (p ^ n) → ℂ) :
    stabRankP p ψ ≤ p ^ n := by
  classical
  refine le_trans (Stabilizer.stabRank_le_of_decomp
    (S := (Finset.univ : Finset (Fin (p ^ n))).image
      (fun idx => (Pi.single idx (1 : ℂ) : Fin (p ^ n) → ℂ))) ?_ ?_) ?_
  · intro σ hσ
    obtain ⟨idx, _, rfl⟩ := Finset.mem_image.mp hσ
    exact isStabP_single p n idx
  · have hψ : ψ = ∑ idx : Fin (p ^ n), ψ idx • (Pi.single idx (1 : ℂ) : Fin (p ^ n) → ℂ) := by
      funext j
      simp [Finset.sum_apply, Pi.single_apply]
    rw [hψ]
    refine Submodule.sum_mem _ (fun idx _ => Submodule.smul_mem _ _ (Submodule.subset_span ?_))
    simp only [Finset.coe_image, Finset.coe_univ, Set.image_univ, Set.mem_range]
    exact ⟨idx, rfl⟩
  · exact le_trans Finset.card_image_le (by simp)

/-! ### Upper bounds from a list of terms -/

/-- A vector written as `Σ αⱼ • σⱼ` over a finite family lies in the span of
    the family's range. -/
theorem mem_span_of_eq_sum {ι : Type*} {r : ℕ} (σ : Fin r → (ι → ℂ)) (α : Fin r → ℂ)
    {v : ι → ℂ} (h : v = ∑ j, α j • σ j) :
    v ∈ Submodule.span ℂ (Set.range σ) := by
  rw [h]
  exact Submodule.sum_mem _ fun j _ =>
    Submodule.smul_mem _ _ (Submodule.subset_span ⟨j, rfl⟩)

/-- **`stabRankP p ψ ≤ r` from `r` stabilizer terms and a pointwise identity
    `ψ = Σ αⱼ • σⱼ`.** -/
theorem stabRankP_le_of_terms (p : ℕ) [Fact p.Prime] {n r : ℕ} (σ : Fin r → (Fin (p ^ n) → ℂ))
    (α : Fin r → ℂ) (hσ : ∀ j, IsStabP p (σ j)) {ψ : Fin (p ^ n) → ℂ}
    (h : ψ = ∑ j, α j • σ j) : stabRankP p ψ ≤ r := by
  classical
  refine le_trans (Stabilizer.stabRank_le_of_decomp (S := Finset.univ.image σ) ?_ ?_)
    (le_trans Finset.card_image_le (by simp))
  · intro τ hτ
    obtain ⟨j, _, rfl⟩ := Finset.mem_image.mp hτ
    exact hσ j
  · rw [Finset.coe_image, Finset.coe_univ, Set.image_univ]
    exact mem_span_of_eq_sum σ α h

/-! ### Explicit sums over `ZMod 2` for `k ≤ 3` -/

theorem sum_fin1_zmod2 (f : (Fin 1 → ZMod 2) → ℂ) :
    ∑ y : Fin 1 → ZMod 2, f y = f ![0] + f ![1] := by
  rw [show (Finset.univ : Finset (Fin 1 → ZMod 2)) = {![0], ![1]} by decide +kernel]
  simp [Finset.sum_insert]

theorem sum_fin2_zmod2 (f : (Fin 2 → ZMod 2) → ℂ) :
    ∑ y : Fin 2 → ZMod 2, f y = f ![0, 0] + f ![0, 1] + f ![1, 0] + f ![1, 1] := by
  rw [show (Finset.univ : Finset (Fin 2 → ZMod 2)) = {![0, 0], ![0, 1], ![1, 0], ![1, 1]}
    by decide +kernel]
  simp [Finset.sum_insert]
  ring

theorem sum_fin3_zmod2 (f : (Fin 3 → ZMod 2) → ℂ) :
    ∑ y : Fin 3 → ZMod 2, f y
      = f ![0, 0, 0] + f ![0, 0, 1] + f ![0, 1, 0] + f ![0, 1, 1]
        + f ![1, 0, 0] + f ![1, 0, 1] + f ![1, 1, 0] + f ![1, 1, 1] := by
  rw [show (Finset.univ : Finset (Fin 3 → ZMod 2))
      = {![0, 0, 0], ![0, 0, 1], ![0, 1, 0], ![0, 1, 1],
         ![1, 0, 0], ![1, 0, 1], ![1, 1, 0], ![1, 1, 1]} by decide +kernel]
  simp [Finset.sum_insert]
  ring

/-- `stabVecP 2` with `k = 1`: two terms. -/
theorem stabVecP_two_k1 (n : ℕ) (x0 : Fin n → ZMod 2) (W : Fin 1 → Fin n → ZMod 2)
    (Q : Fin 1 → Fin 1 → ZMod 2) (l : Fin 1 → ZMod (stabPeriod 2)) (x : Fin n → ZMod 2) :
    stabVecP 2 n 1 x0 W Q l x
      = (if x = affinePtP x0 W ![0] then zeta 2 ^ quadPhaseP Q l ![0] else 0)
      + (if x = affinePtP x0 W ![1] then zeta 2 ^ quadPhaseP Q l ![1] else 0) := by
  unfold stabVecP
  rw [sum_fin1_zmod2]

/-- `stabVecP 2` with `k = 2`: four terms. -/
theorem stabVecP_two_k2 (n : ℕ) (x0 : Fin n → ZMod 2) (W : Fin 2 → Fin n → ZMod 2)
    (Q : Fin 2 → Fin 2 → ZMod 2) (l : Fin 2 → ZMod (stabPeriod 2)) (x : Fin n → ZMod 2) :
    stabVecP 2 n 2 x0 W Q l x
      = (if x = affinePtP x0 W ![0, 0] then zeta 2 ^ quadPhaseP Q l ![0, 0] else 0)
      + (if x = affinePtP x0 W ![0, 1] then zeta 2 ^ quadPhaseP Q l ![0, 1] else 0)
      + (if x = affinePtP x0 W ![1, 0] then zeta 2 ^ quadPhaseP Q l ![1, 0] else 0)
      + (if x = affinePtP x0 W ![1, 1] then zeta 2 ^ quadPhaseP Q l ![1, 1] else 0) := by
  unfold stabVecP
  rw [sum_fin2_zmod2]

/-- `stabVecP 2` with `k = 3`: eight terms. -/
theorem stabVecP_two_k3 (n : ℕ) (x0 : Fin n → ZMod 2) (W : Fin 3 → Fin n → ZMod 2)
    (Q : Fin 3 → Fin 3 → ZMod 2) (l : Fin 3 → ZMod (stabPeriod 2)) (x : Fin n → ZMod 2) :
    stabVecP 2 n 3 x0 W Q l x
      = (if x = affinePtP x0 W ![0, 0, 0] then zeta 2 ^ quadPhaseP Q l ![0, 0, 0] else 0)
      + (if x = affinePtP x0 W ![0, 0, 1] then zeta 2 ^ quadPhaseP Q l ![0, 0, 1] else 0)
      + (if x = affinePtP x0 W ![0, 1, 0] then zeta 2 ^ quadPhaseP Q l ![0, 1, 0] else 0)
      + (if x = affinePtP x0 W ![0, 1, 1] then zeta 2 ^ quadPhaseP Q l ![0, 1, 1] else 0)
      + (if x = affinePtP x0 W ![1, 0, 0] then zeta 2 ^ quadPhaseP Q l ![1, 0, 0] else 0)
      + (if x = affinePtP x0 W ![1, 0, 1] then zeta 2 ^ quadPhaseP Q l ![1, 0, 1] else 0)
      + (if x = affinePtP x0 W ![1, 1, 0] then zeta 2 ^ quadPhaseP Q l ![1, 1, 0] else 0)
      + (if x = affinePtP x0 W ![1, 1, 1] then zeta 2 ^ quadPhaseP Q l ![1, 1, 1] else 0) := by
  unfold stabVecP
  rw [sum_fin3_zmod2]

/-! ### Values of small numerals, for `simp` in the qubit files -/

theorem zmod2_val_zero : (0 : ZMod 2).val = 0 := rfl
theorem zmod2_val_one : (1 : ZMod 2).val = 1 := rfl
theorem zmod4_val_zero : (0 : ZMod (stabPeriod 2)).val = 0 := rfl
theorem zmod4_val_one : (1 : ZMod (stabPeriod 2)).val = 1 := rfl
theorem zmod4_val_two : (2 : ZMod (stabPeriod 2)).val = 2 := rfl
theorem zmod4_val_three : (3 : ZMod (stabPeriod 2)).val = 3 := rfl

theorem stabPeriod_two_div : stabPeriod 2 / 2 = 2 := rfl

/-- The two residues mod 2, for case splits that substitute numerals. -/
theorem zmod2_cases (a : ZMod 2) : a = 0 ∨ a = 1 := by decide +revert

/-- Digit strings on `n` qubits, for `n ≤ 4`, as explicit vectors. -/
theorem fin1_eta_z (x : Fin 1 → ZMod 2) : x = ![x 0] := by
  funext i
  fin_cases i
  rfl

theorem fin2_eta_z (x : Fin 2 → ZMod 2) : x = ![x 0, x 1] := by
  funext i
  fin_cases i <;> rfl

theorem fin3_eta_z (x : Fin 3 → ZMod 2) : x = ![x 0, x 1, x 2] := by
  funext i
  fin_cases i <;> rfl

theorem fin4_eta_z (x : Fin 4 → ZMod 2) : x = ![x 0, x 1, x 2, x 3] := by
  funext i
  fin_cases i <;> rfl

end StabRank
