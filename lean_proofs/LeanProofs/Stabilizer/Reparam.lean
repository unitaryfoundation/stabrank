/-
Reparametrising the flat coordinates of a stabilizer state, and the
exponent table of a term, for any prime `p`.

`stabVecP p n k x0 W Q l` sums over `y ∈ (ZMod p)^k`. An affine bijection
`y = φ(z) = b + A z` of the coordinate space does not change the sum, and
after it the state reads as another `stabVecP`: the base point moves to
`x₀ + Wᵀ b`, the generators become `W' = Aᵀ W`, and the phase is a constant
`ζ^C` times `ζ^quadPhaseP Q' l' z` by `zeta_pow_quadPhaseP_comp` of
`PhaseP.lean`, since every coordinate of `φ z` is affine in `z`
(`stabVecP_reparam`). Injectivity of the parametrisation is carried along
(`affinePtP_reparam_injective`).

Two consequences are drawn here.

* Full support (`stabVecP_full_normal`): a state with `k = n` and an
  injective parametrisation has a surjective one, so choosing the preimage
  `b` of `0` and the preimages of the unit vectors for the columns of `A`
  gives `x₀' = 0` and `W' = I`. Every full-support stabilizer state is
  `ζ^C ζ^(Q'(x) + l'·x)` on all of `(ZMod p)^n`. This is the normal form of
  the dictionary for the top rank at every `n` and every prime.
* The exponent table (`tableOfP`): with pivot columns for `W`, a term is
  `none` off its flat and `some e` on it, with `ζ^e` the amplitude
  (`stabVecP_eq_tableVal`). This is the data the Python dictionary stores
  per state, and what a reflected checker will compare.

The dictionary of two-qutrit states, with the case analysis on `k ≤ 2`, is
`QutritDict2.lean`. The general reduced row echelon form for `0 < k < n`,
which is what the dictionary at three qutrits needs, is not here.
-/
import LeanProofs.Stabilizer.PhaseP
import LeanProofs.Stabilizer.Reflect

namespace StabRank

open Stabilizer

variable {p : ℕ}

/-! ### The affine change of coordinates -/

/-- The change of flat coordinates `z ↦ b + A z`, coordinate `j` being
    `affineP (b j) (A j) z = b j + Σ_t A j t * z t`. -/
def reparamY {k : ℕ} (b : Fin k → ZMod p) (A : Fin k → Fin k → ZMod p)
    (z : Fin k → ZMod p) : Fin k → ZMod p :=
  fun j => affineP (b j) (A j) z

/-- The generators after the change, `W'_t = Σ_j A_jt W_j`. -/
def reparamW {n k : ℕ} (A : Fin k → Fin k → ZMod p) (W : Fin k → Fin n → ZMod p) :
    Fin k → Fin n → ZMod p :=
  fun t i => ∑ j, A j t * W j i

theorem reparamY_apply {k : ℕ} (b : Fin k → ZMod p) (A : Fin k → Fin k → ZMod p)
    (z : Fin k → ZMod p) (j : Fin k) : reparamY b A z j = b j + ∑ t, A j t * z t := rfl

/-- The parametrisation composed with the change of coordinates. -/
theorem affinePtP_reparamY {n k : ℕ} (x0 : Fin n → ZMod p) (W : Fin k → Fin n → ZMod p)
    (b : Fin k → ZMod p) (A : Fin k → Fin k → ZMod p) (z : Fin k → ZMod p) :
    affinePtP x0 W (reparamY b A z) = affinePtP (affinePtP x0 W b) (reparamW A W) z := by
  funext i
  simp only [affinePtP, reparamY, reparamW, affineP]
  have e1 : ∀ j, (b j + ∑ t, A j t * z t) * W j i
      = b j * W j i + ∑ t, z t * (A j t * W j i) := by
    intro j
    rw [add_mul, Finset.sum_mul]
    congr 1
    refine Finset.sum_congr rfl fun t _ => ?_
    ring
  simp only [e1, Finset.sum_add_distrib, Finset.mul_sum]
  rw [Finset.sum_comm (f := fun j t => z t * (A j t * W j i))]
  ring

theorem affinePtP_reparam_injective {n k : ℕ} {x0 : Fin n → ZMod p}
    {W : Fin k → Fin n → ZMod p} (hinj : Function.Injective (affinePtP x0 W))
    {b : Fin k → ZMod p} {A : Fin k → Fin k → ZMod p}
    (hφ : Function.Injective (reparamY b A)) :
    Function.Injective (affinePtP (affinePtP x0 W b) (reparamW A W)) := by
  intro z z' h
  apply hφ
  apply hinj
  rw [affinePtP_reparamY, affinePtP_reparamY, h]

/-- **A stabilizer term under an affine change of its flat coordinates.** -/
theorem stabVecP_reparam [Fact p.Prime] {n k : ℕ} (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (b : Fin k → ZMod p) (A : Fin k → Fin k → ZMod p)
    (hφ : Function.Injective (reparamY b A)) :
    ∃ (C : ℕ) (Q' : Fin k → Fin k → ZMod p) (l' : Fin k → ZMod (stabPeriod p)),
      ∀ x, stabVecP p n k x0 W Q l x
        = zeta p ^ C * stabVecP p n k (affinePtP x0 W b) (reparamW A W) Q' l' x := by
  obtain ⟨C, Q', l', hph⟩ := zeta_pow_quadPhaseP_comp Q l b A
  refine ⟨C, Q', l', fun x => ?_⟩
  have hbij : Function.Bijective (reparamY b A) := Finite.injective_iff_bijective.mp hφ
  unfold stabVecP
  rw [← Equiv.sum_comp (Equiv.ofBijective _ hbij)
    (fun y => if x = affinePtP x0 W y then zeta p ^ quadPhaseP Q l y else 0), Finset.mul_sum]
  refine Finset.sum_congr rfl fun z _ => ?_
  simp only [Equiv.ofBijective_apply]
  rw [affinePtP_reparamY]
  have h := hph z
  change zeta p ^ quadPhaseP Q l (reparamY b A z) = _ at h
  rw [h]
  split_ifs <;> simp

/-! ### Full support: `x₀ = 0`, `W = I` -/

/-- **Every full-support stabilizer term is `ζ^C ζ^(Q'(x) + l'·x)`.** With
    `k = n` an injective parametrisation is surjective; the preimage of `0`
    is the translation and the preimages of the unit vectors the columns of
    the change of basis. -/
theorem stabVecP_full_normal [Fact p.Prime] {n : ℕ} (x0 : Fin n → ZMod p)
    (W : Fin n → Fin n → ZMod p) (Q : Fin n → Fin n → ZMod p)
    (l : Fin n → ZMod (stabPeriod p)) (hinj : Function.Injective (affinePtP x0 W)) :
    ∃ (C : ℕ) (Q' : Fin n → Fin n → ZMod p) (l' : Fin n → ZMod (stabPeriod p)),
      ∀ x, stabVecP p n n x0 W Q l x = zeta p ^ C * stabVecP p n n 0 (idWP p n) Q' l' x := by
  have hsurj := Finite.injective_iff_surjective.mp hinj
  choose g hg using hsurj
  set b : Fin n → ZMod p := g 0 with hb
  set A : Fin n → Fin n → ZMod p := fun j t => g (Pi.single t 1) j - b j with hA
  have hx0' : affinePtP x0 W b = 0 := hg 0
  have hW' : reparamW A W = idWP p n := by
    funext t i
    have h1 := congrFun (hg (Pi.single t 1)) i
    have h2 := congrFun (hg 0) i
    simp only [affinePtP, Pi.single_apply, Pi.zero_apply] at h1 h2
    simp only [reparamW, idWP, hA, sub_mul, Finset.sum_sub_distrib]
    have e : (if i = t then (1 : ZMod p) else 0) = if t = i then 1 else 0 := by
      by_cases h : i = t
      · subst h
        simp
      · rw [if_neg h, if_neg (Ne.symm h)]
    rw [← e]
    linear_combination h1 - h2
  have hφ : Function.Injective (reparamY b A) := by
    intro z z' h
    have h' := congrArg (affinePtP x0 W) h
    rw [affinePtP_reparamY, affinePtP_reparamY, hx0', hW', affinePtP_id, affinePtP_id] at h'
    exact h'
  obtain ⟨C, Q', l', h⟩ := stabVecP_reparam x0 W Q l b A hφ
  refine ⟨C, Q', l', fun x => ?_⟩
  rw [h x, hx0', hW']

/-! ### The exponent table of a term -/

/-- The exponent table of a term with pivot columns `piv`: `none` off the
    flat, and on it the phase exponent in `ZMod D` at the unique preimage. -/
def tableOfP (p : ℕ) [NeZero p] {n k : ℕ} (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (piv : Fin k → Fin n) (x : Fin n → ZMod p) :
    Option (ZMod (stabPeriod p)) :=
  if x = affinePtP x0 W (ysol x0 piv x)
    then some ((quadPhaseP Q l (ysol x0 piv x) : ℕ) : ZMod (stabPeriod p)) else none

/-- The amplitude a table entry stands for. -/
noncomputable def tableVal (p : ℕ) : Option (ZMod (stabPeriod p)) → ℂ
  | none => 0
  | some e => zeta p ^ e.val

/-- **A term with pivot columns is its exponent table.** -/
theorem stabVecP_eq_tableVal [NeZero p] {n k : ℕ} (x0 : Fin n → ZMod p)
    (W : Fin k → Fin n → ZMod p) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (piv : Fin k → Fin n)
    (hpiv : ∀ j j', W j (piv j') = if j = j' then 1 else 0) (x : Fin n → ZMod p) :
    stabVecP p n k x0 W Q l x = tableVal p (tableOfP p x0 W Q l piv x) := by
  rw [stabVecP_eq_ite_of_pivots p x0 W Q l piv hpiv, tableOfP]
  split_ifs
  · simp only [tableVal]
    rw [ZMod.val_natCast, ← zeta_pow_mod]
  · rfl

end StabRank
