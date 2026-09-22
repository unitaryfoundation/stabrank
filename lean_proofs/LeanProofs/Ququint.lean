/-
The ququint foundation: `ω₅ = exp(2πi/5)` and the relations the T5 cells
use, the T5 state on digit strings, and the `ZMod 5` bookkeeping lemmas.

Everything a T5 cell computes lives in `ℤ[ω₅]`: the amplitudes of `|T5⟩`
(up to `1/√5`) and of every single-ququint stabilizer state are `0` or a
power of `ω₅`, and the coefficients of the stored decompositions are integer
combinations of `1, ω₅, ω₅², ω₅³`. Three facts about `ω₅` carry the proofs:

* `ω₅⁵ = 1`, which reduces every exponent mod 5 (`omega5_pow_reduce`,
  `zeta_five_pow`);
* `1 + ω₅ + ω₅² + ω₅³ + ω₅⁴ = 0` (`omega5_geom_sum`), which rewrites `ω₅⁴`
  into the lower powers (`omega5_pow_four`) so that a polynomial identity in
  `ω₅` can be closed by `ring` on its reduced form;
* `{1, ω₅, ω₅², ω₅³}` is linearly independent over `ℚ`
  (`omega5_linearIndependent`, from the fifth cyclotomic polynomial being the
  minimal polynomial of `ω₅`), which gives the zero test of
  `verify_challenge/cert_t5_m1_rank2.py`: an integer combination
  `Σ cₖ ω₅ᵏ` over `k < 5` vanishes if and only if all `cₖ` are equal
  (`omega5_int_comb_eq_zero_iff`, `omega5_sum_val_eq_zero_iff`).

`t5Amp x = ω₅^(x³)/√5` is the amplitude of `|T5⟩ = 5^(-1/2) Σ_x ω₅^(x³)|x⟩`
of `orbit_state("T5")` in `verify_challenge/stabrank_verify.py`, and
`t5Vec m` is `|T5⟩^⊗m` on digit strings, in the style of `tVec` in
`QubitShared.lean`.
-/
import LeanProofs.Stabilizer.IsStabP
import Mathlib.RingTheory.RootsOfUnity.Complex
import Mathlib.RingTheory.Polynomial.Cyclotomic.Roots
import Mathlib.RingTheory.PowerBasis

namespace StabRank

open Complex

instance fact_prime_five : Fact (Nat.Prime 5) := ⟨by norm_num⟩

/-! ### `ω₅` and its relations -/

/-- Primitive fifth root of unity. -/
noncomputable def omega5 : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem omega5_primitive : IsPrimitiveRoot omega5 5 := by
  have h := Complex.isPrimitiveRoot_exp 5 (by norm_num)
  have : ((5 : ℕ) : ℂ) = (5 : ℂ) := by norm_num
  rwa [this] at h

theorem omega5_pow_five : omega5 ^ 5 = 1 := omega5_primitive.pow_eq_one

theorem omega5_ne_zero : omega5 ≠ 0 := Complex.exp_ne_zero _

/-- The cyclotomic relation `1 + ω + ω² + ω³ + ω⁴ = 0`. -/
theorem omega5_geom_sum : 1 + omega5 + omega5 ^ 2 + omega5 ^ 3 + omega5 ^ 4 = 0 := by
  have h := omega5_primitive.geom_sum_eq_zero (by norm_num)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, pow_zero, pow_one, zero_add] at h
  exact h

/-- `ω⁴` in terms of the lower powers; with `omega5_pow_reduce` this puts
    every polynomial in `ω` into the form `a + bω + cω² + dω³`. -/
theorem omega5_pow_four : omega5 ^ 4 = -1 - omega5 - omega5 ^ 2 - omega5 ^ 3 := by
  linear_combination omega5_geom_sum

theorem omega5_pow_mod (n : ℕ) : omega5 ^ n = omega5 ^ (n % 5) :=
  pow_eq_pow_mod n omega5_pow_five

/-- `ω^n = ω^(n-5)` for `n ≥ 5`; with `n` a numeral, `simp` discharges the side
    condition and reduces the exponent without looping. -/
theorem omega5_pow_reduce {n : ℕ} (h : 5 ≤ n) : omega5 ^ n = omega5 ^ (n - 5) := by
  conv_lhs => rw [← Nat.sub_add_cancel h, pow_add, omega5_pow_five, mul_one]

/-- Exponents in `ZMod 5` add. -/
theorem omega5_pow_val_add (x y : ZMod 5) :
    omega5 ^ (x + y).val = omega5 ^ x.val * omega5 ^ y.val := by
  rw [ZMod.val_add, ← omega5_pow_mod, pow_add]

theorem stabPeriod_five : stabPeriod 5 = 5 := rfl
theorem stabPeriod_five_div : stabPeriod 5 / 5 = 1 := rfl

theorem zeta_five : zeta 5 = omega5 := by
  unfold zeta omega5
  rw [stabPeriod_five]
  norm_num

/-- A power of `ζ₅` as a power of `ω₅` with the exponent read in `ZMod 5`. -/
theorem zeta_five_pow (n : ℕ) : zeta 5 ^ n = omega5 ^ (n : ZMod 5).val := by
  rw [zeta_five, ZMod.val_natCast, omega5_pow_mod]

/-! ### Linear independence of `1, ω, ω², ω³` over `ℚ` -/

theorem omega5_isIntegral : IsIntegral ℚ omega5 :=
  ⟨Polynomial.X ^ 5 - Polynomial.C 1, Polynomial.monic_X_pow_sub_C 1 (by norm_num),
    by simp [Polynomial.eval₂_sub, Polynomial.eval₂_X_pow, omega5_pow_five]⟩

/-- The minimal polynomial of `ω₅` over `ℚ` is the fifth cyclotomic
    polynomial, of degree `φ(5) = 4`. -/
theorem minpoly_omega5_natDegree : (minpoly ℚ omega5).natDegree = 4 := by
  rw [← Polynomial.cyclotomic_eq_minpoly_rat omega5_primitive (by norm_num),
    Polynomial.natDegree_cyclotomic, Nat.totient_prime (by norm_num)]

theorem omega5_linearIndependent :
    LinearIndependent ℚ (fun i : Fin 4 => omega5 ^ (i : ℕ)) := by
  have h := linearIndependent_pow (K := ℚ) omega5
  rw [minpoly_omega5_natDegree] at h
  exact h

/-- A rational combination of `1, ω, ω², ω³` vanishes only trivially. -/
theorem omega5_rat_comb_eq_zero {a b c d : ℚ}
    (h : (a : ℂ) + b * omega5 + c * omega5 ^ 2 + d * omega5 ^ 3 = 0) :
    a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  have hli := Fintype.linearIndependent_iff.mp omega5_linearIndependent ![a, b, c, d] ?_
  · exact ⟨hli 0, hli 1, hli 2, hli 3⟩
  · simp only [Fin.sum_univ_four, Rat.smul_def, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons, Matrix.cons_val_three,
      Fin.val_zero, Fin.val_one, Fin.val_two, pow_zero, pow_one]
    have : ((3 : Fin 4) : ℕ) = 3 := rfl
    rw [this]
    linear_combination h

/-- **The equal-coefficient zero test.** An integer combination of
    `1, ω, ω², ω³, ω⁴` is zero if and only if its five coefficients agree. -/
theorem omega5_int_comb_eq_zero_iff (c0 c1 c2 c3 c4 : ℤ) :
    (c0 : ℂ) + c1 * omega5 + c2 * omega5 ^ 2 + c3 * omega5 ^ 3 + c4 * omega5 ^ 4 = 0 ↔
      c1 = c0 ∧ c2 = c0 ∧ c3 = c0 ∧ c4 = c0 := by
  constructor
  · intro h
    have h' : (((c0 - c4 : ℤ) : ℚ) : ℂ) + ((c1 - c4 : ℤ) : ℚ) * omega5
        + ((c2 - c4 : ℤ) : ℚ) * omega5 ^ 2 + ((c3 - c4 : ℤ) : ℚ) * omega5 ^ 3 = 0 := by
      push_cast
      linear_combination h - (c4 : ℂ) * omega5_geom_sum
    obtain ⟨h0, h1, h2, h3⟩ := omega5_rat_comb_eq_zero h'
    have e0 : c0 - c4 = 0 := by exact_mod_cast h0
    have e1 : c1 - c4 = 0 := by exact_mod_cast h1
    have e2 : c2 - c4 = 0 := by exact_mod_cast h2
    have e3 : c3 - c4 = 0 := by exact_mod_cast h3
    omega
  · rintro ⟨rfl, rfl, rfl, rfl⟩
    linear_combination (c4 : ℂ) * omega5_geom_sum

/-! ### `ZMod 5` bookkeeping -/

theorem zmod5_val_zero : (0 : ZMod 5).val = 0 := rfl
theorem zmod5_val_one : (1 : ZMod 5).val = 1 := rfl
theorem zmod5_val_two : (2 : ZMod 5).val = 2 := rfl
theorem zmod5_val_three : (3 : ZMod 5).val = 3 := rfl
theorem zmod5_val_four : (4 : ZMod 5).val = 4 := rfl

theorem zmodP5_val_zero : (0 : ZMod (stabPeriod 5)).val = 0 := rfl
theorem zmodP5_val_one : (1 : ZMod (stabPeriod 5)).val = 1 := rfl
theorem zmodP5_val_two : (2 : ZMod (stabPeriod 5)).val = 2 := rfl
theorem zmodP5_val_three : (3 : ZMod (stabPeriod 5)).val = 3 := rfl
theorem zmodP5_val_four : (4 : ZMod (stabPeriod 5)).val = 4 := rfl

/-- The five residues mod 5, for case splits that substitute numerals. -/
theorem zmod5_cases (a : ZMod 5) : a = 0 ∨ a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 := by
  decide +revert

theorem sum_zmod5 (f : ZMod 5 → ℂ) : ∑ e : ZMod 5, f e = f 0 + f 1 + f 2 + f 3 + f 4 := by
  rw [show (Finset.univ : Finset (ZMod 5)) = {0, 1, 2, 3, 4} by decide]
  simp (config := { decide := true }) [Finset.sum_insert]
  ring

theorem forall_zmod5 (P : ZMod 5 → Prop) : (∀ e, P e) ↔ P 0 ∧ P 1 ∧ P 2 ∧ P 3 ∧ P 4 := by
  constructor
  · intro h
    exact ⟨h 0, h 1, h 2, h 3, h 4⟩
  · rintro ⟨h0, h1, h2, h3, h4⟩ e
    rcases zmod5_cases e with rfl | rfl | rfl | rfl | rfl <;> assumption

/-- The zero test for a coefficient function on `ZMod 5`. -/
theorem omega5_sum_val_eq_zero_iff (c : ZMod 5 → ℤ) :
    ∑ e : ZMod 5, (c e : ℂ) * omega5 ^ e.val = 0 ↔ ∀ e, c e = c 0 := by
  rw [sum_zmod5, forall_zmod5]
  simp only [zmod5_val_zero, zmod5_val_one, zmod5_val_two, zmod5_val_three, zmod5_val_four,
    pow_zero, pow_one, mul_one, true_and]
  exact omega5_int_comb_eq_zero_iff (c 0) (c 1) (c 2) (c 3) (c 4)

/-- Digit strings on one and two ququints as explicit vectors. -/
theorem fin1_eta_z5 (x : Fin 1 → ZMod 5) : x = ![x 0] := by
  funext i
  fin_cases i
  rfl

theorem fin2_eta_z5 (x : Fin 2 → ZMod 5) : x = ![x 0, x 1] := by
  funext i
  fin_cases i <;> rfl

theorem fin1_ext_iff (x y : Fin 1 → ZMod 5) : x = y ↔ x 0 = y 0 := by
  constructor
  · intro h
    rw [h]
  · intro h
    funext i
    rw [Fin.fin_one_eq_zero i, h]

theorem sum_fin1_zmod5 (f : (Fin 1 → ZMod 5) → ℂ) :
    ∑ y : Fin 1 → ZMod 5, f y = f ![0] + f ![1] + f ![2] + f ![3] + f ![4] := by
  rw [show (Finset.univ : Finset (Fin 1 → ZMod 5)) = {![0], ![1], ![2], ![3], ![4]}
    by decide +kernel]
  simp (config := { decide := true }) [Finset.sum_insert]
  ring

/-- `stabVecP 5` with `k = 1`: five terms. -/
theorem stabVecP_five_k1 (n : ℕ) (x0 : Fin n → ZMod 5) (W : Fin 1 → Fin n → ZMod 5)
    (Q : Fin 1 → Fin 1 → ZMod 5) (l : Fin 1 → ZMod (stabPeriod 5)) (x : Fin n → ZMod 5) :
    stabVecP 5 n 1 x0 W Q l x
      = (if x = affinePtP x0 W ![0] then zeta 5 ^ quadPhaseP Q l ![0] else 0)
      + (if x = affinePtP x0 W ![1] then zeta 5 ^ quadPhaseP Q l ![1] else 0)
      + (if x = affinePtP x0 W ![2] then zeta 5 ^ quadPhaseP Q l ![2] else 0)
      + (if x = affinePtP x0 W ![3] then zeta 5 ^ quadPhaseP Q l ![3] else 0)
      + (if x = affinePtP x0 W ![4] then zeta 5 ^ quadPhaseP Q l ![4] else 0) := by
  unfold stabVecP
  rw [sum_fin1_zmod5]

/-! ### The T5 state -/

theorem sqrt5_sq_c : (Real.sqrt 5 : ℂ) ^ 2 = 5 := by
  rw [← Complex.ofReal_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
  norm_num

theorem sqrt5_ne_zero_c : (Real.sqrt 5 : ℂ) ≠ 0 := by
  intro h
  have := congrArg (· ^ 2) h
  simp only [sqrt5_sq_c, zero_pow two_ne_zero] at this
  norm_num at this

/-- The cube of a residue, as the exponent of `ω₅`: the Clifford-orbit
    representative `|T5⟩ = 5^(-1/2) Σ_x ω₅^(x³) |x⟩`. -/
def cubeExp (x : ZMod 5) : ℕ := (x ^ 3).val

theorem cubeExp_zero : cubeExp 0 = 0 := rfl
theorem cubeExp_one : cubeExp 1 = 1 := rfl
theorem cubeExp_two : cubeExp 2 = 3 := rfl
theorem cubeExp_three : cubeExp 3 = 2 := rfl
theorem cubeExp_four : cubeExp 4 = 4 := rfl

/-- The one-ququint amplitude of `|T5⟩`. -/
noncomputable def t5Amp (x : ZMod 5) : ℂ := omega5 ^ cubeExp x / (Real.sqrt 5 : ℂ)

theorem t5Amp_ne_zero (x : ZMod 5) : t5Amp x ≠ 0 :=
  div_ne_zero (pow_ne_zero _ omega5_ne_zero) sqrt5_ne_zero_c

/-- `|T5⟩^⊗m` on digit strings. -/
noncomputable def t5Vec (m : ℕ) : Fin (5 ^ m) → ℂ :=
  fun idx => ∏ i, t5Amp (digitsP 5 m idx i)

theorem t5Vec_ne_zero (m : ℕ) (idx : Fin (5 ^ m)) : t5Vec m idx ≠ 0 :=
  Finset.prod_ne_zero_iff.mpr fun _ _ => t5Amp_ne_zero _

/-- Two amplitudes multiply to a power of `ω₅` over `5`. -/
theorem t5Amp_mul (a b : ZMod 5) :
    t5Amp a * t5Amp b = omega5 ^ cubeExp a * omega5 ^ cubeExp b / 5 := by
  unfold t5Amp
  rw [div_mul_div_comm, ← sq, sqrt5_sq_c]

end StabRank
