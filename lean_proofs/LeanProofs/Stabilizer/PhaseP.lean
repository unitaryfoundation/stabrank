/-
The phase exponent of a stabilizer state under an affine substitution of its
parameters, for any prime `p`.

`stabVecP` carries the phase `ζ_D ^ quadPhaseP Q l y` with `y ∈ (ZMod p)^k`,
`D = stabPeriod p`, and

  quadPhaseP Q l y = (D / p) · Σ_ij Q_ij y_i y_j + Σ_i l_i y_i       (in ℕ).

Slicing a stabilizer state along one qudit restricts `y` to an affine
hyperplane of `(ZMod p)^k`, which is parametrised by `y = y(z)` with every
coordinate `y_i(z) = α_i + Σ_t N_it z_t` an affine function of `z ∈ (ZMod p)^k'`.
This file shows that `quadPhaseP Q l (y(z))` is, modulo `D`, a constant plus
`quadPhaseP Q' l' z` for some `Q'`, `l'`, so the slice is again a `stabVecP`
up to a scalar.

The computation is done in `ZMod D` rather than in `ℕ` modulo `D`. Two maps
carry the two kinds of term:

- `qcast p u = (D / p) · u.val` in `ZMod D` is additive in `u ∈ ZMod p`,
  because `(D / p) · n` mod `D` depends only on `n` mod `p`. The quadratic
  part of the phase is a sum of `qcast (Q_ij y_i y_j)`, and after the
  substitution each such term is `qcast` of a quadratic polynomial in `z`
  over `ZMod p`, which `qcast_add` and `qcast_mul` unfold into the required
  shape without any case analysis on `p`.
- `liftD p u = u.val` in `ZMod D` carries the linear part `l_i · y_i.val`. For
  odd `p` we have `D = p`, so `liftD = qcast` and the same argument applies.
  For `p = 2` we have `D = 4`, and `liftD` is not additive: the sum of two
  bits lifts as `a + b + 2ab` in `ZMod 4`. The correction `2ab` is a
  `qcast` term, so a linear form in `y` becomes a linear form plus a
  quadratic form in `z`; this is why qubit stabilizer states need fourth
  roots of unity in the linear part. The lift of an affine function of `z`
  is handled by induction on the set of summands.

`IsPhaseExp p g` says `g : (Fin k' → ZMod p) → ZMod D` is a constant plus
`phaseZ p Q' l'`, the `ZMod D` reading of `quadPhaseP Q' l'`. The class is
closed under constants, sums, `qcast` of a product of two affine functions,
and `l · liftD` of an affine function, and `isPhaseExp_phaseZ_comp` assembles
the substituted phase from these. `zeta_pow_quadPhaseP_comp` converts back to
the powers of `ζ_D` that `stabVecP` uses.
-/
import LeanProofs.Stabilizer.IsStabP

namespace StabRank

open Stabilizer

variable {p : ℕ}

/-! ### The period for odd `p`, and `(D / p) · p = D` -/

theorem stabPeriod_eq_of_ne_two (hp : p ≠ 2) : stabPeriod p = p := by
  unfold stabPeriod
  split
  · exact absurd rfl hp
  · rfl

instance stabPeriod_neZero [NeZero p] : NeZero (stabPeriod p) := ⟨stabPeriod_ne_zero p⟩

theorem stabPeriod_div_mul_self [NeZero p] : stabPeriod p / p * p = stabPeriod p := by
  rcases eq_or_ne p 2 with rfl | h2
  · rfl
  · rw [stabPeriod_eq_of_ne_two h2, Nat.div_self (NeZero.pos p), one_mul]

theorem stabPeriod_div_eq_one_of_ne_two (hp : p ≠ 2) [NeZero p] : stabPeriod p / p = 1 := by
  rw [stabPeriod_eq_of_ne_two hp, Nat.div_self (NeZero.pos p)]

/-- `(D / p) · n` in `ZMod D` depends only on `n` mod `p`. -/
theorem natCast_div_mul_mod [NeZero p] (n : ℕ) :
    ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p)) * (n : ZMod (stabPeriod p))
      = ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p))
        * ((n % p : ℕ) : ZMod (stabPeriod p)) := by
  have h : stabPeriod p / p * n = stabPeriod p * (n / p) + stabPeriod p / p * (n % p) := by
    conv_lhs => rw [← Nat.div_add_mod n p]
    rw [mul_add, ← mul_assoc, stabPeriod_div_mul_self]
  have h' := congrArg (fun m : ℕ => (m : ZMod (stabPeriod p))) h
  simp only [Nat.cast_add, Nat.cast_mul, ZMod.natCast_self, zero_mul, zero_add] at h'
  exact h'

/-! ### The two lifts `ZMod p → ZMod D` -/

/-- `u.val` read in `ZMod D`. -/
def liftD (p : ℕ) (u : ZMod p) : ZMod (stabPeriod p) := (u.val : ZMod (stabPeriod p))

/-- `(D / p) · u.val` read in `ZMod D`: the coefficient map of the quadratic part. -/
def qcast (p : ℕ) (u : ZMod p) : ZMod (stabPeriod p) :=
  ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p)) * (u.val : ZMod (stabPeriod p))

theorem qcast_natCast [NeZero p] (n : ℕ) :
    qcast p (n : ZMod p)
      = ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p)) * (n : ZMod (stabPeriod p)) := by
  unfold qcast
  rw [ZMod.val_natCast, ← natCast_div_mul_mod]

theorem qcast_zero [NeZero p] : qcast p 0 = 0 := by
  simp [qcast]

theorem qcast_add [NeZero p] (u v : ZMod p) : qcast p (u + v) = qcast p u + qcast p v := by
  conv_lhs => rw [← ZMod.natCast_zmod_val u, ← ZMod.natCast_zmod_val v, ← Nat.cast_add,
    qcast_natCast]
  unfold qcast
  push_cast
  ring

theorem qcast_mul [NeZero p] (u v : ZMod p) :
    qcast p (u * v)
      = ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p)) * (liftD p u * liftD p v) := by
  conv_lhs => rw [← ZMod.natCast_zmod_val u, ← ZMod.natCast_zmod_val v, ← Nat.cast_mul,
    qcast_natCast]
  unfold liftD
  push_cast
  ring

theorem qcast_mul_mul [NeZero p] (a b c : ZMod p) :
    qcast p (a * b * c) = ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p)) *
      ((a.val : ZMod (stabPeriod p)) * (b.val : ZMod (stabPeriod p))
        * (c.val : ZMod (stabPeriod p))) := by
  conv_lhs => rw [← ZMod.natCast_zmod_val a, ← ZMod.natCast_zmod_val b,
    ← ZMod.natCast_zmod_val c, ← Nat.cast_mul, ← Nat.cast_mul, qcast_natCast]
  push_cast
  ring

theorem qcast_mul_mul' [NeZero p] (a b c : ZMod p) :
    qcast p (a * b * c) = ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p)) *
      (liftD p a * liftD p b * liftD p c) :=
  qcast_mul_mul a b c

/-- `qcast` as an additive map. -/
def qcastHom (p : ℕ) [NeZero p] : ZMod p →+ ZMod (stabPeriod p) where
  toFun := qcast p
  map_zero' := qcast_zero
  map_add' := qcast_add

theorem qcast_sum [NeZero p] {ι : Type*} (s : Finset ι) (f : ι → ZMod p) :
    qcast p (∑ i ∈ s, f i) = ∑ i ∈ s, qcast p (f i) :=
  map_sum (qcastHom p) f s

/-- For odd `p`, `D / p = 1` and the two lifts agree. -/
theorem liftD_eq_qcast_of_ne_two (hp : p ≠ 2) [NeZero p] (u : ZMod p) :
    liftD p u = qcast p u := by
  simp [liftD, qcast, stabPeriod_div_eq_one_of_ne_two hp]

/-! ### The phase read in `ZMod D` -/

/-- `quadPhaseP Q l z` read in `ZMod D`, term by term. -/
def phaseZ (p : ℕ) {k : ℕ} (Q : Fin k → Fin k → ZMod p) (l : Fin k → ZMod (stabPeriod p))
    (z : Fin k → ZMod p) : ZMod (stabPeriod p) :=
  ∑ i, ∑ j, qcast p (Q i j * z i * z j) + ∑ i, l i * liftD p (z i)

theorem natCast_quadPhaseP [NeZero p] {k : ℕ} (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (z : Fin k → ZMod p) :
    ((quadPhaseP Q l z : ℕ) : ZMod (stabPeriod p)) = phaseZ p Q l z := by
  unfold quadPhaseP phaseZ
  rw [Nat.cast_add, Nat.cast_mul, Nat.cast_sum, Nat.cast_sum, Finset.mul_sum]
  congr 1
  · refine Finset.sum_congr rfl fun i _ => ?_
    rw [Nat.cast_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [qcast_mul_mul, Nat.cast_mul, Nat.cast_mul]
  · refine Finset.sum_congr rfl fun i _ => ?_
    rw [Nat.cast_mul, ZMod.natCast_zmod_val]
    rfl

theorem phaseZ_add [NeZero p] {k : ℕ} (Q Q' : Fin k → Fin k → ZMod p)
    (l l' : Fin k → ZMod (stabPeriod p)) (z : Fin k → ZMod p) :
    phaseZ p (Q + Q') (l + l') z = phaseZ p Q l z + phaseZ p Q' l' z := by
  unfold phaseZ
  simp only [Pi.add_apply, add_mul, qcast_add, Finset.sum_add_distrib]
  ring

/-! ### Functions of the form `constant + phaseZ` -/

/-- `g z` is a constant plus a quadratic-linear phase in `z`, in `ZMod D`. -/
def IsPhaseExp (p : ℕ) {k : ℕ} (g : (Fin k → ZMod p) → ZMod (stabPeriod p)) : Prop :=
  ∃ (C : ZMod (stabPeriod p)) (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)), ∀ z, g z = C + phaseZ p Q l z

theorem IsPhaseExp.const [NeZero p] {k : ℕ} (c : ZMod (stabPeriod p)) :
    IsPhaseExp p (fun _ : Fin k → ZMod p => c) :=
  ⟨c, 0, 0, fun z => by simp [phaseZ, qcast_zero]⟩

theorem IsPhaseExp.add [NeZero p] {k : ℕ} {g h : (Fin k → ZMod p) → ZMod (stabPeriod p)}
    (hg : IsPhaseExp p g) (hh : IsPhaseExp p h) : IsPhaseExp p (fun z => g z + h z) := by
  obtain ⟨C, Q, l, hg⟩ := hg
  obtain ⟨C', Q', l', hh⟩ := hh
  refine ⟨C + C', Q + Q', l + l', fun z => ?_⟩
  change g z + h z = _
  rw [hg, hh, phaseZ_add]
  ring

theorem IsPhaseExp.sum [NeZero p] {k : ℕ} {ι : Type*} (s : Finset ι)
    (f : ι → (Fin k → ZMod p) → ZMod (stabPeriod p)) (hf : ∀ i ∈ s, IsPhaseExp p (f i)) :
    IsPhaseExp p (fun z => ∑ i ∈ s, f i z) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using IsPhaseExp.const (p := p) (k := k) 0
  | insert a s ha ih =>
    simp only [Finset.sum_insert ha]
    exact (hf a (Finset.mem_insert_self a s)).add
      (ih fun i hi => hf i (Finset.mem_insert_of_mem hi))

/-- `c · liftD (z t)`: a single linear term. -/
theorem IsPhaseExp.mul_liftD_single [NeZero p] {k : ℕ} (c : ZMod (stabPeriod p)) (t : Fin k) :
    IsPhaseExp p (fun z => c * liftD p (z t)) := by
  refine ⟨0, 0, Pi.single t c, fun z => ?_⟩
  simp [phaseZ, qcast_zero, Pi.single_apply, ite_mul]

/-- The affine function `z ↦ α + Σ_t N t · z t` over `ZMod p`. -/
def affineP {k : ℕ} (α : ZMod p) (N : Fin k → ZMod p) (z : Fin k → ZMod p) : ZMod p :=
  α + ∑ t, N t * z t

/-- `qcast` of a constant times two affine functions is a phase. -/
theorem IsPhaseExp.qcast_mul_affine_affine [NeZero p] {k : ℕ} (c α β : ZMod p)
    (N M : Fin k → ZMod p) :
    IsPhaseExp p (fun z => qcast p (c * affineP α N z * affineP β M z)) := by
  refine ⟨qcast p (c * α * β), fun t t' => c * N t * M t',
    fun t => ((stabPeriod p / p : ℕ) : ZMod (stabPeriod p))
      * liftD p (c * α * M t + c * β * N t), fun z => ?_⟩
  have hexp : c * affineP α N z * affineP β M z
      = c * α * β + ∑ t, (c * α * M t + c * β * N t) * z t
        + ∑ t, ∑ t', c * N t * M t' * z t * z t' := by
    unfold affineP
    have e1 : (∑ t, N t * z t) * (∑ t', M t' * z t')
        = ∑ t, ∑ t', N t * z t * (M t' * z t') :=
      Finset.sum_mul_sum _ _ _ _
    calc c * (α + ∑ t, N t * z t) * (β + ∑ t, M t * z t)
        = c * α * β + (c * α * ∑ t, M t * z t + c * β * ∑ t, N t * z t)
          + c * ((∑ t, N t * z t) * (∑ t, M t * z t)) := by ring
      _ = _ := by
        rw [e1]
        simp only [Finset.mul_sum]
        rw [← Finset.sum_add_distrib]
        congr 1
        · congr 1
          refine Finset.sum_congr rfl fun t _ => ?_
          ring
        · refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun t' _ => ?_
          ring
  change qcast p (c * affineP α N z * affineP β M z) = _
  rw [hexp]
  unfold phaseZ
  simp only [qcast_add, qcast_sum, qcast_mul]
  ring_nf

/-- `l · liftD` of an affine function is a phase. For odd `p` the lift is
    additive; for `p = 2` the sum of bits lifts with a `qcast` correction,
    handled by induction on the set of summands. -/
theorem IsPhaseExp.mul_liftD_affine [Fact p.Prime] {k : ℕ} (l : ZMod (stabPeriod p))
    (α : ZMod p) (N : Fin k → ZMod p) :
    IsPhaseExp p (fun z => l * liftD p (affineP α N z)) := by
  rcases eq_or_ne p 2 with rfl | hp2
  · have h2add : ∀ u v : ZMod 2,
        liftD 2 (u + v) = liftD 2 u + liftD 2 v + 2 * (liftD 2 u * liftD 2 v) := by
      intro u v
      rcases zmod2_cases u with rfl | rfl <;> rcases zmod2_cases v with rfl | rfl <;> decide
    have h2mul : ∀ u v : ZMod 2, liftD 2 (u * v) = liftD 2 u * liftD 2 v := by
      intro u v
      rcases zmod2_cases u with rfl | rfl <;> rcases zmod2_cases v with rfl | rfl <;> decide
    have h2l : ∀ l : ZMod (stabPeriod 2), 2 * liftD 2 (l.val : ZMod 2) = 2 * l := by
      intro l
      have h := natCast_div_mul_mod (p := 2) l.val
      rw [stabPeriod_two_div, Nat.cast_ofNat, ZMod.natCast_zmod_val] at h
      unfold liftD
      rw [ZMod.val_natCast, ← h]
    have key : ∀ s : Finset (Fin k),
        IsPhaseExp 2 (fun z => l * liftD 2 (α + ∑ t ∈ s, N t * z t)) := by
      intro s
      induction s using Finset.induction_on with
      | empty => simpa using IsPhaseExp.const (p := 2) (k := k) (l * liftD 2 α)
      | insert t s ht ih =>
        have hU : ∀ z : Fin k → ZMod 2, α + ∑ t' ∈ s, N t' * z t'
            = affineP α (fun t' => if t' ∈ s then N t' else 0) z := by
          intro z
          unfold affineP
          simp only [ite_mul, zero_mul, Finset.sum_ite_mem, Finset.univ_inter]
        have hV : ∀ z : Fin k → ZMod 2, N t * z t = affineP 0 (Pi.single t (N t)) z := by
          intro z
          simp [affineP, Pi.single_apply, ite_mul]
        have hstep : ∀ z : Fin k → ZMod 2, l * liftD 2 (α + ∑ t' ∈ insert t s, N t' * z t')
            = l * liftD 2 (α + ∑ t' ∈ s, N t' * z t') + (l * liftD 2 (N t)) * liftD 2 (z t)
              + qcast 2 ((l.val : ZMod 2) * affineP α (fun t' => if t' ∈ s then N t' else 0) z
                  * affineP 0 (Pi.single t (N t)) z) := by
          intro z
          rw [Finset.sum_insert ht, ← hU, ← hV, qcast_mul_mul', stabPeriod_two_div,
            Nat.cast_ofNat, ← mul_assoc (2 : ZMod (stabPeriod 2)),
            ← mul_assoc (2 : ZMod (stabPeriod 2)), h2l,
            show α + (N t * z t + ∑ t' ∈ s, N t' * z t')
              = (α + ∑ t' ∈ s, N t' * z t') + N t * z t by ring,
            h2add, h2mul]
          ring
        rw [funext hstep]
        exact (ih.add (IsPhaseExp.mul_liftD_single _ _)).add
          (IsPhaseExp.qcast_mul_affine_affine _ _ _ _ _)
    exact key Finset.univ
  · have hlift : ∀ u : ZMod p, liftD p u = qcast p u := liftD_eq_qcast_of_ne_two hp2
    refine ⟨l * qcast p α, 0, fun t => l * liftD p (N t), fun z => ?_⟩
    change l * liftD p (affineP α N z) = _
    rw [hlift, affineP, qcast_add, qcast_sum, mul_add, Finset.mul_sum]
    unfold phaseZ
    simp only [Pi.zero_apply, zero_mul, qcast_zero, Finset.sum_const_zero, zero_add,
      qcast_mul, stabPeriod_div_eq_one_of_ne_two hp2, Nat.cast_one, one_mul, mul_assoc]

/-! ### The substituted phase -/

/-- **The phase after an affine substitution of the parameters is a constant
    plus a phase.** Every coordinate `y_i(z) = affineP (αv i) (Nv i) z`. -/
theorem isPhaseExp_phaseZ_comp [Fact p.Prime] {k k' : ℕ} (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (αv : Fin k → ZMod p)
    (Nv : Fin k → Fin k' → ZMod p) :
    IsPhaseExp p
      (fun z : Fin k' → ZMod p => phaseZ p Q l (fun i => affineP (αv i) (Nv i) z)) := by
  simp only [phaseZ]
  refine IsPhaseExp.add (IsPhaseExp.sum _ _ fun i _ => IsPhaseExp.sum _ _ fun j _ => ?_)
    (IsPhaseExp.sum _ _ fun i _ => ?_)
  · exact IsPhaseExp.qcast_mul_affine_affine (Q i j) (αv i) (αv j) (Nv i) (Nv j)
  · exact IsPhaseExp.mul_liftD_affine (l i) (αv i) (Nv i)

/-- The same statement for the powers of `ζ_D` that `stabVecP` uses. -/
theorem zeta_pow_quadPhaseP_comp [Fact p.Prime] {k k' : ℕ} (Q : Fin k → Fin k → ZMod p)
    (l : Fin k → ZMod (stabPeriod p)) (αv : Fin k → ZMod p)
    (Nv : Fin k → Fin k' → ZMod p) :
    ∃ (C : ℕ) (Q' : Fin k' → Fin k' → ZMod p) (l' : Fin k' → ZMod (stabPeriod p)),
      ∀ z, zeta p ^ quadPhaseP Q l (fun i => affineP (αv i) (Nv i) z)
        = zeta p ^ C * zeta p ^ quadPhaseP Q' l' z := by
  obtain ⟨C, Q', l', h⟩ := isPhaseExp_phaseZ_comp Q l αv Nv
  refine ⟨C.val, Q', l', fun z => ?_⟩
  rw [← pow_add]
  apply zeta_pow_eq_of_mod
  rw [← ZMod.natCast_eq_natCast_iff', Nat.cast_add, natCast_quadPhaseP,
    natCast_quadPhaseP, ZMod.natCast_zmod_val]
  exact h z

end StabRank
