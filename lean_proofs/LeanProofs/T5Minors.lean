/-
The rank-2 exclusion of `|T5⟩` over the 30 single-ququint stabilizer states,
as `verify_challenge/cert_t5_m1_rank2.py` argues it, machine-checked.

A single-ququint stabilizer state is, up to a scalar, one of 30 exponent
vectors (`Shape`): the five points `|x₀⟩` and the 25 full-support states
`Σ_y ω₅^(q y² + l y) |y⟩`. Every amplitude is `0` or a power of `ω₅`, and so
is every amplitude of `√5 |T5⟩`. If `|T5⟩` lay in the span of two states the
three columns `[s | s' | T5]` would be linearly dependent and every `3 × 3`
minor would vanish. Each minor is a signed sum of six monomials `ω₅^e`, so it
is an element of `ℤ[ω₅]` given by five integer coefficients, and it vanishes
exactly when the five coefficients agree (`omega5_sum_val_eq_zero_iff`).

The computable part (`minorNonzero`) accumulates those coefficients from the
exponent data alone; `det_minorMat` and `evalL_eq` identify its value with the
complex determinant, so `minorNonzero = true` is a proof that the minor is
nonzero. `pairs_ok` is the certificate's search, decided in the kernel: for
every ordered pair of distinct shapes some one of the ten row triples has a
nonzero minor. `t5v_not_mem_span_smul` is the consequence the rank bound
needs: no two scalar multiples of shape vectors span the T5 vector, whatever
the scalars and whether or not the shapes differ (a repeated shape is padded
with a different one, which only enlarges the span).
-/
import LeanProofs.Ququint
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

namespace StabRank
namespace T5

/-! ### The 30 shapes and their exponent vectors -/

/-- Exponent data of a single-ququint stabilizer state up to a scalar: a point
    `|x₀⟩`, or the full-support state with phase `ω₅^(q y² + l y)`. -/
inductive Shape
  | pt (x0 : ZMod 5)
  | full (q l : ZMod 5)
  deriving DecidableEq

/-- The exponent of `ω₅` at position `x`, or `none` for a zero amplitude. -/
def Shape.exp : Shape → ZMod 5 → Option (ZMod 5)
  | .pt x0, x => if x = x0 then some 0 else none
  | .full q l, x => some (q * x * x + l * x)

/-- The exponent vector of `√5 |T5⟩`. -/
def t5exp (x : ZMod 5) : Option (ZMod 5) := some (x ^ 3)

/-- The complex amplitude of an exponent. -/
noncomputable def termVal : Option (ZMod 5) → ℂ
  | none => 0
  | some e => omega5 ^ e.val

/-- The complex vector of a shape. -/
noncomputable def Shape.vec (s : Shape) : ZMod 5 → ℂ := fun x => termVal (s.exp x)

/-- `√5 |T5⟩` as a complex vector on `ZMod 5`. -/
noncomputable def t5v : ZMod 5 → ℂ := fun x => termVal (t5exp x)

/-! ### Minors in `ℤ[ω₅]` -/

/-- The exponent of one monomial of a minor: column `a` at row `i`, `b` at
    `j`, `c` at `k`. -/
def tm (a b c : ZMod 5 → Option (ZMod 5)) (i j k : ZMod 5) : Option (ZMod 5) :=
  match a i, b j, c k with
  | some x, some y, some z => some (x + y + z)
  | _, _, _ => none

/-- The six signed monomials of the `3 × 3` minor at rows `i, j, k`, in the
    order of `Matrix.det_fin_three`. -/
def minorTerms (a b c : ZMod 5 → Option (ZMod 5)) (i j k : ZMod 5) :
    List (ℤ × Option (ZMod 5)) :=
  [(1, tm a b c i j k), (-1, tm a b c i k j), (-1, tm a b c j i k),
   (1, tm a b c k i j), (1, tm a b c j k i), (-1, tm a b c k j i)]

/-- The coefficient of `ω₅^e` in a signed list of monomials. -/
def coeffOf (L : List (ℤ × Option (ZMod 5))) (e : ZMod 5) : ℤ :=
  (L.map fun st => if st.2 = some e then st.1 else 0).sum

/-- The complex value of a signed list of monomials. -/
noncomputable def evalL (L : List (ℤ × Option (ZMod 5))) : ℂ :=
  (L.map fun st => (st.1 : ℂ) * termVal st.2).sum

/-- The equal-coefficient test: the minor is nonzero when the five
    coefficients are not all equal. -/
def minorNonzero (a b c : ZMod 5 → Option (ZMod 5)) (i j k : ZMod 5) : Bool :=
  let L := minorTerms a b c i j k
  decide (coeffOf L 1 ≠ coeffOf L 0 ∨ coeffOf L 2 ≠ coeffOf L 0 ∨
    coeffOf L 3 ≠ coeffOf L 0 ∨ coeffOf L 4 ≠ coeffOf L 0)

/-! ### Bridge to the complex determinant -/

theorem termVal_eq_sum (t : Option (ZMod 5)) :
    termVal t = ∑ e : ZMod 5, (if t = some e then (1 : ℂ) else 0) * omega5 ^ e.val := by
  cases t with
  | none => simp [termVal]
  | some x => simp [termVal, Finset.sum_ite_eq]

theorem evalL_eq (L : List (ℤ × Option (ZMod 5))) :
    evalL L = ∑ e : ZMod 5, (coeffOf L e : ℂ) * omega5 ^ e.val := by
  induction L with
  | nil => simp [evalL, coeffOf]
  | cons st L ih =>
    simp only [evalL, coeffOf, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih, termVal_eq_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun e _ => ?_
    push_cast
    split_ifs <;> ring

theorem termVal_tm (a b c : ZMod 5 → Option (ZMod 5)) (i j k : ZMod 5) :
    termVal (a i) * termVal (b j) * termVal (c k) = termVal (tm a b c i j k) := by
  cases ha : a i <;> cases hb : b j <;> cases hc : c k <;>
    simp [termVal, tm, ha, hb, hc, omega5_pow_val_add]

/-- The `3 × 3` matrix of three exponent columns at rows `i, j, k`. -/
noncomputable def minorMat (a b c : ZMod 5 → Option (ZMod 5)) (i j k : ZMod 5) :
    Matrix (Fin 3) (Fin 3) ℂ :=
  !![termVal (a i), termVal (b i), termVal (c i);
     termVal (a j), termVal (b j), termVal (c j);
     termVal (a k), termVal (b k), termVal (c k)]

theorem det_minorMat (a b c : ZMod 5 → Option (ZMod 5)) (i j k : ZMod 5) :
    (minorMat a b c i j k).det = evalL (minorTerms a b c i j k) := by
  rw [Matrix.det_fin_three]
  simp only [minorMat, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons,
    Matrix.empty_val', Matrix.cons_val_fin_one, Matrix.head_fin_const]
  simp only [evalL, minorTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  have e1 := termVal_tm a b c i j k
  have e2 := termVal_tm a b c i k j
  have e3 := termVal_tm a b c j i k
  have e4 := termVal_tm a b c k i j
  have e5 := termVal_tm a b c j k i
  have e6 := termVal_tm a b c k j i
  push_cast
  linear_combination e1 - e2 - e3 + e4 + e5 - e6

/-- `minorNonzero = true` is a proof that the complex minor is nonzero. -/
theorem det_ne_zero_of_minorNonzero (a b c : ZMod 5 → Option (ZMod 5)) (i j k : ZMod 5)
    (h : minorNonzero a b c i j k = true) : (minorMat a b c i j k).det ≠ 0 := by
  rw [det_minorMat, evalL_eq]
  intro h0
  rw [omega5_sum_val_eq_zero_iff] at h0
  simp only [minorNonzero, decide_eq_true_eq] at h
  have h1 := h0 1
  have h2 := h0 2
  have h3 := h0 3
  have h4 := h0 4
  tauto

/-- If `t` lies in the span of `u` and `v`, every `3 × 3` minor of
    `[u | v | t]` vanishes. -/
theorem det_eq_zero_of_mem_span (u v t : ZMod 5 → ℂ)
    (h : t ∈ Submodule.span ℂ ({u, v} : Set (ZMod 5 → ℂ))) (i j k : ZMod 5) :
    Matrix.det !![u i, v i, t i; u j, v j, t j; u k, v k, t k] = 0 := by
  obtain ⟨α, β, hαβ⟩ := Submodule.mem_span_pair.mp h
  have hi := congrFun hαβ i
  have hj := congrFun hαβ j
  have hk := congrFun hαβ k
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul] at hi hj hk
  rw [Matrix.det_fin_three]
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons, Matrix.empty_val',
    Matrix.cons_val_fin_one, Matrix.head_fin_const]
  rw [← hi, ← hj, ← hk]
  ring

/-! ### The search over the 30 shapes -/

def allZ : List (ZMod 5) := [0, 1, 2, 3, 4]

/-- The 30 shapes. -/
def shapes : List Shape :=
  allZ.map Shape.pt ++ allZ.flatMap fun q => allZ.map fun l => Shape.full q l

/-- The ten row triples of a `5 × 3` matrix. -/
def rowTriples : List (ZMod 5 × ZMod 5 × ZMod 5) :=
  [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
   (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]

theorem mem_allZ (x : ZMod 5) : x ∈ allZ := by
  rcases zmod5_cases x with rfl | rfl | rfl | rfl | rfl <;> simp [allZ]

theorem mem_shapes (s : Shape) : s ∈ shapes := by
  cases s <;> simp [shapes, mem_allZ]

/-- **The certificate's search.** For every pair of distinct shapes some
    `3 × 3` minor of `[s | s' | T5]` is nonzero in `ℤ[ω₅]`. -/
theorem pairs_ok : ∀ s ∈ shapes, ∀ s' ∈ shapes, s ≠ s' →
    (rowTriples.any fun r => minorNonzero s.exp s'.exp t5exp r.1 r.2.1 r.2.2) = true := by
  decide +kernel

/-! ### Consequences for spans -/

/-- No two distinct shape vectors span the T5 vector. -/
theorem t5v_not_mem_span_pair (s s' : Shape) (hne : s ≠ s') :
    t5v ∉ Submodule.span ℂ ({s.vec, s'.vec} : Set (ZMod 5 → ℂ)) := by
  intro hmem
  have h := pairs_ok s (mem_shapes s) s' (mem_shapes s') hne
  rw [List.any_eq_true] at h
  obtain ⟨⟨i, j, k⟩, _, hijk⟩ := h
  exact det_ne_zero_of_minorNonzero _ _ _ i j k hijk
    (det_eq_zero_of_mem_span s.vec s'.vec t5v hmem i j k)

theorem exists_ne_shape (s : Shape) : ∃ s', s ≠ s' := by
  by_cases h : s = Shape.pt 0
  · exact ⟨Shape.pt 1, by rw [h]; decide⟩
  · exact ⟨Shape.pt 0, h⟩

/-- The same with the shapes possibly equal: a repeated shape spans no more
    than it does together with any other. -/
theorem t5v_not_mem_span_pair' (s s' : Shape) :
    t5v ∉ Submodule.span ℂ ({s.vec, s'.vec} : Set (ZMod 5 → ℂ)) := by
  by_cases h : s = s'
  · subst h
    obtain ⟨s'', hne⟩ := exists_ne_shape s
    intro hmem
    apply t5v_not_mem_span_pair s s'' hne
    refine Submodule.span_mono ?_ hmem
    intro x hx
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx ⊢
    tauto
  · exact t5v_not_mem_span_pair s s' h

/-- **No two scalar multiples of shape vectors span the T5 vector.** -/
theorem t5v_not_mem_span_smul (c c' : ℂ) (s s' : Shape) :
    t5v ∉ Submodule.span ℂ ({c • s.vec, c' • s'.vec} : Set (ZMod 5 → ℂ)) := by
  intro hmem
  apply t5v_not_mem_span_pair' s s'
  refine Submodule.span_le.mpr ?_ hmem
  rintro x (rfl | rfl)
  · exact Submodule.smul_mem _ _ (Submodule.subset_span (by simp))
  · exact Submodule.smul_mem _ _ (Submodule.subset_span (by simp))

end T5
end StabRank
