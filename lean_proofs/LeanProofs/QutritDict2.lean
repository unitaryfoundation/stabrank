/-
The dictionary of two-qutrit stabilizer states, and its completeness.

`verify_challenge/rank_exclusion.py` builds, for every prime `p` and `n`
qudits, the list of `p^n Π_{j ≤ n} (p^j + 1)` stabilizer states up to phase
and checks every exclusion against it. For a lower bound at the Lean tier
the list has to be proved complete: every `IsStabP 3` vector on two qutrits
is a nonzero multiple of a listed state. This file does that for `n = 2`,
where the count is `9 · 4 · 10 = 360`.

A state is stored as its exponent table (`tableOfP` of `Reparam.lean`): a
function on digit strings, `none` off the flat and `some e` on it with `ω^e`
the amplitude. The normal forms, by the number `k` of generators:

* `k = 0`, points `|x₀⟩`: `9`.
* `k = 1`, lines: `W` in reduced row echelon form, `![![1, a]]` with pivot
  column `0` or `![![0, 1]]` with pivot column `1`, `x₀` zero on the pivot
  column, a phase `q y² + m y`: `(3 · 3 + 3) · 9 = 108`.
* `k = 2`, full support: `x₀ = 0`, `W = I`, `Q` upper triangular, `l` free:
  `27 · 9 = 243`.

Completeness (`isStabP_two_qutrits`). Injectivity of the parametrisation
gives `k ≤ 2`. A point is its own normal form. A line has `W 0 ≠ 0`; if
`W 0 0 ≠ 0` the change of coordinates `y = (z - x₀ 0) / W 0 0` (an instance
of `stabVecP_reparam`) makes the pivot entry `1` and the base point zero
there, and symmetrically for the other column. Full support is
`stabVecP_full_normal`, followed by folding `Q` to upper triangular form,
which does not change the phase (`quadPhaseP_upper2`, decided by the
a polynomial identity in `ZMod 3` once the values are cast back). In each case the phase constant is
absorbed into the scalar.

Distinctness. Each table is normalised so that the exponent at the base
point (`x₀`, or `0` for full support) is `0`, so two listed tables give the
same state up to a scalar only if they are equal; `dict2_nodup` shows the
`360` tables are pairwise distinct (each table is packed into an integer key,
the kernel checks the computed keys against the literals of
`QutritDict2Keys.lean` and, there, that the literals are distinct), which is
the Python count.

What is not here: the reduced row echelon form for `0 < k < n` at general
`n`, which the three-qutrit dictionary (`30240` states) needs, and the
proof that distinct tables are not scalar multiples of each other, stated
only in this comment.
-/
import LeanProofs.Stabilizer.Reparam
import LeanProofs.QutritDict2Keys

namespace StabRank

open Stabilizer

/-- Exponent tables on two qutrits. -/
abbrev Tbl2 := (Fin 2 → ZMod 3) → Option (ZMod (stabPeriod 3))

/-! ### The three residues, as lists -/

def z3 : List (ZMod 3) := [0, 1, 2]

/-- The residues of the phase group, `ZMod (stabPeriod 3)`, kept at that type
    so that the tables below match the parameters syntactically. -/
def zD : List (ZMod (stabPeriod 3)) := [0, 1, 2]

theorem mem_z3 (a : ZMod 3) : a ∈ z3 := by decide +revert

theorem mem_zD (a : ZMod (stabPeriod 3)) : a ∈ zD := by decide +revert

theorem fin2_eta_z3 (x : Fin 2 → ZMod 3) : x = ![x 0, x 1] := by
  funext i
  fin_cases i <;> rfl

theorem fin2_eta_zD (x : Fin 2 → ZMod (stabPeriod 3)) : x = ![x 0, x 1] := by
  funext i
  fin_cases i <;> rfl

/-- The nine digit strings. -/
def vec2 : List (Fin 2 → ZMod 3) := z3.flatMap fun a => z3.map fun b => ![a, b]

theorem mem_vec2 (x : Fin 2 → ZMod 3) : x ∈ vec2 := by
  rw [fin2_eta_z3 x]
  simp only [vec2, List.mem_flatMap, List.mem_map]
  exact ⟨x 0, mem_z3 _, x 1, mem_z3 _, rfl⟩

/-! ### The normal forms -/

/-- `|x₀⟩`. -/
def pointTable (x0 : Fin 2 → ZMod 3) : Tbl2 :=
  tableOfP 3 x0 (fun j => Fin.elim0 j) (fun i _ => Fin.elim0 i) (fun i => Fin.elim0 i)
    (fun j => Fin.elim0 j)

/-- The line `(0, t) + y (1, a)` with phase `ω^(q y² + m y)`. -/
def lineTable0 (a t q : ZMod 3) (m : ZMod (stabPeriod 3)) : Tbl2 :=
  tableOfP 3 ![0, t] ![![1, a]] ![![q]] ![m] ![0]

/-- The line `(s, 0) + y (0, 1)` with phase `ω^(q y² + m y)`. -/
def lineTable1 (s q : ZMod 3) (m : ZMod (stabPeriod 3)) : Tbl2 :=
  tableOfP 3 ![s, 0] ![![0, 1]] ![![q]] ![m] ![1]

/-- Full support with phase `ω^(a x₀² + b x₀ x₁ + c x₁² + m₀ x₀ + m₁ x₁)`. -/
def fullTable (a b c : ZMod 3) (m0 m1 : ZMod (stabPeriod 3)) : Tbl2 :=
  tableOfP 3 0 (idWP 3 2) ![![a, b], ![0, c]] ![m0, m1] ![0, 1]

def pointTables : List Tbl2 := vec2.map pointTable

def lineTables : List Tbl2 :=
  (z3.flatMap fun a => z3.flatMap fun t => z3.flatMap fun q => zD.map fun m =>
    lineTable0 a t q m)
  ++ (z3.flatMap fun s => z3.flatMap fun q => zD.map fun m => lineTable1 s q m)

def fullTables : List Tbl2 :=
  z3.flatMap fun a => z3.flatMap fun b => z3.flatMap fun c => zD.flatMap fun m0 =>
    zD.map fun m1 => fullTable a b c m0 m1

/-- **The dictionary of two-qutrit stabilizer states**, `360` exponent tables. -/
def dict2 : List Tbl2 := pointTables ++ lineTables ++ fullTables

theorem dict2_length : dict2.length = 360 := by decide +kernel

/-! ### Direct evaluators

`tableOfP` decides flat membership through the `Fintype` equality on
`Fin 2 → ZMod 3` and sums over `Finset.univ`; the kernel pays a few thousand
steps per entry for that. The evaluators below compute the same tables
coordinatewise, and `dict2_eq_fast` shows the dictionary is unchanged, so
the kernel check of distinctness runs on them. -/

def pointFast (x0 : Fin 2 → ZMod 3) : Tbl2 := fun x =>
  if x 0 = x0 0 ∧ x 1 = x0 1 then some 0 else none

/-- `q y² + m y` read in `ZMod 3`. -/
def phase1 (q : ZMod 3) (m : ZMod (stabPeriod 3)) (y : ZMod 3) : ZMod (stabPeriod 3) :=
  ((q.val * y.val * y.val + m.val * y.val : ℕ) : ZMod (stabPeriod 3))

def lineFast0 (a t q : ZMod 3) (m : ZMod (stabPeriod 3)) : Tbl2 := fun x =>
  if x 1 = t + x 0 * a then some (phase1 q m (x 0)) else none

def lineFast1 (s q : ZMod 3) (m : ZMod (stabPeriod 3)) : Tbl2 := fun x =>
  if x 0 = s then some (phase1 q m (x 1)) else none

def fullFast (a b c : ZMod 3) (m0 m1 : ZMod (stabPeriod 3)) : Tbl2 := fun x =>
  some ((a.val * (x 0).val * (x 0).val + b.val * (x 0).val * (x 1).val
    + c.val * (x 1).val * (x 1).val + (m0.val * (x 0).val + m1.val * (x 1).val) : ℕ)
      : ZMod (stabPeriod 3))

theorem stabPeriod_three_div : stabPeriod 3 / 3 = 1 := rfl

theorem pointTable_eq (x0 : Fin 2 → ZMod 3) : pointTable x0 = pointFast x0 := by
  funext x
  simp only [pointTable, pointFast, tableOfP, affinePtP, ysol, quadPhaseP, Finset.univ_eq_empty,
    Finset.sum_empty, add_zero, mul_zero, Nat.cast_zero, funext_iff, Fin.forall_fin_two]

theorem lineTable0_eq (a t q : ZMod 3) (m : ZMod (stabPeriod 3)) :
    lineTable0 a t q m = lineFast0 a t q m := by
  funext x
  simp only [lineTable0, lineFast0, tableOfP, affinePtP, ysol, quadPhaseP, phase1,
    Fin.sum_univ_one, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
    sub_zero, mul_one, zero_add, funext_iff, Fin.forall_fin_two, true_and,
    stabPeriod_three_div, one_mul]

theorem lineTable1_eq (s q : ZMod 3) (m : ZMod (stabPeriod 3)) :
    lineTable1 s q m = lineFast1 s q m := by
  funext x
  simp only [lineTable1, lineFast1, tableOfP, affinePtP, ysol, quadPhaseP, phase1,
    Fin.sum_univ_one, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
    sub_zero, mul_one, mul_zero, add_zero, zero_add, funext_iff, Fin.forall_fin_two, and_true,
    stabPeriod_three_div, one_mul]

theorem fullTable_eq (a b c : ZMod 3) (m0 m1 : ZMod (stabPeriod 3)) :
    fullTable a b c m0 m1 = fullFast a b c m0 m1 := by
  funext x
  have hy : ysol (0 : Fin 2 → ZMod 3) ![0, 1] x = x := by
    funext j
    fin_cases j <;> simp [ysol]
  simp only [fullTable, fullFast, tableOfP, hy, affinePtP_id, quadPhaseP, Fin.sum_univ_two,
    Matrix.cons_val_zero, Matrix.cons_val_one, ZMod.val_zero, zero_mul, zero_add,
    stabPeriod_three_div, one_mul, if_true]

theorem pointTable_eq' : pointTable = pointFast := funext pointTable_eq

/-- The dictionary through the direct evaluators. -/
def dict2Fast : List Tbl2 :=
  vec2.map pointFast
  ++ ((z3.flatMap fun a => z3.flatMap fun t => z3.flatMap fun q => zD.map fun m =>
        lineFast0 a t q m)
      ++ (z3.flatMap fun s => z3.flatMap fun q => zD.map fun m => lineFast1 s q m))
  ++ (z3.flatMap fun a => z3.flatMap fun b => z3.flatMap fun c => zD.flatMap fun m0 =>
        zD.map fun m1 => fullFast a b c m0 m1)

theorem dict2_eq_fast : dict2 = dict2Fast := by
  simp only [dict2, dict2Fast, pointTables, lineTables, fullTables, pointTable_eq',
    lineTable0_eq, lineTable1_eq, fullTable_eq]

/-- A table read off at the nine digit strings, in the order of `vec2`. -/
def tableData (t : Tbl2) : List (Option (ZMod (stabPeriod 3))) := vec2.map t

/-- One digit per entry: `3` off the flat, the exponent on it. -/
def entryCode : Option (ZMod (stabPeriod 3)) → ℕ
  | none => 3
  | some e => e.val

/-- The table packed into one natural number, base `4`. The kernel compares
    two such keys in a single step, where it would compare two tables entry by
    entry through the `Fintype` instance on their domain. -/
def keyOf (t : Tbl2) : ℕ := (tableData t).foldr (fun o acc => acc * 4 + entryCode o) 0

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
-- `360` key evaluations by the kernel, a few seconds.
/-- The computed keys are the literals of `QutritDict2Keys.lean`. -/
theorem dict2Fast_keys_eq : dict2Fast.map keyOf = keyLits := by decide +kernel

/-- **The `360` tables are pairwise distinct.** With the exponent at the base
    point normalised to `0`, distinct tables are distinct states up to
    phase, so this is the count of `dictionary(3, 2)`. -/
theorem dict2_nodup : dict2.Nodup := by
  rw [dict2_eq_fast]
  exact List.Nodup.of_map keyOf (dict2Fast_keys_eq ▸ keyLits_nodup)

/-! ### Folding `Q` to upper triangular form -/

/-- `Q_01 + Q_10` above the diagonal, `0` below. -/
def upper2 (Q : Fin 2 → Fin 2 → ZMod 3) : Fin 2 → Fin 2 → ZMod 3 :=
  ![![Q 0 0, Q 0 1 + Q 1 0], ![0, Q 1 1]]

/-- The quadratic part does not see the folding, mod `3`: cast to `ZMod 3`,
    where `(Q i j).val` is `Q i j` again, and the two sums agree as
    polynomials. -/
theorem quadPart_upper2 (Q : Fin 2 → Fin 2 → ZMod 3) (y : Fin 2 → ZMod 3) :
    (∑ i, ∑ j, (Q i j).val * (y i).val * (y j).val) % 3
      = (∑ i, ∑ j, (upper2 Q i j).val * (y i).val * (y j).val) % 3 := by
  rw [← ZMod.natCast_eq_natCast_iff']
  push_cast
  simp only [ZMod.natCast_zmod_val, Fin.sum_univ_two, upper2, Matrix.cons_val_zero,
    Matrix.cons_val_one]
  ring

/-- The phase does not see the folding, mod `3`. -/
theorem quadPhaseP_upper2 (Q : Fin 2 → Fin 2 → ZMod 3) (l : Fin 2 → ZMod (stabPeriod 3))
    (y : Fin 2 → ZMod 3) :
    quadPhaseP (p := 3) Q l y % stabPeriod 3
      = quadPhaseP (p := 3) (upper2 Q) l y % stabPeriod 3 := by
  unfold quadPhaseP
  exact Nat.ModEq.add_right _ (Nat.ModEq.mul_left _ (quadPart_upper2 Q y))

theorem stabVecP_upper2 (Q : Fin 2 → Fin 2 → ZMod 3) (l : Fin 2 → ZMod (stabPeriod 3))
    (x : Fin 2 → ZMod 3) :
    stabVecP 3 2 2 0 (idWP 3 2) Q l x = stabVecP 3 2 2 0 (idWP 3 2) (upper2 Q) l x := by
  unfold stabVecP
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [zeta_pow_eq_of_mod 3 (quadPhaseP_upper2 Q l y)]

/-! ### Completeness -/

theorem idWP_pivots : ∀ j j' : Fin 2, idWP 3 2 j (![0, 1] j') = if j = j' then 1 else 0 := by
  decide

/-- **Every two-qutrit stabilizer state is a nonzero multiple of a listed
    table.** -/
theorem isStabP_two_qutrits {v : Fin (3 ^ 2) → ℂ} (hv : IsStabP 3 v) :
    ∃ (c : ℂ) (t : Tbl2), c ≠ 0 ∧ t ∈ dict2 ∧
      v = fun idx => c * tableVal 3 (t (digitsP 3 2 idx)) := by
  obtain ⟨c, k, x0, W, Q, l, hc, hinj, rfl⟩ := hv
  have hcard := Fintype.card_le_of_injective _ hinj
  simp only [Fintype.card_fun, ZMod.card, Fintype.card_fin] at hcard
  have hk : k ≤ 2 := (Nat.pow_le_pow_iff_right (by norm_num)).mp hcard
  have hz : zeta 3 ≠ 0 := zeta_ne_zero 3
  interval_cases k
  · -- a point
    have hW : W = fun j => Fin.elim0 j := funext fun j => Fin.elim0 j
    have hQ : Q = fun i _ => Fin.elim0 i := funext fun i => Fin.elim0 i
    have hl : l = fun i => Fin.elim0 i := funext fun i => Fin.elim0 i
    subst hW hQ hl
    refine ⟨c, pointTable x0, hc, ?_, ?_⟩
    · exact List.mem_append_left _
        (List.mem_append_left _ (List.mem_map.mpr ⟨x0, mem_vec2 x0, rfl⟩))
    · funext idx
      rw [pointTable, ← stabVecP_eq_tableVal _ _ _ _ _ (fun j => Fin.elim0 j)]
  · -- a line
    have hw : W 0 ≠ 0 := by
      intro h0
      have h := hinj (a₁ := fun _ => 0) (a₂ := fun _ => 1) (by
        funext i
        simp [affinePtP, h0])
      exact absurd (congrFun h 0) (by decide)
    by_cases hw0 : W 0 0 = 0
    · -- pivot in column 1
      have hw1 : W 0 1 ≠ 0 := by
        intro h1
        apply hw
        funext i
        fin_cases i
        exacts [hw0, h1]
      set b : Fin 1 → ZMod 3 := fun _ => -(x0 1) * (W 0 1)⁻¹ with hb
      set A : Fin 1 → Fin 1 → ZMod 3 := fun _ _ => (W 0 1)⁻¹ with hA
      have hφ : Function.Injective (reparamY b A) := by
        intro z z' h
        have h0 := congrFun h 0
        simp only [reparamY_apply, Fin.sum_univ_one, hA] at h0
        have := mul_left_cancel₀ (inv_ne_zero hw1) (add_left_cancel h0)
        funext i
        rw [Fin.fin_one_eq_zero i]
        exact this
      obtain ⟨C, Q', l', h⟩ := stabVecP_reparam x0 W Q l b A hφ
      have hx0'1 : affinePtP x0 W b 1 = 0 := by
        simp only [affinePtP, Fin.sum_univ_one, hb]
        field_simp
        ring
      have hW'0 : reparamW A W 0 0 = 0 := by simp [reparamW, hA, hw0]
      have hW'1 : reparamW A W 0 1 = 1 := by
        simp only [reparamW, Fin.sum_univ_one, hA]
        exact inv_mul_cancel₀ hw1
      have hx0e := fin2_eta_z3 (affinePtP x0 W b)
      rw [hx0'1] at hx0e
      have hWe : reparamW A W = ![![0, 1]] := by
        funext j i
        rw [Fin.fin_one_eq_zero j]
        fin_cases i <;> simp [hW'0, hW'1]
      have hQe : Q' = ![![Q' 0 0]] := by
        funext i j
        rw [Fin.fin_one_eq_zero i, Fin.fin_one_eq_zero j]
        rfl
      have hle : l' = ![l' 0] := by
        funext i
        rw [Fin.fin_one_eq_zero i]
        rfl
      refine ⟨c * zeta 3 ^ C, lineTable1 (affinePtP x0 W b 0) (Q' 0 0) (l' 0),
        mul_ne_zero hc (pow_ne_zero _ hz), ?_, ?_⟩
      · refine List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ ?_))
        simp only [List.mem_flatMap, List.mem_map]
        exact ⟨_, mem_z3 _, _, mem_z3 _, _, mem_zD _, rfl⟩
      · funext idx
        rw [h, lineTable1, ← hx0e, ← hWe, ← hQe, ← hle,
          ← stabVecP_eq_tableVal _ _ _ _ ![1] ?_]
        · ring
        · rw [hWe]
          decide
    · -- pivot in column 0
      set b : Fin 1 → ZMod 3 := fun _ => -(x0 0) * (W 0 0)⁻¹ with hb
      set A : Fin 1 → Fin 1 → ZMod 3 := fun _ _ => (W 0 0)⁻¹ with hA
      have hφ : Function.Injective (reparamY b A) := by
        intro z z' h
        have h0 := congrFun h 0
        simp only [reparamY_apply, Fin.sum_univ_one, hA] at h0
        have := mul_left_cancel₀ (inv_ne_zero hw0) (add_left_cancel h0)
        funext i
        rw [Fin.fin_one_eq_zero i]
        exact this
      obtain ⟨C, Q', l', h⟩ := stabVecP_reparam x0 W Q l b A hφ
      have hx0'0 : affinePtP x0 W b 0 = 0 := by
        simp only [affinePtP, Fin.sum_univ_one, hb]
        field_simp
        ring
      have hW'0 : reparamW A W 0 0 = 1 := by
        simp only [reparamW, Fin.sum_univ_one, hA]
        exact inv_mul_cancel₀ hw0
      have hx0e := fin2_eta_z3 (affinePtP x0 W b)
      rw [hx0'0] at hx0e
      have hWe : reparamW A W = ![![1, reparamW A W 0 1]] := by
        funext j i
        rw [Fin.fin_one_eq_zero j]
        fin_cases i <;> simp [hW'0]
      have hQe : Q' = ![![Q' 0 0]] := by
        funext i j
        rw [Fin.fin_one_eq_zero i, Fin.fin_one_eq_zero j]
        rfl
      have hle : l' = ![l' 0] := by
        funext i
        rw [Fin.fin_one_eq_zero i]
        rfl
      refine ⟨c * zeta 3 ^ C,
        lineTable0 (reparamW A W 0 1) (affinePtP x0 W b 1) (Q' 0 0) (l' 0),
        mul_ne_zero hc (pow_ne_zero _ hz), ?_, ?_⟩
      · refine List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ ?_))
        simp only [List.mem_flatMap, List.mem_map]
        exact ⟨_, mem_z3 _, _, mem_z3 _, _, mem_z3 _, _, mem_zD _, rfl⟩
      · funext idx
        rw [h, lineTable0, ← hx0e, ← hWe, ← hQe, ← hle,
          ← stabVecP_eq_tableVal _ _ _ _ ![0] ?_]
        · ring
        · rw [hWe]
          intro j j'
          fin_cases j
          fin_cases j'
          simp
  · -- full support
    obtain ⟨C, Q', l', h⟩ := stabVecP_full_normal x0 W Q l hinj
    refine ⟨c * zeta 3 ^ C, fullTable (Q' 0 0) (Q' 0 1 + Q' 1 0) (Q' 1 1) (l' 0) (l' 1),
      mul_ne_zero hc (pow_ne_zero _ hz), ?_, ?_⟩
    · refine List.mem_append_right _ ?_
      simp only [fullTables, List.mem_flatMap, List.mem_map]
      exact ⟨_, mem_z3 _, _, mem_z3 _, _, mem_z3 _, _, mem_zD _, _, mem_zD _, rfl⟩
    · funext idx
      rw [h, stabVecP_upper2, fullTable, ← fin2_eta_zD l']
      change c * (zeta 3 ^ C * stabVecP 3 2 2 0 (idWP 3 2) (upper2 Q') l' (digitsP 3 2 idx))
        = c * zeta 3 ^ C
          * tableVal 3 (tableOfP 3 0 (idWP 3 2) (upper2 Q') l' ![0, 1] (digitsP 3 2 idx))
      rw [← stabVecP_eq_tableVal _ _ _ _ ![0, 1] idWP_pivots]
      ring

end StabRank
