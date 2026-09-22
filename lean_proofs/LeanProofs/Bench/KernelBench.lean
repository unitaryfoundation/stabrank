/-
Kernel throughput benchmark for reflected checkers, not imported by the root.

`docs/notes/lean_verified_checker_design.md` sizes the scan-based lower
bounds in field operations mod `ℓ = 65521` and asks, as its first step,
how many such operations per second `decide +kernel` sustains on a
reflected row reduction. This module is that measurement. `gauss` is a
Gaussian elimination over `List (List Nat)` with entries reduced mod `ℓ`:
at each column it picks the first row with a nonzero head, scales it by
the modular inverse of the head (Fermat, `powModAux`), and subtracts the
scaled pivot from every other row. One "operation" is one
multiply-and-reduce `(a + m * b) % ℓ` or `x * m % ℓ`; `gauss` returns the
rank, a checksum, and its own operation count, so the theorems below
record exactly how much work the kernel did.

The matrices are pseudo-random (an LCG seeded per size), so every pivot
is a genuine reduction and the count is about `N³/3` for an `N × N`
matrix. `bench5_kernel` is about `1e5` operations; `KernelBench6.lean` and
`KernelBench7.lean` hold the `1e6` and `1e7` sizes as separate modules so
that each build can be timed and its peak RSS measured on its own;
`NativeBench.lean` runs the same sizes under `native_decide`, for the
comparison only. Nothing here is used by any bound.
-/

namespace StabRank.Bench

/-- The modulus, `65521`, the largest prime below `2^16`; it is `1 mod 48`. -/
def ell : Nat := 65521

/-- One step of a linear congruential generator, `mod 2^31`. -/
def lcg (s : Nat) : Nat := (s * 1103515245 + 12345) % 2147483648

/-- A pseudo-random row of length `c` and the state after it. -/
def genRow : Nat → Nat → List Nat × Nat
  | 0, s => ([], s)
  | c + 1, s =>
    let s' := lcg s
    let r := genRow c s'
    ((s' / 16) % ell :: r.1, r.2)

/-- `r` pseudo-random rows of length `c`. -/
def genRows : Nat → Nat → Nat → List (List Nat)
  | 0, _, _ => []
  | r + 1, c, s =>
    let row := genRow c s
    row.1 :: genRows r c row.2

/-- `b ^ e mod ℓ` by binary exponentiation; `fuel` bounds the number of
    halvings (`20` suffices for `e < 2^20`). -/
def powModAux : Nat → Nat → Nat → Nat → Nat
  | 0, _, _, acc => acc
  | fuel + 1, b, e, acc =>
    if e = 0 then acc
    else powModAux fuel (b * b % ell) (e / 2) (if e % 2 = 1 then acc * b % ell else acc)

/-- The inverse of `a` mod `ℓ`, `a ^ (ℓ - 2)`. -/
def inv (a : Nat) : Nat := powModAux 20 a (ell - 2) 1

/-- `row + m * piv` entrywise mod `ℓ`; one operation per entry. -/
def axpy (m : Nat) (piv row : List Nat) : List Nat :=
  List.zipWith (fun a b => (a + m * b) % ell) row piv

/-- Scale a row by `m` mod `ℓ`; one operation per entry. -/
def scale (m : Nat) (row : List Nat) : List Nat := row.map fun x => x * m % ell

/-- The first row with a nonzero head, and the others (heads still attached). -/
def findPivot : List (List Nat) → Option (List Nat × List (List Nat))
  | [] => none
  | [] :: rest => (findPivot rest).map fun p => (p.1, [] :: p.2)
  | (a :: t) :: rest =>
    if a = 0 then (findPivot rest).map fun p => (p.1, (a :: t) :: p.2)
    else some (a :: t, rest)

/-- Drop the head of a row (the eliminated column). -/
def dropHead : List Nat → List Nat
  | [] => []
  | _ :: t => t

/-- Eliminate the head of `row` against the normalised pivot `pivN` (head
    already dropped, leading entry scaled to `1`). -/
def elimRow (pivN : List Nat) : List Nat → List Nat
  | [] => []
  | a :: t => axpy ((ell - a) % ell) pivN t

/-- Gaussian elimination over the remaining columns. Returns
    `(rank, checksum, operations)`; the checksum is the sum mod `ℓ` of every
    normalised pivot row, so it depends on every operation performed. -/
def gauss : Nat → List (List Nat) → Nat × Nat × Nat
  | 0, _ => (0, 0, 0)
  | c + 1, rows =>
    match findPivot rows with
    | none => gauss c (rows.map dropHead)
    | some (piv, others) =>
      let pivN := scale (inv (piv.headD 1)) (dropHead piv)
      let others' := others.map (elimRow pivN)
      let rest := gauss c others'
      (rest.1 + 1, (rest.2.1 + pivN.foldl (· + ·) 0) % ell,
        rest.2.2 + c + others.length * c)

/-- Rank, checksum, and operation count of the `n × n` pseudo-random matrix
    with seed `s`. -/
def bench (n s : Nat) : Nat × Nat × Nat := gauss n (genRows n n s)

/-! ### Sizes

`#eval bench n s` gives the expected triple; the theorems then ask the kernel
(or the compiler) to confirm it. The third component is the operation
count. -/

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
set_option profiler true in
/-- About `1e5` operations by the kernel. -/
theorem bench5_kernel : bench 67 1 = (67, 4931, 100232) := by
  decide +kernel

end StabRank.Bench
