/-
The same row operation as `KernelBench.lean` with the recursion written
directly through `List.rec` and `Nat.rec` instead of the `brecOn` form that
structural recursion compiles to. The code generator does not accept raw
recursors, so these definitions are `noncomputable`; only the kernel runs
them. The point is the comparison: the kernel unfolds a `brecOn` body,
which is large, once per step, and the raw recursor about halves both the
time and the memory per operation. Not imported by the root.

The work here is `300` repetitions of `row ← row + m · piv` on rows of
length `300`, `9e4` operations, on the same pseudo-random rows as the
other benchmarks.
-/

namespace StabRank.Bench.Rec

def ell : Nat := 65521
def lcg (s : Nat) : Nat := (s * 1103515245 + 12345) % 2147483648

noncomputable def genRow (c : Nat) (s : Nat) : List Nat × Nat :=
  Nat.rec (motive := fun _ => Nat → List Nat × Nat) (fun s => ([], s))
    (fun _ ih s => let s' := lcg s; let r := ih s'; ((s' / 16) % ell :: r.1, r.2)) c s

/-- `row + m · piv` entrywise mod `ℓ`, by `List.rec` on the first list. -/
noncomputable def axpy (m : Nat) (as bs : List Nat) : List Nat :=
  List.rec (motive := fun _ => List Nat → List Nat) (fun _ => [])
    (fun a _ ih bs => List.casesOn (motive := fun _ => List Nat) bs []
      (fun b bs' => (a + m * b) % ell :: ih bs')) as bs

/-- `k` row operations with multipliers from the generator. -/
noncomputable def iter (k : Nat) (s : Nat) (row piv : List Nat) : List Nat :=
  Nat.rec (motive := fun _ => Nat → List Nat → List Nat) (fun _ row => row)
    (fun _ ih s row => ih (lcg s) (axpy (s % ell) piv row)) k s row

noncomputable def sumL (l : List Nat) : Nat :=
  List.rec (motive := fun _ => Nat) 0 (fun a _ ih => a + ih) l

/-- `k` operations on rows of length `n`, `n · k` operations. -/
noncomputable def test (n k : Nat) : Nat :=
  let r := genRow n 1
  let p := genRow n 2
  sumL (iter k 5 r.1 p.1)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
set_option profiler true in
/-- `9e4` operations by the kernel through raw recursors. -/
theorem benchRec : test 300 300 = 10538924 := by
  decide +kernel

end StabRank.Bench.Rec
