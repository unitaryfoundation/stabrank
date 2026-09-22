/-
Splitting a `decide +kernel` over `Fin (c * N)` into `c` slices of length `N`.

A single kernel check over `2^10` indices runs for longer than a build step
is allowed here, so the large cells prove the identity on slices
`⟨q * N + i, _⟩` for `i : Fin N`, one theorem per slice in its own module, and
`forall_fin_of_chunks` puts the slices together.
-/
import Mathlib.Data.Fin.Basic
import Mathlib.Tactic

namespace StabRank

/-- The index `q * N + i` of slice `q`, as an element of `Fin (c * N)`. -/
def chunkIdx {c N : ℕ} (q : Fin c) (i : Fin N) : Fin (c * N) :=
  ⟨q.val * N + i.val, by
    calc q.val * N + i.val < q.val * N + N := by omega
      _ = (q.val + 1) * N := by ring
      _ ≤ c * N := Nat.mul_le_mul_right N q.isLt⟩

/-- A statement about every index of `Fin (c * N)` follows from the same
    statement on each of the `c` slices of length `N`. -/
theorem forall_fin_of_chunks {c N : ℕ} (P : Fin (c * N) → Prop)
    (h : ∀ q : Fin c, ∀ i : Fin N, P (chunkIdx q i)) : ∀ idx, P idx := by
  intro idx
  have hN : 0 < N := by
    rcases Nat.eq_zero_or_pos N with rfl | hN
    · exact absurd idx.isLt (by simp)
    · exact hN
  have hq : idx.val / N < c := by
    rw [Nat.div_lt_iff_lt_mul hN]
    exact idx.isLt
  have := h ⟨idx.val / N, hq⟩ ⟨idx.val % N, Nat.mod_lt _ hN⟩
  convert this using 1
  ext
  simp only [chunkIdx]
  exact (Nat.div_add_mod' _ _).symm

end StabRank
