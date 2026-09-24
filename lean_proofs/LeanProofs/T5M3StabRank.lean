/-
`χ(|T5⟩^⊗3) ≤ 15` against `stabRankP 5`: the product of the five-term
decomposition of `|T5⟩^⊗2` (`t5_m2_stabRankP_le_five`) and the three-term
decomposition of `|T5⟩` (`t5_m1_stabRankP_le_three`), through the
multiplicativity `stabRankP_tensor_le` of `Stabilizer/TensorP.lean`. The
fifteen terms of `bounds/T5-m3-upper-15.json` are the pairwise tensor
products that lemma counts (`x0` and `l` concatenated, `W` and `Q` block
diagonal, coefficients multiplied). `tensorP_t5Vec` identifies
`|T5⟩^⊗n ⊗ |T5⟩^⊗m` with `|T5⟩^⊗(n+m)` on digit strings, in the style of
`tensorP_hVec` in `QubitHStabRank.lean`.
-/
import LeanProofs.T5M2StabRank
import LeanProofs.Stabilizer.TensorP

namespace StabRank

open Stabilizer

/-- `|T5⟩^⊗n ⊗ |T5⟩^⊗m = |T5⟩^⊗(n+m)`. -/
theorem tensorP_t5Vec (n m : ℕ) : tensorP 5 (t5Vec n) (t5Vec m) = t5Vec (n + m) := by
  funext idx
  simp only [tensorP, t5Vec, Equiv.apply_symm_apply, Fin.prod_univ_add]
  rfl

/-- **χ(|T5⟩^⊗3) ≤ 15 against `IsStabP 5`**, from `χ(|T5⟩^⊗2) ≤ 5`,
    `χ(|T5⟩) ≤ 3`, and multiplicativity. -/
theorem t5_m3_stabRankP_le_fifteen : stabRankP 5 (t5Vec 3) ≤ 15 := by
  have h := le_trans (stabRankP_tensor_le (p := 5) (t5Vec 2) (t5Vec 1))
    (Nat.mul_le_mul t5_m2_stabRankP_le_five t5_m1_stabRankP_le_three)
  rw [tensorP_t5Vec] at h
  exact h

end StabRank
