/-
`χ(|T5⟩^⊗4) ≤ 25` against `stabRankP 5`: the tensor square of the five-term
decomposition of `|T5⟩^⊗2` (`t5_m2_stabRankP_le_five`), through the
multiplicativity `stabRankP_tensor_le` of `Stabilizer/TensorP.lean` and
`tensorP_t5Vec` of `T5M3StabRank.lean`. The 25 terms of
`bounds/T5-m4-upper-25.json` are the pairwise tensor products that lemma
counts.
-/
import LeanProofs.T5M3StabRank

namespace StabRank

open Stabilizer

/-- **χ(|T5⟩^⊗4) ≤ 25 against `IsStabP 5`**, from `χ(|T5⟩^⊗2) ≤ 5` and
    multiplicativity. -/
theorem t5_m4_stabRankP_le_twentyfive : stabRankP 5 (t5Vec 4) ≤ 25 := by
  have h := le_trans (stabRankP_tensor_le (p := 5) (t5Vec 2) (t5Vec 2))
    (Nat.mul_le_mul t5_m2_stabRankP_le_five t5_m2_stabRankP_le_five)
  rw [tensorP_t5Vec] at h
  exact h

end StabRank
