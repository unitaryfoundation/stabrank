/-
The three sizes of `KernelBench.lean` under `native_decide`, for the
comparison with the kernel figures only. `native_decide` adds the axiom
`Lean.ofReduceBool` and trusts the compiler and runtime; it is never to be
used in a proof that ends up in a bound. Not imported by the root.
-/
import LeanProofs.Bench.KernelBench

namespace StabRank.Bench

set_option profiler true in
theorem bench5_native : bench 67 1 = (67, 4931, 100232) := by
  native_decide

set_option profiler true in
theorem bench6_native : bench 144 2 = (144, 63513, 995280) := by
  native_decide

set_option profiler true in
theorem bench7_native : bench 311 3 = (311, 29497, 10026640) := by
  native_decide

end StabRank.Bench
