/-
The `3e4` size of `KernelBench.lean`, in its own module so that its build
time and peak RSS can be measured alone and compared with the `1e5` size:
the difference gives the kernel's memory per operation. Not imported by the
root.
-/
import LeanProofs.Bench.KernelBench

namespace StabRank.Bench

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
set_option profiler true in
/-- About `3e4` operations by the kernel. -/
theorem bench4_kernel : bench 45 4 = (45, 17248, 30360) := by
  decide +kernel

end StabRank.Bench
