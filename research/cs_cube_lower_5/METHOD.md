# Finite completeness and arithmetic boundary

This note describes the portable code whose complete saved run and independent audit are now available: 667 base-cover orbits and all 27,088 lift sign tests were excluded. The first producer exit was not observed; its saved result was independently checked. The separate full native challenge-verifier run subsequently completed with an observed zero exit code, reporting PASS and the local verified tier; this is not an external platform decision. The
program must finish every indicated gate and control before printing its
claim. All phase rows represent zero or fourth roots of unity, with first
nonzero phase canonicalized to one. Coefficients retain their actual scales.

## Dictionary and minimum-three gate

For support dimension d, RREF linear bases enumerate each binary subspace
once. Coset representatives with zero pivot coordinates enumerate each affine
translate once. The d linear Z4 coefficients and d(d-1)/2 even quadratic
coefficients give every stabilizer phase on that support. For four qubits
the affine-support counts are 16,120,140,30,1 and phase counts are
1,4,32,512,16384. The 36,720 resulting canonical rays must be distinct.

The seven explicit target-fixing Clifford generators are three adjacent
label transpositions on each two-qubit block and block interchange. A local
label permutation is affine over F2²; its target phase correction has even
mixed coefficient. The known three-cover is recovered by a positive control.
For each first-ray orbit representative, the exact quotient by [u,T] scans
every free ray and all pairs in each projective residual bucket with distinct
target coordinates. Zero residuals or zero solution coefficients are retained
as shorter-rank guards. Every hit is reconstructed over Q(i). Generator
closure, including actual phase factors, must yield exactly one orbit
containing the known triple. Expected cardinalities alone do not certify this.

## Enlarged pair action

Let D_T be the diagonal phase of T and U_pi = D_T P_pi D_T^-1.
The antiunitary A = D_T² K fixes T, is Clifford followed by conjugation, is an
involution and commutes with every U_pi. Actual ray and phase actions verify
these identities. The unitary action has 1,152 elements; its commuting coset
gives 2,304 actions with inverse (gA)^-1 = g^-1 A.

Choose the least ray u in each full-group ray orbit. Its stabilizer partitions
all second rays. If the endpoints lie in different full-group ray orbits,
keep only the later orbit and its point-stabilizer representative v. Its
unordered pair orbit has size |Gu| times |G_u v|. For endpoints in the same
orbit, choose a transporter t_v taking u to v. Reversal takes the suborbit of
v to the suborbit of t_v^-1 u. This involution is independent of the
transporter; keep the lesser of the two representatives. A self-reversed
suborbit has half the ordered-pair weight, while two reversed suborbits give
the full weight. Code verifies the reversal involution, equal partner sizes,
and every pair of ray-orbit weight totals, then the global binomial total.
The whole-group order need not be the full target Clifford group: this
explicit verified subgroup suffices for complete coverage.

## Projected quotient

For retained independent u,v, B=[u,v,T] has rank three because the preceding
gate excludes a rank-two cover. Three independent coordinate rows give each
dictionary ray w its pivot coefficients a_w and quotient residual r_w.
For two nonzero residuals, a four-cover can occur only if r_z=lambda*r_w and
the target coordinates of a_z-lambda*a_w differ. All cross-sigma pairs in
each projective bucket are retained; no first-hit truncation is used.

The fixed 3-by-16 integer linear map L is a deterministic necessary screen.
If Lr is nonzero, its projective key and normalized target coordinate are
used, and all candidate pairs are rechecked in the full residual and exact
identity. False projected collisions are rejections, never assertions of
full equality. All zero projected residuals fall back to the full thirteen
non-pivot coordinates, including true-zero quotient bookkeeping. A true
pair cannot be split across the zero and nonzero projected groups. The
choice of L affects speed only; no random-projection success assumption is
part of the proof.

The finite prime is p=65,521 with i=41,224. Every relevant original
phase-valued matrix has at most five columns. A nonzero Gaussian-integer
k-minor has norm at most k^k <= 5^5 = 3,125 < p; if its image vanished under
the chosen Gaussian prime, its norm would be a positive multiple of p.
Consequently modular ranks are faithful, not probabilistic. The exact
Gaussian-rational solution of every surviving four-cover is checked on all
sixteen amplitudes; the coefficient and minor tests distinguish independent
four-covers from valid shorter identities. Int64 operations have explicit
bounded dimensions; projective combined keys are below 2^48.

The logged primitive pair accounting sums to C(36,720,2) for every retained
case. Every regenerated retained-pair class must complete, and every returned
case index is unique and required. Parallel completion order is irrelevant.

## All-cover orbits and lifts

For independent columns the coefficient vector is unique. Generator action
on actual canonical rays transports each coefficient by the removed phase;
antiunitary action first conjugates it. Complete eight-generator closure
checks coefficient consistency, orbit divisibility and physical word
reconstruction. Every input cover is assigned to exactly one full orbit.
Every orbit is tested, without a product or fidelity exclusion.

For a fixed base T=sum c_j s_j, a globally visible stabilizer extension has
four conditional fibers s_j, P_A s_j, P_B s_j, and
epsilon_j P_B P_A s_j, epsilon_j in {+1,-1}. This follows by taking two
stabilizer generators translating the two projected bits; their Pauli
commutator supplies only a sign. Conversely controlled actual Paulis,
including their fourth-root phases, and an optional CZ realize each choice.
Changing a Pauli representative fixing s_j changes the cross fiber by at
most a sign, already covered. The order P_B P_A and its actual phase are
preserved in the multiplication table.

Each four-qubit ray has exactly 64 distinct actual Pauli images under all
1,024 Pauli/phase labels, with multiplicity sixteen. Integer meet-in-the-
middle comparison keeps every coefficient-fixed transition producing T in
the 01 or 10 fiber. Every ordered pair of complete transitions and all
sixteen epsilon vectors are compared with iT in the 11 fiber. A positive
is independently reconstructed as four 64-amplitude affine-quadratic
stabilizer rows. A negative is valid only after all required pairs finish.
Caps, exceptions and timeouts cannot be negative certificates.

Together with the embedded independent-slice theorem and its regenerated
minimum-three hypothesis, a complete negative lift result excludes every
four-term cube decomposition. The same analytic theorem already excludes
rank at most three. The conclusion is the lower bound five, not equality.
