# Rare-type pivot for the rank-7 scan of |T3>^3: not available by slicing

Companion to `docs/notes/t3_rank7_exclusion.md`, section 4, paragraph
"Rare-type pivot". Result: the lemma proposed there cannot be obtained from
the slice structure, and the two-qutrit analogue of the lemma is false. The
finite claims below are checked by `research/t3_rank7/rare_pivot_m2.py`,
which writes `research/t3_rank7/results/rare_pivot_m2.json`.

Notation is that of the exclusion note. N = 30240 is the number of
three-qutrit stabilizer states, V_3 = span{psi_0, psi_1, psi_2} the Galois
space, G the symmetry group of V_3 (order 2916, 45 orbits on the states).
A rank-7 configuration is a set of seven linearly independent stabilizer
states whose span contains V_3. The support of a stabilizer state is an
affine flat of F_3^3; its dimension k is 0 (point), 1 (line), 2 (plane) or 3
(full support). There are 27 point states, 1053 line states, 9477 plane
states and 19683 full-support states.

## 1. The lemma and what it would buy

Lemma (rare-type pivot, proposed). Every rank-7 configuration contains a
state whose support has dimension at most 1.

The 1080 point and line states form eleven G-orbits (sizes 27 for the
points, and 243, 243, 162, 27, 54, 9, 9, 9, 54, 243 for the lines). The
first pivot of the three-pivot scan runs over one representative per
G-orbit, and every configuration is moved by G so that one of its members
is the representative. If every configuration has a member in a G-invariant
set Lambda, the first pivot can be restricted to the orbits inside Lambda.
The inner-step count of the scan is, per representative i, about N^3 / (6
|Stab(i)|), so restricting to Lambda divides the total by N / |Lambda|. For
the 1080 point and line states this is a factor of 28, as the exclusion note
says. Nothing below depends on the specific set: any G-invariant Lambda
works, and in the orbit-block order of the exclusion note one puts the
orbits of Lambda first.

## 2. The slice route

Fix a slice H = {x_p = c} of F_3^3 and let R be the restriction of a vector
in C^27 to the nine coordinates of H, read as a two-qutrit vector in the
coordinates other than x_p. Let V_2 be the two-qutrit Galois space, spanned
by psi'_0, psi'_1, psi'_2 (the three line states of direction (1, 2) with
the phases of |T3>^2, which are the unique rank-3 decomposition of |T3>^2).

Fact 1 (check 1 of the script). For every slice, R(psi_r) is a nonzero
scalar multiple of psi'_{r - c mod 3}, so R(V_3) = V_2; and R maps every
three-qutrit stabilizer state to zero or to a scalar multiple of a two-qutrit
stabilizer state. On each slice 720 states restrict to zero and every
two-qutrit state has exactly 82 preimages up to scalars (a point state lifts
to itself or to a line through it transverse to H, a line state to itself or
to a plane through it other than H, a full-support state to the plane H or
to a full-support state).

A spanning configuration of V_2 of size s is a set of s linearly independent
two-qutrit stabilizer states whose span contains V_2. For a set tau of
two-qutrit states let L_H(tau) be the set of three-qutrit states whose
restriction to H is a nonzero scalar multiple of a member of tau.

Lemma 2 (transfer). Suppose every spanning configuration of V_2 of size at
most 7 contains a member of tau. Then every rank-7 configuration contains a
member of L_H(tau), hence of the G-invariant set Lambda = G L_H(tau).

Proof. Let S be a rank-7 configuration. Since span(S) contains V_3,
span(R(S)) contains R(V_3) = V_2. Choose M inside S minimal such that
span(R(M)) contains V_2. Minimality makes the vectors R(s), s in M, linearly
independent (a dependent one could be dropped without changing the span), so
they are nonzero and pairwise non-proportional, and by Fact 1 they are, up
to scalars, |M| <= 7 distinct two-qutrit stabilizer states. So R(M) is a
spanning configuration of V_2 of size at most 7 and contains a member of
tau; the corresponding s lies in L_H(tau). QED

The size of Lambda is at most 9 times 82 |tau|, and often less because the
lifts of different slices overlap. The script computes it exactly for each
orbit of the two-qutrit symmetry group (order 324, eight orbits, sizes 9,
54, 27, 18, 3, 6, 81, 162, where 9 is the point orbit, 3 is {psi'_0, psi'_1,
psi'_2}, and 81 and 162 are the two full-support orbits):

| two-qutrit orbit (size, support dim) | closure of its lift at m = 3 | states |
|--------------------------------------|------------------------------|--------|
| points (9, 0)                        | all point and line states    | 1080   |
| psi'_r (3, 1)                        | 1 line orbit, 9 plane orbits | 1890   |
| lines (54, 1)                        | 1 line orbit, 10 plane orbits| 6804   |
| lines (27, 1)                        | 1 line orbit, 5 plane orbits | 4617   |
| lines (18, 1)                        | 1 line orbit, 12 plane orbits| 4536   |
| lines (6, 1)                         | 1 line orbit, 9 plane orbits | 2808   |
| full (81, 2)                         | 1 plane orbit, 13 full orbits| 19926  |
| full (162, 2)                        | 1 plane orbit, 13 full orbits| 20898  |

The closure of the lift of a single two-qutrit state is already the closure
of its whole orbit (check 5). The first row is the point of the exercise:
the point orbit lifts to exactly the 1080 point and line states, so the
proposed lemma is precisely what Lemma 2 gives from the statement "every
spanning configuration of V_2 of size at most 7 contains a point state".

## 3. What holds at m = 2

All ranks are computed mod 2^31 - 1, which is exact at m = 2 (every matrix
has at most nine columns and 9^9 < 2^31 - 1, so the Hadamard argument of
`cert_t3m3_rank7.py` applies to every rank).

Size 3 (check 2). Exactly three states lie in V_2, namely psi'_0, psi'_1,
psi'_2, and they span it. The unique spanning configuration of size 3 is
{psi'_0, psi'_1, psi'_2}, which contains no point state. So the statement
needed for the proposed lemma fails at size 3, and any tau in Lemma 2 must
contain a psi'_r; its lift alone is already 1890 states.

Size 4 (check 3). Every spanning configuration of size 4 contains a psi'_r.
Proof by enumeration: a size-4 configuration avoiding the psi'_r has four
states of nonzero image in the quotient by V_2 spanning one dimension, so it
lies inside the class set of any of its members i, namely the states other
than the psi'_r whose image is parallel to the image of i; for every i the
span of that class set does not contain V_2 (rank pair mod 2^31 - 1). An
example with two psi'_r: psi'_0 = (1 - w3^2) |00> + w3^2 u, where u is the
line state on x1 + x2 = 0 with phase w3^{2 x1^2}, so {psi'_1, psi'_2, |00>,
u} spans V_2.

Size 5 (check 4). There are spanning configurations of size 5 consisting of
full-support states only. Among the 243 full-support states there are 27
class sets of image dimension 2 whose span contains V_2; each has nine
states of rank 5, all in the full-support orbit of size 81, and any five
independent ones among them (54 of the 126 five-subsets) form a spanning
configuration. The first class set consists of the nine states w3^{Q} with

    Q = a (x1^2 + x2^2) + c x1 x2 + d (x1 + x2),

for (a, c, d) in {(0,0,0), (0,1,0), (0,2,1), (1,0,1), (1,1,1), (1,2,2),
(2,0,1), (2,1,1), (2,2,2)}, all invariant under x1 <-> x2; the vectors
invariant under that swap form a 6-dimensional space containing V_2, and
the nine span a 5-dimensional subspace of it that still contains V_2. An
explicit configuration is the first five of the list, dictionary indices
117, 144, 175, 211, 238.

Size 7 (check 4b). The other full-support orbit (size 162) also contains
spanning configurations of size 7, for instance the seven states w3^{Q} with
(a, b, c, d, e) in {(0,0,0,0,1), (0,0,0,0,2), (0,1,0,0,1), (0,1,1,0,0),
(0,1,2,1,2), (1,2,0,2,1), (1,1,1,1,0)} for Q = a x1^2 + b x2^2 + c x1 x2 +
d x1 + e x2 (indices 118, 119, 127, 153, 185, 223, 237).

Consequence (check 5). Any tau satisfying the hypothesis of Lemma 2 contains
a state of each full-support orbit, so Lambda contains the closures of both
lifts, which together are all 19683 full-support states and two plane
orbits, 21870 states in all. The best factor Lemma 2 can give, for any
choice of slice and tau, is 30240 / 21870 = 1.38, and even that requires a
hitting set for all spanning configurations of size at most 7, which was not
established. The slice route is closed.

An exploratory search over unions of the eight two-qutrit orbits (with the
numba scan of the scratch tooling, not part of the script) found a single
minimal hitting set, the union of the orbits of sizes 54, 18, 3, 81 and 162
(318 of the 360 states): the 42 remaining states (points and the line
orbits of sizes 27 and 6) support no spanning configuration of size at most
7. This is consistent with the consequence above and adds nothing to it.

## 4. Conclusion

The two-qutrit analogue of the rare-type lemma holds at sizes chi(T3^2) = 3
and 4 (every spanning configuration contains a psi'_r, a line state) and
fails at size 5 (five full-support states span V_2), so it offers no support
for the three-qutrit statement at size 7, whichever of chi(T3^3) = 7 or 8 is
the truth. The slice structure, the only mechanism at hand that turns a
finite two-qutrit fact into a statement about every rank-7 configuration,
cannot deliver any pivot restriction better than a factor of 1.38, and the
proposed factor of 28 is out of reach by this route. The coset planes x1 +
x2 + x3 = r give nothing either: V_3 restricts there to the single vector
psi_r, and seven restricted states containing psi_r in their span is a
codimension-2 condition with no useful hitting set. If chi(T3^3) = 8, the
lemma is vacuously true and only a proof independent of the scan would be
useful; none is in sight. The recommendation of the exclusion note stands:
run the three-pivot scan in the orbit-block order without a rare-type
restriction.

What was verified computationally (`rare_pivot_m2.py`, about one minute):
Fact 1 on all nine slices; the size-3, size-4, size-5 and size-7 statements
of section 3 in exact arithmetic; the lift table and the 21870-state lower
bound. What was argued: Lemma 2 and the cost accounting of section 1.
