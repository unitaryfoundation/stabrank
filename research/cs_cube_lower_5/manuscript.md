# Excluding four-stabilizer decompositions of three controlled-S states

**Review draft — 6 November 2026.**

## Abstract

We give a computer-assisted exclusion of four-term stabilizer decompositions of three controlled-S magic states, allowing arbitrary complex coefficients and affine supports. A target-specific slice argument reduces every hypothetical four-term decomposition to an independent four-column decomposition of two controlled-S states. The portable certificate regenerates the four-qubit stabilizer dictionary and exhausts 331,913 retained-pair classes under explicit target-fixing unitary and antiunitary actions. It records 5,877 distinct four-column sets representing all base-cover orbits. A necessary-and-sufficient four-fiber test excludes all 667 resulting orbits, checking 1,693 ordered transition pairs and 27,088 crossing-sign assignments. No stored scientific input, family exclusion or fidelity filter is used. Together with the known six-term construction, this gives \(5\leq\chi(|CS\rangle^{\otimes3})\leq6\). The argument does not determine whether the exact rank is five or six.

## 1. Result and context

For a nonzero pure state \(\psi\), its stabilizer rank \(\chi(\psi)\) is the least number of stabilizer rays whose complex span contains \(\psi\). Scalars and normalization do not change this rank. Set

\[
v=(1,1,1,i),\qquad T=v\otimes v,\qquad V=v^{\otimes3}.
\tag{1}
\]

Thus \(|CS\rangle=v/2\). The original tensor factors are two-qubit blocks throughout; a four-qubit stabilizer may be entangled across two such blocks.

**Theorem 1 (computer-assisted lower bound).** No four stabilizer rays span \(V\). Consequently,

\[
5\leq\chi(|CS\rangle^{\otimes3})\leq6.
\tag{2}
\]

The computation is finite and exhaustive after the reductions proved below. The saved complete outputs and the independent arithmetic audit establish the reported finite exclusions. It imposes neither a symmetry of the unknown decomposition nor a product restriction on its summands.

The official Stabilizer Rank Challenge records \(\chi(v)=2\), \(\chi(T)=3\), and the previous interval \(4\leq\chi(V)\leq6\); its three-copy lower bound already uses a block slice-and-lift strategy [1]. The present contribution is the complete four-column exclusion with exact arithmetic and the target-specific handling of rank-deficient slices. Neither slice-and-lift nor finite-field screening is claimed as a new general method. The software lineage and the existing upper witnesses are attributed to the upstream project [1,2].

This result concerns exact rank at one tensor power. It supplies no improved asymptotic simulation algorithm. The proof below uses neither a fidelity filter nor a general relation between inverse fidelity and exact rank.

For reference, the upper bound is explicit. On four bits write

\[
q=x_1x_2+x_3x_4,\quad
f(x)=i^{x_1+x_2+x_3+x_4},\quad
g(x)=(-i)^{x_1+x_2+x_3+x_4}(-1)^q.
\]

The known identity is

\[
T=2\delta_{0000}-\frac{1+i}{2}f-\frac{1-i}{2}g.
\tag{3}
\]

Its three summands are stabilizer phase vectors. Tensoring (3) with
\(v=\mathbf1_4+(i-1)\delta_{11}\) gives six terms. The coefficients in (3) use unnormalized phase vectors: \(\|f\|=\|g\|=4\), whereas \(\|\delta_{0000}\|=1\).

## 2. From an arbitrary global cover to an independent base

A stabilizer phase representative has entries in \(\{0,1,i,-1,-i\}\). Its support is affine; in coordinates \(u\in\mathbb F_2^d\) on that support its phase has the form

\[
i^{\ell_1u_1+\cdots+\ell_du_d+2\sum_{j<k}b_{jk}u_ju_k},
\qquad \ell_j\in\mathbb Z_4,\quad b_{jk}\in\mathbb F_2.
\tag{4}
\]

The constant phase is removed by fixing the first nonzero amplitude to one. This is the standard affine-quadratic description of stabilizer states [3]. Pauli postselection preserves a stabilizer or gives zero. Any two nonzero computational fibers of one stabilizer are Pauli-related, with their actual fourth-root phases retained.

**Proposition 2 (independent block slices).** If \(V\) had a four-term stabilizer decomposition, every summand would have nonzero fibers at all four outcomes of each original block. At each of these twelve outcomes, the four conditional columns would be independent.

The proof is given in Appendix A. Its inputs are an exact census of the minimum-three covers of \(T\), an ancillary-factor inheritance argument, and two analytic exclusions of the support patterns forced by a rank-three outcome. The regenerated exact minimum-three census recovers all sixteen covers in one monomial Clifford orbit of (3) and excludes shorter covers. A direct span argument shows that their three-dimensional spans contain no other stabilizer rays. This does **not** assert \(\chi(T)=4\): its rank remains three. Independence is a necessary condition on the four particular columns induced by a hypothetical global cover.

Choose the third block's outcome \(00\). Proposition 2 reduces any hypothetical global decomposition to

\[
T=\sum_{j=1}^4 c_j s_j,
\qquad c_j\ne0,
\qquad \operatorname{rank}[s_1\ s_2\ s_3\ s_4]=4.
\tag{5}
\]

Every \(s_j\) belongs to the complete four-qubit stabilizer dictionary. After canonical ray gauges are fixed, the coefficients are unique and belong to \(\mathbb Q(i)\), by an invertible four-row minor. This conclusion follows from the slice; it is not an assumption that restricts the original arbitrary complex coefficients.

## 3. Complete enumeration of the base problem

### 3.1. Dictionary and symmetry

Enumerating every affine support and every phase polynomial (4) gives exactly

\[
2^4\prod_{j=1}^4(2^j+1)=36{,}720
\tag{6}
\]

projective four-qubit stabilizer states. The construction uses reduced row-echelon coordinates for linear supports and canonical coset representatives. It checks uniqueness, the phase condition and the count; it does not use a sampled or full-support-only dictionary.

For a permutation \(\pi\) of the four labels of one block, define

\[
(U_\pi a)(x)=\frac{v(x)}{v(\pi x)}a(\pi x).
\tag{7}
\]

Every permutation of \(\mathbb F_2^2\) is affine. The correction in (7) has an even quadratic coefficient, since its exponent is the difference of two single-point indicators modulo four. It is therefore a diagonal Clifford phase. Thus \(U_\pi\) is Clifford and fixes \(v\) exactly. Independent block actions and their interchange give a group of order \(24^2\cdot2=1{,}152\) fixing \(T\).

For completeness of the retained-pair parametrization, first choose a representative of each single-ray orbit, then enumerate the orbits of the second ray under that representative's point stabilizer. Identify reversed pairs by an actual group transporter. This represents every unordered pair: any first endpoint can be transported to a chosen representative, after which precisely its stabilizer remains available. The complete orbit weights sum to \(\binom{36,720}{2}=674,160,840\). The recorded maps replay every endpoint, including the reversal cases; a cardinality comparison alone is not used as the coverage proof.

Complex conjugation \(K\), followed by \(CZ\) on each block, gives a target-fixing antiunitary

\[
\mathcal A_2=(CZ\otimes CZ)K.
\tag{8}
\]

It commutes with the actions (7). The enlarged group has order 2,304. The program checks the physical dictionary actions, pair transporters and coefficient phases. The portable implementation constructs the enlarged-group pair catalogue directly and checks its coverage weights. It obtains 331,913 classes without reading the earlier unitary pair catalogue.

These symmetries preserve liftability. A unitary on the two base blocks extends by identity on the third. The antiunitary extends to \(\mathcal A_3=CZ^{\otimes3}K\), which fixes \(V\); the third \(CZ\) compensates the conjugated \(11\) phase. If an action sends a canonical ray to \(\gamma s'\), its transformed coefficient is \(c\gamma\), or \(\overline c\gamma\) at an antiunitary step. No coefficient magnitude or Pauli-type pattern is used as a substitute for an actual action.

### 3.2. Exact quotient-pair test

Fix two retained rays \(s_1,s_2\), and set \(B=[s_1\ s_2\ T]\). Its rank is three because \(\chi(T)=3\). Choose three invertible pivot rows \(P\). For any dictionary column \(w\), write

\[
\alpha(w)=B_P^{-1}w_P,\qquad r(w)=w-B\alpha(w).
\tag{9}
\]

For two nonzero residuals, a four-column completion exists precisely when
\(r(w')=\lambda r(w)\) and the target component of
\(\alpha(w')-\lambda\alpha(w)\) is nonzero. The resulting identity is solved exactly; four-column independence and all four nonzero coefficients are required. A zero full residual leads to a lower-rank span and is processed separately, not silently discarded as an unexamined case.

The screening field is \(\mathbb F_{65521}\), with \(i\mapsto41224\). It is faithful for the ranks used here. For any \(k\times k\) minor of a phase-valued matrix,

\[
|\det M|^2\leq k^k.
\tag{10}
\]

A nonzero Gaussian integer divisible by the Gaussian prime above 65,521 has norm at least 65,521. Since \(5^5=3{,}125<65{,}521\), ranks of matrices with at most five such columns agree with their finite-field ranks. This excludes a hidden bad-prime or denominator exception.

For speed, the implementation first groups a three-coordinate linear projection of each residual. A nonzero projected residual is normalized, and every same-key pair with different normalized target coefficient is considered. Full residual proportionality is then checked in all sixteen coordinates. Every zero projected residual enters a full-coordinate fallback. Same-key groups retain all cross-pairs, including multiple coefficient values. Thus projected collisions can add work but cannot remove a valid completion. Gaussian-rational reconstruction checks every reported identity.

The scan accounts for all 331,913 prepared retained-pair classes. Each accepted identity is canonicalized by its four ray IDs and exact coefficients. Repeated occurrences remain in the execution accounting and are deduplicated consistently. The number of stored four-sets is not the number of all raw dictionary four-subsets spanning \(T\): quotient symmetry has already selected representatives.

### 3.3. Lift every four-cover orbit

Apply the explicit target-fixing generators to each stored cover, retaining every phase and conjugating coefficients at antiunitary steps. The orbit partition must supply a physical transporter for every stored set. Every orbit is passed to the four-fiber test. There is no product, one-entangled, coefficient-cost or other family filter.

The completed census records 6,922 accepted occurrences and 5,877 distinct canonical four-sets. The complete physical group partition gives 667 four-cover orbits. Every one is tested directly. These counts come from the new unfiltered run.

## 4. A complete four-fiber test

Fix (5). Order the added block's fibers as \(00,10,01,11\), with target values \(T,T,T,iT\). The support projection is full by Proposition 2, so every fiber of a summand has the same number of nonzero entries. Gauge its \(00\) fiber to \(s_j\). Independence fixes the global coefficient to \(c_j\).

Use the Pauli convention

\[
P(a,b,k)=i^kX^aZ^b,\qquad
[P(a,b,k)s](x)=i^k(-1)^{b\cdot(x+a)}s(x+a).
\tag{11}
\]

Each four-qubit stabilizer has 64 distinct actual Pauli images: 16 projective rays with all four global phases. Actual vectors, rather than projective rays, are needed because the coefficients in (5) are fixed.

**Lemma 3 (single-summand compatibility).** Given actual images \(P_As\) and \(P_Bs\), a stabilizer with fibers \(s,P_As,P_Bs,q\) exists exactly when

\[
q=\varepsilon P_BP_As,\qquad \varepsilon\in\{1,-1\}.
\tag{12}
\]

**Proof.** The X projections of a stabilizer's stabilizer group equal the translation space of its affine support. Containment is immediate; equality follows by comparing dimensions with its pure-Z stabilizer subgroup. Full projection onto the added block supplies stabilizer generators translating the two added bits individually. Their action gives the \(10\) and \(01\) fibers and then the \(11\) fiber as the product of the two recorded four-qubit Paulis, times a sign from the added-block Z action.

Changing a recorded Pauli representative multiplies it by an operator fixing \(s\) exactly. Commuting such an operator past the other Pauli changes the product by at most a sign. It does not introduce a new factor of \(i\). Conversely, start with \(s\otimes|++\rangle\), apply controlled-\(P_A\) and then controlled-\(P_B\), and optionally apply \(CZ\) between the controls. These are Clifford operations, including the phase gates induced by fourth-root Pauli scalars, and produce both signs in (12). ∎

A transition is an ordered tuple of actual images satisfying

\[
\sum_j c_jP_{Aj}s_j=T.
\tag{13}
\]

All transitions are enumerated by joining the 4,096 values \(c_1u_1+c_2u_2\) with the 4,096 values \(T-c_3u_3-c_4u_4\). Every colliding index pair is retained. Clearing a common denominator gives exact integer keys for all sixteen complex coordinates.

**Proposition 4 (fixed-base criterion).** The base (5) lifts to four stabilizers spanning \(V\) if and only if two transitions \(A,B\) and a sign vector \(\varepsilon\in\{\pm1\}^4\) satisfy

\[
\sum_j c_j\varepsilon_jP_{Bj}P_{Aj}s_j=iT.
\tag{14}
\]

Necessity and sufficiency follow termwise from Lemma 3. The Pauli product includes the sign \((-1)^{b_B\cdot a_A}\); treating projective Pauli labels as commuting would invalidate the test.

For a base with \(m\) transitions, the criterion checks all \(m^2\) ordered pairs and all \(16m^2\) sign assignments. It does not collapse distinct phase-bearing transitions. A positive control replaces the added controlled-S block by \(|++\rangle\) and verifies the four global stabilizers and all sixty-four amplitudes exactly.

The saved outputs contain 581 bases with one transition, 22 with two, and 64 with four. Thus all 667 bases are excluded after checking
\[
581+22\cdot2^2+64\cdot4^2=1{,}693
\]
ordered pairs and \(16\cdot1{,}693=27{,}088\) crossing-sign assignments. The result reports no incomplete scan, unresolved fallback, transition cap or unprocessed orbit. Two exact positive controls cover coefficient-denominator classes one and two. An independent implementation rebuilt the actual Pauli images, joined the alternative partition of terms, and applied each Pauli product directly. It verified every transition and all 27,088 negative sign decisions, as well as both positive controls.

**Proof of Theorem 1.** Appendix A excludes three-term global covers. By Proposition 2, any four-term cover induces an independent base (5). The complete census and actual symmetry transports place it in a recorded cover orbit. Proposition 4 and the complete negative finite test exclude every orbit. Thus no cover of size at most four exists. Equation (3) supplies the upper bound six.

## 5. Reproducibility, verification and limits

The certificate regenerates its dictionary, exact minimum-three classification, pair catalogue, four-covers and lift decisions. Its runtime dependencies are Python and NumPy plus the accompanying certificate modules. It reads no historical result archive, calls no external solver and requires no network. An embedded source manifest binds the code. Analytic prerequisites are supplied in Appendix A and embedded in the package.

All negative decisions use Gaussian arithmetic or finite-field arithmetic justified by (10). Time and allocation limits have no role in deciding existence: an exceeded limit produces an incomplete result. A success marker is printed only after every required stage completes.

| Quantity | Observed saved result |
|---|---:|
| Four-qubit stabilizer rays | 36,720 |
| Minimum-three covers / cover orbits | 16 / 1 |
| Retained-pair classes | 331,913 |
| Accepted four-cover occurrences | 6,922 |
| Recorded four-sets / complete cover orbits | 5,877 / 667 |
| Ordered transition pairs / crossing signs | 1,693 / 27,088 |
| First clean run, reported elapsed time | 718.19 seconds |
| Full native-verifier repeat, elapsed time | 730.73 seconds |

The clean run used four workers on an AMD Ryzen 9 9955HX workstation with 16 cores, 32 logical processors and 31.18 GiB reported physical memory, running Windows 11, Python 3.12.14 and NumPy 2.5.4. These elapsed times are specific to this environment.

The first producer's final terminal output was not retained; its complete saved results were read back and independently audited. A second full run through the pinned challenge verifier invoked the no-argument adapter, regenerated the entire computation, and completed with an observed zero exit code. It reported the local verification tier as verified and reproduced every scientific result field in 730.73 seconds. This local result is separate from upstream CI or maintainer acceptance.

The independent audit checked all 331,913 completed case records, all 8,428 logged identities, and a physical transporter for each of the 5,877 recorded covers to one of the 667 tested representatives. It repeated every lift decision in 17.71 seconds. It did not repeat the entire negative free-pair scan or enumerate every symmetry orbit again; those steps rest on the reviewed exact producer and its complete execution records. Hashes identify the sources and evidence, not a substitute proof.

The result leaves rank five versus six unresolved. It does not establish multiplicativity, a general small-rank classification or an improved asymptotic exponent. The official project still displayed the earlier 4–6 interval when checked on 6 November 2026. That comparison is not proof of priority over all unpublished work.

**AI use and authorship.** Ruoyu Liu initiated and authorized the work. Codex developed the derivations, code and manuscript; separate AI reviews and arithmetic implementations provided the recorded internal checks. No independent human mathematical or code verification is claimed. The human author list and responsibility statement remain to be established for a scholarly submission.

## Appendix A. Proof of the independent-slice reduction

### A.1. The exact base theorem and its span

The complete minimum-three census uses the same dictionary construction and quotient principle as Section 3 with one retained column. All 126 single-ray symmetry orbits, covering all 36,720 rays, are visited. Zero quotient cases are retained as shorter-rank guards. Exact reconstruction finds sixteen unordered minimal three-covers, all in the target-fixing monomial orbit of (3), and no cover of size two. This is the finite base lemma on which the following analytic argument rests.

The span of \(\delta_0,f,g\) contains no other stabilizer rays. To prove this directly, divide a vector \(a\delta_0+bf+cg\) pointwise by \(f\). Put \(h=g/f\). It equals \(+1\) at the origin and at the nine points where both two-bit blocks are nonzero, and \(-1\) at the six points where exactly one block is nonzero. The vector therefore takes values \(D=a+b+c,E=b+c,F=b-c\) on classes of sizes 1,9,6. Its possible nonzero support sizes are \(1,6,7,9,10,15,16\). A stabilizer support has power-of-two size, leaving only the singleton ray or full support.

For full support divide by \(D\). Write its phase on the six-point class as \(p\in\mathbb Z_4\) and on the nine-point class as \(q\). Single-bit values are \(p\). A two-bit value within one block is also \(p\), forcing the mixed coefficient \(-p\) to be even. A two-bit value crossing blocks is \(q\), forcing \(q-2p\) even. At a three-bit point using both blocks, the quadratic prediction is \(2q-2p=0\pmod4\), whereas its actual value is \(q\). Thus \(q=0\) and \(p=0\) or 2. These give precisely \(f\) and \(g\). The conclusion transports under every Clifford.

### A.2. Ancilla inheritance and one-bit visibility

Every minimum-three decomposition of \(T\otimes|+\rangle\) inherits the \(|+\rangle\) factor termwise. Project the ancilla onto \(+\). No term vanishes and the three projected columns are independent, because \(\chi(T)=3\). Transport the resulting base to (3), and gauge the five-qubit columns as

\[
s_j=u_j\otimes + + w_j\otimes -,
\quad (u_1,u_2,u_3)=(\delta_0,f,g).
\]

Each \(w_j\) is zero or an actual Pauli image of \(u_j\). Nonzero fibers have equal support size and equal amplitude modulus, so no unrecorded scalar magnitude occurs. The minus component obeys

\[
4w_1-(1+i)w_2-(1-i)w_3=0.
\tag{15}
\]

A nonzero \(w_1\) has one entry of modulus one; (15) would require \(4\leq2\sqrt2\) there. Hence \(w_1=0\). Nonzero \(w_2,w_3\) would then be parallel. This is impossible: \(f\) is a product of four one-qubit states, whereas the first two-qubit factor of \(g\) has amplitude determinant 2. Pauli images preserve that entanglement distinction. Thus all \(w_j=0\). Applying \(S\) to the ancilla gives the same statement for \(|+i\rangle\).

A rank-three four-column cover of either five-qubit target contains only repetitions of these three inherited rays: select an independent triple, apply inheritance, and use A.1. Repetitions retain their coefficient freedom; only their sum on each ray is fixed.

Write the last block as \((a,b)\). Its \(b=0,1\) targets are \(T\otimes+\) and \(T\otimes+i\). A hypothetical three-term global cover is visible in both slices. Inheritance would require each term's ancillary factor to change from an X eigenstate to a Y eigenstate under a Pauli, which is impossible. Thus \(\chi(V)\geq4\).

In a four-term cover, suppose only three terms are visible at one \(b\) outcome. They inherit its ancillary X or Y eigenstate. At the opposite outcome each shared term contributes to just one of the corresponding two eigensectors. The one term absent from the first slice contributes to at most both sectors. The two target sectors are nonzero multiples of \(T\), so they require at least six incidences; at most \(3+2=5\) are available. Hence all four terms are visible in both slices. If one slice had rank three, A.1–A.2 would make all four terms inherited repeats. The opposite sectors would receive at most four incidences, again fewer than six. Each one-bit slice therefore has rank four.

### A.3. Full block projection and the only deficient patterns

An affine support projects onto one, two or four values of an original two-qubit block. If its image had at most two values, a permutation (7) would move it into one bit half, contradicting the one-bit visibility just proved. Therefore every summand projects onto all four values. All its fibers on that block have equal size.

A single block outcome has target proportional to \(T\), so its column rank is three or four. At rank three, choose an independent triple. The exact base theorem and A.1 force the fourth ray to repeat one member of (3). Because the orbit maps are monomial, the slice-support sizes are \((1,1,16,16)\) or \((1,16,16,16)\). Equal fiber sizes force global-support sizes \((4,4,64,64)\) or \((4,64,64,64)\). We exclude both next.

### A.4. Two full-support terms agree at at most 54 coordinates

**Lemma 5.** For full-support stabilizer phases \(f_0,g_0\) and any complex \(u,w\), the function \(uf_0+wg_0\) agrees with \(V\) on at most 54 of 64 coordinates. The bound is attained.

**Proof.** Write \(a=x_1x_2,b=x_3x_4,c=x_5x_6\), and let \(Q=a+b+c\) as a Boolean function. A stabilizer phase exponent has affine low bit and quadratic high bit. A nonzero Boolean polynomial of degree at most two on six bits has weight at least 16: the general bound \(2^{n-d}\) follows by splitting on the last variable and inducting. Hence a stabilizer phase lying in any specified set of at most two fourth roots at more than 48 points lies in that set everywhere. For opposite-parity roots the membership equation is quadratic in the high and low bits; for same-parity roots it fixes the affine low bit. A nonconstant two-valued phase takes each value at least 16 times.

Assume at least 55 agreements. If one coefficient vanishes, or the two phases are parallel, agreement at one point fixes the effective scalar to a fourth root. Phase parity would require the nondegenerate quadratic \(Q\) to equal an affine function at more than 36 points, impossible by its Walsh spectrum. Otherwise the circles \(|z|=1\) and \(|u+wz|=1\) have at most two intersections. The ratio \(g_0/f_0\) therefore lies in two fourth roots on the agreement set, hence globally. Each ratio value occurs at an agreement point, so both values of \(u+w(g_0/f_0)\) are themselves fourth roots.

If the ratio values have opposite parity, their indicator is affine; replacing them by any fourth roots yields another stabilizer phase and the preceding 36-point obstruction applies. Otherwise the ratio is a constant times \((-1)^h\), with \(h\) Boolean quadratic. Unless the output is again stabilizer, it has form \(i^q i^{\epsilon h}\), \(\epsilon=\pm1\). Parity agreement at more than 48 points forces \(h=Q+L\) identically for an affine \(L\).

The polar form of \(Q\) is nondegenerate, so translation removes the linear part of \(L\). The target's ratio under that translation is a stabilizer phase. After absorbing this phase and a constant, the discrepancy is therefore

\[
(-1)^{F+r},\qquad F=ab+ac+bc,
\tag{16}
\]

with \(r\) Boolean quadratic; the sign choice \(\epsilon=-1\) only adds the quadratic \(Q\) to \(r\).

The support of \(F\) is the union of the three affine planes \(a=b=1\), \(a=c=1\), \(b=c=1\). They meet at the all-one point, have union size ten, and have full six-dimensional affine hull. A quadratic Boolean function on six bits has weight 0, 64, 32, or \(32\pm2^{5-s}\), where its polar rank is \(2s\). This follows by pairing variables in its alternating form; a nonzero radical linear part makes it balanced. Thus the only nonzero weight below 24 is 16, attained by a codimension-two affine-flat indicator.

If \(\operatorname{wt}(r)\geq24\), then \(\operatorname{wt}(F+r)\geq14\). If \(r=0\), its weight is ten. A codimension-two flat cannot contain nine of the ten points of \(\operatorname{supp}F\): it would contain three points of each constituent affine plane, hence every plane and their full affine hull. Its intersection is at most eight, giving \(\operatorname{wt}(F+r)\geq16+10-16=10\). This contradicts at most nine discrepancies. Finally,
\((1+i)/2+(1-i)(-1)^Q/2=i^Q\) has exactly ten discrepancies, proving sharpness. ∎

Two support-four terms cover at most eight coordinates. Outside them, two full-support terms would need at least 56 agreements. Lemma 5 therefore excludes \((4,4,64,64)\).

### A.5. The one-sparse pattern

For the functions in (3), put \(h=g/f\), \(a_\epsilon=-(1+\epsilon i)/2\), \(b_\epsilon=-(1-\epsilon i)/2\). The Boolean sign \(h\) is nondegenerate quadratic, so its normalized four-bit Walsh transform is \(\sigma(k)/4\) for a quadratic bent sign \(\sigma\). In particular
\(\sigma(k+w)\sigma(k)=\kappa_w\chi_{Jw}(k)\), with \(J\) invertible and \(\chi_t(k)=(-1)^{t\cdot k}\).

A bent sign agrees with a real constant-plus-character function at at most ten of sixteen points. Indeed the latter is constant on two eight-point halves. If both constants are signs, the function is a constant sign or a signed character, whose correlation with a bent sign has magnitude four; otherwise agreement is confined to at most one half.

Suppose nonzero complex \(A,B,C,D\) satisfy

\[
A\chi_u+B\chi_v+Ch\chi_w+D\delta_y
=a_\epsilon+b_\epsilon h+2\delta_0.
\tag{17}
\]

Outside the at most three Walsh coordinates \(\{0,u,v\}\), imaginary parts, multiplied by \(4\sigma(k)\), give

\[
C_I\kappa_w\chi_{Jw}(k)=\epsilon/2-D_I\sigma(k)\chi_y(k)/4.
\]

This holds at least thirteen times. The preceding bent-sign bound forces \(D_I=0\); the remaining nonzero constant forces \(w=0,C_I=\epsilon/2\). Real parts then give

\[
(C_R+1/2)\sigma(k)=1/2-D_R\chi_y(k)/4.
\]

The same bound forces \(C_R=-1/2\), then \(y=0,D_R=2\). Character independence gives \(u=v=0\) and \(A+B=a_\epsilon\).

Now assume the remaining pattern \((4,64,64,64)\) has a rank-three block outcome. Transport that outcome and its base triple to (3). Two full-support fibers repeat \(f\) or \(g\). A full-support six-qubit extension of \(f\) at \(z=0\) has form \(f(x)i^{r(z)}(-1)^{x\cdot Mz}\), and similarly for \(g\); this follows directly from (4). The sparse support has exactly one point \(y_z\) in each block fiber. Divide each fiber equation by \(f\) and the target's nonzero block phase. It has form (17), with \(\epsilon=1\); duplicating \(g\) instead swaps the two constants and gives \(\epsilon=-1\). Thus \(y_z=0\) for all four outcomes. The sparse summand's projection onto either other original block is then a singleton, contradicting A.3. Both deficient patterns are excluded, proving Proposition 2.

## Appendix B. Portable certificate map

This map describes the unfiltered program rather than the older filtered archive. Exact hashes belong in the supplement manifest.

| Component | Responsibility |
|---|---|
| cert_cs3_dictionary.py, cert_cs3_exact.py | Complete affine-support dictionary and exact arithmetic |
| cert_cs3_prepare.py, cert_cs3_pair_primitives.py | Minimum-three census and retained-pair symmetry catalogue |
| cert_cs3_projected.py | Projected quotient scan with complete full-coordinate fallback |
| cert_cs3_cover_orbits.py | Physical symmetry orbits with coefficient gauges |
| cert_cs3_fibers.py | Complete transitions and four-fiber compatibility |
| cert_cs3_analytic.py | Embedded analytic slice reduction |
| cert_cs3_parameters.py, cert_cs3_manifest.py | Fixed limits, arithmetic parameters and source hashes |
| cert_cs3_main.py | Complete execution and success guard |
| cert_cs3_challenge.py | Separate no-argument challenge adapter |

The no-argument adapter delegates to the complete core run. Its integration was checked by a full repeat through the pinned upstream verifier, as reported in Section 5.

## References

[1] Unitary Foundation, *Stabilizer Rank Challenge: CS*, official record, accessed 6 November 2026. [CS page](https://unitaryfoundation.github.io/stabrank/orbits/CS.html); [pinned repository](https://github.com/unitaryfoundation/stabrank/tree/edde871fda354479d56e49b93a2502989bd956a2). The previous three-copy lower bound is recorded as reproduced.

[2] F. Labib and V. Russo, *Stabilizer rank bounds for magic-state orbits*, arXiv:2605.28586v1 (2026). [Primary preprint](https://arxiv.org/abs/2605.28586). Cited for the challenge and its methods; the current CS record is cited separately in [1].

[3] J. Dehaene and B. De Moor, *The Clifford group, stabilizer states, and linear and quadratic operations over GF(2)*, arXiv:quant-ph/0304125 (2003), Section V. [Primary text](https://arxiv.org/abs/quant-ph/0304125).
