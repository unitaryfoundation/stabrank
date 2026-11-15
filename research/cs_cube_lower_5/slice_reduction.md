# Analytic reduction to independent four-qubit block slices

6 November 2026. This note supplies the analytic prerequisite for the portable,
unfiltered certificate. It performs no target computation and does not claim
that the portable program has completed. It requires the regenerated exact
minimum-three census described in A.1. No product-family, one-entangled-family
or fidelity-filter certificate is used.

Let \(v=(1,1,1,i)\), \(T=v\otimes v\) and \(V=v^{\otimes3}\).
The original blocks are pairs of physical qubits. All coefficients may be
arbitrary complex numbers; summands may have any affine support.

**Independent-slice theorem.** Subject to the finite hypothesis in A.1, every
hypothetical four-term stabilizer decomposition of \(V\) has four nonzero,
linearly independent conditional columns at every one of the twelve original
two-qubit block outcomes. Moreover \(\chi(V)\geq4\).

We use the affine-quadratic representation: on an affine support with binary
coordinates \(u\), a stabilizer phase is
\(i^{\sum_j\ell_j u_j+2\sum_{j<k}b_{jk}u_ju_k}\), with
\(\ell_j\in\mathbb Z_4\), \(b_{jk}\in\mathbb F_2\), up to a scalar.
Computational or Pauli postselection gives zero or another stabilizer.
Two nonzero computational fibers have equal amplitude modulus and are related
by an actual Pauli, including its fourth-root scalar. These standard facts
follow from the stabilizer generators or from the constructive representation
in Dehaene and De Moor, Section V, Theorem 5:
[primary text](https://arxiv.org/html/quant-ph/0304125v1).

The target-fixing operations needed here are explicit. For each permutation
\(\pi\) of the four labels of one block, put
\((U_\pi a)(x)=v(x)a(\pi x)/v(\pi x)\).
Every such permutation is affine on \(\mathbb F_2^2\), and the phase correction
has an even mixed coefficient: its exponent is the difference of two
single-point indicators modulo four. Thus \(U_\pi\) is a monomial Clifford
fixing \(v\). This formula, not a claim about the full Clifford little group,
is sufficient for every symmetry step below.

The canonical base identity is

\[
q=x_1x_2+x_3x_4,\qquad
f(x)=i^{x_1+x_2+x_3+x_4},\qquad
g(x)=(-i)^{x_1+x_2+x_3+x_4}(-1)^q,
\]
\[
T=2\delta_0-\frac{1+i}{2}f-\frac{1-i}{2}g. \tag{3}
\]

This identity and the earlier block-lift approach are already recorded by the
[official CS project](https://unitaryfoundation.github.io/stabrank/orbits/CS.html).
They are not claimed as newly invented. Normalization is explicit: the singleton
has norm one and the full-support phase vectors have norm four. The exact
minimum-three completeness required below is a separate finite gate; the old
numerical cube certificate is not substituted for it.


### A.1. The exact base theorem and its span

The finite hypothesis is that the package's regenerated exact census covers every four-qubit stabilizer ray and every minimum-three cover, excludes rank two, and verifies that all resulting triples are in the monomial target-fixing orbit of (3). The historical complete calculation used 126 single-ray orbits covering all 36,720 rays and found sixteen unordered triples. The regenerated verifier must establish coverage and the actual orbit equality, rather than assume them from those expected counts. Zero quotient cases must remain shorter-rank guards. All subsequent statements in this note are conditional on that complete finite hypothesis.

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
\tag{17}
\]

A nonzero \(w_1\) has one entry of modulus one; (17) would require \(4\leq2\sqrt2\) there. Hence \(w_1=0\). Nonzero \(w_2,w_3\) would then be parallel. This is impossible: \(f\) is a product of four one-qubit states, whereas the first two-qubit factor of \(g\) has amplitude determinant 2. Pauli images preserve that entanglement distinction. Thus all \(w_j=0\). Applying \(S\) to the ancilla gives the same statement for \(|+i\rangle\).

A rank-three four-column cover of either five-qubit target contains only repetitions of these three inherited rays: select an independent triple, apply inheritance, and use A.1. Repetitions retain their coefficient freedom; only their sum on each ray is fixed.

Write the last block as \((a,b)\). Its \(b=0,1\) targets are \(T\otimes+\) and \(T\otimes+i\). A hypothetical three-term global cover is visible in both slices. Inheritance would require each term's ancillary factor to change from an X eigenstate to a Y eigenstate under a Pauli, which is impossible. Thus \(\chi(V)\geq4\).

In a four-term cover, suppose only three terms are visible at one \(b\) outcome. They inherit its ancillary X or Y eigenstate. At the opposite outcome each shared term contributes to just one of the corresponding two eigensectors. The one term absent from the first slice contributes to at most both sectors. The two target sectors are nonzero multiples of \(T\), so they require at least six incidences; at most \(3+2=5\) are available. Hence all four terms are visible in both slices. If one slice had rank three, A.1–A.2 would make all four terms inherited repeats. The opposite sectors would receive at most four incidences, again fewer than six. Each one-bit slice therefore has rank four.

### A.3. Full block projection and the only deficient patterns

An affine support projects onto one, two or four values of an original two-qubit block. If its image had at most two values, the explicit local Clifford \(U_\pi\) defined above would move it into one bit half, contradicting the one-bit visibility just proved. Therefore every summand projects onto all four values. All its fibers on that block have equal size.

A single block outcome has target proportional to \(T\), so its column rank is three or four. At rank three, choose an independent triple. The exact base theorem and A.1 force the fourth ray to repeat one member of (3). Because the orbit maps are monomial, the slice-support sizes are \((1,1,16,16)\) or \((1,16,16,16)\). Equal fiber sizes force global-support sizes \((4,4,64,64)\) or \((4,64,64,64)\). We exclude both next.

### A.4. Two full-support terms agree at at most 54 coordinates

**Lemma 5.** For full-support stabilizer phases \(f_0,g_0\) and any complex \(u,w\), the function \(uf_0+wg_0\) agrees with \(V\) on at most 54 of 64 coordinates. The bound is attained.

**Proof.** Write \(a=x_1x_2,b=x_3x_4,c=x_5x_6\), and let \(Q=a+b+c\) as a Boolean function. A stabilizer phase exponent has affine low bit and quadratic high bit. A nonzero Boolean polynomial of degree at most two on six bits has weight at least 16: the general bound \(2^{n-d}\) follows by splitting on the last variable and inducting. Hence a stabilizer phase lying in any specified set of at most two fourth roots at more than 48 points lies in that set everywhere. For opposite-parity roots the membership equation is quadratic in the high and low bits; for same-parity roots it fixes the affine low bit. A nonconstant two-valued phase takes each value at least 16 times.

Assume at least 55 agreements. If one coefficient vanishes, or the two phases are parallel, agreement at one point fixes the effective scalar to a fourth root. Phase parity would require the nondegenerate quadratic \(Q\) to equal an affine function at more than 36 points, impossible by its Walsh spectrum. Otherwise the circles \(|z|=1\) and \(|u+wz|=1\) have at most two intersections. The ratio \(g_0/f_0\) therefore lies in two fourth roots on the agreement set, hence globally. Each ratio value occurs at an agreement point, so both values of \(u+w(g_0/f_0)\) are themselves fourth roots.

If the ratio values have opposite parity, their indicator is affine; replacing them by any fourth roots yields another stabilizer phase and the preceding 36-point obstruction applies. Otherwise the ratio is a constant times \((-1)^h\), with \(h\) Boolean quadratic. Unless the output is again stabilizer, it has form \(i^q i^{\epsilon h}\), \(\epsilon=\pm1\). Parity agreement at more than 48 points forces \(h=Q+L\) identically for an affine \(L\).

The polar form of \(Q\) is nondegenerate, so translation removes the linear part of \(L\). The target's ratio under that translation is a stabilizer phase. After absorbing this phase and a constant, the discrepancy is therefore

\[
(-1)^{F+r},\qquad F=ab+ac+bc,
\tag{18}
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
\tag{19}
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

Now assume the remaining pattern \((4,64,64,64)\) has a rank-three block outcome. Transport that outcome and its base triple to (3). Two full-support fibers repeat \(f\) or \(g\). A full-support six-qubit extension of \(f\) at \(z=0\) has form \(f(x)i^{r(z)}(-1)^{x\cdot Mz}\), and similarly for \(g\); this follows directly from the affine-quadratic representation stated above. The sparse support has exactly one point \(y_z\) in each block fiber. Divide each fiber equation by \(f\) and the target's nonzero block phase. It has form (19), with \(\epsilon=1\); duplicating \(g\) instead swaps the two constants and gives \(\epsilon=-1\). Thus \(y_z=0\) for all four outcomes. The sparse summand's projection onto either other original block is then a singleton, contradicting A.3. Both deficient patterns are excluded, proving the independent-slice theorem stated above.


## Dependency boundary

Beyond the regenerated finite minimum-three hypothesis in A.1, the proof is
analytic. The old 31-pattern enumeration of one known span is unnecessary:
A.1 gives its independent support-and-phase proof. The quadratic-weight and
bent-sign arguments require no sampled classification or optimization. The
support permutations are justified by the explicit Clifford formula above.
No claim that an optimal decomposition is symmetric or has full six-bit
support is used. The conclusion only makes a complete four-column base census
applicable; excluding those bases by the exact four-fiber test remains a
separate complete computation in the portable certificate.

This note is derived from the reviewed manuscript Appendix A and preserves
the analytic scope of the internally accepted historical chain. It does not
itself activate a portable result, report a new execution, establish priority,
or claim human expert review.
