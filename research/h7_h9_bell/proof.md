# Seven- and nine-copy lower bounds for the qubit H-type orbit

We prove $\chi(H^{\otimes7})\ge8$ and $\chi(H^{\otimes9})\ge10$. The five-copy base is $\chi(H^{\otimes5})\ge6$. Running `python verify_challenge/cert_qubit_h_m7_m9_bell.py` replays the five-copy exclusion, checks the local identities below, and evaluates both spectral contradictions with exact rational arithmetic. The subspace and spectral arguments are given here for review.

We work in the phase convention
$$
\omega=e^{i\pi/4},\qquad T=(|0\rangle+\omega|1\rangle)/\sqrt2,
\qquad t_n=T^{\otimes n},\qquad z_n=Z^{\otimes n}t_n.
$$
The single-qubit Clifford $(I+iX)/\sqrt2$ rotates the Bloch vector of the leaderboard's real $H$ state, $(1,0,1)/\sqrt2$, to that of $T$, $(1,1,0)/\sqrt2$. Thus the two conventions have equal stabilizer ranks. [H-type orbit](https://unitaryfoundation.github.io/stabrank/orbits/qubit_H.html)

## The conjugate plane

Every stabilizer ray can be scaled to have entries in $\{0,1,-1,i,-i\}$. A complex span of such rays is the kernel of linear equations over $\mathbb Q(i)$. Apply the automorphism $\omega\mapsto-\omega$ of $\mathbb Q(\omega)$ over $\mathbb Q(i)$ to those equations at the unnormalized target with amplitudes $\omega^{|x|}$. The resulting amplitudes $(-1)^{|x|}\omega^{|x|}$ are proportional to $z_n$. Consequently every stabilizer span containing $t_n$ contains
$$
U_n=\operatorname{span}\{t_n,z_n\}.
$$
These endpoint vectors are orthonormal since $\langle T|Z|T\rangle=0$.

## A shared check eliminates two atoms

Assume $\chi(t_{n-2})\ge r$ and, for contradiction, that $\chi(t_n)\le K=r+1$. Take an independent spanning list for the target and extend it to exactly $K$ independent stabilizer states $s_i$ if necessary. The computational basis spans the ambient space and $K<2^n$ in both applications.

Suppose a signed Pauli $P$ other than $\pm I,\pm Z^{\otimes n}$ fixes two members of the list. Independent single-qubit applications of the Clifford $C=(X+Y)/\sqrt2$ preserve $t_n$ and swap the labels $X,Y$. We may select sites $j,k$ where the unsigned letters of the conjugated $P$ differ. A nonconstant word already has such a pair. For the constant $X$ and $Y$ words, twist just one site.

Write $R=P_j\otimes P_k$ for the unsigned two-site restriction and $b=R(|01\rangle-|10\rangle)/\sqrt2$. Then $\langle b|R|TT\rangle=0$ by antisymmetry. Also $\langle b|TT\rangle\ne0$ because the two one-qubit vectors $P_jT,P_kT$ are nonparallel, since $T$ is not an eigenvector of the nontrivial Pauli $P_jP_k$ up to phase. The map
$$
(\langle b|_{jk}\otimes I)\,(I-P)/2
$$
therefore sends $t_n$ to a nonzero multiple of $t_{n-2}$. It kills both chosen atoms. Pauli postselection and contraction with a stabilizer bra send every stabilizer to a stabilizer or zero. This would span $t_{n-2}$ with at most $K-2<r$ atoms. Thus only global parity can be a common signed nonidentity check.

For each atom let $L_i$ denote its unsigned stabilizer, a Lagrangian subspace of the binary symplectic Pauli space. The identity $|s_i\rangle\langle s_i|=2^{-n}\sum_{P\in S_i}P$, where $S_i$ is the signed stabilizer group, gives
$$
|\langle s_i,s_j\rangle|^2=
\begin{cases}
2^{\dim(L_i\cap L_j)-n},&\text{if their signs agree throughout the intersection},\\
0,&\text{otherwise}.
\end{cases}
$$
Indeed distinct Pauli labels are trace-orthogonal, and a nontrivial relative sign character sums to zero. If the overlap is nonzero, the intersection has dimension at most one by the shared-check argument. Hence $|\langle s_i,s_j\rangle|^2\le 2^{1-n}$ for $i\ne j$.

## The local Bell obstruction

Suppose an atom $A$ is fixed by $+Q=P_jP'_k$, where $P,P'\in\{X,Y\}$. Define $Q'=\bar P_j\bar P'_k$, where the bar swaps $X,Y$. These are commuting independent Pauli operators and their product has unsigned label $Z_jZ_k$. The joint $(-1,-1)$ eigenstate $b_0$ of $Q,Q'$ is a Bell state. Its squared overlap with $TT$ is zero, while the joint $(-1,+1)$ eigenstate has squared overlap $1/4$. Indeed the two-qubit projector for joint eigenvalues $u,v$ is $(I+uQ)(I+vQ')/4$, and the expectations of $Q,Q',QQ'$ on $TT$ are $1/2,1/2,0$, respectively.

Consider another atom $B$ with unsigned stabilizer $L$. Use lowercase $q,q'$ for the unsigned labels of $Q,Q'$ and write $[\cdot,\cdot]$ for the symplectic commutation form. If neither $q'$ nor $q+q'$ is in $L$, there exists $a\in L$ with $[a,q]=0$ and $[a,q']=1$. In fact, absence of such an $a$ would imply
$$
q'\in(L\cap q^\perp)^\perp=L^\perp+\langle q\rangle=L+\langle q\rangle.
$$
Give the Pauli with label $a$ the sign $R$ that fixes $B$, and put $b_1=R_{jk}b_0$ using the unsigned pair restriction. The Bell eigenvalues of $b_1$ are $(-1,+1)$. Apply $(I-R)/2$ followed by $\langle b_1|_{jk}$. This kills $B$. It also kills $A$ because $A$ and $RA$ both lie in the $+1$ eigenspace of $Q$, whereas $b_1$ has eigenvalue $-1$. On $t_n$ the untranslated term survives, and the $Rt_n$ term vanishes since $\langle b_1|R_{jk}|TT\rangle=\langle b_0|TT\rangle=0$ up to phase. Once again at most $K-2$ stabilizers would span $t_{n-2}$.

It follows that each of the $K-1$ atoms other than $A$ carries the unsigned label $q'$ or $q+q'$. Assign each one such label and its sign. There are only four choices. Neither label is global parity, so the shared-check result allows at most one atom per choice. In the two applications $K=7,9$, giving $K-1>4$. This rules out $+Q$ for every pair of sites and every equatorial label pair.

## The spectral contradiction

Put $M=\sum_{j=1}^n C_j$. The local obstruction implies $\langle s_i|M^2|s_i\rangle\le n$, because each cross term is a positive linear combination of equatorial two-site Pauli expectations, all zero or $-1$. Each $C_j$ has eigenvectors $T,ZT$ with eigenvalues $+1,-1$. The endpoint eigenspaces of $M$ have squared eigenvalue $n^2$ and joint projector $P_U$ onto $U_n$. Every other eigenvalue of $M$ is an odd integer for the sizes at issue. Thus $M^2\succeq I+(n^2-1)P_U$ and
$$
p_i=\langle s_i|P_U|s_i\rangle\le\frac{n-1}{n^2-1}=\frac1{n+1}.
$$

Let $S$ have normalized columns $s_i$, let $G=S^\dagger S$, and use $\|\cdot\|_F$ for the Frobenius norm. Since $U_n\subseteq\mathrm{range}(S)$ and $S$ has rank $K$, $\sum_i p_i=\mathrm{Tr}(P_U SS^\dagger)$ is at least the sum of the two smallest eigenvalues of $G$. This follows by diagonalizing $SS^\dagger$ on its range, where it has the same eigenvalues as $G$. For any two-dimensional orthogonal projector $E$ in coefficient space, $\mathrm{Tr}(G-I)=0$ gives
$$
\mathrm{Tr}(EG)
=2+\mathrm{Tr}\big[(E-2I/K)(G-I)\big]
\ge 2-\sqrt{2-4/K}\,\|G-I\|_F.
$$
The overlap bound implies $\|G-I\|_F^2\le K(K-1)2^{1-n}$. Taking $E$ on the two lowest eigenspaces yields the necessary inequality
$$
\frac{K}{n+1}\ \ge\ 2-\sqrt{2(K-2)(K-1)2^{1-n}}.
$$

For both rows, $2-K/(n+1)>0$. The last column subtracts the squared radical from the square of that positive quantity. Its strict positivity contradicts the necessary inequality.

| $n$ | Input $r$ | Excluded $K=r+1$ | $K/(n+1)$ | Exact squared gap |
|---:|---:|---:|---:|---:|
| 7 | 6 | 7 | $7/8$ | $21/64$ |
| 9 | 8 | 9 | $9/10$ | $309/400$ |

The first input comes from the five-copy certificate. The first row supplies the second input. The padding argument covers every smaller rank at the same time.

## The independent five-copy certificate

The base lower bound also appears in [the original attested entry](../../bounds/qubit_H-m5-lower-6.json). The [standalone verifier](../../verify_challenge/cert_qubit_h5_gaussian_base.py) regenerates the following exclusion in a temporary directory using GNU C++17 and two OpenMP threads.

Put $\omega=e^{i\pi/4}$ and define Gaussian vectors $E(x)=\mathbf 1_{|x|\text{ even}}i^{|x|/2}$ and $O(x)=\mathbf 1_{|x|\text{ odd}}i^{(|x|-1)/2}$. The target amplitude vector is proportional to $E+\omega O$. Stabilizer rays have Gaussian representatives whose entries are zero or fourth roots of unity. Every span of these rays containing $E+\omega O$ also contains $E-\omega O$, by conjugating over $\mathbf Q(i)$. Thus it contains the plane $U=\operatorname{span}\{E,O\}$. The same statement applies after Gaussian Pauli postselection and Clifford decoding.

If an atom is fixed by signed Pauli $P$, projection onto its opposite eigenspace removes it. The remaining projected atoms decode to stabilizer states on four qubits or zero. A five-term representation therefore requires rank at most four for every opposite branch associated with each of its nonzero terms.

The physical symmetries used are permutations and the single-site Clifford $C=(X+Y)/\sqrt 2$, which fixes the target ray and interchanges $X,Y$ while flipping the sign of $Z$. An unsigned Pauli representative has $a$ letters $X$, followed by $b$ letters $Z$, followed by identities. If $b>0$, the two signs give equivalent targets. If $b=0$, the code tests the signs independently.

### The branch search

The four-qubit dictionary contains 36,720 rays. The program generates every affine support and each phase $i^{\ell(x)}(-1)^{q(x)}$ on that support, normalized at one point. A representative encoded basis vector for an $X$-containing Pauli branch has nonzero entries at $x$ and $x+u$, where $u$ is its $X$ support, the pivot bit of $x$ is zero and the relative sign is $\lambda(-1)^{v\cdot x}$ for the $Z$ mask $v$. For a diagonal Pauli, the pivot bit is determined by the parity equation. This gives a Gaussian Clifford decoding.

For a branch plane of dimension two, four independent atoms span the target precisely when their images in the quotient span at most two dimensions. The program tests target-contained rays, proportional quotient images, and every quotient line through each symmetry representative. For a line $L$ it tests the rank of atoms in its inverse image. All projected vectors are normalized using the leading nonzero field entry. Hash tables store the entire 16-coordinate vector and compare all entries for equality, including in the event of a hash collision. Each candidate line is tested directly in the full four-qubit space.

Symmetries of a branch include permutations among identical local letters, arbitrary $C$ at identity sites, pairs of $C$ on $Z$ sites, and global $Z^{\otimes5}$ when it preserves the selected branch. If the number of equatorial letters is odd and a $Z$ site is available, combining global parity with $C$ on that site preserves the branch. These transformations preserve the target plane, possibly exchanging its generators.

Each branch record reports its equatorial and diagonal weights `a,b` and retained `outcome`. A `rank_or_lower` of five denotes a proved lower bound of five. A result of two, three, or four comes with a witness and excludes every smaller value. Zero marks the one-dimensional conjugate-plane case, which the filter conservatively permits without a rank assumption. As positive controls, every single-site branch recovers four atoms and the weight-two diagonal branch recovers three. Exactly nine of the 25 branch classes are excluded.

### The five-qubit search

The five-qubit dictionary contains 2,423,520 rays and has 1,581 orbits under the local $C$ operations and permutations. For each representative, the program evaluates the eigenvalues of all Pauli operators on its amplitude table. It discards the representative if any opposite branch has rank at least five. Exactly 224 rays, in seven orbits, remain.

For each retained anchor $s_a$ up to symmetry, take all pairs $s_b,s_c$ that complete an independent triple in the quotient by $U$. Form $W=U+\operatorname{span}\{s_a,s_b,s_c\}$. Any five independent candidate atoms spanning $U$ would include such a triple and lie in this five-dimensional space. Across 173,223 choices the largest dimension of the span of all retained atoms contained in $W$ is four. If a decomposition used fewer than five independent retained rays, one could extend it to five with other retained rays. The retained list spans at least five dimensions since it contains all computational basis rays, as one can also check directly.

### Finite-field validity

For $k\le5$, any $k$ independent Gaussian stabilizer columns have a nonzero minor whose Gaussian norm is at most $k^k\le3125$, by Hadamard's inequality. A Gaussian integer mapping to zero under $i\mapsto256\pmod {65537}$ has norm divisible by 65537. Hence these independent columns remain independent after reduction. If a complex span of $k$ such columns contains a target plane, all relevant augmented minors vanish identically and its reduction contains the reduced plane. The two physical target columns are visibly independent in the reduction. Branch code separately computes the dimension of each reduced plane. These implications suffice for every negative certificate.

For a positive branch witness of length $k\le4$, reduce the augmented minors with one target column at a time. Its Gaussian coordinates have modulus at most two. Each determinant of size $k+1$ consequently has norm at most $4(k+1)^{k+1}\le12500<65537$. Vanishing modulo the chosen prime implies actual vanishing, so these positive controls lift as well.
