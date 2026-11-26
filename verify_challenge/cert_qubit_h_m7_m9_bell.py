"""Exact H-type lower bounds at seven and nine copies.

Replay the five-copy exclusion, verify the two-qubit Clifford and Bell identities
and apply the rank-increment proof in research/h7_h9_bell/proof.md. The proof
explains the Gaussian conjugate plane, Pauli multiplicity and Gram estimate.
This program runs with Python's standard library.
"""

from fractions import Fraction
from itertools import combinations, combinations_with_replacement, product
from math import comb
from pathlib import Path
import subprocess
import sys


LETTERS = "IXYZ"
PAIR_WORDS = tuple(product(LETTERS, repeat=2))
# Nonzero entries encode omega**e, where omega is a primitive eighth root.
TT = (0, 1, 1, 2)
SINGLET = (None, 0, 4, None)
ZERO = (0, 0, 0, 0)
BASE_CLAIM = "CERTIFIED chi(H_5) >= 6"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pauli_action(word, vector):
    """Apply a tensor product of Pauli letters or C=(X+Y)/sqrt(2)."""
    answer = [None] * len(vector)
    for source, exponent in enumerate(vector):
        if exponent is None:
            continue
        target = source
        for site, gate in enumerate(word):
            mask = 1 << (len(word) - site - 1)
            bit = bool(source & mask)
            if gate in "XYC":
                target ^= mask
            if gate == "Y":
                exponent += -2 if bit else 2
            elif gate == "Z" and bit:
                exponent += 4
            elif gate == "C":
                exponent += -1 if bit else 1
        answer[target] = exponent % 8
    return tuple(answer)


def signed(vector, eigenvalue):
    return tuple(None if e is None else (e + (4 if eigenvalue == -1 else 0)) % 8
                 for e in vector)


def overlap(left, right):
    """Unnormalized inner product in the integer basis 1, omega, omega^2, omega^3."""
    out = [0] * 4
    for a, b in zip(left, right):
        if a is not None and b is not None:
            exponent = (b - a) % 8
            out[exponent % 4] += -1 if exponent >= 4 else 1
    return tuple(out)


def squared_norm(value):
    result = [0] * 4
    for i, x in enumerate(value):
        for j, y in enumerate(value):
            exponent = (i - j) % 8
            result[exponent % 4] += (-1 if exponent >= 4 else 1) * x * y
    return tuple(result)


def check_peeling():
    magic = (0, 1)
    require(pauli_action("C", magic) == magic, "C must fix the magic state")
    for letter, (mapped, sign) in {"I": ("I", 1), "X": ("Y", 1),
                                     "Y": ("X", 1), "Z": ("Z", -1)}.items():
        for basis in ((0, None), (None, 0)):
            conjugated = pauli_action("C", pauli_action(letter, pauli_action("C", basis)))
            require(conjugated == signed(pauli_action(mapped, basis), sign),
                    "the local twist conjugated a Pauli incorrectly")
    for first, second in product("IXYZ", repeat=2):
        if first == second:
            continue
        word = (first, second)
        bra = pauli_action(word, SINGLET)
        require(overlap(bra, pauli_action(word, TT)) == ZERO,
                "the translated singlet failed to kill the Pauli-translated input")
        require(overlap(bra, TT) != ZERO, "the translated singlet killed the target")

    for n in (7, 9):
        eligible = 0
        for word in combinations_with_replacement(LETTERS, n):
            if len(set(word)) == 1 and word[0] in "IZ":
                continue
            # For a constant equatorial word, C on the first site swaps X and Y.
            adjusted = list(word)
            if len(set(word)) == 1:
                adjusted[0] = {"X": "Y", "Y": "X"}[word[0]]
            require(any(a != b for a, b in combinations(adjusted, 2)),
                    "a nonexceptional check has no usable pair of sites")
            eligible += 1
        require(eligible == comb(n + 3, 3) - 2, "incomplete Pauli word enumeration")
    print("Checked all 12 distinct local Pauli pairs and every word type at n=7,9.")


# Two-qubit unsigned Pauli labels are written (x1 x2 z1 z2) in binary.
def label(word):
    x = z = 0
    for letter in word:
        x = (x << 1) | int(letter in "XY")
        z = (z << 1) | int(letter in "YZ")
    return (x << 2) | z


def pairing(p, q):
    return (((p >> 2) & (q & 3)) ^ ((p & 3) & (q >> 2))).bit_count() % 2


def local_subspaces():
    zero_space = frozenset({0})
    spaces = [zero_space]
    seen = {zero_space}
    for space in spaces:
        for v in range(1, 16):
            if v in space:
                continue
            enlarged = space.union(u ^ v for u in space)
            if enlarged not in seen:
                spaces.append(enlarged)
                seen.add(enlarged)
    require(len(spaces) == 67, "incomplete enumeration of subspaces of F_2^4")
    return spaces


def check_local_obstruction():
    spaces = local_subspaces()
    words_by_label = {label(w): w for w in PAIR_WORDS}
    require(len(words_by_label) == 16, "local Pauli labels must be distinct")
    swap_xy = {"X": "Y", "Y": "X"}
    for word in product("XY", repeat=2):
        complement = tuple(swap_xy[v] for v in word)
        q, other = label(word), label(complement)
        require(q ^ other == label("ZZ") and pairing(q, other) == 0,
                "equatorial pair identities failed")
        dark = SINGLET if word[0] == word[1] else pauli_action("IC", SINGLET)
        require(pauli_action(word, dark) == signed(dark, -1), "wrong dark Q eigenvalue")
        require(pauli_action(complement, dark) == signed(dark, -1),
                "wrong dark Q' eigenvalue")
        require(overlap(dark, TT) == ZERO, "the dark Bell state's overlap is nonzero")

        for r, r_word in words_by_label.items():
            if pairing(r, q) or not pairing(r, other):
                continue
            bright = pauli_action(r_word, dark)
            require(pauli_action(word, bright) == signed(bright, -1),
                    "wrong bright Q eigenvalue")
            require(pauli_action(complement, bright) == bright, "wrong bright Q' eigenvalue")
            # The inputs have squared norms two and four. A numerator of two
            # therefore means a normalized probability of one quarter.
            require(squared_norm(overlap(bright, TT)) == (2, 0, 0, 0),
                    "wrong bright Bell probability")
            require(overlap(bright, pauli_action(r_word, TT)) == ZERO,
                    "the translated component survived the Bell contraction")

        for space in spaces:
            # The local image of a full stabilizer L can be any of these spaces.
            # Its orthogonal complement consists of supported Pauli checks in L.
            other_supported = all(not pairing(r, other) for r in space)
            product_supported = all(not pairing(r, q ^ other) for r in space)
            eliminator_exists = any(not pairing(r, q) and pairing(r, other) for r in space)
            require(eliminator_exists == (not other_supported and not product_supported),
                    "local commutation alternative failed")
    print("Checked all four equatorial label choices and all 67 local Pauli subspaces.")


def certify_increment(n, previous_lower):
    # Counterhypothesis: pad an independent putative span to this many columns.
    k = previous_lower + 1
    require(k - 2 < previous_lower and 2 <= k < 2**n, "invalid input to peeling")
    # Once each other atom must carry Q' or QQ', there are only four signed slots.
    require(k - 1 > 2 * 2, "the four-slot pigeonhole argument does not apply")

    # M=sum C_j has eigenvalue n-2w at local minus-weight w.
    remaining_min = min((n - 2 * w)**2 for w in range(1, n))
    plane_mass_upper = k * Fraction(n - remaining_min, n*n - remaining_min)
    mu_squared = Fraction(1, 2**(n - 1))
    gram_deviation_squared = 2 * (k - 2) * (k - 1) * mu_squared
    gap = 2 - plane_mass_upper
    require(gap > 0 and gap*gap > gram_deviation_squared,
            "the spectral estimates permit the counterhypothesis")
    print(f"n={n}, excluded K={k}: trace upper {plane_mass_upper}, "
          f"positive squared gap {gap*gap - gram_deviation_squared}")
    return k + 1


def main():
    # Run the base certificate afresh and require its actual success sentinel.
    replay = Path(__file__).with_name("cert_qubit_h5_gaussian_base.py")
    result = subprocess.run([sys.executable, str(replay)], stdout=subprocess.PIPE,
                            text=True, check=True)
    print(result.stdout, end="")
    require(BASE_CLAIM in result.stdout.splitlines(), "five-copy replay omitted its claim")

    check_peeling()
    check_local_obstruction()
    lower7 = certify_increment(7, 6)
    lower9 = certify_increment(9, lower7)
    require((lower7, lower9) == (8, 10), "unexpected finite lower bounds")
    print("CERTIFIED chi(qubit_H^7) >= 8")
    print("CERTIFIED chi(qubit_H^9) >= 10")


if __name__ == "__main__":
    main()
