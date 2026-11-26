"""Verify the exact stabilizer-rank lower bound for the eight-qubit magic cat.

Postselecting its last qubit on zero gives the seven-qubit cat up to a
nonzero scalar. Each stabilizer term is taken to a stabilizer or zero.

Decode the seven-qubit cat's even parity to six qubits. If a signed Pauli
fixes a term of a rank-k decomposition, the opposite projection has rank
at most k-1. Apply this filter recursively at k=5,4,3. Symmetries are
applied to the checks and to whole triples of candidate rays. Enumerating
all maximal signed commuting groups among the surviving checks gives the
candidate rays for the final span test.

For the span test, normalized stabilizer columns and the target have
Gaussian-integer entries of magnitude at most one. For r <= 5 independent
columns, some r-by-r minor has a nonzero determinant D with norm at most
r**r <= 3125. It stays nonzero under reduction i -> 256 modulo 65537.
Cramer's rule therefore carries a putative complex span relation into
the finite field. The C++ kernels exhaust the possible spans there.

The verifier uses Python and a C++17 compiler. All intermediate files
are generated in a fresh temporary directory.
"""


import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import sympy as sp

from cert_cat_m7_pauli_kernels import KERNELS
from stabrank_verify import target_vector

N = 64  # Six bits after decoding even parity on seven qubits.
REPRESENTATIVES = (-31, -7, -1, 1, 7, 31, 64, 66, 78, 192, 196, 448)
TRIPLE_COUNT = 214060736


def physical_word(word):
    return word | ((bin(word).count("1") % 2) << 6)


def swap_bits(word, pair):
    return word ^ pair if bin(word & pair).count("1") == 1 else word


def logical_pauli(x, z, sign):
    """Restrict a parity-preserving physical Hermitian Pauli to the code."""
    logical_x = x % N
    logical_z = (z % N) ^ (N - 1 if z & N else 0)
    phase = (bin(x & z).count("1") - bin(logical_x & logical_z).count("1")) % 4
    if phase not in (0, 2):
        raise ValueError("non-Hermitian logical restriction")
    return sign * (1 - phase) * (logical_x + N * logical_z)


def conjugate_checks(label):
    x = physical_word(abs(label) % N)
    z = abs(label) // N
    sign = 1 if label > 0 else -1
    for j in range(6):
        pair = 3 << j
        yield logical_pauli(swap_bits(x, pair), swap_bits(z, pair), sign)
        # Conjugate by (X+Y)/sqrt(2) on site j and on the parity site.
        pair = (1 << j) | N
        flips = bin(pair & z & ~x).count("1")
        yield logical_pauli(x, z ^ (x & pair), sign * (-1)**flips)


def check_orbits():
    orbit = {}
    for representative in REPRESENTATIVES:
        if representative in orbit:
            raise ValueError("duplicated first-measurement orbit")
        queue = [representative]
        orbit[representative] = representative
        for pauli in queue:
            for other in conjugate_checks(pauli):
                if other not in orbit:
                    orbit[other] = representative
                    queue.append(other)
    expected = set(range(1, N*N)) | set(range(1 - N*N, 0))
    if orbit.keys() != expected:
        raise ValueError("incomplete first-measurement orbits")
    return orbit


def read_results(path, expected_ids):
    rows = {}
    for line in path.read_text().splitlines():
        label, count, possible = map(int, line.split())
        if label in rows or count < 0 or possible not in (0, 1):
            raise ValueError(f"invalid row in {path.name}")
        rows[label] = (count, bool(possible))
    if rows.keys() != set(expected_ids):
        raise ValueError(f"incomplete results in {path.name}")
    return rows


def write_checks(path, checks):
    with path.open("w") as out:
        for label, values in checks.items():
            print(label, "|", *values, file=out)


def normalized_phases(phases):
    """An integer ray key: -1 for a zero and 0..3 for powers of i."""
    first = next(phase for phase in phases if phase >= 0)
    return tuple((phase - first) % 4 if phase >= 0 else -1 for phase in phases)


def ray_permutations(directory, target_phases):
    states = []
    for line in (directory / "candidates.txt").read_text().splitlines():
        amplitudes = list(map(int, line.split("|")[1].split()))
        if len(amplitudes) != 2*N:
            raise ValueError("wrong dimension for candidate ray")
        scale = max(map(abs, amplitudes))
        phases = {(0, 0): -1, (scale, 0): 0, (0, scale): 1,
                  (-scale, 0): 2, (0, -scale): 3}
        gauged = []
        for j in range(N):
            phase = phases[tuple(amplitudes[2*j:2*j+2])]
            gauged.append((phase - target_phases[j]) % 4 if phase >= 0 else -1)
        states.append(normalized_phases(gauged))
    index = {state: j for j, state in enumerate(states)}
    if len(index) != len(states) or len(index) != 1088:
        raise ValueError("candidate ray enumeration mismatch")
    # Dividing coordinates by the full-support target makes its monomial
    # Clifford symmetries act as permutations, up to a common scalar.
    with (directory / "orbit_perms.txt").open("w") as out:
        for j in range(6):
            swap = [swap_bits(physical_word(y), 3 << j) % N for y in range(N)]
            flip = [y ^ (1 << j) for y in range(N)]
            for permutation in (swap, flip):
                action = [index[normalized_phases([s[y] for y in permutation])] for s in states]
                if len(set(action)) != len(states):
                    raise ValueError("symmetry did not permute the candidate rays")
                print(*action, file=out)


def verify_base(directory):
    for name, source in KERNELS.items():
        (directory / name).write_text(source)
    for name in ("pairs", "third", "lagrangians", "five_span"):
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3",
                        f"{name}.cc", "-o", name], cwd=directory, check=True)

    def run(name, *args, output=None):
        command = [str(directory / name), *map(str, args)]
        if output is None:
            return subprocess.run(command, cwd=directory, check=True, text=True,
                                  stdout=subprocess.PIPE).stdout.strip()
        with (directory / output).open("w") as out:
            subprocess.run(command, cwd=directory, check=True, stdout=out)

    orbits = check_orbits()
    run("pairs", directory)
    target_phases = [((bin(y).count("1") + 1) // 2) % 4 for y in range(N)]
    roots = (1, sp.I, -1, -sp.I)
    cat = target_vector("cat", 7)
    expected = [roots[target_phases[y]] for y in range(N)]
    if any(cat[physical_word(y)] != cat[0] * value for y, value in enumerate(expected)):
        raise ValueError("encoded target disagrees with the challenge cat")
    if any(cat[x] != 0 for x in range(2*N) if bin(x).count("1") % 2):
        raise ValueError("cat target has odd support")
    raw = list(map(int, (directory / "C7.dat").read_text().split()))
    if len(raw) != 1 + 2*N or raw[0] != 1 or any(
        a + sp.I*b != value for a, b, value in zip(raw[1::2], raw[2::2], expected)
    ):
        raise ValueError("C++ target disagrees with the encoded cat")

    run("third", 4, "second_states.dat", output="third_probes.log")
    run("lagrangians", 4, 500, "third_probes.log", "second_states.dat", 3,
        output="third_exact.log")
    pair_types = read_results(directory / "third_exact.log", range(236))
    good = {rep: [] for rep in REPRESENTATIVES}
    examined = {rep: set() for rep in REPRESENTATIVES}
    for line in (directory / "second_ids.txt").read_text().splitlines():
        rep, signed_check, pair_type = map(int, line.split())
        if signed_check in examined[rep]:
            raise ValueError("duplicated second measurement")
        examined[rep].add(signed_check)
        if pair_types[pair_type][1]:
            good[rep].append(signed_check)
    signed_labels = set(range(1, 1024)) | set(range(-1023, 0))
    if any(labels != signed_labels for labels in examined.values()):
        raise ValueError("missing second measurement")
    if sum(passed for _, passed in pair_types.values()) != 101:
        raise ValueError("rank-three filters changed")

    write_checks(directory / "first_valid.log", good)
    run("lagrangians", 5, 800, "first_valid.log", "probe_states.dat", 4,
        output="first_exact.log")
    first_types = read_results(directory / "first_exact.log", REPRESENTATIVES)
    allowed = [pauli for pauli, rep in orbits.items() if first_types[rep][1]]
    if sum(passed for _, passed in first_types.values()) != 7 or len(allowed) != 1666:
        raise ValueError("rank-four filters changed")
    write_checks(directory / "global_valid.log", {1: allowed})
    limit = 10000
    run("lagrangians", 6, limit, "global_valid.log", "C7.dat", 3,
        "candidates.txt", output="global_count.log")
    count, _ = read_results(directory / "global_count.log", [1])[1]
    if count >= limit or count != 1088:
        raise ValueError("incomplete or unexpected final candidate list")
    ray_permutations(directory, target_phases)
    summary = run("five_span", directory)
    if summary != (f"unordered triples {TRIPLE_COUNT} orbits 4831 "
                   "independent modulo target 4819, no rank-five span"):
        raise ValueError(f"unexpected final span result: {summary}")
    print(f"{count} candidate rays; {summary}")


def verify_cat8_projection():
    """Check the normalized projection with exact arithmetic."""
    cat7 = target_vector("cat", 7)
    cat8 = target_vector("cat", 8)
    if cat8[::2, :] != cat7 / sp.sqrt(2):
        raise ValueError("the zero outcome on the last qubit did not give cat_7")


def main():
    verify_cat8_projection()
    with TemporaryDirectory(prefix="stabrank-cat-pauli-") as temporary:
        verify_base(Path(temporary))
    print("CERTIFIED chi(cat_8) >= 6")


if __name__ == "__main__":
    main()
