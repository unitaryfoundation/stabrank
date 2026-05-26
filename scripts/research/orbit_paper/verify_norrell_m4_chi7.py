"""Symbolic verification of the closed-form |N_+>^⊗4 = sum_j alpha_j |S_j>
decomposition with chi = 7, for the proof in Appendix A.3.3.

For each indicator pattern a = (a_0, a_1, a_2, a_3) in {0,1}^4, computes the
right-hand side amplitude as an exact symbolic expression in the 12th root
of unity xi = exp(i pi / 6), and checks it equals (-2)^n_2(a) / 36, where
n_2(a) = sum_i a_i. Uses sympy for symbolic simplification.

Run: uv run --with sympy python verify_norrell_m4_chi7.py
"""
import sympy as sp


def main() -> int:
    omega = sp.exp(2 * sp.pi * sp.I / 3)
    xi = sp.exp(sp.pi * sp.I / 6)
    sqrt3 = sp.sqrt(3)
    sigma = sqrt3 / 4

    # Effective alpha on canonical |S_j>:
    a_eff = {
        0: sigma * omega,
        1: sigma * xi,
        2: sigma * xi,
        3: sigma * xi,
        4: sigma * omega,
        5: sigma * xi,
        6: sp.Rational(1, 4) * omega**2,
    }

    # Normalization 1/sqrt(p^k) for k = 3, 4, 2, 2, 3, 2, 4:
    norm = [
        1 / sp.sqrt(27),
        sp.Rational(1, 9),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
        1 / sp.sqrt(27),
        sp.Rational(1, 3),
        sp.Rational(1, 9),
    ]

    def support_ok(j, a):
        a0, a1, a2, a3 = a
        return [
            a1 == 1,
            True,
            a0 == 1 and a3 == 1,
            a2 == 1 and a3 == 1,
            a0 == 1,
            a1 == 1 and a2 == 1,
            True,
        ][j]

    def Q(j, a):
        a0, a1, a2, a3 = a
        return [
            (2 * a0 + a2 + 2 * a3) % 3,
            (2 * a0 + 2 * a1 + a2 + a3) % 3,
            (a1 + a2) % 3,
            (2 * a0 + a1) % 3,
            (2 * a1 + 2 * a2 + a3) % 3,
            (a0 + 2 * a3) % 3,
            0,
        ][j]

    print("Verifying closed-form Norrell m=4 chi=7 decomposition (App A.3.3):")
    print()
    print("  |a|_2 | a                | RHS               | (-2)^n2/36    | residual")
    print("  ------+------------------+-------------------+---------------+---------")
    all_ok = True
    for a in [(a0, a1, a2, a3)
              for a0 in (0, 1) for a1 in (0, 1)
              for a2 in (0, 1) for a3 in (0, 1)]:
        n2 = sum(a)
        target = sp.Rational((-2) ** n2, 36)
        rhs = sum(
            a_eff[j] * norm[j] * omega ** Q(j, a)
            for j in range(7) if support_ok(j, a)
        )
        diff = sp.nsimplify(sp.expand_complex(sp.expand(rhs - target)),
                            rational=False)
        ok = (diff == 0)
        all_ok &= ok
        rhs_str = str(sp.nsimplify(rhs))
        print(f"   {n2}    | {a} | {rhs_str:>17} | {str(target):>13} | "
              f"{('0' if ok else str(diff)):>8}")

    print()
    if all_ok:
        print("ALL 16 INDICATOR-CLASS IDENTITIES SYMBOLICALLY VERIFIED")
        return 0
    print("FAILURE — some residual is nonzero")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
