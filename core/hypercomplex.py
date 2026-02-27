"""
Hypercomplex number systems for Syntonic.

This module provides Quaternions, Octonions, and Sedenions for representing
rotations, orientations, and higher-dimensional algebraic structures.

Quaternions (4D):
    - Non-commutative: q1 * q2 != q2 * q1
    - Associative: (q1 * q2) * q3 == q1 * (q2 * q3)
    - Hamilton product with i² = j² = k² = ijk = -1

Octonions (8D):
    - Non-commutative AND non-associative
    - Cayley-Dickson construction from quaternions
    - Related to exceptional Lie groups (G₂, E₈)

Sedenions (16D):
    - Non-commutative, non-associative, non-alternative
    - Cayley-Dickson construction from octonions
    - WARNING: Has ZERO DIVISORS (ab=0 with a≠0, b≠0)
    - NOT a division algebra
    - Not used in SRT theory (exploration/mathematical completeness)

Usage:
    >>> import syntonic as syn
    >>> q = syn.quaternion(1, 2, 3, 4)  # 1 + 2i + 3j + 4k
    >>> q.norm()
    5.477225575051661
    >>> syn.hypercomplex.I * syn.hypercomplex.J == syn.hypercomplex.K
    True
    >>> # Sedenions have zero divisors:
    >>> a, b = syn.hypercomplex.Sedenion.zero_divisor_pair()
    >>> (a * b).norm()  # Returns ~0
    0.0
"""

from ._core import Octonion, Quaternion, Sedenion

# Unit quaternions (basis elements)
I = Quaternion(0.0, 1.0, 0.0, 0.0)  # i
J = Quaternion(0.0, 0.0, 1.0, 0.0)  # j
K = Quaternion(0.0, 0.0, 0.0, 1.0)  # k

# Unit octonion basis elements
E0 = Octonion(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 1
E1 = Octonion(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # e₁
E2 = Octonion(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # e₂
E3 = Octonion(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # e₃
E4 = Octonion(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # e₄
E5 = Octonion(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # e₅
E6 = Octonion(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # e₆
E7 = Octonion(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # e₇


def quaternion(a, b=0.0, c=0.0, d=0.0):
    """
    Create a quaternion a + bi + cj + dk.

    Args:
        a: Real (scalar) part
        b: i component
        c: j component
        d: k component

    Returns:
        Quaternion object

    Examples:
        >>> q = quaternion(1, 2, 3, 4)  # 1 + 2i + 3j + 4k
        >>> q.real  # property, not method
        1.0
        >>> q.norm()
        5.477225575051661
    """
    return Quaternion(float(a), float(b), float(c), float(d))


def octonion(e0, e1=0.0, e2=0.0, e3=0.0, e4=0.0, e5=0.0, e6=0.0, e7=0.0):
    """
    Create an octonion e0 + e1·e₁ + e2·e₂ + ... + e7·e₇.

    Args:
        e0: Real (scalar) part
        e1-e7: Imaginary basis coefficients

    Returns:
        Octonion object

    Note:
        Octonions are non-associative: (a*b)*c != a*(b*c) in general.
        Use Octonion.associator(a, b, c) to measure non-associativity.

    Examples:
        >>> o = octonion(1, 2, 3, 4, 5, 6, 7, 8)
        >>> o.real()
        1.0
        >>> o.norm()
        14.2828568570857
    """
    return Octonion(
        float(e0),
        float(e1),
        float(e2),
        float(e3),
        float(e4),
        float(e5),
        float(e6),
        float(e7),
    )


def sedenion(
    e0,
    e1=0.0,
    e2=0.0,
    e3=0.0,
    e4=0.0,
    e5=0.0,
    e6=0.0,
    e7=0.0,
    e8=0.0,
    e9=0.0,
    e10=0.0,
    e11=0.0,
    e12=0.0,
    e13=0.0,
    e14=0.0,
    e15=0.0,
):
    """
    Create a sedenion e0 + e1·e₁ + e2·e₂ + ... + e15·e₁₅.

    Args:
        e0: Real (scalar) part
        e1-e15: Imaginary basis coefficients

    Returns:
        Sedenion object

    Warning:
        Sedenions have ZERO DIVISORS! There exist non-zero sedenions a, b
        such that a * b = 0. This means:
        - Division is not always well-defined
        - The composition property ||ab|| = ||a||·||b|| does NOT hold
        - Sedenions are NOT a division algebra

        Use sedenion.has_zero_divisor_with(other) to check before critical operations.

    Note:
        Sedenions are NOT used in SRT theory. Octonions are the "end of the line"
        for exceptional geometry. This is provided for mathematical exploration.

    Examples:
        >>> s = sedenion(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        >>> s.real
        1.0
        >>> s.norm()
        38.6...
        >>> # Zero divisor example:
        >>> a, b = Sedenion.zero_divisor_pair()
        >>> (a * b).norm()  # Returns ~0
        0.0
    """
    return Sedenion(
        float(e0),
        float(e1),
        float(e2),
        float(e3),
        float(e4),
        float(e5),
        float(e6),
        float(e7),
        float(e8),
        float(e9),
        float(e10),
        float(e11),
        float(e12),
        float(e13),
        float(e14),
        float(e15),
    )


# Unit sedenion basis elements
S0 = Sedenion(
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # 1
S1 = Sedenion(
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₁
S2 = Sedenion(
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₂
S3 = Sedenion(
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₃
S4 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₄
S5 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₅
S6 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₆
S7 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₇
S8 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₈
S9 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₉
S10 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # e₁₀
S11 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
)  # e₁₁
S12 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
)  # e₁₂
S13 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
)  # e₁₃
S14 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
)  # e₁₄
S15 = Sedenion(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
)  # e₁₅


__all__ = [
    # Classes
    "Quaternion",
    "Octonion",
    "Sedenion",
    # Factory functions
    "quaternion",
    "octonion",
    "sedenion",
    # Quaternion basis
    "I",
    "J",
    "K",
    # Octonion basis
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    # Sedenion basis
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
]
