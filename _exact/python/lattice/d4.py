"""
D4 Lattice - The D4 root lattice with kissing number K(D4) = 24.

The D4 lattice is significant in SRT as K(D4) = 24 defines the
consciousness threshold. Systems with accumulated phase >= 24
exhibit self-referential coherence.

Example:
    >>> from syntonic_applications.srt.lattice import d4_lattice, K_D4
    >>> D4 = d4_lattice()
    >>> len(D4.roots)
    24
    >>> K_D4
    24
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Tuple

# Kissing number of D4 = consciousness threshold
K_D4: int = 24


@dataclass(frozen=True, slots=True)
class D4Root:
    """
    A root of the D4 lattice with integer coordinates.

    D4 roots have the form (+/-1, +/-1, 0, 0) and permutations,
    with all roots having norm^2 = 2.

    Attributes:
        coords: 4-tuple of integer coordinates

    Properties:
        norm_squared: Always 2 for roots
        dimension: Always 4
    """

    coords: Tuple[int, int, int, int]

    def __post_init__(self):
        """Validate that this is a valid D4 root."""
        if len(self.coords) != 4:
            raise ValueError("D4 root must have 4 coordinates")

    @property
    def dimension(self) -> int:
        """Dimension of the space (always 4)."""
        return 4

    @property
    def norm_squared(self) -> int:
        """Squared norm |lambda|^2."""
        return sum(x * x for x in self.coords)

    @property
    def norm(self) -> float:
        """Euclidean norm |lambda|."""
        return math.sqrt(self.norm_squared)

    def to_float(self) -> Tuple[float, ...]:
        """Convert to float tuple."""
        return tuple(float(x) for x in self.coords)  # export boundary

    def inner_product(self, other: D4Root) -> int:
        """
        Inner product <lambda, mu>.

        Args:
            other: Another D4 root

        Returns:
            Integer inner product
        """
        return sum(a * b for a, b in zip(self.coords, other.coords))

    def __neg__(self) -> D4Root:
        """Negation -lambda."""
        return D4Root(tuple(-x for x in self.coords))

    def __add__(self, other: D4Root) -> Tuple[int, ...]:
        """Vector addition (returns tuple, may not be a root)."""
        return tuple(a + b for a, b in zip(self.coords, other.coords))

    def __iter__(self) -> Iterator[int]:
        """Iterate over coordinates."""
        return iter(self.coords)

    def __getitem__(self, index: int) -> int:
        """Index access to coordinates."""
        return self.coords[index]

    def __str__(self) -> str:
        return f"D4({self.coords})"


class D4Lattice:
    """
    The D4 lattice with kissing number K(D4) = 24.

    D4 is defined as vectors in Z^4 with even coordinate sum:
        D4 = {(x1, x2, x3, x4) in Z^4 : x1 + x2 + x3 + x4 is even}

    The 24 roots are all permutations of (+/-1, +/-1, 0, 0).

    Key properties:
        - 24 roots (kissing number K = 24)
        - 12 positive roots
        - Rank 4
        - Related to consciousness threshold in SRT

    Attributes:
        roots: All 24 roots
        positive_roots: 12 positive roots
        kissing_number: 24

    Example:
        >>> D4 = D4Lattice()
        >>> len(D4.roots)
        24
        >>> D4.kissing_number
        24
    """

    def __init__(self):
        """Initialize D4 lattice by computing all roots."""
        self._roots = self._generate_roots()
        self._positive_roots = [r for r in self._roots if self._is_positive(r)]

    def _generate_roots(self) -> List[D4Root]:
        """Generate all 24 D4 roots."""
        roots = []
        # D4 roots are (+/-1, +/-1, 0, 0) and permutations
        # There are C(4,2) = 6 ways to choose positions for +/-1
        # Each has 2^2 = 4 sign combinations
        # Total: 6 * 4 = 24 roots

        from itertools import combinations, product

        # Choose 2 positions for non-zero entries
        for positions in combinations(range(4), 2):
            # Sign combinations
            for signs in product([-1, 1], repeat=2):
                coords = [0, 0, 0, 0]
                coords[positions[0]] = signs[0]
                coords[positions[1]] = signs[1]
                roots.append(D4Root(tuple(coords)))

        return roots

    def _is_positive(self, root: D4Root) -> bool:
        """
        Check if root is positive using lexicographic order.

        A root is positive if the first nonzero coordinate is positive.
        """
        for x in root.coords:
            if x > 0:
                return True
            if x < 0:
                return False
        return False

    @property
    def roots(self) -> List[D4Root]:
        """All 24 roots of D4."""
        return self._roots.copy()

    @property
    def positive_roots(self) -> List[D4Root]:
        """12 positive roots of D4."""
        return self._positive_roots.copy()

    @property
    def num_roots(self) -> int:
        """Number of roots (24)."""
        return len(self._roots)

    @property
    def kissing_number(self) -> int:
        """Kissing number K(D4) = 24."""
        return K_D4

    @property
    def dimension(self) -> int:
        """Dimension of the adjoint representation (28)."""
        return 28

    @property
    def rank(self) -> int:
        """Rank of D4 (4)."""
        return 4

    @property
    def coxeter_number(self) -> int:
        """Coxeter number of D4 (6)."""
        return 6

    def simple_roots(self) -> List[D4Root]:
        """
        Return 4 simple roots (basis for positive roots).

        Standard choice:
            alpha_1 = (1, -1, 0, 0)
            alpha_2 = (0, 1, -1, 0)
            alpha_3 = (0, 0, 1, -1)
            alpha_4 = (0, 0, 1, 1)
        """
        return [
            D4Root((1, -1, 0, 0)),
            D4Root((0, 1, -1, 0)),
            D4Root((0, 0, 1, -1)),
            D4Root((0, 0, 1, 1)),
        ]

    def cartan_matrix(self) -> List[List[int]]:
        """
        Return 4x4 Cartan matrix of D4.

        A_ij = 2 * <alpha_i, alpha_j> / <alpha_j, alpha_j>
        """
        simple = self.simple_roots()
        n = len(simple)
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = (
                    2 * simple[i].inner_product(simple[j]) // simple[j].norm_squared
                )
        return A

    def roots_by_shell(self) -> dict:
        """
        Group roots by squared norm.

        For D4, all roots have norm^2 = 2.

        Returns:
            Dictionary mapping squared norm to list of roots
        """
        shells = {}
        for root in self._roots:
            n_sq = root.norm_squared
            if n_sq not in shells:
                shells[n_sq] = []
            shells[n_sq].append(root)
        return shells

    def __len__(self) -> int:
        """Number of roots."""
        return len(self._roots)

    def __iter__(self) -> Iterator[D4Root]:
        """Iterate over roots."""
        return iter(self._roots)

    def __repr__(self) -> str:
        return f"D4Lattice(roots={len(self._roots)}, K={self.kissing_number})"


def d4_lattice() -> D4Lattice:
    """
    Create a D4Lattice instance.

    Factory function for D4Lattice.

    Returns:
        D4Lattice instance

    Example:
        >>> D4 = d4_lattice()
        >>> D4.kissing_number
        24
    """
    return D4Lattice()
