"""
T4 Torus - The 4-torus internal geometry T^4 = S^1_7 x S^1_8 x S^1_9 x S^1_10.

The internal geometry of SRT is a 4-dimensional torus where fields can
be Fourier expanded in terms of winding modes.

Example:
    >>> from srt_library.theory_unique_components.srt.geometry import create_torus, winding_state
    >>> torus = create_torus(radius=1.0)
    >>> n = winding_state(1, 0, 0, 0)
    >>> mode = torus.fourier_mode(n, (0.5, 0.0, 0.0, 0.0))
"""

from __future__ import annotations

import cmath
import math
from typing import Dict, Iterator, List, Tuple

from srt_library.theory_unique_components.srt.geometry.winding import WindingState, enumerate_windings


class T4Torus:
    """
    T^4 = S^1_7 x S^1_8 x S^1_9 x S^1_10 torus geometry.

    The fundamental domain is [0, 2*pi*l)^4 where l is the radius.
    Fields on the torus can be Fourier expanded:

        Psi(y) = sum_n Psi_hat(n) * exp(i*n.y/l)

    where n = (n_7, n_8, n_9, n_10) are winding numbers.

    Attributes:
        radius: Fundamental radius l (all radii equal)
        dimension: Always 4

    Example:
        >>> torus = T4Torus(radius=1.0)
        >>> torus.volume
        97.409...  # (2*pi)^4
    """

    def __init__(self, radius: float = 1.0):
        """
        Initialize T^4 torus with given radius.

        Args:
            radius: Common radius l for all four circles (default: 1.0)

        Raises:
            ValueError: If radius <= 0
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self._radius = radius

    @property
    def dimension(self) -> int:
        """Dimension of the torus (always 4)."""
        return 4

    @property
    def radius(self) -> float:
        """Common radius l of the four circles."""
        return self._radius

    @property
    def circumference(self) -> float:
        """Circumference of each circle: 2*pi*l."""
        return 2 * math.pi * self._radius

    @property
    def volume(self) -> float:
        """Volume of T^4: (2*pi*l)^4."""
        return self.circumference**4

    def fourier_mode(
        self, n: WindingState, y: Tuple[float, float, float, float]
    ) -> complex:
        """
        Evaluate single Fourier mode exp(i*n.y/l).

        Args:
            n: Winding state |n_7, n_8, n_9, n_10>
            y: Position on torus (y_7, y_8, y_9, y_10)

        Returns:
            Complex value exp(i*n.y/l)

        Example:
            >>> torus = T4Torus()
            >>> n = WindingState(1, 0, 0, 0)
            >>> torus.fourier_mode(n, (math.pi, 0, 0, 0))
            (-1+0j)  # exp(i*pi) = -1
        """
        # Compute n.y = n_7*y_7 + n_8*y_8 + n_9*y_9 + n_10*y_10
        n_dot_y = n.n7 * y[0] + n.n8 * y[1] + n.n9 * y[2] + n.n10 * y[3]
        phase = n_dot_y / self._radius
        return cmath.exp(1j * phase)

    def fourier_expand(
        self,
        coeffs: Dict[WindingState, complex],
        y: Tuple[float, float, float, float],
    ) -> complex:
        """
        Evaluate Fourier expansion Psi(y) = sum_n Psi_hat(n) * exp(i*n.y/l).

        Args:
            coeffs: Dictionary mapping WindingState to Fourier coefficients
            y: Position on torus

        Returns:
            Complex value of the field at position y

        Example:
            >>> torus = T4Torus()
            >>> coeffs = {WindingState.zero(): 1.0}  # Constant field
            >>> torus.fourier_expand(coeffs, (0, 0, 0, 0))
            (1+0j)
        """
        result = 0j
        for n, coeff in coeffs.items():
            result += coeff * self.fourier_mode(n, y)
        return result

    def dual_lattice_vector(self, n: WindingState) -> Tuple[float, float, float, float]:
        """
        Dual lattice momentum k = 2*pi*n/l.

        In Fourier space, the momentum conjugate to position y is k = 2*pi*n/l.

        Args:
            n: Winding state

        Returns:
            Momentum 4-vector (k_7, k_8, k_9, k_10)
        """
        factor = 2 * math.pi / self._radius
        return (
            n.n7 * factor,
            n.n8 * factor,
            n.n9 * factor,
            n.n10 * factor,
        )

    def momentum_squared(self, n: WindingState) -> float:
        """
        Squared momentum |k|^2 = (2*pi/l)^2 * |n|^2.

        Args:
            n: Winding state

        Returns:
            Squared momentum
        """
        factor_sq = (2 * math.pi / self._radius) ** 2
        return factor_sq * n.norm_squared

    def enumerate_windings(self, max_norm: int = 10) -> Iterator[WindingState]:
        """
        Enumerate all winding states with |n| <= max_norm.

        Args:
            max_norm: Maximum winding norm to enumerate

        Yields:
            WindingState instances
        """
        return enumerate_windings(max_norm)

    def winding_lattice_point(
        self, n: WindingState
    ) -> Tuple[float, float, float, float]:
        """
        Map winding to position on covering space.

        The winding n corresponds to the point (n_7 * l, n_8 * l, n_9 * l, n_10 * l)
        in the covering space R^4.

        Args:
            n: Winding state

        Returns:
            Position 4-vector in covering space
        """
        return (
            n.n7 * self._radius,
            n.n8 * self._radius,
            n.n9 * self._radius,
            n.n10 * self._radius,
        )

    def inner_product_density(self) -> float:
        """
        Normalization for inner products: 1/volume.

        For orthonormal Fourier modes, <exp(in.y)|exp(im.y)> = delta_{nm}.
        The normalization factor is 1/volume.

        Returns:
            1 / (2*pi*l)^4
        """
        return 1.0 / self.volume

    def laplacian_eigenvalue(self, n: WindingState) -> float:
        """
        Eigenvalue of the Laplacian for mode n.

        The Laplacian on T^4 has eigenvalues -|k|^2 = -(2*pi/l)^2 * |n|^2.

        Args:
            n: Winding state

        Returns:
            Negative of squared momentum
        """
        return -self.momentum_squared(n)

    def heat_kernel_term(self, n: WindingState, t: float) -> float:
        """
        Single term in heat kernel expansion: exp(t * laplacian_eigenvalue).

        K(t) = sum_n exp(-t * |k|^2)

        Args:
            n: Winding state
            t: Time parameter

        Returns:
            Heat kernel contribution from mode n
        """
        return math.exp(t * self.laplacian_eigenvalue(n))

    def theta_term(self, n: WindingState, tau: complex) -> complex:
        """
        Single term in theta function: exp(i*pi*tau*|n|^2).

        Theta(tau) = sum_n exp(i*pi*tau*|n|^2)

        Args:
            n: Winding state
            tau: Complex modular parameter

        Returns:
            Theta function contribution from mode n
        """
        return cmath.exp(1j * math.pi * tau * n.norm_squared)

    def count_windings_in_shell(self, norm_squared: int) -> int:
        """
        Count winding states with exactly |n|^2 = norm_squared.

        This is related to the representation numbers of the sum of four squares.

        Args:
            norm_squared: Target squared norm

        Returns:
            Number of winding states with this squared norm
        """
        count = 0
        max_n = int(math.sqrt(norm_squared)) + 1
        for n in enumerate_windings(max_n):
            if n.norm_squared == norm_squared:
                count += 1
        return count

    def shell_windings(self, norm_squared: int) -> List[WindingState]:
        """
        Get all winding states with exactly |n|^2 = norm_squared.

        Args:
            norm_squared: Target squared norm

        Returns:
            List of WindingState with this squared norm
        """
        max_n = int(math.sqrt(norm_squared)) + 1
        return [n for n in enumerate_windings(max_n) if n.norm_squared == norm_squared]

    def __repr__(self) -> str:
        return f"T4Torus(radius={self._radius})"


def create_torus(radius: float = 1.0) -> T4Torus:
    """
    Create a T^4 torus with given radius.

    Factory function for T4Torus.

    Args:
        radius: Common radius for all four circles (default: 1.0)

    Returns:
        T4Torus instance

    Example:
        >>> torus = create_torus(1.0)
        >>> torus.dimension
        4
    """
    return T4Torus(radius=radius)
