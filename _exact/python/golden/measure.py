"""
Golden Measure - The fundamental measure w(n) = exp(-|n|^2/phi) on winding lattice.

The golden measure weights winding configurations according to their
squared norm, with phi as the characteristic scale. This measure
appears in all SRT spectral sums.

Uses exact golden ratio arithmetic from syntonic_applications.exact.GoldenExact.

Example:
    >>> from syntonic_applications.srt.golden import golden_measure, golden_weight
    >>> from syntonic_applications.srt.geometry import winding_state
    >>> n = winding_state(1, 1, 0, 0)
    >>> golden_weight(n)
    0.289...  # exp(-2/phi)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, Optional

from applications.core import (
    partition_function as _partition_function,
)
from applications.core import (
    theta_sum_combined as _theta_sum_combined,
)
from applications.exact import PHI, GoldenExact
from applications.srt.geometry.winding import WindingState, enumerate_windings

if TYPE_CHECKING:
    pass


class GoldenMeasure:
    """
    Golden measure w(n) = exp(-|n|^2 / phi) on winding lattice.

    This is the fundamental measure in SRT that weights winding
    configurations. It appears in partition functions, spectral
    sums, and defines the natural metric on winding space.

    Uses exact golden ratio arithmetic from syntonic_applications.exact.GoldenExact.

    Properties:
        - w(0) = 1 (vacuum has unit weight)
        - w(n) -> 0 as |n| -> infinity (convergence)
        - Sum_n w(n) converges (partition function finite)

    Attributes:
        phi: Golden ratio float value (from exact PHI.eval())
        phi_exact: Exact GoldenExact representation of φ

    Example:
        >>> measure = GoldenMeasure()
        >>> n = WindingState(1, 0, 0, 0)
        >>> measure.weight(n)
        0.538...  # exp(-1/phi)
    """

    def __init__(self, phi: Optional[float] = None):
        """
        Initialize golden measure.

        Uses exact golden ratio φ from syntonic_applications.exact.PHI by default.

        Args:
            phi: Golden ratio float value. If None, uses PHI.eval().
        """
        self._phi_exact = PHI  # Store exact golden ratio
        self._phi = phi if phi is not None else PHI.eval()

    @property
    def phi(self) -> float:
        """Golden ratio float value used for the measure."""
        return self._phi

    @property
    def phi_exact(self) -> GoldenExact:
        """Exact GoldenExact representation of the golden ratio."""
        return self._phi_exact

    def weight(self, n: WindingState) -> float:
        """
        Compute w(n) = exp(-|n|^2 / phi).

        Args:
            n: Winding state

        Returns:
            Golden weight in (0, 1]
        """
        return math.exp(-n.norm_squared / self._phi)

    def log_weight(self, n: WindingState) -> float:
        """
        Compute log(w(n)) = -|n|^2 / phi.

        Useful for numerical stability in large sums.

        Args:
            n: Winding state

        Returns:
            Log of the golden weight
        """
        return -n.norm_squared / self._phi

    def weighted_sum(
        self,
        values: Dict[WindingState, float],
    ) -> float:
        """
        Compute sum_n w(n) * f(n).

        Args:
            values: Dictionary mapping WindingState to function values

        Returns:
            Weighted sum
        """
        return sum(self.weight(n) * v for n, v in values.items())

    def weighted_sum_iter(
        self,
        func: Callable[[WindingState], float],
        max_norm: int = 10,
    ) -> float:
        """
        Compute sum_n w(n) * f(n) over lattice.

        Args:
            func: Function to evaluate at each winding
            max_norm: Maximum winding norm to include

        Returns:
            Weighted sum
        """
        total = 0  # Use exact integer starting value
        for n in enumerate_windings(max_norm):
            total += self.weight(n) * func(n)
        return total

    def partition_function(self, max_norm: int = 10) -> float:
        """
        Compute Z = sum_n w(n) over lattice.

        This is the normalization constant for the golden measure.
        Uses Rust backend for ~50x speedup.

        Args:
            max_norm: Maximum winding norm to include

        Returns:
            Partition function value
        """
        # Use Rust backend for optimized computation
        windings = list(enumerate_windings(max_norm))
        return _partition_function(windings)

    def normalize(
        self,
        values: Dict[WindingState, float],
        max_norm: int = 10,
    ) -> Dict[WindingState, float]:
        """
        Normalize values by partition function.

        Args:
            values: Dictionary of function values
            max_norm: Maximum norm for partition function

        Returns:
            Normalized dictionary: values / Z
        """
        Z = self.partition_function(max_norm)
        return {n: v / Z for n, v in values.items()}

    def expectation(
        self,
        func: Callable[[WindingState], float],
        max_norm: int = 10,
    ) -> float:
        """
        Compute <f> = (sum_n w(n) * f(n)) / Z.

        Args:
            func: Function to average
            max_norm: Maximum winding norm

        Returns:
            Expectation value
        """
        Z = self.partition_function(max_norm)
        weighted = self.weighted_sum_iter(func, max_norm)
        return weighted / Z

    def variance(
        self,
        func: Callable[[WindingState], float],
        max_norm: int = 10,
    ) -> float:
        """
        Compute variance Var(f) = <f^2> - <f>^2.

        Args:
            func: Function to compute variance of
            max_norm: Maximum winding norm

        Returns:
            Variance
        """
        mean = self.expectation(func, max_norm)
        mean_sq = self.expectation(lambda n: func(n) ** 2, max_norm)
        return mean_sq - mean**2

    def theta_sum(
        self,
        t: float,
        max_norm: int = 10,
    ) -> float:
        """
        Compute theta sum: sum_n w(n) * exp(-pi * |n|^2 / t).

        This combines the golden measure with thermal weighting.
        Uses Rust backend for ~50x speedup.

        Args:
            t: Temperature-like parameter
            max_norm: Maximum winding norm

        Returns:
            Theta sum value
        """
        # Use Rust backend for optimized combined computation
        # Rust computes: sum_n exp(-|n|^2 * (1/phi + pi/t))
        windings = list(enumerate_windings(max_norm))
        return _theta_sum_combined(windings, t)

    def effective_temperature(self) -> float:
        """
        Effective temperature T_eff = phi / 2.

        The golden measure corresponds to a thermal state at this temperature.

        Returns:
            Effective temperature
        """
        return self._phi / 2

    def __repr__(self) -> str:
        return f"GoldenMeasure(phi={self._phi:.6f})"


def golden_measure(phi: Optional[float] = None) -> GoldenMeasure:
    """
    Create a GoldenMeasure instance.

    Factory function for GoldenMeasure.

    Args:
        phi: Golden ratio value. If None, uses PHI_NUMERIC.

    Returns:
        GoldenMeasure instance
    """
    return GoldenMeasure(phi=phi)


def golden_weight(n: WindingState, phi: Optional[float] = None) -> float:
    """
    Compute golden weight w(n) = exp(-|n|^2/phi).

    Convenience function for single weight computation.
    Uses exact golden ratio φ from syntonic_applications.exact.PHI by default.

    Args:
        n: Winding state
        phi: Golden ratio float value. If None, uses PHI.eval().

    Returns:
        Golden weight
    """
    if phi is None:
        phi = PHI.eval()  # Use exact golden ratio
    return math.exp(-n.norm_squared / phi)


def compute_partition_function(
    max_norm: int = 10, phi: Optional[float] = None
) -> float:
    """
    Compute partition function Z = sum_n exp(-|n|^2/phi).

    Convenience function for partition function computation.
    Uses Rust backend for ~50x speedup.
    Uses exact golden ratio φ from syntonic_applications.exact.PHI by default.

    Args:
        max_norm: Maximum winding norm
        phi: Golden ratio float value. If None, uses PHI.eval().

    Returns:
        Partition function value
    """
    # Use Rust backend directly
    windings = list(enumerate_windings(max_norm))
    return _partition_function(windings)
