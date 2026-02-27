"""
Theta Series - Θ₄(t) on the W⁴ torus.

The theta series encodes the partition function of winding states
on the 4-torus W⁴ with golden measure weighting.

    Θ₄(t) = Σₙ w(n) · exp(-π|n|²/t)

where w(n) = exp(-|n|²/φ) is the golden measure.

The theta series satisfies the functional equation:
    Θ₄(1/t) = t² · Θ₄(t)

Example:
    >>> from srt_library.theory_unique_components.srt.spectral import theta_series
    >>> theta = theta_series()
    >>> theta.evaluate(1.0)  # Θ₄(1)
    ...
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from srt_library.core import (
    theta_series_derivative as _theta_series_derivative,
)
from srt_library.core import (
    theta_series_evaluate as _theta_series_evaluate,
)
from srt_library.core import (
    theta_series_weighted as _theta_series_weighted,
)
from srt_library.core.exact import PHI
from srt_library.theory_unique_components.srt.geometry.winding import (
    WindingState,
    enumerate_windings_by_norm,
)
from srt_library.theory_unique_components.srt.lattice import NULL_VECTORS, QuadraticForm


class ThetaSeries:
    """
    Theta series Θ₄(t) on the 4-torus.

    The series sums over all winding states |n⟩ = |n₇, n₈, n₉, n₁₀⟩:
        Θ₄(t) = Σₙ w(n) · exp(-π|n|²/t)

    where w(n) = exp(-|n|²/φ) is the golden measure.

    The theta series provides:
    - Spectral information about the torus
    - Connection to heat kernel via Mellin transform
    - Functional equation relating t ↔ 1/t

    Attributes:
        phi: Golden ratio for the golden measure
        max_norm: Maximum |n|² to include in sums

    Example:
        >>> theta = ThetaSeries()
        >>> theta.evaluate(1.0)  # Self-dual point
        ...
        >>> lhs, rhs, err = theta.functional_equation_check(2.0)
        >>> abs(err) < 1e-6  # Functional equation holds
        True
    """

    def __init__(
        self,
        phi: Optional[float] = None,
        max_norm: int = 20,
    ):
        """
        Initialize theta series.

        Uses exact golden ratio φ from srt_library.core.exact.PHI by default.

        Args:
            phi: Golden ratio float value. If None, uses PHI.eval().
            max_norm: Maximum |n|² for winding states to include.
        """
        self._phi_exact = PHI  # Store exact golden ratio
        self._phi = phi if phi is not None else PHI.eval()
        self._q_form = QuadraticForm(self._phi)
        self._max_norm = max_norm
        # Use Rust enumerator for ~50x speedup
        by_norm = enumerate_windings_by_norm(max_norm)
        self._windings = [w for ws in by_norm.values() for w in ws]

    @property
    def phi(self) -> float:
        """Golden ratio value."""
        return self._phi

    @property
    def max_norm(self) -> int:
        """Maximum norm squared for winding states."""
        return self._max_norm

    @property
    def num_terms(self) -> int:
        """Number of winding states in the sum."""
        return len(self._windings)

    def golden_weight(self, n: WindingState) -> float:
        """Compute golden measure w(n) = exp(-|n|²/φ)."""
        return math.exp(-n.norm_squared / self._phi)

    def vigneras_rho(self, v: Tuple[float, ...], t: float) -> float:
        """
        Compute Vignéras rank-4 harmonic Maass kernel weight ρ(λ, τ).

        The kernel acts as a completion term for the indefinite theta series:
            Θ_4(t) = Σ ρ(λ, i/t) exp(-π Q(λ)/t)

        It involves a product of error functions associated with the
        four null vectors defining the Golden Cone.

        Args:
            v: 8-dimensional lattice vector λ
            t: Time parameter (y = 1/t)

        Returns:
            Kernel value ρ(λ, i/t)
        """
        if len(v) != 8:
            raise ValueError("Vector must be 8-dimensional")

        # Q(lambda) metric
        Q = self._q_form.evaluate(v)

        # Avoid division by zero for null vectors
        if abs(Q) < 1e-12:
            return 1.0

        y = 1.0 / t
        sqrt_y = math.sqrt(y)
        sqrt_abs_Q = math.sqrt(abs(Q))
        sqrt_pi = math.sqrt(math.pi)

        product = 1.0
        for c_vector in NULL_VECTORS:
            # B_a(lambda) = v . c_a (Euclidean dot product in embedding)
            b_val = sum(x * c for x, c in zip(v, c_vector))

            # Argument for error function: B_a * sqrt(y) / sqrt(|Q|)
            arg = (b_val * sqrt_y) / sqrt_abs_Q

            # E(t) = erf(sqrt(pi) * t)
            term = math.erf(sqrt_pi * arg)
            product *= term

        return product

    def evaluate(self, t: float) -> float:
        """
        Compute Θ₄(t) = Σₙ w(n) · exp(-π|n|²/t).

        Uses Rust backend for ~50x speedup.

        Args:
            t: The theta series parameter (t > 0)

        Returns:
            Value of Θ₄(t)

        Raises:
            ValueError: If t <= 0
        """
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")

        # Use Rust backend for optimized computation
        return _theta_series_evaluate(self._windings, t)

    def evaluate_without_golden(self, t: float) -> float:
        """
        Compute standard Θ₄(t) = Σₙ exp(-π|n|²/t) without golden weight.

        This is the standard Jacobi theta function on W⁴.
        Uses Rust backend for ~50x speedup.

        Args:
            t: The theta series parameter (t > 0)

        Returns:
            Value of standard Θ₄(t)
        """
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")

        # Use Rust backend with unit weights
        unit_weights = [1.0] * len(self._windings)
        return _theta_series_weighted(self._windings, unit_weights, t)

    def functional_equation_check(
        self,
        t: float,
        use_golden: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Check the functional equation Θ₄(1/t) = t² · Θ₄(t).

        For the standard theta function (without golden weight),
        the functional equation should hold exactly.

        Args:
            t: Parameter to test
            use_golden: If True, use golden-weighted theta

        Returns:
            Tuple of (Θ₄(1/t), t²·Θ₄(t), relative_error)
        """
        if use_golden:
            lhs = self.evaluate(1.0 / t)
            rhs = t * t * self.evaluate(t)
        else:
            lhs = self.evaluate_without_golden(1.0 / t)
            rhs = t * t * self.evaluate_without_golden(t)

        error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-10)
        return (lhs, rhs, error)

    def derivative(self, t: float, order: int = 1) -> float:
        """
        Compute derivative dⁿΘ₄/dtⁿ.

        Uses Rust backend for first derivative, numerical methods for higher orders.

        Args:
            t: Point to evaluate derivative
            order: Order of derivative (1, 2, ...)

        Returns:
            Derivative value
        """
        if order == 1:
            # Use Rust backend for analytical first derivative
            return _theta_series_derivative(self._windings, t)
        elif order == 2:
            # Use finite differences for second derivative
            h = 1e-5 * t
            return (
                self.evaluate(t + h) - 2 * self.evaluate(t) + self.evaluate(t - h)
            ) / (h * h)
        else:
            # Higher orders via recursion
            h = 1e-5 * t
            return (
                self.derivative(t + h, order - 1) - self.derivative(t - h, order - 1)
            ) / (2 * h)

    def mellin_integrand(self, t: float, s: complex) -> complex:
        """
        Compute the Mellin transform integrand t^(s-1) · Θ₄(t).

        The Mellin transform connects theta series to zeta functions.

        Args:
            t: Integration variable
            s: Complex parameter

        Returns:
            Complex value of integrand
        """
        theta_val = self.evaluate(t)
        return (t ** (s - 1)) * theta_val

    def small_t_expansion(self, t: float, terms: int = 3) -> float:
        """
        Compute small-t asymptotic expansion.

        As t → 0+, Θ₄(t) ~ t² · (constant + O(t)).

        Args:
            t: Small parameter
            terms: Number of terms to include

        Returns:
            Asymptotic approximation
        """
        # Use functional equation: Θ₄(t) = (1/t²) · Θ₄(1/t)
        # For small t, 1/t is large, so Θ₄(1/t) approaches 1
        return self.evaluate_without_golden(1.0 / t) / (t * t)

    def partition_function(self, beta: float) -> float:
        """
        Compute partition function Z(β) = Θ₄(1/β).

        In statistical mechanics, β = 1/kT is inverse temperature.

        Args:
            beta: Inverse temperature

        Returns:
            Partition function value
        """
        return self.evaluate(1.0 / beta)

    def __repr__(self) -> str:
        return f"ThetaSeries(phi={self._phi:.6f}, max_norm={self._max_norm}, terms={self.num_terms})"


def theta_series(
    phi: Optional[float] = None,
    max_norm: int = 20,
) -> ThetaSeries:
    """
    Create a ThetaSeries instance.

    Factory function for ThetaSeries.

    Args:
        phi: Golden ratio value. If None, uses PHI_NUMERIC.
        max_norm: Maximum |n|² for winding states.

    Returns:
        ThetaSeries instance

    Example:
        >>> theta = theta_series()
        >>> theta.evaluate(1.0)
        ...
    """
    return ThetaSeries(phi=phi, max_norm=max_norm)
