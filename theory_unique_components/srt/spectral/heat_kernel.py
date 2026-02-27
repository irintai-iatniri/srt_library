"""
Heat Kernel - K(t) = Tr[exp(-t·L²)] on the golden lattice.

The heat kernel trace encodes spectral information about the
Laplacian operator on winding configurations. In SRT, this connects
to the knot Laplacian and determines particle masses.

    K(t) = Σₙ w(n) · exp(-t·λₙ)

where λₙ are eigenvalues of L² and w(n) is the golden measure.

Example:
    >>> from srt_library.theory_unique_components.srt.spectral import heat_kernel
    >>> K = heat_kernel()
    >>> K.evaluate(1.0)
    ...
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from srt_library.core import (
    heat_kernel_trace as _heat_kernel_trace,
)
from srt_library.core import (
    heat_kernel_weighted as _heat_kernel_weighted,
)
from srt_library.core import (
    spectral_zeta_weighted as _spectral_zeta_weighted,
)
from srt_library.core.exact import PHI
from srt_library.theory_unique_components.srt.geometry.winding import (
    WindingState,
    enumerate_windings_by_norm,
    winding_state,
)

# Base eigenvalue for standard Laplacian on W⁴: λₙ = 4π²|n|²
_BASE_EIGENVALUE = 4 * math.pi * math.pi


class HeatKernel:
    """
    Heat kernel trace K(t) = Tr[exp(-t·L²)] on the golden lattice.

    The heat kernel provides:
    - Short-time asymptotics → local geometry
    - Long-time asymptotics → topology (zero modes)
    - Spectral zeta function via Mellin transform

    The Laplacian eigenvalues on W⁴ are λₙ = 4π²|n|² for winding
    state |n⟩. With golden weighting, the trace becomes:

        K(t) = Σₙ w(n) · exp(-4π²t|n|²)

    Attributes:
        phi: Golden ratio for measure
        max_norm: Maximum |n|² in sums

    Example:
        >>> K = HeatKernel()
        >>> K.evaluate(0.01)  # Short time
        ...
        >>> K.evaluate(10.0)  # Long time
        ...
    """

    def __init__(
        self,
        phi: Optional[float] = None,
        max_norm: int = 20,
    ):
        """
        Initialize heat kernel.

        Uses exact golden ratio φ from srt_library.core.exact.PHI by default.

        Args:
            phi: Golden ratio float value. If None, uses PHI.eval().
            max_norm: Maximum |n|² for winding states.
        """
        self._phi_exact = PHI  # Store exact golden ratio
        self._phi = phi if phi is not None else PHI.eval()
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

    def golden_weight(self, n: WindingState) -> float:
        """Compute golden measure w(n) = exp(-|n|²/φ)."""
        return math.exp(-n.norm_squared / self._phi)

    def eigenvalue(self, n: WindingState) -> float:
        """
        Compute Laplacian eigenvalue λₙ = 4π²|n|² for winding state.

        Args:
            n: Winding state

        Returns:
            Eigenvalue λₙ
        """
        return 4 * math.pi * math.pi * n.norm_squared

    def evaluate(self, t: float) -> float:
        """
        Compute K(t) = Tr[exp(-t·L²)] = Σₙ w(n)·exp(-t·λₙ).

        Uses Rust backend for ~50x speedup.

        Args:
            t: Time parameter (t > 0)

        Returns:
            Heat kernel trace value

        Raises:
            ValueError: If t <= 0
        """
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")

        # Use Rust backend for optimized computation
        return _heat_kernel_weighted(self._windings, t, _BASE_EIGENVALUE)

    def evaluate_without_golden(self, t: float) -> float:
        """
        Compute standard heat kernel K(t) = Σₙ exp(-t·λₙ) without golden weight.

        Uses Rust backend for ~50x speedup.

        Args:
            t: Time parameter (t > 0)

        Returns:
            Standard heat kernel trace
        """
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")

        # Use Rust backend for optimized computation
        return _heat_kernel_trace(self._windings, t, _BASE_EIGENVALUE)

    def short_time_expansion(self, t: float, terms: int = 3) -> float:
        """
        Compute short-time asymptotic expansion.

        As t → 0+:
            K(t) ~ (4πt)^(-d/2) · [a₀ + a₁·t + a₂·t² + ...]

        where d = 4 is the dimension and aₖ are Seeley-DeWitt coefficients.

        Args:
            t: Small time parameter
            terms: Number of terms in expansion

        Returns:
            Asymptotic approximation
        """
        d = 4  # Dimension of W⁴
        prefactor = (4 * math.pi * t) ** (-d / 2)

        # For flat torus, a₀ = vol(W⁴) / (4π)^(d/2)
        # We normalize so a₀ = 1 at t = 0
        # Higher coefficients depend on curvature (zero for flat torus)
        a0 = 1.0
        a1 = 0.0  # No curvature term for flat torus
        a2 = 0.0

        coeffs = [a0, a1, a2][:terms]
        expansion = sum(c * (t**k) for k, c in enumerate(coeffs))

        return prefactor * expansion

    def long_time_limit(self) -> float:
        """
        Compute K(t) as t → ∞.

        Only zero modes contribute: K(∞) = w(0) where |0⟩ is the trivial winding.

        Returns:
            Long-time limit (= golden weight of vacuum)
        """
        # Only n = (0,0,0,0) contributes as t → ∞
        return self.golden_weight(winding_state(0, 0, 0, 0))

    def spectral_zeta(self, s: complex, regularize: bool = True) -> complex:
        """
        Compute spectral zeta function ζ_L(s) = Σₙ w(n)/λₙˢ.

        The spectral zeta is related to heat kernel via Mellin transform:
            ζ_L(s) = (1/Γ(s)) ∫₀^∞ t^(s-1) K(t) dt

        Uses Rust backend for real s values (~50x speedup).

        Args:
            s: Complex parameter
            regularize: If True, exclude zero mode

        Returns:
            Spectral zeta value

        Note:
            Convergent for Re(s) > d/2 = 2
        """
        # For real s, use optimized Rust backend
        if isinstance(s, (int, float)) or (isinstance(s, complex) and s.imag == 0):
            s_real = float(s.real if isinstance(s, complex) else s)
            # Rust spectral_zeta_weighted automatically excludes zero mode
            result = _spectral_zeta_weighted(self._windings, s_real, _BASE_EIGENVALUE)
            return complex(result, 0.0)

        # For complex s, use Python implementation
        total = complex(0.0, 0.0)

        for n in self._windings:
            lambda_n = self.eigenvalue(n)
            if lambda_n == 0:
                if regularize:
                    continue  # Skip zero mode
                else:
                    return complex(float("inf"), 0.0)

            w = self.golden_weight(n)
            total += w * (lambda_n ** (-s))

        return total

    def spectral_determinant(self) -> float:
        """
        Compute regularized spectral determinant det'(L²).

        Using zeta regularization:
            log det'(L²) = -ζ'_L(0)

        Returns:
            Regularized determinant (positive)
        """
        # Numerical derivative of zeta at s = 0
        h = 1e-6
        zeta_plus = self.spectral_zeta(complex(h, 0), regularize=True)
        zeta_minus = self.spectral_zeta(complex(-h, 0), regularize=True)
        zeta_prime = (zeta_plus - zeta_minus) / (2 * h)

        return math.exp(-zeta_prime.real)

    def integrated_heat_kernel(
        self, t_min: float, t_max: float, steps: int = 100
    ) -> float:
        """
        Compute ∫[t_min, t_max] K(t) dt numerically.

        Args:
            t_min: Lower integration bound
            t_max: Upper integration bound
            steps: Number of integration steps

        Returns:
            Integral value
        """
        dt = (t_max - t_min) / steps
        total = 0.0

        for i in range(steps):
            t = t_min + (i + 0.5) * dt  # Midpoint rule
            total += self.evaluate(t) * dt

        return total

    def eigenvalue_density(
        self, lambda_max: float, bins: int = 50
    ) -> List[Tuple[float, int]]:
        """
        Compute eigenvalue density histogram.

        Args:
            lambda_max: Maximum eigenvalue to include
            bins: Number of histogram bins

        Returns:
            List of (bin_center, count) tuples
        """
        bin_width = lambda_max / bins
        counts = [0] * bins

        for n in self._windings:
            lambda_n = self.eigenvalue(n)
            if lambda_n < lambda_max:
                bin_idx = min(int(lambda_n / bin_width), bins - 1)
                counts[bin_idx] += 1

        return [(bin_width * (i + 0.5), counts[i]) for i in range(bins)]

    def weyl_law_check(self, lambda_max: float) -> Tuple[int, float, float]:
        """
        Check Weyl's law: N(λ) ~ C·λ^(d/2) as λ → ∞.

        For d = 4: N(λ) ~ (vol / 16π²)·λ²

        Args:
            lambda_max: Maximum eigenvalue

        Returns:
            Tuple of (count, expected, relative_error)
        """
        count = sum(
            1
            for n in self._windings
            if self.eigenvalue(n) <= lambda_max and self.eigenvalue(n) > 0
        )

        # Weyl coefficient for unit 4-torus
        d = 4
        weyl_coeff = 1.0 / (16 * math.pi * math.pi)
        expected = weyl_coeff * (lambda_max ** (d / 2))

        error = abs(count - expected) / max(expected, 1)
        return (count, expected, error)

    def __repr__(self) -> str:
        return f"HeatKernel(phi={self._phi:.6f}, max_norm={self._max_norm})"


def heat_kernel(
    phi: Optional[float] = None,
    max_norm: int = 20,
) -> HeatKernel:
    """
    Create a HeatKernel instance.

    Factory function for HeatKernel.

    Args:
        phi: Golden ratio value. If None, uses PHI_NUMERIC.
        max_norm: Maximum |n|² for winding states.

    Returns:
        HeatKernel instance

    Example:
        >>> K = heat_kernel()
        >>> K.evaluate(1.0)
        ...
    """
    return HeatKernel(phi=phi, max_norm=max_norm)
