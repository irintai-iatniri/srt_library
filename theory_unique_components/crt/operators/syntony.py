"""
Syntony Index S(Ψ) computation.

The syntony index measures the balance between differentiation and
harmonization in a state:

S(Ψ) = 1 - ||D̂[Ψ] - Ĥ[D̂[Ψ]]|| / (||D̂[Ψ] - Ψ|| + ε)

Properties:
- S ∈ [0, 1]
- S = 1: Perfect harmony (D̂ = Ĥ ∘ D̂, no net differentiation)
- S = 0: Maximum differentiation (D̂ creates change that Ĥ cannot undo)
- Golden balance: D + H → 0.382 + 0.618 = 1

This module provides:
- SyntonyComputer: Full computation using operator pair
- Quick variants for estimation without full operator application
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

from srt_library.core.exact import GoldenExact

if TYPE_CHECKING:
    from srt_library.core.state import State
    from srt_library.theory_unique_components.crt.operators.differentiation import DifferentiationOperator
    from srt_library.theory_unique_components.crt.operators.harmonization import HarmonizationOperator


class SyntonyComputer:
    """
    Computes syntony index S(Ψ) using the full operator formulation.

    S(Ψ) = 1 - ||D̂[Ψ] - Ĥ[D̂[Ψ]]|| / (||D̂[Ψ] - Ψ|| + ε)

    This measures how well harmonization "undoes" differentiation.
    When D̂ and Ĥ are in balance, S approaches the golden ratio.

    Example:
        >>> from srt_library.theory_unique_components.crt.operators import DifferentiationOperator, HarmonizationOperator
        >>> D_op = DifferentiationOperator()
        >>> H_op = HarmonizationOperator()
        >>> computer = SyntonyComputer(D_op, H_op)
        >>> S = computer.compute(state)
    """

    def __init__(
        self,
        diff_op: "DifferentiationOperator",
        harm_op: "HarmonizationOperator",
        epsilon: Union[float, GoldenExact] = None,
    ):
        """
        Create a syntony computer.

        Args:
            diff_op: Differentiation operator D̂
            harm_op: Harmonization operator Ĥ
            epsilon: Small constant to avoid division by zero (default: exact small value)
        """
        self.diff_op = diff_op
        self.harm_op = harm_op
        
        # Default epsilon to exact small value
        if epsilon is None:
            epsilon = GoldenExact.nearest(1e-10, 1 << 30)
        elif isinstance(epsilon, float):
            epsilon = GoldenExact.nearest(epsilon, 1 << 30)
        
        self.epsilon = epsilon

    def compute(self, state: "State") -> GoldenExact:
        """
        Compute syntony index S(Ψ).

        S(Ψ) = 1 - ||D̂[Ψ] - Ĥ[D̂[Ψ]]|| / (||D̂[Ψ] - Ψ|| + ε)

        Args:
            state: Input state Ψ

        Returns:
            Syntony index in [0, 1] (exact GoldenExact)
        """
        # Apply D̂[Ψ]
        d_psi = self.diff_op.apply(state)

        # Apply Ĥ[D̂[Ψ]]
        hd_psi = self.harm_op.apply(d_psi)

        # Compute norms as GoldenExact approximations
        numerator_norm = (d_psi - hd_psi).norm()
        numerator = GoldenExact.nearest(numerator_norm, 1 << 30)

        denominator_norm = (d_psi - state).norm()
        denominator_base = GoldenExact.nearest(denominator_norm, 1 << 30)
        denominator = denominator_base + self.epsilon

        # S = 1 - numerator / denominator
        ratio = numerator / denominator
        one = GoldenExact.from_integers(1, 0)
        S = one - ratio

        # For now, return S without clamping (exact arithmetic)
        return S

    def compute_components(self, state: "State") -> dict:
        """
        Compute syntony with detailed components.

        Returns dict with:
        - syntony: The S(Ψ) value
        - diff_magnitude: ||D̂[Ψ] - Ψ||
        - residual: ||D̂[Ψ] - Ĥ[D̂[Ψ]]||
        - d_state: D̂[Ψ]
        - hd_state: Ĥ[D̂[Ψ]]
        """
        # Apply operators
        d_psi = self.diff_op.apply(state)
        hd_psi = self.harm_op.apply(d_psi)

        # Compute magnitudes
        diff_magnitude = (d_psi - state).norm()
        residual = (d_psi - hd_psi).norm()

        # Compute syntony
        S = 1.0 - residual / (diff_magnitude + self.epsilon)
        S = max(0.0, min(1.0, S))

        return {
            "syntony": S,
            "diff_magnitude": diff_magnitude,
            "residual": residual,
            "d_state": d_psi,
            "hd_state": hd_psi,
        }

    def __repr__(self) -> str:
        return f"SyntonyComputer(eps={self.epsilon})"


def syntony_entropy(state: "State") -> float:
    """
    Estimate syntony from entropy of the state.

    Uses normalized entropy as a proxy for syntony:
    - Low entropy → high syntony (ordered, harmonized)
    - High entropy → low syntony (disordered, differentiated)

    This is a quick estimate that doesn't require applying D̂/Ĥ.

    Args:
        state: Input state

    Returns:
        Syntony estimate in [0, 1]
    """
    flat = state.to_list()
    N = len(flat)

    if N < 2:
        return 1.0

    # Compute probability distribution from magnitudes
    magnitudes = [abs(x) for x in flat]
    total = sum(magnitudes)

    if total < 1e-12:
        return 0.5  # Zero state is neutral

    probs = [m / total for m in magnitudes]

    # Compute entropy: H = -Σ p log(p)
    entropy = 0.0
    for p in probs:
        if p > 1e-12:
            entropy -= p * math.log(p)

    # Normalize by max entropy (log N)
    max_entropy = math.log(N)
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0

    # Syntony is inverse of normalized entropy
    # High entropy → low syntony
    return 1.0 - normalized_entropy


def syntony_spectral(state: "State") -> float:
    """
    Estimate syntony from spectral concentration.

    Measures how much energy is concentrated in low frequencies:
    - High concentration in low freq → high syntony (smooth, coherent)
    - Spread across frequencies → low syntony (noisy, differentiated)

    Args:
        state: Input state

    Returns:
        Syntony estimate in [0, 1]
    """
    import cmath

    flat = state.to_list()
    N = len(flat)

    if N < 4:
        return 0.5

    # Convert to complex
    if not isinstance(flat[0], complex):
        flat = [complex(x) for x in flat]

    # Compute DFT
    omega = cmath.exp(-2j * cmath.pi / N)
    freq = []
    for k in range(N):
        s = 0j
        for n in range(N):
            s += flat[n] * (omega ** (k * n))
        freq.append(abs(s) ** 2)  # Power spectrum

    total_power = sum(freq)
    if total_power < 1e-12:
        return 0.5

    # Low-frequency power (first quarter of spectrum)
    low_freq_cutoff = max(1, N // 4)
    low_freq_power = sum(freq[:low_freq_cutoff])

    # Syntony proportional to low-frequency concentration
    return low_freq_power / total_power


def syntony_quick(state: "State") -> float:
    """
    Quick syntony estimate using total variation.

    Total variation measures "roughness":
    - Low TV → high syntony (smooth)
    - High TV → low syntony (rough)

    This is the fastest estimate, using only O(N) operations.

    Args:
        state: Input state

    Returns:
        Syntony estimate in [0, 1]
    """
    flat = state.to_list()
    N = len(flat)

    if N < 2:
        return 1.0

    # Compute total variation: Σ |Ψᵢ₊₁ - Ψᵢ|
    tv = 0.0
    for i in range(N - 1):
        tv += abs(flat[i + 1] - flat[i])

    # Normalize by total magnitude
    total_mag = sum(abs(x) for x in flat)

    if total_mag < 1e-12:
        return 0.5

    normalized_tv = tv / total_mag

    # Map TV to syntony (sigmoid-like transformation)
    # S = 1 / (1 + TV)
    return 1.0 / (1.0 + normalized_tv)
