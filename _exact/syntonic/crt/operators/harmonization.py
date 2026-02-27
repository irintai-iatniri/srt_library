"""
Harmonization Operator Ĥ for CRT.

Implements the full harmonization formula:
Ĥ[Ψ] = Ψ - Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ] + γ(S) Ŝ_op[Ψ] + Δ_NL[Ψ]

Where:
- Q̂ᵢ: High-frequency damping projectors
- βᵢ(S,Δ_D) = β₀ × S × decay_factor  (syntony-dependent damping)
- γ(S): Syntony projection strength
- Ŝ_op: Syntony-promoting operator (projects toward golden measure)
- Δ_NL: Nonlinear correction term
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from syntonic.exact import GoldenExact, Rational, PHI_INVERSE
from syntonic.crt.operators.base import OperatorBase
from syntonic.crt.operators.projectors import (
    DampingProjector,
    create_damping_cascade,
)

if TYPE_CHECKING:
    from syntonic.core.state import State

# Use exact golden inverse from syntonic.exact (for exact-first SGC)
PHI_INV_EXACT = PHI_INVERSE


class HarmonizationOperator(OperatorBase):
    """
    Harmonization Operator Ĥ.

    Decreases complexity/differentiation by:
    1. Damping high-frequency components (via Q̂ᵢ projectors)
    2. Projecting toward a syntony-promoting target
    3. Applying nonlinear corrections

    The damping strengths are syntony-dependent:
    - High syntony (S ≈ 1): Strong harmonization (maintain equilibrium)
    - Low syntony (S ≈ 0): Weak harmonization (allow differentiation)

    This creates complementary dynamics with D̂:
    D + H = S → 0.382 + 0.618 = 1 (golden partition)

    Example:
        >>> H_op = HarmonizationOperator(beta_0=0.618)
        >>> evolved = H_op.apply(state)
    """

    def __init__(
        self,
        beta_0: Union[float, GoldenExact] = None,  # Defaults to 1/φ (exact)
        gamma_0: Union[float, GoldenExact] = 0.1,
        num_dampers: int = 3,
        damping_projectors: Optional[List[DampingProjector]] = None,
        target_generator: Optional[Callable[["State"], "State"]] = None,
        nonlinear_strength: Union[float, GoldenExact] = 0.01,
        exact: bool = True,  # NEW: Default to exact arithmetic for SGC
    ):
        """
        Create a harmonization operator.

        Args:
            beta_0: Base damping strength (default: 1/φ ≈ 0.618, exact)
            gamma_0: Syntony projection strength
            num_dampers: Number of damping levels (if damping_projectors not given)
            damping_projectors: Custom damping projectors (optional)
            target_generator: Function to generate target state (optional)
            nonlinear_strength: Strength of nonlinear correction term
            exact: Use exact arithmetic (default True for SGC zero-entropy)
        """
        self.exact = exact

        # Default beta_0 to exact 1/φ if not provided
        if beta_0 is None:
            beta_0 = PHI_INV_EXACT if exact else PHI_INV

        # Coerce to GoldenExact for exact mode (like differentiation does)
        def to_golden(x):
            if isinstance(x, GoldenExact):
                return x
            if isinstance(x, Rational):
                # Convert Rational to float for nearest approximation in Q(φ)
                # Use high precision (2^30) for exact-like representation
                val = x.numerator / x.denominator
                return GoldenExact.nearest(val, 1 << 30)
            # For float input when exact=True, approximate
            if exact:
                return GoldenExact.nearest(float(x), 1 << 30)
            return float(x)

        if exact:
            self.beta_0 = to_golden(beta_0)
            self.gamma_0 = to_golden(gamma_0)
            self.nonlinear_strength = to_golden(nonlinear_strength)
        else:
            self.beta_0 = float(beta_0) if not isinstance(beta_0, GoldenExact) else beta_0.eval()
            self.gamma_0 = float(gamma_0) if not isinstance(gamma_0, GoldenExact) else gamma_0.eval()
            self.nonlinear_strength = float(nonlinear_strength) if not isinstance(nonlinear_strength, GoldenExact) else nonlinear_strength.eval()

        self.num_dampers = num_dampers

        # Damping projectors (cascade with golden decay)
        if damping_projectors is not None:
            self._dampers = damping_projectors
        else:
            self._dampers = create_damping_cascade(
                num_levels=num_dampers,
                base_cutoff=0.7,
                decay=PHI_INV,
            )

        # Target generator for syntony projection
        self._target_generator = target_generator or self._default_target

    @property
    def dampers(self) -> List[DampingProjector]:
        """Get damping projectors."""
        return self._dampers

    @staticmethod
    def _default_target(state: "State") -> "State":
        """
        Default target generator: normalized mean projection.

        This projects toward a uniform state with the same total magnitude,
        representing the "most harmonized" configuration.
        """
        from syntonic.core.state import State

        flat = state.to_list()
        N = len(flat)

        # Compute mean value
        if isinstance(flat[0], complex):
            mean_val = sum(flat) / N
        else:
            mean_val = sum(flat) / N

        # Target is uniform state with mean value
        target_flat = [mean_val] * N

        return State(
            target_flat, dtype=state.dtype, device=state.device, shape=state.shape
        )

    def _decay_factor(self, level: int, delta_d: Optional[float]) -> float:
        """
        Compute decay factor for damping level.

        Higher levels have stronger decay, modulated by differentiation magnitude.

        Args:
            level: Damping level index
            delta_d: Differentiation magnitude (optional)

        Returns:
            Decay factor in [0, 1]
        """
        # Base decay follows golden ratio - use exact type
        # EXACT→FLOAT BOUNDARY: Result is used for tensor scalar multiplication
        phi_inv_float = PHI_INV_EXACT.eval()
        base_decay = phi_inv_float ** level

        # Modulate by differentiation magnitude if provided
        if delta_d is not None:
            # Stronger damping when more differentiated
            modulation = 1.0 + delta_d
        else:
            modulation = 1.0

        return base_decay * min(modulation, 2.0)  # Cap at 2x

    def _syntony_projection(self, state: "State", S) -> "State":
        """
        Compute syntony-promoting projection Ŝ_op[Ψ].

        Projects state toward target, weighted by syntony.

        Args:
            state: Input state
            S: Current syntony (GoldenExact in exact mode, float otherwise)

        Returns:
            Syntony projection contribution
        """
        target = self._target_generator(state)

        # Direction toward target
        direction = target - state

        # Weight by γ(S) = γ₀ × S (exact arithmetic if in exact mode)
        gamma = self.gamma_0 * S

        # gamma may be GoldenExact or float - State.__mul__ handles both
        # GoldenExact dispatches to mul_scalar_golden for exact arithmetic
        return direction * gamma

    def _nonlinear_correction(self, state: "State", S) -> "State":
        """
        Compute nonlinear correction Δ_NL[Ψ].

        Adds small corrections based on local curvature.

        Args:
            state: Input state
            S: Current syntony (GoldenExact in exact mode, float otherwise)

        Returns:
            Nonlinear correction term
        """
        from syntonic.core.state import State

        # EXACT→FLOAT BOUNDARY: Need float for comparison and loop operations
        nl_strength = self.nonlinear_strength.eval() if hasattr(self.nonlinear_strength, 'eval') else float(self.nonlinear_strength)
        S_float = S.eval() if hasattr(S, 'eval') else float(S)

        if nl_strength < 1e-12:
            # Return zero state
            flat = [0.0] * state.size
            return State(
                flat, dtype=state.dtype, device=state.device, shape=state.shape
            )

        flat = state.to_list()
        N = len(flat)

        # Simple nonlinear correction: cubic damping
        # Δ_NL[Ψ]ᵢ = -η × Ψᵢ³ / (1 + |Ψᵢ|²)
        correction = []
        for i in range(N):
            x = flat[i]
            if isinstance(x, complex):
                x_abs_sq = abs(x) ** 2
                nl = -nl_strength * x * x_abs_sq / (1 + x_abs_sq)
            else:
                x_sq = x * x
                nl = -nl_strength * x * x_sq / (1 + x_sq)
            correction.append(nl)

        # Scale by syntony (stronger correction at higher syntony)
        correction = [c * S_float for c in correction]

        return State(
            correction, dtype=state.dtype, device=state.device, shape=state.shape
        )

    def apply(
        self,
        state: "State",
        syntony: Optional[Union[float, GoldenExact]] = None,
        delta_d: Optional[float] = None,
        **kwargs,
    ) -> "State":
        """
        Apply harmonization operator Ĥ.

        Ĥ[Ψ] = Ψ - Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ] + γ(S) Ŝ_op[Ψ] + Δ_NL[Ψ]

        Args:
            state: Input state Ψ
            syntony: Current syntony S (if None, computed from state)
            delta_d: Differentiation magnitude (optional, for adaptive damping)

        Returns:
            Harmonized state Ĥ[Ψ]
        """
        # Get syntony and convert to exact if needed
        S_val = syntony if syntony is not None else state.syntony

        if self.exact:
            # Convert syntony to GoldenExact (like differentiation does)
            if isinstance(S_val, GoldenExact):
                S = S_val
            elif isinstance(S_val, Rational):
                S = GoldenExact.from_rational(S_val)
            else:
                S = GoldenExact.nearest(float(S_val), 1 << 30)
        else:
            S = float(S_val) if not isinstance(S_val, GoldenExact) else S_val.eval()

        # Start with identity: result = Ψ
        result = state

        # Subtract damping contributions: -Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ]
        for i, damper in enumerate(self._dampers):
            # βᵢ(S,Δ_D) = β₀ × S × decay_factor
            decay = self._decay_factor(i, delta_d)

            if self.exact:
                # Exact arithmetic - keep everything in GoldenExact
                if isinstance(decay, float):
                    decay_exact = GoldenExact.nearest(decay, 1 << 30)
                else:
                    decay_exact = decay
                beta_i = self.beta_0 * S * decay_exact
                
                # Use GoldenExact scalar directly (State.__mul__ dispatches to mul_scalar_golden)
                if not beta_i.is_zero():
                    damped = damper.project(state)
                    high_freq = state - damped
                    result = result - high_freq * beta_i
            else:
                # Float arithmetic path
                beta_i_scalar = self.beta_0 * S * decay
                if beta_i_scalar > 1e-12:
                    damped = damper.project(state)
                    high_freq = state - damped
                    result = result - high_freq * beta_i_scalar

        # Add syntony projection: γ(S) Ŝ_op[Ψ]
        syntony_proj = self._syntony_projection(state, S)
        result = result + syntony_proj

        # Add nonlinear correction: Δ_NL[Ψ]
        nl_correction = self._nonlinear_correction(state, S)
        result = result + nl_correction

        return result

    def harmonization_magnitude(self, state: "State") -> float:
        """
        Compute magnitude of harmonization Δ_H = ||Ĥ[Ψ] - Ψ||.

        Args:
            state: Input state

        Returns:
            Harmonization magnitude
        """
        h_state = self.apply(state)
        diff = h_state - state
        return diff.norm()

    def __repr__(self) -> str:
        return (
            f"HarmonizationOperator(beta_0={self.beta_0}, "
            f"gamma_0={self.gamma_0}, num_dampers={len(self._dampers)})"
        )


def default_harmonization_operator() -> HarmonizationOperator:
    """
    Create a harmonization operator with default SRT parameters.

    Returns:
        HarmonizationOperator with standard settings
    """
    return HarmonizationOperator(
        beta_0=PHI_INV_EXACT,  # 1/φ (exact GoldenExact)
        gamma_0=0.1,
        num_dampers=3,
    )


class SyntonicHarmonization:
    def __init__(self, damping_factor="PHI_INV"):
        """
        Args:
            damping_factor: The rate at which entropy (q) is shed. 
                            Defaults to 1/φ (0.618...) using exact Rational logic.
        """
        from syntonic.exact import PHI_INVERSE, Rational, GoldenExact
        
        # We assume PHI_INV is defined as the exact Rational(89, 144) or GoldenExact approximation
        if damping_factor == "PHI_INV":
            self.damping = PHI_INVERSE # 1/φ
        else:
            self.damping = damping_factor

    def apply(self, state: "State") -> "State":
        """
        Performs LLL Lattice Snapping (Harmonization) on a Syntonic state.
        
        1. Projects the Differentiated gradient onto the Golden Cone.
        2. Snaps to the nearest GoldenExact integer.
        3. Reduces the 'q' deficit (Cooling).
        """
        from syntonic.core.state import State
        
        if state.dtype.name != "syntonic":
             raise TypeError("SRI-HARMONIZATION: Input must be a SyntonicExact field.")

        raw_data = state.data
        
        # 1. The Lattice Snap (The 'H' Operation)
        # Implementing localized helper:
        vals = raw_data.to_list()
        snapped_vals = []
        for v in vals:
            # Identity snap for now as Syntonic IS exact
            snapped_vals.append(v)
            
        from syntonic._core import TensorStorage
        snapped_data = TensorStorage.from_list(snapped_vals, raw_data.shape, "syntonic", raw_data.device.name)

        # 2. Calculate the "Snap Distance" (Entropy Shedding)
        # diff = |raw - snapped|
        snap_distance = (raw_data - snapped_data)

        # 3. Update the Master Equation
        # (Implicitly handled in future)
        
        return State(snapped_data, dtype=state.dtype, device=state.device)
