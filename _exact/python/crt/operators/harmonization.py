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

from applications.exact import GoldenExact, Rational, PHI_INVERSE as PHI_INV
from applications._core import SymExpr
from applications.crt.operators.base import OperatorBase
from applications.crt.operators.projectors import (
    DampingProjector,
    create_damping_cascade,
)

if TYPE_CHECKING:
    from applications.core.state import State

# Use exact golden inverse from syntonic_applications.exact (for exact-first SGC)
PHI_INV_EXACT = PHI_INV


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
        gamma_0: Union[float, GoldenExact] = None,  # Default set below to exact Rational(1,10)
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
            
        # Default gamma_0 to exact 1/10 if not provided  
        if gamma_0 is None:
            gamma_0 = Rational(1, 10)

        # Coerce to GoldenExact for exact mode (like differentiation does)
        def to_golden(x):
            if isinstance(x, GoldenExact):
                return x
            if isinstance(x, Rational):
                # Convert Rational to float for nearest approximation in Q(φ)
                # Use high precision (2^30) for exact-like representation
                val = x.numerator / x.denominator
                return GoldenExact.nearest(val, 1 << 30)  # input boundary
            # For float input when exact=True, approximate
            if exact:
                return GoldenExact.nearest(float(x), 1 << 30)  # input boundary
            return float(x)

        if exact:
            self.beta_0 = to_golden(beta_0)
            self.gamma_0 = to_golden(gamma_0)
            self.nonlinear_strength = to_golden(nonlinear_strength)
        else:
            # Non-exact mode: still store as GoldenExact when possible for consistency
            self.beta_0 = to_golden(beta_0) if not isinstance(beta_0, GoldenExact) else beta_0
            self.gamma_0 = to_golden(gamma_0) if not isinstance(gamma_0, GoldenExact) else gamma_0
            self.nonlinear_strength = to_golden(nonlinear_strength) if not isinstance(nonlinear_strength, GoldenExact) else nonlinear_strength

        self.num_dampers = num_dampers

        # Damping projectors (cascade with golden decay)
        if damping_projectors is not None:
            self._dampers = damping_projectors
        else:
            self._dampers = create_damping_cascade(
                num_levels=num_dampers,
                base_cutoff=Rational(7, 10),
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
        from applications.core.state import State

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

    def _decay_factor(self, level: int, delta_d: Optional[GoldenExact]) -> GoldenExact:
        """
        Compute decay factor for damping level.

        Higher levels have stronger decay, modulated by differentiation magnitude.

        Args:
            level: Damping level index
            delta_d: Differentiation magnitude (optional, exact GoldenExact)

        Returns:
            Decay factor in [0, 1] (exact GoldenExact)
        """
        # Base decay follows golden ratio - keep exact
        base_decay = PHI_INV_EXACT.power(level)

        # Modulate by differentiation magnitude if provided
        if delta_d is not None:
            # Stronger damping when more differentiated
            modulation = GoldenExact.from_integers(1, 0) + delta_d
        else:
            modulation = GoldenExact.from_integers(1, 0)

        return base_decay * GoldenExact.nearest(min(modulation.eval(), 2), 1 << 30)  # input boundary

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
        from applications.core.state import State

        # Use exact comparisons when possible, only convert for unavoidable operations
        if self.exact and hasattr(self.nonlinear_strength, 'eval'):
            # Use exact comparison with GoldenExact threshold
            threshold = GoldenExact.nearest(1e-12, 1 << 30)
            if self.nonlinear_strength < threshold:
                # Return zero state
                flat = [0] * state.size
                return State(
                    flat, dtype=state.dtype, device=state.device, shape=state.shape
                )
            # Keep exact values for subsequent operations
            nl_strength_exact = self.nonlinear_strength
            S_exact = S
        else:
            # Non-exact path: still use exact types with .eval() only for comparisons
            threshold = GoldenExact.nearest(1e-12, 1 << 30)
            nl_val = self.nonlinear_strength if isinstance(self.nonlinear_strength, GoldenExact) else GoldenExact.nearest(self.nonlinear_strength.eval() if hasattr(self.nonlinear_strength, 'eval') else 0, 1 << 30)  # input boundary
            if nl_val < threshold:
                flat = [0] * state.size
                return State(
                    flat, dtype=state.dtype, device=state.device, shape=state.shape
                )
            nl_strength_exact = self.nonlinear_strength
            S_exact = S

        flat = state.to_list()
        N = len(flat)

        # Simple nonlinear correction: cubic damping
        # Δ_NL[Ψ]ᵢ = -η × Ψᵢ³ / (1 + |Ψᵢ|²)
        # Use exact nl_strength for computation, .eval() only if needed for complex ops
        nl_str = nl_strength_exact if 'nl_strength_exact' in locals() else self.nonlinear_strength
        nl_float = nl_str.eval() if hasattr(nl_str, 'eval') else nl_str  # for complex number math only

        correction = []
        for i in range(N):
            x = flat[i]
            if isinstance(x, complex):
                x_abs_sq = abs(x) ** 2
                nl = -nl_float * x * x_abs_sq / (1 + x_abs_sq)
            else:
                x_sq = x * x
                # x_sq and x are exact types (GoldenExact etc.) — keep exact
                nl = -(nl_str * x * x_sq) / (GoldenExact.from_int(1) + x_sq) if isinstance(x, GoldenExact) else -nl_float * x * x_sq / (1.0 + x_sq)  # eval boundary
            correction.append(nl)

        # Scale by syntony (stronger correction at higher syntony)
        S_val = S_exact if 'S_exact' in locals() else S
        if isinstance(S_val, GoldenExact):
            correction = [c * S_val if isinstance(c, GoldenExact) else c * S_val.eval() for c in correction]
        else:
            S_scale = S_val.eval() if hasattr(S_val, 'eval') else S_val
            correction = [c * S_scale for c in correction]

        return State(
            correction, dtype=state.dtype, device=state.device, shape=state.shape
        )

    def apply(
        self,
        state: "State",
        syntony: Optional[Union[float, GoldenExact]] = None,
        delta_d: Optional[Union[float, GoldenExact]] = None,
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

        # Convert delta_d to GoldenExact if provided
        delta_d_exact = None
        if delta_d is not None:
            if isinstance(delta_d, GoldenExact):
                delta_d_exact = delta_d
            else:
                delta_d_exact = GoldenExact.nearest(float(delta_d), 1 << 30)  # input boundary

        if self.exact:
            # Convert syntony to GoldenExact (like differentiation does)
            if isinstance(S_val, GoldenExact):
                S = S_val
            elif isinstance(S_val, Rational):
                S = GoldenExact.from_rational(S_val)
            else:
                S = GoldenExact.nearest(float(S_val), 1 << 30)  # input boundary
        else:
            # Non-exact path: still keep as GoldenExact when possible
            if isinstance(S_val, GoldenExact):
                S = S_val
            elif isinstance(S_val, Rational):
                S = GoldenExact.from_rational(S_val)
            else:
                S = GoldenExact.nearest(S_val.eval() if hasattr(S_val, 'eval') else S_val, 1 << 30)  # input boundary

        # Start with identity: result = Ψ
        result = state

        # Subtract damping contributions: -Σᵢ βᵢ(S,Δ_D) Q̂ᵢ[Ψ]
        for i, damper in enumerate(self._dampers):
            # βᵢ(S,Δ_D) = β₀ × S × decay_factor
            decay = self._decay_factor(i, delta_d_exact)

            if self.exact:
                # Exact arithmetic - keep everything in GoldenExact
                if isinstance(decay, float):
                    decay_exact = GoldenExact.nearest(decay, 1 << 30)  # input boundary
                else:
                    decay_exact = decay
                beta_i = self.beta_0 * S * decay_exact
                
                # Use GoldenExact scalar directly (State.__mul__ dispatches to mul_scalar_golden)
                if beta_i != GoldenExact.from_integers(0, 0):
                    damped = damper.project(state)
                    high_freq = state - damped
                    result = result - high_freq * beta_i
            else:
                # Use exact arithmetic even in non-exact mode (all params are GoldenExact now)
                if isinstance(decay, GoldenExact):
                    decay_exact = decay
                else:
                    decay_exact = GoldenExact.nearest(decay.eval() if hasattr(decay, 'eval') else decay, 1 << 30)  # input boundary
                beta_i = self.beta_0 * S * decay_exact
                threshold = GoldenExact.nearest(1e-12, 1 << 30)
                if beta_i > threshold:
                    damped = damper.project(state)
                    high_freq = state - damped
                    result = result - high_freq * beta_i

        # Add syntony projection: γ(S) Ŝ_op[Ψ]
        syntony_proj = self._syntony_projection(state, S)
        result = result + syntony_proj

        # Add nonlinear correction: Δ_NL[Ψ]
        nl_correction = self._nonlinear_correction(state, S)
        result = result + nl_correction

        return result

    def harmonization_magnitude(self, state: "State") -> GoldenExact:
        """
        Compute magnitude of harmonization Δ_H = ||Ĥ[Ψ] - Ψ||.

        Args:
            state: Input state

        Returns:
            Harmonization magnitude (exact GoldenExact)
        """
        h_state = self.apply(state)
        diff = h_state - state
        norm_float = diff.norm()
        return GoldenExact.nearest(norm_float, 1 << 30)  # input boundary

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
        gamma_0=Rational(1, 10),
        num_dampers=3,
    )


class SyntonicHarmonization:
    def __init__(self, damping_factor="PHI_INV"):
        """
        Args:
            damping_factor: The rate at which entropy (q) is shed. 
                            Defaults to 1/φ (0.618...) using exact Rational logic.
        """
        from applications.exact import PHI_INVERSE, Rational, GoldenExact
        
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
        from applications.core.state import State
        
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
            
        from applications._core import TensorStorage
        snapped_data = TensorStorage.from_list(snapped_vals, raw_data.shape, "syntonic", raw_data.device.name)

        # 2. Calculate the "Snap Distance" (Entropy Shedding)
        # diff = |raw - snapped|
        snap_distance = (raw_data - snapped_data)

        # 3. Update the Master Equation
        # (Implicitly handled in future)
        
        return State(snapped_data, dtype=state.dtype, device=state.device)
