"""
Differentiation Operator D̂ for CRT.

Implements the full differentiation formula:
D̂[Ψ] = Ψ + Σᵢ αᵢ(S) P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]

Where:
- P̂ᵢ: Fourier mode projectors (orthogonal)
- αᵢ(S) = α₀ × (1 - S) × wᵢ  (syntony-dependent coupling)
- ζ(S) = ζ₀ × (1 - S)  (Laplacian diffusion coefficient)
- ∇²: Discrete Laplacian operator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from applications._core import GoldenExact, Rational, PHI_INVERSE
from applications.crt.operators.base import OperatorBase
from applications.crt.operators.projectors import (
    FourierProjector,
    LaplacianOperator,
    create_mode_projectors,
)
from applications._core import TensorStorage

if TYPE_CHECKING:
    from applications.core.state import State

# Use exact golden inverse from syntonic_applications.exact
PHI_INV = PHI_INVERSE


class DifferentiationOperator(OperatorBase):
    """
    Differentiation Operator D̂.

    Increases complexity/differentiation in a state by:
    1. Amplifying selected Fourier modes (via P̂ᵢ projectors)
    2. Adding Laplacian diffusion (∇²)

    The coupling strengths are syntony-dependent:
    - High syntony (S ≈ 1): Weak differentiation
    - Low syntony (S ≈ 0): Strong differentiation

    This creates the fundamental asymmetry: differentiation is stronger
    when the system is far from equilibrium.

    Example:
        >>> D_op = DifferentiationOperator(alpha_0=0.1, zeta_0=0.01)
        >>> evolved = D_op.apply(state)
    """

    def __init__(
        self,
        alpha_0=Rational(1, 10),
        zeta_0=Rational(1, 100),
        num_modes: int = 8,
        projectors: Optional[List[FourierProjector]] = None,
        weights: Optional[List[GoldenExact]] = None,
        laplacian: Optional[LaplacianOperator] = None,
    ):
        """
        Create a differentiation operator.

        Args:
            alpha_0: Base coupling strength for Fourier modes
            zeta_0: Base Laplacian diffusion coefficient
            num_modes: Number of Fourier modes to use (if projectors not given)
            projectors: Custom Fourier projectors (optional)
            weights: Custom mode weights wᵢ (optional, default: golden decay)
            laplacian: Custom Laplacian operator (optional)
        """
        # Coerce base scalars to GoldenExact for exact algebraic arithmetic
        def to_golden(x):
            if isinstance(x, GoldenExact):
                return x
            # Accept Rational-like objects without converting through float()
            if isinstance(x, Rational) or (hasattr(x, "numerator") and hasattr(x, "denominator") and not isinstance(x, float)):
                # Represent rational p/q as GoldenExact(p,0) / GoldenExact(q,0)
                p = x.numerator
                q = x.denominator
                return GoldenExact.from_integers(p, 0) / GoldenExact.from_integers(q, 0)
            # For int, use exact conversion
            if isinstance(x, int):
                return GoldenExact.from_integers(x, 0)
            # For float/unknown types, approximate as GoldenExact (last resort)
            return GoldenExact.nearest(float(x), 1 << 30)  # input boundary

        self.alpha_0 = to_golden(alpha_0)
        self.zeta_0 = to_golden(zeta_0)
        self.num_modes = num_modes

        # Projectors will be initialized on first use if not provided
        self._projectors = projectors
        self._laplacian = laplacian or LaplacianOperator(boundary="periodic")

        # Mode weights (golden ratio decay by default)
        if weights is not None:
            self._weights = weights
        else:
            # Compute exact golden-decay weights using GoldenExact.power
            self._weights = [PHI_INV.power(i) for i in range(num_modes)]

    @property
    def projectors(self) -> List[FourierProjector]:
        """Get Fourier projectors (may be lazily initialized)."""
        return self._projectors or []

    @property
    def weights(self) -> List[float]:
        """Get mode weights."""
        return self._weights

    @property
    def laplacian(self) -> LaplacianOperator:
        """Get Laplacian operator."""
        return self._laplacian

    def _ensure_projectors(self, size: int) -> List[FourierProjector]:
        """Ensure projectors exist for given size."""
        if self._projectors is None:
            self._projectors = create_mode_projectors(
                size=size,
                num_modes=self.num_modes,
                include_dc=False,  # Don't include DC for differentiation
            )
        return self._projectors

    def apply(
        self,
        state: "State",
        syntony: Optional[float] = None,
        **kwargs,
    ) -> "State":
        """
        Apply differentiation operator D̂.

        D̂[Ψ] = Ψ + Σᵢ αᵢ(S) P̂ᵢ[Ψ] + ζ(S) ∇²[Ψ]

        Args:
            state: Input state Ψ
            syntony: Current syntony S (if None, computed from state)

        Returns:
            Differentiated state D̂[Ψ]
        """
        # Get syntony (convert numeric syntony to GoldenExact via nearest)
        S_val = syntony if syntony is not None else state.syntony
        if isinstance(S_val, GoldenExact):
            S_ge = S_val
        elif isinstance(S_val, Rational) or (hasattr(S_val, "numerator") and hasattr(S_val, "denominator") and not isinstance(S_val, float)):
            p = S_val.numerator
            q = S_val.denominator
            S_ge = GoldenExact.from_integers(p, 0) / GoldenExact.from_integers(q, 0)
        elif isinstance(S_val, int):
            S_ge = GoldenExact.from_integers(S_val, 0)
        else:
            # syntony is typically produced as a float by the backend; convert via nearest
            S_ge = GoldenExact.nearest(float(S_val), 1 << 30)  # input boundary

        # Ensure projectors are initialized
        projectors = self._ensure_projectors(state.size)

        # Start with identity: result = Ψ
        result = state

        # Add Fourier mode contributions: Σᵢ αᵢ(S) P̂ᵢ[Ψ]
        for i, projector in enumerate(projectors):
            if i >= len(self._weights):
                break

            # αᵢ(S) = α₀ × (1 - S) × wᵢ (exact GoldenExact arithmetic)
            w_ge = self._weights[i]
            # Ensure w_ge is GoldenExact (should already be from __init__)
            if not isinstance(w_ge, GoldenExact):
                if isinstance(w_ge, Rational) or (hasattr(w_ge, "numerator") and hasattr(w_ge, "denominator") and not isinstance(w_ge, float)):
                    p = w_ge.numerator
                    q = w_ge.denominator
                    w_ge = GoldenExact.from_integers(p, 0) / GoldenExact.from_integers(q, 0)
                elif isinstance(w_ge, int):
                    w_ge = GoldenExact.from_integers(w_ge, 0)
                else:
                    w_ge = GoldenExact.nearest(float(w_ge), 1 << 30)  # input boundary
            
            # All operations now in GoldenExact
            alpha_ge = self.alpha_0 * (GoldenExact.from_integers(1, 0) - S_ge) * w_ge

            # EXACT ARITHMETIC: Use GoldenExact scalar directly via mul_scalar_golden
            # State.__mul__(GoldenExact) dispatches to exact backend
            projected = projector.project(state)
            scaled = projected * alpha_ge  # Uses mul_scalar_golden
            result = result + scaled

        # Add Laplacian diffusion: ζ(S) ∇²[Ψ]
        zeta_ge = self.zeta_0 * (GoldenExact.from_integers(1, 0) - S_ge)

        laplacian_term = self._laplacian.apply(state)
        # Use GoldenExact scalar directly (dispatches to mul_scalar_golden)
        if zeta_ge != GoldenExact.from_integers(0, 0):  # Check exact zero, not approximate
            result = result + laplacian_term * zeta_ge

        return result

    def differentiation_magnitude(self, state: "State") -> GoldenExact:
        """
        Compute magnitude of differentiation Δ_D = ||D̂[Ψ] - Ψ||.

        Args:
            state: Input state

        Returns:
            Differentiation magnitude (exact GoldenExact)
        """
        d_state = self.apply(state)
        diff = d_state - state
        norm_float = diff.norm()
        return GoldenExact.nearest(norm_float, 1 << 30)  # input boundary

    def __repr__(self) -> str:
        return (
            f"DifferentiationOperator(alpha_0={self.alpha_0}, "
            f"zeta_0={self.zeta_0}, num_modes={self.num_modes})"
        )


def default_differentiation_operator() -> DifferentiationOperator:
    """
    Create a differentiation operator with default SRT parameters.

    Returns:
        DifferentiationOperator with standard settings
    """
    return DifferentiationOperator(
        alpha_0=Rational(1, 10),
        zeta_0=Rational(1, 100),
        num_modes=8,
    )


class SyntonicDifferentiation:
    def __init__(self, force_prime=3):
        # We align the operator with a Fermat Prime Force Channel
        # Use Rational from global import (assumed available) or local import
        from applications.exact import Rational
        self.force_prime = Rational(force_prime, 1)

    def apply(self, state: "State") -> "State":
        """
        Performs exact differentiation on a Syntonic-dtype state.
        Calculates the gradient ΔΨ / ΔW across the E8 manifold.
        """
        from applications.core.state import State
        
        if state.dtype.name != "syntonic":
            # Force compliance: SRI must not differentiate floats
            # (Relaxed check: allow string match if DType object equality fails)
            raise TypeError("SRI-DIFFERENTIATION: State must use DType.syntonic for exact gradient calculation.")

        # 1. Access the underlying SyntonicExact data
        # The Rust backend handles the symbolic (a + bφ) arithmetic
        raw_data = state.data
        
        # 2. Shift and Subtract (Symbolic Gradient)
        # We perform a discrete differentiation on the lattice roots
        # result = (state[i+1] - state[i]) * force_prime
        shifted_data = self._roll_syntonic_tensor(raw_data)
        
        # This multiplication happens in Rust using the Mul<Rational> 
        # implementation we just added to SyntonicExact.
        exact_gradient = (shifted_data - raw_data) * self.force_prime
        
        # 3. Update the Deficit Layer
        # Differentiation 'burns' syntony, increasing the q_deficit coefficient
        # Note: This requires iteration or vectorized update.
        # But we must return a State.

        return State(exact_gradient, dtype=state.dtype, device=state.device)

    def _roll_syntonic_tensor(self, tensor):
        """Rolls the lattice coordinates for finite difference calculation."""
        from applications._core import TensorStorage
        vals = tensor.to_list()
        if len(vals) > 0:
            vals = vals[-1:] + vals[:-1]
        return TensorStorage.from_list(vals, tensor.shape, "syntonic", tensor.device.name)
