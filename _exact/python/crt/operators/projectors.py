"""
Projector operators for CRT.

Provides:
- FourierProjector: Projects onto Fourier mode subspaces
- DampingProjector: High-frequency damping via low-pass filtering
- LaplacianOperator: Discrete Laplacian ∇² for diffusion
"""

from __future__ import annotations

import cmath
from typing import TYPE_CHECKING, List, Optional, Union
from applications._core import GoldenExact, Rational

if TYPE_CHECKING:
    from applications.core.state import State


class FourierProjector:
    """
    Projects a state onto a subset of Fourier modes.

    The Fourier projector P̂ᵢ isolates specific frequency components
    of the state. This is used in the differentiation operator to
    selectively amplify certain modes.

    For a 1D state of size N, the DFT gives N frequency components.
    This projector keeps only the specified mode indices and zeros the rest.

    Properties:
    - Idempotent: P̂² = P̂
    - Orthogonal: Different mode projectors are orthogonal
    """

    def __init__(self, mode_indices: List[int], size: int):
        """
        Create a Fourier projector for specific modes.

        Args:
            mode_indices: Which Fourier modes to keep (0 = DC, 1 = first harmonic, etc.)
            size: Size of the state to project
        """
        self.mode_indices = set(mode_indices)
        self.size = size
        self._precompute_dft_matrix()

    def _precompute_dft_matrix(self):
        """Precompute DFT matrix elements for efficiency."""
        N = self.size
        self._omega = cmath.exp(-2j * cmath.pi / N)
        self._omega_inv = cmath.exp(2j * cmath.pi / N)

    def _dft(self, data: List[complex]) -> List[complex]:
        """Compute DFT of data."""
        N = len(data)
        result = []
        for k in range(N):
            s = 0j
            for n in range(N):
                s += data[n] * (self._omega ** (k * n))
            result.append(s)
        return result

    def _idft(self, data: List[complex]) -> List[complex]:
        """Compute inverse DFT of data."""
        N = len(data)
        result = []
        for n in range(N):
            s = 0j
            for k in range(N):
                s += data[k] * (self._omega_inv ** (k * n))
            result.append(s / N)
        return result

    def project(self, state: "State") -> "State":
        """
        Project state onto selected Fourier modes.

        Args:
            state: Input state

        Returns:
            State with only selected Fourier components
        """
        from applications.core.state import State

        flat = state.to_list()
        N = len(flat)

        # Convert to complex if needed
        if not isinstance(flat[0], complex):
            flat = [complex(x) for x in flat]

        # DFT
        freq = self._dft(flat)

        # Zero out non-selected modes
        for k in range(N):
            if k not in self.mode_indices:
                freq[k] = 0j

        # Inverse DFT
        projected = self._idft(freq)

        # Return real part if input was real
        if state.dtype.name != "complex128":
            projected = [x.real for x in projected]

        return State(
            projected, dtype=state.dtype, device=state.device, shape=state.shape
        )

    def __repr__(self) -> str:
        return f"FourierProjector(modes={sorted(self.mode_indices)}, size={self.size})"


class DampingProjector:
    """
    High-frequency damping projector.

    Applies a low-pass filter to dampen high-frequency components.
    Used in the harmonization operator to reduce complexity.

    The damping follows a smooth rolloff:
    D(k) = 1 / (1 + (k/k_c)^order)

    where k_c is the cutoff frequency and order controls rolloff steepness.
    """

    def __init__(
        self,
        cutoff_fraction: Union[float, Rational] = Rational(1, 2),
        order: int = 2,
        size: Optional[int] = None,
    ):
        """
        Create a damping projector.

        Args:
            cutoff_fraction: Fraction of Nyquist frequency for cutoff (0 to 1)
            order: Filter order (higher = steeper rolloff)
            size: Expected state size (optional, determined at runtime if not given)
        """
        self.cutoff_fraction = cutoff_fraction
        self.order = order
        self.size = size

    def _compute_filter(self, N: int) -> List[float]:
        """Compute frequency-domain filter coefficients."""
        k_c = self.cutoff_fraction * (N // 2)
        if k_c < 1:
            k_c = 1

        coeffs = []
        for k in range(N):
            # Map to frequency (handle aliasing)
            freq = k if k <= N // 2 else N - k
            if freq == 0:
                coeffs.append(Rational(1, 1))
            else:
                coeffs.append(Rational(1, 1) / (Rational(1, 1) + Rational(freq, k_c) ** self.order))
        return coeffs

    def _dft(self, data: List[complex]) -> List[complex]:
        """Compute DFT."""
        N = len(data)
        omega = cmath.exp(-2j * cmath.pi / N)
        result = []
        for k in range(N):
            s = 0j
            for n in range(N):
                s += data[n] * (omega ** (k * n))
            result.append(s)
        return result

    def _idft(self, data: List[complex]) -> List[complex]:
        """Compute inverse DFT."""
        N = len(data)
        omega_inv = cmath.exp(2j * cmath.pi / N)
        result = []
        for n in range(N):
            s = 0j
            for k in range(N):
                s += data[k] * (omega_inv ** (k * n))
            result.append(s / N)
        return result

    def project(self, state: "State") -> "State":
        """
        Apply damping to high frequencies.

        Args:
            state: Input state

        Returns:
            State with dampened high-frequency components
        """
        from applications.core.state import State

        flat = state.to_list()
        N = len(flat)

        # Convert to complex if needed
        if not isinstance(flat[0], complex):
            flat = [complex(x) for x in flat]

        # Compute filter
        filter_coeffs = self._compute_filter(N)

        # Apply in frequency domain
        freq = self._dft(flat)
        for k in range(N):
            freq[k] *= filter_coeffs[k]

        # Inverse DFT
        damped = self._idft(freq)

        # Return real part if input was real
        if state.dtype.name != "complex128":
            damped = [x.real for x in damped]

        return State(damped, dtype=state.dtype, device=state.device, shape=state.shape)

    def __repr__(self) -> str:
        return f"DampingProjector(cutoff={self.cutoff_fraction}, order={self.order})"


class LaplacianOperator:
    """
    Discrete Laplacian operator ∇².

    Computes the discrete second derivative using finite differences:
    ∇²[Ψ]ᵢ = Ψᵢ₊₁ - 2Ψᵢ + Ψᵢ₋₁

    With periodic boundary conditions for wraparound.

    For multi-dimensional states, applies sum of second derivatives
    along each axis.
    """

    def __init__(self, boundary: str = "periodic"):
        """
        Create a Laplacian operator.

        Args:
            boundary: Boundary condition ('periodic' or 'zero')
        """
        self.boundary = boundary

    def apply(self, state: "State") -> "State":
        """
        Apply discrete Laplacian.

        Args:
            state: Input state

        Returns:
            ∇²[Ψ]
        """
        from applications.core.state import State

        flat = state.to_list()
        N = len(flat)

        if N < 3:
            # Too small for Laplacian - return zeros as integer for simplicity
            return State(
                [0] * N, dtype=state.dtype, device=state.device, shape=state.shape
            )

        result = []
        for i in range(N):
            if self.boundary == "periodic":
                # Periodic boundary conditions
                i_prev = (i - 1) % N
                i_next = (i + 1) % N
            else:
                # Zero boundary conditions
                i_prev = i - 1 if i > 0 else i
                i_next = i + 1 if i < N - 1 else i

            # ∇²[Ψ]ᵢ = Ψᵢ₊₁ - 2Ψᵢ + Ψᵢ₋₁
            laplacian_i = flat[i_next] - 2 * flat[i] + flat[i_prev]
            result.append(laplacian_i)

        return State(result, dtype=state.dtype, device=state.device, shape=state.shape)

    def __call__(self, state: "State") -> "State":
        """Convenience wrapper."""
        return self.apply(state)

    def __repr__(self) -> str:
        return f"LaplacianOperator(boundary='{self.boundary}')"


def create_mode_projectors(
    size: int,
    num_modes: int = 8,
    include_dc: bool = True,
) -> List[FourierProjector]:
    """
    Create a set of Fourier mode projectors.

    Args:
        size: State size
        num_modes: Number of mode projectors to create
        include_dc: Whether to include DC (k=0) mode

    Returns:
        List of FourierProjector instances
    """
    projectors = []
    start = 0 if include_dc else 1

    for i in range(num_modes):
        mode_idx = start + i
        if mode_idx < size:
            projectors.append(FourierProjector([mode_idx], size))

    return projectors


def create_damping_cascade(
    num_levels: int = 3,
    base_cutoff: float = 0.7,
    decay: Union[float, GoldenExact] = None,  # Golden ratio decay
) -> List[DampingProjector]:
    """
    Create a cascade of damping projectors with decreasing cutoffs.

    Args:
        num_levels: Number of damping levels
        base_cutoff: Initial cutoff fraction
        decay: Decay factor for each level (default: 1/φ)

    Returns:
        List of DampingProjector instances
    """
    # Default to exact golden ratio reciprocal
    if decay is None:
        decay = GoldenExact(-1, 1)  # 1/φ = φ - 1
    
    projectors = []
    cutoff = base_cutoff

    for _ in range(num_levels):
        projectors.append(DampingProjector(cutoff_fraction=cutoff))
        if isinstance(decay, GoldenExact):
            cutoff *= decay.eval()  # Use display value for interface
        else:
            cutoff *= decay

    return projectors
