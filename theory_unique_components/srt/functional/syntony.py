"""
Syntony Functional - S[Ψ] with global bound S ≤ φ.

The syntony functional measures the coherence of a quantum state
with respect to the SRT geometric structure. It is defined as:

    S[Ψ] = φ · Tr[Ψ†·exp(-L²_knot/φ)·Ψ] / Tr[exp(-L²_vac/φ)]

The fundamental theorem of SRT states: S[Ψ] ≤ φ for all physical states.

Example:
    >>> from srt_library.theory_unique_components.srt.functional import syntony_functional
    >>> S = syntony_functional()
    >>> value, is_valid = S.verify_bound(psi)
    >>> is_valid  # Always True for physical states
    True
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from srt_library.core import enumerate_windings_exact_norm
from srt_library.core.exact import PHI
from srt_library.theory_unique_components.srt.geometry.winding import WindingState, winding_state
from srt_library.theory_unique_components.srt.spectral.knot_laplacian import KnotLaplacian

if TYPE_CHECKING:
    from srt_library.core import State


class SyntonyFunctional:
    """
    Syntony functional S[Ψ] with global bound S ≤ φ.

    The syntony functional measures how coherently a state Ψ
    couples to the golden recursion structure of SRT.

    Definition:
        S[Ψ] = φ · ⟨Ψ|exp(-L²_knot/φ)|Ψ⟩ / Z_vac

    where Z_vac = Tr[exp(-L²_vac/φ)] is the vacuum partition function.

    The bound S[Ψ] ≤ φ is saturated only by special "syntonic" states
    that are maximally coherent with the golden structure.

    Attributes:
        phi: Golden ratio (also the global bound)
        laplacian: The knot Laplacian operator
        max_norm: Maximum |n|² for computations

    Example:
        >>> S = SyntonyFunctional()
        >>> S.global_bound
        1.618033988749895
        >>> S.verify_bound(psi)
        (0.8, True)
    """

    def __init__(
        self,
        phi: Optional[float] = None,
        max_norm: int = 20,
    ):
        """
        Initialize syntony functional.

        Uses exact golden ratio φ from srt_library.core.exact.PHI by default.

        Args:
            phi: Golden ratio float value. If None, uses PHI.eval().
            max_norm: Maximum |n|² for winding states.
        """
        self._phi_exact = PHI  # Store exact golden ratio
        self._phi = phi if phi is not None else PHI.eval()
        self._max_norm = max_norm
        self._laplacian = KnotLaplacian(phi=self._phi, max_norm=max_norm)
        self._Z_vac = self._compute_vacuum_partition()

    def _compute_vacuum_partition(self) -> float:
        """Compute vacuum partition function Z_vac = Tr[exp(-L²_vac/φ)]."""
        return self._laplacian.heat_kernel_trace(1.0 / self._phi)

    @property
    def phi(self) -> float:
        """Golden ratio value."""
        return self._phi

    @property
    def global_bound(self) -> float:
        """Global bound on syntony: S[Ψ] ≤ φ."""
        return self._phi

    @property
    def laplacian(self) -> KnotLaplacian:
        """The knot Laplacian operator."""
        return self._laplacian

    @property
    def vacuum_partition(self) -> float:
        """Vacuum partition function Z_vac."""
        return self._Z_vac

    def evaluate(
        self,
        coefficients: Dict[WindingState, complex],
        normalize: bool = True,
    ) -> float:
        """
        Compute S[Ψ] for state Ψ = Σ_n c_n |n⟩.

        S[Ψ] = φ · Σ_n |c_n|² exp(-λ_n/φ) / Z_vac

        Args:
            coefficients: Dict mapping winding states to complex coefficients
            normalize: If True, normalize coefficients to unit norm

        Returns:
            Syntony value S[Ψ]
        """
        if not coefficients:
            return 0.0

        # Compute norm for normalization
        norm_sq = sum(abs(c) ** 2 for c in coefficients.values())
        if norm_sq < 1e-15:
            return 0.0

        # Compute weighted sum
        total = 0.0
        for n, c in coefficients.items():
            prob = abs(c) ** 2
            if normalize:
                prob /= norm_sq

            lambda_n = self._laplacian.eigenvalue(n)
            boltzmann = math.exp(-lambda_n / self._phi)
            total += prob * boltzmann

        return self._phi * total / self._Z_vac

    def evaluate_from_state(
        self, psi: "State", winding_map: Optional[Dict] = None
    ) -> float:
        """
        Compute S[Ψ] for a State object.

        Args:
            psi: A syntonic State object
            winding_map: Optional mapping from state indices to WindingStates

        Returns:
            Syntony value S[Ψ]
        """
        # Extract coefficients from State
        # This depends on how State exposes its data
        if hasattr(psi, "to_numpy"):
            data = psi.to_numpy().flatten()
        elif hasattr(psi, "data"):
            data = list(psi.data)
        else:
            raise TypeError("Cannot extract coefficients from State")

        # Default winding map: index i -> winding with |n|² = i
        if winding_map is None:
            winding_map = self._default_winding_map(len(data))

        coefficients = {}
        for i, c in enumerate(data):
            if i in winding_map:
                coefficients[winding_map[i]] = complex(c)

        return self.evaluate(coefficients)

    def _default_winding_map(self, size: int) -> Dict[int, WindingState]:
        """Create default mapping from indices to winding states."""
        mapping = {}
        idx = 0
        for norm_sq in range(self._max_norm + 1):
            for n in self._enumerate_by_norm(norm_sq):
                if idx >= size:
                    return mapping
                mapping[idx] = n
                idx += 1
        return mapping

    def _enumerate_by_norm(self, norm_sq: int) -> List[WindingState]:
        """Enumerate winding states with given |n|²."""
        # Use Rust enumerator for ~50x speedup
        return enumerate_windings_exact_norm(norm_sq)

    def verify_bound(
        self, coefficients: Dict[WindingState, complex]
    ) -> Tuple[float, bool]:
        """
        Verify that S[Ψ] ≤ φ.

        Args:
            coefficients: Dict mapping winding states to complex coefficients

        Returns:
            Tuple of (syntony_value, satisfies_bound)
        """
        S = self.evaluate(coefficients)
        return (S, S <= self._phi + 1e-10)

    def gradient(
        self,
        coefficients: Dict[WindingState, complex],
    ) -> Dict[WindingState, complex]:
        """
        Compute gradient ∂S/∂c*_n for optimization.

        The gradient points in the direction of increasing syntony.

        Args:
            coefficients: Current state coefficients

        Returns:
            Dict of gradient components
        """
        norm_sq = sum(abs(c) ** 2 for c in coefficients.values())
        if norm_sq < 1e-15:
            return {n: complex(0) for n in coefficients}

        # Current syntony value
        S = self.evaluate(coefficients, normalize=True)

        gradient = {}
        for n, c in coefficients.items():
            lambda_n = self._laplacian.eigenvalue(n)
            boltzmann = math.exp(-lambda_n / self._phi)

            # Gradient of normalized functional
            # ∂S/∂c*_n = (φ/Z) · (boltzmann/norm_sq) · c_n - S · c_n / norm_sq
            dS = (self._phi / self._Z_vac) * (boltzmann / norm_sq) * c - S * c / norm_sq
            gradient[n] = dS

        return gradient

    def optimize_step(
        self,
        coefficients: Dict[WindingState, complex],
        step_size: float = 0.1,
    ) -> Dict[WindingState, complex]:
        """
        Take one gradient ascent step toward maximum syntony.

        Args:
            coefficients: Current state coefficients
            step_size: Learning rate

        Returns:
            Updated coefficients
        """
        grad = self.gradient(coefficients)

        updated = {}
        for n, c in coefficients.items():
            updated[n] = c + step_size * grad.get(n, 0)

        # Renormalize
        norm = math.sqrt(sum(abs(c) ** 2 for c in updated.values()))
        if norm > 1e-15:
            updated = {n: c / norm for n, c in updated.items()}

        return updated

    def ground_state(self) -> Tuple[Dict[WindingState, complex], float]:
        """
        Find the state that minimizes syntony (ground state).

        The ground state is the vacuum |0⟩.

        Returns:
            Tuple of (coefficients, syntony_value)
        """
        vacuum = winding_state(0, 0, 0, 0)
        coefficients = {vacuum: complex(1.0)}
        return (coefficients, self.evaluate(coefficients))

    def excited_state(
        self, n: WindingState
    ) -> Tuple[Dict[WindingState, complex], float]:
        """
        Compute syntony for a single excited state |n⟩.

        Args:
            n: The winding state

        Returns:
            Tuple of (coefficients, syntony_value)
        """
        coefficients = {n: complex(1.0)}
        return (coefficients, self.evaluate(coefficients))

    def coherent_superposition(
        self,
        states: List[WindingState],
        equal_weight: bool = True,
    ) -> Tuple[Dict[WindingState, complex], float]:
        """
        Compute syntony for coherent superposition of states.

        Args:
            states: List of winding states to superpose
            equal_weight: If True, use equal amplitudes

        Returns:
            Tuple of (coefficients, syntony_value)
        """
        if equal_weight:
            amp = 1.0 / math.sqrt(len(states))
            coefficients = {n: complex(amp) for n in states}
        else:
            # Golden-weighted amplitudes
            total = sum(math.exp(-n.norm_squared / (2 * self._phi)) for n in states)
            coefficients = {
                n: complex(math.exp(-n.norm_squared / (2 * self._phi)) / total)
                for n in states
            }

        return (coefficients, self.evaluate(coefficients))

    def thermal_state(self, beta: float) -> Tuple[Dict[WindingState, complex], float]:
        """
        Compute syntony for thermal state at inverse temperature β.

        The thermal state has ρ = exp(-β·L²_knot) / Z(β).

        Args:
            beta: Inverse temperature

        Returns:
            Tuple of (effective_coefficients, syntony_value)
        """
        from srt_library.core import enumerate_windings_by_norm

        # Generate thermal distribution using Rust enumerator (~50x speedup)
        coefficients = {}
        Z = 0.0

        by_norm = enumerate_windings_by_norm(self._max_norm)
        for windings in by_norm.values():
            for n in windings:
                lambda_n = self._laplacian.eigenvalue(n)
                prob = math.exp(-beta * lambda_n)
                coefficients[n] = complex(math.sqrt(prob))
                Z += prob

        # Normalize
        Z_sqrt = math.sqrt(Z)
        coefficients = {n: c / Z_sqrt for n, c in coefficients.items()}

        return (coefficients, self.evaluate(coefficients))

    def __repr__(self) -> str:
        return f"SyntonyFunctional(phi={self._phi:.6f}, bound={self._phi:.6f})"


def syntony_functional(
    phi: Optional[float] = None,
    max_norm: int = 20,
) -> SyntonyFunctional:
    """
    Create a SyntonyFunctional instance.

    Factory function for SyntonyFunctional.

    Args:
        phi: Golden ratio value. If None, uses PHI_NUMERIC.
        max_norm: Maximum |n|² for winding states.

    Returns:
        SyntonyFunctional instance

    Example:
        >>> S = syntony_functional()
        >>> S.global_bound
        1.618033988749895
    """
    return SyntonyFunctional(phi=phi, max_norm=max_norm)


def compute_syntony(
    coefficients: Dict[WindingState, complex],
    phi: Optional[float] = None,
) -> float:
    """
    Quick computation of syntony S[Ψ].

    Convenience function for single computation.

    Args:
        coefficients: Dict mapping winding states to complex coefficients
        phi: Golden ratio value. If None, uses PHI_NUMERIC.

    Returns:
        Syntony value

    Example:
        >>> from srt_library.theory_unique_components.srt.geometry import winding_state
        >>> n = winding_state(1, 0, 0, 0)
        >>> compute_syntony({n: 1.0})
        ...
    """
    S = SyntonyFunctional(phi=phi)
    return S.evaluate(coefficients)
