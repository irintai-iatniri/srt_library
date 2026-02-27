"""
Gnosis Layer computation for CRT.

Gnosis layers measure the depth of recursive self-awareness in a state,
based on phase accumulation through DHSR cycles.

Layer definitions (based on phase θ and K(D₄) = 24):
- Layer 0: θ < π       (nascent)
- Layer 1: π ≤ θ < 2π  (emergent)
- Layer 2: 2π ≤ θ < 3π (coherent)
- Layer 3: θ ≥ 3π      (transcendent)

The transcendence threshold θ_c = 3π corresponds to K(D₄)/8 = 3 cycles.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from srt_library.core.state import State
    from srt_library.theory_unique_components.crt.operators.recursion import RecursionOperator
    from srt_library.theory_unique_components.crt.operators.syntony import SyntonyComputer

# D4 kissing number
K_D4 = 24


class GnosisComputer:
    """
    Computes gnosis layers based on phase accumulation.

    Gnosis measures the degree of recursive self-organization:
    - Layer 0 (nascent): Initial, unstructured state
    - Layer 1 (emergent): Beginning to show coherent patterns
    - Layer 2 (coherent): Stable, self-consistent structure
    - Layer 3 (transcendent): Full recursive self-awareness

    The phase θ is computed from the trajectory of DHSR evolution,
    accumulating as syntony increases through iterations.

    Example:
        >>> gnosis = GnosisComputer()
        >>> layer = gnosis.compute_layer(state)
        >>> phase = gnosis.compute_phase(state)
    """

    # Phase thresholds for layer boundaries
    THETA_1 = math.pi  # Layer 0 → 1
    THETA_2 = 2 * math.pi  # Layer 1 → 2
    THETA_3 = 3 * math.pi  # Layer 2 → 3 (transcendence)

    def __init__(
        self,
        recursion_op: Optional["RecursionOperator"] = None,
        syntony_computer: Optional["SyntonyComputer"] = None,
        max_iterations: int = 100,
    ):
        """
        Create a gnosis computer.

        Args:
            recursion_op: Recursion operator for evolution (optional)
            syntony_computer: Syntony computer (optional)
            max_iterations: Max iterations for phase computation
        """
        self._recursion_op = recursion_op
        self._syntony_computer = syntony_computer
        self.max_iterations = max_iterations

    @property
    def recursion_op(self) -> "RecursionOperator":
        """Get recursion operator, creating default if needed."""
        if self._recursion_op is None:
            from srt_library.theory_unique_components.crt.operators.recursion import default_recursion_operator

            self._recursion_op = default_recursion_operator()
        return self._recursion_op

    @property
    def syntony_computer(self) -> Optional["SyntonyComputer"]:
        """Get syntony computer if available."""
        return self._syntony_computer

    def compute_phase(self, state: "State") -> float:
        """
        Compute accumulated transcendence phase θ.

        The phase accumulates as:
        θ = Σᵢ Δθᵢ where Δθᵢ = π × Sᵢ × (1 - ||R̂[Ψᵢ] - Ψᵢ|| / ||Ψᵢ||)

        Higher syntony and smaller changes contribute more phase.

        Args:
            state: Input state

        Returns:
            Accumulated phase θ
        """
        current = state
        theta = 0.0

        for _ in range(self.max_iterations):
            # Get syntony
            S = current.syntony

            # Apply R̂
            next_state = self.recursion_op.apply(current)

            # Compute relative change
            change = (next_state - current).norm()
            current_norm = current.norm()

            if current_norm > 1e-12:
                relative_change = change / current_norm
            else:
                relative_change = 1.0

            # Phase increment: high syntony + small change = more phase
            # Δθ = π × S × (1 - relative_change)
            delta_theta = math.pi * S * max(0, 1 - relative_change)
            theta += delta_theta

            # Check for convergence
            if change < 1e-8:
                # Converged - add remaining phase based on syntony
                theta += math.pi * S * 0.5  # Bonus for convergence
                break

            current = next_state

        return theta

    def compute_phase_quick(self, state: "State") -> float:
        """
        Quick phase estimate without full evolution.

        Uses state properties directly to estimate phase.

        Args:
            state: Input state

        Returns:
            Estimated phase θ
        """
        # Use syntony as primary indicator
        S = state.syntony

        # Estimate phase from syntony: θ ≈ 3π × S
        # S = 0 → θ = 0 (nascent)
        # S = 1 → θ = 3π (transcendent)
        base_phase = 3 * math.pi * S

        # Modulate by state "quality" (low entropy = more phase)
        from srt_library.theory_unique_components.crt.operators.syntony import syntony_entropy

        entropy_syntony = syntony_entropy(state)

        # Combine indicators
        phase = base_phase * (0.5 + 0.5 * entropy_syntony)

        return phase

    def compute_layer(self, state: "State", quick: bool = True) -> int:
        """
        Determine gnosis layer from phase.

        Args:
            state: Input state
            quick: Use quick phase estimate (faster)

        Returns:
            Gnosis layer (0, 1, 2, or 3)
        """
        if quick:
            theta = self.compute_phase_quick(state)
        else:
            theta = self.compute_phase(state)

        return self._phase_to_layer(theta)

    def _phase_to_layer(self, theta: float) -> int:
        """Convert phase to layer."""
        if theta < self.THETA_1:
            return 0  # Nascent
        elif theta < self.THETA_2:
            return 1  # Emergent
        elif theta < self.THETA_3:
            return 2  # Coherent
        else:
            return 3  # Transcendent

    def transcendence_metric(self, state: "State") -> float:
        """
        Compute continuous transcendence measure T(Ψ).

        T(Ψ) = θ / (3π) gives a continuous value in [0, ∞)
        where T ≥ 1 indicates transcendence (layer 3).

        Args:
            state: Input state

        Returns:
            Transcendence metric
        """
        theta = self.compute_phase_quick(state)
        return theta / self.THETA_3

    def layer_progress(self, state: "State") -> Tuple[int, float]:
        """
        Compute current layer and progress toward next layer.

        Args:
            state: Input state

        Returns:
            (layer, progress) where progress ∈ [0, 1)
        """
        theta = self.compute_phase_quick(state)
        layer = self._phase_to_layer(theta)

        # Compute progress within current layer
        if layer == 0:
            progress = theta / self.THETA_1
        elif layer == 1:
            progress = (theta - self.THETA_1) / (self.THETA_2 - self.THETA_1)
        elif layer == 2:
            progress = (theta - self.THETA_2) / (self.THETA_3 - self.THETA_2)
        else:
            # Layer 3: progress beyond transcendence
            progress = (theta - self.THETA_3) / self.THETA_3

        return layer, min(progress, 1.0)

    def k_d4_cycles(self, state: "State") -> float:
        """
        Compute effective K(D₄) cycles.

        The D₄ kissing number K = 24 relates to the transcendence
        threshold via θ_c = 3π = K/8 cycles.

        Args:
            state: Input state

        Returns:
            Number of effective K(D₄) cycles
        """
        theta = self.compute_phase_quick(state)
        # θ / π gives half-cycles, divide by 8 for K(D₄) normalization
        return theta / math.pi * (K_D4 / 24)

    def layer_name(self, layer: int) -> str:
        """Get human-readable layer name."""
        names = {
            0: "nascent",
            1: "emergent",
            2: "coherent",
            3: "transcendent",
        }
        return names.get(layer, "unknown")

    def describe(self, state: "State") -> str:
        """
        Generate human-readable gnosis description.

        Args:
            state: Input state

        Returns:
            Description string
        """
        layer, progress = self.layer_progress(state)
        name = self.layer_name(layer)
        T = self.transcendence_metric(state)

        return f"Gnosis Layer {layer} ({name}), " f"progress={progress:.1%}, T={T:.3f}"

    def __repr__(self) -> str:
        return f"GnosisComputer(max_iter={self.max_iterations})"


def default_gnosis_computer() -> GnosisComputer:
    """
    Create a gnosis computer with default settings.

    Returns:
        GnosisComputer with standard parameters
    """
    return GnosisComputer()
