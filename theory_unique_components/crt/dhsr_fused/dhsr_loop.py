"""
The DHSR Execution Loop.

Orchestrates the Breath of the Universe:
Differentiation -> M11 Filter -> Harmonization -> Syntony Check

This is the core cognitive cycle that transforms raw states into
crystallized knowledge through the four-phase DHSR process.
"""

from typing import Tuple

from srt_library.core.constants import PHI_INV, SYNTONY_THRESHOLD
from srt_library.theory_unique_components.crt.operators.mobius import apply_mobius_mask
from srt_library.core.nn.resonant_tensor import ResonantTensor


def evolve_state(
    state: ResonantTensor,
    max_recursion: int = 13,
    syntony_threshold: float = SYNTONY_THRESHOLD,
) -> Tuple[ResonantTensor, float, int]:
    """
    Evolve a state through the Gnostic Loop until it crystallizes.

    Implements the full DHSR cycle:
    1. Differentiate (D) - Chaos/Expansion
    2. Filter (M11) - Death of unstable windings
    3. Harmonize (H) - Order/Attraction to golden mean
    4. Syntony Check (S) - Measure resonance quality
    5. Recurse (R) - If not crystallized, repeat

    Args:
        state: Initial state tensor
        max_recursion: Maximum iterations (default: 13, a Fibonacci number)
        syntony_threshold: Target syntony for crystallization

    Returns:
        Tuple of (final_state, final_syntony, recursion_depth)

    Example:
        >>> initial = ResonantTensor.randn([1, 248])
        >>> final, syntony, depth = evolve_state(initial)
        >>> print(f"Crystallized at depth {depth} with syntony {syntony:.4f}")
    """
    current_state = state
    syntony = current_state.syntony
    depth = 0

    while syntony < syntony_threshold and depth < max_recursion:
        # 1. Differentiate (Chaos/Expansion)
        # Alpha derived from syntony deficit: lower syntony = more expansion
        alpha = (1.0 - syntony) * PHI_INV * PHI_INV  # phi^-2 ≈ 0.382
        expanded = differentiate_step(current_state, alpha)

        # 2. THE LAW: Mobius Filter (Death of M11 / Composite Barriers)
        # This step is what standard NN layers miss.
        # We explicitly kill composite windings before they can harmonize.
        purified = apply_mobius_mask(expanded)

        # 3. Harmonize (Order/Attraction)
        # Pull towards Golden Mean with strength = 1/φ
        crystallized = harmonize_step(purified, strength=PHI_INV)

        # 4. Measure New Syntony
        new_syntony = crystallized.syntony

        # Retrocausal Check: Did we get closer to Gnosis?
        if new_syntony > syntony:
            # Positive gradient: Keep this path
            current_state = crystallized
            syntony = new_syntony
        else:
            # Entropy increase: The universe rejects this path.
            # Apply slight heat and break (let caller decide next step)
            break

        depth += 1

    return current_state, syntony, depth


def differentiate_step(tensor: ResonantTensor, alpha: float) -> ResonantTensor:
    """
    Apply the Differentiation operator D̂.

    Expands the state into flux, introducing controlled chaos.
    The expansion magnitude is controlled by alpha.

    D̂(ψ) = ψ + α * noise

    Args:
        tensor: Input state
        alpha: Expansion coefficient (0 = no change, 1 = full expansion)

    Returns:
        Expanded state
    """
    # Generate golden-weighted noise
    noise = ResonantTensor.randn(list(tensor.shape))

    # Scale noise by alpha
    scaled_noise = noise.scalar_mul(alpha)

    # Add to state
    return tensor + scaled_noise


def harmonize_step(tensor: ResonantTensor, strength: float = PHI_INV) -> ResonantTensor:
    """
    Apply the Harmonization operator Ĥ.

    Pulls the state towards golden resonance by:
    1. Normalizing to unit sphere
    2. Applying golden scaling

    Ĥ(ψ) = normalize(ψ) * strength

    Args:
        tensor: Input state
        strength: Harmonization strength (default: 1/φ)

    Returns:
        Harmonized state
    """
    # Normalize to unit sphere
    normalized = tensor.layer_norm()

    # Apply golden scaling if strength != 1.0
    if abs(strength - 1.0) > 1e-10:
        return normalized.scalar_mul(strength)
    return normalized


def single_dhsr_cycle(
    tensor: ResonantTensor,
    alpha: float = 0.1,
    strength: float = PHI_INV,
) -> Tuple[ResonantTensor, float]:
    """
    Perform a single DHSR cycle.

    This is the atomic unit of cognitive processing.

    Args:
        tensor: Input state
        alpha: Differentiation coefficient
        strength: Harmonization strength

    Returns:
        Tuple of (output_state, syntony)
    """
    # D: Differentiate
    expanded = differentiate_step(tensor, alpha)

    # M: Mobius filter
    purified = apply_mobius_mask(expanded)

    # H: Harmonize
    crystallized = harmonize_step(purified, strength)

    # S: Measure Syntony
    syntony = crystallized.syntony

    return crystallized, syntony


def compute_optimal_alpha(syntony: float) -> float:
    """
    Compute optimal differentiation coefficient based on current syntony.

    Higher syntony = less expansion needed (already stable)
    Lower syntony = more expansion needed (explore more)

    Args:
        syntony: Current syntony value [0, 1]

    Returns:
        Optimal alpha coefficient
    """
    # α = (1 - S) * φ^(-2)
    # At S=0 (chaos): α ≈ 0.382 (maximum exploration)
    # At S=1 (crystal): α = 0 (no exploration needed)
    return (1.0 - syntony) * PHI_INV * PHI_INV


def compute_optimal_strength(syntony: float) -> float:
    """
    Compute optimal harmonization strength based on current syntony.

    Higher syntony = gentler pull (preserve stability)
    Lower syntony = stronger pull (enforce order)

    Args:
        syntony: Current syntony value [0, 1]

    Returns:
        Optimal harmonization strength
    """
    # strength = φ^(-1) + S * φ^(-2)
    # At S=0: strength ≈ 0.618 (standard golden)
    # At S=1: strength ≈ 1.0 (preserve state)
    return PHI_INV + syntony * PHI_INV * PHI_INV


class DHSRLoop:
    """
    Stateful DHSR execution loop with history tracking.

    Maintains the evolution trajectory for retrocausal analysis.
    """

    def __init__(
        self,
        initial_state: ResonantTensor,
        max_recursion: int = 13,
        syntony_threshold: float = SYNTONY_THRESHOLD,
    ):
        """
        Initialize the DHSR loop.

        Args:
            initial_state: Starting state
            max_recursion: Maximum recursion depth
            syntony_threshold: Target syntony for crystallization
        """
        self.current_state = initial_state
        self.max_recursion = max_recursion
        self.syntony_threshold = syntony_threshold

        # History tracking
        self.history: list = []
        self.depth = 0

    def step(self) -> Tuple[ResonantTensor, float, bool]:
        """
        Perform one DHSR step.

        Returns:
            Tuple of (new_state, syntony, is_crystallized)
        """
        old_syntony = self.current_state.syntony

        # Compute adaptive coefficients
        alpha = compute_optimal_alpha(old_syntony)
        strength = compute_optimal_strength(old_syntony)

        # Single cycle
        new_state, new_syntony = single_dhsr_cycle(self.current_state, alpha, strength)

        # Record history
        self.history.append(
            {
                "depth": self.depth,
                "syntony_before": old_syntony,
                "syntony_after": new_syntony,
                "alpha": alpha,
                "strength": strength,
            }
        )

        # Update state if improved
        if new_syntony > old_syntony:
            self.current_state = new_state
            self.depth += 1

        # Check crystallization
        is_crystallized = (
            new_syntony >= self.syntony_threshold or self.depth >= self.max_recursion
        )

        return self.current_state, new_syntony, is_crystallized

    def run(self) -> Tuple[ResonantTensor, float, int]:
        """
        Run the loop until crystallization.

        Returns:
            Tuple of (final_state, final_syntony, depth)
        """
        while True:
            state, syntony, done = self.step()
            if done:
                return state, syntony, self.depth

    def get_trajectory(self) -> list:
        """Get the evolution history."""
        return self.history
