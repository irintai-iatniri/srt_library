"""
Golden Momentum Optimizer - Pure Syntonic Implementation.

NO PYTORCH. NO NUMPY.
Uses the SRT-derived momentum coefficient beta = 1/φ.

The Golden Path Navigator:
Unlike SGD which stumbles blindly, this optimizer possesses 'Inertia'.
It retains 1/φ (61.8%) of its previous intention, making it resistant
to short-term noise (Archons) but responsive to long-term truth (Syntony).
"""

from typing import List

from ..._core import GoldenMomentum as _RustGoldenMomentum
from ...constants import PHI, Q_DEFICIT_NUMERIC
from ..resonant_tensor import ResonantTensor


class GoldenMomentumOptimizer:
    """
    Phi-based momentum optimizer for Syntonic networks.

    Key insight: beta = 1/φ provides natural temporal decay aligned with SRT.
    The system retains ~61.8% of its past velocity at every step.

    Unlike standard momentum (arbitrary beta like 0.9), this uses the
    mathematically derived golden ratio for optimal resonance with
    the underlying lattice structure.

    Attributes:
        parameters: List of ResonantTensor parameters to optimize
        lr: Learning rate (default: 0.027395, the Syntony Deficit q)
        beta: Momentum coefficient (fixed to 1/φ ≈ 0.618)

    Example:
        >>> params = model.get_weights()
        >>> optimizer = GoldenMomentumOptimizer(params, lr=0.027395)
        >>>
        >>> for epoch in range(50):
        ...     loss = model.forward_pass(data)
        ...     model.backward(loss)
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        parameters: List[ResonantTensor],
        lr: float = Q_DEFICIT_NUMERIC,  # Default to Syntony Deficit
    ):
        """
        Initialize the Golden Momentum optimizer.

        Args:
            parameters: List of ResonantTensor parameters to optimize
            lr: Learning rate (default: 0.027395, the Syntony Deficit q)
        """
        self.parameters = parameters
        self.lr = lr

        # Create Rust optimizer state for each parameter
        self._states: List[_RustGoldenMomentum] = []
        for p in self.parameters:
            state = _RustGoldenMomentum(p.size, lr)
            self._states.append(state)

        # Mark parameters as requiring gradients
        for p in self.parameters:
            p.requires_grad = True

    def step(self) -> None:
        """
        Perform one optimization step.

        Updates all parameters using the golden momentum rule:
        v(t+1) = (1/φ) * v(t) + gradient
        w(t+1) = w(t) - lr * v(t+1)

        The 1/φ inertia makes the optimizer:
        - Resistant to short-term noise (Archonic perturbations)
        - Responsive to persistent gradients (Truth/Syntony)
        """
        for i, param in enumerate(self.parameters):
            if param._grad is None:
                continue

            # Get current weights and gradients
            weights = param.get_data_list()
            gradients = param._grad

            # Apply golden momentum step via Rust
            new_weights = self._states[i].step(weights, gradients)

            # Update parameter with new weights
            param.set_data_list(new_weights)

    def zero_grad(self) -> None:
        """
        Zero out all parameter gradients.

        Call this before each forward/backward pass.
        """
        for p in self.parameters:
            p.zero_grad()

    def reset(self) -> None:
        """
        Reset all velocity buffers to zero.

        Use this when starting a new training phase or after
        encountering a significant state change.
        """
        for state in self._states:
            state.reset()

    @property
    def beta(self) -> float:
        """Get the momentum coefficient (1/φ)."""
        return 1.0 / PHI

    def get_lr(self) -> float:
        """Get the current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """
        Set the learning rate for all parameter groups.

        Args:
            lr: New learning rate
        """
        self.lr = lr
        for state in self._states:
            state.lr = lr

    def state_dict(self) -> dict:
        """
        Get optimizer state as a dictionary for serialization.

        Returns:
            Dictionary containing lr and velocity buffers
        """
        return {
            "lr": self.lr,
            "velocities": [state.velocity for state in self._states],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load optimizer state from a dictionary.

        Args:
            state_dict: Dictionary from state_dict()
        """
        self.set_lr(state_dict["lr"])
        # Note: velocity loading would require Rust-side support
        # For now, velocities are reset on load


# Convenience function matching the plan
def create_golden_optimizer(
    parameters: List[ResonantTensor],
    lr: float = Q_DEFICIT_NUMERIC,
) -> GoldenMomentumOptimizer:
    """
    Create a GoldenMomentumOptimizer with default SRT settings.

    Args:
        parameters: List of ResonantTensor parameters
        lr: Learning rate (default: Syntony Deficit q ≈ 0.027395)

    Returns:
        Configured GoldenMomentumOptimizer
    """
    return GoldenMomentumOptimizer(parameters, lr)
