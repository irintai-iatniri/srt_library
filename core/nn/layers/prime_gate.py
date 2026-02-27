"""
Prime Syntony Gate - Neural Network Layer

Implements Fibonacci Prime resonance boosting for neural networks.
Dimensions matching Fibonacci Prime indices get φ^n boost with normalization.
"""

import torch
import torch.nn as nn

from ..._core import fibonacci_resonance_boost, is_transcendence_gate


class PrimeSyntonyGate(nn.Module):
    """
    Neural layer that applies Fibonacci Prime resonance boosting.

    If input dimension matches a Fibonacci Prime index (transcendence gate),
    normalize to unit sphere and apply φ^n boost.

    Special case: dim=4 (Material Anomaly) gets 0.9× destabilization.

    Based on Grand Synthesis theory: neural networks should resonate
    with ontological transcendence gates.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.is_resonant = is_transcendence_gate(dim)
        self.boost_factor = fibonacci_resonance_boost(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_resonant:
            # Lock to unit sphere ("Syntony 1.0")
            x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
            return x_norm * self.boost_factor
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, resonant={self.is_resonant}, boost={self.boost_factor:.3f}"


__all__ = ["PrimeSyntonyGate"]
