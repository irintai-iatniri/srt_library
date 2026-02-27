"""
Gnosis-GELU Activation

Combines GoldenGELU with Gnosis-aware gating.
"""

import torch
import torch.nn as nn
from ..._core import (
    gnosis_score,
    golden_gelu_forward,
)


class GnosisGELU(nn.Module):
    """GELU activation with Gnosis-aware modulation."""

    def __init__(self, consciousness_threshold: float = 24.0):
        super().__init__()
        self.threshold = consciousness_threshold

    def forward(self, x: torch.Tensor, syntony: float = 0.618) -> torch.Tensor:
        # Apply GoldenGELU
        output = golden_gelu_forward(x.numpy())

        # Modulate by Gnosis score if above consciousness threshold
        creativity = 1.0 - syntony  # Novelty
        g = gnosis_score(syntony, creativity)

        return torch.from_numpy(output) * (1.0 + 0.1 * g)


__all__ = ["GnosisGELU"]
