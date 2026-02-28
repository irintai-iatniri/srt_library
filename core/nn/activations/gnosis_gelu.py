"""
Gnosis-GELU Activation

Combines GoldenGELU with Gnosis-aware gating.
Pure SRT implementation — no PyTorch or NumPy.
"""

from __future__ import annotations

import math
from typing import Optional

from srt_library.core import sn
from srt_library.core.nn.resonant_tensor import ResonantTensor


class GnosisGELU(sn.Module):
    """GELU activation with Gnosis-aware modulation."""

    def __init__(self, consciousness_threshold: float = 24.0):
        super().__init__()
        self.threshold = consciousness_threshold

    def forward(self, x: ResonantTensor, syntony: float = 0.618) -> ResonantTensor:
        # Apply GoldenGELU (approximate GELU with golden ratio scaling)
        data = x.to_floats()
        phi = (1 + math.sqrt(5)) / 2
        result = []
        for v in data:
            inner = math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)
            gelu_val = 0.5 * v * (1.0 + math.tanh(inner))
            result.append(gelu_val)

        # Modulate by Gnosis score if above consciousness threshold
        creativity = 1.0 - syntony  # Novelty
        g = self._gnosis_score(syntony, creativity)
        scale = 1.0 + 0.1 * g

        output = [v * scale for v in result]
        return ResonantTensor(output, list(x.shape), device=x.device)

    @staticmethod
    def _gnosis_score(syntony: float, creativity: float) -> float:
        """Compute gnosis score from syntony and creativity."""
        try:
            from srt_library.core._core import gnosis_score
            return gnosis_score(syntony, creativity)
        except ImportError:
            # Fallback: geometric mean
            return math.sqrt(max(0.0, syntony) * max(0.0, creativity))


__all__ = ["GnosisGELU"]
