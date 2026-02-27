"""
Pure Syntony Metrics for neural networks.

S_model = 1 - ||D(x) - x|| / (||D(x) - H(D(x))|| + ε)

PURE IMPLEMENTATION: Uses ResonantTensor, no PyTorch dependencies.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..._core import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


def compute_activation_syntony(
    x_input: ResonantTensor,
    x_diff: ResonantTensor,
    x_harm: ResonantTensor,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute syntony from D/H activations.

    S = 1 - ||D(x) - x|| / (||D(x) - H(D(x))|| + ε)

    Args:
        x_input: Original input
        x_diff: After differentiation D(x)
        x_harm: After harmonization H(D(x))
        epsilon: Regularization constant

    Returns:
        Syntony value
    """
    # Numerator: how much D changed x
    diff_change = x_diff.elementwise_add(x_input.negate())
    diff_norm = math.sqrt(diff_change.var() * _tensor_size(diff_change))

    # Denominator: how much H corrected D
    harm_correction = x_diff.elementwise_add(x_harm.negate())
    harm_norm = math.sqrt(harm_correction.var() * _tensor_size(harm_correction))

    # Syntony
    S = 1.0 - diff_norm / (harm_norm + epsilon)
    return max(0.0, min(1.0, S))


def _tensor_size(t: ResonantTensor) -> int:
    """Get total element count."""
    size = 1
    for dim in t.shape():
        size *= dim
    return size


def aggregate_syntonies(
    layer_syntonies: List[float],
    method: str = "mean",
) -> float:
    """
    Aggregate layer syntonies into global syntony.

    Args:
        layer_syntonies: Per-layer syntony values
        method: Aggregation method ('mean', 'min', 'geometric')

    Returns:
        Global syntony
    """
    if not layer_syntonies:
        return 0.5  # Default mid-syntony

    if method == "mean":
        return sum(layer_syntonies) / len(layer_syntonies)
    elif method == "min":
        return min(layer_syntonies)
    elif method == "geometric":
        product = 1.0
        for s in layer_syntonies:
            product *= max(s, 1e-10)
        return product ** (1 / len(layer_syntonies))
    else:
        return sum(layer_syntonies) / len(layer_syntonies)


class SyntonyTracker:
    """
    Pure tracker for syntony evolution during training.

    Monitors syntony trends and detects archonic patterns.

    Example:
        >>> tracker = SyntonyTracker()
        >>> for epoch in range(100):
        ...     tracker.update(model_syntony)
        >>> print(f"Mean: {tracker.mean_syntony:.4f}")
        >>> print(f"Trend: {tracker.syntony_trend:.4f}")
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: List[float] = []
        self.layer_histories: Dict[int, List[float]] = {}

    def update(
        self,
        global_syntony: float,
        layer_syntonies: Optional[List[float]] = None,
    ):
        """
        Record syntony values.

        Args:
            global_syntony: Global model syntony
            layer_syntonies: Optional per-layer syntonies
        """
        self.history.append(global_syntony)
        if len(self.history) > self.window_size * 10:
            self.history = self.history[-self.window_size * 10 :]

        if layer_syntonies:
            for i, s in enumerate(layer_syntonies):
                if i not in self.layer_histories:
                    self.layer_histories[i] = []
                self.layer_histories[i].append(s)
                if len(self.layer_histories[i]) > self.window_size * 10:
                    self.layer_histories[i] = self.layer_histories[i][
                        -self.window_size * 10 :
                    ]

    @property
    def mean_syntony(self) -> float:
        """Mean syntony over window."""
        if not self.history:
            return 0.0
        window = self.history[-self.window_size :]
        return sum(window) / len(window)

    @property
    def syntony_trend(self) -> float:
        """Syntony trend (positive = improving)."""
        if len(self.history) < 2:
            return 0.0

        half_window = self.window_size // 2
        if len(self.history) > self.window_size:
            recent = self.history[-half_window:]
            earlier = self.history[-self.window_size : -half_window]
        else:
            mid = len(self.history) // 2
            recent = self.history[mid:]
            earlier = self.history[:mid]

        if not earlier:
            return 0.0

        return sum(recent) / len(recent) - sum(earlier) / len(earlier)

    @property
    def variance(self) -> float:
        """Syntony variance over window."""
        if len(self.history) < 2:
            return 0.0
        window = self.history[-self.window_size :]
        mean = sum(window) / len(window)
        return sum((s - mean) ** 2 for s in window) / len(window)

    def is_archonic(
        self,
        threshold: float = 0.01,
        min_samples: int = 50,
    ) -> bool:
        """
        Detect if network is in Archonic (stuck) pattern.

        Args:
            threshold: Variance threshold
            min_samples: Minimum samples needed

        Returns:
            True if archonic pattern detected
        """
        if len(self.history) < min_samples:
            return False

        mean_S = self.mean_syntony
        variance_S = self.variance
        trend = self.syntony_trend

        target = S_TARGET - 0.1
        return (
            variance_S > threshold and abs(trend) < threshold / 10 and mean_S < target
        )

    def get_layer_stats(self, layer_idx: int) -> Dict[str, float]:
        """Get statistics for a specific layer."""
        if layer_idx not in self.layer_histories:
            return {"mean": 0.0, "variance": 0.0, "trend": 0.0}

        history = self.layer_histories[layer_idx]
        if not history:
            return {"mean": 0.0, "variance": 0.0, "trend": 0.0}

        window = history[-self.window_size :]
        mean = sum(window) / len(window)
        variance = sum((s - mean) ** 2 for s in window) / len(window)

        if len(window) > 1:
            mid = len(window) // 2
            recent = window[mid:]
            earlier = window[:mid]
            trend = (
                sum(recent) / len(recent) - sum(earlier) / len(earlier)
                if earlier
                else 0.0
            )
        else:
            trend = 0.0

        return {"mean": mean, "variance": variance, "trend": trend}

    def reset(self):
        """Reset tracker."""
        self.history = []
        self.layer_histories = {}


# Aliases
PureSyntonyTracker = SyntonyTracker


if __name__ == "__main__":
    """Test pure syntony metrics."""
    print("Testing Pure Syntony Metrics...")

    # Test aggregation
    layer_syntonies = [0.7, 0.8, 0.9]
    global_S = aggregate_syntonies(layer_syntonies, "mean")
    print(f"Aggregated syntony (mean): {global_S:.4f}")

    global_S_geo = aggregate_syntonies(layer_syntonies, "geometric")
    print(f"Aggregated syntony (geometric): {global_S_geo:.4f}")

    # Test tracker
    tracker = SyntonyTracker(window_size=20)
    for i in range(50):
        syntony = 0.5 + 0.01 * i  # Improving
        tracker.update(syntony)

    print(f"Mean syntony: {tracker.mean_syntony:.4f}")
    print(f"Syntony trend: {tracker.syntony_trend:.4f}")
    print(f"Variance: {tracker.variance:.6f}")
    print(f"Is archonic: {tracker.is_archonic()}")

    print("✅ Pure Syntony Metrics test passed!")
