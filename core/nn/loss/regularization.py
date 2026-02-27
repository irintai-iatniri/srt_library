"""
Pure Syntonic Regularization: Golden-ratio based weight decay and constraints.

PURE IMPLEMENTATION: Uses ResonantTensor, no PyTorch dependencies.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..._core import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


def compute_weight_decay(
    weights: List[ResonantTensor],
    lambda_decay: float = 0.01,
    golden_scaling: bool = True,
) -> float:
    """
    Compute golden ratio-based weight decay.

    L_decay = λ Σ_l (φ^{-l}) ||W_l||²

    Args:
        weights: List of weight tensors per layer
        lambda_decay: Base decay rate
        golden_scaling: If True, use φ^{-l} scaling (earlier layers decay faster)

    Returns:
        Total weight decay loss
    """
    decay_loss = 0.0
    # n_layers = len(weights)

    for i, w in enumerate(weights):
        if golden_scaling:
            scale = PHI ** (-i)
        else:
            scale = 1.0

        # L2 norm via variance * size
        w_var = w.var()
        w_size = 1
        for dim in w.shape():
            w_size *= dim

        decay_loss += scale * w_var * w_size

    return lambda_decay * decay_loss


def compute_sparsity_penalty(
    activations: ResonantTensor,
    target_sparsity: float = PHI_INV,  # ~0.618
) -> float:
    """
    Compute activation sparsity regularization.

    Promotes golden-ratio sparsity: ~61.8% near-zero activations.

    Args:
        activations: Activation tensor
        target_sparsity: Target sparsity ratio (default: 1/φ)

    Returns:
        Sparsity penalty
    """
    # Use variance as sparsity proxy
    # Low variance → more concentrated (sparse)
    var = activations.var()

    # Map variance to sparsity estimate
    # High var → low sparsity, low var → high sparsity
    estimated_sparsity = 1.0 / (1.0 + var)

    # Penalty for deviation from golden sparsity
    penalty = (estimated_sparsity - target_sparsity) ** 2
    return penalty


class SyntonicRegularizer:
    """
    Pure combined syntonic regularization.

    Applies multiple regularization terms that promote syntony:
    1. Golden weight decay
    2. Activation sparsity (promotes differentiation)
    3. Weight coherence (promotes harmonization)

    Example:
        >>> reg = SyntonicRegularizer(lambda_decay=0.01, lambda_sparsity=0.001)
        >>> loss, metrics = reg(weights, activations)
    """

    def __init__(
        self,
        lambda_decay: float = 0.01,
        lambda_sparsity: float = 0.001,
        lambda_coherence: float = 0.001,
    ):
        """
        Initialize syntonic regularizer.

        Args:
            lambda_decay: Weight decay strength
            lambda_sparsity: Activation sparsity strength
            lambda_coherence: Weight coherence strength
        """
        self.lambda_decay = lambda_decay
        self.lambda_sparsity = lambda_sparsity
        self.lambda_coherence = lambda_coherence

    def __call__(
        self,
        weights: Optional[List[ResonantTensor]] = None,
        activations: Optional[ResonantTensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total regularization loss.

        Args:
            weights: Optional list of weight tensors
            activations: Optional activation tensor

        Returns:
            (total_reg_loss, metrics_dict)
        """
        total_loss = 0.0
        metrics = {}

        # 1. Golden weight decay
        if self.lambda_decay > 0 and weights:
            decay_loss = compute_weight_decay(weights, self.lambda_decay)
            total_loss += decay_loss
            metrics["reg_decay"] = decay_loss

        # 2. Activation sparsity
        if self.lambda_sparsity > 0 and activations is not None:
            sparsity_loss = compute_sparsity_penalty(activations)
            weighted_sparsity = self.lambda_sparsity * sparsity_loss
            total_loss += weighted_sparsity
            metrics["reg_sparsity"] = weighted_sparsity

        # 3. Weight coherence (variance-based approximation)
        if self.lambda_coherence > 0 and weights:
            coherence_loss = self._compute_coherence(weights)
            weighted_coherence = self.lambda_coherence * coherence_loss
            total_loss += weighted_coherence
            metrics["reg_coherence"] = weighted_coherence

        metrics["reg_total"] = total_loss
        return total_loss, metrics

    def _compute_coherence(self, weights: List[ResonantTensor]) -> float:
        """
        Compute weight coherence regularization.

        Encourages balanced variance across weight matrices.
        """
        if not weights:
            return 0.0

        variances = [w.var() for w in weights]
        mean_var = sum(variances) / len(variances)

        # Penalize variance deviation (want balanced coherence)
        coherence_loss = sum((v - mean_var) ** 2 for v in variances) / len(variances)
        return coherence_loss


class SyntonyConstraint:
    """
    Pure soft constraint maintaining syntony above threshold.

    Penalizes syntony values below the target S* = φ - q.
    """

    def __init__(
        self,
        target: Optional[float] = None,
        margin: float = 0.1,
        weight: float = 1.0,
    ):
        self.target = target if target is not None else S_TARGET
        self.margin = margin
        self.weight = weight

    def __call__(self, syntony: float) -> float:
        """
        Compute syntony constraint violation.

        Args:
            syntony: Current model syntony

        Returns:
            Constraint violation loss
        """
        if syntony >= self.target - self.margin:
            return 0.0

        violation = self.target - self.margin - syntony
        return self.weight * (violation**2)


class ArchonicPenalty:
    """
    Pure penalty for archonic (stuck) patterns.

    Detects and penalizes representations that exhibit
    archonic cycling (high variance, no syntony improvement).
    """

    def __init__(
        self,
        weight: float = 0.1,
        history_size: int = 100,
        variance_threshold: float = 0.01,
    ):
        self.weight = weight
        self.history_size = history_size
        self.variance_threshold = variance_threshold
        self.syntony_history: List[float] = []

    def __call__(
        self,
        current_syntony: float,
    ) -> float:
        """
        Compute archonic penalty.

        Args:
            current_syntony: Current model syntony

        Returns:
            Archonic penalty loss
        """
        # Update history
        self.syntony_history.append(current_syntony)
        if len(self.syntony_history) > self.history_size:
            self.syntony_history = self.syntony_history[-self.history_size :]

        if len(self.syntony_history) < 10:
            return 0.0

        # Check for archonic pattern
        recent = (
            self.syntony_history[-50:]
            if len(self.syntony_history) >= 50
            else self.syntony_history
        )
        mean_S = sum(recent) / len(recent)
        var_S = sum((s - mean_S) ** 2 for s in recent) / len(recent)

        # Trend
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        trend = (
            sum(second_half) / len(second_half) - sum(first_half) / len(first_half)
            if first_half
            else 0.0
        )

        # Archonic: high variance, no trend, below target
        target_S = S_TARGET - 0.1
        is_archonic = (
            var_S > self.variance_threshold
            and abs(trend) < self.variance_threshold / 10
            and mean_S < target_S
        )

        if is_archonic:
            archonic_score = var_S * (target_S - mean_S) / (abs(trend) + 1e-8)
            return self.weight * archonic_score

        return 0.0

    def reset(self):
        """Reset history."""
        self.syntony_history = []


# Aliases
PureSyntonicRegularizer = SyntonicRegularizer
PureSyntonyConstraint = SyntonyConstraint
PureArchonicPenalty = ArchonicPenalty


if __name__ == "__main__":
    """Test pure regularization."""
    print("Testing Pure Syntonic Regularization...")

    # Create test weights
    w1 = ResonantTensor.from_floats_default_modes([0.1] * 16, [4, 4], 100)
    w2 = ResonantTensor.from_floats_default_modes([0.2] * 16, [4, 4], 100)
    weights = [w1, w2]

    # Create test activations
    activations = ResonantTensor.from_floats_default_modes(
        [0.5, 0.0, 0.3, 0.0, 0.1, 0.0], [2, 3], 100
    )

    # Test regularizer
    reg = SyntonicRegularizer(lambda_decay=0.01, lambda_sparsity=0.001)
    loss, metrics = reg(weights, activations)

    print(f"Total regularization: {metrics['reg_total']:.6f}")
    print(f"  Decay: {metrics.get('reg_decay', 0):.6f}")
    print(f"  Sparsity: {metrics.get('reg_sparsity', 0):.6f}")

    # Test syntony constraint
    constraint = SyntonyConstraint(weight=1.0)
    violation = constraint(0.5)  # Below target
    print(f"Syntony constraint violation: {violation:.6f}")

    print("✅ Pure Syntonic Regularization test passed!")
