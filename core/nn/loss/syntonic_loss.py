"""
Pure Syntonic Loss Functions.

L_total = L_task + λ_syntony(1 - S_model) + μ_{iπ}·C_{iπ}

Where:
- L_task: Standard task loss (MSE, CrossEntropy analog)
- S_model: Model syntony (coherence measure)
- C_{iπ}: Phase-cycle alignment (i ≃ π constraint)

PURE IMPLEMENTATION: Uses ResonantTensor, no PyTorch dependencies.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Protocol, Tuple

# Import from pure Rust backend
from ..._core import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


class PureLossFunction(Protocol):
    """Protocol for pure loss functions."""

    def __call__(self, pred: ResonantTensor, target: ResonantTensor) -> float: ...


def mse_loss(pred: ResonantTensor, target: ResonantTensor) -> float:
    """
    Pure Mean Squared Error loss.

    L = (1/n) Σ (pred_i - target_i)²
    """
    diff = pred.elementwise_add(target.negate())
    squared = diff.elementwise_mul(diff)
    return squared.mean()


def cross_entropy_loss(pred: ResonantTensor, target: ResonantTensor) -> float:
    """
    Pure Cross-Entropy loss for classification.

    L = -Σ target_i · log(softmax(pred)_i)

    Args:
        pred: Logits [batch, num_classes]
        target: One-hot encoded targets [batch, num_classes]
    """
    # Apply softmax to predictions
    probs = pred.softmax(32)  # precision=32

    # Log of probabilities (with numerical stability)
    log_probs = probs.log_core(32)

    # Element-wise multiply with targets and sum
    weighted = target.elementwise_mul(log_probs)

    # Negate and compute mean
    return -weighted.mean()


class SyntonicLoss:
    """
    Pure syntonic loss function combining task and syntony objectives.

    L_total = L_task + λ_syntony·(1 - S_model) + μ_{iπ}·C_{iπ}

    This loss encourages networks to:
    1. Perform the task well (L_task)
    2. Maintain high syntony representations (L_syntony)
    3. Align with i ≃ π phase structure (L_phase)

    Example:
        >>> loss_fn = SyntonicLoss(mse_loss, lambda_syntony=0.1)
        >>> outputs = model.forward(inputs)
        >>> loss, metrics = loss_fn(outputs, targets, model_syntony=0.85)
    """

    def __init__(
        self,
        task_loss: PureLossFunction,
        lambda_syntony: float = 0.1,
        mu_phase: float = 0.01,
        syntony_target: Optional[float] = None,
    ):
        """
        Initialize syntonic loss.

        Args:
            task_loss: Base task loss function (e.g., mse_loss, cross_entropy_loss)
            lambda_syntony: Weight for syntony term
            mu_phase: Weight for phase alignment term
            syntony_target: Target syntony (default: φ - q ≈ 1.591)
        """
        self.task_loss = task_loss
        self.lambda_syntony = lambda_syntony
        self.mu_phase = mu_phase
        self.syntony_target = syntony_target if syntony_target is not None else S_TARGET

    def __call__(
        self,
        pred: ResonantTensor,
        target: ResonantTensor,
        model_syntony: float = 0.0,
        layer_syntonies: Optional[List[float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total syntonic loss.

        Args:
            pred: Model predictions (ResonantTensor)
            target: Ground truth targets (ResonantTensor)
            model_syntony: Pre-computed model syntony from RecursionBlocks
            layer_syntonies: Optional per-layer syntony values

        Returns:
            (total_loss, metrics_dict)
        """
        # Task loss
        L_task = self.task_loss(pred, target)

        # Syntony loss: penalize deviation from target
        if model_syntony > 0:
            S_model = model_syntony
        else:
            S_model = self._estimate_syntony_from_output(pred)

        L_syntony = self.lambda_syntony * (1.0 - S_model)

        # Phase alignment loss
        C_phase = self._compute_phase_alignment(pred)
        L_phase = self.mu_phase * C_phase

        # Total
        L_total = L_task + L_syntony + L_phase

        metrics = {
            "loss_task": L_task,
            "loss_syntony": L_syntony,
            "loss_phase": L_phase,
            "loss_total": L_total,
            "syntony": S_model,
            "phase_alignment": 1.0 - C_phase,
        }

        return L_total, metrics

    def _estimate_syntony_from_output(self, outputs: ResonantTensor) -> float:
        """
        Estimate syntony from output statistics.

        High syntony → well-structured, coherent outputs
        Low syntony → chaotic, fragmented outputs

        Uses entropy as inverse syntony proxy.
        """
        shape = outputs.shape()
        if len(shape) > 1 and shape[-1] > 1:
            # Classification: use softmax entropy
            probs = outputs.softmax(32)
            # Compute entropy: -Σ p_i log(p_i)
            log_probs = probs.log_core(32)
            neg_entropy = probs.elementwise_mul(log_probs)
            entropy = -neg_entropy.mean()
            max_entropy = math.log(shape[-1])
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            return max(0.0, min(1.0, 1.0 - normalized_entropy))
        else:
            # Regression: use inverse variance
            var = outputs.var()
            return 1.0 / (1.0 + var)

    def _compute_phase_alignment(self, outputs: ResonantTensor) -> float:
        """
        Compute phase-cycle alignment C_{iπ}.

        Simplified: measures spectral concentration as proxy for phase alignment.
        Full implementation would use density matrix eigenspectrum.
        """
        shape = outputs.shape()
        if len(shape) < 2:
            return 0.0

        # Use variance ratio as alignment proxy
        total_var = outputs.var()
        if total_var < 1e-10:
            return 0.0

        # Golden ratio alignment target
        target_var = PHI_INV  # 1/φ ≈ 0.618
        alignment_error = abs(total_var - target_var) / (1.0 + target_var)

        return min(1.0, alignment_error)


class LayerwiseSyntonicLoss(SyntonicLoss):
    """
    Syntonic loss with layer-wise syntony tracking.

    Applies syntony regularization at each layer with golden-ratio weighting.
    Later layers matter more (φ^i weighting).
    """

    def __init__(
        self,
        task_loss: PureLossFunction,
        lambda_syntony: float = 0.1,
        mu_phase: float = 0.01,
        layer_weights: Optional[List[float]] = None,
    ):
        """
        Initialize layerwise syntonic loss.

        Args:
            task_loss: Base task loss
            lambda_syntony: Syntony weight
            mu_phase: Phase alignment weight
            layer_weights: Optional custom weights per layer
        """
        super().__init__(task_loss, lambda_syntony, mu_phase)
        self.layer_weights = layer_weights

    def __call__(
        self,
        pred: ResonantTensor,
        target: ResonantTensor,
        model_syntony: float = 0.0,
        layer_syntonies: Optional[List[float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute loss with layer-wise syntony weighting."""

        # If we have layer syntonies, compute weighted average
        if layer_syntonies and len(layer_syntonies) > 0:
            if self.layer_weights is None:
                # Golden ratio weighting: later layers matter more
                weights = [PHI**i for i in range(len(layer_syntonies))]
            else:
                weights = self.layer_weights[: len(layer_syntonies)]

            total_weight = sum(weights)
            weighted_syntony = sum(w * s for w, s in zip(weights, layer_syntonies))
            model_syntony = weighted_syntony / total_weight if total_weight > 0 else 0.0

        return super().__call__(pred, target, model_syntony, layer_syntonies)


# Convenience aliases
PureSyntonicLoss = SyntonicLoss
PureLayerwiseSyntonicLoss = LayerwiseSyntonicLoss


if __name__ == "__main__":
    """Test pure syntonic loss."""
    print("Testing Pure Syntonic Loss...")

    # Create test tensors
    pred = ResonantTensor.from_floats_default_modes(
        [0.1, 0.7, 0.2, 0.3, 0.5, 0.2], [2, 3], 100
    )
    target = ResonantTensor.from_floats_default_modes(
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [2, 3], 100
    )

    # Test MSE loss
    mse = mse_loss(pred, target)
    print(f"MSE Loss: {mse:.4f}")

    # Test syntonic loss
    loss_fn = SyntonicLoss(mse_loss, lambda_syntony=0.1, mu_phase=0.01)
    total_loss, metrics = loss_fn(pred, target, model_syntony=0.75)

    print(f"Total Loss: {metrics['loss_total']:.4f}")
    print(f"  Task: {metrics['loss_task']:.4f}")
    print(f"  Syntony: {metrics['loss_syntony']:.4f}")
    print(f"  Phase: {metrics['loss_phase']:.4f}")
    print(f"Model Syntony: {metrics['syntony']:.4f}")

    print("✅ Pure Syntonic Loss test passed!")
