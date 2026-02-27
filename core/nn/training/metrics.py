"""
Pure Training Metrics for Syntonic Networks.

Metrics computation and aggregation for syntony-aware training.
Pure Python - no PyTorch dependencies.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920
S_TARGET = PHI - Q_DEFICIT


@dataclass
class SyntonyMetrics:
    """
    Metrics for tracking syntony.

    Attributes:
        global_syntony: Overall model syntony
        layer_syntonies: Per-layer syntony values
        syntony_trend: Recent syntony trend
        is_archonic: Whether archonic pattern detected
        phase_alignment: Phase alignment with iπ constraint
    """

    global_syntony: float = 0.0
    layer_syntonies: List[float] = field(default_factory=list)
    syntony_trend: float = 0.0
    is_archonic: bool = False
    phase_alignment: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "global_syntony": self.global_syntony,
            "layer_syntonies": self.layer_syntonies,
            "syntony_trend": self.syntony_trend,
            "is_archonic": self.is_archonic,
            "phase_alignment": self.phase_alignment,
        }


@dataclass
class TrainingMetrics:
    """
    Comprehensive training metrics.

    Attributes:
        fitness: RES fitness (negative loss + syntony)
        loss: Total loss
        task_loss: Task-specific loss
        syntony_loss: Syntony regularization loss
        phase_loss: Phase alignment loss
        accuracy: Classification accuracy (if applicable)
        syntony: Syntony metrics
    """

    fitness: float = 0.0
    loss: float = 0.0
    task_loss: float = 0.0
    syntony_loss: float = 0.0
    phase_loss: float = 0.0
    accuracy: Optional[float] = None
    syntony: SyntonyMetrics = field(default_factory=SyntonyMetrics)

    # RES-specific metrics
    generation: int = 0
    population_diversity: float = 0.0
    mutation_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "fitness": self.fitness,
            "loss": self.loss,
            "task_loss": self.task_loss,
            "syntony_loss": self.syntony_loss,
            "phase_loss": self.phase_loss,
            "generation": self.generation,
            "population_diversity": self.population_diversity,
            "mutation_rate": self.mutation_rate,
        }
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy
        result.update(self.syntony.to_dict())
        return result


def compute_syntony_from_weights(
    weights: List[float], mode_norms: Optional[List[float]] = None
) -> float:
    """
    Compute syntony from weight distribution.

    Uses golden weight measure: S = Σ|w_i|²exp(-i²/φ) / Σ|w_i|²

    Args:
        weights: Weight values
        mode_norms: Optional mode norms (defaults to i²)

    Returns:
        Syntony value [0, 1]
    """
    if not weights:
        return 0.5

    # Compute energies
    energies = [w * w for w in weights]
    total_energy = sum(energies)

    if total_energy < 1e-10:
        return 0.5

    # Compute golden weights
    weighted_energy = 0.0
    for i, e in enumerate(energies):
        norm = mode_norms[i] if mode_norms and i < len(mode_norms) else i * i
        golden_weight = math.exp(-norm / PHI)
        weighted_energy += e * golden_weight

    syntony = weighted_energy / total_energy
    return max(0.0, min(1.0, syntony))


def compute_accuracy(predictions: List[int], targets: List[int]) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices

    Returns:
        Accuracy [0, 1]
    """
    if not predictions or not targets:
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def compute_mse(predictions: List[float], targets: List[float]) -> float:
    """
    Compute mean squared error.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        MSE loss
    """
    if not predictions or not targets:
        return 0.0

    n = min(len(predictions), len(targets))
    mse = sum((p - t) ** 2 for p, t in zip(predictions[:n], targets[:n])) / n
    return mse


def compute_cross_entropy(
    probs: List[float], targets: List[int], num_classes: int
) -> float:
    """
    Compute cross-entropy loss.

    Args:
        probs: Softmax probabilities (flattened, batch_size x num_classes)
        targets: Target class indices
        num_classes: Number of classes

    Returns:
        Cross-entropy loss
    """
    if not probs or not targets:
        return 0.0

    batch_size = len(targets)
    epsilon = 1e-10
    total_loss = 0.0

    for i, target in enumerate(targets):
        # Get probability of correct class
        start_idx = i * num_classes
        if start_idx + target < len(probs):
            p = probs[start_idx + target]
            total_loss -= math.log(max(p, epsilon))

    return total_loss / batch_size


def check_archonic_pattern(
    syntonies: List[float],
    window_size: int = 20,
    variance_threshold: float = 0.01,
    trend_threshold: float = 0.001,
) -> bool:
    """
    Check for archonic pattern in syntony sequence.

    An archonic pattern is high variance (cycling) with no net trend.

    Args:
        syntonies: Sequence of syntony values
        window_size: Window for analysis
        variance_threshold: Minimum variance for cycling
        trend_threshold: Maximum trend for stuck pattern

    Returns:
        True if archonic pattern detected
    """
    if len(syntonies) < window_size:
        return False

    recent = syntonies[-window_size:]
    mean_S = sum(recent) / len(recent)
    var_S = sum((s - mean_S) ** 2 for s in recent) / len(recent)

    # Check for high variance (cycling)
    if var_S < variance_threshold:
        return False

    # Check for no trend (stuck)
    mid = len(recent) // 2
    trend = sum(recent[mid:]) / len(recent[mid:]) - sum(recent[:mid]) / len(
        recent[:mid]
    )

    # Archonic if cycling (high var) but stuck (low trend)
    return abs(trend) < trend_threshold and mean_S < S_TARGET - 0.1


def compute_syntony_gap(
    current_syntony: float,
    target: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute gap between current and target syntony.

    Args:
        current_syntony: Current model syntony
        target: Target syntony (default: φ - q)

    Returns:
        Dictionary with gap metrics
    """
    target = target if target is not None else S_TARGET

    gap = target - current_syntony
    relative_gap = gap / target if target != 0 else 0.0

    return {
        "syntony_gap": gap,
        "syntony_relative_gap": relative_gap,
        "syntony_ratio": current_syntony / target if target != 0 else 0.0,
        "target": target,
    }


def compute_population_diversity(population: List[List[float]]) -> float:
    """
    Compute diversity of a population of weight vectors.

    Uses mean pairwise distance.

    Args:
        population: List of weight vectors

    Returns:
        Diversity measure [0, ∞)
    """
    if len(population) < 2:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Euclidean distance
            dist = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(population[i], population[j]))
            )
            total_distance += dist
            count += 1

    return total_distance / count if count > 0 else 0.0


def compute_weight_statistics(weights: List[float]) -> Dict[str, float]:
    """
    Compute statistics for weight values.

    Args:
        weights: Weight values

    Returns:
        Dictionary with statistics
    """
    if not weights:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "norm": 0.0}

    mean = sum(weights) / len(weights)
    var = sum((w - mean) ** 2 for w in weights) / len(weights)
    std = math.sqrt(var)
    norm = math.sqrt(sum(w * w for w in weights))

    return {
        "mean": mean,
        "std": std,
        "max": max(weights),
        "min": min(weights),
        "norm": norm,
    }


class MetricsAggregator:
    """
    Aggregate metrics over multiple batches/generations.

    Example:
        >>> aggregator = MetricsAggregator()
        >>> for gen in range(100):
        ...     metrics = compute_generation_metrics()
        ...     aggregator.update(metrics)
        >>> summary = aggregator.compute()
    """

    def __init__(self):
        """Initialize aggregator."""
        self._metrics: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Add batch metrics.

        Args:
            metrics: Dictionary of metric values
            count: Number of samples in batch
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self._metrics:
                    self._metrics[key] = []
                    self._counts[key] = 0
                self._metrics[key].append(value * count)
                self._counts[key] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute aggregated metrics.

        Returns:
            Dictionary of mean metric values
        """
        result = {}
        for key, values in self._metrics.items():
            total = sum(values)
            count = self._counts[key]
            result[key] = total / count if count > 0 else 0.0
        return result

    def reset(self):
        """Reset aggregator."""
        self._metrics = {}
        self._counts = {}


class SyntonyTracker:
    """
    Track syntony over time with trend analysis.

    Provides rolling statistics and archonic detection.
    """

    def __init__(self, window_size: int = 50):
        """
        Initialize tracker.

        Args:
            window_size: Window for rolling statistics
        """
        self.window_size = window_size
        self._history: List[float] = []

    def add(self, syntony: float):
        """Add a syntony value."""
        self._history.append(syntony)

    def get_trend(self) -> float:
        """Get recent trend (positive = improving)."""
        if len(self._history) < 10:
            return 0.0

        recent = (
            self._history[-self.window_size :]
            if len(self._history) >= self.window_size
            else self._history
        )
        mid = len(recent) // 2
        return sum(recent[mid:]) / len(recent[mid:]) - sum(recent[:mid]) / len(
            recent[:mid]
        )

    def get_mean(self) -> float:
        """Get mean of recent values."""
        if not self._history:
            return 0.5

        recent = (
            self._history[-self.window_size :]
            if len(self._history) >= self.window_size
            else self._history
        )
        return sum(recent) / len(recent)

    def get_variance(self) -> float:
        """Get variance of recent values."""
        if len(self._history) < 2:
            return 0.0

        recent = (
            self._history[-self.window_size :]
            if len(self._history) >= self.window_size
            else self._history
        )
        mean = sum(recent) / len(recent)
        return sum((s - mean) ** 2 for s in recent) / len(recent)

    def is_archonic(self) -> bool:
        """Check for archonic pattern."""
        return check_archonic_pattern(self._history, self.window_size)

    def get_best(self) -> float:
        """Get best syntony achieved."""
        return max(self._history) if self._history else 0.0

    def get_history(self) -> List[float]:
        """Get full history."""
        return list(self._history)

    def clear(self):
        """Clear history."""
        self._history = []
