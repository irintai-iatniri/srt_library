"""
Syntonic Training - Training utilities for syntonic networks.

Provides training loop, callbacks, and metrics for
syntony-aware neural network training.

Uses pure Python + ResonantTensor (no PyTorch dependencies).
"""

# Pure Python implementations (no PyTorch)
from .callbacks import (
    ArchonicEarlyStop,
    Callback,
    FitnessPlateauCallback,
    MetricsLogger,
    SyntonyCallback,
    SyntonyCheckpoint,
    default_callbacks,
)
from .metrics import (
    MetricsAggregator,
    SyntonyMetrics,
    TrainingMetrics,
    check_archonic_pattern,
    compute_accuracy,
    compute_mse,
    compute_syntony_from_weights,
    compute_syntony_gap,
)
from .metrics import (
    SyntonyTracker as SyntonyMetricTracker,
)
from .trainer import (
    RESTrainingConfig,
    RetrocausalTrainer,
    SyntonyTracker,
)
from .gradient_trainer import (
    GradientConfig,
    GradientTrainer,
)

__all__ = [
    # Trainer
    "RetrocausalTrainer",
    "RESTrainingConfig",
    "SyntonyTracker",
    # Gradient Trainer
    "GradientTrainer",
    "GradientConfig",
    # Callbacks
    "Callback",
    "SyntonyCallback",
    "ArchonicEarlyStop",
    "SyntonyCheckpoint",
    "MetricsLogger",
    "FitnessPlateauCallback",
    "default_callbacks",
    # Metrics
    "TrainingMetrics",
    "SyntonyMetrics",
    "MetricsAggregator",
    "SyntonyMetricTracker",
    "compute_syntony_from_weights",
    "compute_accuracy",
    "compute_mse",
    "check_archonic_pattern",
    "compute_syntony_gap",
]
