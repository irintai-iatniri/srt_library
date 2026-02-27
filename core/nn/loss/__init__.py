"""
Pure Syntonic Loss Functions - Task + Syntony optimization.

PURE IMPLEMENTATION: Uses ResonantTensor, no PyTorch dependencies.

L_total = L_task + λ(1 - S_model) + μC_{iπ}
"""

from ..loss.phase_alignment import (
    GoldenPhaseScheduler,
    IPiConstraint,
    PhaseAlignmentLoss,
    compute_phase_alignment,
)
from ..loss.regularization import (
    ArchonicPenalty,
    SyntonicRegularizer,
    SyntonyConstraint,
    compute_sparsity_penalty,
    compute_weight_decay,
)
from ..loss.syntonic_loss import (
    LayerwiseSyntonicLoss,
    SyntonicLoss,
    cross_entropy_loss,
    mse_loss,
)
from ..loss.syntony_metrics import (
    SyntonyTracker,
    aggregate_syntonies,
    compute_activation_syntony,
)

__all__ = [
    # Syntony metrics
    "compute_activation_syntony",
    "aggregate_syntonies",
    "SyntonyTracker",
    # Loss functions
    "SyntonicLoss",
    "LayerwiseSyntonicLoss",
    "mse_loss",
    "cross_entropy_loss",
    # Phase alignment
    "PhaseAlignmentLoss",
    "compute_phase_alignment",
    "IPiConstraint",
    "GoldenPhaseScheduler",
    # Regularization
    "SyntonicRegularizer",
    "SyntonyConstraint",
    "ArchonicPenalty",
    "compute_weight_decay",
    "compute_sparsity_penalty",
]
