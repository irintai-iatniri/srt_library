"""
Syntonic Neural Networks - Pure CRT-Native Deep Learning.

BREAKING CHANGE: This module has been purified to use ResonantTensor
and Retrocausal RES instead of PyTorch dependencies.

Key Features:
- DHSR-structured layers (DifferentiationLayer, HarmonizationLayer, RecursionBlock)
- Pure syntonic loss functions (SyntonicLoss with syntony penalty)
- Retrocausal RES training (replaces gradient-based optimizers)
- Complete architectures (PureSyntonicMLP, PureResonantTransformer)
- Archonic pattern detection and escape mechanisms

Core Insight:
    L_total = L_task + λ(1 - S_model) + μC_{iπ}

Where S_model is the model syntony, optimizing for both
task performance AND representational coherence.

Source: CRT.md §12.2
"""

# Core Tensor (Python wrapper around Rust ResonantTensor)
# Constants
import math

# Activations
from .golden_gelu import GoldenGELU

# Layers (Pure Rust-backed)
from .layers import (
    AdaptiveGate,
    DeepRecursionNet,
    DifferentiationLayer,
    DifferentiationModule,
    GoldenNorm,
    HarmonizationLayer,
    MultiHeadSyntonicAttention,
    PixelShuffle,
    PositionalEncoding,
    RecursionBlock,
    ResonantLinear,
    SyntonicAttention,
    SyntonicConv1d,
    SyntonicConv2d,
    SyntonicConvTranspose2d,
    SyntonicEmbedding,
    SyntonicGate,
    SyntonicGRU,
    SyntonicGRUCell,
    SyntonicLSTM,
    SyntonicLSTMCell,
    SyntonicNorm,
    Upsample,
    WindingEmbedding,
)
from . import functional as F
from .loss.phase_alignment import (
    GoldenPhaseScheduler,
    IPiConstraint,
    PhaseAlignmentLoss,
    compute_phase_alignment,
)
from .loss.regularization import (
    ArchonicPenalty,
    SyntonicRegularizer,
    SyntonyConstraint,
    compute_sparsity_penalty,
    compute_weight_decay,
)

# Pure Loss Functions (ResonantTensor-based)
from .loss.syntonic_loss import (
    LayerwiseSyntonicLoss,
    SyntonicLoss,
    cross_entropy_loss,
    mse_loss,
)
from .loss.syntony_metrics import (
    SyntonyTracker,
    aggregate_syntonies,
    compute_activation_syntony,
)

# Retrocausal RES Optimization (replaces gradient-based optimizers)
from .optim import (
    RESConfig,
    ResonantEvolver,
    RESResult,
    RetrocausalConfig,
    compare_convergence,
    create_retrocausal_evolver,
    create_standard_evolver,
)
from .resonant_tensor import ResonantTensor

# Pure Training
from .training.trainer import (
    RESTrainingConfig,
    RetrocausalTrainer,
    train_with_retrocausal_res,
)

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920071658
S_TARGET = PHI - Q_DEFICIT

__all__ = [
    # Core Tensor
    "ResonantTensor",
    # Activations
    "GoldenGELU",
    # Layers
    "DifferentiationLayer",
    "DifferentiationModule",
    "HarmonizationLayer",
    "SyntonicGate",
    "AdaptiveGate",
    "RecursionBlock",
    "DeepRecursionNet",
    "SyntonicNorm",
    "GoldenNorm",
    "ResonantLinear",
    # Conv
    "SyntonicConv1d",
    "SyntonicConv2d",
    "SyntonicConvTranspose2d",
    # Embedding
    "SyntonicEmbedding",
    "PositionalEncoding",
    "WindingEmbedding",
    # Attention
    "SyntonicAttention",
    "MultiHeadSyntonicAttention",
    # Recurrent
    "SyntonicGRUCell",
    "SyntonicGRU",
    "SyntonicLSTMCell",
    "SyntonicLSTM",
    # Pixel ops
    "PixelShuffle",
    "Upsample",
    # Functional
    "F",
    # Pure Loss
    "SyntonicLoss",
    "LayerwiseSyntonicLoss",
    "mse_loss",
    "cross_entropy_loss",
    "SyntonyTracker",
    "aggregate_syntonies",
    "compute_activation_syntony",
    "PhaseAlignmentLoss",
    "compute_phase_alignment",
    "IPiConstraint",
    "GoldenPhaseScheduler",
    "SyntonicRegularizer",
    "SyntonyConstraint",
    "ArchonicPenalty",
    "compute_weight_decay",
    "compute_sparsity_penalty",
    # Retrocausal RES (replaces gradient optimizers)
    "RetrocausalConfig",
    "create_retrocausal_evolver",
    "create_standard_evolver",
    "compare_convergence",
    "ResonantEvolver",
    "RESConfig",
    "RESResult",
    # Pure Training
    "RetrocausalTrainer",
    "RESTrainingConfig",
    "train_with_retrocausal_res",
    # Constants
    "PHI",
    "Q_DEFICIT",
    "S_TARGET",
]
