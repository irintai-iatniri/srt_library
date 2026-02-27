"""
Syntonic Neural Network Layers - DHSR-structured layers.

This module provides the fundamental building blocks for syntonic neural networks:
- DifferentiationLayer: D̂ operator expanding complexity
- HarmonizationLayer: Ĥ operator building coherence
- RecursionBlock: Complete R̂ = Ĥ ∘ D̂ cycle
- SyntonicGate: Adaptive mixing based on syntony
- SyntonicNorm: Golden-ratio aware normalization
- ResonantLinear: Linear layer in Q(φ)
"""

from ..activations.gnosis_gelu import GnosisGELU
from ..layers.differentiation import (
    DifferentiationLayer,
    DifferentiationModule,
)
from ..layers.gnosis import GnosisLayer
from ..layers.harmonization import HarmonizationLayer
from ..layers.normalization import (
    GoldenNorm,
    SyntonicNorm,
)
from ..layers.prime_syntony_gate import (
    PrimeSyntonyGate,
    SRTTransformerBlock,
    WindingAttention,
    get_stable_dimensions,
    suggest_network_dimensions,
)
from ..layers.recursion import (
    DeepRecursionNet,
    RecursionBlock,
)
from ..layers.resonant_linear import ResonantLinear
from ..layers.syntonic_gate import (
    AdaptiveGate,
    SyntonicGate,
)

from ..layers.attention import (
    MultiHeadSyntonicAttention,
    SyntonicAttention,
)
from ..layers.conv import (
    SyntonicConv1d,
    SyntonicConv2d,
    SyntonicConvTranspose2d,
)
from ..layers.embedding import (
    PositionalEncoding,
    SyntonicEmbedding,
    WindingEmbedding,
)
from ..layers.pixel_ops import (
    PixelShuffle,
    Upsample,
)
from ..layers.recurrent import (
    SyntonicGRU,
    SyntonicGRUCell,
    SyntonicLSTM,
    SyntonicLSTMCell,
)

__all__ = [
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
    "GnosisLayer",
    "GnosisGELU",
    "PrimeSyntonyGate",
    "WindingAttention",
    "SRTTransformerBlock",
    "get_stable_dimensions",
    "suggest_network_dimensions",
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
]
