"""
Theory Unique Components - SRT/CRT specific modules + architecture implementations.

Pre-built architectures that embed the DHSR cycle throughout:
- SyntonicMLP: Fully-connected with RecursionBlocks
- SyntonicConv: Convolutional with recursion
- CRTTransformer: Transformer with syntonic attention
"""

from .embeddings import (
    PurePositionalEncoding,
    PureSyntonicEmbedding,
    PureWindingEmbedding,
)
from .embeddings import PurePositionalEncoding as PositionalEncoding
from .embeddings import PureSyntonicEmbedding as SyntonicEmbedding
from .embeddings import PureWindingEmbedding as WindingEmbedding

from .syntonic_attention import (
    PureMultiHeadSyntonicAttention,
    PureSyntonicAttention,
)
from .syntonic_attention import PureMultiHeadSyntonicAttention as MultiHeadSyntonicAttention
from .syntonic_attention import PureSyntonicAttention as GnosisAttention
from .syntonic_attention import PureSyntonicAttention as SyntonicAttention

from .syntonic_cnn import (
    PureSyntonicCNN1d,
    PureSyntonicCNN2d,
    PureSyntonicConv1d,
    PureSyntonicConv2d,
)
from .syntonic_cnn import PureSyntonicCNN1d as RecursionConvBlock
from .syntonic_cnn import PureSyntonicCNN1d as SyntonicCNN
from .syntonic_cnn import PureSyntonicCNN2d as SyntonicCNN2d
from .syntonic_cnn import PureSyntonicConv2d as SyntonicConv2d

from .syntonic_mlp import (
    PureDeepSyntonicMLP,
    PureSyntonicLinear,
    PureSyntonicMLP,
)
from .syntonic_mlp import PureSyntonicLinear as SyntonicLinear
from .syntonic_mlp import PureSyntonicMLP as SyntonicMLP

from .syntonic_transformer import (
    PureDHTransformerLayer,
    PureSyntonicTransformer,
    PureSyntonicTransformerEncoder,
)

from srt_library.core.nn.layers.prime_syntony_gate import (
    PrimeSyntonyGate,
    SRTTransformerBlock,
    WindingAttention,
    get_stable_dimensions,
    suggest_network_dimensions,
)

__all__ = [
    "SyntonicMLP", "SyntonicLinear", "PureSyntonicMLP", "PureSyntonicLinear", "PureDeepSyntonicMLP",
    "SyntonicConv2d", "RecursionConvBlock", "SyntonicCNN", "SyntonicCNN2d",
    "PureSyntonicConv1d", "PureSyntonicConv2d", "PureSyntonicCNN1d", "PureSyntonicCNN2d",
    "SyntonicEmbedding", "WindingEmbedding", "PositionalEncoding",
    "PureSyntonicEmbedding", "PureWindingEmbedding", "PurePositionalEncoding",
    "SyntonicAttention", "GnosisAttention", "MultiHeadSyntonicAttention",
    "PureSyntonicAttention", "PureMultiHeadSyntonicAttention",
    "PureDHTransformerLayer", "PureSyntonicTransformerEncoder", "PureSyntonicTransformer",
    "PrimeSyntonyGate", "WindingAttention", "SRTTransformerBlock",
    "get_stable_dimensions", "suggest_network_dimensions",
]
