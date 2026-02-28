"""
Prime Syntony Gate Layer for SRT/CRT Neural Networks

Implements the Prime Syntony Gate as described in The_Grand_Synthesis.md.
This layer applies resonance boosts at Fibonacci prime dimensions.

Pure SRT implementation — no PyTorch or NumPy.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

from srt_library.core import sn
from srt_library.core.nn.resonant_tensor import ResonantTensor

# SRT Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio


class PrimeSyntonyGate(sn.Module):
    """
    Prime Syntony Gate Layer

    Applies resonance amplification at Fibonacci prime dimensions.
    According to CRT, these dimensions correspond to "transcendence gates"
    where consciousness emerges.

    Args:
        dim: Dimension of the input feature space
        boost_scale: Scaling factor for resonance boost (default: 1.0)
        anomaly_penalty: Penalty for the "material anomaly" at dim=4 (default: 0.9)
    """

    def __init__(
        self, dim: int, boost_scale: float = 1.0, anomaly_penalty: float = 0.9
    ):
        super().__init__()
        self.dim = dim
        self.boost_scale = boost_scale
        self.anomaly_penalty = anomaly_penalty

        # Fibonacci prime indices (transcendence gates)
        self.fib_prime_indices = {3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47}

        # Determine if this dimension is resonant
        self.is_resonant = dim in self.fib_prime_indices

        # Calculate resonance boost factor
        if self.is_resonant:
            if dim == 4:
                self.boost_factor = (PHI ** dim) * anomaly_penalty * boost_scale
            else:
                self.boost_factor = (PHI ** dim) * boost_scale
        else:
            self.boost_factor = 1.0

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """Apply Prime Syntony Gate transformation."""
        if self.is_resonant:
            # Normalize to unit sphere, then apply boost
            data = x.to_floats()
            # L2 norm along last dimension
            shape = x.shape
            last_dim = shape[-1]
            batch_size = len(data) // last_dim

            result = []
            for b in range(batch_size):
                start = b * last_dim
                chunk = data[start:start + last_dim]
                norm = math.sqrt(sum(v * v for v in chunk))
                if norm > 1e-12:
                    result.extend(v / norm * self.boost_factor for v in chunk)
                else:
                    result.extend(v * self.boost_factor for v in chunk)

            return ResonantTensor(result, list(shape), device=x.device)
        else:
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, resonant={self.is_resonant}, boost={self.boost_factor:.3f}"


class WindingAttention(sn.Module):
    """
    Winding Attention Layer with Mersenne-stabilized dimensions.

    Implements attention with dimension constraints based on Mersenne primes
    for stability, as per SRT matter generation rules.

    Args:
        embed_dim: Embedding dimension (should be Mersenne prime for stability)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        if not self._is_mersenne_dimension(embed_dim):
            import warnings
            warnings.warn(
                f"embed_dim={embed_dim} is not a Mersenne prime. "
                "Consider using: 3, 7, 31, 127 for stability."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Attention projections
        from srt_library.core.nn.layers.resonant_linear import ResonantLinear
        self.q_proj = ResonantLinear(embed_dim, embed_dim)
        self.k_proj = ResonantLinear(embed_dim, embed_dim)
        self.v_proj = ResonantLinear(embed_dim, embed_dim)
        self.out_proj = ResonantLinear(embed_dim, embed_dim)

        self.dropout_p = dropout
        self.syntony_gate = PrimeSyntonyGate(embed_dim)

    def _is_mersenne_dimension(self, dim: int) -> bool:
        mersenne_primes = {3, 7, 31, 127}
        return dim in mersenne_primes

    def forward(
        self,
        query: ResonantTensor,
        key: ResonantTensor,
        value: ResonantTensor,
    ) -> ResonantTensor:
        """Multi-head attention with winding stabilization."""
        # Simple attention: Q @ K^T / sqrt(d) -> softmax -> @ V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = q.matmul(k.transpose(-2, -1))
        attn = attn.scalar_mul(1.0 / scale)
        attn.softmax(dim=-1)

        output = attn.matmul(v)
        output = self.out_proj(output)
        output = self.syntony_gate(output)
        return output


class SRTTransformerBlock(sn.Module):
    """
    SRT Transformer Block with Prime-stabilized components.

    Args:
        embed_dim: Embedding dimension (Mersenne prime)
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension (should be ~4x embed_dim)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * embed_dim

        self.attention = WindingAttention(embed_dim, num_heads, dropout)

        from srt_library.core.nn.layers.resonant_linear import ResonantLinear
        from srt_library.core.nn.layers.normalization import SyntonicNorm
        self.ff1 = ResonantLinear(embed_dim, ff_dim)
        self.ff_gate = PrimeSyntonyGate(ff_dim)
        self.ff2 = ResonantLinear(ff_dim, embed_dim)

        self.norm1 = SyntonicNorm(embed_dim)
        self.norm2 = SyntonicNorm(embed_dim)
        self.dropout_p = dropout

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        # Multi-head attention
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_out = self.ff1(x)
        ff_out = self.ff_gate(ff_out)
        ff_out = self.ff2(ff_out)
        x = self.norm2(x + ff_out)

        return x


# Utility functions for dimension validation

def get_stable_dimensions(max_dim: int = 128) -> list:
    """
    Get all Mersenne prime dimensions up to max_dim.
    These are the "stable" dimensions for SRT neural networks.
    """
    mersenne_primes = []
    p = 2
    while True:
        mp = (1 << p) - 1
        if mp > max_dim:
            break
        mersenne_primes.append(mp)
        p += 1
    return mersenne_primes


def suggest_network_dimensions(
    input_dim: int, output_dim: int, num_layers: int
) -> list:
    """
    Suggest stable dimensions for a neural network architecture.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        num_layers: Number of layers

    Returns:
        List of suggested dimensions for each layer
    """
    stable_dims = get_stable_dimensions(max(input_dim, output_dim) * 4)

    dimensions = [input_dim]

    for i in range(1, num_layers):
        target_dim = stable_dims[
            min(i * len(stable_dims) // num_layers, len(stable_dims) - 1)
        ]
        dimensions.append(target_dim)

    dimensions.append(output_dim)
    return dimensions
