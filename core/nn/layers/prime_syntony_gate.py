"""
Prime Syntony Gate Layer for SRT/CRT Neural Networks

Implements the Prime Syntony Gate as described in The_Grand_Synthesis.md.
This layer applies resonance boosts at Fibonacci prime dimensions.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

# SRT Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio


class PrimeSyntonyGate(nn.Module):
    """
    Prime Syntony Gate Layer

    Applies resonance amplification at Fibonacci prime dimensions.
    According to CRT, these dimensions correspond to "transcendence gates"
    where consciousness emerges.

    Args:
        dim (int): Dimension of the input feature space
        boost_scale (float): Scaling factor for resonance boost (default: 1.0)
        anomaly_penalty (float): Penalty for the "material anomaly" at dim=4 (default: 0.9)
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
                # The "Material Anomaly" - composite structure creates prime reality
                # but with imperfect resonance
                self.boost_factor = (PHI**dim) * anomaly_penalty * boost_scale
            else:
                # Pure Fibonacci prime resonance
                self.boost_factor = (PHI**dim) * boost_scale
        else:
            self.boost_factor = 1.0

        # Register as buffer for device compatibility
        self.register_buffer(
            "boost", torch.tensor(self.boost_factor, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Prime Syntony Gate transformation.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Transformed tensor with resonance boost applied if dimension is prime-gated
        """
        if self.is_resonant:
            # Normalize to unit sphere (Gnostic crystallization)
            x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
            # Apply resonance boost
            return x_norm * self.boost
        else:
            # No transformation for non-resonant dimensions
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, resonant={self.is_resonant}, boost={self.boost_factor:.3f}"


class WindingAttention(nn.Module):
    """
    Winding Attention Layer with Mersenne-stabilized dimensions

    Implements attention with dimension constraints based on Mersenne primes
    for stability, as per SRT matter generation rules.

    Args:
        embed_dim (int): Embedding dimension (should be Mersenne prime for stability)
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # Validate dimension stability
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
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Prime Syntony Gate for attention output
        self.syntony_gate = PrimeSyntonyGate(embed_dim)

    def _is_mersenne_dimension(self, dim: int) -> bool:
        """Check if dimension corresponds to a Mersenne prime."""
        mersenne_primes = {3, 7, 31, 127}  # First few Mersenne primes
        return dim in mersenne_primes

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-head attention with winding stabilization.

        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            attn_mask: Attention mask (batch, num_heads, seq_len, seq_len)

        Returns:
            Attention output (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections and reshape
        q = (
            self.q_proj(query)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention computation
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project out
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        # Apply Prime Syntony Gate
        output = self.syntony_gate(output)

        return output


class SRTTransformerBlock(nn.Module):
    """
    SRT Transformer Block with Prime-stabilized components

    Implements a transformer block where all dimensions are constrained
    to Mersenne primes for stability, and includes Prime Syntony Gates.

    Args:
        embed_dim (int): Embedding dimension (Mersenne prime)
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension (should be ~4x embed_dim)
        dropout (float): Dropout probability
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

        # Multi-head attention with winding stabilization
        self.attention = WindingAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            PrimeSyntonyGate(ff_dim),  # Resonance boost in FF layer
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Multi-head attention
        attn_output = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# Utility functions for dimension validation


def get_stable_dimensions(max_dim: int = 128) -> list[int]:
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
) -> list[int]:
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

    # Start from input, end at output, interpolate stable dimensions
    dimensions = [input_dim]

    for i in range(1, num_layers):
        # Interpolate between stable dimensions
        target_dim = stable_dims[
            min(i * len(stable_dims) // num_layers, len(stable_dims) - 1)
        ]
        dimensions.append(target_dim)

    dimensions.append(output_dim)
    return dimensions
