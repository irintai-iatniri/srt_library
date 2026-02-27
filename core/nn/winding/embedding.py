"""
Winding State Embedding - Maps winding states to neural embeddings.

This module provides WindingStateEmbedding, which maps discrete T^4 winding
states |n₇, n₈, n₉, n₁₀⟩ to continuous learned embeddings using the Rust backend.

.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from ..._core import ResonantTensor, WindingState, enumerate_windings

PHI = (1 + math.sqrt(5)) / 2


class WindingStateEmbedding:
    """
    Embedding layer for winding states.

    Maps discrete winding states |n₇, n₈, n₉, n₁₀⟩ to learned continuous
    embeddings. Each unique winding state gets its own learnable parameter vector
    stored as a ResonantTensor.
    """

    def __init__(
        self,
        max_n: int = 5,
        embed_dim: int = 64,
        init_scale: Optional[float] = None,
        precision: int = 100,
    ):
        """
        Initialize winding state embedding.

        Args:
            max_n: Maximum winding number (enumerates [-max_n, max_n]^4)
            embed_dim: Embedding dimension
            init_scale: Initialization scale (defaults to 1/sqrt(embed_dim))
            precision: Lattice precision for exact arithmetic
        """
        self.max_n = max_n
        self.embed_dim = embed_dim
        self.precision = precision

        # Enumerate all winding states in the range
        self.windings = enumerate_windings(max_n)
        self.num_windings = len(self.windings)

        # Create embeddings for each winding state
        self.embeddings: Dict[str, ResonantTensor] = {}
        self.mode_norms: Dict[str, int] = {}

        if init_scale is None:
            init_scale = 1.0 / math.sqrt(embed_dim)

        for w in self.windings:
            key = self._winding_key(w)
            mode_norm_sq = w.norm_squared
            self.mode_norms[key] = mode_norm_sq

            # Golden initialization: scale by exp(-|n|²/(2φ))
            golden_scale = math.exp(-mode_norm_sq / (2 * PHI))

            # Generate embedding data
            embed_data = [
                random.gauss(0, init_scale * golden_scale) for _ in range(embed_dim)
            ]
            mode_norms_list = [float(i**2) for i in range(embed_dim)]

            self.embeddings[key] = ResonantTensor(
                embed_data, [embed_dim], mode_norms_list, precision
            )

    def _winding_key(self, w: WindingState) -> str:
        """Generate string key for winding state."""
        return f"{w.n7},{w.n8},{w.n9},{w.n10}"

    def forward(self, winding: WindingState) -> ResonantTensor:
        """
        Get embedding for a single winding state.

        Args:
            winding: WindingState object

        Returns:
            Embedding ResonantTensor of shape (embed_dim,)
        """
        key = self._winding_key(winding)
        if key not in self.embeddings:
            raise ValueError(
                f"Winding state {winding} not in embedding vocabulary. "
                f"max_n={self.max_n} may be too small."
            )
        return self.embeddings[key]

    def batch_forward(self, windings: List[WindingState]) -> ResonantTensor:
        """
        Get embeddings for a batch of winding states.

        Args:
            windings: List of WindingState objects

        Returns:
            Embeddings ResonantTensor of shape (batch_size, embed_dim)
        """
        # Collect all embeddings as flat data
        batch_data = []
        for w in windings:
            embed = self.forward(w)
            batch_data.extend(embed.to_floats())

        # Create batched tensor
        batch_size = len(windings)
        mode_norms = [float(i**2) for i in range(batch_size * self.embed_dim)]

        return ResonantTensor(
            batch_data, [batch_size, self.embed_dim], mode_norms, self.precision
        )

    def get_mode_norm(self, winding: WindingState) -> int:
        """Get mode norm squared |n|² for a winding state."""
        key = self._winding_key(winding)
        return self.mode_norms[key]

    def get_all_mode_norms(self, windings: List[WindingState]) -> List[float]:
        """Get mode norms for a batch of windings."""
        return [float(self.get_mode_norm(w)) for w in windings]

    def parameters(self) -> List[ResonantTensor]:
        """Return all embedding tensors for training."""
        return list(self.embeddings.values())

    def __repr__(self) -> str:
        return (
            f"WindingStateEmbedding(max_n={self.max_n}, embed_dim={self.embed_dim}, "
            f"num_windings={self.num_windings})"
        )


if __name__ == "__main__":
    # Test the pure WindingStateEmbedding
    print("Testing WindingStateEmbedding...")

    embed = WindingStateEmbedding(max_n=2, embed_dim=16)
    print(f"Embedding: {embed}")

    # Get some windings
    windings = embed.windings[:3]
    print(f"First 3 windings: {windings}")

    # Single embedding
    e = embed.forward(windings[0])
    print(f"Single embedding shape: {e.shape}")
    print(f"Single embedding syntony: {e.syntony:.4f}")

    # Batch
    X = embed.batch_forward(windings)
    print(f"Batch shape: {X.shape}")

    print("SUCCESS")
