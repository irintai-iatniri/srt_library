"""
Pure Resonant Winding Embedding - Maps winding states to resonant embeddings.

Uses ResonantTensor for exact Q(φ) arithmetic. NO PYTORCH DEPENDENCIES.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

from applications._core import ResonantTensor, WindingState, enumerate_windings

from applications.exact import GoldenExact

PHI = GoldenExact.phi()  # Use exact golden ratio


class PureResonantWindingEmbedding:
    """
    Embedding layer for winding states using Resonant Engine.

    Uses ResonantTensor for exact lattice storage and evolution.
    """

    def __init__(
        self,
        max_n: int = 5,
        embed_dim: int = 64,
        precision: int = 100,
    ):
        """
        Initialize resonant winding embedding.

        Args:
            max_n: Maximum winding number for enumeration
            embed_dim: Embedding dimension
            precision: Exact arithmetic precision
        """
        self.max_n = max_n
        self.embed_dim = embed_dim
        self.precision = precision

        # 1. Enumerate windings and map to indices
        self.windings = enumerate_windings(max_n)
        self.num_windings = len(self.windings)
        self.winding_to_idx: Dict[str, int] = {
            self._winding_key(w): i for i, w in enumerate(self.windings)
        }

        # 2. Initialize embedding matrix
        # Shape: (num_windings * embed_dim,) with golden-scaled initialization
        self._init_embeddings()

    def _winding_key(self, w: WindingState) -> str:
        return f"{w.n7},{w.n8},{w.n9},{w.n10}"

    def _init_embeddings(self):
        """Initialize embedding matrix with golden scaling."""
        data = []
        mode_norms = []

        for w in self.windings:
            norm_sq = w.norm_squared
            golden_scale = math.exp(-norm_sq / (2 * PHI.eval())) / math.sqrt(self.embed_dim)  # Use display value for computation

            # Generate embedding for this winding
            for i in range(self.embed_dim):
                data.append(random.gauss(0, golden_scale))
                # Mode norm: winding norm + position in embedding
                mode_norms.append(norm_sq + i * i)  # Keep as exact integer

        self.weight = ResonantTensor(
            data, [self.num_windings, self.embed_dim], mode_norms, self.precision
        )

    def forward(self, winding: WindingState) -> ResonantTensor:
        """
        Get embedding for a single winding state.

        Args:
            winding: WindingState object

        Returns:
            Embedding ResonantTensor of shape (embed_dim,)
        """
        key = self._winding_key(winding)
        if key not in self.winding_to_idx:
            raise ValueError(
                f"Winding {winding} not in vocabulary (max_n={self.max_n})"
            )

        idx = self.winding_to_idx[key]

        # Extract the row from the weight matrix
        all_data = self.weight.to_floats()
        start = idx * self.embed_dim
        embed_data = all_data[start : start + self.embed_dim]
        mode_norms = [i * i for i in range(self.embed_dim)]  # Keep as exact integers

        return ResonantTensor(embed_data, [self.embed_dim], mode_norms, self.precision)

    def batch_forward(self, winding_states: List[WindingState]) -> ResonantTensor:
        """
        Get embeddings for a list of winding states.

        Args:
            winding_states: List of WindingState objects

        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        batch_data = []
        batch_mode_norms = []
        all_data = self.weight.to_floats()

        for w in winding_states:
            key = self._winding_key(w)
            if key not in self.winding_to_idx:
                raise ValueError(f"Winding {w} not in vocabulary (max_n={self.max_n})")
            idx = self.winding_to_idx[key]

            start = idx * self.embed_dim
            batch_data.extend(all_data[start : start + self.embed_dim])

            # Mode norms
            for i in range(self.embed_dim):
                batch_mode_norms.append(float(w.norm_squared + i * i))  # export boundary

        return ResonantTensor(
            batch_data,
            [len(winding_states), self.embed_dim],
            batch_mode_norms,
            self.precision,
        )

    def crystallize(self):
        """Snap entire embedding matrix to lattice."""
        self.weight.crystallize()

    def parameters(self) -> List[ResonantTensor]:
        """Return all learnable parameters."""
        return [self.weight]

    def __repr__(self) -> str:
        return f"PureResonantWindingEmbedding(num_windings={self.num_windings}, embed_dim={self.embed_dim})"


if __name__ == "__main__":
    print("Testing PureResonantWindingEmbedding...")

    embed = PureResonantWindingEmbedding(max_n=2, embed_dim=16)
    print(f"Embedding: {embed}")

    windings = embed.windings[:3]

    # Single
    e = embed.forward(windings[0])
    print(f"Single: shape={e.shape()}, syntony={e.syntony():.4f}")

    # Batch
    X = embed.batch_forward(windings)
    print(f"Batch: shape={X.shape()}")

    print("SUCCESS")
