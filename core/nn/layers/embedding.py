"""
Syntonic Embedding Layers — Pure SRT implementations.

Token embedding maps discrete tokens to points on the resonance manifold.
Positional encoding uses golden-ratio frequencies for non-aliasing structure.

Includes:
- SyntonicEmbedding: Token lookup table using ResonantTensor
- PositionalEncoding: Golden ratio-based positional encodings
- WindingEmbedding: Embeddings via winding numbers on a torus

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import List, Optional

from srt_library.core import sn
from srt_library.core.nn.layers import HarmonizationLayer, SyntonicNorm
from srt_library.core.nn.layers.resonant_linear import ResonantLinear
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


class SyntonicEmbedding(sn.Module):
    """
    Token embedding with golden-ratio initialization.

    Maps discrete tokens to points on the resonance manifold.
    Supports optional harmonization post-lookup.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        harmonize: bool = False,
        scale_by_sqrt_dim: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.harmonize_enabled = harmonize
        self.scale_by_sqrt_dim = scale_by_sqrt_dim
        self.device = device

        # Golden-ratio scaled initialization
        std = PHI_INV / math.sqrt(embedding_dim)
        embedding_data = []
        for i in range(num_embeddings * embedding_dim):
            # Deterministic pseudo-random with golden ratio spacing
            val = (sum([(i * 7919 + j * 104729) % 1000 / 1000.0 for j in range(12)]) - 6.0) * std
            embedding_data.append(val)

        self.weight = sn.Parameter(
            [num_embeddings, embedding_dim], init="zeros", device=device
        )
        # Override with our golden-ratio initialization
        self.weight.tensor.set_data_list(embedding_data)

        if harmonize:
            self.harm = HarmonizationLayer(embedding_dim, embedding_dim, device=device)
            self.norm = SyntonicNorm(embedding_dim, device=device)

    def forward(self, token_indices: List[int]) -> ResonantTensor:
        """
        Look up embeddings for token indices.

        Args:
            token_indices: List of integer token indices

        Returns:
            ResonantTensor of shape (len(token_indices), embedding_dim)
        """
        table_floats = self.weight.to_list()
        embedding_data = []

        for idx in token_indices:
            if self.padding_idx is not None and idx == self.padding_idx:
                embedding_data.extend([0.0] * self.embedding_dim)
            else:
                start = idx * self.embedding_dim
                end = start + self.embedding_dim
                embedding_data.extend(table_floats[start:end])

        embeddings = ResonantTensor(
            embedding_data, [len(token_indices), self.embedding_dim], device=self.device
        )

        if self.scale_by_sqrt_dim:
            scale = math.sqrt(self.embedding_dim)
            embeddings = embeddings.scalar_mul(scale)

        if self.harmonize_enabled:
            embeddings = self.harm.forward(embeddings)
            embeddings = self.norm.forward(embeddings)

        return embeddings


class PositionalEncoding(sn.Module):
    """
    Positional encoding with golden ratio frequencies.

    PE(pos, 2i) = sin(pos / φ^(2i/d))
    PE(pos, 2i+1) = cos(pos / φ^(2i/d))
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        use_golden: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_golden = use_golden
        self.pe_cache = self._compute_pe(max_len, d_model, use_golden)

    def _compute_pe(self, max_len: int, d_model: int, use_golden: bool) -> List[List[float]]:
        pe = []
        for pos in range(max_len):
            row = []
            for i in range(d_model):
                if use_golden:
                    div_term = PHI ** (i / d_model)
                else:
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))
                if i % 2 == 0:
                    row.append(math.sin(pos / div_term))
                else:
                    row.append(math.cos(pos / div_term))
            pe.append(row)
        return pe

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """Add positional encoding to input (batch, seq, d_model) or (seq, d_model)."""
        x_floats = x.to_floats()

        if len(x.shape) == 2:
            seq_len, d_model = x.shape
            batch_size = 1
        elif len(x.shape) == 3:
            batch_size, seq_len, d_model = x.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if seq_len > self.max_len:
            self.pe_cache = self._compute_pe(seq_len, d_model, self.use_golden)

        output_data = []
        for b in range(batch_size):
            for s in range(seq_len):
                for d in range(d_model):
                    if len(x.shape) == 2:
                        idx = s * d_model + d
                    else:
                        idx = b * (seq_len * d_model) + s * d_model + d
                    output_data.append(x_floats[idx] + self.pe_cache[s][d])

        return ResonantTensor(output_data, x.shape, device=x.device)


class WindingEmbedding(sn.Module):
    """
    Embedding using winding number structure on a torus.

    Maps discrete tokens to continuous positions using coprime
    winding numbers for rich, non-degenerate structure.

    e(t) = [cos(2πw₁t/V), sin(2πw₁t/V), ...]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_windings: int = 8,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_windings = num_windings
        self.device = device

        self.windings = self._generate_coprimes(num_windings)

        winding_dim = 2 * num_windings
        self.projection = ResonantLinear(winding_dim, embedding_dim, device=device)
        self.norm = SyntonicNorm(embedding_dim, device=device)

    def _generate_coprimes(self, n: int) -> List[int]:
        coprimes = [1, 2]
        while len(coprimes) < n:
            next_val = int(coprimes[-1] * PHI) + 1
            while any(math.gcd(next_val, c) > 1 for c in coprimes):
                next_val += 1
            coprimes.append(next_val)
        return coprimes[:n]

    def forward(self, token_indices: List[int]) -> ResonantTensor:
        """Compute winding embeddings for token indices."""
        if isinstance(token_indices[0], list):
            flat_indices = [idx for batch in token_indices for idx in batch]
        else:
            flat_indices = token_indices

        all_features = []
        for idx in flat_indices:
            t = float(idx) / self.num_embeddings
            features = []
            for w in self.windings:
                angle = 2 * math.pi * w * t
                features.append(math.cos(angle))
                features.append(math.sin(angle))
            all_features.extend(features)

        winding_dim = 2 * self.num_windings
        winding_tensor = ResonantTensor(
            all_features, [len(flat_indices), winding_dim], device=self.device
        )

        embeddings = self.projection.forward(winding_tensor)
        embeddings = self.norm.forward(embeddings)
        return embeddings


__all__ = [
    "SyntonicEmbedding",
    "PositionalEncoding",
    "WindingEmbedding",
]
