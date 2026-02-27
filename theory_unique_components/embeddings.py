"""
Pure Syntonic Embeddings: Embedding layers with DHSR structure.

Includes:
- PurePositionalEncoding: Golden ratio-based positional encodings
- PureWindingEmbedding: Embeddings using winding numbers
- PureSyntonicEmbedding: Token embeddings with harmonization (limited)

 - Pure Rust backend.

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import List, Optional

from srt_library.core.nn.layers import HarmonizationLayer, SyntonicNorm
from srt_library.core.nn.layers.resonant_linear import ResonantLinear
from srt_library.core.nn.layers.resonant_parameter import ResonantParameter
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


class PurePositionalEncoding:
    """
    Positional encoding with golden ratio frequencies.

    Uses golden ratio-spaced frequencies for richer
    positional information without aliasing.

    PE(pos, 2i) = sin(pos / φ^(2i/d))
    PE(pos, 2i+1) = cos(pos / φ^(2i/d))

    Example:
        >>> from srt_library.core.nn.resonant_tensor import ResonantTensor
        >>> pe = PurePositionalEncoding(d_model=512, max_len=5000)
        >>> x = ResonantTensor([0.1] * 512 * 100, [1, 100, 512])
        >>> x_with_pe = pe.forward(x)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,  # Not implemented in pure version
        use_golden: bool = True,
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability (not implemented)
            use_golden: Use golden ratio frequencies (vs standard)
        """
        self.d_model = d_model
        self.max_len = max_len
        self.use_golden = use_golden
        self.dropout_p = dropout

        # Precompute positional encodings
        self.pe_cache = self._compute_pe(max_len, d_model, use_golden)

    def _compute_pe(
        self, max_len: int, d_model: int, use_golden: bool
    ) -> List[List[float]]:
        """Precompute positional encoding table."""
        pe = []

        for pos in range(max_len):
            row = []
            for i in range(d_model):
                if use_golden:
                    # Golden ratio frequencies: φ^(i/d)
                    div_term = PHI ** (i / d_model)
                else:
                    # Standard sinusoidal
                    div_term = math.exp(i * (-math.log(10000.0) / d_model))

                if i % 2 == 0:
                    # Even indices: sin
                    row.append(math.sin(pos / div_term))
                else:
                    # Odd indices: cos
                    row.append(math.cos(pos / div_term))

            pe.append(row)

        return pe

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model) or (seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x_floats = x.to_floats()

        # Determine shape
        if len(x.shape) == 2:
            # (seq_len, d_model)
            seq_len, d_model = x.shape
            batch_size = 1
        elif len(x.shape) == 3:
            # (batch, seq_len, d_model)
            batch_size, seq_len, d_model = x.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

        if seq_len > self.max_len:
            # Rebuild cache if needed
            self.pe_cache = self._compute_pe(seq_len, d_model, self.use_golden)

        # Add positional encoding
        output_data = []
        for b in range(batch_size):
            for s in range(seq_len):
                for d in range(d_model):
                    if len(x.shape) == 2:
                        idx = s * d_model + d
                    else:
                        idx = b * (seq_len * d_model) + s * d_model + d

                    # Add PE to input
                    output_data.append(x_floats[idx] + self.pe_cache[s][d])

        return ResonantTensor(output_data, x.shape, device=x.device)

    def __repr__(self) -> str:
        return f"PurePositionalEncoding(d_model={self.d_model}, max_len={self.max_len}, golden={self.use_golden})"


class PureWindingEmbedding:
    """
    Embedding using winding number structure.

    Maps discrete tokens to continuous positions on
    a torus, using winding numbers for rich structure.

    The embedding is: e(t) = [cos(2πw₁t/V), sin(2πw₁t/V), ...]

    Where w_i are coprime winding numbers.

    Example:
        >>> from srt_library.core.nn.resonant_tensor import ResonantTensor
        >>> embed = PureWindingEmbedding(num_embeddings=100, embedding_dim=64)
        >>> # Simulate token indices (would come from data)
        >>> tokens = [5, 10, 15, 20]  # Token indices
        >>> embeddings = embed.forward(tokens)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_windings: int = 8,
        learnable: bool = False,  # Not supported in pure version
        device: str = "cpu",
    ):
        """
        Initialize winding embedding.

        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Output embedding dimension
            num_windings: Number of winding frequencies
            learnable: Make winding numbers learnable (not supported)
            device: Device placement
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_windings = num_windings
        self.device = device

        # Generate coprime winding numbers
        self.windings = self._generate_coprimes(num_windings)

        # Project winding features to embedding dim
        winding_dim = 2 * num_windings  # cos and sin for each
        self.projection = ResonantLinear(winding_dim, embedding_dim, device=device)
        self.norm = SyntonicNorm(embedding_dim, device=device)

    def _generate_coprimes(self, n: int) -> List[int]:
        """Generate n coprime numbers using Fibonacci-like sequence."""
        coprimes = [1, 2]
        while len(coprimes) < n:
            # Use golden ratio for spacing
            next_val = int(coprimes[-1] * PHI) + 1
            # Ensure coprime with all previous
            while any(math.gcd(next_val, c) > 1 for c in coprimes):
                next_val += 1
            coprimes.append(next_val)
        return coprimes[:n]

    def forward(self, token_indices: List[int]) -> ResonantTensor:
        """
        Compute winding embeddings.

        Args:
            token_indices: List of token indices (batch * seq_len)
                          Can be 1D list or nested list [[batch1_tokens], [batch2_tokens]]

        Returns:
            Embeddings tensor
        """
        # Flatten if nested
        if isinstance(token_indices[0], list):
            flat_indices = [idx for batch in token_indices for idx in batch]
            batch_size = len(token_indices)
            _seq_len = len(token_indices[0])
        else:
            flat_indices = token_indices
            batch_size = 1
            _seq_len = len(token_indices)

        # Compute winding features for each token
        all_features = []
        for idx in flat_indices:
            # Normalize token index to [0, 1)
            t = float(idx) / self.num_embeddings

            # Compute winding features
            features = []
            for w in self.windings:
                angle = 2 * math.pi * w * t
                features.append(math.cos(angle))
                features.append(math.sin(angle))

            all_features.extend(features)

        # Shape: (batch * seq_len, 2 * num_windings)
        winding_dim = 2 * self.num_windings
        winding_tensor = ResonantTensor(
            all_features, [len(flat_indices), winding_dim], device=self.device
        )

        # Project to embedding dimension
        embeddings = self.projection.forward(winding_tensor)
        embeddings = self.norm.forward(embeddings)

        # Reshape to (batch, seq_len, embed_dim) if batched
        if batch_size > 1:
            # Currently [batch*seq_len, embed_dim], need to mark as 3D
            # For now, return as 2D - proper 3D reshape would need API support
            pass

        return embeddings

    def __repr__(self) -> str:
        return f"PureWindingEmbedding(vocab={self.num_embeddings}, embed_dim={self.embedding_dim}, windings={self.num_windings})"


class PureSyntonicEmbedding:
    """
    Embedding layer with harmonization.

    NOTE: This is a LIMITED pure implementation. True embedding lookup
    requires indexing into a large parameter table, which is memory-intensive
    in pure Python. This version is primarily for demonstration.

    For production, use PureWindingEmbedding which generates embeddings
    mathematically rather than via lookup.

    Example:
        >>> embed = PureSyntonicEmbedding(num_embeddings=100, embedding_dim=64)
        >>> tokens = [5, 10, 15]
        >>> embeddings = embed.forward(tokens)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        harmonize: bool = True,
        scale_by_sqrt_dim: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize syntonic embedding.

        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
            padding_idx: Padding token index
            harmonize: Apply harmonization to embeddings
            scale_by_sqrt_dim: Scale embeddings by sqrt(dim)
            device: Device placement
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.harmonize_enabled = harmonize
        self.scale_by_sqrt_dim = scale_by_sqrt_dim
        self.device = device

        # Initialize embedding table with golden-scaled variance
        std = PHI_INV / math.sqrt(embedding_dim)
        embedding_data = []
        for i in range(num_embeddings * embedding_dim):
            # Simple random initialization (Gaussian approximation using central limit)
            # In production, would use proper random number generation
            val = (
                sum([(i * 7919 + j * 104729) % 1000 / 1000.0 for j in range(12)]) - 6.0
            ) * std
            embedding_data.append(val)

        self.embedding_table = ResonantParameter(
            embedding_data, [num_embeddings, embedding_dim], device=device
        )

        if harmonize:
            self.harm = HarmonizationLayer(embedding_dim, embedding_dim, device=device)
            self.norm = SyntonicNorm(embedding_dim, device=device)

    def forward(self, token_indices: List[int]) -> ResonantTensor:
        """
        Get embeddings for token indices.

        Args:
            token_indices: List of token indices

        Returns:
            Embeddings tensor (len(indices), embed_dim)
        """
        # Extract embeddings by indexing into table
        table_floats = self.embedding_table.to_floats()

        embedding_data = []
        for idx in token_indices:
            if self.padding_idx is not None and idx == self.padding_idx:
                # Zero padding
                embedding_data.extend([0.0] * self.embedding_dim)
            else:
                # Extract row from table
                start = idx * self.embedding_dim
                end = start + self.embedding_dim
                embedding_data.extend(table_floats[start:end])

        embeddings = ResonantTensor(
            embedding_data, [len(token_indices), self.embedding_dim], device=self.device
        )

        # Scale by sqrt(dim) for stable attention
        if self.scale_by_sqrt_dim:
            scale = math.sqrt(self.embedding_dim)
            embeddings = embeddings.scalar_mul(scale)

        # Harmonize embeddings
        if self.harmonize_enabled:
            embeddings = self.harm.forward(embeddings)
            embeddings = self.norm.forward(embeddings)

        return embeddings

    def __repr__(self) -> str:
        return f"PureSyntonicEmbedding(vocab={self.num_embeddings}, embed_dim={self.embedding_dim})"


if __name__ == "__main__":
    # Test the pure embeddings
    from srt_library.core.nn.resonant_tensor import ResonantTensor

    print("=" * 70)
    print("Testing PurePositionalEncoding...")
    print("=" * 70)

    pe = PurePositionalEncoding(d_model=8, max_len=100, use_golden=True)
    print(f"Positional encoding: {pe}")

    # Test with 2D input (seq_len, d_model)
    x_2d = ResonantTensor([0.1] * 8 * 5, [5, 8])
    x_pe = pe.forward(x_2d)
    print(f"Input shape: {x_2d.shape}, syntony: {x_2d.syntony:.4f}")
    print(f"Output shape: {x_pe.shape}, syntony: {x_pe.syntony:.4f}")

    # Test with 3D input (batch, seq_len, d_model)
    x_3d = ResonantTensor([0.1] * 8 * 5 * 2, [2, 5, 8])
    x_pe_3d = pe.forward(x_3d)
    print(f"Batched output shape: {x_pe_3d.shape}, syntony: {x_pe_3d.syntony:.4f}")

    print("\n" + "=" * 70)
    print("Testing PureWindingEmbedding...")
    print("=" * 70)

    winding_embed = PureWindingEmbedding(
        num_embeddings=100, embedding_dim=16, num_windings=4
    )
    print(f"Winding embedding: {winding_embed}")
    print(f"Coprime windings: {winding_embed.windings}")

    # Test with single sequence
    tokens = [5, 10, 15, 20, 25]
    embeddings = winding_embed.forward(tokens)
    print(f"Embeddings shape: {embeddings.shape}, syntony: {embeddings.syntony:.4f}")

    print("\n" + "=" * 70)
    print("Testing PureSyntonicEmbedding...")
    print("=" * 70)

    syntonic_embed = PureSyntonicEmbedding(
        num_embeddings=50, embedding_dim=16, harmonize=True
    )
    print(f"Syntonic embedding: {syntonic_embed}")

    tokens2 = [1, 5, 10]
    embeddings2 = syntonic_embed.forward(tokens2)
    print(f"Embeddings shape: {embeddings2.shape}, syntony: {embeddings2.syntony:.4f}")

    print("\n" + "=" * 70)
    print("SUCCESS - All pure embedding classes working!")
    print("=" * 70)
