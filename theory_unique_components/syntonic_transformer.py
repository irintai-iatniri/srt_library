"""
Pure Syntonic Transformer: Transformer architecture with DHSR structure.

NO PYTORCH DEPENDENCIES - uses sn.Module and ResonantTensor.

Includes:
- PureDHTransformerLayer: Single transformer layer using D→H cycle
- PureSyntonicTransformerEncoder: Encoder-only transformer stack
- PureSyntonicTransformer: Generic encoder (for pre-embedded inputs)
- PureSyntonicTransformerLM: Language model with vocab embedding + positional encoding

Architecture Notes:
- Uses SyntonicNorm (golden-ratio normalization) instead of standard LayerNorm
- Residual connections scaled by 1/φ for harmonic structure
- Pre-norm architecture (normalize before sublayer, better gradient flow)
- GELU activation in FFN
- Multi-head attention with ResonantTensor exact arithmetic

Usage Examples:

    # Encoder-only (when embeddings provided externally)
    >>> from srt_library.theory_unique_components.syntonic_transformer_pure import PureSyntonicTransformer
    >>> from srt_library.core.nn.resonant_tensor import ResonantTensor
    >>> encoder = PureSyntonicTransformer(d_model=64, n_layers=3, output_dim=10)
    >>> x = ResonantTensor([0.1] * 64 * 10, [10, 64])  # Shape: (seq, d_model)
    >>> output = encoder(x)  # Shape: (1, output_dim) - average pooled

    # Language model (with built-in embeddings)
    >>> from srt_library.theory_unique_components.syntonic_transformer_pure import PureSyntonicTransformerLM
    >>> lm = PureSyntonicTransformerLM(vocab_size=1000, d_model=64, n_layers=3)
    >>> tokens = [1, 5, 10, 15]
    >>> logits = lm(tokens)  # Shape: (4, 1000) - per-token logits

Embedding Types:
- 'winding': Deterministic winding embeddings (memory efficient, rich structure)
- 'syntonic': Learned embeddings with harmonization (trainable, syntony-aware)
- 'learned': Standard learned embeddings (trainable, conventional)

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from srt_library.core import sn as sn
from .syntonic_attention import (
    PureMultiHeadSyntonicAttention,
)
from srt_library.core.nn.layers import SyntonicNorm
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


def _gelu_rt(x: ResonantTensor) -> ResonantTensor:
    """Apply GELU activation (tanh approximation)."""
    data = x.to_floats()
    # GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    result = []
    for v in data:
        inner = sqrt_2_pi * (v + 0.044715 * v**3)
        gelu = 0.5 * v * (1.0 + math.tanh(inner))
        result.append(gelu)
    shape = x.shape
    mode_norms = [float(i * i) for i in range(len(result))]
    return ResonantTensor(result, shape, mode_norms, 100, device=x.device)


class PureDHTransformerLayer(sn.Module):
    """
    Transformer layer with D→H structure.

    Pure Python + ResonantTensor implementation.

    Standard transformer layer components:
    1. Self-Attention with harmonization
    2. FFN with differentiation/harmonization

    Example:
        >>> layer = PureDHTransformerLayer(d_model=64, n_heads=4)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> y = layer(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize DH transformer layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            precision: ResonantTensor precision
            device: Device placement
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.precision = precision
        self.device = device

        # Self-attention
        self.self_attn = PureMultiHeadSyntonicAttention(
            d_model, n_heads, dropout, precision, device=device
        )

        # FFN weights (shape [out, in] for matmul which does A @ W^T)
        self.ffn_w1 = sn.Parameter([d_ff, d_model], init="kaiming", device=device)
        self.ffn_w2 = sn.Parameter([d_model, d_ff], init="kaiming", device=device)

        # Normalization layers (golden-ratio aware)
        self.norm1 = SyntonicNorm(d_model, device=device)
        self.norm2 = SyntonicNorm(d_model, device=device)

        self.dropout = sn.Dropout(dropout)

        self._attn_syntony: Optional[float] = None
        self._ffn_syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, d_model)

        Returns:
            Output (seq_len, d_model)
        """
        # === Attention sublayer ===
        # Pre-norm
        x_norm = self.norm1.forward(x)

        # Attention
        attn_out = self.self_attn(x_norm, x_norm, x_norm)
        self._attn_syntony = self.self_attn.syntony

        # Residual (golden scaled)
        attn_scaled = attn_out.scalar_mul(1.0 / PHI)
        x = x.elementwise_add(attn_scaled)

        # === FFN sublayer ===
        # Pre-norm
        x_norm = self.norm2.forward(x)

        # FFN: W2(GELU(W1(x)))
        ffn_hidden = x_norm.matmul(self.ffn_w1.tensor)
        ffn_hidden = _gelu_rt(ffn_hidden)
        ffn_out = ffn_hidden.matmul(self.ffn_w2.tensor)

        self._ffn_syntony = ffn_out.syntony

        # Residual
        ffn_scaled = ffn_out.scalar_mul(1.0 / PHI)
        x = x.elementwise_add(ffn_scaled)

        return x

    @property
    def syntony(self) -> float:
        """Get average layer syntony."""
        attn_s = self._attn_syntony if self._attn_syntony else 0.5
        ffn_s = self._ffn_syntony if self._ffn_syntony else 0.5
        return (attn_s + ffn_s) / 2


class PureSyntonicTransformerEncoder(sn.Module):
    """
    Stack of DH transformer encoder layers.

    Pure Python + ResonantTensor implementation.

    Example:
        >>> encoder = PureSyntonicTransformerEncoder(d_model=64, n_layers=3)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> y = encoder(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.1,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize encoder stack.

        Args:
            d_model: Model dimension
            n_heads: Attention heads per layer
            n_layers: Number of layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            precision: ResonantTensor precision
            device: Device placement
        """
        super().__init__()

        self.layers = sn.ModuleList(
            [
                PureDHTransformerLayer(
                    d_model, n_heads, d_ff, dropout, precision, device=device
                )
                for _ in range(n_layers)
            ]
        )
        self.n_layers = n_layers
        self.device = device

        # Final normalization
        self.final_norm = SyntonicNorm(d_model, device=device)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """Forward through encoder stack."""
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.final_norm.forward(x)
        return x

    @property
    def syntony(self) -> float:
        """Get average syntony across layers."""
        syntonies = [layer.syntony for layer in self.layers]
        return sum(syntonies) / len(syntonies) if syntonies else 0.5


class PureSyntonicTransformer(sn.Module):
    """
    Simple encoder-only syntonic transformer.

    For classification or embedding tasks.

    Example:
        >>> model = PureSyntonicTransformer(d_model=64, n_layers=3, output_dim=10)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> logits = model(x)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        output_dim: int = 10,
        dropout: float = 0.1,
        precision: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize syntonic transformer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            output_dim: Output dimension (num classes)
            dropout: Dropout probability
            precision: ResonantTensor precision
            device: Device placement
        """
        super().__init__()

        self.d_model = d_model
        self.precision = precision
        self.device = device

        # Encoder
        self.encoder = PureSyntonicTransformerEncoder(
            d_model, n_heads, n_layers, d_ff, dropout, precision, device=device
        )

        # Output projection (shape [out, in] for matmul which does A @ W^T)
        self.output_proj = sn.Parameter(
            [output_dim, d_model], init="kaiming", device=device
        )

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Forward pass.

        Args:
            x: Input (seq_len, d_model)

        Returns:
            Output logits (output_dim,) - average pooled
        """
        # Encode
        encoded = self.encoder(x)

        # Average pool over sequence
        data = encoded.to_floats()
        shape = encoded.shape
        seq_len, d_model = shape

        pooled = []
        for d in range(d_model):
            col_sum = sum(data[s * d_model + d] for s in range(seq_len))
            pooled.append(col_sum / seq_len)

        pooled_rt = ResonantTensor(
            pooled, [1, d_model], [float(i * i) for i in range(d_model)], self.precision
        )

        # Project to output
        output = pooled_rt.matmul(self.output_proj.tensor)

        return output

    @property
    def syntony(self) -> float:
        """Get model syntony."""
        return self.encoder.syntony


class PureSyntonicTransformerLM(sn.Module):
    """
    Language model with vocabulary embedding and positional encoding.

    Complete transformer for text generation and classification:
    1. Token embedding (vocab_size -> d_model)
    2. Positional encoding (golden ratio frequencies)
    3. Transformer encoder
    4. Output projection (d_model -> vocab_size)

    Example:
        >>> lm = PureSyntonicTransformerLM(
        ...     vocab_size=1000,
        ...     d_model=64,
        ...     n_layers=3,
        ...     embedding_type='winding'
        ... )
        >>> tokens = [5, 10, 15, 20]
        >>> logits = lm(tokens)  # Shape: [4, 1000]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        precision: int = 100,
        embedding_type: str = "winding",  # 'winding', 'learned', 'syntonic'
        use_golden_pe: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize language model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            precision: ResonantTensor precision
            embedding_type: Type of token embedding
                - 'winding': Deterministic winding embeddings (recommended)
                - 'syntonic': Learned with harmonization
                - 'learned': Standard learned embeddings
            use_golden_pe: Use golden ratio frequencies for positional encoding
            device: Device placement
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.precision = precision
        self.device = device

        # Import embedding classes
        from .embeddings import (
            PurePositionalEncoding,
            PureSyntonicEmbedding,
            PureWindingEmbedding,
        )

        # Token embedding
        if embedding_type == "winding":
            self.token_embedding = PureWindingEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                num_windings=8,
                device=device,
            )
        elif embedding_type == "syntonic":
            self.token_embedding = PureSyntonicEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                harmonize=True,
                scale_by_sqrt_dim=True,
                device=device,
            )
        else:  # 'learned'
            self.token_embedding = PureSyntonicEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                harmonize=False,
                scale_by_sqrt_dim=True,
                device=device,
            )

        # Positional encoding
        self.pos_encoding = PurePositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
            use_golden=use_golden_pe,
        )

        # Transformer encoder
        self.encoder = PureSyntonicTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            precision=precision,
            device=device,
        )

        # Output projection to vocabulary (shape [out, in] for matmul)
        self.output_proj = sn.Parameter(
            [vocab_size, d_model], init="kaiming", device=device
        )

    def forward(self, token_indices: List[int]) -> ResonantTensor:
        """
        Forward pass.

        Args:
            token_indices: List of token indices [seq_len]

        Returns:
            Logits (seq_len, vocab_size)
        """
        # Embed tokens: [seq_len, d_model]
        embeddings = self.token_embedding.forward(token_indices)

        # Add positional encoding: [seq_len, d_model]
        embeddings = self.pos_encoding.forward(embeddings)

        # Encode: [seq_len, d_model]
        encoded = self.encoder(embeddings)

        # Project to vocabulary: [seq_len, vocab_size]
        # encoded @ output_proj^T
        logits = encoded.matmul(self.output_proj.tensor)

        return logits

    @property
    def syntony(self) -> float:
        """Get model syntony from encoder."""
        return self.encoder.syntony


if __name__ == "__main__":
    print("=" * 70)
    print("Pure Syntonic Transformer Test Suite")
    print("=" * 70)

    d_model = 32
    seq_len = 8

    # Create random input
    data = [random.gauss(0, 0.5) for _ in range(seq_len * d_model)]
    mode_norms = [float(i * i) for i in range(len(data))]
    x = ResonantTensor(data, [seq_len, d_model], mode_norms, 100)

    print(f"\nInput shape: {x.shape}")
    print(f"Input syntony: {x.syntony:.4f}")

    # Test 1: Single Layer
    print("\n" + "=" * 70)
    print("Test 1: PureDHTransformerLayer")
    print("=" * 70)
    layer = PureDHTransformerLayer(d_model=d_model, n_heads=4, d_ff=64)
    y = layer(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Layer syntony: {layer.syntony:.4f}")
    assert y.shape == x.shape, "Shape should be preserved"
    print("  ✓ Pass")

    # Test 2: Encoder Stack
    print("\n" + "=" * 70)
    print("Test 2: PureSyntonicTransformerEncoder (2 layers)")
    print("=" * 70)
    encoder = PureSyntonicTransformerEncoder(
        d_model=d_model, n_heads=4, n_layers=2, d_ff=64
    )
    encoded = encoder(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {encoded.shape}")
    print(f"  Encoder syntony: {encoder.syntony:.4f}")
    assert encoded.shape == x.shape, "Shape should be preserved"
    print("  ✓ Pass")

    # Test 3: Generic Encoder (embeddings provided)
    print("\n" + "=" * 70)
    print("Test 3: PureSyntonicTransformer (encoder-only, for embeddings)")
    print("=" * 70)
    model = PureSyntonicTransformer(
        d_model=d_model, n_heads=4, n_layers=2, d_ff=64, output_dim=10
    )
    logits = model(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  Model syntony: {model.syntony:.4f}")
    print("  Note: Output is average-pooled over sequence dimension")
    print("  ✓ Pass")

    # Test 4: Language Model (with winding embeddings)
    print("\n" + "=" * 70)
    print("Test 4: PureSyntonicTransformerLM (winding embeddings)")
    print("=" * 70)
    lm = PureSyntonicTransformerLM(
        vocab_size=100,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=128,
        embedding_type="winding",
    )

    # Test with token indices
    tokens = [5, 10, 15, 20, 25, 30, 35, 40]
    lm_logits = lm(tokens)

    print(f"  Input tokens: {tokens}")
    print(f"  Output shape: {lm_logits.shape}")
    print(f"  Expected:     [{len(tokens)}, {lm.vocab_size}]")
    print(f"  Model syntony: {lm.syntony:.4f}")
    assert lm_logits.shape == [len(tokens), lm.vocab_size], "Shape mismatch"
    print("  ✓ Pass")

    # Test 5: Language Model (with syntonic embeddings)
    print("\n" + "=" * 70)
    print("Test 5: PureSyntonicTransformerLM (syntonic embeddings)")
    print("=" * 70)
    lm2 = PureSyntonicTransformerLM(
        vocab_size=50,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        embedding_type="syntonic",
    )

    tokens2 = [1, 5, 10]
    lm2_logits = lm2(tokens2)

    print(f"  Input tokens: {tokens2}")
    print(f"  Output shape: {lm2_logits.shape}")
    print(f"  Expected:     [{len(tokens2)}, {lm2.vocab_size}]")
    print(f"  Model syntony: {lm2.syntony:.4f}")
    assert lm2_logits.shape == [len(tokens2), lm2.vocab_size], "Shape mismatch"
    print("  ✓ Pass")

    print("\n" + "=" * 70)
    print("✓ All tests passed! Pure Syntonic Transformer verified!")
    print("=" * 70)
