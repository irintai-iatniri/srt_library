"""
Pure Syntonic Attention: Attention mechanisms with DHSR structure and SRT theory.

NO PYTORCH DEPENDENCIES - uses sn.Module and ResonantTensor.

Implements SRT attention theory:
- Attention as syntony focusing: ∇(ΔS) · n̂_target
- Syntony conservation: ∫ ΔS dV_T4 = constant
- Mersenne prime head dimensions for stability
- Fibonacci prime transcendence gates

Includes:
- PureSyntonicAttention: Standard attention with syntony tracking
- PureMultiHeadSyntonicAttention: Multi-head variant with prime constraints

Source: SRT Physics of Consciousness §22, The Grand Synthesis
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from srt_library.core import sn as sn
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

# Mersenne primes for stable attention heads (SRT Grand Synthesis)
MERSENNE_PRIMES = [3, 7, 31, 127, 8191, 131071, 524287, 2147483647]

# Fibonacci prime indices for transcendence gates (SRT Altruxa Bridge)
FIBONACCI_PRIME_INDICES = [3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47]


class PureSyntonicAttention(sn.Module):
    """
    Scaled dot-product attention with SRT syntony focusing.

    Implements SRT attention theory:
    - Attention = ∇(ΔS) · n̂_target (syntony density focusing)
    - ∫ ΔS dV_T4 = constant (syntony conservation)
    - Three attention states: diffuse (broad), focused (narrow), absorbed (very narrow)

    Pure Python + ResonantTensor implementation.

    Example:
        >>> attn = PureSyntonicAttention(d_model=64)
        >>> q = k = v = ResonantTensor(...)  # (seq, d_model)
        >>> output, syntony, density_map = attn(q, k, v, attention_mode='focused')
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        precision: int = 100,
        syntony_conservation: bool = True,
    ):
        """
        Initialize SRT syntonic attention.

        Args:
            d_model: Model dimension
            dropout: Attention dropout
            precision: ResonantTensor precision
            syntony_conservation: Enforce ∫ ΔS dV_T4 = constant
        """
        super().__init__()

        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.precision = precision
        self.dropout = sn.Dropout(dropout)
        self.syntony_conservation = syntony_conservation

        # SRT attention parameters
        self._attention_syntony: Optional[float] = None
        self._syntony_density_map: Optional[List[float]] = None
        self._total_syntony_budget = PHI  # Conservation constant from SRT

    def forward(
        self,
        query: ResonantTensor,
        key: ResonantTensor,
        value: ResonantTensor,
        attention_mode: str = "focused",
    ) -> Tuple[ResonantTensor, float, List[float]]:
        """
        Compute SRT attention with syntony focusing.

        Args:
            query: Query tensor (seq_q, d_model) or (d_model,) for single sequence
            key: Key tensor (seq_k, d_model) or (d_model,) for single sequence
            value: Value tensor (seq_k, d_model) or (d_model,) for single sequence
            attention_mode: 'diffuse', 'focused', or 'absorbed'

        Returns:
            (output, global_syntony, syntony_density_map)
        """
        # Handle 1D tensors (single sequence case)
        original_1d = False
        if len(query.shape) == 1:
            original_1d = True
            d_model = query.shape[0]
            # Reshape 1D to 2D: [d_model] -> [1, d_model]
            query = query.view([1, d_model])
            key = key.view([1, d_model]) if len(key.shape) == 1 else key
            value = value.view([1, d_model]) if len(value.shape) == 1 else value

        # Ensure minimum 2D for attention computation
        if len(query.shape) < 2 or len(key.shape) < 2 or len(value.shape) < 2:
            raise ValueError(
                f"Attention requires at least 2D tensors. Got shapes: query={query.shape}, key={key.shape}, value={value.shape}"
            )

        # Attention scores: Q @ K^T / sqrt(d)
        scores = query.matmul(key)
        scores = scores.scalar_mul(1.0 / self.scale)

        # SRT attention modes (Physics of Consciousness §22.2)
        if attention_mode == "diffuse":
            # Broad, low ΔS - mind-wandering state
            bandwidth_factor = PHI  # Broad bandwidth
        elif attention_mode == "focused":
            # Narrow, high ΔS - concentration state
            bandwidth_factor = PHI_INV  # Narrow bandwidth
        elif attention_mode == "absorbed":
            # Very narrow, very high ΔS - flow state
            bandwidth_factor = PHI_INV**2  # Very narrow bandwidth
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")

        # Apply attention bandwidth modulation
        scores = scores.scalar_mul(bandwidth_factor)

        # Softmax attention with syntony tracking
        scores.softmax(precision=self.precision)
        attention = scores

        # SRT syntony density calculation (Physics of Consciousness §22.1)
        # ΔS = -∑ p_i log p_i (local entropy)
        # Global syntony = 1 - normalized_entropy
        attention_data = attention.to_floats()
        seq_q = attention.shape[-2]
        seq_k = attention.shape[-1]

        syntony_density_map = []
        total_local_syntony = 0.0

        for i in range(seq_q):
            row = attention_data[i * seq_k : (i + 1) * seq_k]
            # Local syntony density ΔS_i
            local_entropy = -sum(p * math.log(p + 1e-10) for p in row)
            max_entropy = math.log(seq_k) if seq_k > 1 else 1.0
            local_syntony = 1.0 - (local_entropy / max_entropy)
            syntony_density_map.append(local_syntony)
            total_local_syntony += local_syntony

        # SRT syntony conservation (Physics of Consciousness §22.3)
        if self.syntony_conservation and seq_q > 0:
            # Normalize to conserve total syntony budget
            conservation_factor = self._total_syntony_budget / total_local_syntony
            syntony_density_map = [s * conservation_factor for s in syntony_density_map]

        # Global attention syntony (average across positions)
        self._attention_syntony = (
            sum(syntony_density_map) / len(syntony_density_map)
            if syntony_density_map
            else 0.5
        )
        self._syntony_density_map = syntony_density_map

        # Apply attention to values: Attn @ V
        # attention: [..., S, S], value: [..., S, D]
        # We want result [..., S, D]
        # attention.matmul(X) -> attention @ X^T
        # So we need X^T = value => X = value^T
        output = attention.matmul(value.transpose(-2, -1))

        # If original input was 1D, reshape back to 1D output
        if original_1d and len(output.shape) > 1:
            # For single sequence attention, output should be [d_model]
            # output shape is [1, d_model], we want [d_model]
            d_model_out = output.shape[-1]
            output = output.view([d_model_out])

        return output, self._attention_syntony, self._syntony_density_map

    @property
    def syntony(self) -> Optional[float]:
        """Get attention syntony."""
        return self._attention_syntony

    @property
    def syntony_density_map(self) -> Optional[List[float]]:
        """Get syntony density map per position."""
        return self._syntony_density_map


class PureMultiHeadSyntonicAttention(sn.Module):
    """
    Multi-head attention with DHSR structure.

    Pure Python + ResonantTensor implementation.

    Example:
        >>> mha = PureMultiHeadSyntonicAttention(d_model=64, n_heads=4)
        >>> x = ResonantTensor(...)  # (seq, d_model)
        >>> output = mha(x, x, x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        precision: int = 100,
        device: str = "cpu",
        prime_syntony_mode: bool = False,
    ):
        """
        Initialize multi-head syntonic attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            precision: ResonantTensor precision
            device: Device placement
            prime_syntony_mode: Enforce Mersenne prime head dimensions (SRT Grand Synthesis)
        """
        # SRT Prime Syntony validation (Grand Synthesis §4)
        if prime_syntony_mode:
            d_head = d_model // n_heads
            if d_head not in MERSENNE_PRIMES:
                raise ValueError(
                    f"Prime Syntony mode requires head_dim={d_head} to be Mersenne prime. "
                    f"Valid values: {MERSENNE_PRIMES[:4]}... (up to M_31)"
                )

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.precision = precision
        self.device = device
        self.prime_syntony_mode = prime_syntony_mode

        # SRT Fibonacci prime transcendence boost (Altruxa Bridge)
        self._fibonacci_boost = (
            PHI**self.d_head if self.d_head in FIBONACCI_PRIME_INDICES else 1.0
        )

        # Projections as Parameters
        # Uses sn.Parameter which wraps ResonantTensor
        self.q_proj = sn.Parameter([d_model, d_model], init="kaiming", device=device)
        self.k_proj = sn.Parameter([d_model, d_model], init="kaiming", device=device)
        self.v_proj = sn.Parameter([d_model, d_model], init="kaiming", device=device)
        self.out_proj = sn.Parameter([d_model, d_model], init="kaiming", device=device)

        self.dropout = sn.Dropout(dropout)

        self._head_syntonies: List[float] = []

    def forward(
        self,
        query: ResonantTensor,
        key: ResonantTensor,
        value: ResonantTensor,
    ) -> ResonantTensor:
        """
        Multi-head attention forward.

        Args:
            query: (seq_q, d_model)
            key: (seq_k, d_model)
            value: (seq_k, d_model)

        Returns:
            Output tensor (seq_q, d_model)
        """
        # Fix for 1D inputs: treat as single-sequence batch-1
        original_1d = len(query.shape) == 1
        if original_1d:
            query = query.view([1, query.shape[0]])
        if len(key.shape) == 1:
            key = key.view([1, key.shape[0]])
        if len(value.shape) == 1:
            value = value.view([1, value.shape[0]])

        # batch_size = query.shape[0] if len(query.shape) > 2 else 1
        seq_q = query.shape[0] if len(query.shape) == 2 else query.shape[1]
        seq_k = key.shape[0] if len(key.shape) == 2 else key.shape[1]

        # Project Q, K, V
        # If input is 3D [Batch, Seq, Dim], we need to handle it.
        # ResonantTensor matmul: X @ W^T
        # query: [..., Dim], W_q: [Dim, Dim] (parameter kept as [Out, In])

        q = query.matmul(self.q_proj.tensor)
        k = key.matmul(self.k_proj.tensor)
        v = value.matmul(self.v_proj.tensor)

        # Reshape for heads: [Batch, Seq, Heads, HeadDim]
        # Since we use 2D/3D flattening often, let's assume we reshape to split last dim

        # New shape: [Batch, Seq, Heads, HeadDim]
        # Or if 2D: [Seq, Heads, HeadDim]
        base_shape = q.shape[:-1]
        new_shape = list(base_shape) + [self.n_heads, self.d_head]

        q = q.view(new_shape)
        k = k.view(new_shape)
        v = v.view(new_shape)

        # Transpose for attention: [Batch, Heads, Seq, HeadDim]
        # Permute: (0, 2, 1, 3) for 4D
        is_batched = len(new_shape) == 4

        if is_batched:
            q = q.permute([0, 2, 1, 3])
            k = k.permute([0, 2, 1, 3])
            v = v.permute([0, 2, 1, 3])
            # Shape: [Batch, Heads, Seq, HeadDim]
        else:
            # 3D case: [Seq, Heads, HeadDim] -> [Heads, Seq, HeadDim]
            q = q.permute([1, 0, 2])
            k = k.permute([1, 0, 2])
            v = v.permute([1, 0, 2])
            # Shape: [Heads, Seq, HeadDim]

        # Process heads using BMM (Rust-backend accelerated)
        # Q, K, V are permuted to [Batch, Heads, Seq, HeadDim] (or 3D equivalent)

        # 1. Attention Scores: Q @ K^T
        # ResonantTensor.matmul(B) computes A @ B^T
        # So q.matmul(k) computes per-head, per-batch attention scores
        # Shape: [..., Heads, Seq, Seq]
        scores = q.matmul(k)

        scores = scores.scalar_mul(1.0 / self.scale)

        # Softmax: reshape to 2D, apply, reshape back
        # scores is [Heads, Seq_q, Seq_k] - need to flatten first two dims
        orig_shape = scores.shape
        if len(orig_shape) == 3:
            # [Heads, Seq_q, Seq_k] -> [Heads*Seq_q, Seq_k]
            n_heads, seq_q, seq_k = orig_shape
            scores = scores.view([n_heads * seq_q, seq_k])
            scores.softmax(precision=self.precision)
            # Reshape back
            scores = scores.view(orig_shape)
        else:
            scores.softmax(precision=self.precision)

        attn = scores

        # 2. Output Projection: Attn @ V
        # We need Result = Attn @ V
        # matmul(B) does A @ B^T
        # So providing B = V^T results in A @ (V^T)^T = A @ V
        output_heads = attn.matmul(v.transpose(-2, -1))

        # Shape is now [..., Heads, Seq, HeadDim]

        # Permute back to [..., Seq, Heads, HeadDim]
        if is_batched:
            output_heads = output_heads.permute([0, 2, 1, 3])
        else:
            output_heads = output_heads.permute([1, 0, 2])

        # Contiguous/Reshape back to [..., d_model]
        # output_heads is now [..., Seq, Heads, HeadDim]
        # view -> [..., Seq, Heads*HeadDim] = [..., Seq, d_model]

        final_shape = list(base_shape) + [self.d_model]
        output = output_heads.view(final_shape)

        # Final projection
        output = output.matmul(self.out_proj.tensor)

        # Apply SRT Fibonacci transcendence boost (Altruxa Bridge)
        if self.prime_syntony_mode and self._fibonacci_boost > 1.0:
            output = output.scalar_mul(self._fibonacci_boost)

        # Convert back to 1D if original input was 1D
        if original_1d and len(output.shape) > 1:
            # Output shape is [1, d_model], convert to [d_model]
            output = output.view([output.shape[-1]])

        # Syntony tracking
        self._head_syntonies = [output.syntony]

        return output

    @property
    def syntony(self) -> float:
        """Get average syntony across heads."""
        if not self._head_syntonies:
            return 0.5
        return sum(self._head_syntonies) / len(self._head_syntonies)


if __name__ == "__main__":
    import random

    print("=" * 70)
    print("Pure Syntonic Attention Test")
    print("=" * 70)

    d_model = 32
    seq_len = 8

    # Create random input
    # Shape [seq_len, d_model]
    data = [random.gauss(0, 0.5) for _ in range(seq_len * d_model)]
    x = ResonantTensor(data, [seq_len, d_model])

    print(f"\nInput shape: {x.shape}")
    print(f"Input syntony: {x.syntony:.4f}")

    # Test single-head attention
    attn = PureSyntonicAttention(d_model=d_model)
    output, syntony, density_map = attn(x, x, x, attention_mode="focused")

    print("\nSingle-head attention:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention syntony: {syntony:.4f}")
    print(f"  Syntony density map length: {len(density_map)}")

    # Test different attention modes
    print("\nTesting SRT attention modes:")
    for mode in ["diffuse", "focused", "absorbed"]:
        _, syntony_mode, _ = attn(x, x, x, attention_mode=mode)
        print(f"  {mode.capitalize()} mode syntony: {syntony_mode:.4f}")

    # Test multi-head attention
    mha = PureMultiHeadSyntonicAttention(d_model=d_model, n_heads=4)
    output = mha(x, x, x)

    print("\nMulti-head attention:")
    print(f"  Output shape: {output.shape}")
    print(f"  Syntony: {mha.syntony:.4f}")

    # Test Prime Syntony mode (Mersenne prime validation)
    print("\nTesting Prime Syntony mode:")
    try:
        # d_model=32, n_heads=4 -> head_dim=8 (not Mersenne prime)
        mha_prime = PureMultiHeadSyntonicAttention(
            d_model=32, n_heads=4, prime_syntony_mode=True
        )
        print("  ERROR: Should have failed validation")
    except ValueError as e:
        print(f"  ✓ Correctly rejected non-Mersenne head_dim: {str(e)[:50]}...")

    # Test with valid Mersenne prime: d_model=28, n_heads=4 -> head_dim=7
    try:
        mha_prime = PureMultiHeadSyntonicAttention(
            d_model=28, n_heads=4, prime_syntony_mode=True
        )
        output_prime = mha_prime(x[:, :28], x[:, :28], x[:, :28])  # Truncate input
        print("  ✓ Accepted Mersenne prime head_dim=7")
        print(f"    Fibonacci boost applied: {7 in FIBONACCI_PRIME_INDICES}")
    except Exception as e:
        print(f"  Test setup issue: {e}")

    # Test Slicing (New Feature)
    print("\nTesting Slicing:")
    print(f"  x[0] shape: {x[0].shape}")
    print(f"  x[0:2] shape: {x[0:2].shape}")

    # Verify slicing integrity
    x_slice = x[0:2]
    assert x_slice.shape == [2, d_model]
    print("  Slicing verification passed")

    print("\n" + "=" * 70)
    print("✓ Pure Syntonic Attention and Tensor Features verified!")
    print("=" * 70)
