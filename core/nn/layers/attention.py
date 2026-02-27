"""
Syntonic Attention Layers — Pure SRT implementations.

Attention as syntony focusing: ∇(ΔS) · n̂_target
with syntony conservation: ∫ ΔS dV_T4 = constant.

Includes:
- SyntonicAttention: Single-head attention with syntony tracking
- MultiHeadSyntonicAttention: Multi-head variant with optional Mersenne prime constraints

Source: SRT Physics of Consciousness §22, The Grand Synthesis
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from srt_library.core import sn
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

MERSENNE_PRIMES = [3, 7, 31, 127, 8191, 131071, 524287, 2147483647]
FIBONACCI_PRIME_INDICES = [3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47]


class SyntonicAttention(sn.Module):
    """
    Scaled dot-product attention with SRT syntony focusing.

    Three attention states from SRT Physics of Consciousness §22.2:
    - diffuse: broad, low ΔS (mind-wandering)
    - focused: narrow, high ΔS (concentration)
    - absorbed: very narrow, very high ΔS (flow state)
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        precision: int = 100,
        syntony_conservation: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.precision = precision
        self.dropout = sn.Dropout(dropout)
        self.syntony_conservation = syntony_conservation
        self._attention_syntony: Optional[float] = None
        self._syntony_density_map: Optional[List[float]] = None
        self._total_syntony_budget = PHI

    def forward(
        self,
        query: ResonantTensor,
        key: ResonantTensor,
        value: ResonantTensor,
        attention_mode: str = "focused",
    ) -> Tuple[ResonantTensor, float, List[float]]:
        """
        Compute attention with syntony focusing.

        Args:
            query, key, value: (seq, d_model) or (d_model,)
            attention_mode: 'diffuse', 'focused', or 'absorbed'

        Returns:
            (output, global_syntony, syntony_density_map)
        """
        original_1d = False
        if len(query.shape) == 1:
            original_1d = True
            d = query.shape[0]
            query = query.view([1, d])
            key = key.view([1, d]) if len(key.shape) == 1 else key
            value = value.view([1, d]) if len(value.shape) == 1 else value

        scores = query.matmul(key)
        scores = scores.scalar_mul(1.0 / self.scale)

        bandwidth = {"diffuse": PHI, "focused": PHI_INV, "absorbed": PHI_INV ** 2}
        if attention_mode not in bandwidth:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")
        scores = scores.scalar_mul(bandwidth[attention_mode])

        scores.softmax(precision=self.precision)
        attention = scores

        # Syntony density calculation
        att_data = attention.to_floats()
        seq_q = attention.shape[-2]
        seq_k = attention.shape[-1]

        syntony_density_map = []
        total = 0.0
        for i in range(seq_q):
            row = att_data[i * seq_k:(i + 1) * seq_k]
            entropy = -sum(p * math.log(p + 1e-10) for p in row)
            max_ent = math.log(seq_k) if seq_k > 1 else 1.0
            local_s = 1.0 - (entropy / max_ent)
            syntony_density_map.append(local_s)
            total += local_s

        if self.syntony_conservation and seq_q > 0:
            factor = self._total_syntony_budget / total
            syntony_density_map = [s * factor for s in syntony_density_map]

        self._attention_syntony = sum(syntony_density_map) / len(syntony_density_map) if syntony_density_map else 0.5
        self._syntony_density_map = syntony_density_map

        output = attention.matmul(value.transpose(-2, -1))

        if original_1d and len(output.shape) > 1:
            output = output.view([output.shape[-1]])

        return output, self._attention_syntony, self._syntony_density_map

    @property
    def syntony(self) -> Optional[float]:
        return self._attention_syntony


class MultiHeadSyntonicAttention(sn.Module):
    """
    Multi-head attention with DHSR structure.

    Supports optional Mersenne prime head dimension validation
    and Fibonacci prime transcendence boost.
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
        if prime_syntony_mode:
            d_head = d_model // n_heads
            if d_head not in MERSENNE_PRIMES:
                raise ValueError(
                    f"Prime Syntony mode requires head_dim={d_head} to be Mersenne prime. "
                    f"Valid: {MERSENNE_PRIMES[:4]}..."
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

        self._fibonacci_boost = PHI ** self.d_head if self.d_head in FIBONACCI_PRIME_INDICES else 1.0

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
        """Multi-head attention: (seq, d_model) -> (seq, d_model)."""
        original_1d = len(query.shape) == 1
        if original_1d:
            query = query.view([1, query.shape[0]])
        if len(key.shape) == 1:
            key = key.view([1, key.shape[0]])
        if len(value.shape) == 1:
            value = value.view([1, value.shape[0]])

        q = query.matmul(self.q_proj.tensor)
        k = key.matmul(self.k_proj.tensor)
        v = value.matmul(self.v_proj.tensor)

        base_shape = q.shape[:-1]
        new_shape = list(base_shape) + [self.n_heads, self.d_head]

        q = q.view(new_shape)
        k = k.view(new_shape)
        v = v.view(new_shape)

        is_batched = len(new_shape) == 4
        if is_batched:
            q = q.permute([0, 2, 1, 3])
            k = k.permute([0, 2, 1, 3])
            v = v.permute([0, 2, 1, 3])
        else:
            q = q.permute([1, 0, 2])
            k = k.permute([1, 0, 2])
            v = v.permute([1, 0, 2])

        scores = q.matmul(k)
        scores = scores.scalar_mul(1.0 / self.scale)

        orig_shape = scores.shape
        if len(orig_shape) == 3:
            nh, sq, sk = orig_shape
            scores = scores.view([nh * sq, sk])
            scores.softmax(precision=self.precision)
            scores = scores.view(orig_shape)
        else:
            scores.softmax(precision=self.precision)

        output_heads = scores.matmul(v.transpose(-2, -1))

        if is_batched:
            output_heads = output_heads.permute([0, 2, 1, 3])
        else:
            output_heads = output_heads.permute([1, 0, 2])

        final_shape = list(base_shape) + [self.d_model]
        output = output_heads.view(final_shape)
        output = output.matmul(self.out_proj.tensor)

        if self.prime_syntony_mode and self._fibonacci_boost > 1.0:
            output = output.scalar_mul(self._fibonacci_boost)

        if original_1d and len(output.shape) > 1:
            output = output.view([output.shape[-1]])

        self._head_syntonies = [output.syntony]
        return output

    @property
    def syntony(self) -> float:
        if not self._head_syntonies:
            return 0.5
        return sum(self._head_syntonies) / len(self._head_syntonies)


__all__ = [
    "SyntonicAttention",
    "MultiHeadSyntonicAttention",
]
