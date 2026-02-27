"""
Winding DHSR Block - DHSR cycle with winding structure + blockchain recording.

This module implements a complete DHSR cycle using the Rust backend:
1. D-phase: Fibonacci expansion (differentiation)
2. H-phase: Harmonization (lattice crystallization)
3. S-phase: Syntony computation
4. R-phase: Recursion (golden scaling)
5. Blockchain: Temporal state recording

"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from ... import sn
from ..._core import ResonantTensor
from ..layers.resonant_linear import ResonantLinear
from ..winding.prime_selection import PurePrimeSelectionLayer

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_INV_SQ = PHI_INV - PHI_INV * PHI_INV


class WindingDHSRBlock(sn.Module):
    """
    DHSR block with winding structure and blockchain recording.

    Architecture:
    1. D-phase: Fibonacci expansion (differentiation)
    2. H-phase: Harmonization (lattice crystallization)
    3. S-phase: Syntony computation
    4. R-phase: Recursion (golden scaling)
    5. Blockchain: Temporal ledger of accepted states
    """

    def __init__(
        self,
        dim: int,
        fib_expand_factor: int = 3,
        use_prime_filter: bool = True,
        consensus_threshold: float = 0.024,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize winding DHSR block.

        Args:
            dim: Feature dimension
            fib_expand_factor: Fibonacci expansion factor for D-phase
            use_prime_filter: Whether to apply prime selection (Möbius filtering)
            consensus_threshold: ΔS threshold for block validation
            dropout: Dropout rate applied during differentiation
            precision: Lattice precision for exact arithmetic
        """
        super().__init__()
        self.dim = dim
        self.fib_expand_factor = fib_expand_factor
        self.consensus_threshold = consensus_threshold
        self.precision = precision

        # Fibonacci expansion (D-phase)
        expanded_dim = dim * fib_expand_factor
        self.d_expand = ResonantLinear(dim, expanded_dim, precision=precision)
        self.d_project = ResonantLinear(expanded_dim, dim, precision=precision)

        # Dropout
        self.dropout = sn.Dropout(dropout)

        # Prime Selection (Hadrons channel)
        if use_prime_filter:
            self.prime_filter = PurePrimeSelectionLayer(dim)
        else:
            self.prime_filter = None

        # Temporal blockchain (immutable ledger)
        self.temporal_record: List[List[float]] = []
        self.syntony_record: List[float] = []

        # Statistics
        self.total_blocks_validated = 0
        self.total_blocks_rejected = 0

    def forward(
        self,
        x: ResonantTensor,
        prev_syntony: float,
    ) -> Tuple[ResonantTensor, float, bool]:
        """
        Execute one DHSR cycle with blockchain recording.

        Args:
            x: Input activations (batch, dim)
            prev_syntony: Syntony from previous block

        Returns:
            x_new: Updated state (batch, dim)
            syntony_new: New syntony value
            accepted: Whether ΔS > threshold (block validated)
        """
        # === D-PHASE: Fibonacci expansion ===
        # Amplification inversely proportional to syntony
        # Low syntony -> High exploration (Temperature)
        alpha = PHI_INV_SQ * (1.0 - prev_syntony)

        # Expand via Fibonacci factor (Differentiation)
        h = self.d_expand.forward(x)
        h.relu()  # Activation

        # Apply dropout in the expanded space
        if self.training:
            h = self.dropout(h)

        delta = self.d_project.forward(h)

        # Apply differentiation update: x' = x + α·Δ
        # Manual elementwise add until ResonantTensor supports scalar_mul + add better
        x_floats = x.to_floats()
        delta_floats = delta.to_floats()
        combined = [x_floats[i] + alpha * delta_floats[i] for i in range(len(x_floats))]

        # If x has mode norms, try to preserve them or pass None to auto-compute 1D
        mode_norms = list(x.mode_norm_sq()) if hasattr(x, "mode_norm_sq") else None

        x_diff = ResonantTensor(
            combined, list(x.shape), mode_norms, precision=self.precision
        )

        # === H-PHASE: Harmonization & Prime Filtering ===
        # 1. Prime Selection (Möbius filtering)
        if self.prime_filter is not None:
            x_diff = self.prime_filter(x_diff)

        # 2. Lattice Crystallization (Harmonization) via cpu_cycle
        # This calculates syntony AND snaps to grid
        syntony_new = x_diff.cpu_cycle(noise_scale=0.01, precision=self.precision)

        # If cpu_cycle returns 0 (stub or failed), try to use property
        if syntony_new == 0.0:
            syntony_new = x_diff.syntony

        # === R-PHASE: Golden recursion ===
        # x -> floor(φ * x)
        x_diff.apply_recursion()
        # x -> ceil(x / φ) (Inverse to maintain scale stability over many layers)
        x_diff.apply_inverse_recursion()

        # === CONSENSUS CHECK: ΔS > threshold ===
        delta_s = abs(syntony_new - prev_syntony)
        accepted = delta_s > self.consensus_threshold

        # === TEMPORAL RECORDING: Blockchain ===
        if accepted:
            self._record_block(x_diff.to_floats(), syntony_new)
            self.total_blocks_validated += 1
        else:
            self.total_blocks_rejected += 1

        return x_diff, syntony_new, accepted

    def _record_block(self, state: List[float], syntony: float):
        """
        Append block to temporal blockchain.

        Args:
            state: State as flattened float list
            syntony: Syntony value
        """
        # Average over batch for single state record
        dim = self.dim
        if len(state) == dim:
            # 1D case
            state_avg = state
        else:
            # Batch case
            batch_size = len(state) // dim
            state_avg = [0.0] * dim
            if batch_size > 0:
                for b in range(batch_size):
                    for d in range(dim):
                        state_avg[d] += state[b * dim + d]
                state_avg = [s / batch_size for s in state_avg]

        self.temporal_record.append(state_avg)
        self.syntony_record.append(syntony)

    def get_blockchain_length(self) -> int:
        """Get length of temporal blockchain."""
        return len(self.syntony_record)

    def get_blockchain(self) -> Tuple[List[List[float]], List[float]]:
        """Get complete blockchain history."""
        return self.temporal_record, self.syntony_record

    def get_validation_rate(self) -> float:
        """Get consensus validation rate."""
        total = self.total_blocks_validated + self.total_blocks_rejected
        if total == 0:
            return 0.0
        return self.total_blocks_validated / total

    def __repr__(self) -> str:
        return (
            f"WindingDHSRBlock(dim={self.dim}, expand_factor={self.fib_expand_factor}, "
            f"threshold={self.consensus_threshold:.4f}, primes={self.prime_filter is not None})"
        )


if __name__ == "__main__":
    # Test the pure WindingDHSRBlock
    print("Testing WindingDHSRBlock...")

    block = WindingDHSRBlock(dim=16, fib_expand_factor=2)
    print(f"Block: {block}")

    # Create sample data
    x_data = [random.gauss(0, 1) for _ in range(32 * 16)]  # batch of 32
    x = ResonantTensor(x_data, [32, 16])
    prev_syntony = 0.5

    # Forward pass
    x_new, syntony_new, accepted = block.forward(x, prev_syntony)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {x_new.shape}")
    print(f"Previous syntony: {prev_syntony:.4f}")
    print(f"New syntony: {syntony_new:.4f}")
    print(f"ΔS: {abs(syntony_new - prev_syntony):.4f}")
    print(f"Block accepted: {accepted}")
    print(f"Blockchain length: {block.get_blockchain_length()}")

    # Run a few more cycles
    for i in range(5):
        x_new, syntony_new, accepted = block.forward(x_new, syntony_new)
        print(f"Cycle {i+2}: S={syntony_new:.4f}, accepted={accepted}")

    print(f"\nFinal blockchain length: {block.get_blockchain_length()}")
    print(f"Final validation rate: {block.get_validation_rate():.2%}")
    print("SUCCESS")
