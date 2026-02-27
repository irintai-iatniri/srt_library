"""
WindingNet Pure - Complete winding-aware neural network architecture (Pure Python).

This module implements the full WindingNet architecture using:
1. sn (Syntonic Network) module instead of torch.nn
2. ResonantTensor for all computations
3. Pure Python / Rust backend

Architecture:
- Winding embedding layer (maps |n₇,n₈,n₉,n₁₀⟩ to embeddings)
- Fibonacci hierarchy (layer dims grow as F_k)
- ResonantWindingDHSRBlocks (one per Fibonacci level)
- Prime selection filtering (Möbius function)
- Temporal blockchain recording
- Syntony consensus (ΔS > threshold validation)
"""

from __future__ import annotations

import math
from typing import Dict, List

from ... import sn
from ..._core import ResonantTensor, WindingState
from srt_library.theory_unique_components.syntonic_mlp import PureSyntonicLinear
from ..layers.syntonic_gate import SyntonicGate
from ..winding.fibonacci_hierarchy import FibonacciHierarchy
try:
    from srt_library.theory_unique_components.resonant.resonant_dhsr_block import ResonantWindingDHSRBlock
except ImportError:
    ResonantWindingDHSRBlock = None
try:
    from srt_library.theory_unique_components.resonant.resonant_embedding import PureResonantWindingEmbedding
except ImportError:
    PureResonantWindingEmbedding = None

PHI = (1 + math.sqrt(5)) / 2
Q_DEFICIT = 0.027395146920


class PureWindingNet(sn.Module):
    """
    Neural network operating on winding space (Pure Python).

    Architecture:
    - Winding embedding
    - Fibonacci hierarchy
    - Resonant DHSR blocks
    - Syntony consensus
    """

    def __init__(
        self,
        max_winding: int = 5,
        base_dim: int = 64,
        num_blocks: int = 3,
        output_dim: int = 2,
        use_prime_filter: bool = True,
        consensus_threshold: float = 0.024,
        dropout: float = 0.1,
        precision: int = 100,
    ):
        """
        Initialize PureWindingNet.

        Args:
            max_winding: Maximum winding number for enumeration
            base_dim: Base embedding dimension
            num_blocks: Number of DHSR blocks (Fibonacci levels)
            output_dim: Output dimension (number of classes)
            use_prime_filter: Whether to use prime selection in blocks
            consensus_threshold: ΔS threshold for block validation
            dropout: Dropout rate
            precision: ResonantTensor precision
        """
        super().__init__()

        self.max_winding = max_winding
        self.base_dim = base_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.precision = precision

        # 1. Resonant Winding embedding layer
        self.winding_embed = PureResonantWindingEmbedding(
            max_n=max_winding, embed_dim=base_dim, precision=precision
        )

        # 2. Fibonacci hierarchy
        self.fib_hierarchy = FibonacciHierarchy(num_blocks)
        layer_dims = self.fib_hierarchy.get_layer_dims(base_dim)
        self.layer_dims = layer_dims

        # 3. Resonant DHSR blocks (one per Fibonacci level)
        self.blocks = sn.ModuleList()
        for i in range(num_blocks):
            block = ResonantWindingDHSRBlock(
                dim=layer_dims[i],
                fib_expand_factor=self.fib_hierarchy.get_expansion_factor(i),
                use_prime_filter=use_prime_filter,
                consensus_threshold=consensus_threshold,
                precision=precision,
                dropout=dropout,
            )
            self.blocks.append(block)

        # 4. Transitions between Fibonacci levels
        self.transitions = sn.ModuleList()
        for i in range(num_blocks - 1):
            self.transitions.append(
                PureSyntonicLinear(layer_dims[i], layer_dims[i + 1])
            )

        # 5. Output projection
        final_dim = layer_dims[num_blocks - 1]
        self.output_proj = PureSyntonicLinear(final_dim, output_dim)

        # 6. SyntonicGate for geometric folding (non-linearity)
        # Creates the decision boundaries needed for non-linear problems like XOR
        self.gates = sn.ModuleList()
        for i in range(num_blocks):
            self.gates.append(SyntonicGate(layer_dims[i]))
        # Final gate before output projection
        self.final_gate = SyntonicGate(layer_dims[num_blocks - 1])

        # 7. Mode norms per layer (precomputed lists)
        self.mode_norms = [[float(j * j) for j in range(dim)] for dim in layer_dims]

        # Network-level metrics
        self.network_syntony = 0.0
        self.layer_syntonies = []

    def forward(self, winding_states: List[WindingState]) -> ResonantTensor:
        """
        Forward pass.

        Args:
            winding_states: List of WindingState objects

        Returns:
            predictions: Logits (ResonantTensor)
        """
        # 1. Embed windings
        x = self.winding_embed.batch_forward(winding_states)

        # Initial syntony
        syntony = 0.5
        syntonies = []

        # 2. Pass through DHSR blocks with SyntonicGate activations
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            gate = self.gates[i]

            # DHSR cycle
            x_prev = x  # Save for gating
            x, syntony_new, accepted = block(
                x,
                self.mode_norms[i],
                syntony,
            )

            # Apply SyntonicGate (geometric folding)
            # Gate adaptively mixes original and processed based on syntony
            x = gate.forward(x_prev, x)

            syntonies.append(syntony_new)
            syntony = syntony_new

            # Transition to next level
            if i < len(self.transitions):
                transition = self.transitions[i]
                x = transition(x)

        # 3. Final metrics
        self.network_syntony = sum(syntonies) / len(syntonies) if syntonies else 0.5
        self.layer_syntonies = syntonies

        # 4. Apply final SyntonicGate before output projection
        # Critical for solving non-linear problems like XOR
        x_before_gate = x
        x = self.final_gate.forward(x_before_gate, x)

        # 5. Output projection
        return self.output_proj(x)

    def get_blockchain_stats(self) -> Dict[str, float]:
        """Get blockchain and consensus statistics."""
        total_validated = 0
        total_rejected = 0
        blockchain_length = 0

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            total_validated += block.total_blocks_validated
            total_rejected += block.total_blocks_rejected
            blockchain_length += block.get_blockchain_length()

        total_cycles = total_validated + total_rejected

        return {
            "total_cycles": total_cycles,
            "validated_blocks": total_validated,
            "rejected_blocks": total_rejected,
            "validation_rate": total_validated / max(total_cycles, 1),
            "network_syntony": self.network_syntony,
            "blockchain_length": blockchain_length,
            "layer_syntonies": self.layer_syntonies,
        }

    def get_weights(self) -> List[ResonantTensor]:
        """Get all learnable weights (PureModel protocol)."""
        weights = []

        # 0. Embeddings
        weights.append(self.winding_embed.weight)

        # 1. Transitions
        for i in range(len(self.transitions)):
            layer = self.transitions[i]
            # Check for linear layer parameters
            if hasattr(layer.linear, "weight"):
                # ResonantLinear wraps weight in Parameter, so get tensor
                weights.append(layer.linear.weight.tensor)
            if hasattr(layer.linear, "bias") and layer.linear.bias is not None:
                weights.append(layer.linear.bias.tensor)

        # 2. Output projection
        if hasattr(self.output_proj.linear, "weight"):
            weights.append(self.output_proj.linear.weight.tensor)
        if (
            hasattr(self.output_proj.linear, "bias")
            and self.output_proj.linear.bias is not None
        ):
            weights.append(self.output_proj.linear.bias.tensor)

        return weights

    def set_weights(self, weights: List[ResonantTensor]) -> None:
        """Set weights (PureModel protocol)."""
        idx = 0

        # 0. Embeddings
        self.winding_embed.weight = weights[idx]
        idx += 1

        # 1. Transitions
        for i in range(len(self.transitions)):
            layer = self.transitions[i]
            if hasattr(layer.linear, "weight"):
                layer.linear.weight.tensor = weights[idx]
                idx += 1
            if hasattr(layer.linear, "bias") and layer.linear.bias is not None:
                layer.linear.bias.tensor = weights[idx]
                idx += 1

        # 2. Output projection
        if hasattr(self.output_proj.linear, "weight"):
            self.output_proj.linear.weight.tensor = weights[idx]
            idx += 1
        if (
            hasattr(self.output_proj.linear, "bias")
            and self.output_proj.linear.bias is not None
        ):
            self.output_proj.linear.bias.tensor = weights[idx]
            idx += 1

    @property
    def syntony(self) -> float:
        """Get model syntony (PureModel protocol)."""
        return self.network_syntony


if __name__ == "__main__":
    from ...srt.geometry.winding import WindingState

    print("=" * 60)
    print("PureWindingNet Test (No PyTorch)")
    print("=" * 60)

    # Create simple windings
    w1 = WindingState(7, 8, 0, 0)
    w2 = WindingState(1, 2, 0, 0)
    windings = [w1, w2]

    model = PureWindingNet(max_winding=10, base_dim=32, num_blocks=2, output_dim=2)

    # Forward pass
    logits = model(windings)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits data: {logits.to_floats()}")

    # Stats
    stats = model.get_blockchain_stats()
    print(f"Network syntony: {stats['network_syntony']:.4f}")

    print("\n✓ PureWindingNet Verified")
