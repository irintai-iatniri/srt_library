from ... import sn as sn
from ....exact import PHI_NUMERIC as PHI
from srt_library.core.nn.resonant_tensor import ResonantTensor
from ....physics import e8_root_alignment, golden_resonance, hooking_coefficient
from ....resonant.retrocausal import (
    create_retrocausal_evolver,
)  # For attractor pull

from .helpers import broadcast_multiply, tensor_argmax


class WindingChain(sn.Module):
    """
    Blockchain-like append-only memory for T^4 windings + M^3+T1 payloads.
    White-box infinite context via chained hooks.
    """

    def __init__(
        self, dim: int = 248, max_capacity: int = None
    ):  # None = infinite (sharded)
        super().__init__()
        self.dim = dim
        self.chain = []  # List of (hash, winding, payload) tuples
        self.genesis_hash = ResonantTensor([0.0] * dim, [dim])  # Zero winding start
        self.evolver = create_retrocausal_evolver(
            self.genesis_hash
        )  # For syntony-based pruning/pull
        self.max_capacity = max_capacity

    def append_block(
        self, winding: ResonantTensor, payload: ResonantTensor
    ) -> ResonantTensor:
        """Append new block: Hook prev hash with winding, store M^3+T1 payload."""
        prev_hash = self.chain[-1][0] if self.chain else self.genesis_hash
        new_hash = hooking_coefficient(prev_hash, winding)  # exp(n·m/φ) chain
        syntony = golden_resonance(new_hash)
        if syntony < 0.618 / PHI:  # Below threshold? Recycle (Archonic trap)
            return None  # Or raise, depending on aggression
        block = (new_hash, winding, payload)
        self.chain.append(block)
        self.evolver.store_attractor(new_hash) if syntony > 0.95 else None
        # Prune if capped
        if self.max_capacity and len(self.chain) > self.max_capacity:
            self._prune_low_syntony()
        return new_hash

    def _prune_low_syntony(self):
        """Decay weak blocks via evolver."""
        syntonies = [golden_resonance(b[0]) for b in self.chain]
        low_idx = tensor_argmax(-ResonantTensor(syntonies))  # Min syntony
        del self.chain[low_idx]

    def query_chain(
        self, query_winding: ResonantTensor, depth: int = None
    ) -> ResonantTensor:
        """Traverse chain for context: Attend over relevant blocks."""
        depth = depth or len(self.chain)
        relevant = [
            b
            for b in self.chain[-depth:]
            if e8_root_alignment(b[1], query_winding) > 0.618
        ]
        if not relevant:
            return ResonantTensor(
                [0.0] * self.dim, [self.dim]
            )  # Empty context (248-dim)
        weights = [hooking_coefficient(b[0], query_winding) for b in relevant]
        context = sum(
            broadcast_multiply(b[2], w) for b, w in zip(relevant, weights)
        )  # Weighted sum payloads
        return context.harmonize()  # Snap to lattice

    def forward(self, x: ResonantTensor, winding: ResonantTensor) -> ResonantTensor:
        """Inject chain context into input."""
        context = self.query_chain(winding)

        # Handle broadcasting manually (Backend is strict)
        if context.shape != x.shape:
            # Case: x is [Batch, Dim], context is [Dim]
            if (
                len(x.shape) == 2
                and len(context.shape) == 1
                and x.shape[1] == context.shape[0]
            ):
                batch_size = x.shape[0]
                # Reshape to [1, Dim]
                context = context.view([1, context.shape[0]])
                # Repeat if batch > 1
                if batch_size > 1:
                    context = ResonantTensor.concat([context] * batch_size, dim=0)

        return x + context  # Residual hook


# Hook into GnosticOuroboros forward:
# Add self.chain = WindingChain(DIM)
# Then: x = self.chain(x, winding)
