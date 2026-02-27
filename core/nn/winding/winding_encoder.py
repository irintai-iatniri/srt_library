"""
PureWindingEncoder: Maps winding states to resonant embeddings.

"""

import math
import random

from ..._core import GoldenExact, ResonantTensor, enumerate_windings


class PureWindingEncoder:
    """
    Pure Python Winding Encoder.
    Uses ResonantTensor as an embedding table.
    """

    def __init__(self, max_n=5, embed_dim=64, precision=100):
        self.max_n = max_n
        self.embed_dim = embed_dim
        self.precision = precision

        # 1. Enumerate all possible windings within max_n
        self.windings = enumerate_windings(max_n)
        self.num_windings = len(self.windings)

        # 2. Map winding state to index
        self.winding_to_idx = {self._key(w): i for i, w in enumerate(self.windings)}

        # 3. Create embedding table as a ResonantTensor
        # Initialize with Golden Measure: weight ~ exp(-|n|²/2φ)
        lattice = []
        mode_norms = []
        PHI = (1 + math.sqrt(5)) / 2

        for w in self.windings:
            norm_sq = w.norm_squared
            # Scaled initialization: small random integers, attenuated by norm
            scale = math.exp(-norm_sq / (2 * PHI))

            for _ in range(embed_dim):
                # Random integer components (-1, 0, 1) scaled by golden measure logic
                # For pure lattice, we just use small integers
                if random.random() < scale:
                    a = random.randint(-1, 1)
                    b = 0
                else:
                    a, b = 0, 0
                lattice.append(GoldenExact.from_integers(a, b))
                mode_norms.append(float(norm_sq))

        self.weight = ResonantTensor.from_golden_exact(
            lattice, [self.num_windings, embed_dim], mode_norms
        )

    def _key(self, w):
        return (w.n7, w.n8, w.n9, w.n10)

    def encode(self, winding_states):
        """
        Encode a list of WindingState objects into a ResonantTensor.

        Args:
            winding_states: List of WindingState

        Returns:
            ResonantTensor of shape [batch, embed_dim]
        """
        indices = []

        for w in winding_states:
            idx = self.winding_to_idx.get(self._key(w))
            if idx is None:
                raise ValueError(f"Winding {w} out of range (max_n={self.max_n})")
            indices.append(idx)

        # Use native index_select (now available in backend)
        return self.weight.index_select(indices, 0)

    def __repr__(self):
        return f"PureWindingEncoder(max_n={self.max_n}, num_windings={self.num_windings}, embed_dim={self.embed_dim})"
