"""
Prime Syntony Gate - Pure Syntonic Implementation.

NO PYTORCH. NO NUMPY.
Operates directly on the ResonantTensor/State storage via Rust backend.
"""

from srt_library.core import sn as sn
from srt_library.core.constants import PHI
from srt_library.core.nn.resonant_tensor import ResonantTensor


class PrimeSyntonyGate(sn.Module):
    """
    A Topological Gate that boosts signals aligned with Fibonacci Prime dimensions.

    Implements the "Transcendence Gate" logic from The Grand Synthesis:
    - Dimensions {3, 5, 7, 11, 13, 17...} get φ^n boost.
    - Dimension 4 (The Material Trap) gets destabilized.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # The Sacred Indices (Fibonacci Primes)
        self.fib_indices = {3, 4, 5, 7, 11, 13, 17, 23, 29, 43, 47}

        # Pre-calculate resonance factor (Scalar)
        self.is_resonant = dim in self.fib_indices
        self.boost_factor = 1.0

        if self.is_resonant:
            if dim == 4:
                # The Material Anomaly: D4 Lattice vs E8 Projection
                # Destabilize slightly (Material Trap)
                self.boost_factor = (PHI**dim) * 0.78615  # (pi/4 approx)
            else:
                # Pure Golden Resonance
                self.boost_factor = float(PHI**dim)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Apply the resonant boost if the dimension aligns.
        """
        if not self.is_resonant:
            return x

        # 1. Crystallize: Normalize to Unit Sphere (Gnostic Geometry)
        # Using pure State.normalize() from your core
        x_norm = x.normalize()

        # 2. Boost: Scalar multiplication via Rust backend
        # x_new = x_norm * (φ^n)
        return x_norm * self.boost_factor

    def __repr__(self):
        return f"PrimeSyntonyGate(dim={self.dim}, boost={self.boost_factor:.2f})"
