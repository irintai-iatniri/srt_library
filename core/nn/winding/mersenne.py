"""
Mersenne Stability - Implements Axiom 6 (The Principle of Prime Syntony).

Determines stability of winding modes based on the primality of
Mersenne Harmonic Volumes M_p = 2^p - 1.

Stable Generations: p = 2, 3, 5, 7, 13, 17...
Unstable Barrier: p = 11 (M_11 = 2047 = 23 * 89)
"""

from typing import List

from ... import sn
from ..._core import ResonantTensor
from ..winding.fibonacci_hierarchy import FibonacciHierarchy


class MersenneOracle:
    """
    Oracle for querying the stability of winding recursion depths.
    """

    # Known Mersenne primes indices p
    STABLE_INDICES = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}

    @staticmethod
    def is_stable(p: int) -> bool:
        """Check if recursion depth p is stable (Axiom 6)."""
        return p in MersenneOracle.STABLE_INDICES

    @staticmethod
    def get_stability_factor(p: int) -> float:
        """
        Returns stability coefficient:
        1.0 = Perfectly Stable (M_p is Prime)
        0.0 = Totally Unstable (M_p is Composite barrier)
        """
        if p in MersenneOracle.STABLE_INDICES:
            return 1.0
        if p == 11:
            return 0.0  # The Great Barrier
        return 0.0


class MersenneStabilityGate(sn.Module):
    """
    Neural implementation of the Generation Limit.
    Acts as a 'Physics Filter' to block unstable generation depths.
    """

    def __init__(self, recursion_depth: int):
        super().__init__()
        self.p = recursion_depth
        self.stability = MersenneOracle.get_stability_factor(self.p)

        # Stability is a fixed geometric constant
        # We register it as a buffer so it saves with the model but doesn't update via gradient
        self.register_buffer("stability_factor", ResonantTensor([self.stability], [1]))

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        if self.stability == 1.0:
            return x
        elif self.stability == 0.0:
            return x.zeros_like()
        else:
            return x.scalar_mul(self.stability)


class MersenneHierarchy(FibonacciHierarchy):
    """
    Network topology that strictly follows Stable Mersenne Generations.

    Maps logical layers to Prime Winding Depths:
    Layer 0 -> p=2 (Gen 1)
    Layer 1 -> p=3 (Gen 2)
    Layer 2 -> p=5 (Gen 3)
    Layer 3 -> p=7 (Heavy Sector)
    Layer 4 -> p=13 (Gauge Boson Scale)
    """

    # Sequence of primes used for generations
    P_SEQUENCE = [2, 3, 5, 7, 13, 17, 19, 31]

    def get_layer_p(self, logical_layer: int) -> int:
        """Get the winding depth p for a given logical network layer."""
        if logical_layer >= len(self.P_SEQUENCE):
            raise ValueError(
                "Exceeded maximum stable generations in Standard Model scope"
            )
        return self.P_SEQUENCE[logical_layer]

    def _fib_at_index(self, n: int) -> int:
        """Helper to get the nth Fibonacci number safely."""
        # We use the parent class's generator
        seq = self._fibonacci(n + 1)
        return seq[-1]

    def get_layer_dims_practical(self, base_dim: int) -> List[int]:
        """
        Dimensions scale by the Fibonacci shadow of the winding depth p.

        We use F_p instead of M_p = 2^p - 1 because M_p grows too rapidly
        for practical tensor dimensions (e.g., M_13 = 8191).
        """
        dims = []
        for i in range(self.max_depth):
            p = self.get_layer_p(i)

            # Use Fibonacci(p) as the tractable projection of the volume
            scale_factor = self._fib_at_index(p)

            dims.append(base_dim * scale_factor)

        return dims

    def get_layer_dims_theory(
        self, base_dim: int, use_true_volume: bool = False
    ) -> List[int]:
        """
        Get dimensions for each layer.

        Args:
            base_dim: The starting dimension unit.
            use_true_volume: If True, scales by M_p = 2^p - 1 (Huge!).
                             If False, scales by F_p (Fibonacci Shadow, manageable).
        """
        dims = []
        for i in range(self.max_depth):
            p = self.get_layer_p(i)

            if use_true_volume:
                # The exact Mersenne Volume (Explodes quickly!)
                # p=13 -> scale=8191
                scale_factor = (2**p) - 1
            else:
                # The Golden Shadow (Manageable)
                # p=13 -> scale=233
                scale_factor = self._fib_at_index(p)

            dims.append(base_dim * scale_factor)

        return dims
