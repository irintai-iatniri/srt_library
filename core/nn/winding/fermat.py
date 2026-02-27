"""
Fermat Force Selection - Implements Pillar III (The Architect).

Determines valid interaction types (Forces) based on Fermat Primality.
F_n = 2^(2^n) + 1

Valid Forces: n=0, 1, 2, 3, 4 (Primes)
Forbidden: n=5 (Composite: Euler's Counterexample) -> The Gauge Limit.
"""

from ... import sn
from ..._core import ResonantTensor


from enum import Enum

class FermatForce(Enum):
    STRONG = 0  # F_0 = 3   (Color Triality)
    ELECTROWEAK = 1  # F_1 = 5   (Symmetry Breaking)
    DARK = 2  # F_2 = 17  (Topological Firewall)
    GRAVITY = 3  # F_3 = 257 (Geometric Container)
    VERSAL = 4  # F_4 = 65537 (Syntonic Repulsion)
    # n=5 is mathematically forbidden (Interaction Collapse)


class FermatInteractionGate(sn.Module):
    """
    Filters interaction channels.
    If a system tries to evolve a 6th force (index 5), this gate
    dissipates the energy because F_5 is composite.
    """

    def __init__(self, force_index: int):
        super().__init__()
        self.force_index = force_index
        # Hardcoded primality check for F_n
        # F_0..F_4 are prime. F_5 is composite.
        self.is_valid = force_index < 5

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        if self.is_valid:
            return x
        else:
            # The Geometry Factorizes: Coherence cannot be maintained.
            # Return disconnected noise or zero
            return x.zeros_like()
