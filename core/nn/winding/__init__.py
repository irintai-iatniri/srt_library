"""
Winding Neural Networks - Number-theoretic deep learning with SRT structure.

This module provides winding-aware neural network architectures that integrate:
- T^4 torus winding states |n₇, n₈, n₉, n₁₀⟩
- Fibonacci hierarchy (golden ratio scaling)
- Prime selection (Möbius filtering)
- DHSR dynamics (differentiation-harmonization cycles)
- Temporal blockchain (immutable state recording)
- Syntony consensus (ΔS > threshold validation)
- Resonant Engine (exact Q(φ) lattice arithmetic)

Example:
    >>> from syntonic_applications.nn.winding import WindingNet
    >>> from syntonic_applications.physics.fermions.windings import *
    >>>
    >>> # Standard WindingNet (float-based DHSR)
    >>> model = WindingNet(max_winding=5, base_dim=64, num_blocks=3, output_dim=2)
    >>> windings = [ELECTRON_WINDING, MUON_WINDING, UP_WINDING]
    >>> y = model(windings)
    >>>
    >>> # After training, crystallize to Q(φ) and use exact inference
    >>> model.crystallize_weights(precision=100)
    >>> y_exact = model.forward_exact(windings)
"""

from ..winding.dhsr_block import WindingDHSRBlock
from ..winding.embedding import WindingStateEmbedding
from ..winding.fibonacci_hierarchy import FibonacciHierarchy

# Pure (PyTorch-free) versions
from ..winding.prime_selection import PurePrimeSelectionLayer
from ..winding.prime_selection import (
    PurePrimeSelectionLayer as PrimeSelectionLayer,
)
# Lazy imports to avoid circular dependency (resonant_dhsr_block → nn.winding.prime_selection → here)
_ResonantWindingDHSRBlock = None
_PureResonantWindingEmbedding = None

def __getattr__(name):
    global _ResonantWindingDHSRBlock, _PureResonantWindingEmbedding
    if name == 'ResonantWindingDHSRBlock':
        if _ResonantWindingDHSRBlock is None:
            from srt_library.theory_unique_components.resonant.resonant_dhsr_block import ResonantWindingDHSRBlock as _cls
            _ResonantWindingDHSRBlock = _cls
        return _ResonantWindingDHSRBlock
    if name == 'PureResonantWindingEmbedding':
        if _PureResonantWindingEmbedding is None:
            from srt_library.theory_unique_components.resonant.resonant_embedding import PureResonantWindingEmbedding as _cls
            _PureResonantWindingEmbedding = _cls
        return _PureResonantWindingEmbedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from ..winding.syntony import PureWindingSyntonyComputer
from ..winding.syntony import (
    PureWindingSyntonyComputer as WindingSyntonyComputer,
)
from ..winding.winding_net import PureWindingNet as WindingNet

__all__ = [
    "WindingStateEmbedding",
    "FibonacciHierarchy",
    "PrimeSelectionLayer",
    "WindingSyntonyComputer",
    "WindingDHSRBlock",
    "ResonantWindingDHSRBlock",
    "WindingNet",
    # Pure versions (no PyTorch)
    "PurePrimeSelectionLayer",
    "PureWindingSyntonyComputer",
    "PureResonantWindingEmbedding",
]
