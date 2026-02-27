"""
Syntonic Optimization - Retrocausal RES and Golden Momentum Training.

Provides two optimization strategies:

1. Retrocausal RES (Resonant Evolutionary Search):
   - Syntony as fitness, not loss gradients
   - Attractor-guided evolution

2. Golden Momentum:
   - Gradient-based with phi-derived momentum (beta = 1/φ)
   - Resistant to noise, responsive to persistent patterns
"""

# Re-export Retrocausal RES from Rust backend
from ..._core import (
    GoldenMomentum,
    RESConfig,
    ResonantEvolver,
    RESResult,
)

# Golden Momentum optimizer
from ..optim.golden_momentum import (
    GoldenMomentumOptimizer,
    create_golden_optimizer,
)

# Retrocausal components (from theory_unique_components)
try:
    from srt_library.theory_unique_components.resonant.retrocausal import (
        RetrocausalConfig,
        compare_convergence,
        create_retrocausal_evolver,
        create_standard_evolver,
    )
except ImportError:
    RetrocausalConfig = None
    compare_convergence = None
    create_retrocausal_evolver = None
    create_standard_evolver = None

__all__ = [
    "GoldenMomentumOptimizer", "GoldenMomentum", "create_golden_optimizer",
    "RetrocausalConfig", "create_retrocausal_evolver",
    "create_standard_evolver", "compare_convergence",
    "ResonantEvolver", "RESConfig", "RESResult",
]
