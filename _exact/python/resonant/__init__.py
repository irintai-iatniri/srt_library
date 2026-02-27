"""
Resonant Engine for Syntonic.

Hardware-native SRT/CRT architecture:
- GPU = D̂ (Differentiation) - chaotic flux generator
- CPU = Ĥ (Harmonization) - exact lattice crystallization
- PCIe = Phase boundary with φ-dwell timing

The Resonant Engine resolves the "No Float Paradox" by treating floats as
ephemeral shadows cast by the eternal exact lattice. Floats exist only during
D-phase; they are destroyed each cycle by crystallization.

Example:
    >>> from syntonic_applications.resonant import ResonantTensor, ResonantEvolver, RESConfig
    >>>
    >>> # Create from float data
    >>> tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
    >>> print(f"Initial syntony: {tensor.syntony:.4f}")
    >>>
    >>> # Run CPU cycle (D → H)
    >>> syntony = tensor.cpu_cycle(noise_scale=0.1, precision=100)
    >>> print(f"Post-cycle syntony: {syntony:.4f}")
    >>>
    >>> # Evolution with RES
    >>> config = RESConfig(population_size=32, survivor_count=8)
    >>> evolver = ResonantEvolver(tensor, config)
    >>> result = evolver.run()
    >>> print(f"Final syntony: {result.final_syntony:.4f}")
"""

from applications._core import (
    GoldenExact,
    RESConfig,
    ResonantEvolver,
    ResonantTensor,
    RESResult,
)

# Retrocausal module
from . import retrocausal

# Constants
PHI = 1.6180339887498948482
PHI_INV = 0.6180339887498948482
PHI_INV_SQ = 0.3819660112501051518
Q_DEFICIT = 0.027395146920  # Universal syntony deficit (NOT a hyperparameter!)

__all__ = [
    # Core types
    "ResonantTensor",
    "GoldenExact",
    # Evolver types
    "ResonantEvolver",
    "RESConfig",
    "RESResult",
    # Retrocausal module
    "retrocausal",
    # Constants
    "PHI",
    "PHI_INV",
    "PHI_INV_SQ",
    "Q_DEFICIT",
]
