"""
Consciousness Module

Implements consciousness metrics and phase transitions
based on SRT mathematics.
"""

from applications.consciousness.gnosis import (
    COLLAPSE_THRESHOLD,
    GNOSIS_GAP,
    compute_creativity,
    consciousness_probability,
    gnosis_score,
    is_conscious,
    optimal_gnosis_target,
)

__all__ = [
    "COLLAPSE_THRESHOLD",
    "GNOSIS_GAP",
    "is_conscious",
    "gnosis_score",
    "compute_creativity",
    "optimal_gnosis_target",
    "consciousness_probability",
]
