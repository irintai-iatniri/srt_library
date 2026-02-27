"""
SRT Geometry - T^4 torus and winding state classes.

The internal geometry of SRT is the 4-torus T^4 = S^1_7 x S^1_8 x S^1_9 x S^1_10
where the subscripts denote the compact internal dimensions.

Classes:
    WindingState - Winding numbers |n_7, n_8, n_9, n_10>
    T4Torus - The 4-torus geometry

Functions:
    winding_state() - Factory for WindingState
    create_torus() - Factory for T4Torus
"""

from applications.srt.geometry.torus import (
    T4Torus,
    create_torus,
)
from applications.srt.geometry.winding import (
    WindingState,
    count_windings,
    winding_state,
)

# Alias for consistency with other factory functions
t4_torus = create_torus

__all__ = [
    "WindingState",
    "winding_state",
    "T4Torus",
    "create_torus",
    "t4_torus",
]
