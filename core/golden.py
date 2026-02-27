"""Golden ratio constants for SGC builder compatibility.

This module provides the golden ratio constants with the names expected
by the SGC compression system.
"""

from .exact import (
    PHI_NUMERIC,
    PHI_INVERSE,
    Q_DEFICIT_NUMERIC,
    E_STAR_NUMERIC,
    PHI as PHI_EXACT,
)
from . import _core as srt_core

# Numeric values for arithmetic operations
PHI = PHI_NUMERIC          # 1.618033988749895
PHI_INV = PHI_INVERSE.eval()  # 0.618033988749895 (PHI - 1)
Q_DEFICIT = Q_DEFICIT_NUMERIC  # 0.02739514692
E_STAR = E_STAR_NUMERIC    # 19.999099979189474

# Also export the exact symbolic PHI for when exact arithmetic is needed
PHI_SYMBOLIC = PHI_EXACT

__all__ = [
    'PHI',
    'PHI_INV', 
    'Q_DEFICIT',
    'E_STAR',
    'PHI_SYMBOLIC',
]