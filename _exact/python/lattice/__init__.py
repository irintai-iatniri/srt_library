"""
SRT Lattice - E8 and D4 lattice structures.

The E8 lattice is the fundamental structure in SRT from which the
Standard Model gauge groups emerge. The D4 lattice (kissing number 24)
defines the consciousness threshold.

Classes:
    D4Root - Root of D4 lattice
    D4Lattice - D4 lattice with 24 roots
    E8Root - Root of E8 lattice with exact coordinates
    E8Lattice - E8 lattice with 240 roots
    GoldenProjector - P_phi: R^8 -> R^4 projection
    GoldenCone - 36 roots in golden cone (= Phi+(E6))
    QuadraticForm - Q(lambda) = |P_parallel|^2 - |P_perp|^2

Functions:
    d4_lattice() - Factory for D4Lattice
    e8_lattice() - Factory for E8Lattice
    golden_projector() - Factory for GoldenProjector
    golden_cone() - Factory for GoldenCone
    quadratic_form() - Factory for QuadraticForm
    compute_Q() - Quick Q(lambda) computation

Constants:
    K_D4 - Kissing number of D4 (24)
    NULL_VECTORS - The 4 null vectors defining the Golden Cone
"""

from applications.srt.lattice.d4 import (
    K_D4,
    D4Lattice,
    D4Root,
    d4_lattice,
)
from applications.srt.lattice.e8 import (
    E8Lattice,
    E8Root,
    e8_lattice,
)
from applications.srt.lattice.golden_cone import (
    GoldenCone,
    GoldenProjector,
    golden_cone,
    golden_projector,
)
from applications.srt.lattice.quadratic_form import (
    NULL_VECTORS,
    QuadraticForm,
    compute_Q,
    quadratic_form,
)

__all__ = [
    "D4Root",
    "D4Lattice",
    "d4_lattice",
    "K_D4",
    "E8Root",
    "E8Lattice",
    "e8_lattice",
    "GoldenProjector",
    "golden_projector",
    "GoldenCone",
    "golden_cone",
    "QuadraticForm",
    "quadratic_form",
    "compute_Q",
    "NULL_VECTORS",
]
