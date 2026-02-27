"""
SRT Corrections - Correction factors (1 +/- q/N) for algebraic structures.

These corrections arise from heat kernel regularization and appear
in all physical predictions from SRT.

Classes:
    CorrectionFactors - Compute correction factors for structures

Functions:
    correction_factors() - Factory for CorrectionFactors
"""

from applications.srt.corrections.factors import (
    CorrectionFactors,
    correction_factors,
)

__all__ = [
    "CorrectionFactors",
    "correction_factors",
]
