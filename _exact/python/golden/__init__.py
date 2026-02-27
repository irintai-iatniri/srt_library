"""
SRT Golden Operations - Golden ratio measure and recursion map.

The golden ratio phi is the organizing principle of SRT. This module
provides the golden measure w(n) = exp(-|n|^2/phi) and the golden
recursion map R: n -> floor(phi*n).

Classes:
    GoldenMeasure - Golden-weighted measure on winding lattice
    GoldenRecursionMap - The golden recursion R: n -> floor(phi*n)

Functions:
    golden_measure() - Factory for GoldenMeasure
    golden_recursion() - Factory for GoldenRecursionMap
    golden_weight(n) - Quick weight computation
    apply_golden_recursion(n) - Quick recursion application
"""

from applications.srt.golden.measure import (
    GoldenMeasure,
    compute_partition_function,
    golden_measure,
    golden_weight,
)
from applications.srt.golden.recursion import (
    GoldenRecursionMap,
    apply_golden_recursion,
    golden_recursion,
)

# Aliases for consistency
golden_recursion_map = golden_recursion

__all__ = [
    "GoldenMeasure",
    "golden_measure",
    "golden_weight",
    "compute_partition_function",
    "GoldenRecursionMap",
    "golden_recursion",
    "golden_recursion_map",
    "apply_golden_recursion",
]
