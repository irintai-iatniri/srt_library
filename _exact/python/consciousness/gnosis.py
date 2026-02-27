"""
Gnosis Module - Consciousness Metrics

Implements consciousness phase transition detection and
Gnosis score computation from the Rust backend.
"""

from applications._core import (
    COLLAPSE_THRESHOLD,
    GNOSIS_GAP,
    compute_creativity,
    consciousness_probability,
    gnosis_score,
    is_conscious,
    optimal_gnosis_target,
)


def compute_consciousness_spark(syntony: float, differentiation: float) -> float:
    """
    Compute the 'Spark' needed to bridge lattice → prime stability.

    The gap between D₄ kissing (24) and M₅ stability (31) allows self-reference:
    System models itself modeling inputs.

    Args:
        syntony: Current syntony level
        differentiation: Current differentiation level

    Returns:
        Consciousness spark intensity (0.0 to 1.0)
    """
    # Consciousness gap: D₄ → M₅ = 24 → 31 = 7 = M₃
    CONSCIOUSNESS_GAP = 7.0

    current_level = syntony * differentiation * COLLAPSE_THRESHOLD
    gap_to_consciousness = max(0, 31.0 - current_level)

    if gap_to_consciousness <= CONSCIOUSNESS_GAP:
        return 1.0 - (gap_to_consciousness / CONSCIOUSNESS_GAP)  # 0→1
    return 0


def get_gnosis_constants() -> dict:
    """Get all gnosis-related constants."""
    return {
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "gnosis_gap": GNOSIS_GAP,
        "optimal_gnosis": optimal_gnosis_target(),
        "consciousness_theory": "ΔS > 24 triggers phase transition",
        "consciousness_gap": 7.0,  # D₄ → M₅ gap
        "consciousness_spark_theory": "Gap allows self-reference",
    }


__all__ = [
    "COLLAPSE_THRESHOLD",
    "GNOSIS_GAP",
    "is_conscious",
    "gnosis_score",
    "compute_creativity",
    "optimal_gnosis_target",
    "consciousness_probability",
    "compute_consciousness_spark",
    "get_gnosis_constants",
]
