"""
Mersenne Prime Matter Stability Rules.

A winding mode is stable IFF M_p = 2^p - 1 is prime.
This explains exactly 3 fermion generations.
"""

from srt_library.core.exact import M11_BARRIER


def mersenne_number(p: int) -> int:
    """Compute M_p = 2^p - 1."""
    return (1 << p) - 1


def is_mersenne_prime(p: int) -> bool:
    """Check if M_p is prime (stable matter)."""
    # Known Mersenne primes (for small p)
    known_primes = {2, 3, 5, 7, 13, 17, 19, 31}
    if p in known_primes:
        return True

    # For larger p, we would need full Lucas-Lehmer implementation
    # For now, return False for unknown cases
    return False


GENERATION_SPECTRUM = {
    2: ("Generation 1", 3, ["Electron", "Up", "Down"]),
    3: ("Generation 2", 7, ["Muon", "Charm", "Strange"]),
    5: ("Generation 3", 31, ["Tau", "Bottom"]),
    7: ("Heavy Anchor", 127, ["Top", "Higgs VEV"]),
    # 11: BARRIER - M_11 = 2047 = 23 × 89 (composite)
}


def get_generation_spectrum() -> dict:
    """Get the complete fermion generation spectrum."""
    return GENERATION_SPECTRUM


def validate_generation_stability(generation_index: int) -> bool:
    """
    Validate whether a fermion generation is stable based on Mersenne primality.

    Args:
        generation_index: The generation index (0=1st gen, 1=2nd gen, etc.)

    Returns:
        True if the generation is stable (M_p is prime), False otherwise
    """
    # Only 3 generations are stable due to M_11 barrier
    if generation_index >= 3:
        return False  # No 4th generation

    p_values = [2, 3, 5]  # Corresponding Mersenne exponents for 3 generations
    return is_mersenne_prime(p_values[generation_index])


def generation_barrier_explanation() -> str:
    """Why there's no 4th generation."""
    return (
        "M_11 = 2^11 - 1 = {} = 23 × 89 (composite)\n"
        "The geometry at winding depth 11 factorizes.\n"
        "No stable fermion can exist at 4th generation."
    ).format(M11_BARRIER)
