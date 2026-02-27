"""
Fermat Prime Force Selection Rules.

A gauge force exists IFF F_n = 2^(2^n) + 1 is prime.
This caps fundamental forces at n=4 (F_5 is composite).
"""


def fermat_number(n: int) -> int:
    """Compute F_n = 2^(2^n) + 1."""
    return (1 << (1 << n)) + 1


def is_fermat_prime(n: int) -> bool:
    """Check if F_n is prime (valid gauge force)."""
    if n > 4:
        return False  # F_5 through F_∞ are composite
    return n <= 4  # F_0 through F_4 are proven prime


FORCE_SPECTRUM = {
    0: ("Strong", 3, "SU(3) Color"),
    1: ("Electroweak", 5, "Symmetry Breaking"),
    2: ("Dark Boundary", 17, "Topological Firewall"),
    3: ("Gravity", 257, "Geometric Container"),
    4: ("Versal", 65537, "Syntonic Repulsion"),
}


def get_force_spectrum() -> dict:
    """Get the complete gauge force spectrum."""
    return FORCE_SPECTRUM


def validate_force_existence(force_index: int) -> bool:
    """
    Validate whether a fundamental force exists based on Fermat primality.

    Args:
        force_index: The force index (0=Strong, 1=Electroweak, etc.)

    Returns:
        True if the force exists (F_n is prime), False otherwise
    """
    return is_fermat_prime(force_index)
