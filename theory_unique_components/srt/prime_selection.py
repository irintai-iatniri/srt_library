"""
Prime Selection Rules - Python Wrapper

Re-exports the Python implementations for Fermat, Mersenne, and Lucas
prime-selection utilities so the rest of the codebase can import
these functions from `srt_library.theory_unique_components.srt.prime_selection`.
"""

# Fermat primes / force spectrum
from srt_library.theory_unique_components.srt.fermat_forces import (
    fermat_number,
    get_force_spectrum,
    is_fermat_prime,
    validate_force_existence,
)

# Lucas numbers / dark sector
from srt_library.theory_unique_components.srt.lucas_shadow import (
    dark_matter_mass_prediction,
    get_shadow_spectrum,
    is_lucas_prime,
    lucas_number,
    shadow_phase,
)

# Mersenne primes / generation spectrum
from srt_library.theory_unique_components.srt.mersenne_matter import (
    generation_barrier_explanation,
    get_generation_spectrum,
    is_mersenne_prime,
    mersenne_number,
    validate_generation_stability,
)

__all__ = [
    # Fermat
    "fermat_number",
    "is_fermat_prime",
    "get_force_spectrum",
    "validate_force_existence",
    # Mersenne
    "mersenne_number",
    "is_mersenne_prime",
    "get_generation_spectrum",
    "generation_barrier_explanation",
    "validate_generation_stability",
    # Lucas
    "lucas_number",
    "shadow_phase",
    "is_lucas_prime",
    "dark_matter_mass_prediction",
    "get_shadow_spectrum",
]
